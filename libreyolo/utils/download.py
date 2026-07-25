"""Download helpers for LibreYOLO model weights."""

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

_YOLONAS_LICENSE_NOTICE_SHOWN = False
_DOWNLOAD_RETRIES = 3
_DOWNLOAD_BACKOFF_SECONDS = 1.0
_DOWNLOAD_TIMEOUT = (10, 60)
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
logger = logging.getLogger(__name__)


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from env var or cached login."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def _notify_yolonas_license_once() -> None:
    """Print Deci's YOLO-NAS license terms once per process before download."""
    global _YOLONAS_LICENSE_NOTICE_SHOWN
    if _YOLONAS_LICENSE_NOTICE_SHOWN:
        return
    _YOLONAS_LICENSE_NOTICE_SHOWN = True
    print(
        "\n"
        "─────────────────────────────────────────────────────────────────────\n"
        "YOLO-NAS weights are distributed by Deci.AI under a proprietary\n"
        "license (non-commercial, no redistribution, no production use\n"
        "without a separate agreement). By downloading, you accept those\n"
        "terms. Full license text:\n"
        "  https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md\n"
        "─────────────────────────────────────────────────────────────────────\n"
    )


def _detect_family_from_filename(filename: str) -> Optional[str]:
    """Return model family hint from filename (for download routing only)."""
    fl = filename.lower()
    if re.search(r"librerfdetr", fl):
        return "rfdetr"
    if re.search(r"libreyolox", fl):
        return "yolox"
    if re.search(r"libreyolo9", fl):
        return "yolo9"
    return None


def _response_total_size(response, offset: int) -> int:
    """Return the complete object size for a full or ranged response."""
    content_range = response.headers.get("content-range", "")
    match = re.match(r"bytes\s+\d+-\d+/(\d+|\*)", content_range, re.IGNORECASE)
    if match and match.group(1) != "*":
        return int(match.group(1))

    content_length = int(response.headers.get("content-length", 0))
    if content_length <= 0:
        return 0
    if response.status_code == 206:
        return offset + content_length
    return content_length


def _download_once(url: str, partial: Path, headers: dict[str, str]) -> None:
    """Stream one request, resuming a partial file when the server permits it."""
    offset = partial.stat().st_size if partial.exists() else 0
    request_headers = dict(headers)
    if offset:
        request_headers["Range"] = f"bytes={offset}-"

    response = requests.get(
        url,
        stream=True,
        headers=request_headers,
        timeout=_DOWNLOAD_TIMEOUT,
    )
    try:
        if response.status_code == 416 and offset:
            complete_range = re.match(
                r"bytes\s+\*/(\d+)",
                response.headers.get("content-range", ""),
                re.IGNORECASE,
            )
            if complete_range and int(complete_range.group(1)) == offset:
                return

            # The saved offset is invalid for this object. Remove only this
            # download's temporary file so the retry starts cleanly.
            partial.unlink(missing_ok=True)

        response.raise_for_status()

        append = offset > 0 and response.status_code == 206
        if offset and not append:
            logger.warning(
                "Download server ignored the resume request; restarting from byte 0."
            )
            offset = 0

        total_size = _response_total_size(response, offset)
        downloaded = offset
        last_logged = -1
        mode = "ab" if append else "wb"
        with open(partial, mode) as f:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(100 * downloaded / total_size)
                        if percent % 25 == 0 and percent != last_logged:
                            last_logged = percent
                            logger.info(
                                "Downloading: %d%% (%.1f/%.1f MB)",
                                percent,
                                downloaded / 1024 / 1024,
                                total_size / 1024 / 1024,
                            )

        if total_size > 0 and downloaded != total_size:
            raise IOError(
                f"Incomplete download: got {downloaded} of {total_size} bytes"
            )
    finally:
        response.close()


def download_weights(model_path: str, size: str):
    """Download weights from Hugging Face if not found locally."""
    path = Path(model_path)
    if path.exists():
        return

    from libreyolo.models.base.model import BaseModel

    url = None
    for cls in BaseModel._registry:
        url = cls.get_download_url(path.name)
        if url:
            break

    # RF-DETR is lazily registered — try loading it if no match yet
    if url is None:
        try:
            from libreyolo.models import _ensure_rfdetr

            _ensure_rfdetr()
            for cls in BaseModel._registry:
                url = cls.get_download_url(path.name)
                if url:
                    break
        except (ModuleNotFoundError, ImportError):
            pass

    if url is None:
        raise ValueError(f"Could not determine download URL for '{path.name}'.")

    notice = cls.get_download_notice(path.name, url)
    if notice:
        logger.warning(notice)

    logger.info(
        "Model weights not found at %s. Attempting download from %s...",
        model_path,
        url,
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    host = urlparse(url).netloc
    is_hf = host.endswith("huggingface.co")

    if "cloudfront.net" in host or host.endswith("deci.ai"):
        _notify_yolonas_license_once()

    headers = {}
    token = _get_hf_token()
    if token and is_hf:
        # Only attach the HF token to HF URLs — never leak it to third parties.
        headers["Authorization"] = f"Bearer {token}"
    elif is_hf and not token:
        logger.info(
            "Tip: Run `huggingface-cli login` or set HF_TOKEN for faster downloads."
        )

    # Stream to a temp file and rename at the end so a killed process can
    # never leave a truncated weight at the final path (loading one fails
    # with an opaque zip error and requires a manual delete).
    partial = path.with_name(path.name + ".part")
    last_error: Exception | None = None
    for attempt in range(_DOWNLOAD_RETRIES + 1):
        try:
            _download_once(url, partial, headers)
            os.replace(partial, path)
            logger.info("Download complete.")
            break
        except Exception as e:
            last_error = e
            if attempt == _DOWNLOAD_RETRIES:
                partial_size = partial.stat().st_size if partial.exists() else 0
                raise RuntimeError(
                    f"Failed to download weights from {url} after "
                    f"{_DOWNLOAD_RETRIES + 1} attempts: {e}. "
                    f"Partial download kept at {partial} ({partial_size} bytes)."
                ) from e

            partial_size = partial.stat().st_size if partial.exists() else 0
            logger.warning(
                "Download interrupted (%s). Retrying %d/%d from byte %d.",
                e,
                attempt + 1,
                _DOWNLOAD_RETRIES,
                partial_size,
            )
            time.sleep(_DOWNLOAD_BACKOFF_SECONDS * (2**attempt))

    if last_error is not None and not path.exists():
        # Defensive guard for type checkers and unusual filesystem races.
        raise RuntimeError(f"Failed to download weights from {url}") from last_error

    # Let the matched family verify the freshly downloaded file before anything
    # loads it (e.g. checksum-pin a third-party CDN object). This runs for every
    # download path — the LibreYOLO(...) factory and the per-family loaders all
    # funnel through here — so the check cannot be bypassed. HF-hosted LibreYOLO
    # weights use the trusting default and this is a no-op.
    cls.verify_downloaded_file(str(path), url)
