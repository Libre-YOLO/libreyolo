"""Download helpers for LibreYOLO model weights."""

import logging
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

_YOLONAS_LICENSE_NOTICE_SHOWN = False
logger = logging.getLogger(__name__)

_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_LOCK_POLL_SECONDS = 0.1
_DOWNLOAD_LOCK_TIMEOUT_SECONDS = 60 * 60
_DOWNLOAD_LOCK_STALE_SECONDS = 6 * 60 * 60
_DOWNLOAD_TIMEOUT = (10, 120)


class WeightDownloadError(RuntimeError):
    """Raised when model weights cannot be downloaded into a verified cache."""


class WeightVerificationError(WeightDownloadError):
    """Raised when downloaded bytes fail a family integrity check."""


class WeightDownloadLockTimeout(WeightDownloadError, TimeoutError):
    """Raised when another process holds a target download lock too long."""


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


@contextmanager
def _download_lock(path: Path):
    """Serialize downloads targeting ``path`` across processes."""
    lock_path = path.with_name(path.name + ".lock")
    deadline = time.monotonic() + _DOWNLOAD_LOCK_TIMEOUT_SECONDS
    fd = None

    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                stale = (
                    time.time() - lock_path.stat().st_mtime
                    > _DOWNLOAD_LOCK_STALE_SECONDS
                )
            except FileNotFoundError:
                continue
            if stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise WeightDownloadLockTimeout(
                    f"Timed out waiting for download lock '{lock_path}'."
                )
            time.sleep(_DOWNLOAD_LOCK_POLL_SECONDS)

    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _download_to_verified_temp(
    *,
    url: str,
    partial: Path,
    headers: dict[str, str],
    verifier,
) -> None:
    """Download to ``partial`` and verify it without exposing a final file."""
    response = requests.get(
        url,
        stream=True,
        headers=headers,
        timeout=_DOWNLOAD_TIMEOUT,
    )
    try:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        next_progress = 25

        with open(partial, "xb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = min(100, int(100 * downloaded / total_size))
                    while percent >= next_progress:
                        logger.info(
                            "Downloading: %d%% (%.1f/%.1f MiB)",
                            next_progress,
                            downloaded / 1024 / 1024,
                            total_size / 1024 / 1024,
                        )
                        next_progress += 25
            file.flush()
            os.fsync(file.fileno())

        if downloaded == 0:
            raise IOError("Downloaded response was empty.")
        if total_size > 0 and downloaded != total_size:
            raise IOError(
                f"Incomplete download: got {downloaded} of {total_size} bytes"
            )

        try:
            verifier(str(partial), url)
        except Exception as error:
            raise WeightVerificationError(
                f"Downloaded weights from {url} failed verification: {error}"
            ) from error
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _publish_verified_temp(partial: Path, destination: Path) -> bool:
    """Atomically publish ``partial`` without replacing ``destination``.

    Windows rename is create-if-absent. POSIX rename replaces an existing
    file, so use a same-directory hard link there to get the same no-clobber
    contract. The caller removes the temporary name after publication.
    """
    try:
        if os.name == "nt":
            os.rename(partial, destination)
        else:
            os.link(partial, destination)
    except FileExistsError:
        return False
    return True


def download_weights(model_path: str, size: str):
    """Download weights from Hugging Face if not found locally."""
    path = Path(model_path)

    # An existing path is user-owned input, not a cache entry managed by this
    # helper.  In particular, do not checksum, delete, or replace it merely
    # because its filename also matches an official auto-download route.
    if os.path.lexists(path):
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
    path.parent.mkdir(parents=True, exist_ok=True)

    host = urlparse(url).netloc
    is_hf = host.endswith("huggingface.co")

    headers = {}
    token = _get_hf_token()
    if token and is_hf:
        # Only attach the HF token to HF URLs — never leak it to third parties.
        headers["Authorization"] = f"Bearer {token}"
    elif is_hf and not token:
        logger.info(
            "Tip: Run `huggingface-cli login` or set HF_TOKEN for faster downloads."
        )

    # Stream to a temp file and publish it at the end so a killed process can
    # never leave a truncated weight at the final path (loading one fails
    # with an opaque zip error and requires a manual delete).
    with _download_lock(path):
        # Another caller, or the user, may have created the destination while
        # this caller was waiting for the lock.  It is no longer ours to
        # inspect or replace.
        if os.path.lexists(path):
            return

        if notice:
            logger.warning(notice)
        if "cloudfront.net" in host or host.endswith("deci.ai"):
            _notify_yolonas_license_once()
        logger.info(
            "Model weights not found at %s. Attempting download from %s...",
            model_path,
            url,
        )

        last_error = None
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            partial = path.with_name(
                f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.part"
            )
            try:
                _download_to_verified_temp(
                    url=url,
                    partial=partial,
                    headers=headers,
                    verifier=cls.verify_downloaded_file,
                )
                published = _publish_verified_temp(partial, path)
                try:
                    partial.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    logger.warning(
                        "Could not remove temporary download %s: %s",
                        partial,
                        cleanup_error,
                    )
                if published:
                    logger.info("Download complete.")
                else:
                    logger.info(
                        "Weights appeared at %s during download; preserving "
                        "the existing filesystem entry.",
                        path,
                    )
                return
            except Exception as error:
                last_error = error
                try:
                    partial.unlink()
                except FileNotFoundError:
                    pass
                if attempt < _DOWNLOAD_ATTEMPTS:
                    delay = 2 ** (attempt - 1)
                    logger.warning(
                        "Download attempt %d/%d failed; retrying in %ds: %s",
                        attempt,
                        _DOWNLOAD_ATTEMPTS,
                        delay,
                        error,
                    )
                    time.sleep(delay)

        if isinstance(last_error, WeightVerificationError):
            raise last_error
        raise WeightDownloadError(
            f"Failed to download weights from {url}: {last_error}"
        ) from last_error
