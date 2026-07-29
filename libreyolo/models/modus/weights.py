# SPDX-License-Identifier: MIT

"""External-only MODUS checkpoint resolution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HF_REPO = "EPFL-VILAB/MODUS"
HF_REVISION = "8428a81602c19141e422b1e1795dddcb5d2bc14b"
ACCESS_URL = f"https://huggingface.co/{HF_REPO}"
COMPLETE_MARKER = ".libreyolo-modus-snapshot.json"
REQUIRED_FILES = (
    "model.safetensors",
    "ae.safetensors",
    "llm_config.json",
    "vit_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)

WEIGHT_TERMS_NOTICE = (
    "\n"
    "----------------------------------------------------------------\n"
    "LibreMODUS loads weights only from the user's local files or directly\n"
    "from EPFL-VILAB/MODUS. LibreYOLO does not redistribute or mirror them.\n"
    "The upstream model card currently labels the terms 'bagel-derived' and\n"
    "requests research-only use. Review and accept the current upstream terms\n"
    "before downloading; your Hugging Face credentials and use remain yours.\n"
    f"  Upstream: {ACCESS_URL}\n"
    "----------------------------------------------------------------"
)

_notice_shown = False


def notify_weight_terms_once() -> None:
    global _notice_shown
    if not _notice_shown:
        _notice_shown = True
        logger.warning(WEIGHT_TERMS_NOTICE)


def _required_files_exist(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_FILES)


def validate_snapshot(path: str | Path) -> Path:
    """Validate the released directory checkpoint without executing remote code."""
    snapshot = Path(path).expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"MODUS checkpoint directory {snapshot} is incomplete; missing: "
            f"{', '.join(missing)}."
        )
    return snapshot


def _download_complete(path: Path) -> bool:
    marker = path / COMPLETE_MARKER
    if not marker.is_file() or not _required_files_exist(path):
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return metadata == {"repo": HF_REPO, "revision": HF_REVISION}


def resolve_modus_snapshot(
    *,
    checkpoint_path: Optional[str | Path] = None,
    token: Optional[str] = None,
    download_dir: Optional[str | Path] = None,
) -> Path:
    """Return a complete local snapshot, downloading only from upstream."""
    notify_weight_terms_once()
    if checkpoint_path is not None:
        return validate_snapshot(checkpoint_path)

    destination = Path(download_dir or "weights/LibreMODUS14b-a7b").resolve()
    if _download_complete(destination):
        return destination

    try:
        from huggingface_hub import get_token, snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as exc:
        raise ImportError(
            "LibreMODUS requires the 'modus' extra. Install with:\n"
            "    pip install 'libreyolo[modus]'"
        ) from exc

    effective_token = token.strip() if isinstance(token, str) else token
    if not effective_token:
        effective_token = get_token()
    if not effective_token:
        raise PermissionError(
            "Downloading EPFL-VILAB/MODUS requires authentication with the "
            "user's own Hugging Face account. Review and accept the upstream "
            f"terms at {ACCESS_URL}, then run `hf auth login`, set HF_TOKEN, "
            "or pass token=. A complete local checkpoint_path does not require "
            "Hugging Face authentication."
        )

    logger.info(
        "Downloading external MODUS snapshot %s@%s -> %s",
        HF_REPO,
        HF_REVISION,
        destination,
    )
    try:
        snapshot_download(
            repo_id=HF_REPO,
            revision=HF_REVISION,
            local_dir=str(destination),
            token=effective_token,
            allow_patterns=list(REQUIRED_FILES) + ["README.md", ".gitattributes"],
        )
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise PermissionError(
                "Hugging Face denied access to EPFL-VILAB/MODUS. Review the "
                f"upstream terms/access page at {ACCESS_URL}, then authenticate "
                "with `hf auth login`, HF_TOKEN, or token=."
            ) from exc
        raise

    if not _required_files_exist(destination):
        missing = [
            name for name in REQUIRED_FILES if not (destination / name).is_file()
        ]
        raise FileNotFoundError(
            "The upstream MODUS download completed without required files: "
            f"{', '.join(missing)}."
        )
    (destination / COMPLETE_MARKER).write_text(
        json.dumps({"repo": HF_REPO, "revision": HF_REVISION}) + "\n",
        encoding="utf-8",
    )
    return destination
