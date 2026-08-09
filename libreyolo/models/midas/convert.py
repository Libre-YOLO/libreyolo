"""MiDaS official-checkpoint identification and download conversion."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

import torch

from ...utils.serialization import (
    load_untrusted_torch_file,
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)


UPSTREAM_URLS = {
    "s": (
        "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
    ),
    "l": "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt",
}
UPSTREAM_SHA256 = {
    "midas_v21_small_256.pt": (
        "70d6b9c891758c67f974a6097fb0c608c7ee67fb81ac3e5588847d5596d56fca"
    ),
    "dpt_large_384.pt": (
        "2f21e586477d90cb9624c7eef5df7891edca49a1c4795ee2cb631fd4daa6ca69"
    ),
}
INPUT_SIZES = {"s": 256, "l": 384}


def is_upstream_state_dict(state_dict: dict) -> bool:
    """Return true only for the two native-keyed MiDaS architectures."""
    has_decoder = (
        "scratch.refinenet1.resConfUnit1.conv1.weight" in state_dict
        and "scratch.output_conv.4.weight" in state_dict
    )
    has_dpt = "pretrained.model.cls_token" in state_dict
    has_small = "pretrained.layer1.3.0.conv_dw.weight" in state_dict
    return has_decoder and (has_dpt or has_small)


def detect_size(state_dict: dict) -> str | None:
    cls_token = state_dict.get("pretrained.model.cls_token")
    if cls_token is not None and tuple(cls_token.shape) == (1, 1, 1024):
        return "l"
    stem = state_dict.get("pretrained.layer1.0.weight")
    if stem is not None and tuple(stem.shape[:2]) == (32, 3):
        return "s"
    return None


def wrap_upstream_state_dict(state_dict: dict, size: str) -> dict:
    """Add strict LibreYOLO depth metadata without changing learned tensors."""
    if not is_upstream_state_dict(state_dict):
        raise ValueError("Downloaded file is not a supported MiDaS state dict.")
    detected = detect_size(state_dict)
    if detected != size:
        raise ValueError(
            f"MiDaS checkpoint signature is {detected!r}, expected {size!r}."
        )
    return wrap_libreyolo_checkpoint(
        dict(state_dict),
        model_family="midas",
        size=size,
        task="depth",
        nc=1,
        names={0: "depth"},
        imgsz=INPUT_SIZES[size],
    )


def verify_and_wrap_download(local_path: str, source_url: str) -> None:
    """Checksum-pin an official raw pickle, then metadata-wrap it in place."""
    source_name = Path(urlparse(source_url).path).name
    expected = UPSTREAM_SHA256.get(source_name)
    if expected is None:
        raise RuntimeError(
            f"Refusing unpinned MiDaS download {source_name!r}; no SHA-256 is known."
        )

    digest = hashlib.sha256()
    with open(local_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for MiDaS checkpoint {source_name!r}: "
            f"expected {expected}, got {actual}."
        )

    state_dict = load_untrusted_torch_file(
        local_path,
        map_location="cpu",
        context="official MiDaS release checkpoint",
    )
    size = detect_size(state_dict)
    if size is None or UPSTREAM_URLS.get(size) != source_url:
        raise RuntimeError(
            f"MiDaS release asset {source_name!r} has an unexpected architecture."
        )
    wrapped = wrap_upstream_state_dict(state_dict, size)
    errors = validate_checkpoint_metadata(wrapped, strict=True)
    if errors:
        raise RuntimeError("Invalid converted MiDaS metadata: " + "; ".join(errors))

    path = Path(local_path)
    temporary = path.with_suffix(path.suffix + ".wrapped")
    torch.save(wrapped, temporary)
    os.replace(temporary, path)


__all__ = [
    "INPUT_SIZES",
    "UPSTREAM_SHA256",
    "UPSTREAM_URLS",
    "detect_size",
    "is_upstream_state_dict",
    "verify_and_wrap_download",
    "wrap_upstream_state_dict",
]
