"""Automatic mixed-precision helpers."""

from __future__ import annotations

from typing import Final

_AMP_DTYPE_ALIASES: Final = {
    "float16": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


def normalize_amp_dtype(value: str) -> str:
    """Return the canonical AMP dtype name or raise for unsupported values."""
    normalized = str(value).strip().lower()
    try:
        return _AMP_DTYPE_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(_AMP_DTYPE_ALIASES))
        raise ValueError(f"amp_dtype must be one of: {allowed}; got {value!r}") from exc


def torch_amp_dtype(value: str):
    """Resolve a canonical AMP dtype name to its torch dtype."""
    import torch

    canonical = normalize_amp_dtype(value)
    return torch.float16 if canonical == "float16" else torch.bfloat16
