"""DINO-DETR preprocessing and checkpoint helpers."""

from __future__ import annotations

from typing import Any


def unwrap_dinodetr_checkpoint(checkpoint: Any) -> dict:
    """Return parameters from an upstream or LibreYOLO checkpoint mapping."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dict, got {type(checkpoint).__name__}")
    for key in ("model", "model_state_dict", "state_dict", "ema"):
        inner = checkpoint.get(key)
        if isinstance(inner, dict):
            return inner
    return checkpoint


__all__ = ["unwrap_dinodetr_checkpoint"]
