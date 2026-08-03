"""Postprocessing for the LibreDeepLabv3 semantic family."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


def semantic_logits(
    output: Any,
    original_size: Tuple[int, int],
) -> torch.Tensor:
    """Return ``[B, C, orig_h, orig_w]`` logits before semantic argmax."""
    logits = output
    if isinstance(logits, dict):
        logits = logits.get(
            "semantic_logits",
            logits.get("predictions", logits.get("out")),
        )
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
    if logits.ndim != 4:
        raise ValueError(
            f"DeepLabv3 output must have shape [B, C, H, W], got {tuple(logits.shape)}."
        )

    orig_w, orig_h = original_size
    logits = logits.float()
    if tuple(logits.shape[-2:]) != (orig_h, orig_w):
        logits = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
    return logits


def postprocess(
    output: Any,
    original_size: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    """Convert raw DeepLabv3 logits to a CPU semantic-class mask."""
    logits = semantic_logits(output, original_size)
    return {"semantic": logits.argmax(dim=1)[0].detach().cpu()}


__all__ = ["postprocess", "semantic_logits"]
