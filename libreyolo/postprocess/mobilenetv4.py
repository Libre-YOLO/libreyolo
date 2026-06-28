"""Postprocessing for LibreMobileNetV4 classification: logits -> probability vector.

The runner wraps the returned ``{"probs": vector}`` into ``Results.probs`` so
``result.probs.top1 / .top5 / .top1conf`` behave like the rest of the ecosystem.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def postprocess(
    output: Any,
    *,
    conf_thres: float = 0.0,
    iou_thres: float = 0.0,
    original_size=None,
    max_det: int = 0,
    ratio: float = 1.0,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Convert raw classifier logits to a per-image probability vector."""
    logits = output
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if isinstance(logits, dict):
        logits = logits.get("logits", logits.get("predictions"))
    logits = torch.as_tensor(logits).float()
    if logits.ndim > 1:
        logits = logits.reshape(-1) if logits.shape[0] == 1 else logits[0]
    probs = F.softmax(logits, dim=-1)
    return {"probs": probs}
