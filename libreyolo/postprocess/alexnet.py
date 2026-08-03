"""AlexNet classification postprocessing: logits to probabilities."""

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
    """Convert raw AlexNet logits to a per-image probability vector."""
    del conf_thres, iou_thres, original_size, max_det, ratio, kwargs

    logits = output
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if isinstance(logits, dict):
        logits = logits.get("logits", logits.get("predictions"))
    logits = torch.as_tensor(logits).float()
    if logits.ndim > 1:
        logits = logits.reshape(-1) if logits.shape[0] == 1 else logits[0]
    return {"probs": F.softmax(logits, dim=-1)}
