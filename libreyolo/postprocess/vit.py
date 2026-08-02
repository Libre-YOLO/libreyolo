"""Postprocessing for classic ViT classification logits."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def postprocess(output: Any, **kwargs) -> Dict[str, torch.Tensor]:
    """Convert one classifier output into a probability vector."""
    del kwargs
    logits = output
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if isinstance(logits, dict):
        logits = logits.get("logits", logits.get("predictions"))
    logits = torch.as_tensor(logits).float()
    if logits.ndim > 1:
        logits = logits.reshape(-1) if logits.shape[0] == 1 else logits[0]
    return {"probs": F.softmax(logits, dim=-1)}
