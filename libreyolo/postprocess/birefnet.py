"""Postprocessing for the LibreBiRefNet matte family.

The network emits a single-channel logit map at the fixed native resolution.
Turning it into the ``matte`` task's Results payload is: ``sigmoid`` -> resize
to the original image canvas with bilinear interpolation -> clamp to ``[0, 1]``.
Bilinear (not nearest) matters: nearest gives jagged hair/fur edges users
notice immediately.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


def postprocess(output: Any, original_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    """Convert a BiRefNet logit map to a ``{"matte": (H, W) float32}`` payload.

    Args:
        output: The network output, a ``(1, 1, S, S)`` logit tensor (or a
            list/tuple whose last element is that tensor).
        original_size: ``(orig_w, orig_h)`` of the source image.

    Returns:
        ``{"matte": tensor}`` where the tensor is ``(orig_h, orig_w)`` float32 in
        ``[0, 1]`` on the CPU.
    """
    logits = output
    if isinstance(logits, (list, tuple)):
        logits = logits[-1]
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)

    orig_w, orig_h = original_size
    matte = torch.sigmoid(logits.float())
    if matte.dim() == 4:
        matte = F.interpolate(matte, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        matte = matte[0, 0]
    elif matte.dim() == 3:
        matte = F.interpolate(matte.unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False)[0, 0]
    matte = matte.clamp(0.0, 1.0)
    return {"matte": matte.detach().cpu()}
