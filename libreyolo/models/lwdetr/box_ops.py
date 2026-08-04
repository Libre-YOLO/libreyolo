"""Box conversions for LW-DETR.

Ported from LW-DETR (https://github.com/Atten4Vis/LW-DETR).
Copyright (c) 2024 Baidu. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
Modified from DETR (https://github.com/facebookresearch/detr).
Copyright (c) Facebook, Inc. and its affiliates.
"""

from __future__ import annotations

import torch

__all__ = ["box_cxcywh_to_xyxy", "box_xyxy_to_cxcywh"]


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert ``(cx, cy, w, h)`` boxes to ``(x1, y1, x2, y2)``."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Convert ``(x1, y1, x2, y2)`` boxes to ``(cx, cy, w, h)``."""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)
