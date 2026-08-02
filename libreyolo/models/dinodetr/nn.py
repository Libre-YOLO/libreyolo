"""Temporary construction skeleton for the DINO-DETR family.

The native architecture replaces this module in the next porting commit.  The
skeleton exists so factory recognition can be reviewed independently from the
ported model math.
"""

from __future__ import annotations

import torch
from torch import nn


class LibreDINODETRModel(nn.Module):
    """Minimal shape-compatible module used during family registration."""

    def __init__(self, size: str, nc: int = 91):
        super().__init__()
        self.size = size
        self.nc = nc
        self.num_queries = 900
        self.num_select = 300
        self.backbone = nn.Identity()
        self.transformer = nn.Identity()
        self.class_embed = nn.Linear(1, nc)
        self.bbox_embed = nn.Linear(1, 4)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = images.shape[0]
        value = images.mean(dim=(1, 2, 3), keepdim=True).reshape(batch, 1, 1)
        value = value.expand(batch, self.num_queries, 1)
        return {
            "pred_logits": self.class_embed(value),
            "pred_boxes": self.bbox_embed(value).sigmoid(),
        }


__all__ = ["LibreDINODETRModel"]
