"""RetinaNet architecture placeholder for the skeleton commit."""

from __future__ import annotations

import torch
from torch import nn


class LibreRetinaNetModel(nn.Module):
    """Construction-only placeholder replaced by the native graph next commit."""

    def __init__(self, size: str, num_classes: int) -> None:
        super().__init__()
        self.size = size
        self.num_classes = num_classes
        self.backbone = nn.Identity()
        self.head = nn.Identity()

    def forward(self, images: torch.Tensor):
        raise NotImplementedError("RetinaNet architecture lands in commit 2")


__all__ = ["LibreRetinaNetModel"]
