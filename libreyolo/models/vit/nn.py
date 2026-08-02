"""Native ViT architecture placeholder for the family-skeleton commit."""

from __future__ import annotations

import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    """Construction-only placeholder replaced by the native graph next commit."""

    def __init__(self, size: str = "ti", num_classes: int = 1000):
        super().__init__()
        if size not in {"ti", "s", "b", "l"}:
            raise ValueError("Unknown ViT size. Choose from: ti, s, b, l.")
        self.size = size
        self.num_classes = num_classes
        self.patch_embed = nn.Identity()
        self.blocks = nn.Identity()
        self.norm = nn.Identity()
        self.head = nn.Linear(1, num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        weight = self.head.weight
        self.head = nn.Linear(1, num_classes).to(device=weight.device, dtype=weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        raise NotImplementedError("The native ViT graph lands in the architecture commit.")
