"""Temporary VGG construction surface for the factory skeleton."""

from __future__ import annotations

import torch
import torch.nn as nn


class VGG(nn.Module):
    """Minimal construction stub replaced by the native VGG graph next."""

    def __init__(self, size: str = "16", num_classes: int = 1000) -> None:
        super().__init__()
        if size not in {"16", "19", "16bn", "19bn"}:
            raise ValueError(f"Unknown VGG size {size!r}.")
        self.size = size
        self.num_classes = num_classes
        self.features = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.avgpool = nn.Identity()
        self.classifier = nn.Sequential(nn.Flatten(1), nn.Linear(3, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.avgpool(self.features(x)))


__all__ = ["VGG"]
