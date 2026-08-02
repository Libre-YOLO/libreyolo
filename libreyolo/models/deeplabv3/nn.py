"""DeepLabv3 network definition.

The temporary bootstrap graph in this commit keeps the new factory family
constructible. The torchvision-compatible backbone and ASPP graph land in the
next port commit, before any checkpoint conversion or parity claim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


SIZE_CONFIGS = {
    "r50": {"backbone": "resnet50", "output_stride": 8},
    "r101": {"backbone": "resnet101", "output_stride": 8},
    "mv3": {"backbone": "mobilenet_v3_large", "output_stride": 16},
}


class LibreDeepLabv3Net(nn.Module):
    """Bootstrap dense graph replaced by the native port in the next commit."""

    def __init__(self, size: str = "r50", num_classes: int = 21) -> None:
        super().__init__()
        if size not in SIZE_CONFIGS:
            raise ValueError(
                f"Unknown DeepLabv3 size {size!r}; choose from {tuple(SIZE_CONFIGS)}."
            )
        self.size = size
        self.num_classes = int(num_classes)
        self.backbone = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.classifier = nn.Conv2d(8, self.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(F.relu(self.backbone(x), inplace=False))


__all__ = ["LibreDeepLabv3Net", "SIZE_CONFIGS"]
