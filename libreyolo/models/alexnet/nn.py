"""Native AlexNet graph.

The complete torchvision-compatible architecture is added in the architecture
commit. This lightweight shell keeps the factory-registration commit
independently importable and constructible.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """Constructible shell for the AlexNet classification graph."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Identity()
        self.avgpool = nn.Identity()
        self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.avgpool(self.features(x)))


__all__ = ["AlexNet"]
