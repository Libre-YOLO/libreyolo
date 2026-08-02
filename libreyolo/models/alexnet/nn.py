"""Native AlexNet graph with torchvision-compatible module names.

Derived from ``pytorch/vision`` v0.26.0 at commit
``336d36e8db990a905498c73933e35231876e28bc`` under BSD-3-Clause. The graph is
the single-tower "one weird trick" variant: 64 conv1 channels, no local
response normalization, and no grouped convolutions. Keeping the upstream
``features`` / ``avgpool`` / ``classifier`` layout lets the official state dict
load with ``strict=True``. See the family ``NOTICE`` for attribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """Five-convolution AlexNet feature extractor and three-layer classifier."""

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


__all__ = ["AlexNet"]
