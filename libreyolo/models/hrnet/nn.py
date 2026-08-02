"""Temporary HRNet graph scaffold.

The exact MIT-licensed HRNet graph is added in the architecture commit that
follows this factory-registration commit.
"""

from __future__ import annotations

import torch
from torch import nn


class HRNetPoseModel(nn.Module):
    """Small shape-compatible scaffold used while registering the family."""

    def __init__(self, width: int, num_keypoints: int = 17) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width, width, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.stage3 = nn.Identity()
        self.final_layer = nn.Conv2d(width, num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.final_layer(x)
