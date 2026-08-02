"""Native Faster R-CNN architecture scaffold.

The completed implementation is derived from torchvision v0.26.0 at commit
336d36e8db990a905498c73933e35231876e28bc (BSD-3-Clause). Faster R-CNN is the
landmark two-stage detector that coupled a Region Proposal Network with RoI
classification and box regression; the variants exposed here are modernized
torchvision models rather than the paper's original VGG16 network.
"""

from __future__ import annotations

import torch
from torch import nn

FASTER_RCNN_CONFIGS = {
    "n": {"backbone": "mobilenet_v3_large", "min_size": 320, "max_size": 640},
    "s": {"backbone": "mobilenet_v3_large", "min_size": 800, "max_size": 1333},
    "m": {"backbone": "resnet50_fpn", "min_size": 800, "max_size": 1333},
    "l": {"backbone": "resnet50_fpn_v2", "min_size": 800, "max_size": 1333},
}


class LibreFasterRCNNModel(nn.Module):
    """Construction placeholder replaced by the native port in commit 2."""

    def __init__(self, size: str, num_classes: int = 91) -> None:
        super().__init__()
        if size not in FASTER_RCNN_CONFIGS:
            raise ValueError(
                f"Unknown Faster R-CNN size '{size}'. "
                f"Valid sizes: {', '.join(FASTER_RCNN_CONFIGS)}"
            )
        self.size = size
        self.num_classes = num_classes
        self.backbone = nn.Identity()
        self.rpn = nn.Identity()
        self.roi_heads = nn.Identity()
        self.transform = nn.Identity()

    def forward(self, images: torch.Tensor):
        raise NotImplementedError("Faster R-CNN architecture is not wired yet")
