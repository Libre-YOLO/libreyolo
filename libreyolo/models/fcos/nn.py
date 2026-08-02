"""FCOS inference-graph scaffold.

The checkpoint-compatible ResNet-50/FPN/head graph is added in the next port
step. Keeping this small module importable lets the factory and registry
contracts land independently of the architecture.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LibreFCOSModel(nn.Module):
    """Importable FCOS scaffold with the final raw-output contract."""

    def __init__(self, num_classes: int = 91) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone = nn.Identity()
        self.head = nn.Identity()

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        batch = images.shape[0]
        return {
            "cls_logits": images.new_zeros((batch, 0, self.num_classes)),
            "bbox_regression": images.new_zeros((batch, 0, 4)),
            "bbox_ctrness": images.new_zeros((batch, 0, 1)),
            "anchors": images.new_zeros((batch, 0, 4)),
            "level_sizes": torch.zeros(
                (batch, 0), dtype=torch.int64, device=images.device
            ),
        }


__all__ = ["LibreFCOSModel"]
