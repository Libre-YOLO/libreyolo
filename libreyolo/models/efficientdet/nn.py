"""Native EfficientDet graph.

The complete Apache-2.0 ``effdet``-compatible architecture is added in the
next port commit. This small shell keeps the factory integration independently
loadable while the graph is implemented and parity-gated.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LibreEfficientDetModel(nn.Module):
    """Temporary graph shell used by the factory-skeleton commit."""

    def __init__(self, size: str, num_classes: int = 90) -> None:
        super().__init__()
        self.size = size
        self.num_classes = num_classes
        self.backbone = nn.Identity()
        self.fpn = nn.Identity()
        self.class_net = nn.Identity()
        self.box_net = nn.Identity()

    def forward(self, x: torch.Tensor):
        del x
        return [], []


__all__ = ["LibreEfficientDetModel"]
