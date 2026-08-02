"""Neural-network shell for the LibreFCN family.

The complete torchvision-compatible architecture lands in the next port
commit. Keeping this shell buildable lets the factory and checkpoint
recognition contract stand on their own.
"""

from __future__ import annotations

import torch
from torch import nn


class LibreFCNModel(nn.Module):
    """Buildable FCN shell used while the native graph is introduced."""

    def __init__(self, size: str = "r50", num_classes: int = 21) -> None:
        super().__init__()
        if size not in ("r50", "r101"):
            raise ValueError(f"Unknown FCN size {size!r}")
        self.size = size
        self.num_classes = int(num_classes)
        self.backbone = nn.Identity()
        self.classifier = nn.Identity()
        self.aux_classifier = nn.Identity()

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("The LibreFCN architecture is not implemented yet.")


__all__ = ["LibreFCNModel"]
