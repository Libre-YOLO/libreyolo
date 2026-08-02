"""Native SSD300 network components.

The architecture is implemented in the next port commit.  This shell exists so
the family can register and its checkpoint discriminator can be tested before
the graph is introduced.
"""

from __future__ import annotations

import torch
from torch import nn


class LibreSSDModel(nn.Module):
    """Construction shell for the fixed 300 px SSD graph."""

    def __init__(self, num_classes: int = 91) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, images: torch.Tensor):
        raise NotImplementedError("SSD300 architecture is not implemented yet")


__all__ = ["LibreSSDModel"]
