"""MiDaS network construction.

The first commit intentionally keeps a tiny shape-correct scaffold. The native
MiDaS v2.1 Small and DPT-Large modules replace it in the architecture commit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _MiDaSScaffold(nn.Module):
    """Parameter-free dense-output scaffold used while wiring the family."""

    def __init__(self, size: str):
        super().__init__()
        self.size = size
        self.pretrained = nn.Identity()
        self.scratch = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True)


def build_midas_model(size: str) -> nn.Module:
    if size not in {"s", "l"}:
        raise ValueError(f"Unknown MiDaS size {size!r}; expected 's' or 'l'.")
    return _MiDaSScaffold(size)
