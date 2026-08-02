"""Temporary DeiT construction skeleton.

The complete native transformer graph is added in the architecture commit.
This small graph keeps the family wrapper constructible while the factory
registration is introduced separately.
"""

from __future__ import annotations

import torch
import torch.nn as nn


ARCH_DEFS = {
    "t": {"embed_dim": 192, "depth": 12, "num_heads": 3},
    "s": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "b": {"embed_dim": 768, "depth": 12, "num_heads": 12},
}


class _PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class DeiT(nn.Module):
    """Constructible placeholder with DeiT's public module surface."""

    def __init__(self, size: str = "t", num_classes: int = 1000) -> None:
        super().__init__()
        if size not in ARCH_DEFS:
            raise ValueError(f"Unknown DeiT size {size!r}; choose from {list(ARCH_DEFS)}.")
        dim = int(ARCH_DEFS[size]["embed_dim"])
        self.size = size
        self.num_classes = num_classes
        self.embed_dim = dim
        self.patch_embed = _PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, dim))
        self.blocks = nn.Sequential(*[nn.Identity() for _ in range(12)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        weight = self.head.weight
        self.head = nn.Linear(self.embed_dim, num_classes).to(
            device=weight.device, dtype=weight.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        return self.head(self.norm(cls)[:, 0])
