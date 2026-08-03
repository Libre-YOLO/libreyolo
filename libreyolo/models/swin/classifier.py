"""Swin V1 classification head built on LibreYOLO's shared native tower.

The shared :class:`SwinBackbone` already mirrors the Apache-2.0 timm Swin V1
module graph. This module adds only the final normalization, global average
pool, and linear ImageNet head. Parameter names remain aligned with timm so
the released Microsoft checkpoints load without changing learned tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import SWIN_CONFIGS
from .nn import SwinBackbone, SwinDims


class SwinClassifierHead(nn.Module):
    """Global-average classifier head with timm-compatible ``head.fc`` keys."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=(1, 2)))


class SwinClassifier(SwinBackbone):
    """Swin V1 tiny/small/base/large image classifier."""

    def __init__(self, size: str = "t", num_classes: int = 1000) -> None:
        if size not in SWIN_CONFIGS:
            raise ValueError(
                f"Unknown Swin size {size!r}; choose from {list(SWIN_CONFIGS)}."
            )
        spec = SWIN_CONFIGS[size]
        dims = SwinDims(
            embed_dim=spec["embed_dim"],
            depths=spec["depths"],
            num_heads=spec["num_heads"],
            window_size=7,
            patch_size=4,
            out_indices=(3,),
            tf_order=False,
        )
        super().__init__(dims)
        # At the fixed 224px classifier resolution, the final stage is one
        # 7x7 window. timm disables cyclic shift when a stage is no larger
        # than its window; the shared dense-backbone path normally runs at
        # larger resolutions and therefore keeps the configured shift.
        # LibreSwin rejects non-native resolutions before this graph runs.
        for block in self.layers[-1].blocks:
            block.shift_size = 0
        self.size = size
        self.num_classes = num_classes
        self.num_features = spec["embed_dim"] * 8
        self.norm = nn.LayerNorm(self.num_features)
        self.head = SwinClassifierHead(self.num_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def reset_classifier(self, num_classes: int) -> None:
        """Replace the classification layer while preserving device and dtype."""
        self.num_classes = num_classes
        weight = self.head.fc.weight
        self.head.fc = nn.Linear(self.num_features, num_classes).to(
            device=weight.device, dtype=weight.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))


__all__ = ["SwinClassifier", "SwinClassifierHead"]
