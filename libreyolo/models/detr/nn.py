"""Temporary DETR architecture shell used by the family-registration commit.

The native Apache-2.0 architecture lands in the next port commit. Keeping this
small shell separate makes the registration and architecture changes auditable
as two self-contained commits, as required by the model-port workflow.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _BoxMLP(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, 4),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if index < len(self.layers) - 1 else layer(x)
        return x


class LibreDETRModel(nn.Module):
    """Minimal shape-compatible shell; replaced by the native port next."""

    def __init__(self, size: str, nc: int, num_queries: int = 100) -> None:
        super().__init__()
        del size
        self.num_queries = num_queries
        self.backbone = nn.Identity()
        self.transformer = nn.Identity()
        self.class_embed = nn.Linear(256, nc + 1)
        self.bbox_embed = _BoxMLP()

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        hidden = x.new_zeros((x.shape[0], self.num_queries, 256))
        return {
            "pred_logits": self.class_embed(hidden),
            "pred_boxes": self.bbox_embed(hidden).sigmoid(),
        }
