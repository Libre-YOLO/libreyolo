"""FOMO training loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FOMOLoss(nn.Module):
    """Weighted pixel-wise CrossEntropyLoss for FOMO grid targets."""

    def __init__(
        self,
        num_classes: int = 1,
        fg_weight: float = 100.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        weights = torch.ones(1 + num_classes, dtype=torch.float32, device=device)
        weights[1:] = fg_weight
        self.register_buffer("weights", weights)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> dict:
        """Compute FOMO loss."""
        loss = F.cross_entropy(logits, targets, weight=self.weights)
        return {"total_loss": loss, "ce": float(loss.item())}
