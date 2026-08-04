"""Training-time validation loss adapter for D-FINE detection."""

from __future__ import annotations

from torch import nn

from ..base.detr_validation_loss import DETRValidationLoss
from .nn import LibreDFINEModel


class DFINEValidationLoss(DETRValidationLoss):
    """Evaluate D-FINE's criterion from the validation forward pass."""

    family = "D-FINE"

    def __init__(self, model: nn.Module, criterion: nn.Module) -> None:
        if not isinstance(model, LibreDFINEModel) or getattr(
            model.decoder, "enable_mask_head", False
        ):
            raise TypeError("D-FINE validation loss supports the detect model only")
        super().__init__(model, criterion)


__all__ = ["DFINEValidationLoss"]
