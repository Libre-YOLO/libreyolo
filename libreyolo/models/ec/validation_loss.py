"""Training-time validation loss adapter for EC detection."""

from __future__ import annotations

from torch import nn

from ..base.detr_validation_loss import DETRValidationLoss
from .nn import LibreECModel


class ECValidationLoss(DETRValidationLoss):
    """Evaluate EC's criterion from the validation forward pass."""

    family = "EC"

    def __init__(self, model: nn.Module, criterion: nn.Module) -> None:
        if type(model) is not LibreECModel:
            raise TypeError("EC validation loss supports the detect model only")
        super().__init__(model, criterion)


__all__ = ["ECValidationLoss"]
