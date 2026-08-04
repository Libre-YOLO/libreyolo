"""Training-time validation loss adapter for RT-DETR detection."""

from __future__ import annotations

from torch import nn

from ..base.detr_validation_loss import DETRValidationLoss
from .nn import RTDETRModel


class RTDETRValidationLoss(DETRValidationLoss):
    """Evaluate RT-DETR's criterion from the validation forward pass."""

    family = "RT-DETR"

    def __init__(self, model: nn.Module, criterion: nn.Module) -> None:
        if not isinstance(model, RTDETRModel):
            raise TypeError("RT-DETR validation loss supports the detect model only")
        super().__init__(model, criterion)


__all__ = ["RTDETRValidationLoss"]
