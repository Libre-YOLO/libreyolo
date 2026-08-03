"""Training-time validation loss adapter for RT-DETRv2 detection."""

from __future__ import annotations

from torch import nn

from ..base.detr_validation_loss import DETRValidationLoss
from .nn import RTDETRv2Model


class RTDETRv2ValidationLoss(DETRValidationLoss):
    """Evaluate RT-DETRv2's criterion from the validation forward pass."""

    family = "RT-DETRv2"

    def __init__(self, model: nn.Module, criterion: nn.Module) -> None:
        if not isinstance(model, RTDETRv2Model):
            raise TypeError("RT-DETRv2 validation loss supports the detect model only")
        super().__init__(model, criterion)


__all__ = ["RTDETRv2ValidationLoss"]
