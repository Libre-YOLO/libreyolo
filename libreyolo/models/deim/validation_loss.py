"""Training-time validation loss adapter for DEIM detection."""

from __future__ import annotations

from torch import nn

from ..base.detr_validation_loss import DETRValidationLoss
from ..dfine.nn import LibreDFINEModel
from .nn import LibreDEIMModel


class DEIMValidationLoss(DETRValidationLoss):
    """Evaluate DEIM's criterion from the validation forward pass.

    RT-DETRv4 is DEIM's criterion over the D-FINE model, so both model
    classes are accepted here.
    """

    family = "DEIM"

    def __init__(self, model: nn.Module, criterion: nn.Module) -> None:
        supported = isinstance(model, (LibreDEIMModel, LibreDFINEModel))
        if not supported or getattr(model.decoder, "enable_mask_head", False):
            raise TypeError("DEIM validation loss supports the detect model only")
        super().__init__(model, criterion)


__all__ = ["DEIMValidationLoss"]
