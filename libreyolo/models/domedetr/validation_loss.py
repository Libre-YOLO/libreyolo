"""Training-time validation loss adapter for Dome-DETR detection."""

from __future__ import annotations

from torch import nn

from ..base.detr_validation_loss import DETRValidationLoss
from .nn import LibreDOMEDETRModel


class DOMEDETRValidationLoss(DETRValidationLoss):
    """Evaluate Dome-DETR's criterion from the validation forward pass.

    The shared DETR adapter drives the decoder's ``emit_loss_outputs`` flag so
    every layer is scored, not just ``eval_idx``. That works unchanged here:
    Dome-DETR reuses D-FINE's decoder stack, and PAQI only changes how many
    queries reach it.
    """

    family = "Dome-DETR"

    def __init__(self, model: nn.Module, criterion: nn.Module) -> None:
        if not isinstance(model, LibreDOMEDETRModel):
            raise TypeError("Dome-DETR validation loss supports the detect model only")
        super().__init__(model, criterion)


__all__ = ["DOMEDETRValidationLoss"]
