"""Training-time validation loss adapter for DEIMv2 detection."""

from __future__ import annotations

from typing import Any, Callable

from torch import nn

from ..base.detr_validation_loss import DETRValidationLoss
from .nn import LibreDEIMv2Model


class DEIMv2ValidationLoss(DETRValidationLoss):
    """Evaluate DEIMv2's criterion from the validation forward pass."""

    family = "DEIMv2"

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        *,
        epoch: Callable[[], int] | int = 0,
    ) -> None:
        if type(model) is not LibreDEIMv2Model:
            raise TypeError("DEIMv2 validation loss supports the detect model only")
        super().__init__(model, criterion)
        self._epoch = epoch

    def criterion_kwargs(self) -> dict[str, Any]:
        # DEIMv2's matcher switches at a configured epoch. Reading the epoch
        # per call keeps validation on the same matcher training is using.
        epoch = self._epoch() if callable(self._epoch) else self._epoch
        return {"epoch": int(epoch)}


__all__ = ["DEIMv2ValidationLoss"]
