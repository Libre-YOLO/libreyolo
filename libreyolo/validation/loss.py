"""Internal protocol and shared plumbing for training-time validation loss."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Protocol, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .config import ValidationConfig

logger = logging.getLogger(__name__)


class ValidationLossAdapter(Protocol):
    """Compute a model-family loss from an existing validation forward pass.

    ``image_size`` is ``(height, width)``, or ``None`` for tasks whose loss
    does not depend on input geometry (classification, restoration): those
    adapters take targets already aligned with the prediction, so there is
    nothing to rescale. Implementations must be safe to run on rank 0 while a
    distributed process group is initialized; validation in
    :class:`~libreyolo.training.trainer.BaseTrainer` is rank-0-only.

    An implementation whose family needs the model to produce extra outputs
    may also define ``forward_scope() -> ContextManager[None]``. The validator
    enters it around the whole pass, including CUDA-graph capture, and the
    model must return to its inference-shaped output on exit.
    """

    max_labels: int | None

    def __call__(
        self,
        predictions: Any,
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int] | None,
    ) -> Mapping[str, torch.Tensor | float]:
        """Return ``loss`` and optional ``loss/<component>`` scalar values."""


class ValidationLossMixin:
    """Accumulate an opt-in family loss over a validation pass.

    The bookkeeping is identical whatever the task is: average an adapter's
    scalars over batches, publish them under ``metrics/``, and never let an
    adapter failure cost the epoch its real metrics. Only the call site
    differs, so each validator decides where to call
    :meth:`_accumulate_validation_loss` and what to pass as ``predictions``.

    Validators that use this must call :meth:`_init_validation_loss` from
    ``__init__`` and :meth:`_reset_validation_loss` wherever they reset
    per-run metric state.
    """

    def _init_validation_loss(
        self,
        loss_adapter: Optional["ValidationLossAdapter"],
        *,
        config: Optional["ValidationConfig"] = None,
    ) -> None:
        config = self.config if config is None else config
        if loss_adapter is not None and getattr(config, "augment", False):
            raise ValueError(
                "Validation loss cannot be combined with augmented validation. "
                "The training validator uses augment=False."
            )
        self.loss_adapter = loss_adapter
        self._reset_validation_loss()

    def _reset_validation_loss(self) -> None:
        self._active_loss_adapter = getattr(self, "loss_adapter", None)
        self._validation_loss_totals: Dict[str, float] = {}
        self._validation_loss_batches = 0

    def _accumulate_validation_loss(
        self,
        predictions: Any,
        targets: Any,
        *,
        image_size: tuple[int, int] | None,
    ) -> None:
        """Accumulate the adapter's loss for one batch, or disable it."""
        adapter = getattr(self, "_active_loss_adapter", None)
        if adapter is None:
            return

        try:
            with self._autocast_context():
                values = adapter(predictions, targets, image_size=image_size)
            scalars = self._validation_loss_scalars(values)
        except Exception as exc:
            # Validation loss is auxiliary to the established metrics. A family
            # adapter failure must not discard mAP/accuracy or best-checkpoint
            # selection for the epoch, and partial averages are unsafe to
            # publish.
            self._active_loss_adapter = None
            self._validation_loss_totals = {}
            self._validation_loss_batches = 0
            if (
                isinstance(exc, torch.cuda.OutOfMemoryError)
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
            logger.warning(
                "Validation loss failed and was disabled for this validation "
                "pass; %s metrics will continue: %s",
                getattr(self, "task", "validation"),
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return

        for name, value in scalars.items():
            self._validation_loss_totals[name] = (
                self._validation_loss_totals.get(name, 0.0) + value
            )
        self._validation_loss_batches += 1

    @staticmethod
    def _validation_loss_scalars(
        values: Mapping[str, torch.Tensor | float],
    ) -> Dict[str, float]:
        if not isinstance(values, Mapping):
            raise TypeError("Validation loss adapter must return a mapping")
        if "loss" not in values:
            raise ValueError("Validation loss adapter must return a 'loss' value")

        scalars: Dict[str, float] = {}
        for name, value in values.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Validation loss metric names must be non-empty strings"
                )
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(
                        f"Validation loss metric {name!r} must be scalar, "
                        f"got shape {tuple(value.shape)}"
                    )
                scalars[name] = float(value.detach().float().item())
            else:
                scalars[name] = float(value)
        return scalars

    def _validation_loss_metrics(self) -> Dict[str, float]:
        batches = int(getattr(self, "_validation_loss_batches", 0))
        if batches == 0:
            return {}
        return {
            f"metrics/{name}": value / batches
            for name, value in getattr(self, "_validation_loss_totals", {}).items()
        }


__all__ = ["ValidationLossAdapter", "ValidationLossMixin"]
