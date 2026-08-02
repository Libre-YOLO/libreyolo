"""Internal protocol for training-time validation loss adapters."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import torch


class ValidationLossAdapter(Protocol):
    """Compute a model-family loss from an existing validation forward pass.

    ``image_size`` is ``(height, width)``. Implementations must be safe to run
    on rank 0 while a distributed process group is initialized; validation in
    :class:`~libreyolo.training.trainer.BaseTrainer` is rank-0-only.
    """

    max_labels: int | None

    def __call__(
        self,
        predictions: Any,
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> Mapping[str, torch.Tensor | float]:
        """Return ``loss`` and optional ``loss/<component>`` scalar values."""


__all__ = ["ValidationLossAdapter"]
