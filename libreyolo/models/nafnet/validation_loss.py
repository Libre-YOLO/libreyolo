"""Validation-loss adapter for NAFNet paired restoration.

The restore validator already produces the cropped model output it scores
PSNR/SSIM from, and the criterion is a pixel-wise charbonnier against the
paired target: no assignment, no second forward. The adapter is handed the
output *before* the validator's display clamp, because training's loss sees
the raw output and the two numbers are only comparable if this one does too.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

__all__ = ["NAFNetValidationLoss"]


class NAFNetValidationLoss:
    """Charbonnier loss over the restored output and its paired target."""

    # Restoration targets are images, not padded label rows.
    max_labels = None

    def __init__(self, *, device: torch.device) -> None:
        self.device = device

    def __call__(
        self,
        predictions: Any,
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> Mapping[str, torch.Tensor | float]:
        del image_size  # The pair is already aligned by the validator.
        # Imported here so the one definition in the trainer stays canonical.
        from .trainer import charbonnier_loss

        pred = self._as_batch(predictions, name="prediction")
        target = self._as_batch(targets, name="target").to(
            device=pred.device, dtype=pred.dtype
        )
        if pred.shape != target.shape:
            raise ValueError(
                "NAFNet validation loss needs matching prediction and target "
                f"shapes, got {tuple(pred.shape)} and {tuple(target.shape)}"
            )
        loss = charbonnier_loss(pred, target)
        return {"loss": loss, "loss/restore": loss}

    def _as_batch(self, value: Any, *, name: str) -> torch.Tensor:
        if isinstance(value, Mapping):
            value = value.get("restored", value.get("predictions"))
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"NAFNet validation loss needs a tensor {name}, got "
                f"{type(value).__name__}"
            )
        if value.ndim != 4:
            raise ValueError(
                f"NAFNet validation {name} must be [batch, 3, H, W], got "
                f"shape {tuple(value.shape)}"
            )
        return value.to(device=self.device, dtype=torch.float32)
