"""Training-time validation loss adapter for YOLO-NAS detection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..base.validation_loss import padded_targets_to_flat_pixels
from .loss import PPYoloELoss
from .nn import LibreYOLONASModel


class YOLONASValidationLoss:
    """Evaluate PP-YOLOE's training loss from eval-mode raw head outputs."""

    max_labels = 100

    def __init__(self, model: nn.Module, *, max_labels: int) -> None:
        if type(model) is not LibreYOLONASModel:
            raise TypeError(
                "YOLO-NAS validation loss supports the detect model only"
            )

        self.max_labels = int(max_labels)
        if self.max_labels < 1:
            raise ValueError("YOLO-NAS validation-loss max_labels must be at least 1")
        self.device = next(model.parameters()).device
        self.num_classes = int(model.nc)
        self.loss = PPYoloELoss(
            num_classes=self.num_classes,
            use_static_assigner=False,
            use_varifocal_loss=True,
            distributed_normalize=False,
        ).to(self.device)

    def __call__(
        self,
        predictions: Any,
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> Mapping[str, torch.Tensor | float]:
        del image_size  # YOLO-NAS assigns in pixels; the head carries the anchors.
        if not isinstance(predictions, Mapping) or "raw_predictions" not in predictions:
            raise ValueError(
                "YOLO-NAS validation loss requires eval output containing "
                "raw_predictions"
            )

        prepared = padded_targets_to_flat_pixels(
            targets[:, : self.max_labels],
            num_classes=self.num_classes,
            device=self.device,
            family="YOLO-NAS",
        )
        _, log_losses = self.loss(predictions["raw_predictions"], prepared)
        # ``log_losses`` is [cls, iou, dfl, total], each already multiplied by
        # its configured weight, so the three components sum to the total.
        return {
            "loss": log_losses[3],
            "loss/cls": log_losses[0],
            "loss/iou": log_losses[1],
            "loss/dfl": log_losses[2],
        }


__all__ = ["YOLONASValidationLoss"]
