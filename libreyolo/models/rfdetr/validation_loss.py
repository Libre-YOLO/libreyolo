"""Training-time validation loss adapter for RF-DETR detection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .nn import LibreRFDETRModel


class RFDETRValidationLoss:
    """Evaluate RF-DETR's full criterion from its eval-mode output dictionary."""

    max_labels = 300

    def __init__(self, model: nn.Module) -> None:
        if type(model) is not LibreRFDETRModel or any(
            bool(getattr(model, name, False))
            for name in (
                "segmentation",
                "pose",
                "obb",
                "classification",
                "semantic",
            )
        ):
            raise TypeError(
                "RF-DETR validation loss supports the standard detect model only"
            )

        self.device = next(model.parameters()).device
        self.num_classes = int(model.nb_classes)
        self.criterion, _ = model.build_criterion_and_postprocess(
            distributed_normalize=False
        )
        self.criterion.to(self.device)
        self.criterion.eval()

    def __call__(
        self,
        predictions: Any,
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> Mapping[str, torch.Tensor | float]:
        if not isinstance(predictions, Mapping) or not {
            "pred_logits",
            "pred_boxes",
        }.issubset(predictions):
            raise ValueError(
                "RF-DETR validation loss requires pred_logits and pred_boxes"
            )

        target_list = self._prepare_targets(
            targets[:, : self.max_labels],
            image_size=image_size,
            num_classes=self.num_classes,
            device=self.device,
        )
        loss_dict = self.criterion(predictions, target_list)
        weighted = [
            value * self.criterion.weight_dict[name]
            for name, value in loss_dict.items()
            if name in self.criterion.weight_dict
        ]
        if not weighted:
            raise RuntimeError("RF-DETR criterion returned no weighted losses")
        total = sum(weighted[1:], weighted[0])

        zero = predictions["pred_logits"].sum() * 0.0

        def component(prefix: str) -> torch.Tensor:
            return sum(
                (
                    value
                    for name, value in loss_dict.items()
                    if name == prefix or name.startswith(prefix + "_")
                ),
                zero,
            )

        return {
            "loss": total,
            "loss/ce": component("loss_ce"),
            "loss/bbox": component("loss_bbox"),
            "loss/giou": component("loss_giou"),
        }

    @staticmethod
    def _prepare_targets(
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int],
        num_classes: int,
        device: torch.device,
    ) -> list[dict[str, torch.Tensor]]:
        """Convert padded validation ``xyxy,class`` pixels to DETR targets."""
        if targets.ndim != 3 or targets.shape[-1] < 5:
            raise ValueError(
                "RF-DETR validation targets must have shape [batch, labels, >=5]"
            )

        source = targets[..., :5].to(
            device=device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        height, width = image_size
        scale = torch.tensor(
            [width, height, width, height],
            dtype=torch.float32,
            device=device,
        )
        target_list = []
        for image_targets in source:
            valid = (image_targets[:, 2] > image_targets[:, 0]) & (
                image_targets[:, 3] > image_targets[:, 1]
            )
            rows = image_targets[valid]
            labels = rows[:, 4]
            invalid_labels = (
                (labels < 0) | (labels >= num_classes) | (labels != labels.round())
            )
            if invalid_labels.any():
                bad_label = float(labels[invalid_labels][0].item())
                raise ValueError(
                    f"RF-DETR validation target class {bad_label:g} is outside "
                    f"[0, {num_classes - 1}]"
                )

            xyxy = (rows[:, :4] / scale).clamp(0.0, 1.0)
            boxes = torch.empty_like(xyxy)
            boxes[:, :2] = (xyxy[:, :2] + xyxy[:, 2:]) * 0.5
            boxes[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
            target_list.append(
                {
                    "labels": labels.long(),
                    "boxes": boxes,
                }
            )
        return target_list


__all__ = ["RFDETRValidationLoss"]
