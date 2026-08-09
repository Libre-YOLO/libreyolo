"""Validation-loss adapter for YOLOv7.

Simpler than its YOLOX sibling: this net returns the three raw head maps from
``forward`` whether or not targets are given, and the criterion consumes
exactly those. The validator's raw predictions are therefore already the
criterion's input, so there is no scoped output flag and no second forward.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ..base.validation_loss import check_padded_targets, valid_target_mask

__all__ = ["YOLOv7ValidationLoss"]


class YOLOv7ValidationLoss:
    """Score a validation batch with YOLOv7's own SimOTA criterion."""

    def __init__(
        self,
        criterion: Any,
        *,
        num_classes: int,
        device: torch.device,
        max_labels: int | None = None,
    ) -> None:
        self.criterion = criterion
        self.num_classes = int(num_classes)
        self.device = device
        self.family = "yolo7"
        self.max_labels = max_labels

    def __call__(
        self,
        predictions: Any,
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> Mapping[str, torch.Tensor | float]:
        del image_size  # Validator targets are already in pixel coordinates.
        return self._components(
            self.criterion(self._head_maps(predictions), self._labels(targets))
        )

    def _head_maps(self, predictions: Any) -> Sequence[torch.Tensor]:
        if isinstance(predictions, Mapping):
            predictions = predictions.get("head_out", predictions.get("predictions"))
        if not isinstance(predictions, (list, tuple)) or not predictions:
            raise TypeError(
                "yolo7 validation loss needs the raw head maps, got "
                f"{type(predictions).__name__}"
            )
        return predictions

    def _labels(self, targets: torch.Tensor) -> torch.Tensor:
        """Convert padded ``xyxy`` validator targets to ``[cls, cx, cy, w, h]``.

        The trainer builds the same tensor from normalized ``xyxy``; the
        validator's rows are already in pixels, so this converts one step
        less and lands on the same values. Zero-padded rows stay all-zero,
        which is how the criterion counts real boxes.
        """
        check_padded_targets(targets, family=self.family)
        source = targets[..., :5].to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=self.device.type == "cuda",
        )

        labels = source.new_zeros(source.shape[:2] + (5,))
        for index, image_targets in enumerate(source):
            rows = image_targets[valid_target_mask(image_targets)]
            if rows.numel() == 0:
                continue
            classes = rows[:, 4]
            invalid = (
                (classes < 0)
                | (classes >= self.num_classes)
                | (classes != classes.round())
            )
            if invalid.any():
                raise ValueError(
                    f"yolo7 validation target class "
                    f"{float(classes[invalid][0].item()):g} is outside "
                    f"[0, {self.num_classes - 1}]"
                )
            xyxy = rows[:, :4]
            labels[index, : rows.shape[0], 0] = classes
            labels[index, : rows.shape[0], 1:3] = (xyxy[:, :2] + xyxy[:, 2:]) * 0.5
            labels[index, : rows.shape[0], 3:5] = xyxy[:, 2:] - xyxy[:, :2]
        return labels

    def _components(self, values: Any) -> Mapping[str, torch.Tensor | float]:
        if not isinstance(values, Mapping) or "total_loss" not in values:
            raise ValueError("yolo7 criterion must return a 'total_loss' value")
        # ``iou_loss`` already carries its reg_weight, so the components sum to
        # the total; ``num_fg`` is a diagnostic, not a term.
        out: dict[str, torch.Tensor | float] = {"loss": values["total_loss"]}
        for name, value in values.items():
            if name == "total_loss" or not name.endswith("_loss"):
                continue
            out[f"loss/{name[: -len('_loss')]}"] = value
        return out
