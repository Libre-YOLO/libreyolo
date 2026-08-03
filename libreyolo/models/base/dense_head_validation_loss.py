"""Validation-loss adapter for the dense-head g2 detectors.

RTMDet and PicoDet both return their raw head outputs
(``(cls_scores, bbox_preds)``) from ``forward`` in eval as well as train, and
both criteria take the same ``(cls_scores, bbox_preds, gt_boxes_list,
gt_labels_list)`` call. So the validator's existing raw predictions are
already what the criterion wants: no scoped output flag, no second forward,
just the target conversion and the assignment the criterion runs itself.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .validation_loss import check_padded_targets, valid_target_mask

__all__ = ["DenseHeadValidationLoss"]

# The criteria return their aggregate under this key; everything else prefixed
# ``loss_`` is a weighted component that sums to it.
_TOTAL_KEY = "total_loss"


class DenseHeadValidationLoss:
    """Run a dense-head detector's own criterion over a validation batch."""

    def __init__(
        self,
        criterion: torch.nn.Module,
        *,
        num_classes: int,
        device: torch.device,
        family: str,
        max_labels: int | None = None,
    ) -> None:
        self.criterion = criterion
        self.num_classes = int(num_classes)
        self.device = device
        self.family = family
        self.max_labels = max_labels
        self.criterion.eval()

    def __call__(
        self,
        predictions: Any,
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> Mapping[str, torch.Tensor | float]:
        del image_size  # Validator targets are already in pixel coordinates.
        cls_scores, bbox_preds = self._head_outputs(predictions)
        gt_boxes_list, gt_labels_list = self._targets(targets)
        values = self.criterion(cls_scores, bbox_preds, gt_boxes_list, gt_labels_list)
        return self._components(values)

    def _head_outputs(
        self, predictions: Any
    ) -> tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        if isinstance(predictions, Mapping):
            predictions = (
                predictions.get("cls_scores"),
                predictions.get("bbox_preds"),
            )
        if not isinstance(predictions, (list, tuple)) or len(predictions) != 2:
            raise TypeError(
                f"{self.family} validation loss needs the (cls_scores, "
                f"bbox_preds) head output, got {type(predictions).__name__}"
            )
        cls_scores, bbox_preds = predictions
        if not isinstance(cls_scores, (list, tuple)) or not isinstance(
            bbox_preds, (list, tuple)
        ):
            raise TypeError(
                f"{self.family} validation loss needs per-level head output "
                "sequences"
            )
        return cls_scores, bbox_preds

    def _targets(
        self, targets: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Split padded ``xyxy`` validator targets into the criterion's lists.

        The trainers build these from ``cxcywh`` rows; the validator hands over
        ``xyxy`` already, so this converts one step less than training does and
        lands on the same values.
        """
        check_padded_targets(targets, family=self.family)
        source = targets[..., :5].to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=self.device.type == "cuda",
        )

        gt_boxes_list: list[torch.Tensor] = []
        gt_labels_list: list[torch.Tensor] = []
        for image_targets in source:
            rows = image_targets[valid_target_mask(image_targets)]
            labels = rows[:, 4]
            invalid = (
                (labels < 0)
                | (labels >= self.num_classes)
                | (labels != labels.round())
            )
            if invalid.any():
                raise ValueError(
                    f"{self.family} validation target class "
                    f"{float(labels[invalid][0].item()):g} is outside "
                    f"[0, {self.num_classes - 1}]"
                )
            gt_boxes_list.append(rows[:, :4])
            gt_labels_list.append(labels.long())
        return gt_boxes_list, gt_labels_list

    def _components(self, values: Any) -> Mapping[str, torch.Tensor | float]:
        if not isinstance(values, Mapping) or _TOTAL_KEY not in values:
            raise ValueError(
                f"{self.family} criterion must return a {_TOTAL_KEY!r} value"
            )
        out: dict[str, torch.Tensor | float] = {"loss": values[_TOTAL_KEY]}
        for name, value in values.items():
            # ``num_pos`` and friends are diagnostics, not weighted terms.
            if name == _TOTAL_KEY or not name.startswith("loss_"):
                continue
            out[f"loss/{name[len('loss_'):]}"] = value
        return out
