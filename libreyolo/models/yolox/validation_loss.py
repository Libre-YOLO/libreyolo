"""Validation-loss adapter for YOLOX (and the YOLOv7 trainer that reuses it).

Unlike the dense-head g2 detectors, this head's eval branch sigmoids obj/cls
and skips the grid bookkeeping ``get_losses`` consumes, so the inference
tensor alone cannot be scored. A scoped ``emit_loss_outputs`` flag makes the
head assemble the training-shaped tensors from the same conv outputs and stash
them; this adapter converts the validator's targets and calls the head's own
``get_losses``. No extra convolutions run, and the returned inference tensor
is unchanged, so predictions and mAP are untouched.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Mapping

import torch

from ..base.validation_loss import check_padded_targets, valid_target_mask

__all__ = ["YOLOXValidationLoss"]


class YOLOXValidationLoss:
    """Score a validation batch with the YOLOX head's own ``get_losses``."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        num_classes: int,
        device: torch.device,
        family: str = "yolox",
        max_labels: int | None = None,
    ) -> None:
        self.head = self._head(model, family=family)
        self.num_classes = int(num_classes)
        self.device = device
        self.family = family
        self.max_labels = max_labels

    @staticmethod
    def _head(model: torch.nn.Module, *, family: str) -> torch.nn.Module:
        for module in model.modules():
            if isinstance(getattr(module, "emit_loss_outputs", None), bool) and hasattr(
                module, "get_losses"
            ):
                return module
        raise TypeError(
            f"{family} validation loss needs a head declaring emit_loss_outputs; "
            "none was found"
        )

    def forward_scope(self) -> contextlib.AbstractContextManager:
        """Make the head stash training-shaped outputs for the whole pass."""

        @contextlib.contextmanager
        def scope() -> Iterator[None]:
            self.head.emit_loss_outputs = True
            try:
                yield
            finally:
                self.head.emit_loss_outputs = False
                self.head._loss_cache = None

        return scope()

    def __call__(
        self,
        predictions: Any,
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> Mapping[str, torch.Tensor | float]:
        del predictions, image_size  # Scored from the head's stashed tensors.
        cache = getattr(self.head, "_loss_cache", None)
        if cache is None:
            raise ValueError(
                f"{self.family} validation loss found no stashed head output; "
                "the forward scope was not active for this batch"
            )

        values = self.head.get_losses(
            None,  # get_losses accepts imgs but never reads it.
            cache["x_shifts"],
            cache["y_shifts"],
            cache["expanded_strides"],
            self._labels(targets),
            cache["outputs"],
            cache["origin_preds"],
            dtype=cache["dtype"],
        )
        return self._components(values)

    def _labels(self, targets: torch.Tensor) -> torch.Tensor:
        """Convert padded ``xyxy`` validator targets to YOLOX's label tensor.

        ``get_losses`` counts a row as real when it sums to more than zero and
        reads the first ``num_gt`` rows, so real boxes must be packed to the
        front of each image's block.
        """
        check_padded_targets(targets, family=self.family)
        source = targets[..., :5].to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=self.device.type == "cuda",
        )

        batch, max_rows = source.shape[0], source.shape[1]
        labels = source.new_zeros((batch, max_rows, 5))
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
                    f"{self.family} validation target class "
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
            raise ValueError(
                f"{self.family} criterion must return a 'total_loss' value"
            )
        # This head names components with a ``_loss`` suffix and reports
        # ``iou_loss`` already multiplied by its reg_weight, so the components
        # sum to the total. ``num_fg`` is a diagnostic, not a term.
        out: dict[str, torch.Tensor | float] = {"loss": values["total_loss"]}
        for name, value in values.items():
            if name == "total_loss" or not name.endswith("_loss"):
                continue
            out[f"loss/{name[: -len('_loss')]}"] = value
        return out
