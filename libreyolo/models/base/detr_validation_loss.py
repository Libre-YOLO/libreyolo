"""Shared validation-loss adapter for the DETR-line detection families.

D-FINE, DEIM, DEIMv2, EC, RT-DETR and RT-DETRv2 all pair a set-prediction
decoder with a criterion that consumes the training-shaped output dict and
returns already-weighted named losses. The per-family adapters differ only in
which criterion they build, so the mechanics live here.

Denoising is out of scope by construction: contrastive-denoising groups need
the ground truth at forward time and validation forwards without it. The
reported total therefore covers the main, auxiliary-decoder, encoder and
pre-decoder terms, never the ``dn_*`` ones.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, ContextManager

import torch
from torch import nn

from .validation_loss import (
    emit_loss_outputs,
    loss_output_modules,
    padded_targets_to_detr,
)

# Criterions decorate a base loss name per supervised output: ``loss_bbox``
# becomes ``loss_bbox_aux_0``, ``loss_bbox_enc_0``, ``loss_bbox_pre`` and so
# on. Components are reported per base name, so the decorations are stripped.
_DECORATION = re.compile(r"_(?:aux|enc|dn)_\d+$|_dn_pre$|_dn_final$|_pre$")

# RT-DETR's criterion returns the pre-summed total alongside its terms.
_AGGREGATE_KEYS = frozenset({"total_loss"})


def _base_name(name: str) -> str:
    while True:
        stripped = _DECORATION.sub("", name)
        if stripped == name:
            return name
        name = stripped


class DETRValidationLoss:
    """Evaluate a DETR-line criterion from the validation forward pass."""

    #: Human-readable family name used in error messages.
    family: str = "DETR"
    max_labels = 300

    def __init__(self, model: nn.Module, criterion: nn.Module) -> None:
        self._check_eval_layer_is_last(model)
        self.model = model
        self.criterion = criterion.eval()
        self.device = next(model.parameters()).device
        self.num_classes = int(criterion.num_classes)

    def _check_eval_layer_is_last(self, model: nn.Module) -> None:
        """Reject a decoder whose inference layer is not its last layer.

        With ``eval_idx == num_layers - 1`` (every shipped size) the extra
        layers scored under the loss scope are exactly the auxiliary ones and
        the reported main terms stay the layer the metrics come from. A
        smaller ``eval_idx`` would silently report a different layer.
        """
        for module in loss_output_modules(model):
            eval_idx = getattr(module, "eval_idx", None)
            num_layers = getattr(module, "num_layers", None)
            if eval_idx is None or num_layers is None:
                continue
            if int(eval_idx) != int(num_layers) - 1:
                raise TypeError(
                    f"{self.family} validation loss needs the last decoder "
                    f"layer to be the evaluated one, got eval_idx={eval_idx} "
                    f"of {num_layers} layers"
                )

    def forward_scope(self) -> ContextManager[None]:
        """Turn on the decoder's training-shaped output for one pass."""
        return emit_loss_outputs(self.model)

    def __call__(
        self,
        predictions: Any,
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> Mapping[str, torch.Tensor | float]:
        self._check_predictions(predictions)
        target_list = padded_targets_to_detr(
            targets[:, : self.max_labels],
            image_size=image_size,
            num_classes=self.num_classes,
            device=self.device,
            family=self.family,
        )
        loss_dict = self.criterion(predictions, target_list, **self.criterion_kwargs())
        return self._report(loss_dict, predictions)

    def criterion_kwargs(self) -> dict[str, Any]:
        """Extra keyword arguments this family's criterion takes."""
        return {}

    def _check_predictions(self, predictions: Any) -> None:
        if not isinstance(predictions, Mapping) or not {
            "pred_logits",
            "pred_boxes",
        }.issubset(predictions):
            raise ValueError(
                f"{self.family} validation loss requires pred_logits and pred_boxes"
            )
        if "aux_outputs" not in predictions:
            raise ValueError(
                f"{self.family} validation loss requires the decoder's "
                "loss-shaped output; the forward scope was not active"
            )

    def _report(
        self,
        loss_dict: Mapping[str, Any],
        predictions: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor | float]:
        # These criterions apply ``weight_dict`` internally, so the terms are
        # already weighted and the total is their plain sum. Reporting per base
        # name therefore makes the components sum back to the total, the same
        # contract the YOLO9 and RF-DETR adapters follow.
        terms = {
            name: value
            for name, value in loss_dict.items()
            if isinstance(value, torch.Tensor) and name not in _AGGREGATE_KEYS
        }
        if not terms:
            raise RuntimeError(f"{self.family} criterion returned no losses")

        zero = predictions["pred_logits"].sum() * 0.0
        components: dict[str, torch.Tensor | float] = {}
        for name, value in terms.items():
            label = _base_name(name).removeprefix("loss_") or "other"
            key = f"loss/{label}"
            components[key] = components.get(key, zero) + value

        values: dict[str, torch.Tensor | float] = {"loss": sum(terms.values(), zero)}
        values.update(components)
        return values


__all__ = ["DETRValidationLoss"]
