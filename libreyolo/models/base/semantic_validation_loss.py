"""Validation-loss adapter for the dense semantic-segmentation families.

SegFormer, LingBotVision and the RF-DETR semantic segmenter (which the DINOv2
family reuses wholesale) all compute the same thing inside ``forward`` when
handed targets: upsample the decode-head logits to the target resolution, then
``cross_entropy`` with ``ignore_index``. Recomputing it here from the logits
the semantic validator already extracted is the same arithmetic without a
second forward, and it keeps validation's forward signature (no ``targets=``)
unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

__all__ = ["SemanticValidationLoss", "SemanticValidationLossMixin"]


class SemanticValidationLoss:
    """Cross-entropy over dense logits already aligned to the target mask."""

    # Semantic targets are dense masks, not padded label rows.
    max_labels = None

    def __init__(
        self, *, device: torch.device, family: str, ignore_index: int
    ) -> None:
        self.device = device
        self.family = family
        self.ignore_index = int(ignore_index)

    def __call__(
        self,
        predictions: Any,
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> Mapping[str, torch.Tensor | float]:
        del image_size  # The validator aligned logits to the mask already.
        logits = self._logits(predictions)
        mask = self._mask(targets)

        if tuple(logits.shape[-2:]) != tuple(mask.shape[-2:]):
            # Training upsamples to the target resolution before scoring, so
            # match that rather than scoring at a different scale.
            logits = F.interpolate(
                logits, size=mask.shape[-2:], mode="bilinear", align_corners=False
            )

        if bool((mask != self.ignore_index).any()):
            loss = F.cross_entropy(logits, mask, ignore_index=self.ignore_index)
        else:
            # Same guard the families use in training: cross_entropy is NaN
            # when every pixel is ignored, and a NaN would poison the average
            # for the whole epoch.
            loss = logits.sum() * 0.0
        return {"loss": loss, "loss/sem": loss}

    def _logits(self, predictions: Any) -> torch.Tensor:
        logits = predictions
        if isinstance(logits, Mapping):
            logits = logits.get("semantic_logits", logits.get("logits"))
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"{self.family} validation loss needs dense logits, got "
                f"{type(predictions).__name__}"
            )
        if logits.ndim != 4:
            raise ValueError(
                f"{self.family} validation logits must be [batch, classes, "
                f"H, W], got shape {tuple(logits.shape)}"
            )
        return logits.to(device=self.device, dtype=torch.float32)

    def _mask(self, targets: Any) -> torch.Tensor:
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if targets.ndim != 3:
            raise ValueError(
                f"{self.family} validation targets must be [batch, H, W] "
                f"class ids, got shape {tuple(targets.shape)}"
            )
        return targets.to(device=self.device, dtype=torch.long)


class SemanticValidationLossMixin:
    """Opt a dense semantic trainer into ``val_loss=True``.

    Mixed in ahead of the family's own trainer so it overrides whatever gate
    that trainer inherits.
    """

    def validate_validation_loss_config(self) -> None:
        if not getattr(self.config, "val_loss", False):
            return
        task = getattr(getattr(self, "wrapper_model", None), "task", "semantic")
        if task != "semantic":
            raise ValueError(
                f"val_loss=True currently supports {self.get_model_family()} "
                "semantic segmentation only; other tasks are not supported"
            )

    def build_validation_loss_adapter(self, model: torch.nn.Module):
        # IGNORE_INDEX is a class attribute on every semantic net in the repo;
        # read it from the model rather than hardcoding 255 in a second place.
        inner = getattr(model, "model", model)
        ignore_index = getattr(
            inner, "IGNORE_INDEX", getattr(model, "IGNORE_INDEX", None)
        )
        if ignore_index is None:
            raise TypeError(
                f"{self.get_model_family()} validation loss needs the model's "
                "IGNORE_INDEX; none was found"
            )
        return SemanticValidationLoss(
            device=self.device,
            family=self.get_model_family(),
            ignore_index=ignore_index,
        )
