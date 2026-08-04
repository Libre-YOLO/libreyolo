"""Validation adapter for variable-resolution Faster R-CNN input."""

from __future__ import annotations

import logging
from typing import Any

from ...validation.detection_validator import DetectionValidator

logger = logging.getLogger(__name__)


class FasterRCNNValidator(DetectionValidator):
    """Run unresized images one at a time through the in-graph transform."""

    def _setup_dataloader(self):
        if self.config.batch_size != 1:
            logger.info(
                "Faster R-CNN validation uses batch_size=1 because source "
                "images retain their original dimensions."
            )
            self.config.batch_size = 1
        return super()._setup_dataloader()

    def _slice_batch_predictions(self, preds: Any, batch_idx: int) -> Any:
        if not isinstance(preds, list):
            raise TypeError(
                "Faster R-CNN validation expected a list of detection dicts"
            )
        return preds[batch_idx]
