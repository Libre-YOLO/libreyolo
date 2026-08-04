"""Validation adapter for variable-resolution Mask R-CNN input."""

from __future__ import annotations

import logging
from typing import Any

from ...validation.detection_validator import SegmentationValidator

logger = logging.getLogger(__name__)


class MaskRCNNValidator(SegmentationValidator):
    """Run unresized images one at a time through the in-graph transform."""

    def _setup_dataloader(self):
        if self.config.batch_size != 1:
            logger.info(
                "Mask R-CNN validation uses batch_size=1 because source "
                "images retain their original dimensions."
            )
            self.config.batch_size = 1
        return super()._setup_dataloader()

    def _slice_batch_predictions(self, preds: Any, batch_idx: int) -> Any:
        if not isinstance(preds, list):
            raise TypeError(
                "Mask R-CNN validation expected a list of detection dictionaries"
            )
        return preds[batch_idx]
