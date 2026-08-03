"""Validation adapter for variable-resolution FCOS input."""

from __future__ import annotations

import logging

from ...validation.detection_validator import DetectionValidator

logger = logging.getLogger(__name__)


class FCOSValidator(DetectionValidator):
    """Run aspect-resized images one at a time because padded widths vary."""

    def _setup_dataloader(self):
        if self.config.batch_size != 1:
            logger.info(
                "FCOS validation uses batch_size=1 because aspect-preserving "
                "inputs can have different padded dimensions."
            )
            self.config.batch_size = 1
        return super()._setup_dataloader()


__all__ = ["FCOSValidator"]
