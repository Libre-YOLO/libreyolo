"""Validation adapter for variable-resolution RetinaNet inputs."""

from __future__ import annotations

import logging

from ...validation.detection_validator import DetectionValidator

logger = logging.getLogger(__name__)


class RetinaNetValidator(DetectionValidator):
    """Run one aspect-preserved, upstream-sized image per validation batch."""

    def _setup_dataloader(self):
        if self.config.batch_size != 1:
            logger.info(
                "RetinaNet validation uses batch_size=1 because upstream "
                "aspect-resized images have variable shapes."
            )
            self.config.batch_size = 1
        return super()._setup_dataloader()


__all__ = ["RetinaNetValidator"]
