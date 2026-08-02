"""LibreMaskRCNN: wire Mask R-CNN into the LibreYOLO factory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch.nn as nn

from ...validation.preprocessors import FasterRCNNValPreprocessor
from ..faster_rcnn.model import LibreFasterRCNN
from .nn import LibreMaskRCNNModel


class LibreMaskRCNN(LibreFasterRCNN):
    """Mask R-CNN, the defining two-stage instance-segmentation architecture."""

    FAMILY = "mask_rcnn"
    FILENAME_PREFIX = "LibreMaskRCNN"
    INPUT_SIZES = {"r50": 800}
    SUPPORTED_TASKS = ("detect", "segment")
    DEFAULT_TASK = "segment"
    TASK_INPUT_SIZES = {
        "detect": INPUT_SIZES,
        "segment": INPUT_SIZES,
    }
    TRAIN_CONFIG = None
    val_preprocessor_class = FasterRCNNValPreprocessor

    def __init__(
        self,
        model_path=None,
        size: str = "r50",
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Claim only two-stage checkpoints carrying the Mask R-CNN head."""
        return (
            "roi_heads.mask_predictor.mask_fcn_logits.weight" in weights_dict
            and "roi_heads.box_predictor.cls_score.weight" in weights_dict
            and any(key.startswith("rpn.head.") for key in weights_dict)
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return "r50" if cls.can_load(weights_dict) else None

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        detected = super().detect_size_from_filename(filename)
        if detected is not None:
            return detected
        lower = Path(filename).name.lower()
        if lower.startswith("maskrcnn_resnet50_fpn_v2"):
            return "r50"
        return None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "segment" if cls.can_load(state_dict) else None

    def _init_model(self) -> nn.Module:
        head_width = 91 if self.nb_classes == 80 else self.nb_classes + 1
        return LibreMaskRCNNModel(size=self.size, num_classes=head_width)

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Mask R-CNN is currently inference-only; sampled-RoI and mask "
            "training are not implemented."
        )
