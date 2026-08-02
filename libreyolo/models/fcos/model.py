"""LibreFCOS: wire the FCOS family into the LibreYOLO factory."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .nn import LibreFCOSModel
from .utils import preprocess_image, preprocess_numpy


class LibreFCOS(BaseModel):
    """FCOS ResNet-50/FPN, the landmark anchor-free per-pixel detector."""

    FAMILY = "fcos"
    FILENAME_PREFIX = "LibreFCOS"
    INPUT_SIZES = {"r50": 800}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = None

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
        if isinstance(model_path, str):
            self._load_weights(model_path)
        self.model.eval()

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Claim FCOS only through its centerness branch and P6/P7 FPN."""
        return (
            "head.regression_head.bbox_ctrness.weight" in weights_dict
            and "head.classification_head.cls_logits.weight" in weights_dict
            and "backbone.fpn.extra_blocks.p6.weight" in weights_dict
            and "backbone.fpn.extra_blocks.p7.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Recognize the only permissive pretrained variant, ResNet-50/FPN."""
        if not cls.can_load(weights_dict):
            return None
        stem = weights_dict.get("backbone.body.conv1.weight")
        if stem is not None and tuple(stem.shape) == (64, 3, 7, 7):
            return "r50"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "head.classification_head.cls_logits.weight"
        if key not in weights_dict:
            return None
        width = int(weights_dict[key].shape[0])
        return 80 if width == 91 else width

    def _init_model(self) -> nn.Module:
        head_width = 91 if self.nb_classes == 80 else self.nb_classes
        return LibreFCOSModel(num_classes=head_width)

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {"backbone": self.model.backbone, "head": self.model.head}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ):
        return preprocess_image(
            image,
            color_format=color_format,
            input_size=int(input_size or self.input_size),
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> dict:
        del output, conf_thres, iou_thres, original_size, max_det, kwargs
        return {
            "num_detections": 0,
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "classes": np.zeros((0,), dtype=np.int64),
        }

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "FCOS is currently inference-only; dense assignment and loss "
            "training are not implemented."
        )

    def _strict_loading(self) -> bool:
        return True


__all__ = ["LibreFCOS"]
