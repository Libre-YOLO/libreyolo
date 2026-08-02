"""LibreSSD: wire SSD300 into the LibreYOLO checkpoint factory."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.coco import COCO91_TO_COCO80
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .nn import LibreSSDModel
from .utils import postprocess, preprocess_image


class LibreSSD(BaseModel):
    """SSD300, the historic multi-scale single-shot detector (ECCV 2016).

    LibreYOLO ships the fixed 300 px VGG16 COCO variant as an inference-only
    museum family.  It is not presented as a modern accuracy recommendation.
    """

    FAMILY = "ssd"
    FILENAME_PREFIX = "LibreSSD"
    INPUT_SIZES = {"300": 300}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = None
    TTA_FIXED_SIZE = True

    def __init__(
        self,
        model_path=None,
        size: str = "300",
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
        self._arch_num_classes = self.model.num_classes
        self.model.eval()

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Recognize SSD's VGG extras and paired MultiBox heads together."""
        cls_key = "head.classification_head.module_list.0.weight"
        reg_key = "head.regression_head.module_list.5.weight"
        extra_key = "backbone.extra.4.2.weight"
        if not all(key in weights_dict for key in (cls_key, reg_key, extra_key)):
            return False
        cls_weight = weights_dict[cls_key]
        reg_weight = weights_dict[reg_key]
        extra_weight = weights_dict[extra_key]
        return (
            getattr(cls_weight, "ndim", 0) == 4
            and int(cls_weight.shape[0]) % 4 == 0
            and tuple(reg_weight.shape[:2]) == (16, 256)
            and tuple(extra_weight.shape[:2]) == (256, 128)
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return "300" if cls.can_load(weights_dict) else None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "head.classification_head.module_list.0.weight"
        if key not in weights_dict:
            return None
        head_width = int(weights_dict[key].shape[0]) // 4
        return 80 if head_width == 91 else head_width - 1

    def _init_model(self) -> nn.Module:
        head_width = 91 if self.nb_classes == 80 else self.nb_classes + 1
        return LibreSSDModel(num_classes=head_width)

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "head": self.model.head,
        }

    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy

        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ):
        effective_size = input_size if input_size is not None else self.input_size
        if int(effective_size) != 300:
            raise ValueError("SSD300 has a fixed 300 px input canvas")
        return preprocess_image(
            image,
            input_size=effective_size,
            color_format=color_format,
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
        actual_input_size = kwargs.get("input_size", self.input_size)
        class_map = (
            COCO91_TO_COCO80
            if self._arch_num_classes == 91 and self.nb_classes == 80
            else None
        )
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
            class_map=class_map,
            input_size=actual_input_size,
        )

    def train(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "SSD is currently inference-only; MultiBox matching, hard-negative "
            "mining, and training losses are not implemented."
        )

    def _strict_loading(self) -> bool:
        return True


__all__ = ["LibreSSD"]
