"""LibreEfficientDet factory integration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput
from ...validation.preprocessors import EfficientDetValPreprocessor
from ..base import BaseModel
from .config import INPUT_SIZES, SCALE_CONFIGS
from .nn import LibreEfficientDetModel
from .utils import postprocess, preprocess_image

_CLASS_PREDICT_KEY = "class_net.predict.conv_pw.weight"
_BOX_PREDICT_KEY = "box_net.predict.conv_pw.weight"
_BIFPN_KEY = "fpn.cell.0.fnode.0.combine.edge_weights"
_BACKBONE_KEY = "backbone.conv_stem.weight"
_ANCHORS_PER_LOCATION = 9
_COCO_SPARSE_CLASSES = 90


class LibreEfficientDet(BaseModel):
    """EfficientDet D0-D4, the 2020 BiFPN and compound-scaling detector.

    This museum family follows the TensorFlow-ported checkpoints published by
    the Apache-2.0 ``rwightman/efficientdet-pytorch`` project. It is shipped
    inference-only; focal-loss training and anchor assignment remain upstream.
    """

    FAMILY = "efficientdet"
    FILENAME_PREFIX = "LibreEfficientDet"
    INPUT_SIZES = INPUT_SIZES
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = None
    TTA_FIXED_SIZE = True
    val_preprocessor_class = EfficientDetValPreprocessor

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Require the learned BiFPN fusion plus both separable prediction heads."""
        required = (_BIFPN_KEY, _CLASS_PREDICT_KEY, _BOX_PREDICT_KEY, _BACKBONE_KEY)
        if not all(key in weights_dict for key in required):
            return False
        try:
            class_weight = weights_dict[_CLASS_PREDICT_KEY]
            box_weight = weights_dict[_BOX_PREDICT_KEY]
            return (
                tuple(weights_dict[_BIFPN_KEY].shape) == (2,)
                and int(class_weight.shape[0]) % _ANCHORS_PER_LOCATION == 0
                and int(box_weight.shape[0]) == 4 * _ANCHORS_PER_LOCATION
                and tuple(weights_dict[_BACKBONE_KEY].shape[1:]) == (3, 3, 3)
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        weight = weights_dict.get(_CLASS_PREDICT_KEY)
        if weight is None or getattr(weight, "ndim", 0) != 4:
            return None
        fpn_channels = int(weight.shape[1])
        matches = [
            size
            for size, cfg in SCALE_CONFIGS.items()
            if cfg.fpn_channels == fpn_channels
        ]
        return matches[0] if len(matches) == 1 else None

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        detected = super().detect_size_from_filename(filename)
        if detected is not None:
            return detected
        match = re.search(
            r"(?:tf_)?efficientdet[_-](d[0-4])(?:_|-|\.|$)", Path(filename).name.lower()
        )
        return match.group(1) if match else None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        weight = weights_dict.get(_CLASS_PREDICT_KEY)
        if weight is None or getattr(weight, "ndim", 0) != 4:
            return None
        architectural = int(weight.shape[0]) // _ANCHORS_PER_LOCATION
        if architectural <= 0:
            return None
        return 80 if architectural == _COCO_SPARSE_CLASSES else architectural

    @classmethod
    def convert_upstream_state_dict(cls, weights_dict: dict) -> Optional[dict]:
        """Require the explicit converter because TF/static padding is not serialized."""
        del weights_dict
        return None

    def __init__(
        self,
        model_path=None,
        size: str = "d0",
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ) -> None:
        self._arch_num_classes = (
            _COCO_SPARSE_CLASSES if nb_classes == 80 else nb_classes
        )
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        return LibreEfficientDetModel(
            size=self.size, num_classes=self._arch_num_classes
        )

    def _rebuild_for_checkpoint_classes(
        self, new_nb_classes: int, state_dict: dict
    ) -> None:
        """Rebuild the prediction head to the checkpoint's serialized width."""
        class_weight = state_dict.get(_CLASS_PREDICT_KEY)
        if class_weight is None or getattr(class_weight, "ndim", 0) != 4:
            raise RuntimeError(
                f"EfficientDet checkpoint is missing {_CLASS_PREDICT_KEY!r}"
            )
        channels = int(class_weight.shape[0])
        if channels % _ANCHORS_PER_LOCATION:
            raise RuntimeError(
                "EfficientDet checkpoint class head is not divisible by nine anchors"
            )
        self._arch_num_classes = channels // _ANCHORS_PER_LOCATION
        super()._rebuild_for_checkpoint_classes(new_nb_classes, state_dict)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "neck": self.model.fpn,
            "class_head": self.model.class_net,
            "box_head": self.model.box_net,
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
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        effective = self.input_size if input_size is None else int(input_size)
        if effective != self.input_size:
            raise ValueError(
                f"EfficientDet {self.size} has a fixed {self.input_size}x{self.input_size} graph; "
                f"got imgsz={effective}."
            )
        return preprocess_image(image, input_size=effective, color_format=color_format)

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 100,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        actual_input_size = kwargs.pop("input_size", self.input_size)
        if kwargs.pop("letterbox", False):
            original_width, original_height = original_size
            ratio = min(
                float(actual_input_size) / original_height,
                float(actual_input_size) / original_width,
            )
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            max_det=max_det,
            ratio=ratio,
            sparse_coco=(
                self.nb_classes == 80
                and self._arch_num_classes == _COCO_SPARSE_CLASSES
            ),
            **kwargs,
        )

    def train(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "EfficientDet is shipped inference-only. Its focal-loss and anchor-matching "
            "training recipe is not implemented in LibreYOLO yet."
        )

    def _strict_loading(self) -> bool:
        return True


__all__ = ["LibreEfficientDet"]
