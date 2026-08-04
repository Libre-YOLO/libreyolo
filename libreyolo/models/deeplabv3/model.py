"""LibreYOLO wrapper for DeepLabv3 semantic segmentation.

DeepLabv3 introduced the image-level pooling branch and improved atrous
spatial pyramid pooling that became a standard dense-prediction baseline. This
family exposes the three torchvision COCO-with-VOC-label checkpoints as
inference-only semantic models: dilated ResNet-50, dilated ResNet-101, and
dilated MobileNetV3-Large.

The implementation is derived from the BSD-3-Clause torchvision release pinned
in this family's NOTICE file. It is DeepLabv3, not DeepLabv3+; there is no
decoder or CRF.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ...postprocess.deeplabv3 import (
    postprocess as _deeplabv3_postprocess,
)
from ...postprocess.deeplabv3 import semantic_logits
from ...tasks import normalize_task
from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from .convert import convert_upstream_deeplabv3_state_dict
from .nn import SIZE_CONFIGS, LibreDeepLabv3Net
from .utils import preprocess_numpy


VOC_NAMES: dict[int, str] = {
    0: "__background__",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
}


class LibreDeepLabv3(BaseModel):
    """DeepLabv3 r50/r101/mv3 family for 21-class semantic segmentation."""

    FAMILY: ClassVar[str] = "deeplabv3"
    FILENAME_PREFIX: ClassVar[str] = "LibreDeepLabv3"
    WEIGHT_EXT: ClassVar[str] = ".pt"
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("semantic",)
    DEFAULT_TASK: ClassVar[str] = "semantic"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    INPUT_SIZES: ClassVar[Dict[str, int]] = {size: 520 for size in SIZE_CONFIGS}
    TRAIN_CONFIG: ClassVar[None] = None

    # The released torchvision preset resizes the short side to 520. LibreYOLO
    # maps that deployment resolution onto its fixed square backend contract.
    semantic_resize_mode: ClassVar[str] = "stretch"
    # The native heads upsample logits to the input canvas and all three
    # backbones accept arbitrary positive spatial dimensions. Their feature
    # strides differ (ResNet=8, MobileNetV3=16), so there is no shared divisor.
    semantic_imgsz_divisor: ClassVar[int] = 1
    TTA_FIXED_SIZE: ClassVar[bool] = True

    _UPSTREAM_URL: ClassVar[str] = "https://github.com/pytorch/vision"

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys = set(weights_dict)
        has_aspp = {
            "classifier.0.convs.0.0.weight",
            "classifier.0.convs.1.0.weight",
            "classifier.0.convs.4.1.weight",
            "classifier.0.project.0.weight",
            "classifier.4.weight",
        }.issubset(keys)
        has_backbone = "backbone.conv1.weight" in keys or "backbone.0.0.weight" in keys
        return has_aspp and has_backbone

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        aspp = weights_dict.get("classifier.0.convs.0.0.weight")
        if aspp is None or getattr(aspp, "ndim", 0) != 4:
            return None
        in_channels = int(aspp.shape[1])
        if in_channels == 960:
            return "mv3"
        if in_channels != 2048:
            return None

        prefix = "backbone.layer3."
        indices = {
            int(key[len(prefix) :].split(".", 1)[0])
            for key in weights_dict
            if key.startswith(prefix) and key[len(prefix) :].split(".", 1)[0].isdigit()
        }
        depth = len(indices)
        return {6: "r50", 23: "r101"}.get(depth)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        head = weights_dict.get("classifier.4.weight")
        if head is not None and getattr(head, "ndim", 0) >= 1:
            return int(head.shape[0])
        return None

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        return convert_upstream_deeplabv3_state_dict(state_dict)

    def __init__(
        self,
        model_path=None,
        size: str = "r50",
        nb_classes: int = 21,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else "semantic"
        if resolved_task != "semantic":
            raise ValueError(
                f"LibreDeepLabv3 supports only task='semantic'; got {task!r}."
            )
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )
        if self.nb_classes == 21:
            self.names = dict(VOC_NAMES)
        self.model.eval()
        if self.model_path is not None:
            self._load_weights(str(self.model_path))

    def _init_model(self) -> nn.Module:
        return LibreDeepLabv3Net(size=self.size, num_classes=self.nb_classes)

    def _prepare_model_for_state_dict(self, state_dict: dict) -> None:
        detected = self.detect_size(state_dict)
        if detected is None or detected == self.size:
            return
        self.size = detected
        self.input_size = self.INPUT_SIZES[detected]
        self.model = self._init_model().to(self.device)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"backbone": self.model.backbone, "head": self.model.classifier}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: int | tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, Image.Image, tuple[int, int], float]:
        effective_size = (
            input_size if input_size is not None else self._get_input_size()
        )
        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        chw, ratio = preprocess_numpy(np.asarray(img.convert("RGB")), effective_size)
        return torch.from_numpy(chw).unsqueeze(0), img, original_size, ratio

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess_semantic_logits(
        self,
        output: Any,
        original_size: tuple[int, int],
        **kwargs,
    ) -> torch.Tensor:
        return semantic_logits(output, original_size)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        return _deeplabv3_postprocess(output, original_size)

    def _validate_loaded_state_dict_for_task(
        self,
        state_dict: dict,
        checkpoint: dict | None = None,
    ) -> None:
        if not self.can_load(state_dict):
            raise RuntimeError(
                "Checkpoint does not look like a DeepLabv3 ASPP semantic model."
            )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training DeepLabv3 is out of scope for the inference-only museum "
            f"port. Train with the pinned torchvision recipe at {self._UPSTREAM_URL} "
            "and convert the checkpoint with weights/convert_deeplabv3_weights.py."
        )


__all__ = ["LibreDeepLabv3", "VOC_NAMES"]
