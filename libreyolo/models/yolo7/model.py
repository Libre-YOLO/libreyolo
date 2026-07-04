"""LibreYOLO7 — YOLOv7 detector, native port of MIT MultimediaTechLab/YOLO.

Source provenance: architecture and weights derive from MultimediaTechLab/YOLO
(MIT, (c) 2024 Kin-Yiu Wong & Hao-Tang Tsui) — the authors' own MIT re-release
of YOLOv7. NOT the GPL-3.0 ``WongKinYiu/yolov7``. The native modules mirror the
upstream names so ``v7.pt`` loads with no remapping (see ``blocks.py`` / ``net.py``).

Inference-only in this release. Single size ``b`` (upstream ships one v7 model).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...utils.image_loader import ImageInput
from ...validation.preprocessors import YOLO9ValPreprocessor
from ..base import BaseModel
from .net import YOLOv7Model


class LibreYOLO7(BaseModel):
    """YOLOv7 object detector (anchor-based, implicit-knowledge head)."""

    FAMILY = "yolo7"
    FILENAME_PREFIX = "LibreYOLO7"
    INPUT_SIZES = {"b": 640}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    # Letterbox + RGB + /255 + gray(114) pad — same contract as YOLO9.
    val_preprocessor_class = YOLO9ValPreprocessor

    # =====================================================================
    # Registry classmethods
    # =====================================================================
    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # ImplicitA/M (YOLOR implicit knowledge) heads are unique to YOLOv7.
        return any("implicit_a.implicit" in k for k in weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return "b" if cls.can_load(weights_dict) else None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        for k, v in weights_dict.items():
            if k.endswith("heads.0.head_conv.weight"):
                return int(v.shape[0]) // 3 - 5
        return None

    # =====================================================================
    # Init
    # =====================================================================
    def __init__(self, model_path=None, size: str = "b", nb_classes: int = 80,
                 device: str = "auto", **kwargs):
        super().__init__(model_path=model_path, size=size, nb_classes=nb_classes,
                         device=device, **kwargs)
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        return YOLOv7Model(num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"model": self.model}

    def _strict_loading(self) -> bool:
        return True  # native names match v7.pt exactly (verified 564/564)

    # =====================================================================
    # Inference pipeline
    # =====================================================================
    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy
        return preprocess_numpy

    def _preprocess(self, image: ImageInput, color_format: str = "auto",
                    input_size: Optional[int] = None):
        from .utils import preprocess_image
        size = input_size if input_size is not None else self.input_size
        return preprocess_image(image, input_size=size, color_format=color_format)

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(self, output: Any, conf_thres: float, iou_thres: float,
                     original_size: Tuple[int, int], max_det: int = 300,
                     ratio: float = 1.0, **kwargs) -> Dict:
        from ...postprocess.yolo7 import postprocess as _pp
        input_size = kwargs.get("input_size", self.input_size)
        return _pp(
            output,
            self.model.anchors,
            self.model.strides,
            self.nb_classes,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=input_size,
            original_size=original_size,
            ratio=ratio,
            max_det=max_det,
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreYOLO7 is inference-only in this LibreYOLO release. Training for "
            "YOLOv7 is not yet implemented; use YOLO9 or RF-DETR for custom training."
        )
