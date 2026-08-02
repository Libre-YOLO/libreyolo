"""LibreMiDaS: MiDaS relative monocular depth in the LibreYOLO factory.

MiDaS introduced zero-shot monocular relative depth through scale-and-shift
invariant training across mixed datasets. Its later DPT decoder established the
transformer dense-prediction pattern inherited by Depth Anything and related
families.

Predictions are relative inverse depth: larger values mean closer, and values
have no metric unit or cross-image scale. This first integration is
inference-only and ships two complementary official variants:

* ``s``: MiDaS v2.1 Small, EfficientNet-Lite3, 256-pixel upper-bound resize,
  ImageNet mean/std normalization.
* ``l``: DPT-Large, ViT-L/16, 384-pixel minimal resize, mean/std 0.5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from .nn import build_midas_model
from .utils import IMGSZ_DIVISOR, preprocess_numpy


class LibreMiDaS(BaseModel):
    """MiDaS image-to-relative-inverse-depth family."""

    FAMILY = "midas"
    FILENAME_PREFIX = "LibreMiDaS"
    WEIGHT_EXT = ".pt"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": 256, "l": 384}
    SUPPORTED_TASKS = ("depth",)
    DEFAULT_TASK = "depth"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None

    depth_imgsz_divisor = IMGSZ_DIVISOR
    depth_resize_mode = "letterbox"
    SUPPORTS_BATCHED_PREDICT = False

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        has_decoder = (
            "scratch.refinenet1.resConfUnit1.conv1.weight" in weights_dict
            and "scratch.output_conv.4.weight" in weights_dict
        )
        has_dpt = "pretrained.model.cls_token" in weights_dict
        has_small = "pretrained.layer1.3.0.conv_dw.weight" in weights_dict
        return has_decoder and (has_dpt or has_small)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        cls_token = weights_dict.get("pretrained.model.cls_token")
        if cls_token is not None and tuple(cls_token.shape) == (1, 1, 1024):
            return "l"
        stem = weights_dict.get("pretrained.layer1.0.weight")
        if stem is not None and tuple(stem.shape[:2]) == (32, 3):
            return "s"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    def __init__(
        self,
        model_path,
        size: str = "l",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        self.model.eval()
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.nb_classes = 1
        self.names = {0: "depth"}

    def _init_model(self) -> nn.Module:
        return build_midas_model(self.size)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "pretrained": self.model.pretrained,
            "scratch": self.model.scratch,
        }

    @staticmethod
    def _get_preprocess_numpy():
        # The family needs the size code in addition to the common input size;
        # validation uses the dense-task path and the network owns normalization.
        return None

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_res = input_size if input_size is not None else self._get_input_size()
        if effective_res % self.depth_imgsz_divisor:
            raise ValueError(
                f"MiDaS imgsz={effective_res} must be divisible by "
                f"{self.depth_imgsz_divisor}."
            )
        img = ImageLoader.load(image, color_format=color_format).convert("RGB")
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), effective_res, self.size)
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

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
    ) -> Dict:
        del conf_thres, iou_thres, max_det, kwargs
        depth = output
        if isinstance(depth, dict):
            depth = depth.get("depth", depth.get("predictions"))
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        orig_w, orig_h = original_size
        depth = F.interpolate(
            depth.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=True,
        )
        return {"depth": depth[0, 0].cpu()}

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "MiDaS training is not implemented in LibreYOLO. This museum-tier "
            "port supports pretrained inference, zero-shot validation, and export."
        )
