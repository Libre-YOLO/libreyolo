"""LibreYOLO wrapper for Depth Anything 3 monocular depth estimation.

The vendored architecture comes from ByteDance-Seed/Depth-Anything-3 at commit
``41736238f5bced4debf3f2a12375d2466874866d`` under Apache-2.0. This family
implements only DA3MONO-LARGE, whose Apache-2.0 checkpoint directly predicts
positive relative depth. :class:`LibreDepthAnything3Net` converts that output
to LibreYOLO's relative inverse-depth contract (higher means closer, no metric
unit) inside ``forward``.

The Apache-2.0 DA3 metric and any-view Small/Base checkpoints are not exposed:
metric depth does not match the task contract, while any-view models require a
future multi-image/camera API. The non-commercial Large/Giant/Nested weights
are never referenced by download code. Training is out of scope.
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
from .nn import LibreDepthAnything3Net
from .utils import preprocess_numpy


class LibreDepthAnything3(BaseModel):
    """DA3MONO-LARGE: image to dense relative inverse-depth map."""

    FAMILY = "depth_anything3"
    FILENAME_PREFIX = "LibreDepthAnything3"
    WEIGHT_EXT = ".pt"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"l": 504}
    SUPPORTED_TASKS = ("depth",)
    DEFAULT_TASK = "depth"
    REQUIRE_TASK_SUFFIX = True

    depth_imgsz_divisor = 14
    depth_resize_mode = "letterbox"
    SUPPORTS_BATCHED_PREDICT = False

    _UPSTREAM_URL = "https://github.com/ByteDance-Seed/Depth-Anything-3"

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return any(
            key.startswith("backbone.pretrained.") for key in weights_dict
        ) and any(key.startswith("head.scratch.") for key in weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        cls_token = weights_dict.get("backbone.pretrained.cls_token")
        if (
            cls_token is not None
            and getattr(cls_token, "ndim", 0) >= 1
            and int(cls_token.shape[-1]) == 1024
        ):
            return "l"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1

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
        return LibreDepthAnything3Net()

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"backbone": self.model.backbone, "head": self.model.head}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_res = input_size if input_size is not None else self._get_input_size()
        if effective_res % self.depth_imgsz_divisor:
            raise ValueError(
                f"Depth Anything 3 imgsz={effective_res} must be divisible by "
                f"{self.depth_imgsz_divisor} (DINOv2 patch grid)."
            )
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), effective_res)
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
        inverse_depth = output
        if isinstance(inverse_depth, dict):
            inverse_depth = inverse_depth.get("depth", inverse_depth.get("predictions"))
        orig_w, orig_h = original_size
        inverse_depth = F.interpolate(
            inverse_depth.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=True,
        )
        return {"depth": inverse_depth[0, 0].cpu()}

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training Depth Anything 3 is out of scope for LibreYOLO. Use the "
            f"upstream project at {self._UPSTREAM_URL} and convert a compatible "
            "DA3MONO-LARGE checkpoint with "
            "weights/convert_depth_anything3_weights.py."
        )

    def export(
        self,
        format: str = "onnx",
        *,
        opset: int = 17,
        **kwargs,
    ) -> str:
        if format.lower() not in {
            "onnx",
            "torchscript",
            "executorch",
            "tensorrt",
            "openvino",
        }:
            raise NotImplementedError(
                f"Export to {format!r} is not implemented for Depth Anything 3; "
                "supported formats are ONNX, TorchScript, ExecuTorch, "
                "TensorRT, and OpenVINO."
            )
        return super().export(format=format, opset=opset, **kwargs)
