"""LibreHRNet top-down human-pose wrapper (inference only).

HRNet keeps a high-resolution stream throughout repeated multi-scale fusion.
The pose family consumes person crops and emits one COCO-17 heatmap per crop.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import numpy as np
import torch
from torch import nn

from ..base import BaseModel
from .nn import HRNetPoseModel
from .utils import preprocess_crop_image, preprocess_numpy


class LibreHRNet(BaseModel):
    """Top-down HRNet pose estimator: person crops to COCO-17 heatmaps."""

    FAMILY = "hrnet"
    FILENAME_PREFIX = "LibreHRNet"
    INPUT_SIZES: ClassVar[dict[str, tuple[int, int]]] = {
        "w32": (256, 192),
        "w48": (384, 288),
    }
    SUPPORTED_TASKS = ("pose",)
    DEFAULT_TASK = "pose"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    TTA_ENABLED = False
    POSE_NUM_KEYPOINTS = 17

    _STAGE_KEY = "stage3.0.branches.0.0.conv1.weight"
    _SIGNATURE_KEYS = (
        "transition1.0.0.weight",
        _STAGE_KEY,
        "final_layer.weight",
    )

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        if not all(key in weights_dict for key in cls._SIGNATURE_KEYS):
            return False
        stem = weights_dict.get("conv1.weight")
        stage = weights_dict[cls._STAGE_KEY]
        head = weights_dict["final_layer.weight"]
        return bool(
            getattr(stem, "shape", None) == torch.Size((64, 3, 3, 3))
            and getattr(stage, "ndim", 0) == 4
            and int(stage.shape[0]) in (32, 48)
            and getattr(head, "shape", None)
            in (torch.Size((17, 32, 1, 1)), torch.Size((17, 48, 1, 1)))
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        stage = weights_dict.get(cls._STAGE_KEY)
        if stage is None or getattr(stage, "ndim", 0) != 4:
            return None
        return {32: "w32", 48: "w48"}.get(int(stage.shape[0]))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_num_keypoints(cls, weights_dict: dict) -> Optional[int]:
        head = weights_dict.get("final_layer.weight")
        return int(head.shape[0]) if head is not None and getattr(head, "ndim", 0) == 4 else None

    def __init__(
        self,
        model_path=None,
        size: str = "w32",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        self.num_keypoints = self.POSE_NUM_KEYPOINTS
        self.keypoint_dim = 3
        if model_path is not None and isinstance(model_path, str):
            self._load_weights(model_path)
        # HRNet's released pose head is fixed to the COCO person category.
        # Keep this semantic name even when a metadata-less upstream file was
        # auto-wrapped with the generic one-class fallback.
        self.names = {0: "person"}
        self.model.eval()

    def _init_model(self) -> nn.Module:
        width = 32 if self.size == "w32" else 48
        return HRNetPoseModel(width=width, num_keypoints=self.POSE_NUM_KEYPOINTS)

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {
            "stem": self.model.conv1,
            "backbone": self.model.stage3,
            "head": self.model.final_layer,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(self, image, color_format="auto", input_size=None):
        return preprocess_crop_image(
            image,
            input_size=input_size or self._get_input_size(),
            color_format=color_format,
        )

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output,
        conf_thres,
        iou_thres,
        original_size,
        max_det=1,
        ratio=1.0,
        **kwargs,
    ) -> dict:
        del output, conf_thres, iou_thres, original_size, max_det, ratio, kwargs
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "classes": np.zeros((0,), dtype=np.int64),
            "keypoints": np.zeros((0, self.POSE_NUM_KEYPOINTS, 3), dtype=np.float32),
        }

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreHRNet is inference-only. Pose training requires a keypoint-aware "
            "data path and augmentations that LibreYOLO does not yet provide."
        )
