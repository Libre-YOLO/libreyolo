"""Shared ``BaseModel`` implementation for the Darknet families.

``DarknetFamily`` holds all logic common to YOLOv2/v3/v4; the concrete
``LibreYOLO2`` / ``LibreYOLO3`` / ``LibreYOLO4`` classes are thin subclasses that
declare their sizes, cfg mapping, and anchor count. The runtime graph is the
cfg-driven :class:`~libreyolo.models.darknet.net.DarknetNet`, wrapped in a
family-named holder so state-dict keys carry a stable per-family prefix
(``yolo3.layers.0.conv.weight``) for checkpoint discrimination.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...postprocess.darknet_yolo import postprocess as darknet_postprocess
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import DarknetValPreprocessor
from ..base import BaseModel
from . import build_darknet
from .net import DarknetNet
from .preprocess import preprocess_image as _dk_preprocess_image
from .preprocess import preprocess_numpy as _dk_preprocess_numpy


class _DarknetHolder(nn.Module):
    """Wrap a :class:`DarknetNet` under a family-named submodule attribute.

    Keys become ``<family>.layers.<i>...`` so each family's checkpoints are
    self-identifying, independent of the (shared) engine.
    """

    def __init__(self, family_attr: str, net: DarknetNet):
        super().__init__()
        self.add_module(family_attr, net)
        self._family_attr = family_attr

    @property
    def net(self) -> DarknetNet:
        return getattr(self, self._family_attr)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class DarknetFamily(BaseModel):
    """Base class for the cfg-driven Darknet detector families."""

    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    val_preprocessor_class = DarknetValPreprocessor

    # --- subclass contract -------------------------------------------------
    CFG_BY_SIZE: Dict[str, str] = {}          # size code -> bundled cfg name
    CONV_COUNT_TO_SIZE: Dict[int, str] = {}   # conv-layer count -> size code
    ANCHORS_PER_HEAD: int = 3                  # 3 for v3/v4, 5 for v2 region
    DEFAULT_SIZE: str = "b"

    def __init__(
        self,
        model_path=None,
        size: str | None = None,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        if size is None:
            size = self.DEFAULT_SIZE
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )
        if isinstance(model_path, str):
            self._load_weights(model_path)

    # =====================================================================
    # Registry classmethods
    # =====================================================================
    @classmethod
    def _own_layer_indices_with(cls, weights_dict: dict, suffix: str) -> Dict[str, Any]:
        prefix = f"{cls.FAMILY}.layers."
        found: Dict[str, Any] = {}
        for key, value in weights_dict.items():
            if key.startswith(prefix) and key.endswith(suffix):
                idx = key[len(prefix):].split(".", 1)[0]
                found[idx] = value
        return found

    @classmethod
    def _conv_count(cls, weights_dict: dict) -> int:
        return len(cls._own_layer_indices_with(weights_dict, ".conv.weight"))

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        prefix = f"{cls.FAMILY}."
        if not any(k.startswith(prefix) for k in weights_dict):
            return False
        return cls.detect_size(weights_dict) is not None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return cls.CONV_COUNT_TO_SIZE.get(cls._conv_count(weights_dict))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # Detection-head convs are the only ones with a conv bias and no BN
        # (Darknet ``activation=linear`` output layers). Their out-channels is
        # ``anchors_per_head * (5 + nc)``.
        conv_bias = cls._own_layer_indices_with(weights_dict, ".conv.bias")
        bn_layers = set(cls._own_layer_indices_with(weights_dict, ".bn.weight"))
        head_ocs = [v.shape[0] for idx, v in conv_bias.items() if idx not in bn_layers]
        if not head_ocs:
            return None
        nc = head_ocs[0] // cls.ANCHORS_PER_HEAD - 5
        return nc if nc > 0 else None

    # =====================================================================
    # Model lifecycle
    # =====================================================================
    def _init_model(self) -> nn.Module:
        cfg_name = self.CFG_BY_SIZE[self.size]
        net = build_darknet(cfg_name, num_classes=self.nb_classes)
        return _DarknetHolder(self.FAMILY, net)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"net": self.model.net}

    def _strict_loading(self) -> bool:
        return True

    # =====================================================================
    # Inference pipeline
    # =====================================================================
    @staticmethod
    def _get_preprocess_numpy():
        return _dk_preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        size = input_size if input_size is not None else self.input_size
        return _dk_preprocess_image(image, input_size=size, color_format=color_format)

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        specs = self.model.net.detection_specs()
        input_size = kwargs.get("input_size", self.input_size)
        return darknet_postprocess(
            output,
            specs,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=input_size,
            original_size=original_size,
            ratio=ratio,
            max_det=max_det,
        )

    # =====================================================================
    # Training is out of scope for the Darknet families in LibreYOLO v1.
    # =====================================================================
    def train(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} is inference-only in this LibreYOLO release. "
            "Training/fine-tuning for the Darknet-lineage families "
            "(YOLOv2/v3/v4) is not yet implemented; use a modern trainable "
            "family (YOLO9, RF-DETR) for custom training."
        )
