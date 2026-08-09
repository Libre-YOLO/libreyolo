"""LibreSwinIR super-resolution model family.

SwinIR uses residual Swin Transformer blocks for image restoration. This v1
family includes 4x super-resolution inference and validation for the official
lightweight, real-world medium, and real-world large checkpoints. Training is
out of scope because the real-world variants require degradation and GAN
training pipelines that LibreYOLO does not currently provide.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.swinir import postprocess as _swinir_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .nn import SwinIR
from .utils import forward_with_optional_tiling, preprocess_image, preprocess_numpy


SWINIR_SIZE_CONFIGS: dict[str, dict[str, Any]] = {
    "s": {
        "scale": 4,
        "embed_dim": 60,
        "depths": (6, 6, 6, 6),
        "num_heads": (6, 6, 6, 6),
        "upsampler": "pixelshuffledirect",
        "resi_connection": "1conv",
        "pad_multiple": 8,
    },
    "m": {
        "scale": 4,
        "embed_dim": 180,
        "depths": (6, 6, 6, 6, 6, 6),
        "num_heads": (6, 6, 6, 6, 6, 6),
        "upsampler": "nearest+conv",
        "resi_connection": "1conv",
        "pad_multiple": 8,
    },
    "l": {
        "scale": 4,
        "embed_dim": 240,
        "depths": (6, 6, 6, 6, 6, 6, 6, 6, 6),
        "num_heads": (8, 8, 8, 8, 8, 8, 8, 8, 8),
        "upsampler": "nearest+conv",
        "resi_connection": "3conv",
        "pad_multiple": 8,
    },
}

_LAYER_KEY = re.compile(r"^layers\.(\d+)\.")


def _swinir_fingerprint(state_dict: dict) -> tuple[int, int] | None:
    conv_first = state_dict.get("conv_first.weight")
    if conv_first is None or getattr(conv_first, "ndim", 0) != 4:
        return None
    if tuple(conv_first.shape[1:]) != (3, 3, 3):
        return None
    if "patch_embed.norm.weight" not in state_dict or "norm.weight" not in state_dict:
        return None
    if not any(key.endswith("attn.relative_position_bias_table") for key in state_dict):
        return None
    layer_indexes = {
        int(match.group(1))
        for key in state_dict
        if (match := _LAYER_KEY.match(key)) is not None
    }
    if not layer_indexes or layer_indexes != set(range(max(layer_indexes) + 1)):
        return None
    return int(conv_first.shape[0]), len(layer_indexes)


class LibreSwinIR(BaseModel):
    """SwinIR RGB 4x super-resolution with optional halo-padded tiling."""

    FAMILY = "swinir"
    FILENAME_PREFIX = "LibreSwinIR"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": 64, "m": 64, "l": 64}
    SUPPORTED_TASKS = ("restore",)
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    DEFAULT_TASK = "restore"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return cls.detect_size(weights_dict) is not None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        fingerprint = _swinir_fingerprint(weights_dict)
        return {(60, 4): "s", (180, 6): "m", (240, 9): "l"}.get(fingerprint)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "restore" if cls.can_load(state_dict) else None

    def __init__(
        self,
        model_path=None,
        size: str = "m",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        tile: int = 0,
        tile_pad: int = 16,
        **kwargs,
    ) -> None:
        del nb_classes
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        self._default_tile = int(tile)
        self._default_tile_pad = int(tile_pad)
        self._tile_settings: ContextVar[tuple[int, int] | None] = ContextVar(
            f"libreyolo_swinir_tile_settings_{id(self)}", default=None
        )
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.nb_classes = 1
        self.names = {0: "image"}

    @property
    def restore_scale(self) -> int:
        return int(SWINIR_SIZE_CONFIGS[self.size]["scale"])

    @property
    def _pad_multiple(self) -> int:
        return int(SWINIR_SIZE_CONFIGS[self.size]["pad_multiple"])

    def _init_model(self) -> nn.Module:
        config = SWINIR_SIZE_CONFIGS[self.size]
        return SwinIR(
            upscale=int(config["scale"]),
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=config["depths"],
            embed_dim=int(config["embed_dim"]),
            num_heads=config["num_heads"],
            mlp_ratio=2.0,
            upsampler=str(config["upsampler"]),
            resi_connection=str(config["resi_connection"]),
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {name: module for name, module in self.model.named_children()}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def __call__(self, source=None, **kwargs):
        tile = kwargs.pop("tile", None)
        tile_pad = kwargs.pop("tile_pad", None)
        inherited = self._tile_settings.get()
        default_tile, default_tile_pad = inherited or (
            self._default_tile,
            self._default_tile_pad,
        )
        settings = (
            default_tile if tile is None else int(tile),
            default_tile_pad if tile_pad is None else int(tile_pad),
        )
        token = self._tile_settings.set(settings)
        try:
            return super().__call__(source, **kwargs)
        finally:
            self._tile_settings.reset(token)

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        del input_size
        return preprocess_image(
            image,
            pad_multiple=self._pad_multiple,
            color_format=color_format,
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        settings = self._tile_settings.get()
        tile_size, tile_pad = settings or (
            self._default_tile,
            self._default_tile_pad,
        )
        return forward_with_optional_tiling(
            self.model,
            input_tensor,
            scale=self.restore_scale,
            tile_size=tile_size,
            tile_pad=tile_pad,
        )

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
        del conf_thres, iou_thres, max_det, ratio, kwargs
        return {
            "restored": _swinir_postprocess(
                output, original_size, scale=self.restore_scale
            )
        }

    def train(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "LibreSwinIR ships inference and validation only. Real-world SwinIR "
            "training requires degradation and GAN pipelines that are outside "
            "the v1 restore trainer scope."
        )


__all__ = ["LibreSwinIR", "SWINIR_SIZE_CONFIGS"]
