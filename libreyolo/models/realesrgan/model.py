"""LibreRealESRGAN super-resolution model family.

Real-ESRGAN is a practical blind super-resolution upscaler. This family ports
the two upstream generator architectures and extends the ``restore`` task with
scale-aware output (``Results.restore_scale``):

* ``x4``  - RealESRGAN_x4plus  (RRDBNet, 23 blocks, quality default, 4x)
* ``x2``  - RealESRGAN_x2plus  (RRDBNet, 23 blocks, 2x)
* ``x4t`` - realesr-general-x4v3 (SRVGGNetCompact, fast/video tier, 4x)

Inference and validation only. Real-ESRGAN training is a GAN over a synthetic
degradation pipeline and is out of scope for v1 (see the family docs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.realesrgan import postprocess as _realesrgan_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .nn import RRDBNet, SRVGGNetCompact
from .utils import forward_with_optional_tiling, preprocess_image, preprocess_numpy


# Per-size architecture. ``pad_multiple`` is the input divisibility required by
# the network (RRDBNet x2 pixel-unshuffles by 2; the 4x models are unconstrained).
REALESRGAN_SIZE_CONFIGS: dict[str, dict[str, Any]] = {
    "x4": {
        "arch": "rrdbnet",
        "scale": 4,
        "num_feat": 64,
        "num_block": 23,
        "num_grow_ch": 32,
        "pad_multiple": 1,
    },
    "x2": {
        "arch": "rrdbnet",
        "scale": 2,
        "num_feat": 64,
        "num_block": 23,
        "num_grow_ch": 32,
        "pad_multiple": 2,
    },
    "x4t": {
        "arch": "srvgg",
        "scale": 4,
        "num_feat": 64,
        "num_conv": 32,
        "act_type": "prelu",
        "pad_multiple": 1,
    },
}


def _is_rrdbnet_state_dict(sd: dict) -> bool:
    required = ("conv_first.weight", "conv_body.weight", "conv_up1.weight", "conv_last.weight")
    return all(k in sd for k in required) and "body.0.rdb1.conv1.weight" in sd


def _is_srvgg_state_dict(sd: dict) -> bool:
    if "conv_first.weight" in sd or "body.0.rdb1.conv1.weight" in sd:
        return False
    first = sd.get("body.0.weight")
    prelu = sd.get("body.1.weight")
    if first is None or getattr(first, "ndim", 0) != 4:
        return False
    if prelu is None or getattr(prelu, "ndim", 0) != 1:
        return False
    # the pre-PixelShuffle conv maps num_feat -> num_out_ch * upscale**2 (3*16)
    return any(
        k.startswith("body.")
        and getattr(v, "ndim", 0) == 4
        and int(v.shape[0]) == 48
        for k, v in sd.items()
    )


class LibreRealESRGAN(BaseModel):
    """Real-ESRGAN RGB super-resolution.

    Predict runs at native image resolution, reflect-pads to the network's
    divisibility factor, and returns ``Results.restored`` on a canvas ``scale``
    times the input (``Results.restore_scale`` records the factor). Large inputs
    can be processed seam-free with ``model.predict(img, tile=512)``.
    """

    FAMILY = "realesrgan"
    FILENAME_PREFIX = "LibreRealESRGAN"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"x4": 64, "x2": 64, "x4t": 64}
    SUPPORTED_TASKS = ("restore",)
    DEFAULT_TASK = "restore"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return (
            _is_rrdbnet_state_dict(weights_dict) or _is_srvgg_state_dict(weights_dict)
        ) and cls.detect_size(weights_dict) is not None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        if _is_srvgg_state_dict(weights_dict):
            return "x4t"
        if _is_rrdbnet_state_dict(weights_dict):
            conv_first = weights_dict.get("conv_first.weight")
            if conv_first is None or getattr(conv_first, "ndim", 0) != 4:
                return None
            # in_channels encodes the pixel-unshuffle factor: 3 -> x4, 12 -> x2.
            return {3: "x4", 12: "x2"}.get(int(conv_first.shape[1]))
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "restore" if cls.can_load(state_dict) else None

    def __init__(
        self,
        model_path=None,
        size: str = "x4",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        tile: int = 0,
        tile_pad: int = 10,
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
        # Tiling state (see ``__call__`` / ``_forward``). Defaults from the
        # constructor let ``LibreYOLO("...pt", tile=512)`` set a persistent tile.
        self._tile = int(tile)
        self._tile_pad = int(tile_pad)
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.nb_classes = 1
        self.names = {0: "image"}

    @property
    def restore_scale(self) -> int:
        """Integer output-to-input upscale factor for this model."""

        return int(REALESRGAN_SIZE_CONFIGS[self.size]["scale"])

    @property
    def _pad_multiple(self) -> int:
        return int(REALESRGAN_SIZE_CONFIGS[self.size]["pad_multiple"])

    def _init_model(self) -> nn.Module:
        cfg = REALESRGAN_SIZE_CONFIGS[self.size]
        if cfg["arch"] == "rrdbnet":
            return RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                scale=int(cfg["scale"]),
                num_feat=int(cfg["num_feat"]),
                num_block=int(cfg["num_block"]),
                num_grow_ch=int(cfg["num_grow_ch"]),
            )
        return SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=int(cfg["num_feat"]),
            num_conv=int(cfg["num_conv"]),
            upscale=int(cfg["scale"]),
            act_type=str(cfg["act_type"]),
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {name: module for name, module in self.model.named_children()}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def __call__(self, source=None, **kwargs):
        """Capture ``tile`` / ``tile_pad`` before dispatching to the runner.

        These control the seam-free tiled forward but must not leak into the
        shared postprocess kwargs, so they are stashed on the instance and read
        back in ``_forward``.
        """

        tile = kwargs.pop("tile", None)
        tile_pad = kwargs.pop("tile_pad", None)
        prev = (self._tile, self._tile_pad)
        if tile is not None:
            self._tile = int(tile)
        if tile_pad is not None:
            self._tile_pad = int(tile_pad)
        try:
            return super().__call__(source, **kwargs)
        finally:
            self._tile, self._tile_pad = prev

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
        return forward_with_optional_tiling(
            self.model,
            input_tensor,
            scale=self.restore_scale,
            tile_size=self._tile,
            tile_pad=self._tile_pad,
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
            "restored": _realesrgan_postprocess(
                output, original_size, scale=self.restore_scale
            )
        }

    def train(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "LibreRealESRGAN ships inference and validation only. Real-ESRGAN "
            "training is a GAN over a synthetic degradation pipeline and is out "
            "of scope; the paired restore trainer would run but would not match "
            "upstream quality. See the restore task docs."
        )


__all__ = ["LibreRealESRGAN", "REALESRGAN_SIZE_CONFIGS"]
