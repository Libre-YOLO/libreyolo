"""LibreYOLO wrapper for ZipDepth monocular depth estimation.

ZipDepth (https://github.com/fabiotosi92/ZipDepth, ECCV 2026, University of
Bologna, MIT) is a ~6M-parameter reparameterizable CNN distilled from Depth
Anything V2 Large. It predicts the same primitive as LibreYOLO's other depth
family (``Results.depth_map``: dense relative inverse depth, higher = closer,
no metric unit), at a fraction of the compute: it is the speed/edge tier of the
``depth`` task, next to ``depth_anything`` as the quality tier.

Both code and weights are MIT, so LibreYOLO rehosts the converted checkpoints
(unlike Depth Anything V2 Base/Large/Giant, which are CC-BY-NC). Note the
weights were trained on pseudo-labels produced by Depth Anything V2 Large; see
the family ``NOTICE`` for the provenance chain.

Two checkpoints ship, differing only in the trained 2x upsampling head:

- ``LibreZipDepthb-depth.pt``: convex upsampling via ``F.unfold`` (GPU/CPU).
- ``LibreZipDepthbnpu-depth.pt``: unfold-free upsampling for NPU/edge
  compilers that lack gather/unfold support. Use it when exporting to
  constrained runtimes; outputs are visually equivalent.

Training is out of scope (see :meth:`LibreZipDepth.train`); export follows the
fixed-resolution dense contract (stretch-resize to the exported canvas, resize
the depth map back to the original canvas).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from .nn import LibreZipDepthNet
from .utils import IMGSZ_DIVISOR, preprocess_numpy

logger = logging.getLogger(__name__)


class LibreZipDepth(BaseModel):
    """ZipDepth: image -> dense relative inverse-depth map (lightweight CNN)."""

    FAMILY = "zipdepth"
    FILENAME_PREFIX = "LibreZipDepth"
    WEIGHT_EXT = ".pt"
    # Default short-side inference resolution (upstream default). Both size
    # codes are the 'base' capacity; 'bnpu' swaps in the unfold-free decoder
    # trained for NPU/edge compilers.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "b": 384,
        "bnpu": 384,
    }
    SUPPORTED_TASKS = ("depth",)
    DEFAULT_TASK = "depth"

    # Encoder stride; the depth dataset and validator enforce divisibility.
    depth_imgsz_divisor = IMGSZ_DIVISOR
    # Validation letterboxes to a fixed square (padded pixels are invalid depth,
    # excluded from metrics); predict uses the native keep-aspect resize, so val
    # metrics on non-square images are a documented approximation of predict.
    depth_resize_mode = "letterbox"

    # Keep-aspect preprocessing yields per-image variable sizes, so the stacked
    # single-forward batched-predict path does not apply.
    SUPPORTS_BATCHED_PREDICT = False

    # dims[0] // 2 of the stem output uniquely identifies the capacity tier
    # (only 'base' has released checkpoints today).
    _STEM_CH_TO_VARIANT: ClassVar[Dict[int, str]] = {
        12: "small",
        24: "base",
        32: "large",
        48: "giant",
    }
    _VARIANT_TO_SIZE: ClassVar[Dict[str, str]] = {"base": "b"}

    _UPSTREAM_URL = "https://github.com/fabiotosi92/ZipDepth"

    # ====================================================================
    # Checkpoint detection
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # The stem + convex-upsampling head key pair is unique to ZipDepth; no
        # other family uses ``encoder.stem_half`` or ``decoder.convex_up``.
        has_stem = "encoder.stem_half.conv.weight" in weights_dict
        has_up = any(k.startswith("decoder.convex_up.") for k in weights_dict)
        if has_stem and has_up and any(".fused_conv." in k for k in weights_dict):
            raise ValueError(
                "This looks like a fused (reparameterized) ZipDepth state dict. "
                "LibreYOLO checkpoints store the unfused training-form weights "
                "and fuse at load time; re-convert from the unfused upstream "
                "checkpoint with weights/convert_zipdepth_weights.py."
            )
        return has_stem and has_up

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        stem = weights_dict.get("encoder.stem_half.conv.weight")
        if stem is None or getattr(stem, "ndim", 0) != 4:
            return None
        variant = cls._STEM_CH_TO_VARIANT.get(int(stem.shape[0]))
        size = cls._VARIANT_TO_SIZE.get(variant) if variant else None
        if size is None:
            return None
        # The NPU decoder flavor is a separately trained checkpoint, not a
        # graph rewrite: discriminate on its unfold-free head keys.
        if any(k.startswith("decoder.convex_up.where_conv.") for k in weights_dict):
            return f"{size}npu"
        return size

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # The single class slot is a schema placeholder; depth has no classes.
        return 1

    # ====================================================================
    # Construction
    # ====================================================================

    def __init__(
        self,
        model_path,
        size: str = "b",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,  # always 1 ("depth"); ignore any caller-provided value
            device=device,
            task=task,
            **kwargs,
        )
        self.model.eval()
        # A file path is stored but not loaded by BaseModel; load it here (a raw
        # state_dict passed as ``model_path`` is already loaded by BaseModel).
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        # Reparameterize for inference once weights are in place. Checkpoints
        # always store the unfused form; fusing is idempotent and confined to
        # the live module, so nothing fused is ever re-saved.
        if model_path is not None:
            self.model.fuse_for_inference()
        # Enforce the single depth slot last, after any loading path.
        self.nb_classes = 1
        self.names = {0: "depth"}

    def _init_model(self) -> nn.Module:
        return LibreZipDepthNet(size=self.size)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "encoder": self.model.encoder,
            "decoder": self.model.decoder,
        }

    # ====================================================================
    # Inference surface
    # ====================================================================

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_res = (
            input_size if input_size is not None else self._get_input_size()
        )
        if effective_res % self.depth_imgsz_divisor:
            raise ValueError(
                f"ZipDepth imgsz={effective_res} must be divisible by "
                f"{self.depth_imgsz_divisor} (encoder stride)."
            )
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), effective_res)
        img_tensor = torch.from_numpy(chw).unsqueeze(0)
        return img_tensor, img, (orig_w, orig_h), ratio

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
        depth = output
        if isinstance(depth, dict):
            depth = depth.get("depth", depth.get("predictions"))
        orig_w, orig_h = original_size
        # align_corners=True matches upstream infer_image, so converted
        # checkpoints reproduce upstream depth maps on the original canvas.
        depth = F.interpolate(
            depth.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=True,
        )
        return {"depth": depth[0, 0].cpu()}

    # ====================================================================
    # Weights I/O
    # ====================================================================
    # Weights load (and auto-download) through BaseModel. A missing checkpoint
    # is fetched from the LibreYOLO Hugging Face org via the inherited
    # ``get_download_url`` -> ``LibreYOLO/LibreZipDepth{b,bnpu}-depth``. The
    # offline path is to convert an upstream checkpoint with
    # ``weights/convert_zipdepth_weights.py``. Code and weights are MIT.

    # ====================================================================
    # Out of scope
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training ZipDepth in LibreYOLO is out of scope: the upstream "
            "recipe distills Depth Anything V2 Large pseudo-labels over ~14M "
            "images, which is not reproducible as a LibreYOLO training run. "
            f"To fine-tune, train upstream at {self._UPSTREAM_URL} and convert "
            "the result with weights/convert_zipdepth_weights.py."
        )
