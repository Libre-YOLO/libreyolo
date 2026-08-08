"""LibreDOMEDETR preprocessing helpers.

Dome-DETR inherits D-FINE's eval transform: plain square resize to the eval
size, ``uint8 / 255``, no letterbox and no ImageNet normalisation. The only
difference is the size itself (800 rather than 640).
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader


def unwrap_domedetr_checkpoint(checkpoint: Mapping | Any):
    """Extract the state dict from an upstream Dome-DETR checkpoint.

    Upstream saves ``{"model": state_dict}`` for this family (no EMA copy in
    the published ``best_ckpts_dome_2026`` files), but the EMA layout is
    handled too since the training script can emit it.
    """
    if not isinstance(checkpoint, Mapping):
        return checkpoint

    ema = checkpoint.get("ema")
    if isinstance(ema, Mapping):
        module = ema.get("module")
        if isinstance(module, Mapping):
            return module

    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value

    return checkpoint


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size: int = 800) -> Tuple[np.ndarray, float]:
    """RGB HWC uint8 -> CHW float32 in [0, 1], square-resized. Ratio is always 1.0."""
    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), 1.0


def preprocess_image(
    image: ImageInput,
    input_size: int = 800,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()

    img_chw, ratio = preprocess_numpy(np.array(img), input_size=input_size)
    return torch.from_numpy(img_chw).unsqueeze(0), original_img, original_size, ratio
