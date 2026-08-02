"""DETR inference preprocessing and checkpoint helpers.

Official DETR validation resizes the short image side to 800 with a 1333-pixel
long-side cap. LibreYOLO's checkpoint families expose one fixed square canvas,
so this port uses one PIL bilinear resize to 800x800 followed by the same RGB
ImageNet normalization. The fixed canvas has no padded pixels, which gives the
native model an all-false padding mask and keeps export deterministic.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 800,
) -> Tuple[np.ndarray, float]:
    """Stretch-resize an RGB image to the fixed DETR square and normalize."""
    resized = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    array = (array - mean) / std
    return array.transpose(2, 0, 1), 1.0


def preprocess_image(
    image: ImageInput,
    input_size: int = 800,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load and preprocess one image for native DETR inference."""
    loaded = ImageLoader.load(image, color_format=color_format)
    original_width, original_height = loaded.size
    chw, ratio = preprocess_numpy(np.asarray(loaded), input_size)
    return (
        torch.from_numpy(chw).unsqueeze(0),
        loaded,
        (original_width, original_height),
        ratio,
    )


def unwrap_detr_checkpoint(checkpoint: Any) -> dict:
    """Extract the raw DETR state dict from official or LibreYOLO wrappers."""
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a checkpoint dictionary, got {type(checkpoint).__name__}"
        )
    for key in ("model", "state_dict", "model_state_dict", "ema"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint
