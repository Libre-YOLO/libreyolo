"""Preprocessing and checkpoint helpers for Deformable DETR.

The released models were evaluated with ImageNet normalization after resizing
the short side to 800 pixels (capped at 1333). LibreYOLO's fixed-shape runtime
uses an 800 x 800 PIL bilinear resize; boxes are rescaled independently along
the two image axes during postprocessing.
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
    """Resize RGB HWC input to a normalized square CHW array."""
    resized = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    return ((array - mean) / std).transpose(2, 0, 1), 1.0


def preprocess_image(
    image: ImageInput,
    input_size: int = 800,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load and preprocess one image for Deformable DETR inference."""
    img = ImageLoader.load(image, color_format=color_format)
    orig_w, orig_h = img.size
    chw, ratio = preprocess_numpy(np.asarray(img), input_size=input_size)
    return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio


def unwrap_deformable_detr_checkpoint(checkpoint: Any) -> dict:
    """Return parameters from an upstream or LibreYOLO checkpoint mapping."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dict, got {type(checkpoint).__name__}")
    for key in ("model", "model_state_dict", "state_dict", "ema"):
        inner = checkpoint.get(key)
        if isinstance(inner, dict):
            return inner
    return checkpoint
