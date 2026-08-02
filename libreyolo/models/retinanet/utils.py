"""Inference helpers for the RetinaNet family."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int = 800
) -> Tuple[np.ndarray, float]:
    """Return a temporary fixed-square ImageNet-normalized skeleton input."""
    image = Image.fromarray(np.asarray(img_rgb_hwc, dtype=np.uint8)).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - _MEAN) / _STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)), 1.0


def preprocess_image(
    image: ImageInput,
    input_size: int = 800,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load one image for the construction-only skeleton."""
    loaded = ImageLoader.load(image, color_format=color_format)
    original_size = loaded.size
    image_chw, ratio = preprocess_numpy(np.asarray(loaded), input_size)
    return torch.from_numpy(image_chw).unsqueeze(0), loaded, original_size, ratio


__all__ = ["preprocess_image", "preprocess_numpy"]
