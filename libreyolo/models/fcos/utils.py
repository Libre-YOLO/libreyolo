"""Preprocessing helpers for the FCOS family scaffold."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int = 800
) -> Tuple[np.ndarray, float]:
    """Return an RGB CHW float tensor; exact aspect resize lands with the graph."""
    del input_size
    image = np.asarray(img_rgb_hwc, dtype=np.float32) / 255.0
    return np.ascontiguousarray(image.transpose(2, 0, 1)), 1.0


def preprocess_image(
    image: ImageInput,
    color_format: str = "auto",
    input_size: int = 800,
) -> tuple[torch.Tensor, Image.Image, tuple[int, int], float]:
    """Load one image for the importable scaffold."""
    loaded = ImageLoader.load(image, color_format=color_format)
    original_size = loaded.size
    image_chw, ratio = preprocess_numpy(np.asarray(loaded), input_size)
    return torch.from_numpy(image_chw).unsqueeze(0), loaded, original_size, ratio


__all__ = ["preprocess_image", "preprocess_numpy"]
