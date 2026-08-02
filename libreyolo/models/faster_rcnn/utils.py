"""Inference helpers for the Faster R-CNN family."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader


def preprocess_numpy(img_rgb_hwc: np.ndarray) -> Tuple[np.ndarray, float]:
    """Convert RGB HWC uint8 input to unresized CHW float input."""
    image = np.asarray(img_rgb_hwc, dtype=np.float32) / 255.0
    return image.transpose(2, 0, 1), 1.0


def preprocess_image(
    image: ImageInput, color_format: str = "auto"
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load an image without resizing or ImageNet normalization.

    Both operations belong to the model's in-graph GeneralizedRCNNTransform.
    """
    loaded = ImageLoader.load(image, color_format=color_format)
    original_size = loaded.size
    image_chw, ratio = preprocess_numpy(np.array(loaded))
    return torch.from_numpy(image_chw).unsqueeze(0), loaded, original_size, ratio
