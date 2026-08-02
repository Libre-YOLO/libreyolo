"""Image preprocessing helpers for SSD300."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader


SSD_IMAGE_MEAN = (0.48235 * 255.0, 0.45882 * 255.0, 0.40784 * 255.0)


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 300,
) -> Tuple[np.ndarray, float]:
    """Resize RGB pixels directly to SSD's fixed canvas and subtract its mean."""
    if isinstance(input_size, (list, tuple)):
        input_h, input_w = int(input_size[0]), int(input_size[1])
    else:
        input_h = input_w = int(input_size)
    resized = cv2.resize(
        img_rgb_hwc,
        (input_w, input_h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    resized -= np.asarray(SSD_IMAGE_MEAN, dtype=np.float32)
    return np.ascontiguousarray(resized.transpose(2, 0, 1)), 1.0


def preprocess_image(
    image: ImageInput,
    input_size: int = 300,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load and preprocess one image for native SSD inference."""
    loaded = ImageLoader.load(image, color_format=color_format)
    original_size = loaded.size
    original_image = loaded.copy()
    chw, ratio = preprocess_numpy(np.asarray(loaded), input_size)
    return (
        torch.from_numpy(chw).unsqueeze(0),
        original_image,
        original_size,
        ratio,
    )


__all__ = ["SSD_IMAGE_MEAN", "preprocess_image", "preprocess_numpy"]
