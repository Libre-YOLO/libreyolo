# Derived from torchvision v0.26.0 GeneralizedRCNNTransform normalization and
# fixed-size resize behavior. Upstream commit:
# 336d36e8db990a905498c73933e35231876e28bc (BSD-3-Clause).
"""Image preprocessing helpers for SSD300."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...postprocess.ssd import postprocess
from ...utils.image_loader import ImageInput, ImageLoader


SSD_IMAGE_MEAN = (0.48235, 0.45882, 0.40784)
SSD_IMAGE_STD = (1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0)


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 300,
) -> Tuple[np.ndarray, float]:
    """Resize RGB pixels directly to SSD's fixed canvas and subtract its mean."""
    if isinstance(input_size, (list, tuple)):
        input_h, input_w = int(input_size[0]), int(input_size[1])
    else:
        input_h = input_w = int(input_size)
    image = torch.from_numpy(np.array(img_rgb_hwc, copy=True, order="C"))
    image = image.permute(2, 0, 1).to(dtype=torch.float32) / 255.0
    mean = image.new_tensor(SSD_IMAGE_MEAN).view(3, 1, 1)
    std = image.new_tensor(SSD_IMAGE_STD).view(3, 1, 1)
    image = (image - mean) / std
    image = F.interpolate(
        image.unsqueeze(0),
        size=(input_h, input_w),
        mode="bilinear",
        align_corners=False,
    )[0]
    return np.ascontiguousarray(image.numpy()), 1.0


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


__all__ = [
    "SSD_IMAGE_MEAN",
    "SSD_IMAGE_STD",
    "postprocess",
    "preprocess_image",
    "preprocess_numpy",
]
