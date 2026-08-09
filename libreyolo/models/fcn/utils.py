"""Input helpers for the LibreFCN semantic-segmentation family."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int | tuple[int, int] = 520,
) -> Tuple[np.ndarray, float]:
    """Stretch RGB input to the export canvas and scale it to ``[0, 1]``."""
    if isinstance(input_size, int):
        height = width = input_size
    else:
        height, width = int(input_size[0]), int(input_size[1])
    resized = cv2.resize(img_rgb_hwc, (width, height), interpolation=cv2.INTER_LINEAR)
    array = np.ascontiguousarray(resized, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1), 1.0


def preprocess_image(
    image: ImageInput,
    input_size: int,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load and stretch one image; channel normalization stays in the graph."""
    loaded = ImageLoader.load(image, color_format=color_format)
    original_size = loaded.size
    image_chw, ratio = preprocess_numpy(np.asarray(loaded), input_size)
    return torch.from_numpy(image_chw).unsqueeze(0), loaded, original_size, ratio


__all__ = ["preprocess_image", "preprocess_numpy"]
