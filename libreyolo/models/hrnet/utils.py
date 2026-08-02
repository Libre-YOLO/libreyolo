"""HRNet crop preprocessing helpers."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch

from ...utils.image_loader import ImageLoader

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def _size_hw(input_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(input_size, int):
        return input_size, input_size
    if len(input_size) != 2:
        raise ValueError(f"input_size must be an int or (height, width), got {input_size!r}")
    return int(input_size[0]), int(input_size[1])


def preprocess_numpy(
    image_rgb_hwc: np.ndarray,
    input_size: int | Sequence[int],
) -> tuple[np.ndarray, tuple[float, float]]:
    """Resize an already-aligned person crop and apply ImageNet normalization."""
    input_h, input_w = _size_hw(input_size)
    original_h, original_w = image_rgb_hwc.shape[:2]
    resized = cv2.resize(image_rgb_hwc, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)
    return chw, (input_w / original_w, input_h / original_h)


def preprocess_crop_image(image, input_size, color_format: str = "auto"):
    """Preprocess a person crop for the native single-crop compatibility path."""
    original = ImageLoader.load(image, color_format=color_format)
    original_size = original.size
    chw, ratio = preprocess_numpy(np.asarray(original), input_size)
    return torch.from_numpy(chw).unsqueeze(0), original, original_size, ratio
