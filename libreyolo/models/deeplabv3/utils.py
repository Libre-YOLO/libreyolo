"""DeepLabv3 preprocessing helpers."""

from __future__ import annotations

import cv2
import numpy as np

from ...postprocess.deeplabv3 import postprocess, semantic_logits


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def _input_size_hw(input_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(input_size, int):
        return input_size, input_size
    if len(input_size) != 2:
        raise ValueError(
            f"input_size must be int or (height, width), got {input_size!r}"
        )
    return int(input_size[0]), int(input_size[1])


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int | tuple[int, int] = 520,
) -> tuple[np.ndarray, float]:
    """Resize RGB to the fixed deployment canvas and apply ImageNet normalization."""
    input_h, input_w = _input_size_hw(input_size)
    resized = cv2.resize(
        img_rgb_hwc,
        (input_w, input_h),
        interpolation=cv2.INTER_LINEAR,
    )
    arr = np.ascontiguousarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)
    return chw, 1.0


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "postprocess",
    "preprocess_numpy",
    "semantic_logits",
]
