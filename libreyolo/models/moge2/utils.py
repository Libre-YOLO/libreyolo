"""Image preprocessing for the MoGe-2 normal family."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

PATCH_SIZE = 14


def make_divisible(value: float, divisor: int = PATCH_SIZE) -> int:
    """Round to the nearest positive multiple of the DINOv2 patch size."""
    return max(divisor, int(round(value / divisor) * divisor))


def compute_input_hw(
    orig_h: int,
    orig_w: int,
    input_size: int,
) -> Tuple[int, int]:
    """Keep aspect ratio while scaling the short side to ``input_size``."""
    scale = float(input_size) / min(orig_h, orig_w)
    return make_divisible(orig_h * scale), make_divisible(orig_w * scale)


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 518,
) -> Tuple[np.ndarray, float]:
    """Return a patch-aligned RGB CHW tensor in ``[0, 1]``.

    MoGe accepts arbitrary aspect ratios. Native prediction therefore keeps
    the input aspect ratio, scales the short side, and rounds both dimensions
    to the ViT patch grid. ImageNet normalization remains inside the network,
    matching the official implementation and checkpoint buffers.
    """
    image = np.asarray(img_rgb_hwc)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    orig_h, orig_w = image.shape[:2]
    new_h, new_w = compute_input_hw(orig_h, orig_w, input_size)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(chw), 1.0


__all__ = [
    "PATCH_SIZE",
    "compute_input_hw",
    "make_divisible",
    "preprocess_numpy",
]
