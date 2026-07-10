"""Preprocessing for the ZipDepth family."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# ZipDepth's encoder downsamples by 32 (stem /4 + three stride-2 stages), and
# upstream rounds both input dimensions to multiples of 32.
IMGSZ_DIVISOR = 32


def make_divisible(value: float, divisor: int = IMGSZ_DIVISOR) -> int:
    """Round to the nearest multiple of ``divisor`` (at least ``divisor``).

    Mirrors upstream ``inference/predictor.py::make_divisible`` exactly,
    including Python banker's rounding, so converted checkpoints see the same
    input geometry as upstream inference.
    """
    return max(divisor, int(round(value / divisor) * divisor))


def compute_input_hw(orig_h: int, orig_w: int, input_size: int) -> Tuple[int, int]:
    """Model input (H, W) for an image: keep aspect, short side ~= input_size."""
    scale = input_size / min(orig_h, orig_w)
    return make_divisible(orig_h * scale), make_divisible(orig_w * scale)


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int = 384
) -> Tuple[np.ndarray, float]:
    """Resize an RGB image to ZipDepth's native input geometry.

    Reproduces upstream ``DepthInference.image2tensor``: keep the aspect ratio,
    scale the short side to ``input_size`` with both dimensions rounded to the
    nearest multiple of 32, bilinear interpolation on uint8. ImageNet
    normalization lives inside the network (checkpoint ``mean``/``std``
    buffers), so the returned array is plain ``(3, H, W)`` float32 in
    ``[0, 1]``. The second return value is the resize ratio (always ``1.0``:
    depth maps are resized back to the original canvas in ``_postprocess``).
    """
    img = np.asarray(img_rgb_hwc)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    orig_h, orig_w = img.shape[:2]
    new_h, new_w = compute_input_hw(orig_h, orig_w, input_size)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(chw), 1.0
