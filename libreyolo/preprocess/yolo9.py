"""YOLO9 input preprocessing.

Moved verbatim from ``libreyolo/models/yolo9/utils.py``, which re-exports
everything here for backward compatibility. Lives outside ``models/`` so the
ONNX backend can import it without pulling torch; see the package docstring.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..postprocess.yolo9 import ImageSize, _input_size_hw
from ..utils.image_loader import ImageInput, ImageLoader
from . import as_batched_input


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: ImageSize = 640,
) -> Tuple[np.ndarray, float]:
    """
    Preprocess RGB HWC uint8 image for YOLOv9 inference.

    Letterbox resize + normalize to 0-1 range.

    Args:
        img_rgb_hwc: Input image as RGB HWC uint8 numpy array.
        input_size: Target size for the model as int or (height, width).

    Returns:
        Tuple of (preprocessed CHW float32 array in RGB 0-1, ratio).
    """
    orig_h, orig_w = img_rgb_hwc.shape[:2]
    input_h, input_w = _input_size_hw(input_size)
    ratio = min(input_h / orig_h, input_w / orig_w)
    new_h = max(int(orig_h * ratio), 1)
    new_w = max(int(orig_w * ratio), 1)

    resized = cv2.resize(img_rgb_hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    arr = np.ascontiguousarray(padded, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), ratio


def preprocess_image(
    image: ImageInput, input_size: ImageSize = 640, color_format: str = "auto"
):
    """
    Preprocess image for YOLOv9 inference.

    Args:
        image: Input image (path, PIL, numpy, tensor, bytes, etc.)
        input_size: Target size for resizing as int or (height, width).
        color_format: Color format hint ("auto", "rgb", "bgr")

    Returns:
        Tuple of (preprocessed_tensor, original_image, original_size)
    """
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size  # (width, height)
    original_img = img.copy()

    img_chw, _ = preprocess_numpy(np.array(img), input_size)
    return as_batched_input(img_chw), original_img, original_size


__all__ = ["preprocess_numpy", "preprocess_image"]
