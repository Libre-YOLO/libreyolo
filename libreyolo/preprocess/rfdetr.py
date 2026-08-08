"""RF-DETR input preprocessing.

Moved verbatim from ``libreyolo/models/rfdetr/utils.py``, which re-exports it for backward
compatibility. Lives outside ``models/`` so the ONNX backend can import it
without pulling torch; see the package docstring.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 560,
) -> Tuple[np.ndarray, float]:
    """
    Preprocess RGB HWC uint8 image for RF-DETR inference.

    Simple resize + ImageNet normalization.

    Args:
        img_rgb_hwc: Input image as RGB HWC uint8 numpy array.
        input_size: Target size for the model.

    Returns:
        Tuple of (preprocessed CHW float32 array with ImageNet norm, ratio).
    """
    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1), 1.0


__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "preprocess_numpy"]
