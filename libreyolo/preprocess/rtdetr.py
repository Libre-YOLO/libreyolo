"""RT-DETR input preprocessing.

Moved verbatim from ``libreyolo/models/rtdetr/utils.py``, which re-exports it for backward
compatibility. Lives outside ``models/`` so the ONNX backend can import it
without pulling torch; see the package docstring.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize and normalize image for RTDETR inference. Returns float32 NCHW array.

    Args:
        img_rgb_hwc: Input image in RGB format, HWC layout
        input_size: Target input size (square)

    Returns:
        Tuple of (preprocessed image as float32 NCHW array, original size (h, w))
    """
    import cv2

    h, w = img_rgb_hwc.shape[:2]
    img = cv2.resize(img_rgb_hwc, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
    return img, (h, w)


__all__ = ["preprocess_numpy"]
