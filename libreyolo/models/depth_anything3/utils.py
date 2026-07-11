"""Native DA3 keep-aspect preprocessing without upstream runtime dependencies."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def _nearest_multiple(value: int, divisor: int) -> int:
    down = (value // divisor) * divisor
    up = down + divisor
    return max(divisor, up if abs(up - value) <= abs(value - down) else down)


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int = 504
) -> Tuple[np.ndarray, float]:
    """Reproduce DA3's default upper-bound resize for one RGB image.

    The longest side is resized to ``input_size`` and both dimensions are then
    resized to the nearest multiple of the 14-pixel patch size. The returned
    CHW array remains in ``[0, 1]`` because normalization occurs in the network.
    """
    image = np.asarray(img_rgb_hwc)
    height, width = image.shape[:2]
    scale = input_size / float(max(width, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    image = cv2.resize(
        image, (resized_width, resized_height), interpolation=interpolation
    )

    final_width = _nearest_multiple(resized_width, 14)
    final_height = _nearest_multiple(resized_height, 14)
    if (final_width, final_height) != (resized_width, resized_height):
        interpolation = (
            cv2.INTER_CUBIC
            if final_width > resized_width or final_height > resized_height
            else cv2.INTER_AREA
        )
        image = cv2.resize(
            image, (final_width, final_height), interpolation=interpolation
        )

    chw = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return chw, 1.0
