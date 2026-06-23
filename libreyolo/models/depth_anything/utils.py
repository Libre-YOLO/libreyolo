"""Preprocessing for the Depth Anything V2 family."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ._vendor.util.transform import PrepareForNet, Resize


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int = 518
) -> Tuple[np.ndarray, float]:
    """Resize an RGB image to Depth Anything V2's native input geometry.

    Reproduces upstream ``infer_image``: keep the aspect ratio and scale the
    short side up to a multiple of 14 (the DINOv2 patch grid) with cubic
    interpolation. ImageNet normalization is applied inside the network, so the
    returned array is a plain ``(3, H, W)`` float32 in ``[0, 1]``. The second
    return value is the resize ratio (always ``1.0``: depth maps are resized
    back to the original canvas in ``_postprocess``, not rescaled by ratio).
    """
    image = np.asarray(img_rgb_hwc, dtype=np.float32) / 255.0
    resize = Resize(
        width=input_size,
        height=input_size,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_CUBIC,
    )
    sample = PrepareForNet()(resize({"image": image}))
    return sample["image"], 1.0
