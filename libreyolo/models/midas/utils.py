"""Image preprocessing helpers for MiDaS."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


IMGSZ_DIVISOR = 32


def _constrain_to_multiple(
    value: float,
    *,
    multiple: int,
    min_value: int = 0,
    max_value: int | None = None,
) -> int:
    """Match MiDaS's nearest-multiple resize rule."""
    constrained = int(np.round(value / multiple) * multiple)
    if max_value is not None and constrained > max_value:
        constrained = int(np.floor(value / multiple) * multiple)
    if constrained < min_value:
        constrained = int(np.ceil(value / multiple) * multiple)
    return constrained


def _resize_shape(width: int, height: int, input_size: int, size: str) -> tuple[int, int]:
    scale_height = input_size / height
    scale_width = input_size / width

    if size == "s":
        # v2.1 Small uses ``upper_bound``: the whole image fits within the
        # requested canvas while preserving aspect ratio.
        scale = min(scale_width, scale_height)
        new_height = _constrain_to_multiple(
            scale * height,
            multiple=IMGSZ_DIVISOR,
            max_value=input_size,
        )
        new_width = _constrain_to_multiple(
            scale * width,
            multiple=IMGSZ_DIVISOR,
            max_value=input_size,
        )
    elif size == "l":
        # DPT-Large uses ``minimal``: choose the target-side scale closest to
        # one, then round both axes to the ViT/DPT multiple.
        scale = (
            scale_width
            if abs(1.0 - scale_width) < abs(1.0 - scale_height)
            else scale_height
        )
        new_height = _constrain_to_multiple(
            scale * height,
            multiple=IMGSZ_DIVISOR,
        )
        new_width = _constrain_to_multiple(
            scale * width,
            multiple=IMGSZ_DIVISOR,
        )
    else:
        raise ValueError(f"Unknown MiDaS size {size!r}; expected 's' or 'l'.")

    return new_width, new_height


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int,
    size: str,
) -> Tuple[np.ndarray, float]:
    """Reproduce the official per-variant MiDaS input geometry.

    The returned tensor is RGB float32 in ``[0, 1]``. Variant-specific
    normalization lives inside the network so native and exported runtimes
    consume the same public input contract.
    """
    image = np.asarray(img_rgb_hwc, dtype=np.float32) / 255.0
    height, width = image.shape[:2]
    new_width, new_height = _resize_shape(width, height, input_size, size)
    image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC,
    )
    chw = np.ascontiguousarray(image.transpose(2, 0, 1)).astype(np.float32)
    return chw, 1.0
