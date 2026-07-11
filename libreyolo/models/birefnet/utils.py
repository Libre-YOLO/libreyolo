"""Preprocessing for the LibreBiRefNet family.

Reproduces the upstream BiRefNet inference transform exactly (the matte quality
depends on it): ``Image.convert('RGB')`` -> bilinear resize to the fixed native
square -> ``/255`` -> ImageNet normalization. The resize deliberately squashes
the aspect ratio (upstream ``transforms.Resize((H, W))``); the matte is resized
back to the original canvas in ``_postprocess``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

# ImageNet statistics, matching upstream ``transforms.Normalize``.
_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size: int = 1024) -> Tuple[np.ndarray, float]:
    """Resize an RGB ``HxWx3`` array to BiRefNet's native square and normalize.

    Uses PIL bilinear resize (identical to upstream ``transforms.Resize``), then
    ``/255`` and ImageNet normalization. Returns ``((3, S, S) float32, ratio)``;
    ``ratio`` is always ``1.0`` (the matte is resized back to the original canvas
    in ``_postprocess``, not rescaled).
    """
    arr = np.asarray(img_rgb_hwc)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    pil = Image.fromarray(arr.astype(np.uint8), mode="RGB").resize(
        (input_size, input_size), Image.BILINEAR
    )
    chw = np.asarray(pil, dtype=np.float32) / 255.0
    chw = (chw - _MEAN) / _STD
    chw = np.ascontiguousarray(chw.transpose(2, 0, 1))
    return chw, 1.0


def preprocess_image(
    image,
    input_size: int = 1024,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load an image and return ``(tensor, pil_rgb, (orig_w, orig_h), ratio)``."""
    from ...utils.image_loader import ImageLoader

    img = ImageLoader.load(image, color_format=color_format)
    orig_w, orig_h = img.size
    chw, ratio = preprocess_numpy(np.asarray(img), input_size)
    tensor = torch.from_numpy(chw).unsqueeze(0)
    return tensor, img, (orig_w, orig_h), ratio
