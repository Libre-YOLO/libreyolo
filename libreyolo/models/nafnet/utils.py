"""Utility helpers for LibreNAFNet."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader


def _pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
) -> torch.Tensor:
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if not pad_h and not pad_w:
        return tensor
    mode = "reflect" if h > 1 and w > 1 and pad_h < h and pad_w < w else "replicate"
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)


def preprocess_image(
    image: ImageInput,
    *,
    pad_multiple: int,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load RGB image as float [0, 1] tensor and pad without resizing."""

    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    tensor = _pad_to_multiple(tensor, int(pad_multiple))
    return tensor, img, original_size, 1.0


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size: int | tuple[int, int]):
    """Exporter calibration helper: convert RGB HWC to CHW float [0, 1]."""

    del input_size
    arr = np.asarray(img_rgb_hwc, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr.transpose(2, 0, 1).astype(np.float32), 1.0


__all__ = ["preprocess_image", "preprocess_numpy"]

