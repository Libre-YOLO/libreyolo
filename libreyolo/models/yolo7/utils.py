"""YOLOv7 preprocessing: letterbox, RGB, [0,1], gray(114) pad (YOLOv5/v7 style)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader

_PAD_VALUE = 114


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size: int = 640) -> Tuple[np.ndarray, float]:
    """Letterbox an RGB HWC uint8 image to ``input_size``; normalize to [0,1]."""
    orig_h, orig_w = img_rgb_hwc.shape[:2]
    ratio = min(input_size / orig_h, input_size / orig_w)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
    resized = Image.fromarray(img_rgb_hwc).resize((new_w, new_h), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (input_size, input_size), (_PAD_VALUE,) * 3)
    padded.paste(resized, (0, 0))
    arr = np.array(padded, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), ratio


def preprocess_image(
    image: ImageInput, input_size: int = 640, color_format: str = "auto"
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()
    chw, ratio = preprocess_numpy(np.array(img), input_size)
    return torch.from_numpy(chw).unsqueeze(0), original_img, original_size, ratio
