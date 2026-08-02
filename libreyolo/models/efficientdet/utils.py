"""EfficientDet preprocessing and compatibility re-exports."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int = 512
) -> Tuple[np.ndarray, float]:
    """Match ``effdet`` eval resize: keep aspect, pad at bottom/right."""
    image = Image.fromarray(img_rgb_hwc)
    width, height = image.size
    scale = min(input_size / height, input_size / width)
    scaled_h, scaled_w = int(height * scale), int(width * scale)
    resized = image.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
    fill = tuple(int(round(255 * value)) for value in IMAGENET_MEAN)
    canvas = Image.new("RGB", (input_size, input_size), color=fill)
    canvas.paste(resized, (0, 0))
    array = np.asarray(canvas, dtype=np.float32)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32) * 255.0
    std = np.asarray(IMAGENET_STD, dtype=np.float32) * 255.0
    array = (array - mean) / std
    return np.ascontiguousarray(array.transpose(2, 0, 1)), scale


def preprocess_image(
    image: ImageInput,
    input_size: int = 512,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    loaded = ImageLoader.load(image, color_format=color_format)
    original_size = loaded.size
    original_img = loaded.copy()
    chw, ratio = preprocess_numpy(np.asarray(loaded), input_size=input_size)
    return torch.from_numpy(chw).unsqueeze(0), original_img, original_size, ratio


def postprocess(*args, **kwargs):
    """Placeholder until the raw graph passes the upstream parity gate."""
    del args, kwargs
    raise NotImplementedError("EfficientDet postprocessing is not implemented yet.")


__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "postprocess", "preprocess_image", "preprocess_numpy"]
