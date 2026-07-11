"""Image preprocessing and dataset helpers for LibrePatchCore."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ...data.anomaly_dataset import (
    resolve_anomaly_root,
    resolve_anomaly_test_samples,
    resolve_good_training_images,
)

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def preprocess_numpy(image_rgb_hwc: np.ndarray, input_size: int = 224) -> tuple[np.ndarray, float]:
    """Resize, center-crop, and ImageNet-normalize an RGB image."""
    image = Image.fromarray(np.asarray(image_rgb_hwc, dtype=np.uint8), mode="RGB")
    width, height = image.size
    resize_short = int(round(input_size / 0.875))
    scale = resize_short / min(width, height)
    resized = image.resize(
        (max(input_size, int(round(width * scale))), max(input_size, int(round(height * scale)))),
        Image.Resampling.BILINEAR,
    )
    left = (resized.width - input_size) // 2
    top = (resized.height - input_size) // 2
    crop = resized.crop((left, top, left + input_size, top + input_size))
    array = np.asarray(crop, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)), 1.0


def preprocess_image(
    image: ImageInput,
    *,
    input_size: int = 224,
    color_format: str = "auto",
) -> tuple[torch.Tensor, Image.Image, tuple[int, int], float]:
    original = ImageLoader.load(image, color_format=color_format)
    if not isinstance(original, Image.Image):
        original = Image.fromarray(np.asarray(original)).convert("RGB")
    original = original.convert("RGB")
    chw, ratio = preprocess_numpy(np.asarray(original), input_size)
    return torch.from_numpy(chw).unsqueeze(0), original, original.size, ratio


def iter_preprocessed_batches(
    paths: Iterable, batch_size: int, input_size: int
) -> Iterable[torch.Tensor]:
    batch: list[torch.Tensor] = []
    for path in paths:
        tensor, _, _, _ = preprocess_image(str(path), input_size=input_size)
        batch.append(tensor)
        if len(batch) == batch_size:
            yield torch.cat(batch, dim=0)
            batch = []
    if batch:
        yield torch.cat(batch, dim=0)


__all__ = [
    "iter_preprocessed_batches",
    "preprocess_image",
    "preprocess_numpy",
    "resolve_anomaly_root",
    "resolve_anomaly_test_samples",
    "resolve_good_training_images",
]
