"""CenterNet image preprocessing and checkpoint helpers.

The affine transform and BGR normalization match xingyizhou/CenterNet commit
4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c (MIT).
"""

from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np
import torch

from ...utils.image_loader import ImageInput, ImageLoader

CENTERNET_MEAN = np.array([0.408, 0.447, 0.470], dtype=np.float32)
CENTERNET_STD = np.array([0.289, 0.274, 0.278], dtype=np.float32)


def _third_point(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    direction = first - second
    return second + np.array([-direction[1], direction[0]], dtype=np.float32)


def _direction(point: np.ndarray, rotation: float) -> np.ndarray:
    sine, cosine = np.sin(rotation), np.cos(rotation)
    return np.array(
        [point[0] * cosine - point[1] * sine, point[0] * sine + point[1] * cosine],
        dtype=np.float32,
    )


def get_affine_transform(
    center: np.ndarray,
    scale: float | np.ndarray,
    output_size: tuple[int, int],
    *,
    inverse: bool = False,
) -> np.ndarray:
    """Return CenterNet's zero-rotation source/canvas affine matrix."""
    if not isinstance(scale, np.ndarray):
        scale = np.array([scale, scale], dtype=np.float32)
    source_width = float(scale[0])
    destination_width, destination_height = output_size
    source_direction = _direction(
        np.array([0.0, -0.5 * source_width], dtype=np.float32), 0.0
    )
    destination_direction = np.array([0.0, -0.5 * destination_width], dtype=np.float32)

    source = np.zeros((3, 2), dtype=np.float32)
    destination = np.zeros((3, 2), dtype=np.float32)
    source[0] = center
    source[1] = center + source_direction
    source[2] = _third_point(source[0], source[1])
    destination[0] = [0.5 * destination_width, 0.5 * destination_height]
    destination[1] = destination[0] + destination_direction
    destination[2] = _third_point(destination[0], destination[1])
    if inverse:
        return cv2.getAffineTransform(destination, source)
    return cv2.getAffineTransform(source, destination)


def image_geometry(
    original_size: tuple[int, int], input_size: int
) -> tuple[np.ndarray, float]:
    """Return the official center and scalar affine extent for one image."""
    width, height = original_size
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    return center, float(max(width, height))


def preprocess_bgr(
    image_bgr_hwc: np.ndarray, input_size: int = 512
) -> Tuple[np.ndarray, float]:
    """Affine-warp a BGR HWC image and return normalized CHW float32."""
    height, width = image_bgr_hwc.shape[:2]
    center, scale = image_geometry((width, height), input_size)
    transform = get_affine_transform(center, scale, (input_size, input_size))
    warped = cv2.warpAffine(
        image_bgr_hwc,
        transform,
        (input_size, input_size),
        flags=cv2.INTER_LINEAR,
    )
    normalized = ((warped / 255.0 - CENTERNET_MEAN) / CENTERNET_STD).astype(np.float32)
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    return chw, input_size / scale


def preprocess_numpy(
    image_rgb_hwc: np.ndarray, input_size: int = 512
) -> Tuple[np.ndarray, float]:
    """Convert an RGB HWC image through CenterNet's BGR affine pipeline."""
    image_bgr = np.ascontiguousarray(np.asarray(image_rgb_hwc)[..., ::-1])
    return preprocess_bgr(image_bgr, input_size=input_size)


def preprocess_image(
    image: ImageInput,
    input_size: int = 512,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
    """Load one image and reproduce official fixed-resolution preprocessing."""
    loaded = ImageLoader.load(image, color_format=color_format)
    original_size = loaded.size
    chw, ratio = preprocess_numpy(np.asarray(loaded), input_size=input_size)
    return torch.from_numpy(chw).unsqueeze(0), loaded, original_size, ratio


def unwrap_centernet_checkpoint(checkpoint: Any) -> dict:
    """Extract parameters from official or LibreYOLO checkpoint containers."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")
    for key in ("model", "state_dict", "model_state_dict", "ema"):
        inner = checkpoint.get(key)
        if isinstance(inner, dict):
            return inner
    return checkpoint


__all__ = [
    "CENTERNET_MEAN",
    "CENTERNET_STD",
    "get_affine_transform",
    "image_geometry",
    "preprocess_bgr",
    "preprocess_image",
    "preprocess_numpy",
    "unwrap_centernet_checkpoint",
]
