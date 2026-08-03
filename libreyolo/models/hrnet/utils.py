"""HRNet top-down pose geometry, preprocessing, and heatmap decoding.

Affine, box-to-center/scale, and decoding arithmetic is adapted from
``lib/utils/transforms.py``, ``lib/core/inference.py``,
``lib/dataset/coco.py``, and ``demo/inference.py`` in
``leoxiaobin/deep-high-resolution-net.pytorch`` at commit
``6f69e4676ad8d43d0d61b64b1b9726f0c369e7b1`` (MIT License).

Copyright (c) Microsoft. Written by Bin Xiao.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch

from ...postprocess.hrnet import (
    decode_heatmaps,
    flip_back,
    flip_back_tensor,
    get_max_preds,
    transform_preds,
)
from ...utils.image_loader import ImageLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PIXEL_STD = 200.0
BOX_SCALE_PADDING = 1.25

__all__ = [
    "decode_heatmaps",
    "flip_back",
    "flip_back_tensor",
    "get_max_preds",
    "transform_preds",
]


def size_hw(input_size: int | Sequence[int]) -> tuple[int, int]:
    """Normalize an integer or ``(height, width)`` input size."""
    if isinstance(input_size, int):
        return input_size, input_size
    if len(input_size) != 2:
        raise ValueError(
            f"input_size must be an int or (height, width), got {input_size!r}"
        )
    height, width = int(input_size[0]), int(input_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"input_size values must be positive, got {input_size!r}")
    return height, width


def box_to_center_scale(
    box_xyxy: Sequence[float] | np.ndarray,
    input_size: int | Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one LibreYOLO ``xyxy`` person box to upstream center/scale."""
    box = np.asarray(box_xyxy, dtype=np.float32).reshape(-1)
    if box.size != 4:
        raise ValueError(f"person box must contain four xyxy values, got {box_xyxy!r}")
    x1, y1, x2, y2 = (float(value) for value in box)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"person box must have positive area, got {box.tolist()}")

    center = np.zeros(2, dtype=np.float32)
    center[0] = x1 + width * 0.5
    center[1] = y1 + height * 0.5

    input_h, input_w = size_hw(input_size)
    aspect_ratio = input_w * 1.0 / input_h
    if width > aspect_ratio * height:
        height = width * 1.0 / aspect_ratio
    elif width < aspect_ratio * height:
        width = height * aspect_ratio

    scale = np.asarray(
        [width / PIXEL_STD, height / PIXEL_STD],
        dtype=np.float32,
    )
    return center, scale * BOX_SCALE_PADDING


def _get_dir(src_point: Sequence[float], rotation_radians: float) -> list[float]:
    sine, cosine = np.sin(rotation_radians), np.cos(rotation_radians)
    return [
        src_point[0] * cosine - src_point[1] * sine,
        src_point[0] * sine + src_point[1] * cosine,
    ]


def _get_third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.asarray([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray | Sequence[float] | float,
    rotation: float,
    output_size_wh: Sequence[int],
    shift: np.ndarray | None = None,
    *,
    inverse: bool = False,
) -> np.ndarray:
    """Return the upstream three-point affine transform."""
    if not isinstance(scale, (np.ndarray, list, tuple)):
        scale = np.asarray([scale, scale], dtype=np.float32)
    scale_array = np.asarray(scale, dtype=np.float32)
    shift_array = (
        np.zeros(2, dtype=np.float32)
        if shift is None
        else np.asarray(shift, dtype=np.float32)
    )
    scale_pixels = scale_array * PIXEL_STD
    source_width = scale_pixels[0]
    destination_width = float(output_size_wh[0])
    destination_height = float(output_size_wh[1])

    rotation_radians = np.pi * rotation / 180
    source_direction = _get_dir([0, source_width * -0.5], rotation_radians)
    destination_direction = np.asarray(
        [0, destination_width * -0.5],
        dtype=np.float32,
    )

    source = np.zeros((3, 2), dtype=np.float32)
    destination = np.zeros((3, 2), dtype=np.float32)
    source[0, :] = center + scale_pixels * shift_array
    source[1, :] = center + source_direction + scale_pixels * shift_array
    destination[0, :] = [destination_width * 0.5, destination_height * 0.5]
    destination[1, :] = destination[0, :] + destination_direction
    source[2, :] = _get_third_point(source[0, :], source[1, :])
    destination[2, :] = _get_third_point(destination[0, :], destination[1, :])

    if inverse:
        return cv2.getAffineTransform(destination, source)
    return cv2.getAffineTransform(source, destination)


def affine_transform(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine transform to one xy coordinate."""
    homogeneous = np.asarray([point[0], point[1], 1.0]).T
    return np.dot(transform, homogeneous)[:2]


def warp_person_crop(
    image_rgb_hwc: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    input_size: int | Sequence[int],
) -> np.ndarray:
    """Warp one person region using the official HRNet affine convention."""
    input_h, input_w = size_hw(input_size)
    transform = get_affine_transform(
        center,
        scale,
        0,
        (input_w, input_h),
    )
    return cv2.warpAffine(
        image_rgb_hwc,
        transform,
        (input_w, input_h),
        flags=cv2.INTER_LINEAR,
    )


def normalize_crop(crop_rgb_hwc: np.ndarray) -> np.ndarray:
    """Apply the official ``ToTensor`` plus ImageNet normalization sequence."""
    if crop_rgb_hwc.ndim != 3 or crop_rgb_hwc.shape[2] != 3:
        raise ValueError(
            "HRNet expects an RGB HWC image with three channels, got "
            f"shape {crop_rgb_hwc.shape}"
        )
    chw = np.ascontiguousarray(crop_rgb_hwc.transpose(2, 0, 1))
    tensor = torch.from_numpy(chw)
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(dtype=torch.float32).div(255)
    else:
        tensor = tensor.to(dtype=torch.float32)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor.sub_(mean).div_(std).numpy()


def preprocess_box_numpy(
    image_rgb_hwc: np.ndarray,
    box_xyxy: Sequence[float] | np.ndarray,
    input_size: int | Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine-warp and normalize one person box from a full RGB image."""
    center, scale = box_to_center_scale(box_xyxy, input_size)
    crop = warp_person_crop(image_rgb_hwc, center, scale, input_size)
    return normalize_crop(crop), center, scale


def preprocess_numpy(
    image_rgb_hwc: np.ndarray,
    input_size: int | Sequence[int],
) -> tuple[np.ndarray, tuple[float, float]]:
    """Preprocess an already-cropped image as one full-frame person box."""
    original_h, original_w = image_rgb_hwc.shape[:2]
    chw, _center, _scale = preprocess_box_numpy(
        image_rgb_hwc,
        (0.0, 0.0, float(original_w), float(original_h)),
        input_size,
    )
    input_h, input_w = size_hw(input_size)
    return chw, (input_w / original_w, input_h / original_h)


def preprocess_crop_image(image, input_size, color_format: str = "auto"):
    """Preprocess one already-cropped person image for the native path."""
    original = ImageLoader.load(image, color_format=color_format)
    original_size = original.size
    chw, ratio = preprocess_numpy(np.asarray(original), input_size)
    return torch.from_numpy(chw).unsqueeze(0), original, original_size, ratio
