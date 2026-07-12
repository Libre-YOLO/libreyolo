"""EC pose training/validation transforms.

ECPose uses the same preprocessing contract at train, validation, and
inference time: resize directly to the square model input, convert BGR to RGB,
scale to [0, 1], then apply ImageNet normalization. No letterbox padding is
used, so non-square images intentionally stretch to the model input.

The keypoint-aware helper implementations live in
:mod:`libreyolo.data.augment.pose`; this module keeps only the EC-specific
pieces (direct-stretch target scaling and the 114 affine border fill — the
YOLO-NAS pose recipe fills with 127).
"""

from __future__ import annotations

import random  # noqa: F401  (historical module attribute)
from typing import Optional, Sequence

import cv2  # noqa: F401  (historical module attribute)
import numpy as np

from ...data.augment.constants import (  # noqa: F401
    IMAGENET_MEAN as _IMAGENET_MEAN,
    IMAGENET_STD as _IMAGENET_STD,
)
from ...data.augment.pose import (
    AFFINE_INTERPOLATIONS as _AFFINE_INTERPOLATIONS,
    brightness_contrast as _brightness_contrast,
    build_target as _build_target,
    clip_keypoints_to_image,
    finalize_image as _finalize_image,
    random_affine_pose,
)
from ...data.pose_metadata import validate_flip_idx
from ...training.augment import augment_hsv

_AFFINE_BORDER_VALUE = 114


def _as_hw(input_dim) -> tuple[int, int]:
    if isinstance(input_dim, int):
        return int(input_dim), int(input_dim)
    if len(input_dim) != 2:
        raise ValueError(f"input_dim must be int or (h, w), got {input_dim!r}")
    return int(input_dim[0]), int(input_dim[1])


def _random_affine(
    img: np.ndarray,
    bboxes: np.ndarray,
    kpts: np.ndarray,
    *,
    degrees: float,
    translate: float,
    scale_range: tuple[float, float],
    interpolation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EC affine: shared keypoint-aware affine with the EC 114 border fill."""
    return random_affine_pose(
        img,
        bboxes,
        kpts,
        degrees=degrees,
        translate=translate,
        scale_range=scale_range,
        interpolation=interpolation,
        border_value=_AFFINE_BORDER_VALUE,
    )


def _scale_targets_direct(
    bboxes: np.ndarray,
    kpts: np.ndarray,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> None:
    if len(bboxes) == 0:
        return
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)
    bboxes[:, [0, 2]] *= scale_x
    bboxes[:, [1, 3]] *= scale_y
    kpts[..., 0] *= scale_x
    kpts[..., 1] *= scale_y


class ECPoseTrainTransform:
    """Train-time EC pose transform: augmentation plus direct square resize."""

    def __init__(
        self,
        num_keypoints: int,
        flip_idx: Optional[Sequence[int]] = None,
        max_labels: int = 100,
        flip_prob: float = 0.5,
        hsv_prob: float = 0.5,
        brightness_contrast_prob: float = 0.5,
        affine_prob: float = 0.75,
        degrees: float = 5.0,
        translate: float = 0.1,
        scale: tuple[float, float] = (0.75, 1.5),
        affine_interpolation: str = "linear",
        imagenet_norm: bool = True,
        to_rgb: bool = True,
    ):
        self.num_keypoints = num_keypoints
        self.max_labels = max_labels
        self.imagenet_norm = imagenet_norm
        self.to_rgb = to_rgb
        self.hsv_prob = hsv_prob
        self.brightness_contrast_prob = brightness_contrast_prob
        self.affine_prob = affine_prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.affine_interpolation = _AFFINE_INTERPOLATIONS.get(
            affine_interpolation, cv2.INTER_LINEAR
        )
        validated_flip_idx = validate_flip_idx(flip_idx, num_keypoints)
        if validated_flip_idx is not None:
            self.flip_idx = np.asarray(validated_flip_idx, dtype=np.int64)
            self.flip_prob = flip_prob
        else:
            self.flip_idx = None
            self.flip_prob = 0.0

    def __call__(self, img, bboxes_norm, cls, kpts_norm, input_dim):
        h, w = img.shape[:2]
        dst_hw = _as_hw(input_dim)

        bboxes = bboxes_norm.astype(np.float32).reshape(-1, 4)
        bboxes[:, [0, 2]] *= w
        bboxes[:, [1, 3]] *= h
        kpts = kpts_norm.astype(np.float32).reshape(-1, self.num_keypoints, 3)
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        cls = cls.astype(np.float32).reshape(-1)

        if self.hsv_prob > 0 and random.random() < self.hsv_prob:
            augment_hsv(img)
        if (
            self.brightness_contrast_prob > 0
            and random.random() < self.brightness_contrast_prob
        ):
            _brightness_contrast(img)

        if self.flip_idx is not None and random.random() < self.flip_prob:
            img = img[:, ::-1]
            if len(bboxes):
                bboxes[:, 0] = w - bboxes[:, 0]
                kpts[..., 0] = w - kpts[..., 0]
                kpts = kpts[:, self.flip_idx, :]

        if self.affine_prob > 0 and random.random() < self.affine_prob:
            img, bboxes, kpts = _random_affine(
                img,
                bboxes,
                kpts,
                degrees=self.degrees,
                translate=self.translate,
                scale_range=self.scale,
                interpolation=self.affine_interpolation,
            )

        img = cv2.resize(
            np.ascontiguousarray(img),
            (dst_hw[1], dst_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        _scale_targets_direct(bboxes, kpts, src_hw=(h, w), dst_hw=dst_hw)
        clip_keypoints_to_image(kpts, width=dst_hw[1], height=dst_hw[0])

        target = _build_target(cls, bboxes, kpts, self.num_keypoints, self.max_labels)
        img = _finalize_image(np.ascontiguousarray(img), self.to_rgb, self.imagenet_norm)
        return img, target


class ECPoseValTransform:
    """Validation EC pose transform: direct square resize only."""

    def __init__(
        self,
        num_keypoints: int,
        max_labels: int = 100,
        imagenet_norm: bool = True,
        to_rgb: bool = True,
    ):
        self.num_keypoints = num_keypoints
        self.max_labels = max_labels
        self.imagenet_norm = imagenet_norm
        self.to_rgb = to_rgb

    def __call__(self, img, bboxes_norm, cls, kpts_norm, input_dim):
        h, w = img.shape[:2]
        dst_hw = _as_hw(input_dim)

        bboxes = bboxes_norm.astype(np.float32).reshape(-1, 4)
        bboxes[:, [0, 2]] *= w
        bboxes[:, [1, 3]] *= h
        kpts = kpts_norm.astype(np.float32).reshape(-1, self.num_keypoints, 3)
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        cls = cls.astype(np.float32).reshape(-1)

        img = cv2.resize(
            np.ascontiguousarray(img),
            (dst_hw[1], dst_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        _scale_targets_direct(bboxes, kpts, src_hw=(h, w), dst_hw=dst_hw)
        clip_keypoints_to_image(kpts, width=dst_hw[1], height=dst_hw[0])

        target = _build_target(cls, bboxes, kpts, self.num_keypoints, self.max_labels)
        img = _finalize_image(np.ascontiguousarray(img), self.to_rgb, self.imagenet_norm)
        return img, target
