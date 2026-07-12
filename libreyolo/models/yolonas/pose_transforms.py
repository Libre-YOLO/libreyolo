"""YOLO-NAS pose training/validation transforms.

Keypoint-aware preprocessing for the YOLO-format pose pipeline. Both transforms
take a raw BGR image plus normalized labels and return:

- ``image``: ``(3, H, W)`` float32 BGR in ``[0, 1]``
- ``target``: ``(max_labels, 5 + 3K)`` float32 — rows are
  ``[cls, cx, cy, w, h, kx1, ky1, v1, ...]`` in letterboxed pixel coordinates.

Augmentation follows the public SuperGradients YOLO-NAS pose recipe where it
is practical for YOLO-format labels: keypoint-aware hflip, brightness/contrast,
HSV jitter, random affine, resize, and padding. Training pads in the center;
validation pads bottom/right.

The keypoint-aware helper implementations live in
:mod:`libreyolo.data.augment.pose`; this module keeps only the YOLO-NAS
specifics (capped letterbox with the 127 pad/border fill — the EC pose recipe
fills with 114 and stretches without letterboxing).
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
from .utils import YOLO_NAS_POSE_PAD_VALUE, YOLO_NAS_POSE_RESIZE_SIZE


def _letterbox(
    img: np.ndarray,
    input_dim,
    *,
    padding_mode: str = "center",
) -> tuple[np.ndarray, float, int, int]:
    """Resize-and-center-pad into ``input_dim``; return image, ratio, x/y pad."""
    ih, iw = input_dim
    h, w = img.shape[:2]
    resize_size = min(YOLO_NAS_POSE_RESIZE_SIZE, ih, iw)
    r = min(resize_size / h, resize_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((ih, iw, 3), YOLO_NAS_POSE_PAD_VALUE, dtype=np.uint8)
    if padding_mode == "bottom_right":
        pad_x = 0
        pad_y = 0
    elif padding_mode == "center":
        pad_x = (iw - nw) // 2
        pad_y = (ih - nh) // 2
    else:
        raise ValueError(f"Unsupported padding_mode={padding_mode!r}")
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, r, pad_x, pad_y


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
    """YOLO-NAS affine: shared keypoint-aware affine with the 127 border fill."""
    return random_affine_pose(
        img,
        bboxes,
        kpts,
        degrees=degrees,
        translate=translate,
        scale_range=scale_range,
        interpolation=interpolation,
        border_value=YOLO_NAS_POSE_PAD_VALUE,
    )


def _apply_letterbox_to_targets(
    bboxes: np.ndarray, kpts: np.ndarray, ratio: float, pad_x: int, pad_y: int
):
    """Transform cxcywh boxes and xy keypoints into letterboxed pixel space."""
    if len(bboxes) == 0:
        return
    bboxes *= ratio
    bboxes[:, 0] += pad_x
    bboxes[:, 1] += pad_y
    kpts[..., :2] *= ratio
    kpts[..., 0] += pad_x
    kpts[..., 1] += pad_y


class YOLONASPoseTrainTransform:
    """Train-time pose transform: HSV jitter + keypoint-aware hflip + letterbox."""

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
        imagenet_norm: bool = False,
        to_rgb: bool = False,
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
        # A horizontal flip needs the left/right keypoint permutation; without
        # one, flipping would corrupt keypoint identities.
        validated_flip_idx = validate_flip_idx(flip_idx, num_keypoints)
        if validated_flip_idx is not None:
            self.flip_idx = np.asarray(validated_flip_idx, dtype=np.int64)
            self.flip_prob = flip_prob
        else:
            self.flip_idx = None
            self.flip_prob = 0.0

    def __call__(self, img, bboxes_norm, cls, kpts_norm, input_dim):
        h, w = img.shape[:2]

        # Normalized -> original-image pixels.
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

        img, r, pad_x, pad_y = _letterbox(
            np.ascontiguousarray(img), input_dim, padding_mode="center"
        )
        _apply_letterbox_to_targets(bboxes, kpts, r, pad_x, pad_y)
        clip_keypoints_to_image(kpts, width=input_dim[1], height=input_dim[0])

        target = _build_target(
            cls, bboxes, kpts, self.num_keypoints, self.max_labels
        )
        img = _finalize_image(np.ascontiguousarray(img), self.to_rgb, self.imagenet_norm)
        return img, target


class YOLONASPoseValTransform:
    """Validation pose transform: letterbox only, no augmentation."""

    def __init__(
        self,
        num_keypoints: int,
        max_labels: int = 100,
        imagenet_norm: bool = False,
        to_rgb: bool = False,
    ):
        self.num_keypoints = num_keypoints
        self.max_labels = max_labels
        self.imagenet_norm = imagenet_norm
        self.to_rgb = to_rgb

    def __call__(self, img, bboxes_norm, cls, kpts_norm, input_dim):
        h, w = img.shape[:2]
        bboxes = bboxes_norm.astype(np.float32).reshape(-1, 4)
        bboxes[:, [0, 2]] *= w
        bboxes[:, [1, 3]] *= h
        kpts = kpts_norm.astype(np.float32).reshape(-1, self.num_keypoints, 3)
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        cls = cls.astype(np.float32).reshape(-1)

        img, r, pad_x, pad_y = _letterbox(
            np.ascontiguousarray(img), input_dim, padding_mode="bottom_right"
        )
        _apply_letterbox_to_targets(bboxes, kpts, r, pad_x, pad_y)
        clip_keypoints_to_image(kpts, width=input_dim[1], height=input_dim[0])

        target = _build_target(
            cls, bboxes, kpts, self.num_keypoints, self.max_labels
        )
        img = _finalize_image(np.ascontiguousarray(img), self.to_rgb, self.imagenet_norm)
        return img, target
