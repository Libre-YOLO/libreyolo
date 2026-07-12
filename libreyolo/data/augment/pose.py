"""Shared keypoint-aware pose augmentation helpers.

Moved verbatim from the formerly duplicated helper blocks in
``models/ec/pose_transforms.py`` and ``models/yolonas/pose_transforms.py``.

``random_affine_pose`` takes ``border_value`` as a REQUIRED keyword because
the families deliberately disagree: EC fills affine borders with 114,
YOLO-NAS pose with 127 (``YOLO_NAS_POSE_PAD_VALUE``). There is no correct
shared default.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .constants import IMAGENET_MEAN, IMAGENET_STD

AFFINE_INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def finalize_image(img: np.ndarray, to_rgb: bool, imagenet_norm: bool) -> np.ndarray:
    """HWC uint8 BGR -> CHW float32, optionally RGB + ImageNet-normalized."""
    if to_rgb:
        img = img[:, :, ::-1]
    img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
    img /= 255.0
    if imagenet_norm:
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(img, dtype=np.float32)


def brightness_contrast(img: np.ndarray) -> None:
    """In-place brightness/contrast jitter for uint8 BGR images."""
    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(-0.2, 0.2) * 255.0
    img[:] = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def clip_keypoints_to_image(
    kpts: np.ndarray,
    *,
    width: int | float,
    height: int | float,
) -> np.ndarray:
    """Clip keypoint coordinates and hide points outside the image canvas."""
    if kpts.size == 0:
        return kpts
    width = float(width)
    height = float(height)
    outside = (
        (kpts[..., 0] < 0.0)
        | (kpts[..., 0] > width)
        | (kpts[..., 1] < 0.0)
        | (kpts[..., 1] > height)
    )
    kpts[..., 0] = np.clip(kpts[..., 0], 0.0, width)
    kpts[..., 1] = np.clip(kpts[..., 1], 0.0, height)
    kpts[..., 2] = np.where(outside, 0.0, kpts[..., 2])
    return kpts


def random_affine_pose(
    img: np.ndarray,
    bboxes: np.ndarray,
    kpts: np.ndarray,
    *,
    degrees: float,
    translate: float,
    scale_range: tuple[float, float],
    interpolation: int,
    border_value: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a keypoint-aware affine transform in original image space."""
    h, w = img.shape[:2]
    angle = random.uniform(-degrees, degrees)
    scale = random.uniform(*scale_range)
    tx = random.uniform(-translate, translate) * w
    ty = random.uniform(-translate, translate) * h

    matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
    matrix[:, 2] += (tx, ty)
    warped = cv2.warpAffine(
        img,
        matrix,
        dsize=(w, h),
        flags=interpolation,
        borderValue=(border_value,) * 3,
    )

    if len(bboxes) == 0:
        return warped, bboxes, kpts

    xyxy = np.concatenate(
        [
            bboxes[:, :2] - bboxes[:, 2:] * 0.5,
            bboxes[:, :2] + bboxes[:, 2:] * 0.5,
        ],
        axis=1,
    )
    corners = np.stack(
        [
            xyxy[:, [0, 1]],
            xyxy[:, [2, 1]],
            xyxy[:, [2, 3]],
            xyxy[:, [0, 3]],
        ],
        axis=1,
    )
    ones = np.ones((*corners.shape[:2], 1), dtype=np.float32)
    warped_corners = np.concatenate([corners, ones], axis=2) @ matrix.T
    new_xyxy = np.concatenate(
        [warped_corners.min(axis=1), warped_corners.max(axis=1)], axis=1
    )
    new_xyxy[:, [0, 2]] = new_xyxy[:, [0, 2]].clip(0, w)
    new_xyxy[:, [1, 3]] = new_xyxy[:, [1, 3]].clip(0, h)
    bboxes[:, :2] = (new_xyxy[:, :2] + new_xyxy[:, 2:]) * 0.5
    bboxes[:, 2:] = new_xyxy[:, 2:] - new_xyxy[:, :2]

    points = kpts[..., :2]
    warped_points = (
        np.concatenate([points, np.ones((*points.shape[:2], 1), dtype=np.float32)], axis=2)
        @ matrix.T
    )
    kpts[..., :2] = warped_points
    clip_keypoints_to_image(kpts, width=w, height=h)
    return warped, bboxes, kpts


def build_target(
    cls: np.ndarray,
    bboxes_px: np.ndarray,
    kpts_px: np.ndarray,
    num_keypoints: int,
    max_labels: int,
) -> np.ndarray:
    """Assemble the padded ``(max_labels, 5 + 3K)`` target slab.

    Valid rows are written contiguously from the front — the pose loss relies
    on this front-packing to slice each image's objects.
    """
    target = np.zeros((max_labels, 5 + 3 * num_keypoints), dtype=np.float32)
    if len(bboxes_px) == 0:
        return target

    keep = (
        (bboxes_px[:, 2] * bboxes_px[:, 3] > 1.0)
        & ((kpts_px[..., 2] > 0).sum(axis=1) >= 1)
    )
    bboxes_px, cls, kpts_px = bboxes_px[keep], cls[keep], kpts_px[keep]
    n = min(len(bboxes_px), max_labels)
    if n == 0:
        return target

    target[:n, 0] = cls[:n]
    target[:n, 1:5] = bboxes_px[:n]
    # Reshape the first ``n`` (capped) instances, not ``len(kpts_px)``: when an
    # image has more instances than ``max_labels`` the two differ and the old
    # ``reshape(len(kpts_px), -1)`` either raised a shape error or, when the
    # element counts happened to divide, silently mis-broadcast coordinates.
    target[:n, 5:] = kpts_px[:n].reshape(n, -1)
    return target
