"""Pre-processing for the LibrePPOCR pipeline.

Mirrors PaddleOCR's inference transforms (Apache-2.0):
- det: ``DetResizeForTest`` (limit_type "max", side 960) + ImageNet-constant
  normalization (``tools/infer/predict_det.py``)
- crop: ``get_rotate_crop_image`` perspective warp with the 90-degree rotation
  of tall crops (``tools/infer/utility.py``)
- rec: aspect-preserving resize to height 48, width-padded per batch bucket
  with symmetric [-1, 1] normalization (``tools/infer/predict_rec.py``)

PaddleOCR feeds cv2-decoded BGR images to both networks and applies the
RGB-ordered ImageNet constants to the BGR channels as-is; the converted
weights bake that convention in, so LibreYOLO converts RGB inputs to BGR
before normalization. Keep that in mind before "fixing" the channel order.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np
import torch

# Applied to BGR channels in this order, matching upstream.
_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

REC_IMAGE_SHAPE = (3, 48, 320)  # C, H, W


def det_resize(
    img: np.ndarray,
    limit_side_len: int = 960,
    max_side_limit: int = 4000,
) -> Tuple[np.ndarray, float, float]:
    """Resize so the long side fits ``limit_side_len``, rounded to /32.

    Returns the resized image and ``(ratio_h, ratio_w)``.
    """
    h, w = img.shape[:2]
    if h + w < 64:
        pad = np.zeros((max(32, h), max(32, w), img.shape[2]), dtype=np.uint8)
        pad[:h, :w] = img
        img = pad
        h, w = img.shape[:2]

    if max(h, w) > limit_side_len:
        ratio = limit_side_len / h if h > w else limit_side_len / w
    else:
        ratio = 1.0
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)
    if max(resize_h, resize_w) > max_side_limit:
        ratio = max_side_limit / max(resize_h, resize_w)
        resize_h, resize_w = int(resize_h * ratio), int(resize_w * ratio)

    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)
    resized = cv2.resize(img, (resize_w, resize_h))
    return resized, resize_h / h, resize_w / w


def det_normalize(img_bgr: np.ndarray) -> torch.Tensor:
    """Normalize a BGR uint8 image to the det network's (1, 3, H, W) input."""
    x = img_bgr.astype(np.float32) / 255.0
    x = (x - _DET_MEAN) / _DET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Perspective-warp a text quad to an upright rectangle crop.

    Tall crops (h/w >= 1.5) are rotated 90 degrees so vertical text lines
    enter the recognizer horizontally, matching upstream.
    """
    points = np.asarray(points, dtype=np.float32)
    assert points.shape == (4, 2), "quad must be 4x2"
    crop_w = int(
        max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
    )
    crop_h = int(
        max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))
    )
    pts_std = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])
    matrix = cv2.getPerspectiveTransform(points, pts_std)
    dst = cv2.warpPerspective(
        img,
        matrix,
        (crop_w, crop_h),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_h, dst_w = dst.shape[:2]
    if dst_h * 1.0 / max(dst_w, 1) >= 1.5:
        dst = np.rot90(dst)
    return dst


def rec_resize_norm(img_bgr: np.ndarray, max_wh_ratio: float) -> np.ndarray:
    """Resize a BGR crop to height 48 preserving aspect, pad to the bucket width.

    Returns a ``(3, 48, W)`` float32 array normalized to ``[-1, 1]``.
    """
    img_c, img_h, _ = REC_IMAGE_SHAPE
    img_w = int(img_h * max_wh_ratio)
    h, w = img_bgr.shape[:2]
    ratio = w / float(h)
    resized_w = img_w if math.ceil(img_h * ratio) > img_w else int(math.ceil(img_h * ratio))
    resized = cv2.resize(img_bgr, (resized_w, img_h)).astype(np.float32)
    resized = resized.transpose(2, 0, 1) / 255.0
    resized -= 0.5
    resized /= 0.5
    padded = np.zeros((img_c, img_h, img_w), dtype=np.float32)
    padded[:, :, :resized_w] = resized
    return padded


def rec_batches(
    crops: List[np.ndarray],
    batch_size: int = 6,
) -> List[Tuple[List[int], np.ndarray]]:
    """Bucket crops by aspect ratio into padded batches, upstream-style.

    Crops are sorted by width/height ratio; each chunk shares one padded
    width driven by the widest member (never narrower than 320/48). Returns
    ``(original_indices, batch_tensor)`` pairs.
    """
    img_c, img_h, img_w = REC_IMAGE_SHAPE
    ratios = [c.shape[1] / float(c.shape[0]) for c in crops]
    indices = np.argsort(np.array(ratios))

    batches: List[Tuple[List[int], np.ndarray]] = []
    for beg in range(0, len(crops), batch_size):
        chunk = [int(i) for i in indices[beg : beg + batch_size]]
        max_wh_ratio = img_w / img_h
        for i in chunk:
            max_wh_ratio = max(max_wh_ratio, ratios[i])
        norm = [rec_resize_norm(crops[i], max_wh_ratio) for i in chunk]
        batches.append((chunk, np.stack(norm)))
    return batches


__all__ = [
    "det_resize",
    "det_normalize",
    "get_rotate_crop_image",
    "rec_resize_norm",
    "rec_batches",
    "REC_IMAGE_SHAPE",
]
