"""Ring-list segment-polygon helpers for the numpy pipelines.

Moved verbatim from ``libreyolo/models/yolo9/transforms.py``. Instances are
lists of polygon rings (``(N, 2)`` float arrays), the format produced by the
YOLO datasets.

NOTE: RF-DETR's segment helpers in :mod:`libreyolo.data.augment.rfdetr` use a
different data model (dense-mask-backed rings for crop paths) and are
deliberately NOT merged with these — see the augmentations refactor plan.
"""

import cv2
import numpy as np


def copy_segments(segments):
    if segments is None:
        return None
    return [[ring.copy() for ring in instance] for instance in segments]


def transform_segments(segments, scale=1.0, padw=0.0, padh=0.0, width=None, height=None):
    if segments is None:
        return None
    transformed = []
    for instance in segments:
        rings = []
        for ring in instance:
            r = ring.astype(np.float32, copy=True)
            r[:, 0] = r[:, 0] * scale + padw
            r[:, 1] = r[:, 1] * scale + padh
            if width is not None:
                r[:, 0] = np.clip(r[:, 0], 0, width)
            if height is not None:
                r[:, 1] = np.clip(r[:, 1], 0, height)
            rings.append(r)
        transformed.append(rings)
    return transformed


def flip_segments_lr(segments, width):
    if segments is None:
        return None
    flipped = []
    for instance in segments:
        rings = []
        for ring in instance:
            r = ring.astype(np.float32, copy=True)
            r[:, 0] = width - r[:, 0]
            rings.append(r)
        flipped.append(rings)
    return flipped


def flip_segments_ud(segments, height):
    if segments is None:
        return None
    flipped = []
    for instance in segments:
        rings = []
        for ring in instance:
            r = ring.astype(np.float32, copy=True)
            r[:, 1] = height - r[:, 1]
            rings.append(r)
        flipped.append(rings)
    return flipped


def filter_segments(segments, keep_mask):
    if segments is None:
        return None
    keep = np.asarray(keep_mask, dtype=bool)
    return [segments[i] for i in range(min(len(segments), len(keep))) if keep[i]]


def rasterize_segments(segments, image_shape, mask_shape, max_masks):
    masks = np.zeros((max_masks, mask_shape[0], mask_shape[1]), dtype=np.float32)
    if not segments:
        return masks

    img_h, img_w = image_shape
    mask_h, mask_w = mask_shape
    sx = mask_w / max(float(img_w), 1.0)
    sy = mask_h / max(float(img_h), 1.0)

    for idx, instance in enumerate(segments[:max_masks]):
        polygons = []
        for ring in instance:
            if ring is None or len(ring) < 3:
                continue
            poly = ring.astype(np.float32, copy=True)
            poly[:, 0] *= sx
            poly[:, 1] *= sy
            polygons.append(np.round(poly).astype(np.int32))
        if polygons:
            cv2.fillPoly(masks[idx], polygons, color=1)
    return masks
