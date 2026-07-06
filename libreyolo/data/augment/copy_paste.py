"""Copy-paste instance augmentation (numpy/cv2, torch-free).

Implements the copy-paste method described by Ghiasi et al. 2021, "Simple
Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation"
(arXiv:2012.07177): take a random subset of instances from a source image,
optionally horizontally flip and rescale the source, then hard-paste the source
pixels lying under each chosen instance mask onto the destination image with no
blending (the paper found simple pasting works as well as blended pasting).
Pasted instances are appended to the destination annotations; pre-existing
destination instances whose mask is mostly covered by the paste are dropped.

Instances follow this package's ring-list convention: a per-instance list of
``(N, 2)`` float polygon rings, matching :mod:`libreyolo.data.augment.segments`.
The module is deliberately numpy/cv2 only so it stays importable without torch.
"""

import random

import cv2
import numpy as np


def _draw_uniform01(rng):
    """One draw from ``[0, 1)`` that works for the stdlib ``random`` module,
    a ``random.Random`` instance, or a numpy ``Generator`` (all expose
    ``.random()``)."""
    return float(rng.random())


def _draw_uniform(rng, low, high):
    """One draw from ``[low, high]``; ``.uniform`` is common to the stdlib
    ``random`` module/instance and numpy ``Generator``."""
    return float(rng.uniform(low, high))


def _flip_rings_lr(rings, width):
    """Mirror each ring's x about ``width`` (x -> width - x); pure geometry."""
    out = []
    for ring in rings:
        r = np.asarray(ring, dtype=np.float32).copy()
        if len(r):
            r[:, 0] = width - r[:, 0]
        out.append(r)
    return out


def _scale_rings(rings, fx, fy):
    """Scale each ring's coordinates by ``(fx, fy)`` about the origin."""
    out = []
    for ring in rings:
        r = np.asarray(ring, dtype=np.float32).copy()
        if len(r):
            r[:, 0] *= fx
            r[:, 1] *= fy
        out.append(r)
    return out


def _clip_rings(rings, width, height):
    """Clamp ring coordinates into the ``[0, width] x [0, height]`` canvas."""
    out = []
    for ring in rings:
        r = np.asarray(ring, dtype=np.float32).copy()
        if len(r):
            r[:, 0] = np.clip(r[:, 0], 0.0, float(width))
            r[:, 1] = np.clip(r[:, 1], 0.0, float(height))
        out.append(r)
    return out


def _fill_instance_mask(rings, height, width):
    """Rasterize an instance's rings into a ``(height, width)`` bool mask.

    ``cv2.fillPoly`` clips polygons to the canvas, so rings that spill past the
    edges are cropped rather than raising.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = []
    for ring in rings:
        r = np.asarray(ring, dtype=np.float32)
        if r is None or len(r) < 3:
            continue
        polygons.append(np.round(r).astype(np.int32))
    if polygons:
        cv2.fillPoly(mask, polygons, color=1)
    return mask.astype(bool)


def copy_paste(
    img,
    boxes,
    labels,
    segments,
    src_img,
    src_boxes,
    src_labels,
    src_segments,
    *,
    max_instances=None,
    scale_range=(0.5, 1.5),
    flip_prob=0.5,
    occluded_ioa_drop=0.8,
    rng=None,
):
    """Paste a random subset of source instances onto the destination image.

    Args:
        img: Destination image ``(H, W, C)`` (or ``(H, W)``); not mutated.
        boxes: Destination boxes ``(N, 4)`` xyxy in ``img`` pixel coords.
        labels: Destination class ids ``(N,)``.
        segments: Destination instances (list of ring lists) in ``img`` coords.
        src_img: Source image the pasted pixels are taken from.
        src_boxes: Source boxes ``(M, 4)`` xyxy in ``src_img`` pixel coords.
        src_labels: Source class ids ``(M,)``.
        src_segments: Source instances (list of ring lists) in ``src_img`` coords.
        max_instances: Cap on how many source instances may be pasted.
        scale_range: Uniform range for the source scale-jitter factor.
        flip_prob: Probability the source is horizontally flipped before pasting.
        occluded_ioa_drop: Destination instances whose mask has intersection-
            over-own-area with the union paste mask strictly above this value
            are dropped (they are considered occluded).
        rng: ``random`` module, a ``random.Random``, or a numpy ``Generator``.
            Defaults to the stdlib ``random`` module for determinism under the
            library's global seeding.

    Returns:
        ``(img, boxes, labels, segments)`` with pasted instances appended and
        occluded destination instances removed. Box/label/segment counts stay
        aligned.
    """
    if rng is None:
        rng = random

    height, width = img.shape[:2]
    out_img = img.copy()

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    labels = np.asarray(labels).reshape(-1).copy()
    segments = list(segments) if segments else []

    src_segments = list(src_segments) if src_segments else []
    n_src = len(src_segments)
    if n_src == 0 or src_img is None:
        # Nothing to paste from: identity (image already copied above).
        return out_img, boxes, labels, segments

    src = np.asarray(src_img)
    src_h, src_w = src.shape[:2]
    src_boxes = np.asarray(src_boxes, dtype=np.float32).reshape(-1, 4).copy()
    src_labels = np.asarray(src_labels).reshape(-1)
    # Work on private ring copies so the caller's source arrays stay intact.
    src_rings = [[np.asarray(r, dtype=np.float32).copy() for r in inst] for inst in src_segments]

    # Optionally mirror the whole source (image + boxes + polygons).
    if _draw_uniform01(rng) < flip_prob:
        src = np.ascontiguousarray(src[:, ::-1])
        if len(src_boxes):
            src_boxes[:, [0, 2]] = src_w - src_boxes[:, [2, 0]]
        src_rings = [_flip_rings_lr(inst, src_w) for inst in src_rings]

    # Scale-jitter, then map source coordinates into the destination frame. The
    # source image is resized to (W*s, H*s) and dropped at the destination's
    # top-left corner (cropped if larger, leaving a border if smaller), so a
    # single factor per axis carries both the source->destination resize and the
    # jitter.
    scale = _draw_uniform(rng, scale_range[0], scale_range[1])
    fx = (width / max(src_w, 1)) * scale
    fy = (height / max(src_h, 1)) * scale
    new_w = max(1, int(round(src_w * fx)))
    new_h = max(1, int(round(src_h * fy)))
    src_resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if out_img.ndim == 3:
        src_canvas = np.zeros((height, width, out_img.shape[2]), dtype=out_img.dtype)
    else:
        src_canvas = np.zeros((height, width), dtype=out_img.dtype)
    crop_h = min(new_h, height)
    crop_w = min(new_w, width)
    src_canvas[:crop_h, :crop_w] = src_resized[:crop_h, :crop_w]

    src_rings = [_scale_rings(inst, fx, fy) for inst in src_rings]

    # Random subset: keep each source instance with probability 0.5, but never
    # an empty subset (fall back to the lowest draw) so a paste always happens.
    draws = [_draw_uniform01(rng) for _ in range(n_src)]
    selected = [i for i, d in enumerate(draws) if d < 0.5]
    if not selected:
        selected = [int(np.argmin(draws))]
    if max_instances is not None:
        selected = selected[: max(0, int(max_instances))]

    union_mask = np.zeros((height, width), dtype=bool)
    pasted_boxes = []
    pasted_labels = []
    pasted_segments = []
    for i in selected:
        inst_rings = src_rings[i]
        instance_mask = _fill_instance_mask(inst_rings, height, width)
        if not instance_mask.any():
            # Instance fell entirely outside the destination canvas.
            continue
        union_mask |= instance_mask
        clipped = _clip_rings(inst_rings, width, height)
        pts = np.concatenate([r for r in clipped if len(r)], axis=0)
        pasted_boxes.append(
            [
                float(pts[:, 0].min()),
                float(pts[:, 1].min()),
                float(pts[:, 0].max()),
                float(pts[:, 1].max()),
            ]
        )
        pasted_labels.append(src_labels[i] if i < len(src_labels) else 0)
        pasted_segments.append(clipped)

    if not union_mask.any():
        return out_img, boxes, labels, segments

    # Hard paste (no blending): overwrite destination pixels under the union of
    # the pasted instance masks with the source pixels.
    out_img[union_mask] = src_canvas[union_mask]

    # Drop destination instances mostly hidden by the paste. Intersection-over-
    # area uses each instance's own mask area as the denominator.
    keep = np.ones(len(segments), dtype=bool)
    for j, inst in enumerate(segments):
        inst_mask = _fill_instance_mask(inst, height, width)
        area = int(inst_mask.sum())
        if area == 0:
            continue
        inter = int(np.logical_and(inst_mask, union_mask).sum())
        if inter / area > occluded_ioa_drop:
            keep[j] = False

    kept_segments = [segments[j] for j in range(len(segments)) if keep[j]]
    if len(boxes) == len(keep):
        boxes = boxes[keep]
    if len(labels) == len(keep):
        labels = labels[keep]

    if pasted_boxes:
        boxes = np.concatenate(
            [boxes.reshape(-1, 4), np.asarray(pasted_boxes, dtype=np.float32)], axis=0
        )
        label_dtype = labels.dtype if labels.size else np.float32
        labels = np.concatenate(
            [labels.reshape(-1), np.asarray(pasted_labels, dtype=label_dtype)]
        )
        kept_segments = kept_segments + pasted_segments

    return out_img, boxes, labels, kept_segments
