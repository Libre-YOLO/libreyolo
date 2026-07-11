"""Postprocessing for the LibrePPOCR family: DB map-to-quads and CTC decode.

The detection stage follows PaddleOCR's ``DBPostProcess`` in its default
``quad`` mode (Apache-2.0, ``ppocr/postprocess/db_postprocess.py``): threshold
the shrink-probability map, find contours, take each contour's minimum-area
rectangle, score it by the mean probability inside, and expand ("unclip") it
by ``area * unclip_ratio / perimeter``.

Upstream unclips with pyclipper's round-joined polygon offset and then takes
the minimum-area rectangle of the result. For a rectangle, the round-joined
offset by distance ``d`` is the rectangle grown by ``d`` on every side plus
quarter-circle corners, whose minimum-area rectangle is exactly the original
rectangle with ``w + 2d`` by ``h + 2d`` at the same center and angle. That
identity lets the unclip be computed analytically, so LibreYOLO does not need
the pyclipper/shapely dependencies (differences vs upstream are below the
integer rounding pyclipper itself applies).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Order the 4 corners of a min-area rect as tl, tr, br, bl.

    Mirrors upstream ``get_mini_boxes``: sort by x, then resolve each side's
    top/bottom by y.
    """
    pts = sorted(points.tolist(), key=lambda p: p[0])
    if pts[1][1] > pts[0][1]:
        i1, i4 = 0, 1
    else:
        i1, i4 = 1, 0
    if pts[3][1] > pts[2][1]:
        i2, i3 = 2, 3
    else:
        i2, i3 = 3, 2
    return np.array([pts[i1], pts[i2], pts[i3], pts[i4]], dtype=np.float32)


def _box_score_fast(prob_map: np.ndarray, quad: np.ndarray) -> float:
    """Mean probability inside the quad (fast mode: bbox-cropped fillPoly)."""
    h, w = prob_map.shape[:2]
    box = quad.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(prob_map[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def _quad_area_perimeter(quad: np.ndarray) -> Tuple[float, float]:
    x = quad[:, 0]
    y = quad[:, 1]
    area = 0.5 * abs(
        float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    )
    perimeter = float(np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1).sum())
    return area, perimeter


def db_postprocess(
    prob_map: np.ndarray,
    dest_width: int,
    dest_height: int,
    *,
    thresh: float = 0.3,
    box_thresh: float = 0.6,
    max_candidates: int = 1000,
    unclip_ratio: float = 1.5,
    min_size: int = 3,
) -> Tuple[np.ndarray, List[float]]:
    """Convert a DB shrink-probability map to text quads on the source canvas.

    Args:
        prob_map: ``(H, W)`` float probability map (network resolution).
        dest_width: Source-image width to scale the quads back to.
        dest_height: Source-image height to scale the quads back to.
        thresh: Binarization threshold on the probability map.
        box_thresh: Minimum mean probability for a kept region.
        max_candidates: Maximum number of contours considered.
        unclip_ratio: DB box expansion ratio.
        min_size: Minimum short side (in map pixels) of a kept quad.

    Returns:
        ``(boxes, scores)`` where boxes is ``(N, 4, 2)`` float32 in source-image
        pixels ordered tl, tr, br, bl, and scores is the per-box mean
        probability.
    """
    height, width = prob_map.shape[:2]
    bitmap = (prob_map > thresh).astype(np.uint8)

    contours, _ = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[np.ndarray] = []
    scores: List[float] = []
    for contour in contours[:max_candidates]:
        rect = cv2.minAreaRect(contour)
        sside = min(rect[1])
        if sside < min_size:
            continue
        quad = _order_quad(cv2.boxPoints(rect))
        score = _box_score_fast(prob_map, quad)
        if score < box_thresh:
            continue

        area, perimeter = _quad_area_perimeter(quad)
        if perimeter < 1e-3:
            continue
        distance = area * unclip_ratio / perimeter
        (cx, cy), (rw, rh), angle = rect
        expanded = ((cx, cy), (rw + 2.0 * distance, rh + 2.0 * distance), angle)
        if min(expanded[1]) < min_size + 2:
            continue
        box = _order_quad(cv2.boxPoints(expanded))

        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box)
        scores.append(score)

    if not boxes:
        return np.zeros((0, 4, 2), dtype=np.float32), []
    return np.stack(boxes).astype(np.float32), scores


def sort_quads_reading_order(quads: np.ndarray) -> np.ndarray:
    """Indices sorting quads top-to-bottom, then left-to-right in a ~10 px band.

    Mirrors upstream ``sorted_boxes`` (``tools/infer/predict_system.py``),
    keyed on each quad's top-left corner. Returns the permutation so callers
    can reorder parallel arrays (scores) alongside the quads.
    """
    n = quads.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = sorted(range(n), key=lambda i: (quads[i][0][1], quads[i][0][0]))
    for i in range(n - 1):
        for j in range(i, -1, -1):
            if abs(quads[order[j + 1]][0][1] - quads[order[j]][0][1]) < 10 and (
                quads[order[j + 1]][0][0] < quads[order[j]][0][0]
            ):
                order[j], order[j + 1] = order[j + 1], order[j]
            else:
                break
    return np.asarray(order, dtype=np.int64)


def ctc_decode(
    probs: np.ndarray,
    charset: Sequence[str],
) -> List[Tuple[str, float]]:
    """Greedy CTC decode of ``(B, T, C)`` per-timestep char probabilities.

    ``charset`` is the full CTC alphabet including the blank at index 0
    (blank + dictionary + space for PP-OCRv5). Repeats are collapsed before
    blank removal; per-text confidence is the mean probability of the kept
    timesteps (0.0 when nothing is kept).
    """
    preds_idx = probs.argmax(axis=2)
    preds_prob = probs.max(axis=2)

    results: List[Tuple[str, float]] = []
    for indices, confidences in zip(preds_idx, preds_prob):
        selection = np.ones(len(indices), dtype=bool)
        selection[1:] = indices[1:] != indices[:-1]
        selection &= indices != 0  # CTC blank
        chars = [charset[i] for i in indices[selection]]
        kept = confidences[selection]
        conf = float(kept.mean()) if kept.size else 0.0
        results.append(("".join(chars), conf))
    return results


__all__ = ["db_postprocess", "sort_quads_reading_order", "ctc_decode"]
