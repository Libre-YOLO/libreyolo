"""Tolerant parsing of VLM detection output into the LibreYOLO detection dict.

Vision-language detectors emit a JSON array of ``{"label", "bbox"}`` objects as
*generated text*, not a tensor. That text can arrive wrapped in markdown fences,
prefixed with prose, or truncated mid-array when generation hits the token
budget. These helpers turn that noisy text into the plain detection dict that
``InferenceRunner._wrap_results`` already knows how to turn into ``Results``.

Everything here is pure (no torch, no model) so it can be unit-tested offline.

The coordinate contract follows the documented LFM2-VL schema: ``bbox`` is
``[x1, y1, x2, y2]`` normalized to ``[0, 1]`` relative to the original image.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from numbers import Integral
from typing import Dict, List, Optional, Tuple

__all__ = [
    "extract_detections",
    "normalize_bbox",
    "to_xyxy",
    "resolve_label",
    "finalize_detection_dict",
    "build_detection_dict",
]

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```$")
# A flat ``{...}`` object with no nested braces; detection items are flat.
_OBJECT = re.compile(r"\{[^{}]*\}")


def _iter_balanced_arrays(text: str):
    """Yield every balanced top-level ``[...]`` substring, left to right.

    A model can emit a bracketed array in prose (e.g. "normalized to [0,1]")
    before the real detection array, so the caller must try each one rather than
    committing to the first.
    """
    i = 0
    n = len(text)
    while i < n:
        start = text.find("[", i)
        if start == -1:
            return
        depth = 0
        end = -1
        for j in range(start, n):
            ch = text[j]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            return  # unterminated; nothing more to yield
        yield text[start : end + 1]
        i = end + 1


def _is_detection_dict(d) -> bool:
    """A dict that looks like a detection item (carries a label or a box key).

    Keyed on the union of markers used by the families, since the parser has no
    access to a family's ``BBOX_KEY``. Used to prefer the real detection array
    over a dict-bearing preamble.
    """
    return isinstance(d, dict) and ("label" in d or "bbox" in d or "bbox_2d" in d)


def _loads_object(blob: str) -> Optional[dict]:
    for candidate in (blob, blob.replace("'", '"')):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_detections(text: str) -> List[dict]:
    """Extract ``{"label", "bbox"}`` dicts from possibly-noisy model text.

    Defensive against markdown fences, surrounding prose, single quotes, and a
    truncated (unterminated) array. Returns ``[]`` rather than raising on any
    unparseable input, so an empty or chatty "no objects found" reply maps to
    zero detections.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    cleaned = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text.strip())).strip()

    # Collect DETECTION-shaped dicts (carrying a label/box key) from EVERY
    # top-level array, so a dict-bearing preamble (a restated schema example, a
    # reasoning/metadata array) cannot shadow the real detections. A bracketed
    # array in prose (e.g. "[0,1]") or a bare ``bbox`` list contributes nothing;
    # an echoed example whose label is out of vocabulary is dropped downstream.
    # An any-dict array is remembered only as a last resort for an unforeseen key.
    collected = []
    fallback = None
    for array in _iter_balanced_arrays(cleaned):
        try:
            data = json.loads(array)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, list):
            continue
        collected.extend(d for d in data if _is_detection_dict(d))
        if fallback is None:
            dicts = [d for d in data if isinstance(d, dict)]
            if dicts:
                fallback = dicts
    # Recover flat objects the bracket scan missed and merge them in (deduped):
    # a real array truncated mid-content behind a complete (schema-echo) array is
    # never yielded by the scan, and a bare object has no enclosing array at all.
    # _OBJECT matches brace-flat objects, which detection items are. Any-shape
    # recoveries are kept only as a last resort.
    seen = {repr(d) for d in collected}
    recovered = []
    for blob in _OBJECT.findall(cleaned):
        obj = _loads_object(blob)
        if obj is None:
            continue
        recovered.append(obj)
        if _is_detection_dict(obj) and repr(obj) not in seen:
            collected.append(obj)
            seen.add(repr(obj))

    if collected:
        return collected
    if fallback is not None:
        return fallback
    return recovered


def normalize_bbox(bbox) -> Optional[Tuple[float, float, float, float]]:
    """Validate/clean a normalized ``[x1, y1, x2, y2]`` box.

    Returns a 4-tuple clamped to ``[0, 1]`` with corners ordered, or None if the
    value is not four finite numbers. Coordinates are assumed already normalized
    to ``[0, 1]`` per the detection prompt contract.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
        return None
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in vals)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def resolve_label(label, name_to_id: Dict[str, int]) -> Optional[int]:
    """Map a free-text label to a class id (case-insensitive exact match).

    Returns None for labels outside the vocabulary, which the caller drops.
    This is what makes an open-vocabulary generator behave like a closed-set
    detector against a fixed ``names`` mapping.
    """
    if not isinstance(label, str):
        return None
    return name_to_id.get(label.strip().lower())


def to_xyxy(box, box_format: str = "xyxy"):
    """Convert a 4-value box in the given layout to ``[x1, y1, x2, y2]``.

    Supported layouts: ``xyxy`` (corners, the default), ``xywh`` (top-left plus
    width/height), and ``cxcywh`` (center plus width/height). Returns None if the
    value is not four finite numbers or the layout is unknown.
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        a, b, c, d = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if box_format == "xyxy":
        return [a, b, c, d]
    if box_format == "xywh":
        return [a, b, a + c, b + d]
    if box_format == "cxcywh":
        return [a - c / 2.0, b - d / 2.0, a + c / 2.0, b + d / 2.0]
    return None


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _empty_detection_dict() -> dict:
    return {
        "boxes": [],
        "scores": [],
        "classes": [],
        "num_detections": 0,
    }


def _as_rows(value) -> list:
    """Return a concrete sequence for list/array/tensor-like output."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return []
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def finalize_detection_dict(
    boxes,
    scores,
    class_ids,
    original_size: Tuple[int, int],
    *,
    conf_thres: float = 0.0,
    iou_thres: Optional[float] = None,
    max_det: int = 300,
    classes: Optional[List[int]] = None,
    num_classes: Optional[int] = None,
) -> dict:
    """Validate pixel-space detections and apply class-aware NMS.

    This is the common finalization contract for generative VLM adapters and
    discriminative open-vocabulary adapters. Malformed, non-finite, fractional-
    class, and zero-area rows are dropped; boxes are clipped to the image;
    output is stably score-sorted and capped only after class-aware NMS.
    """
    if isinstance(max_det, bool) or not isinstance(max_det, Integral):
        raise ValueError("max_det must be an integer.")
    if max_det <= 0:
        return _empty_detection_dict()
    try:
        width, height = (float(v) for v in original_size)
        conf_value = float(conf_thres)
    except (TypeError, ValueError):
        return _empty_detection_dict()
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        return _empty_detection_dict()
    if not math.isfinite(conf_value):
        raise ValueError("conf_thres must be finite.")

    if iou_thres is not None:
        try:
            iou_value = float(iou_thres)
        except (TypeError, ValueError) as exc:
            raise ValueError("iou_thres must be a finite number between 0 and 1.") from exc
        if not math.isfinite(iou_value) or not 0.0 <= iou_value <= 1.0:
            raise ValueError("iou_thres must be a finite number between 0 and 1.")
    else:
        iou_value = None

    allowed_classes = None
    if classes is not None:
        try:
            allowed_values = [float(value) for value in classes]
        except (TypeError, ValueError) as exc:
            raise ValueError("classes must contain integer class ids.") from exc
        if any(not math.isfinite(value) or not value.is_integer() for value in allowed_values):
            raise ValueError("classes must contain integer class ids.")
        allowed_classes = {int(value) for value in allowed_values}

    candidates = []
    for index, (box, score, class_id) in enumerate(
        zip(_as_rows(boxes), _as_rows(scores), _as_rows(class_ids))
    ):
        if isinstance(box, (str, bytes)):
            continue
        try:
            box_values = list(box)
        except (TypeError, ValueError):
            continue
        if len(box_values) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in box_values)
            score_value = float(score)
            class_value = float(class_id)
        except (TypeError, ValueError):
            continue
        if not all(
            math.isfinite(value)
            for value in (x1, y1, x2, y2, score_value, class_value)
        ):
            continue
        if not class_value.is_integer():
            continue
        class_int = int(class_value)
        if num_classes is not None and not 0 <= class_int < num_classes:
            continue
        if allowed_classes is not None and class_int not in allowed_classes:
            continue
        if score_value < conf_value:
            continue

        x1, x2 = min(width, max(0.0, x1)), min(width, max(0.0, x2))
        y1, y2 = min(height, max(0.0, y1)), min(height, max(0.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        normalized = (x1, y1, x2, y2)
        candidates.append((index, normalized, score_value, class_int))

    if not candidates:
        return _empty_detection_dict()

    # Stable ordering makes equal synthetic VLM scores deterministic.
    candidates.sort(key=lambda row: (-row[2], row[0]))
    kept = []
    seen = set()
    for candidate in candidates:
        _, box, _, class_id = candidate
        exact_key = (class_id, *box)
        if exact_key in seen:
            continue
        if iou_value is not None and any(
            class_id == kept_row[3] and _iou_xyxy(box, kept_row[1]) > iou_value
            for kept_row in kept
        ):
            continue
        seen.add(exact_key)
        kept.append(candidate)
        if len(kept) >= max_det:
            break

    return {
        "boxes": [list(row[1]) for row in kept],
        "scores": [row[2] for row in kept],
        "classes": [row[3] for row in kept],
        "num_detections": len(kept),
    }


def build_detection_dict(
    items: List[dict],
    name_to_id: Dict[str, int],
    original_size: Tuple[int, int],
    conf_thres: float = 0.0,
    max_det: int = 300,
    classes: Optional[List[int]] = None,
    default_score: float = 1.0,
    bbox_key: str = "bbox",
    coord_divisor: float = 1.0,
    box_format: str = "xyxy",
    iou_thres: Optional[float] = None,
) -> dict:
    """Turn parsed items into the ``InferenceRunner`` detection dict.

    Boxes are read from ``item[bbox_key]``, divided by ``coord_divisor`` to
    reach the ``[0, 1]`` space (1.0 for already-normalized LFM2-VL output, 1000.0
    for Qwen-style ``bbox_2d`` on a 0-1000 scale), converted from ``box_format``
    to corner layout (``xyxy`` / ``xywh`` / ``cxcywh``), then scaled to pixel
    ``xyxy`` against ``original_size`` (W, H). Labels outside ``name_to_id`` and
    malformed boxes are skipped. If ``classes`` is provided, that class filter is
    applied before the ``max_det`` cap so requested classes are not dropped by an
    earlier out-of-filter prediction. ``default_score`` is the synthetic per-box
    confidence (the VLM emits none); rows below ``conf_thres`` are dropped so
    ``conf=`` still filters.
    """
    width, height = original_size
    boxes: List[List[float]] = []
    scores: List[float] = []
    class_ids: List[int] = []

    for item in items:
        class_id = resolve_label(item.get("label"), name_to_id)
        if class_id is None:
            continue
        raw = item.get(bbox_key)
        box = None
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            try:
                scaled = [float(v) / coord_divisor for v in raw]
            except (TypeError, ValueError):
                scaled = None
            box = normalize_bbox(to_xyxy(scaled, box_format)) if scaled else None
        if box is None:
            continue
        x1, y1, x2, y2 = box
        boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
        scores.append(default_score)
        class_ids.append(class_id)

    return finalize_detection_dict(
        boxes,
        scores,
        class_ids,
        original_size,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        classes=classes,
    )
