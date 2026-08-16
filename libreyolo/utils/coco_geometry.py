"""Shared COCO bounding-box cleanup.

Provenance for the adapted clipping rule is recorded in
``THIRD_PARTY_NOTICES.txt``.
"""

import math
from collections.abc import Sequence
from numbers import Real
from typing import Optional


def clipped_coco_bbox_xyxy(
    bbox: Sequence[float], image_width: float, image_height: float
) -> Optional[tuple[float, float, float, float]]:
    """Convert COCO ``xywh`` to clipped ``xyxy``, or omit an empty box."""

    try:
        raw_values = tuple(bbox)
    except TypeError as exc:
        raise ValueError("COCO bbox must contain four finite numbers.") from exc
    if len(raw_values) != 4 or any(
        isinstance(value, bool) or not isinstance(value, Real) for value in raw_values
    ):
        raise ValueError("COCO bbox must contain four finite numbers.")
    if (
        isinstance(image_width, bool)
        or isinstance(image_height, bool)
        or not isinstance(image_width, Real)
        or not isinstance(image_height, Real)
    ):
        raise ValueError("COCO image dimensions must be finite and positive.")
    try:
        raw_x, raw_y, raw_width, raw_height = map(float, raw_values)
        image_width = float(image_width)
        image_height = float(image_height)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("COCO geometry must contain finite numbers.") from exc
    if not all(
        math.isfinite(value)
        for value in (
            raw_x,
            raw_y,
            raw_width,
            raw_height,
            image_width,
            image_height,
        )
    ):
        raise ValueError("COCO geometry must contain finite numbers.")
    if image_width <= 0.0 or image_height <= 0.0:
        raise ValueError("COCO image dimensions must be finite and positive.")

    x1 = max(0.0, raw_x)
    y1 = max(0.0, raw_y)
    x2 = min(image_width, raw_x + max(0.0, raw_width))
    y2 = min(image_height, raw_y + max(0.0, raw_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
