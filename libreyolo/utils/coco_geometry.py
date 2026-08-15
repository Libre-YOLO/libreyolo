"""Shared COCO bounding-box cleanup.

Provenance for the adapted clipping rule is recorded in
``THIRD_PARTY_NOTICES.txt``.
"""

from collections.abc import Sequence
from typing import Optional


def clipped_coco_bbox_xyxy(
    bbox: Sequence[float], image_width: float, image_height: float
) -> Optional[tuple[float, float, float, float]]:
    """Convert COCO ``xywh`` to clipped ``xyxy``, or omit an empty box."""

    raw_x, raw_y, raw_width, raw_height = bbox
    x1 = max(0.0, raw_x)
    y1 = max(0.0, raw_y)
    x2 = min(image_width, x1 + max(0.0, raw_width))
    y2 = min(image_height, y1 + max(0.0, raw_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
