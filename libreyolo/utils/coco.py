"""COCO category-id helpers shared across model families.

COCO's ``instances_*.json`` uses non-contiguous category ids in ``1..90`` (11
ids are unused). DETR-lineage detectors trained on the raw annotations
therefore emit a 91-wide classification head whose column index *is* the COCO
category id, while LibreYOLO's user-facing interface is the contiguous
80-class YOLO ordering. Families with a 91-wide head map through
:data:`COCO91_TO_COCO80` on the way out.
"""

from __future__ import annotations

__all__ = ["COCO91_CATEGORY_IDS", "COCO91_TO_COCO80"]

# The 80 COCO category ids that carry annotations, in ascending order. Their
# position in this list is the contiguous YOLO class index.
COCO91_CATEGORY_IDS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)

COCO91_TO_COCO80: dict[int, int] = {
    category_id: index for index, category_id in enumerate(COCO91_CATEGORY_IDS)
}
