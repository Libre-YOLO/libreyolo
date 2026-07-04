"""Mosaic tile-placement math shared by the mosaic dataset wrappers.

Moved verbatim from ``libreyolo/training/augment.py`` (originally adapted
from the official YOLOX repository). This module holds only the quadrant
placement arithmetic — the piece that is provably identical between the
YOLOX and YOLO9 mosaic wrappers. The wrappers themselves stay separate
because they differ in tile scaling, canvas allocation, mixup RNG order,
and label handling.
"""


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    """Return (large_coords, small_coords) for placing an image in the 2x2 mosaic."""
    # Top-left
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # Top-right
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # Bottom-left
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # Bottom-right
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord
