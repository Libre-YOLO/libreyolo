"""YOLO-NAS training transforms (compatibility shim).

The implementation moved to :mod:`libreyolo.data.augment.yolonas`. This
module re-exports the full historical surface, including the shared helpers
that were previously imported here and the stdlib/third-party module
attributes, so existing imports and monkeypatch dotted paths keep working.
"""

from __future__ import annotations

import random  # noqa: F401  (historical module attribute)

import cv2  # noqa: F401  (historical module attribute)
import numpy as np  # noqa: F401  (historical module attribute)

from libreyolo.data.augment.boxes import adjust_box_anns, xyxy2cxcywh  # noqa: F401
from libreyolo.data.augment.color import augment_hsv  # noqa: F401
from libreyolo.data.augment.geometry import mirror, random_affine  # noqa: F401
from libreyolo.data.augment.yolonas import (  # noqa: F401
    YOLONASAffineMixupDataset,
    YOLONASTrainTransform,
    preproc,
)
