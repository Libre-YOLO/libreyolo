"""YOLOv9 training transforms (compatibility shim).

The implementation moved to :mod:`libreyolo.data.augment.yolo9`. This module
re-exports the full historical surface — classes, module-level helpers,
underscore-private segment helpers, and the stdlib/third-party module
attributes — so existing imports (including the RT-DETR trainer's) and
monkeypatch dotted paths keep working.
"""

import random  # noqa: F401  (historical module attribute)

import cv2  # noqa: F401  (historical module attribute)
import numpy as np  # noqa: F401  (historical module attribute)

from libreyolo.data.obb import normalize_obb_angle  # noqa: F401
from libreyolo.data.augment.color import augment_hsv  # noqa: F401
from libreyolo.data.augment.geometry import mirror, random_affine  # noqa: F401
from libreyolo.data.augment.segments import (  # noqa: F401
    copy_segments as _copy_segments,
    filter_segments as _filter_segments,
    flip_segments_lr as _flip_segments_lr,
    flip_segments_ud as _flip_segments_ud,
    rasterize_segments as _rasterize_segments,
    transform_segments as _transform_segments,
)
from libreyolo.data.augment.yolo9 import (  # noqa: F401
    YOLO9MosaicMixupDataset,
    YOLO9TrainTransform,
    YOLO9ValTransform,
    preproc,
)
