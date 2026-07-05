"""RF-DETR pose training transforms (compatibility shim).

The implementation moved to :mod:`libreyolo.data.augment.rfdetr`, where the
formerly re-declared copies of ``_resolve_training_size`` /
``_resize_square`` / ``_resize_shortest_side`` are deduplicated onto the seg
definitions (the pose ``_resolve_training_size`` was truth-table-identical).
This module re-exports the full historical surface so existing imports and
monkeypatch dotted paths keep working.
"""

from __future__ import annotations

import random  # noqa: F401  (historical module attribute; monkeypatched in tests)
from typing import Optional, Sequence  # noqa: F401

import cv2  # noqa: F401  (historical module attribute)
import numpy as np  # noqa: F401  (historical module attribute)

from ...data.augment.constants import (  # noqa: F401
    IMAGENET_MEAN as _IMAGENET_MEAN,
    IMAGENET_STD as _IMAGENET_STD,
)
from ...data.augment.rfdetr import (  # noqa: F401
    RFDETRPoseTransform,
    _build_target,
    _crop_pose,
    _cxcywh_to_xyxy,
    _resize_shortest_side,
    _resize_square,
    _resolve_training_size,
    _xyxy_to_cxcywh,
    compute_multi_scale_scales,
)

__all__ = ["RFDETRPoseTransform"]
