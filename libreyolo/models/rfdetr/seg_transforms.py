"""RF-DETR seg/det transforms (compatibility shim).

The implementation moved to :mod:`libreyolo.data.augment.rfdetr`. This module
re-exports the full historical surface — classes, underscore helpers, and the
stdlib/third-party module attributes — so existing imports (including the EC
seg trainer's) and monkeypatch dotted paths like
``libreyolo.models.rfdetr.seg_transforms.random.random`` keep working.
``__all__`` is kept byte-identical to the pre-move module.
"""

from __future__ import annotations

import random  # noqa: F401  (historical module attribute; monkeypatched in tests)

import cv2  # noqa: F401  (historical module attribute)
import numpy as np  # noqa: F401  (historical module attribute)

from ...data.obb import normalize_obb_angle, scale_xywhr  # noqa: F401
from ...data.augment.constants import (  # noqa: F401
    IMAGENET_MEAN as _IMAGENET_MEAN,
    IMAGENET_STD as _IMAGENET_STD,
)
from ...data.augment.rfdetr import (  # noqa: F401
    RFDETRDetTransform,
    RFDETRSegPassThroughDataset,
    RFDETRSegTransform,
    _copy_segments,
    _crop_segments,
    _dense_mask,
    _filter_segments,
    _flip_segments_lr,
    _instance_dense_mask,
    _materialize_dense_masks_for_crop,
    _rasterize_segments,
    _resize_shortest_side,
    _resize_square,
    _resolve_training_size,
    _scale_segments_xy,
    _set_dense_mask,
    compute_multi_scale_scales,
)

__all__ = ["RFDETRSegTransform", "RFDETRSegPassThroughDataset"]
