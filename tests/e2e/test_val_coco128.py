"""
val_128: Validation sanity check for all catalog pretrained models.

Runs model.val() on coco128.yaml (128 images) and checks mAP50-95 >= 0.18.
Purpose: catch catastrophic regressions (broken preprocessing, wrong class
mapping, etc.) — NOT exact mAP benchmarking.

Usage:
    pytest tests/e2e/test_val_coco128.py -v -m e2e
    pytest tests/e2e/test_val_coco128.py::test_val_coco128[yolox-n] -v
    pytest tests/e2e/test_val_coco128.py -k "rfdetr" -v
"""

import pytest

from libreyolo import LibreYOLO
from .conftest import (
    ALL_MODEL_WEIGHT_PARAMS,
    cuda_cleanup,
    require_test_weights,
)

pytestmark = pytest.mark.e2e

MIN_MAP = 0.18  # Uniform threshold for all models


@pytest.mark.parametrize(
    "family,size,weights",
    ALL_MODEL_WEIGHT_PARAMS,
)
def test_val_coco128(family, size, weights):
    """Validate a pretrained model on coco128 and check mAP >= 0.18."""
    weights = require_test_weights(weights)
    # coco128 ships box-only YOLO labels. Mask R-CNN defaults to segment, so
    # use its shared checkpoint in detect mode for this detection-mAP gate;
    # mask parity and segmentation validation dispatch are covered separately.
    model_kwargs = {"task": "detect"} if family == "mask_rcnn" else {}
    model = LibreYOLO(weights, size=size, **model_kwargs)

    # The portable Deformable DETR implementation deliberately uses GridSample
    # instead of its upstream CUDA extension. Multi-scale encoder attention is
    # memory-heavy at 800 px, so validate it one image at a time.
    batch = 1 if family in {"deformable_detr", "mask_rcnn"} else 16
    results = model.val(data="coco128.yaml", batch=batch, conf=0.001, iou=0.6)

    map50_95 = results["metrics/mAP50-95"]
    map50 = results["metrics/mAP50"]

    print(f"\n  {weights} (size={size}): mAP50-95={map50_95:.4f}, mAP50={map50:.4f}")

    assert map50_95 >= MIN_MAP, (
        f"mAP50-95={map50_95:.4f} below threshold {MIN_MAP} — "
        f"model may be broken (wrong preprocessing, class mapping, etc.)"
    )

    cuda_cleanup()
