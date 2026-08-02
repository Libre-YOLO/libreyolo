"""SSD300 default-box decode and public class-mapping tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.postprocess.ssd import _default_boxes, postprocess
from libreyolo.utils.coco import COCO91_TO_COCO80

pytestmark = pytest.mark.unit


def _raw_outputs() -> dict[str, torch.Tensor]:
    logits = torch.full((1, 8732, 91), -20.0)
    logits[..., 0] = 0.0
    return {
        "bbox_regression": torch.zeros(1, 8732, 4),
        "cls_logits": logits,
    }


def test_default_box_inventory_and_first_pair():
    boxes = _default_boxes(device=torch.device("cpu"), dtype=torch.float32)
    assert boxes.shape == (8732, 4)
    np.testing.assert_allclose(
        boxes[0].numpy(),
        np.array([-6.5, -6.5, 14.5, 14.5], dtype=np.float32),
        atol=1e-6,
    )


def test_postprocess_maps_sparse_coco_ids_and_scales_direct_resize():
    outputs = _raw_outputs()
    outputs["cls_logits"][0, 100, 13] = 20.0
    result = postprocess(
        outputs,
        conf_thres=0.5,
        original_size=(600, 150),
        class_map=COCO91_TO_COCO80,
    )

    assert result["num_detections"] == 1
    assert result["classes"].tolist() == [11]
    expected = _default_boxes(device=torch.device("cpu"), dtype=torch.float32)[100]
    expected[[0, 2]] *= 2.0
    expected[[1, 3]] *= 0.5
    expected[0::2].clamp_(0, 600)
    expected[1::2].clamp_(0, 150)
    np.testing.assert_allclose(result["boxes"][0], expected.numpy(), atol=1e-5)


def test_postprocess_drops_unassigned_coco_head_slots():
    outputs = _raw_outputs()
    outputs["cls_logits"][0, 200, 12] = 20.0
    result = postprocess(
        outputs,
        conf_thres=0.5,
        class_map=COCO91_TO_COCO80,
    )

    assert result["num_detections"] == 0
    assert result["boxes"].shape == (0, 4)


def test_postprocess_applies_per_class_nms_and_max_det():
    outputs = _raw_outputs()
    outputs["cls_logits"][0, 0, 1] = 20.0
    outputs["cls_logits"][0, 1, 1] = 19.0
    outputs["bbox_regression"][0, 1, 0] = -0.35
    outputs["bbox_regression"][0, 1, 1] = -0.35
    result = postprocess(
        outputs,
        conf_thres=0.5,
        iou_thres=0.45,
        max_det=1,
        class_map=COCO91_TO_COCO80,
    )

    assert result["num_detections"] == 1
    assert result["scores"][0] > 0.99


def test_postprocess_rejects_non_ssd_raw_shapes():
    with pytest.raises(ValueError, match="8,732 anchors"):
        postprocess(
            {
                "bbox_regression": torch.zeros(1, 10, 4),
                "cls_logits": torch.zeros(1, 10, 91),
            }
        )
