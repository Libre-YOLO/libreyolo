"""Faster R-CNN canonical detection and label-space tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.postprocess.faster_rcnn import postprocess
from libreyolo.utils.coco import COCO91_TO_COCO80

pytestmark = pytest.mark.unit


def test_coco_sparse_labels_are_contiguous_and_do_not_consume_budget():
    outputs = [
        {
            "boxes": torch.tensor(
                [[-1.0, -2.0, 20.0, 22.0], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
            ),
            "scores": torch.tensor([0.99, 0.90, 0.80]),
            # 12 is an unused COCO category id; 1 and 90 map to 0 and 79.
            "labels": torch.tensor([12, 1, 90]),
        }
    ]
    result = postprocess(
        outputs,
        conf_thres=0.1,
        original_size=(10, 10),
        max_det=2,
        class_map=COCO91_TO_COCO80,
    )
    assert result["num_detections"] == 2
    np.testing.assert_array_equal(result["classes"], np.array([0, 79]))
    np.testing.assert_allclose(result["scores"], np.array([0.9, 0.8]))
    np.testing.assert_allclose(
        result["boxes"], np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    )


def test_custom_head_labels_are_shifted_past_background():
    outputs = {
        "boxes": torch.tensor([[0.0, 0.0, 8.0, 9.0], [1.0, 1.0, 4.0, 4.0]]),
        "scores": torch.tensor([0.75, 0.2]),
        "labels": torch.tensor([3, 1]),
    }
    result = postprocess(outputs, conf_thres=0.5, max_det=10)
    assert result["num_detections"] == 1
    assert result["classes"].dtype == np.int64
    np.testing.assert_array_equal(result["classes"], np.array([2]))


def test_empty_output_uses_canonical_shapes_and_dtypes():
    result = postprocess(
        (
            torch.zeros((1, 0, 4)),
            torch.zeros((1, 0)),
            torch.zeros((1, 0), dtype=torch.int64),
        )
    )
    assert result["num_detections"] == 0
    assert result["boxes"].shape == (0, 4)
    assert result["scores"].shape == (0,)
    assert result["classes"].shape == (0,)
    assert result["boxes"].dtype == np.float32
    assert result["classes"].dtype == np.int64
