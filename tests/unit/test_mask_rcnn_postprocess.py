"""Mask R-CNN canonical mask, label, and alignment tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.postprocess.mask_rcnn import postprocess
from libreyolo.utils.coco import COCO91_TO_COCO80

pytestmark = pytest.mark.unit


def test_masks_stay_aligned_after_sparse_label_filtering_and_sorting():
    masks = torch.zeros(3, 1, 4, 5)
    masks[0] = 0.9
    masks[1, :, 1:3, 1:4] = 0.6
    masks[2, :, 0:2, 0:2] = 0.7
    outputs = [
        {
            "boxes": torch.tensor(
                [
                    [-1.0, -2.0, 20.0, 22.0],
                    [1.0, 1.0, 4.0, 3.0],
                    [0.0, 0.0, 2.0, 2.0],
                ]
            ),
            "scores": torch.tensor([0.99, 0.90, 0.80]),
            "labels": torch.tensor([12, 1, 90]),
            "masks": masks,
        }
    ]
    result = postprocess(
        outputs,
        conf_thres=0.1,
        original_size=(5, 4),
        max_det=2,
        class_map=COCO91_TO_COCO80,
    )

    assert result["num_detections"] == 2
    np.testing.assert_array_equal(result["classes"], np.array([0, 79]))
    assert result["masks"].shape == (2, 4, 5)
    assert result["masks"].dtype == np.bool_
    np.testing.assert_array_equal(result["masks"][0], masks[1, 0] >= 0.5)
    np.testing.assert_array_equal(result["masks"][1], masks[2, 0] >= 0.5)


def test_detect_task_can_omit_masks():
    result = postprocess(
        {
            "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
        },
        include_masks=False,
    )
    assert result["num_detections"] == 1
    assert "masks" not in result


def test_empty_segment_output_has_original_canvas_shape():
    result = postprocess(
        (
            torch.zeros((0, 4)),
            torch.zeros((0,)),
            torch.zeros((0,), dtype=torch.int64),
            torch.zeros((0, 7, 9)),
        ),
        original_size=(9, 7),
    )
    assert result["masks"].shape == (0, 7, 9)
    assert result["masks"].dtype == np.bool_
