"""RetinaNet candidate filtering, scaling, and class-aware NMS tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.postprocess.retinanet import (
    level_anchor_counts,
    postprocess,
    resize_geometry,
)

pytestmark = [pytest.mark.unit, pytest.mark.retinanet]


def _output(original_size=(64, 64), input_size=64, classes=3):
    resized_h, resized_w, _, _ = resize_geometry(original_size, input_size)
    anchors = sum(level_anchor_counts(resized_h, resized_w))
    return torch.zeros(1, anchors, 4 + classes)


def test_same_box_different_classes_survives_class_aware_nms():
    output = _output()
    output[0, 0, :4] = torch.tensor([8.0, 8.0, 40.0, 40.0])
    output[0, 0, 4:] = torch.tensor([0.9, 0.8, 0.0])
    result = postprocess(
        output,
        conf_thres=0.5,
        iou_thres=0.5,
        original_size=(64, 64),
        input_size=64,
    )
    assert result["num_detections"] == 2
    np.testing.assert_array_equal(result["classes"], np.array([0, 1]))
    np.testing.assert_allclose(result["scores"], np.array([0.9, 0.8]))


def test_overlapping_same_class_is_suppressed_and_class_filter_applies():
    output = _output()
    output[0, 0, :4] = torch.tensor([4.0, 4.0, 44.0, 44.0])
    output[0, 1, :4] = torch.tensor([5.0, 5.0, 43.0, 43.0])
    output[0, 0, 4:] = torch.tensor([0.9, 0.7, 0.0])
    output[0, 1, 4:] = torch.tensor([0.8, 0.0, 0.0])
    result = postprocess(
        output,
        conf_thres=0.5,
        iou_thres=0.5,
        original_size=(64, 64),
        input_size=64,
        classes=[0],
    )
    assert result["num_detections"] == 1
    np.testing.assert_array_equal(result["classes"], np.array([0]))
    np.testing.assert_allclose(result["scores"], np.array([0.9]))


def test_non_square_geometry_restores_and_clips_coordinates():
    original_size = (200, 100)
    output = _output(original_size=original_size, input_size=64)
    resized_h, resized_w, scale_x, scale_y = resize_geometry(original_size, 64)
    output[0, 0, :4] = torch.tensor([-5.0, 10.0, resized_w + 10.0, resized_h - 5.0])
    output[0, 0, 4] = 0.9
    result = postprocess(
        output,
        conf_thres=0.5,
        original_size=original_size,
        input_size=64,
    )
    assert result["num_detections"] == 1
    np.testing.assert_allclose(
        result["boxes"][0],
        np.array([0.0, 10.0 / scale_y, 200.0, (resized_h - 5.0) / scale_y]),
        rtol=0,
        atol=2e-5,
    )
    assert scale_x == resized_w / original_size[0]


def test_per_level_topk_and_max_det_are_enforced():
    output = _output(classes=2)
    first_level = level_anchor_counts(64, 64)[0]
    candidates = min(first_level, 550)
    output[0, :candidates, :4] = torch.tensor([1.0, 1.0, 2.0, 2.0])
    output[0, :candidates, 4] = torch.linspace(0.51, 0.99, candidates)
    result = postprocess(
        output,
        conf_thres=0.5,
        iou_thres=1.0,
        original_size=(64, 64),
        input_size=64,
        topk_candidates=100,
        max_det=25,
    )
    assert result["num_detections"] == 25
    assert np.all(result["scores"][:-1] >= result["scores"][1:])


def test_empty_and_bad_geometry_contracts():
    result = postprocess(
        _output(),
        conf_thres=0.5,
        original_size=(64, 64),
        input_size=64,
    )
    assert result["boxes"].shape == (0, 4)
    assert result["scores"].dtype == np.float32
    assert result["classes"].dtype == np.int64

    with pytest.raises(ValueError, match="anchor count"):
        postprocess(
            torch.zeros(1, 1, 7),
            original_size=(64, 64),
            input_size=64,
        )
