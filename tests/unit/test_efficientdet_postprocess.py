"""EfficientDet anchor/decode/NMS contracts."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from libreyolo.postprocess.efficientdet import (
    decode_candidates,
    generate_anchors,
    postprocess,
)
from libreyolo.validation.preprocessors import EfficientDetValPreprocessor

pytestmark = pytest.mark.unit


def _synthetic_output(input_size: int = 128):
    class_outputs = []
    box_outputs = []
    for stride in (8, 16, 32, 64, 128):
        cells = input_size // stride
        class_outputs.append(torch.full((1, 810, cells, cells), -100.0))
        box_outputs.append(torch.zeros((1, 36, cells, cells)))
    # First anchor: category ids 1 and 13 are valid; id 12 is a COCO gap.
    class_outputs[0][0, 0, 0, 0] = 10.0
    class_outputs[0][0, 11, 0, 0] = 9.5
    class_outputs[0][0, 12, 0, 0] = 9.0
    return class_outputs, box_outputs


def test_anchor_shape_order_and_known_coordinates():
    anchors = generate_anchors(512)
    assert anchors.shape == (49104, 4)
    assert torch.equal(anchors[0], torch.tensor([-12.0, -12.0, 20.0, 20.0]))
    # Anchor configuration changes before the spatial location changes.
    assert torch.equal(anchors[9], torch.tensor([-12.0, -4.0, 20.0, 28.0]))


def test_decode_candidates_marks_sparse_coco_gaps():
    candidates = decode_candidates(
        _synthetic_output(), input_size=128, max_candidates=3, sparse_coco=True
    )[0]
    assert candidates.shape == (3, 6)
    assert candidates[:, 5].long().tolist() == [0, -1, 11]
    assert torch.allclose(candidates[0, :4], torch.tensor([-12.0, -12.0, 20.0, 20.0]))


def test_postprocess_filters_gaps_and_scales_top_left_padding():
    result = postprocess(
        _synthetic_output(),
        input_size=128,
        original_size=(64, 48),
        ratio=2.0,
        conf_thres=0.5,
        iou_thres=0.5,
        max_candidates=3,
        max_det=10,
    )
    assert result["num_detections"] == 2
    assert result["classes"].tolist() == [0, 11]
    assert result["boxes"].tolist() == [[0.0, 0.0, 10.0, 10.0]] * 2


def test_postprocess_honors_threshold_and_max_det():
    empty = postprocess(
        _synthetic_output(), input_size=128, conf_thres=1.0, max_candidates=3
    )
    assert empty["num_detections"] == 0

    one = postprocess(
        _synthetic_output(), input_size=128, conf_thres=0.5, max_candidates=3, max_det=1
    )
    assert one["num_detections"] == 1
    assert one["classes"].tolist() == [0]


def test_validation_preprocessor_matches_family_pipeline():
    from libreyolo.models.efficientdet.utils import preprocess_numpy

    rng = np.random.default_rng(7)
    bgr = rng.integers(0, 256, size=(40, 80, 3), dtype=np.uint8)
    targets = np.array([[4.0, 5.0, 20.0, 30.0, 3.0]], dtype=np.float32)
    preprocessor = EfficientDetValPreprocessor((64, 64), max_labels=4)
    actual_image, actual_targets = preprocessor(bgr, targets, (64, 64))
    expected_image, ratio = preprocess_numpy(bgr[:, :, ::-1], input_size=64)

    np.testing.assert_array_equal(actual_image, expected_image)
    np.testing.assert_allclose(actual_targets[0, :4], targets[0, :4] * ratio)
    assert actual_targets[0, 4] == 3
    assert np.count_nonzero(actual_targets[1:]) == 0
    assert preprocessor.uses_letterbox
    assert preprocessor.custom_normalization
    assert preprocessor.wants_unresized_image
