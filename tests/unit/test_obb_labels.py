"""Tests for OBB label parsing."""

import math

import numpy as np
import pytest

from libreyolo.data.obb import (
    corners_to_xywhr,
    parse_yolo_obb_label_line,
    scale_xywhr,
    xywhr_iou,
    xywhr_to_proxy_xyxy,
)

pytestmark = pytest.mark.unit


def test_parse_yolo_obb_label_line():
    cls_id, corners = parse_yolo_obb_label_line(
        "1 0.10 0.20 0.50 0.20 0.50 0.40 0.10 0.40",
        num_classes=3,
    )

    assert cls_id == 1
    assert corners.shape == (4, 2)
    assert corners.dtype == np.float32
    np.testing.assert_allclose(
        corners,
        np.array(
            [[0.10, 0.20], [0.50, 0.20], [0.50, 0.40], [0.10, 0.40]],
            dtype=np.float32,
        ),
    )


def test_parse_yolo_obb_label_line_accepts_split_parts():
    cls_id, corners = parse_yolo_obb_label_line(
        ["0", "0", "0", "1", "0", "1", "1", "0", "1"], num_classes=1
    )

    assert cls_id == 0
    assert corners.shape == (4, 2)


def test_parse_yolo_obb_label_line_can_clip_crop_boundary_rows():
    cls_id, corners = parse_yolo_obb_label_line(
        "0 -0.01 0.2 1.01 0.2 1.01 0.4 -0.01 0.4",
        num_classes=1,
        clip=True,
    )

    assert cls_id == 0
    np.testing.assert_allclose(
        corners,
        np.array([[0.0, 0.2], [1.0, 0.2], [1.0, 0.4], [0.0, 0.4]], dtype=np.float32),
    )


def test_corners_to_xywhr_and_proxy_box():
    _, corners = parse_yolo_obb_label_line(
        "0 0.10 0.20 0.50 0.20 0.50 0.40 0.10 0.40",
        num_classes=1,
    )

    xywhr = corners_to_xywhr(corners)
    proxy = xywhr_to_proxy_xyxy(xywhr)

    np.testing.assert_allclose(xywhr[:4], [0.30, 0.30, 0.40, 0.20], atol=1e-6)
    assert xywhr[4] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(proxy, [0.10, 0.20, 0.50, 0.40], atol=1e-6)


def test_corners_to_xywhr_rejects_degenerate_corners_by_default():
    corners = np.array([[0.5, 0.1], [0.5, 0.1], [0.5, 0.9], [0.5, 0.9]], dtype=np.float32)

    with pytest.raises(ValueError, match="width and height"):
        corners_to_xywhr(corners)


def test_scale_xywhr_can_clamp_degenerate_transform_outputs():
    scaled = scale_xywhr(
        np.array([0.5, 0.5, 0.0, 0.2, 0.0], dtype=np.float32),
        200.0,
        100.0,
        min_size=1e-4,
    )

    np.testing.assert_allclose(scaled[:2], [100.0, 50.0], atol=1e-6)
    assert scaled[2] > 0.0
    assert scaled[3] > 0.0


def test_xywhr_iou_handles_rotated_identity_and_disjoint_boxes():
    box = [32.0, 32.0, 20.0, 10.0, 0.5]

    assert xywhr_iou(box, box) == pytest.approx(1.0, abs=1e-6)
    assert xywhr_iou(box, [100.0, 100.0, 20.0, 10.0, 0.5]) == pytest.approx(0.0)


def test_xywhr_iou_is_pi_periodic():
    box = [32.0, 32.0, 20.0, 10.0, 0.5]
    same_box_pi_period = [32.0, 32.0, 20.0, 10.0, 0.5 + math.pi]

    assert xywhr_iou(box, same_box_pi_period) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("0 0.5 0.5 0.2 0.2", "Expected 9 fields"),
        ("1.5 0 0 1 0 1 1 0 1", "integer"),
        ("2 0 0 1 0 1 1 0 1", "out of range"),
        ("0 0 0 1 0 1 1 0 1.1", r"\[0, 1\]"),
        ("0 0 0 1 0 nan 1 0 1", "finite"),
        ("0 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5", "non-degenerate"),
        ("0 0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4", "non-degenerate"),
    ],
)
def test_parse_yolo_obb_label_line_rejects_invalid_rows(line, message):
    with pytest.raises(ValueError, match=message):
        parse_yolo_obb_label_line(line, num_classes=2)


def test_rotated_iou_matrix_matches_opencv_reference():
    """The vectorized IoU must stay exact: it replaces the OpenCV path in
    rotated NMS and OBB validation, so any drift silently moves mAP."""
    import torch

    from libreyolo.utils.box_ops import rotated_iou_matrix

    rng = np.random.default_rng(0)

    def sample(n):
        boxes = np.empty((n, 5), dtype=np.float32)
        boxes[:, 0:2] = rng.uniform(0, 200, (n, 2))
        boxes[:, 2:4] = rng.uniform(10, 70, (n, 2))
        boxes[:, 4] = rng.uniform(-math.pi / 2, math.pi / 2, n)
        return boxes

    a, b = sample(40), sample(30)
    actual = rotated_iou_matrix(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    expected = np.array(
        [[xywhr_iou(box_a, box_b) for box_b in b] for box_a in a], dtype=np.float32
    )

    assert (expected > 1e-6).any()  # the sample must actually contain overlaps
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_rotated_iou_matrix_edge_cases():
    import torch

    from libreyolo.utils.box_ops import rotated_iou_matrix

    box = torch.tensor([[50.0, 50.0, 40.0, 20.0, 0.3]])
    assert float(rotated_iou_matrix(box, box)) == pytest.approx(1.0, abs=1e-5)

    half_turn = box + torch.tensor([[0.0, 0.0, 0.0, 0.0, math.pi]])
    assert float(rotated_iou_matrix(box, half_turn)) == pytest.approx(1.0, abs=1e-5)

    far_away = torch.tensor([[900.0, 900.0, 40.0, 20.0, 0.3]])
    assert float(rotated_iou_matrix(box, far_away)) == pytest.approx(0.0, abs=1e-6)

    empty = torch.zeros((0, 5))
    assert rotated_iou_matrix(empty, box).shape == (0, 1)
    assert rotated_iou_matrix(box, empty).shape == (1, 0)


def test_rotated_nms_is_greedy_and_class_aware():
    """Greedy semantics: the highest-scoring box suppresses same-class
    overlaps and never suppresses a different class."""
    import torch

    from libreyolo.postprocess.yolo9 import _rotated_nms_keep_indices

    boxes = torch.tensor(
        [
            [50.0, 50.0, 40.0, 20.0, 0.0],  # best of its class
            [51.0, 50.0, 40.0, 20.0, 0.0],  # near-duplicate, same class -> dropped
            [50.0, 50.0, 40.0, 20.0, 0.0],  # same box, different class -> kept
            [200.0, 200.0, 40.0, 20.0, 0.0],  # far away -> kept
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    classes = torch.tensor([0, 0, 1, 0])

    keep = _rotated_nms_keep_indices(boxes, scores, classes, 0.45, 300)

    assert keep.tolist() == [0, 2, 3]


def test_rotated_nms_respects_max_det():
    import torch

    from libreyolo.postprocess.yolo9 import _rotated_nms_keep_indices

    boxes = torch.stack(
        [
            torch.tensor([100.0 * i, 100.0 * i, 20.0, 10.0, 0.0])
            for i in range(1, 8)
        ]
    )
    scores = torch.linspace(0.9, 0.3, 7)
    classes = torch.zeros(7, dtype=torch.long)

    keep = _rotated_nms_keep_indices(boxes, scores, classes, 0.45, max_det=3)

    assert keep.tolist() == [0, 1, 2]
