"""Regression tests for detection-head geometry caches."""

import pytest
import torch

from libreyolo.models.yolo9.nn import DDetect
from libreyolo.models.yolo9_e2e.nn import YOLO9E2EDetect
from libreyolo.models.yolo9_p2.nn import LibreYOLO9P2Model
from libreyolo.models.yolox.nn import YOLOXHead

pytestmark = pytest.mark.unit


def _raw_predictions(head, shapes, *, dtype=torch.float32, device="cpu"):
    return [
        torch.zeros((1, head.no, height, width), dtype=dtype, device=device)
        for height, width in shapes
    ]


def test_yolo9_anchor_cache_tracks_every_feature_shape():
    head = DDetect(nc=2, ch=(4, 8, 16), reg_max=4, stride=(8, 16, 32)).eval()

    first = [
        torch.zeros(1, 4, 8, 8),
        torch.zeros(1, 8, 4, 4),
        torch.zeros(1, 16, 2, 2),
    ]
    second = [first[0], torch.zeros(1, 8, 5, 4), first[2]]

    first_decoded, _ = head(first)
    second_decoded, _ = head(second)

    assert first_decoded.shape[-1] == 84
    assert second_decoded.shape[-1] == 88
    assert head.anchors.shape[-1] == 88


def test_yolo9_e2e_anchor_cache_tracks_every_feature_shape():
    head = YOLO9E2EDetect(
        nc=2, ch=(4, 8, 16), reg_max=4, stride=(8, 16, 32)
    ).eval()
    first = _raw_predictions(head, [(8, 8), (4, 4), (2, 2)])
    second = _raw_predictions(head, [(8, 8), (4, 4), (3, 2)])

    first_decoded, _ = head._inference(first)
    second_decoded, _ = head._inference(second)

    assert first_decoded.shape[-1] == 84
    assert second_decoded.shape[-1] == 86
    assert head.anchors.shape[-1] == 86


def test_yolo9_p2_anchor_cache_tracks_fourth_feature_shape():
    head = LibreYOLO9P2Model(config="t", nb_classes=2).head.eval()
    first = _raw_predictions(head, [(8, 8), (4, 4), (2, 2), (1, 1)])
    second = _raw_predictions(head, [(8, 8), (4, 4), (2, 2), (1, 2)])

    head._inference_anchors(first)
    anchors, _ = head._inference_anchors(second)

    assert anchors.shape[-1] == 86


def test_yolo9_anchor_cache_tracks_dtype_and_device_and_is_nonpersistent():
    head = DDetect(nc=2, ch=(4,), reg_max=4, stride=(8,)).eval()

    float_anchors, _ = head._inference_anchors(
        _raw_predictions(head, [(2, 3)], dtype=torch.float32)
    )
    double_anchors, _ = head._inference_anchors(
        _raw_predictions(head, [(2, 3)], dtype=torch.float64)
    )
    meta_anchors, _ = head._inference_anchors(
        _raw_predictions(head, [(2, 3)], dtype=torch.float64, device="meta")
    )

    assert float_anchors.dtype == torch.float32
    assert double_anchors.dtype == torch.float64
    assert meta_anchors.device.type == "meta"
    assert {"anchors", "strides"} <= dict(head.named_buffers()).keys()
    assert "anchors" not in head.state_dict()
    assert "strides" not in head.state_dict()


def test_yolox_training_grid_cache_tracks_shape_dtype_and_device():
    head = YOLOXHead(
        num_classes=2, width=0.25, strides=[8], in_channels=[64]
    )

    output = torch.zeros(1, 7, 2, 3, dtype=torch.float64)
    _, grid = head.get_output_and_grid(output, 0, 8, output.type())
    assert grid.shape == (1, 6, 2)
    assert grid.dtype == torch.float64
    assert grid.device.type == "cpu"

    output = torch.zeros(1, 7, 2, 4, dtype=torch.float32)
    _, grid = head.get_output_and_grid(output, 0, 8, output.type())
    assert grid.shape == (1, 8, 2)
    assert grid.dtype == torch.float32

    meta_output = torch.empty(1, 7, 2, 4, device="meta")
    _, grid = head.get_output_and_grid(meta_output, 0, 8, meta_output.type())
    assert grid.device.type == "meta"
