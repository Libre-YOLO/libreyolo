"""Unit tests for HRNet pose result decoding and rescoring."""

from __future__ import annotations

import numpy as np
import pytest

from libreyolo.models.hrnet.utils import box_to_center_scale
from libreyolo.postprocess.hrnet import oks_nms, postprocess_hrnet

pytestmark = pytest.mark.unit


def _constant_pose_heatmaps(values: list[float]) -> np.ndarray:
    heatmaps = np.zeros((len(values), 17, 8, 6), dtype=np.float32)
    for instance, value in enumerate(values):
        heatmaps[instance, :, 4, 3] = value
    return heatmaps


def test_postprocess_preserves_boxes_and_rescores_poses():
    boxes = np.asarray(
        [[10, 20, 110, 220], [200, 50, 300, 250]],
        dtype=np.float32,
    )
    geometry = [box_to_center_scale(box, (256, 192)) for box in boxes]
    centers = np.stack([item[0] for item in geometry])
    scales = np.stack([item[1] for item in geometry])

    result = postprocess_hrnet(
        _constant_pose_heatmaps([0.8, 0.5]),
        centers,
        scales,
        boxes,
        np.asarray([0.9, 0.6], dtype=np.float32),
        oks_threshold=1.0,
    )

    assert result["num_detections"] == 2
    assert np.array_equal(result["boxes"], boxes)
    assert np.allclose(result["scores"], [0.72, 0.30])
    assert result["keypoints"].shape == (2, 17, 3)
    assert np.allclose(result["keypoints"][:, :, 2], [[0.8] * 17, [0.5] * 17])


def test_postprocess_suppresses_duplicate_poses_by_oks():
    boxes = np.asarray([[10, 20, 110, 220], [10, 20, 110, 220]], dtype=np.float32)
    center, scale = box_to_center_scale(boxes[0], (256, 192))
    result = postprocess_hrnet(
        _constant_pose_heatmaps([0.9, 0.8]),
        np.stack([center, center]),
        np.stack([scale, scale]),
        boxes,
        np.asarray([0.9, 0.8], dtype=np.float32),
        oks_threshold=0.9,
    )

    assert result["num_detections"] == 1
    assert result["scores"][0] == pytest.approx(0.81)


def test_oks_nms_empty_input():
    keep = oks_nms(
        np.zeros((0, 17, 3), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )
    assert keep.shape == (0,)
