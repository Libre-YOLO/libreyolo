"""Regression test for the pose ``build_target`` reshape crash.

When an image has more keypoint instances than ``max_labels``, the capped row
count ``n`` differs from ``len(kpts_px)``. The old
``kpts_px[:n].reshape(len(kpts_px), -1)`` then raised a numpy shape error
(``n * K * 3`` elements cannot fill ``len(kpts_px)`` rows), crashing pose
training on any crowded image. This pins the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from libreyolo.data.augment.pose import build_target

pytestmark = pytest.mark.unit


def _valid_instances(m, k):
    cls = np.zeros(m, dtype=np.float32)
    # [x, y, w, h] with w*h > 1 so the keep-filter passes.
    bboxes_px = np.tile(np.array([10.0, 10.0, 20.0, 20.0], dtype=np.float32), (m, 1))
    kpts_px = np.zeros((m, k, 3), dtype=np.float32)
    kpts_px[..., 0] = 5.0
    kpts_px[..., 1] = 5.0
    kpts_px[..., 2] = 2.0  # visible → keep-filter passes
    return cls, bboxes_px, kpts_px


def test_build_target_caps_instances_over_max_labels_without_crashing():
    k, max_labels = 5, 10
    m = max_labels + 5  # more instances than slots
    cls, bboxes_px, kpts_px = _valid_instances(m, k)

    target = build_target(cls, bboxes_px, kpts_px, num_keypoints=k, max_labels=max_labels)

    assert target.shape == (max_labels, 5 + 3 * k)
    # Every slot is filled (all instances valid, capped at max_labels).
    assert int((target[:, 3] > 0).sum()) == max_labels
    # Keypoints made it into the tail columns.
    assert np.any(target[:, 5:] > 0)


def test_build_target_normal_case_unaffected():
    k, max_labels = 5, 20
    m = 3  # fewer instances than slots
    cls, bboxes_px, kpts_px = _valid_instances(m, k)

    target = build_target(cls, bboxes_px, kpts_px, num_keypoints=k, max_labels=max_labels)

    assert target.shape == (max_labels, 5 + 3 * k)
    assert int((target[:, 3] > 0).sum()) == m
