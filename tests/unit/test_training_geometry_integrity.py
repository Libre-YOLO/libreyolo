"""Regression tests for training-target geometry trust boundaries."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_random_affine_drops_box_moved_fully_off_canvas(monkeypatch):
    import libreyolo.data.augment.geometry as geometry

    monkeypatch.setattr(
        geometry,
        "get_affine_matrix",
        lambda *_args, **_kwargs: (
            np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 100.0]]),
            1.0,
        ),
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    targets = np.array([[4.0, 4.0, 12.0, 12.0, 0.0]], dtype=np.float32)

    _, transformed = geometry.random_affine(
        image,
        targets,
        target_size=(32, 32),
    )

    assert transformed.shape == (0, 5)


def test_fomo_grid_rejects_zero_height_target():
    from libreyolo.models.fomo.dataset import FOMOAugmentedDataset

    class _GhostDataset:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            image = np.zeros((3, 32, 32), dtype=np.float32)
            targets = np.array([[0.0, 16.0, 16.0, 8.0, 0.0]], dtype=np.float32)
            return image, targets, (32, 32), 0

    dataset = FOMOAugmentedDataset(_GhostDataset(), input_size=32, grid_size=4)

    _, grid, _, _ = dataset[0]

    assert int(grid.sum()) == 0


@pytest.mark.parametrize("family", ["yolonas", "ec", "rfdetr"])
@pytest.mark.parametrize("flip_idx", [[0, 0], [0, 2], [0]])
def test_pose_transforms_reject_invalid_flip_permutations(family, flip_idx):
    if family == "yolonas":
        from libreyolo.models.yolonas.pose_transforms import (
            YOLONASPoseTrainTransform as Transform,
        )
    elif family == "ec":
        from libreyolo.models.ec.pose_transforms import ECPoseTrainTransform as Transform
    else:
        from libreyolo.data.augment.rfdetr import RFDETRPoseTransform as Transform

    with pytest.raises(ValueError, match="flip_idx"):
        Transform(num_keypoints=2, flip_idx=flip_idx)


def test_yolonas_pose_transform_clips_outside_keypoint_and_hides_it(monkeypatch):
    import libreyolo.models.yolonas.pose_transforms as transforms

    def _outside_affine(image, boxes, keypoints, **_kwargs):
        keypoints = keypoints.copy()
        keypoints[0, 0, 0] = -5.0
        return image, boxes, keypoints

    monkeypatch.setattr(transforms, "_random_affine", _outside_affine)
    monkeypatch.setattr(transforms.random, "random", lambda: 0.0)
    transform = transforms.YOLONASPoseTrainTransform(
        num_keypoints=2,
        flip_idx=None,
        hsv_prob=0.0,
        brightness_contrast_prob=0.0,
        affine_prob=1.0,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    classes = np.array([0.0], dtype=np.float32)
    keypoints = np.array(
        [[[0.25, 0.5, 2.0], [0.75, 0.5, 2.0]]],
        dtype=np.float32,
    )

    _, target = transform(image, boxes, classes, keypoints, (64, 64))

    assert target[0, 5] == 0.0
    assert target[0, 7] == 0.0
    assert target[0, 8] == pytest.approx(48.0)
    assert target[0, 10] == 2.0


@pytest.mark.parametrize("family", ["yolo9", "rfdetr"])
def test_segment_transforms_drop_positive_box_with_empty_mask(family):
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    targets = np.array([[10.0, 10.0, 30.0, 30.0, 0.0]], dtype=np.float32)
    empty_instance = [[]]

    if family == "yolo9":
        from libreyolo.data.augment.yolo9 import YOLO9TrainTransform

        transform = YOLO9TrainTransform(
            max_labels=4,
            flip_prob=0.0,
            vertical_flip_prob=0.0,
            hsv_prob=0.0,
            mask_downsample_ratio=1,
        )
        _, labels, masks = transform(image, targets, (40, 40), empty_instance)
        assert (labels[:, 0] == -1).all()
    else:
        from libreyolo.data.augment.rfdetr import RFDETRSegTransform

        transform = RFDETRSegTransform(
            max_labels=4,
            flip_prob=0.0,
            imgsz=40,
            mask_downsample_ratio=1,
        )
        _, labels, masks = transform(image, targets, (40, 40), empty_instance)
        assert (labels[:, 3:5] == 0).all()

    assert masks.shape == (0, 40, 40)
