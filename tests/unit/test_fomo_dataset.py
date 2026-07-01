"""Unit tests for LibreFOMO datasets and data transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.fomo]


class TestFOMOTrainTransform:
    """Verify that FOMOTrainTransform performs direct stretch resizing and scales coordinates."""

    def test_stretch_resize_and_coordinate_scaling(self) -> None:
        from libreyolo.models.fomo.dataset import FOMOTrainTransform

        img = np.zeros((200, 400, 3), dtype=np.uint8)
        targets = np.array([[150.0, 50.0, 250.0, 150.0, 0.0]], dtype=np.float32)

        transform = FOMOTrainTransform(max_labels=10, flip_prob=0.0, hsv_prob=0.0)
        assert transform.wants_unresized_image is True

        img_out, targets_out = transform(img, targets, (96, 96))

        assert img_out.shape == (3, 96, 96)

        expected = np.array([0.0, 48.0, 48.0, 24.0, 48.0], dtype=np.float32)
        np.testing.assert_allclose(targets_out[0], expected, atol=1e-4)

    def test_empty_targets(self) -> None:
        from libreyolo.models.fomo.dataset import FOMOTrainTransform

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        targets = np.zeros((0, 5), dtype=np.float32)

        transform = FOMOTrainTransform(max_labels=10, flip_prob=0.0, hsv_prob=0.0)
        img_out, targets_out = transform(img, targets, (96, 96))

        assert img_out.shape == (3, 96, 96)
        assert targets_out.shape == (10, 5)
        assert targets_out.sum() == 0.0


class TestFOMODatasets:
    """Verify FOMOYOLODataset and FOMOAugmentedDataset functionality on synthetic data."""

    def test_fomo_yolo_dataset(self, tmp_path: Path) -> None:
        from libreyolo.models.fomo.dataset import FOMOYOLODataset

        # Set up synthetic image and label
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        img_path = img_dir / "sample.jpg"
        lbl_path = lbl_dir / "sample.txt"

        # 96x96 synthetic image
        Image.new("RGB", (96, 96), color=(255, 128, 64)).save(img_path)
        # YOLO box format: cls cx cy w h
        # One point at cx=0.5, cy=0.5
        lbl_path.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        dataset = FOMOYOLODataset(
            img_files=[img_path],
            label_files=[lbl_path],
            input_size=96,
            grid_size=12,
        )

        assert len(dataset) == 1
        img_tensor, grid, img_info, idx = dataset[0]

        # Check shapes and types
        assert img_tensor.shape == (3, 96, 96)
        assert img_tensor.dtype == torch.float32
        assert grid.shape == (12, 12)
        assert grid.dtype == torch.long
        assert img_info == (96, 96)
        assert idx == 0

        # Center should be class 0 + 1 = 1
        assert grid[6, 6] == 1
        assert grid.sum() == 1

    def test_fomo_yolo_dataset_polygon_labels(self, tmp_path: Path) -> None:
        from libreyolo.models.fomo.dataset import FOMOYOLODataset

        # Set up synthetic image and label
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        img_path = img_dir / "sample_poly.jpg"
        lbl_path = lbl_dir / "sample_poly.txt"

        Image.new("RGB", (96, 96), color=(255, 128, 64)).save(img_path)
        lbl_path.write_text("0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n", encoding="utf-8")

        dataset = FOMOYOLODataset(
            img_files=[img_path],
            label_files=[lbl_path],
            input_size=96,
            grid_size=12,
        )

        assert len(dataset) == 1
        _, grid, _, _ = dataset[0]

        assert grid[6, 6] == 1
        assert grid.sum() == 1

    def test_fomo_augmented_dataset(self) -> None:
        from libreyolo.models.fomo.dataset import FOMOAugmentedDataset

        # Create a mock augmented dataset that returns (img, targets, img_info, img_id)
        # YOLOX-style image shape (C, H, W) in BGR
        mock_img = np.zeros((3, 96, 96), dtype=np.uint8)
        # target row: [class, cx, cy, w, h, ...]
        mock_targets = np.array([
            [0.0, 48.0, 48.0, 10.0, 10.0]
        ], dtype=np.float32)
        mock_img_info = (96, 96)
        mock_img_id = 42

        class MockBaseDataset:
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return mock_img, mock_targets, mock_img_info, mock_img_id
            def close_mosaic(self):
                self.closed = True

        base_ds = MockBaseDataset()
        base_ds.closed = False

        dataset = FOMOAugmentedDataset(
            augmented_dataset=base_ds,
            input_size=96,
            grid_size=12,
        )

        assert len(dataset) == 1
        img_tensor, grid, img_info, img_id = dataset[0]

        assert img_tensor.shape == (3, 96, 96)
        assert grid.shape == (12, 12)
        assert img_info == (96, 96)
        assert img_id == 42

        assert grid[6, 6] == 1
        assert grid.sum() == 1

        dataset.close_mosaic()
        assert base_ds.closed is True
