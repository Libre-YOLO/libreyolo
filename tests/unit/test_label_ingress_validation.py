"""Trust-boundary tests for normalized text-label ingestion."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from libreyolo.data.dataset import YOLODataset
from libreyolo.data.pose_dataset import YOLOPoseDataset
from libreyolo.data.semantic_dataset import SemanticDataset

pytestmark = pytest.mark.unit


def _write_detection_sample(tmp_path: Path, rows: str) -> tuple[Path, Path]:
    image_path = tmp_path / "images" / "train" / "sample.jpg"
    label_path = tmp_path / "labels" / "train" / "sample.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), color="white").save(image_path)
    label_path.write_text(rows, encoding="utf-8")
    return image_path, label_path


@pytest.mark.parametrize(
    ("invalid_row", "message"),
    [
        ("0.5 0.5 0.5 0.2 0.2", "must be an integer"),
        ("1 0.5 0.5 0.2 0.2", "out of range"),
        ("0 nan 0.5 0.2 0.2", "must be finite"),
        ("0 1.1 0.5 0.2 0.2", "normalized to [0, 1]"),
        ("0 0.5 0.5 0.0 0.2", "must be positive"),
        ("0 0.1 0.1 0.9 0.1 0.9", "even number"),
        ("0 0.1 0.1 0.2 0.2 0.3 0.3", "non-degenerate"),
        ("0 0.1 0.1 0.9 0.1 0.9 1.1", "normalized to [0, 1]"),
    ],
)
def test_yolo_dataset_rejects_malformed_rows_with_file_and_row_context(
    tmp_path, invalid_row, message
):
    image_path, label_path = _write_detection_sample(
        tmp_path,
        f"0 0.5 0.5 0.2 0.2\n{invalid_row}\n",
    )

    with pytest.raises(ValueError) as exc_info:
        YOLODataset(
            img_files=[image_path],
            label_files=[label_path],
            load_segments=True,
            num_classes=1,
        )

    assert f"{label_path}:2:" in str(exc_info.value)
    assert message in str(exc_info.value)


def test_yolo_dataset_keeps_missing_and_empty_label_files(tmp_path):
    image_path, label_path = _write_detection_sample(tmp_path, "")
    missing_label = label_path.with_name("missing.txt")

    empty_dataset = YOLODataset(
        img_files=[image_path],
        label_files=[label_path],
        num_classes=1,
    )
    missing_dataset = YOLODataset(
        img_files=[image_path],
        label_files=[missing_label],
        num_classes=1,
    )

    assert empty_dataset.annotations[0][0].shape == (0, 5)
    assert missing_dataset.annotations[0][0].shape == (0, 5)


@pytest.mark.parametrize(
    ("invalid_row", "message"),
    [
        ("0.5 0.5 0.5 0.2 0.2 0.5 0.5 2", "must be an integer"),
        ("1 0.5 0.5 0.2 0.2 0.5 0.5 2", "out of range"),
        ("0 0.5 0.5 0.2 0.2 nan 0.5 2", "must be finite"),
        ("0 0.5 0.5 1.1 0.2 0.5 0.5 2", "normalized to [0, 1]"),
        ("0 0.5 0.5 0.0 0.2 0.5 0.5 2", "must be positive"),
        ("0 0.5 0.5 0.2 0.2 1.1 0.5 2", "normalized to [0, 1]"),
        ("0 0.5 0.5 0.2 0.2 0.5 0.5 1.5", "visibility"),
        ("0 0.5 0.5 0.2 0.2 0.5 0.5", "Expected 8 fields"),
    ],
)
def test_pose_dataset_rejects_malformed_rows_with_file_and_row_context(
    tmp_path, invalid_row, message
):
    image_path = tmp_path / "sample.jpg"
    label_path = tmp_path / "sample.txt"
    image_path.write_bytes(b"")
    label_path.write_text(
        f"0 0.5 0.5 0.2 0.2 0.5 0.5 2\n{invalid_row}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        YOLOPoseDataset(
            [image_path],
            num_keypoints=1,
            label_files=[label_path],
            num_classes=1,
        )

    assert f"{label_path}:2:" in str(exc_info.value)
    assert message in str(exc_info.value)


def test_semantic_polygon_fallback_rejects_malformed_row_with_context(tmp_path):
    image_path, label_path = _write_detection_sample(
        tmp_path,
        "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n0 0.1 0.1 nan 0.1 0.9 0.9 0.1 0.9\n",
    )
    dataset = SemanticDataset(
        {
            "train": str(image_path.parent),
            "train_img_files": [image_path],
            "names": {0: "object"},
            "nc": 1,
        },
        split="train",
        imgsz=32,
    )

    with pytest.raises(ValueError) as exc_info:
        dataset[0]

    assert f"{label_path}:2:" in str(exc_info.value)
    assert "must be finite" in str(exc_info.value)


def test_pose_dataset_keeps_empty_label_file(tmp_path):
    image_path = tmp_path / "sample.jpg"
    label_path = tmp_path / "sample.txt"
    image_path.write_bytes(b"")
    label_path.write_text("", encoding="utf-8")

    dataset = YOLOPoseDataset(
        [image_path],
        num_keypoints=2,
        label_files=[label_path],
        num_classes=1,
    )

    boxes, classes, keypoints = dataset.labels[0]
    assert boxes.shape == (0, 4)
    assert classes.shape == (0,)
    assert keypoints.shape == (0, 2, 3)
    assert all(array.dtype == np.float32 for array in (boxes, classes, keypoints))


def test_fomo_dataset_rejects_out_of_range_class_with_context(tmp_path):
    from libreyolo.models.fomo.dataset import FOMOYOLODataset

    label_path = tmp_path / "sample.txt"
    label_path.write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    dataset = FOMOYOLODataset([], [], 32, 4, num_classes=1)

    with pytest.raises(ValueError) as exc_info:
        dataset._load_labels(str(label_path))

    assert f"{label_path}:1:" in str(exc_info.value)
    assert "out of range" in str(exc_info.value)
