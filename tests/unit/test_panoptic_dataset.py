"""Tests for the COCO-panoptic dataset loader (issue #555)."""

import json

import numpy as np
import pytest
import yaml
from PIL import Image

from libreyolo.data.panoptic_dataset import (
    VOID_SEGMENT_ID,
    PanopticDataset,
    panoptic_collate_fn,
    resolve_panoptic_data,
    rgb2id,
)

pytestmark = pytest.mark.unit


def _id2rgb(segment_id: int) -> tuple:
    """Inverse of the COCO encoding, for building fixtures."""
    return (segment_id % 256, (segment_id // 256) % 256, segment_id // 65536)


def _build_dataset(tmp_path, *, names=None, categories=None, segments=None):
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "panoptic").mkdir(parents=True)

    Image.new("RGB", (4, 2), (10, 20, 30)).save(root / "images" / "a.jpg")

    # Left half segment id 1, right half id 3226956, plus one void column.
    seg = np.zeros((2, 4, 3), dtype=np.uint8)
    seg[:, 0:2] = _id2rgb(1)
    seg[:, 2:3] = _id2rgb(3226956)
    # column 3 stays (0,0,0) -> VOID
    Image.fromarray(seg).save(root / "panoptic" / "a.png")

    categories = categories or [
        {"id": 1, "name": "person", "isthing": 1},
        {"id": 92, "name": "sky", "isthing": 0},
    ]
    segments = segments or [
        {"id": 1, "category_id": 1, "area": 4, "iscrowd": 0},
        {"id": 3226956, "category_id": 92, "area": 2, "iscrowd": 1},
    ]
    ann = {
        "images": [{"id": 7, "file_name": "a.jpg"}],
        "annotations": [
            {"image_id": 7, "file_name": "a.png", "segments_info": segments}
        ],
        "categories": categories,
    }
    (root / "panoptic.json").write_text(json.dumps(ann), encoding="utf-8")

    cfg = {
        "path": str(root),
        "val": "images",
        "annotations": {"val": "panoptic.json"},
        "panoptic_dir": {"val": "panoptic"},
        "names": names if names is not None else {0: "person", 1: "sky"},
    }
    yaml_path = tmp_path / "panoptic.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return yaml_path


def test_rgb2id_matches_the_coco_encoding():
    # id = R + 256*G + 256*256*B
    rgb = np.array([[[76, 61, 49]]], dtype=np.uint8)
    assert int(rgb2id(rgb)[0, 0]) == 76 + 256 * 61 + 65536 * 49 == 3226956
    assert int(rgb2id(np.zeros((1, 1, 3), dtype=np.uint8))[0, 0]) == VOID_SEGMENT_ID


def test_rgb2id_rejects_non_rgb():
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        rgb2id(np.zeros((2, 2), dtype=np.uint8))


def test_rgb2id_does_not_overflow_uint8_arithmetic():
    """Naive uint8 math would wrap; ids must survive as full integers."""
    rgb = np.array([[[255, 255, 255]]], dtype=np.uint8)
    assert int(rgb2id(rgb)[0, 0]) == 255 + 256 * 255 + 65536 * 255


def test_dataset_loads_segment_map_and_remaps_categories(tmp_path):
    dataset = PanopticDataset(resolve_panoptic_data(_build_dataset(tmp_path)), "val")
    assert len(dataset) == 1
    # 'person'(raw 1) -> train 0, 'sky'(raw 92) -> train 1; things = {0}
    assert dataset.category_id_to_train_id == {1: 0, 92: 1}
    assert dataset.thing_train_ids == {0}
    assert dataset.nc == 2

    img_path, seg_map, segments_info, info = dataset[0]
    assert img_path.endswith("a.jpg")
    assert tuple(seg_map.shape) == (2, 4)
    assert seg_map[0, 0].item() == 1
    assert seg_map[0, 2].item() == 3226956
    assert seg_map[0, 3].item() == VOID_SEGMENT_ID
    assert info["height"] == 2 and info["width"] == 4

    by_id = {s["id"]: s for s in segments_info}
    assert by_id[1]["category_id"] == 0 and by_id[1]["iscrowd"] == 0
    assert by_id[3226956]["category_id"] == 1
    assert by_id[3226956]["iscrowd"] == 1  # crowd flag survives


def test_dataset_rejects_category_missing_from_names(tmp_path):
    """A category the model cannot name would score as a permanent false negative."""
    yaml_path = _build_dataset(tmp_path, names={0: "person"})  # 'sky' missing
    with pytest.raises(ValueError, match="missing from the dataset YAML"):
        PanopticDataset(resolve_panoptic_data(yaml_path), "val")


def test_resolve_requires_annotations_and_panoptic_dir(tmp_path):
    cfg = {"val": "images", "names": {0: "person"}}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="annotations"):
        resolve_panoptic_data(p)

    cfg["annotations"] = {"val": "x.json"}
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="panoptic_dir"):
        resolve_panoptic_data(p)


def test_collate_keeps_per_image_structure(tmp_path):
    dataset = PanopticDataset(resolve_panoptic_data(_build_dataset(tmp_path)), "val")
    paths, maps, infos, meta = panoptic_collate_fn([dataset[0]])
    assert len(paths) == len(maps) == len(infos) == len(meta) == 1
    assert tuple(maps[0].shape) == (2, 4)  # not stacked, not resized
