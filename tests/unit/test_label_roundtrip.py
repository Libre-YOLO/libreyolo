"""Unit tests for LibreLabel v1 (boxes only): the format oracle + dataset R/W."""

import pytest

from libreyolo.label.labelio import (
    format_label_text,
    parse_label_text,
    sanitize_boxes,
)

pytestmark = pytest.mark.unit


def test_cls_serializes_as_integer():
    # The LibreYOLO loader does int(parts[0]); a float token would abort.
    txt = format_label_text([{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.4}])
    first = txt.split()[0]
    assert first == "0"
    int(first)  # must not raise
    parsed, has_non_box = parse_label_text(txt)
    assert has_non_box is False
    assert parsed[0]["cls"] == 0


def test_polygon_line_is_flagged_not_parsed_as_box():
    boxes, has_non_box = parse_label_text("2 0.10 0.10 0.40 0.12 0.25 0.48\n")
    assert boxes == []
    assert has_non_box is True


def test_roundtrip_precision():
    src = [{"cls": 3, "cx": 0.123456, "cy": 0.654321, "w": 0.111111, "h": 0.222222}]
    parsed, _ = parse_label_text(format_label_text(src))
    assert parsed[0]["cls"] == 3
    for k in ("cx", "cy", "w", "h"):
        assert abs(parsed[0][k] - src[0][k]) < 1e-6


def test_empty_list_is_empty_file():
    assert format_label_text([]) == ""


def test_sanitize_clamps_and_drops():
    out = sanitize_boxes(
        [
            {"cls": 0, "cx": 1.4, "cy": -0.2, "w": 0.5, "h": 0.5},  # clamp
            {"cls": 9, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},  # cls >= nc -> drop
            {"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.1},  # zero area -> drop
        ],
        nc=2,
    )
    assert len(out) == 1
    assert out[0]["cx"] == 1.0 and out[0]["cy"] == 0.0


def _make_dataset(root, with_images_in_root=False):
    """Create a minimal YOLO dataset under ``root``; return the data.yaml path."""
    from PIL import Image

    (root / "images" / "train").mkdir(parents=True)
    Image.new("RGB", (20, 10), (123, 80, 200)).save(
        root / "images" / "train" / "a.jpg"
    )
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"  # absent split -> skipped (keeps the fixture at 1 image)
        "nc: 2\n"
        "names:\n  0: cat\n  1: dog\n",
        encoding="utf-8",
    )
    return yaml_path


def test_dataset_roundtrip(tmp_path):
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_dataset(tmp_path)))
    assert len(ds) == 1
    assert ds.writable
    assert ds.names == ["cat", "dog"]

    boxes, editable = ds.read_label(0)
    assert boxes == [] and editable is True

    ds.write_label(0, [{"cls": 1, "cx": 0.5, "cy": 0.5, "w": 0.25, "h": 0.5}])
    lp = tmp_path / "labels" / "train" / "a.txt"
    assert lp.exists()
    assert lp.read_text().strip().split()[0] == "1"  # integer class token

    back, editable = ds.read_label(0)
    assert editable is True
    assert back[0]["cls"] == 1
    assert abs(back[0]["w"] - 0.25) < 1e-6


def test_ambiguous_images_path_is_read_only(tmp_path):
    # A root that itself sits under an 'images' segment => 'images' appears twice
    # in each image path, which would corrupt the labels<->images mapping.
    from libreyolo.label.dataset import DatasetSession

    root = tmp_path / "my" / "images" / "proj"
    yaml_path = _make_dataset(root)
    ds = DatasetSession(str(yaml_path))
    assert ds.writable is False
    with pytest.raises(RuntimeError):
        ds.write_label(0, [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])


def test_assist_class_map_synonyms_and_normalization():
    from libreyolo.label.assist import build_class_map

    resolve = build_class_map(["person", "motorcycle", "potted plant", "tv"])
    assert resolve("person") == 0
    assert resolve("PERSON") == 0  # case-insensitive
    assert resolve("people") == 0  # synonym
    assert resolve("motorbike") == 1  # PASCAL->COCO synonym
    assert resolve("potted plant") == 2  # space normalised to underscore
    assert resolve("pottedplant") == 2  # synonym
    assert resolve("tvmonitor") == 3  # synonym -> tv
    assert resolve("zebra") is None  # not in the dataset -> unmapped (not dropped silently)


def test_assist_engine_never_imports_write_label():
    # The trust contract: the assist module must contain no write path to disk.
    import inspect

    from libreyolo.label import assist

    assert "write_label" not in inspect.getsource(assist)


def test_annotation_parse_format_roundtrip():
    from libreyolo.label.labelio import (
        format_annotations,
        parse_annotations,
        sanitize_annotations,
    )

    text = "0 0.5 0.5 0.2 0.2\n1 0.10 0.10 0.40 0.10 0.25 0.48\n"
    anns = parse_annotations(text)
    assert anns[0]["type"] == "box" and anns[0]["cls"] == 0
    assert anns[1]["type"] == "poly" and len(anns[1]["points"]) == 6
    assert format_annotations(anns).splitlines()[0].split()[0] == "0"  # integer cls
    # a 2-vertex "polygon" (4 nums) is invalid and dropped
    assert sanitize_annotations([{"type": "poly", "cls": 0, "points": [0.1, 0.1, 0.2, 0.2]}], nc=2) == []


def test_polygon_roundtrip(tmp_path):
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_dataset(tmp_path)))
    ds.write_label(0, [
        {"type": "poly", "cls": 1, "points": [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]},
        {"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
    ])
    anns, editable = ds.read_label(0)
    assert editable is True and len(anns) == 2
    poly = [a for a in anns if a["type"] == "poly"][0]
    box = [a for a in anns if a["type"] == "box"][0]
    assert poly["cls"] == 1 and len(poly["points"]) == 8
    assert box["cls"] == 0 and abs(box["w"] - 0.2) < 1e-6
    lines = (tmp_path / "labels" / "train" / "a.txt").read_text().strip().splitlines()
    assert any(len(ln.split()) == 9 for ln in lines)  # polygon line (cls + 8 coords)
    assert any(len(ln.split()) == 5 for ln in lines)  # box line
