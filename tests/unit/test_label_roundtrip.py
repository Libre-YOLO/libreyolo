"""Unit tests for LibreLabel v1 (boxes only): the format oracle + dataset R/W."""

import os

import pytest

from pathlib import Path

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


def test_sanitize_rejects_invalid_box_instead_of_clamping_or_dropping():
    with pytest.raises(ValueError, match="normalized"):
        sanitize_boxes(
            [{"cls": 0, "cx": 1.4, "cy": -0.2, "w": 0.5, "h": 0.5}],
            nc=2,
        )
    with pytest.raises(ValueError, match="outside"):
        sanitize_boxes(
            [{"cls": 9, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
            nc=2,
        )
    with pytest.raises(ValueError, match="positive"):
        sanitize_boxes(
            [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.1}],
            nc=2,
        )


def _make_dataset(root, with_images_in_root=False, task=None):
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
        + (f"task: {task}\n" if task else "")
        + "names:\n  0: cat\n  1: dog\n",
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
    # A 2-vertex "polygon" is rejected; a save never silently drops a shape.
    with pytest.raises(ValueError, match="3 coordinate"):
        sanitize_annotations(
            [{"type": "poly", "cls": 0, "points": [0.1, 0.1, 0.2, 0.2]}],
            nc=2,
        )


def test_polygon_roundtrip(tmp_path):
    from libreyolo.label.dataset import DatasetSession

    # 9-field rows are oriented-box/polygon ambiguous; the dataset must declare
    # task: segment for read_label to treat them as editable polygons.
    ds = DatasetSession(str(_make_dataset(tmp_path, task="segment")))
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


def test_obb_roundtrip(tmp_path):
    from libreyolo.label.dataset import DatasetSession

    # task: obb makes 9-field rows editable too — an oriented box is a 4-corner quad,
    # stored/loaded as a 4-vertex polygon byte-identically (no corruption).
    ds = DatasetSession(str(_make_dataset(tmp_path, task="obb")))
    quad = [0.2, 0.1, 0.5, 0.2, 0.4, 0.5, 0.1, 0.4]   # a rotated rectangle's 4 corners
    ds.write_label(0, [{"type": "poly", "cls": 1, "points": quad}])
    anns, editable = ds.read_label(0)
    assert editable is True and len(anns) == 1
    assert anns[0]["type"] == "poly" and len(anns[0]["points"]) == 8
    lines = (tmp_path / "labels" / "train" / "a.txt").read_text().strip().splitlines()
    assert len(lines) == 1 and len(lines[0].split()) == 9  # cls + 8 coords (oriented box)


def test_upload_never_overwrites(tmp_path):
    # Uploads run BEFORE the create-project guard, so they must refuse to touch
    # an existing dataset or an existing file (labels would describe wrong pixels).
    from libreyolo.label.dataset import save_uploaded_image

    dst = tmp_path / "proj"
    save_uploaded_image(str(dst), "a.jpg", b"xx")
    with pytest.raises(FileExistsError):
        save_uploaded_image(str(dst), "a.jpg", b"yy")
    assert (dst / "images" / "train" / "a.jpg").read_bytes() == b"xx"

    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "data.yaml").write_text("train: .\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_uploaded_image(str(taken), "b.jpg", b"zz")


def _make_nested_dataset(tmp_path):
    # Two same-named images in different subfolders (train: . scans recursively).
    from PIL import Image

    root = tmp_path / "ds"
    for sub in ("a", "b"):
        (root / sub).mkdir(parents=True)
        Image.new("RGB", (16, 12), (60, 60, 60)).save(root / sub / "0001.jpg")
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: .\ntask: detect\nnc: 1\nnames:\n  0: thing\n",
        encoding="utf-8",
    )
    return root


def test_export_uniquifies_same_basenames(tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.export import export_dataset

    root = _make_nested_dataset(tmp_path)
    ds = DatasetSession(str(root / "data.yaml"))
    ds.write_label(0, [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5}])
    ds.write_label(1, [{"cls": 0, "cx": 0.25, "cy": 0.25, "w": 0.2, "h": 0.2}])

    out = tmp_path / "out"
    res = export_dataset(ds, dst=str(out), formats=("yolo",), split="none")
    assert res["counts"]["train"] == 2
    imgs = sorted(p.name for p in (out / "images" / "train").iterdir())
    lbls = sorted(p.name for p in (out / "labels" / "train").iterdir())
    assert len(imgs) == 2 and len(set(imgs)) == 2      # no silent overwrite
    assert {p.rsplit(".", 1)[0] for p in imgs} == {p.rsplit(".", 1)[0] for p in lbls}


def test_export_refuses_nonempty_destination(tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.export import export_dataset

    root = _make_nested_dataset(tmp_path)
    ds = DatasetSession(str(root / "data.yaml"))
    out = tmp_path / "out"
    export_dataset(ds, dst=str(out), formats=("yolo",), split="none")
    with pytest.raises(ValueError, match="not empty"):
        export_dataset(ds, dst=str(out), formats=("yolo",), split="none")


def test_inplace_export_uniquifies_same_basenames(tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.export import export_dataset

    root = _make_nested_dataset(tmp_path)
    ds = DatasetSession(str(root / "data.yaml"))
    res = export_dataset(ds, in_place=True, split="none")
    assert res["in_place"] is True
    imgs = sorted(p.name for p in (root / "images" / "train").iterdir())
    assert len(imgs) == 2 and len(set(imgs)) == 2      # both survived the flatten


def test_shared_label_file_makes_session_read_only(tmp_path):
    # a.jpg and a.png both derive labels/a.txt: saving one would clobber the other.
    from PIL import Image

    from libreyolo.label.dataset import DatasetSession

    root = tmp_path / "ds"
    root.mkdir()
    Image.new("RGB", (16, 12), (10, 10, 10)).save(root / "a.jpg")
    Image.new("RGB", (16, 12), (20, 20, 20)).save(root / "a.png")
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: .\nnc: 1\nnames:\n  0: thing\n",
        encoding="utf-8",
    )
    ds = DatasetSession(str(root / "data.yaml"))
    assert ds.writable is False
    assert "same label file" in ds.reason
    with pytest.raises(RuntimeError):
        ds.write_label(0, [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])


def test_shared_label_file_detection_is_casefolded(tmp_path):
    from PIL import Image

    from libreyolo.label.dataset import DatasetSession

    root = tmp_path / "ds"
    (root / "images" / "train").mkdir(parents=True)
    Image.new("RGB", (16, 12), (10, 10, 10)).save(
        root / "images" / "train" / "foo.jpg"
    )
    Image.new("RGB", (16, 12), (20, 20, 20)).save(
        root / "images" / "train" / "FOO.png"
    )
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\ntask: detect\n"
        "nc: 1\nnames:\n  0: thing\n",
        encoding="utf-8",
    )

    session = DatasetSession(str(root / "data.yaml"))

    assert session.writable is False
    assert "same label file" in session.reason


def test_case_distinct_posix_items_are_rejected_instead_of_omitted(tmp_path):
    from PIL import Image

    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.export import export_dataset

    if os.name == "nt":
        pytest.skip("Windows paths are case-insensitive")
    root = tmp_path / "ds"
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    upper = images / "A.jpg"
    lower = images / "a.jpg"
    Image.new("RGB", (16, 12), (10, 10, 10)).save(upper)
    Image.new("RGB", (16, 12), (20, 20, 20)).save(lower)
    if upper.samefile(lower):
        pytest.skip("the test filesystem is case-insensitive")
    (labels / "A.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels / "a.txt").write_text("0 0.4 0.4 0.2 0.2\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\ntask: detect\n"
        "nc: 1\nnames:\n  0: thing\n",
        encoding="utf-8",
    )

    session = DatasetSession(str(root / "data.yaml"))

    assert len(session._items) == 2
    assert session._label_clash is True
    assert session.writable is False
    with pytest.raises(ValueError, match="share one derived label file"):
        export_dataset(session, dst=str(tmp_path / "export"))


def test_wizard_segment_project_polygon_roundtrip(tmp_path):
    # The wizard can now create task: segment projects; polygons must round-trip.
    from PIL import Image

    from libreyolo.label.dataset import DatasetSession, create_uploaded_project

    dst = tmp_path / "segproj"
    (dst / "images" / "train").mkdir(parents=True)
    Image.new("RGB", (32, 24), (40, 40, 40)).save(dst / "images" / "train" / "a.jpg")
    yaml_path = create_uploaded_project(str(dst), name="Seg", classes=["blob"], task="segment")
    ds = DatasetSession(yaml_path)
    assert ds.meta()["task"] == "segment"
    assert ds.writable
    ds.write_label(0, [{"type": "poly", "cls": 0,
                        "points": [0.1, 0.1, 0.6, 0.15, 0.4, 0.7]}])
    back, editable = ds.read_label(0)
    assert editable and back[0]["type"] == "poly" and len(back[0]["points"]) == 6


def _tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_linked_project_never_touches_source(tmp_path):
    from PIL import Image

    from libreyolo.label.dataset import DatasetSession, create_linked_project

    src = tmp_path / "photos"
    for sub in ("a", "b"):
        (src / sub).mkdir(parents=True)
        Image.new("RGB", (16, 12), (70, 70, 70)).save(src / sub / "0001.jpg")
    before = _tree(src)

    yaml_path = create_linked_project(
        str(src), name="My photos", classes=["cat"],
        projects_dir=str(tmp_path / "managed"))
    assert Path(yaml_path).is_relative_to(tmp_path / "managed")

    ds = DatasetSession(yaml_path)
    meta = ds.meta()
    assert meta["linked"] is True and meta["name"] == "My photos"
    assert ds.writable and len(ds) == 2

    ds.write_label(0, [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4}])
    ds.write_label(1, [{"cls": 0, "cx": 0.3, "cy": 0.3, "w": 0.2, "h": 0.2}])
    # labels landed in the managed dir with unique names, source untouched
    labs = list((Path(yaml_path).parent / "labels" / "train").glob("*.txt"))
    assert len(labs) == 2 and len({p.name for p in labs}) == 2
    assert _tree(src) == before

    back, editable = ds.read_label(0)
    assert editable and back[0]["cls"] == 0


def test_linked_project_refuses_source_mutations(tmp_path):
    from PIL import Image

    from libreyolo.label.dataset import DatasetSession, create_linked_project
    from libreyolo.label.export import export_dataset

    src = tmp_path / "photos"
    src.mkdir()
    for i in range(2):
        Image.new("RGB", (16, 12), (i * 40,) * 3).save(src / f"im{i}.jpg")
    yaml_path = create_linked_project(str(src), classes=["cat"],
                                      projects_dir=str(tmp_path / "managed"))
    ds = DatasetSession(yaml_path)

    with pytest.raises(RuntimeError, match="linked"):
        ds.resolve_duplicates([0, 1])
    with pytest.raises(ValueError, match="[Ll]inked"):
        export_dataset(ds, in_place=True, split="none")

    # ...but a copy export produces a normal trainable dataset
    out = tmp_path / "out"
    res = export_dataset(ds, dst=str(out), formats=("yolo",), split="none")
    assert res["counts"]["train"] == 2
    assert (out / "data.yaml").exists()


def test_sidecar_settings_roundtrip(tmp_path):
    from libreyolo.label.dataset import DatasetSession, update_sidecar

    yaml_path = _make_dataset(tmp_path)
    update_sidecar(str(yaml_path), name="Cats", description="cat photos",
                   instructions="tight boxes only")
    meta = DatasetSession(str(yaml_path)).meta()
    assert meta["name"] == "Cats"
    assert meta["description"] == "cat photos"
    assert meta["instructions"] == "tight boxes only"
