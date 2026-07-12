"""LibreLabel data-layer integrity and rollback contracts."""

from __future__ import annotations

import json
import math
import multiprocessing
import queue
import threading
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from libreyolo.label.dataset import (
    DatasetSession,
    create_uploaded_project,
    save_uploaded_image,
)
from libreyolo.label.export import _assign_splits, export_dataset
from libreyolo.label.labelio import (
    format_annotations,
    parse_annotations,
    sanitize_annotations,
    sanitize_boxes,
)

pytestmark = pytest.mark.unit


def _process_upload(root, name, payload, ready, start, outcomes):
    """Spawn-safe worker for the cross-process stem reservation regression."""
    ready.put(True)
    start.wait(10)
    try:
        save_uploaded_image(root, name, payload)
    except Exception as exc:  # noqa: BLE001 - serialized test outcome
        outcomes.put(("error", type(exc).__name__))
    else:
        outcomes.put(("ok", name))


def _process_label_write(yaml_path, cx, ready, start, attempting, outcomes):
    """Spawn-safe worker for the cross-process label CAS regression."""
    session = DatasetSession(yaml_path)
    revision = session.label_rev(0)
    ready.put(revision)
    start.wait(10)
    attempting.put(True)
    annotation = {
        "type": "box",
        "cls": 0,
        "cx": cx,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
    }
    try:
        session.write_label(0, [annotation], expected_rev=revision)
    except Exception as exc:  # noqa: BLE001 - serialized test outcome
        outcomes.put(("error", type(exc).__name__, str(exc)))
    else:
        outcomes.put(("ok", cx, ""))


def _process_sidecar_update(root, key, value, ready, start, loaded, release, outcomes):
    """Force competing sidecar updates to overlap when no process lock exists."""
    import libreyolo.label.dataset as dataset_module

    original_load = dataset_module.load_sidecar

    def synchronized_load(base):
        result = original_load(base)
        loaded.put(key)
        release.wait(10)
        return result

    dataset_module.load_sidecar = synchronized_load
    ready.put(True)
    start.wait(10)
    try:
        dataset_module.update_sidecar(root, **{key: value})
    except Exception as exc:  # noqa: BLE001 - serialized test outcome
        outcomes.put(("error", key, type(exc).__name__, str(exc)))
    else:
        outcomes.put(("ok", key))


def _dataset(root: Path, names=("cat", "dog"), task: str | None = "detect") -> DatasetSession:
    for subdir, filename, color in (
        ("a", "same.jpg", "red"),
        ("b", "same.png", "blue"),
    ):
        image = root / "images" / "train" / subdir / filename
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 16), color).save(image)
    lines = [
        f"path: {root.as_posix()}",
        "train: images/train",
        f"nc: {len(names)}",
        "names:",
        *(f"  {i}: {name}" for i, name in enumerate(names)),
    ]
    if task:
        lines.append(f"task: {task}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return DatasetSession(str(root / "data.yaml"))


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _directory_snapshot(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
    }


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        ({"cls": 0.5, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}, "integer"),
        ({"cls": 2, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}, "outside"),
        ({"cls": 0, "cx": math.nan, "cy": 0.5, "w": 0.2, "h": 0.2}, "finite"),
        ({"cls": 0, "cx": 1.1, "cy": 0.5, "w": 0.2, "h": 0.2}, "normalized"),
        ({"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.2}, "positive"),
        ({"type": "poly", "cls": 0, "points": [0.1, 0.1, 0.2, 0.2]}, "3 coordinate"),
        ({"type": "poly", "cls": 0, "points": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]}, "non-zero"),
    ],
)
def test_annotation_ingress_rejects_instead_of_truncating_or_dropping(annotation, message):
    with pytest.raises(ValueError, match=message):
        sanitize_annotations([annotation], nc=2)


def test_box_ingress_rejects_out_of_bounds_and_nonfinite():
    with pytest.raises(ValueError, match="normalized"):
        sanitize_boxes([{"cls": 0, "cx": -0.1, "cy": 0.5, "w": 0.2, "h": 0.2}], nc=1)
    with pytest.raises(ValueError, match="finite"):
        sanitize_boxes([{"cls": 0, "cx": 0.5, "cy": 0.5, "w": math.inf, "h": 0.2}], nc=1)
    with pytest.raises(ValueError, match="six-decimal precision"):
        sanitize_boxes([{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 1e-7, "h": 0.2}], nc=1)


def test_annotation_ingress_rejects_geometry_lost_by_disk_quantization():
    with pytest.raises(ValueError, match="six-decimal precision"):
        sanitize_annotations(
            [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 1e-7, "h": 0.2}],
            nc=1,
            task="detect",
        )
    with pytest.raises(ValueError, match="non-zero area|distinct points"):
        sanitize_annotations(
            [{
                "type": "poly",
                "cls": 0,
                "points": [0.1, 0.1, 0.9, 0.1, 0.9, 0.1000001, 0.1, 0.1000001],
            }],
            nc=1,
            task="obb",
        )


def test_write_rejects_subprecision_geometry_without_publishing_label(tmp_path):
    detect = _dataset(tmp_path / "detect", task="detect")
    with pytest.raises(ValueError, match="six-decimal precision"):
        detect.write_label(
            0,
            [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 1e-7, "h": 0.2}],
        )
    assert not detect.has_label_file(0)

    obb = _dataset(tmp_path / "obb-subprecision", task="obb")
    with pytest.raises(ValueError, match="non-zero area|distinct points"):
        obb.write_label(
            0,
            [{
                "type": "poly",
                "cls": 0,
                "points": [0.25, 0.5, 0.75, 0.5, 0.75, 0.5000001, 0.25, 0.5000001],
            }],
        )
    assert not obb.has_label_file(0)


def test_write_rejects_whole_payload_without_changing_existing_label(tmp_path):
    session = _dataset(tmp_path / "ds")
    good = {"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
    session.write_label(0, [good])
    label = session._items[0][1]
    before = label.read_bytes()

    with pytest.raises(ValueError, match="class id"):
        session.write_label(0, [good, {**good, "cls": 0.25}])

    assert label.read_bytes() == before


def test_segment_polygon_is_clipped_to_persisted_canvas(tmp_path):
    session = _dataset(tmp_path / "seg", task="segment")
    session.write_label(
        0,
        [{"type": "poly", "cls": 0, "points": [-0.2, 0.2, 0.4, 0.2, 0.4, 1.2]}],
    )

    annotations, editable = session.read_label(0)
    assert editable
    points = annotations[0]["points"]
    assert len(points) >= 6
    assert all(0.0 <= value <= 1.0 for value in points)


def test_segment_rejects_diagonal_polygon_outside_canvas_corner(tmp_path):
    session = _dataset(tmp_path / "segment", task="segment")
    points = [-0.45, 0.35, 0.35, -0.45, 0.36, -0.44, -0.44, 0.36]

    with pytest.raises(ValueError, match="does not overlap"):
        session.write_label(
            0, [{"type": "poly", "cls": 0, "points": points}]
        )


def test_obb_clipping_translates_rectangle_without_shearing(tmp_path):
    session = _dataset(tmp_path / "obb", task="obb")
    session.write_label(
        0,
        [{
            "type": "poly",
            "cls": 0,
            "points": [-0.1, 0.2, 0.3, 0.2, 0.3, 0.6, -0.1, 0.6],
        }],
    )

    annotations, editable = session.read_label(0)
    assert editable
    assert annotations[0]["points"] == pytest.approx([0.0, 0.2, 0.4, 0.2, 0.4, 0.6, 0.0, 0.6])


def test_obb_rejects_degenerate_quad(tmp_path):
    session = _dataset(tmp_path / "obb", task="obb")
    with pytest.raises(ValueError, match="non-zero area"):
        session.write_label(
            0,
            [{"type": "poly", "cls": 0, "points": [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7]}],
        )
    assert not session.has_label_file(0)


def test_obb_rejects_axis_aligned_box_rows(tmp_path):
    session = _dataset(tmp_path / "obb", task="obb")
    with pytest.raises(ValueError, match="4 corners"):
        session.write_label(
            0,
            [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.2}],
        )


def test_obb_rejects_geometry_entirely_outside_canvas(tmp_path):
    session = _dataset(tmp_path / "obb", task="obb")
    with pytest.raises(ValueError, match="does not overlap"):
        session.write_label(
            0,
            [{
                "type": "poly",
                "cls": 0,
                "points": [-0.9, 0.2, -0.5, 0.2, -0.5, 0.6, -0.9, 0.6],
            }],
        )


def test_obb_rejects_diagonal_geometry_outside_canvas_corner(tmp_path):
    session = _dataset(tmp_path / "obb", task="obb")
    cx = cy = -0.05
    length, height = 0.8, 0.02
    inv = math.sqrt(0.5)
    ux, uy = inv * length / 2, -inv * length / 2
    vx, vy = inv * height / 2, inv * height / 2
    corners = [
        cx - ux - vx, cy - uy - vy,
        cx + ux - vx, cy + uy - vy,
        cx + ux + vx, cy + uy + vy,
        cx - ux + vx, cy - uy + vy,
    ]

    with pytest.raises(ValueError, match="does not overlap"):
        session.write_label(
            0, [{"type": "poly", "cls": 0, "points": corners}]
        )


def test_obb_rejects_non_rectangular_quad(tmp_path):
    session = _dataset(tmp_path / "obb", task="obb")
    with pytest.raises(ValueError, match="oriented rectangle"):
        session.write_label(
            0,
            [{
                "type": "poly",
                "cls": 0,
                "points": [0.1, 0.1, 0.8, 0.1, 0.7, 0.7, 0.2, 0.7],
            }],
        )


def test_rotated_obb_survives_six_decimal_disk_quantization():
    cx, cy, width, height, angle = 0.5, 0.5, 0.2, 0.08, 0.37
    ux, uy = math.cos(angle) * width / 2, math.sin(angle) * width / 2
    vx, vy = -math.sin(angle) * height / 2, math.cos(angle) * height / 2
    corners = [
        cx - ux - vx, cy - uy - vy,
        cx + ux - vx, cy + uy - vy,
        cx + ux + vx, cy + uy + vy,
        cx - ux + vx, cy - uy + vy,
    ]
    text = format_annotations([{"type": "poly", "cls": 0, "points": corners}])

    clean = sanitize_annotations(parse_annotations(text), nc=1, task="obb")

    assert len(clean) == 1


def test_thin_rotated_obb_survives_six_decimal_disk_quantization():
    cx, cy, width, height, angle = 0.5, 0.5, 0.5, 0.001, 0.37
    ux, uy = math.cos(angle) * width / 2, math.sin(angle) * width / 2
    vx, vy = -math.sin(angle) * height / 2, math.cos(angle) * height / 2
    corners = [
        cx - ux - vx, cy - uy - vy,
        cx + ux - vx, cy + uy - vy,
        cx + ux + vx, cy + uy + vy,
        cx - ux + vx, cy - uy + vy,
    ]
    text = format_annotations([{"type": "poly", "cls": 0, "points": corners}])

    clean = sanitize_annotations(parse_annotations(text), nc=1, task="obb")

    assert len(clean) == 1


def test_export_uniquifies_stems_across_image_suffixes(tmp_path):
    session = _dataset(tmp_path / "source", task="detect")
    session.write_label(0, [{"cls": 0, "cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2}])
    session.write_label(1, [{"cls": 1, "cx": 0.6, "cy": 0.6, "w": 0.3, "h": 0.3}])

    out = tmp_path / "out"
    export_dataset(session, dst=str(out), formats=("yolo", "voc"), split="none")

    images = sorted((out / "images" / "train").iterdir())
    labels = sorted((out / "labels" / "train").iterdir())
    xml = sorted((out / "voc" / "train" / "Annotations").iterdir())
    assert [path.stem.casefold() for path in images] == ["same", "same_2"]
    assert {path.stem.casefold() for path in labels} == {"same", "same_2"}
    assert {path.stem.casefold() for path in xml} == {"same", "same_2"}
    assert {path.read_text(encoding="utf-8").split()[0] for path in labels} == {"0", "1"}


def test_export_rejects_source_images_that_already_share_one_label_path(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    Image.new("RGB", (12, 8), "red").save(root / "same.jpg")
    Image.new("RGB", (12, 8), "blue").save(root / "same.png")
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: .\nnc: 1\nnames:\n  0: thing\n",
        encoding="utf-8",
    )
    session = DatasetSession(str(root / "data.yaml"))

    with pytest.raises(ValueError, match="share one derived label"):
        export_dataset(session, dst=str(tmp_path / "out"), formats=("yolo",), split="none")

    assert not (tmp_path / "out").exists()


def test_native_coco_project_is_view_only_and_cannot_export_empty_labels(tmp_path):
    root = tmp_path / "native"
    images = root / "images" / "train"
    annotations = root / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    Image.new("RGB", (12, 8), "red").save(images / "a.jpg")
    (annotations / "train.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 12, "height": 8}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 3]}
                ],
                "categories": [{"id": 1, "name": "thing"}],
            }
        ),
        encoding="utf-8",
    )
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\n"
        "annotations:\n  train: annotations/train.json\n"
        "names:\n  0: thing\nnc: 1\n",
        encoding="utf-8",
    )
    session = DatasetSession(str(root / "data.yaml"))

    assert session.writable is False
    with pytest.raises(ValueError, match="cannot preserve"):
        export_dataset(session, dst=str(tmp_path / "out"), formats=("yolo",))
    assert not (tmp_path / "out").exists()


def test_taskless_dataset_rejects_transforming_exports(tmp_path):
    session = _dataset(tmp_path / "source", task=None)
    with pytest.raises(ValueError, match="does not declare a task"):
        export_dataset(session, dst=str(tmp_path / "coco"), formats=("coco",))
    with pytest.raises(ValueError, match="detection annotations require boxes"):
        session.write_label(
            0,
            [{
                "type": "poly",
                "cls": 0,
                "points": [0.1, 0.1, 0.8, 0.1, 0.4, 0.7],
            }],
        )

    with pytest.raises(ValueError, match="does not declare a task"):
        export_dataset(
            session, dst=str(tmp_path / "yolo"), formats=("yolo",), split="none"
        )


def test_taskless_nonquad_polygons_infer_segment_and_reject_voc(tmp_path):
    root = tmp_path / "source"
    initial = _dataset(root, task=None)
    label = initial._items[0][1]
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("0 0.1 0.1 0.8 0.1 0.4 0.7\n", encoding="utf-8")
    session = DatasetSession(str(root / "data.yaml"))

    assert session.meta()["task"] == "segment"
    with pytest.raises(ValueError, match="discard"):
        export_dataset(session, dst=str(tmp_path / "voc"), formats=("voc",))


def test_taskless_quad_dataset_is_globally_view_only(tmp_path):
    root = tmp_path / "source"
    initial = _dataset(root, task=None)
    label = initial._items[0][1]
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text(
        "0 0.1 0.1 0.7 0.1 0.7 0.6 0.1 0.6\n", encoding="utf-8"
    )
    session = DatasetSession(str(root / "data.yaml"))

    assert session._task_ambiguous is True
    assert session.writable is False
    assert session.read_label(1)[1] is False
    with pytest.raises(RuntimeError, match="ambiguous"):
        session.write_label(
            1,
            [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
        )


@pytest.mark.parametrize(
    ("task", "label_text"),
    [
        ("detect", "0 0.1 0.1 0.8 0.1 0.4 0.7\n"),
        ("obb", "0 0.5 0.5 0.2 0.2\n"),
    ],
)
def test_declared_task_mismatch_is_read_only(tmp_path, task, label_text):
    root = tmp_path / task
    initial = _dataset(root, task=task)
    label = initial._items[0][1]
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text(label_text, encoding="utf-8")
    session = DatasetSession(str(root / "data.yaml"))

    annotations, editable = session.read_label(0)
    assert annotations
    assert editable is False
    with pytest.raises(RuntimeError, match="read-only"):
        session.write_label(0, annotations)


def test_segment_coco_export_preserves_rectangular_mask(tmp_path):
    session = _dataset(tmp_path / "source", task="segment")
    session.write_label(0, [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5}])

    out = tmp_path / "out"
    export_dataset(session, dst=str(out), formats=("coco",), split="none")
    coco = json.loads((out / "annotations" / "instances_train.json").read_text(encoding="utf-8"))

    assert len(coco["annotations"][0]["segmentation"][0]) == 8
    assert coco["annotations"][0]["area"] > 0


@pytest.mark.parametrize(
    ("task", "formats", "message"),
    [
        ("segment", ("voc",), "discard"),
        ("obb", ("coco",), "discard"),
        ("pose", ("yolo",), "cannot preserve"),
        ("semantic", ("yolo",), "cannot preserve"),
    ],
)
def test_export_rejects_lossy_task_format_pairs(tmp_path, task, formats, message):
    session = _dataset(tmp_path / task, task=task)
    if task == "pose":
        session._lossy_export = True
    with pytest.raises(ValueError, match=message):
        export_dataset(session, dst=str(tmp_path / "out"), formats=formats, split="none")
    assert not (tmp_path / "out").exists()


def test_export_rejects_unknown_format_and_inplace_ignored_options(tmp_path):
    session = _dataset(tmp_path / "source")
    with pytest.raises(ValueError, match="Unsupported"):
        export_dataset(session, dst=str(tmp_path / "out"), formats=("bogus",), split="none")
    with pytest.raises(ValueError, match="YOLO only"):
        export_dataset(session, formats=("coco",), split="none", in_place=True)
    with pytest.raises(ValueError, match="cannot be zipped"):
        export_dataset(session, formats=("yolo",), split="none", in_place=True, make_zip=True)


def test_copy_export_failure_leaves_new_destination_absent(tmp_path, monkeypatch):
    from libreyolo.label import export as export_module

    session = _dataset(tmp_path / "source")
    real_copy = export_module.shutil.copy2
    calls = 0

    def fail_second_copy(src, dst, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected copy failure")
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(export_module.shutil, "copy2", fail_second_copy)
    out = tmp_path / "out"
    with pytest.raises(OSError, match="injected"):
        export_dataset(session, dst=str(out), formats=("yolo",), split="none")

    assert not out.exists()
    assert not list(tmp_path.glob(".out-librelabel-*"))


def test_copy_export_rejects_destination_inside_recursive_source(tmp_path):
    root = tmp_path / "source"
    session = _dataset(root, task="detect")
    destination = root / "images" / "train" / "export"

    with pytest.raises(ValueError, match="inside recursive source split"):
        export_dataset(
            session, dst=str(destination), formats=("yolo",), split="none"
        )

    assert not destination.exists()
    assert len(DatasetSession(str(root / "data.yaml"))) == len(session)


@pytest.mark.parametrize("stale_kind", ["image", "label"])
def test_in_place_export_rejects_unrelated_target_content(tmp_path, stale_kind):
    root = tmp_path / "source"
    session = _dataset(root, task="detect")
    if stale_kind == "image":
        stale = root / "images" / "val" / "stale.jpg"
        stale.parent.mkdir(parents=True)
        Image.new("RGB", (12, 8), "black").save(stale)
    else:
        stale = root / "labels" / "val" / "stale.txt"
        stale.parent.mkdir(parents=True)
        stale.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="unrelated"):
        export_dataset(session, formats=("yolo",), split="none", in_place=True)

    assert stale.exists()


def test_copy_export_failure_preserves_preexisting_empty_destination(tmp_path, monkeypatch):
    from libreyolo.label import export as export_module

    session = _dataset(tmp_path / "source")
    out = tmp_path / "out"
    out.mkdir()

    def fail_yaml(*args, **kwargs):
        raise OSError("injected yaml failure")

    monkeypatch.setattr(export_module, "_write_yaml", fail_yaml)
    with pytest.raises(OSError, match="injected"):
        export_dataset(session, dst=str(out), formats=("yolo",), split="none")

    assert out.is_dir() and not any(out.iterdir())


def test_copy_export_zip_uses_final_root_and_yaml_path(tmp_path):
    import yaml

    session = _dataset(tmp_path / "source")
    out = tmp_path / "out"
    result = export_dataset(
        session, dst=str(out), formats=("yolo",), split="none", make_zip=True
    )

    config = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert Path(config["path"]) == out.absolute()
    with zipfile.ZipFile(result["zip"]) as archive:
        assert archive.namelist()
        assert all(name.startswith("out/") for name in archive.namelist())


def test_explicit_detect_export_remains_reexportable(tmp_path):
    source = _dataset(tmp_path / "source", task="detect")
    first = tmp_path / "first"
    export_dataset(source, dst=str(first), formats=("yolo",), split="none")

    reopened = DatasetSession(str(first / "data.yaml"))
    second = tmp_path / "second"
    export_dataset(reopened, dst=str(second), formats=("yolo",), split="none")

    assert reopened._task == "detect"
    assert (second / "data.yaml").exists()


def test_zip_publication_race_rolls_back_dataset_without_overwrite(tmp_path, monkeypatch):
    from libreyolo.label import export as export_module

    session = _dataset(tmp_path / "source")
    out = tmp_path / "out"
    zip_target = Path(str(out.absolute()) + ".zip")
    real_zip = export_module._zip_dir

    def race_zip(*args, **kwargs):
        real_zip(*args, **kwargs)
        zip_target.write_bytes(b"other exporter")

    monkeypatch.setattr(export_module, "_zip_dir", race_zip)
    with pytest.raises(FileExistsError):
        export_dataset(
            session, dst=str(out), formats=("yolo",), split="none", make_zip=True
        )

    assert not out.exists()
    assert zip_target.read_bytes() == b"other exporter"


def test_zip_cleanup_failure_does_not_report_failure_after_publication(
    tmp_path, monkeypatch
):
    session = _dataset(tmp_path / "source")
    out = tmp_path / "out"
    real_unlink = Path.unlink

    def fail_zip_temp_cleanup(path, *args, **kwargs):
        if path.name.endswith(".zip.tmp"):
            raise OSError("injected cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_zip_temp_cleanup)
    result = export_dataset(
        session, dst=str(out), formats=("yolo",), split="none", make_zip=True
    )

    assert out.is_dir()
    assert Path(result["zip"]).is_file()


def test_zip_publication_does_not_require_hard_links(tmp_path, monkeypatch):
    from libreyolo.label import export as export_module

    session = _dataset(tmp_path / "source")
    monkeypatch.setattr(
        export_module.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unsupported")),
    )

    result = export_dataset(
        session,
        dst=str(tmp_path / "out"),
        formats=("yolo",),
        split="none",
        make_zip=True,
    )

    assert Path(result["zip"]).is_file()


def test_in_place_export_rolls_back_all_moves_on_mid_commit_failure(tmp_path, monkeypatch):
    from libreyolo.label import export as export_module

    root = tmp_path / "source"
    session = _dataset(root)
    session.write_label(0, [{"cls": 0, "cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2}])
    session.write_label(1, [{"cls": 1, "cx": 0.6, "cy": 0.6, "w": 0.3, "h": 0.3}])
    before = _file_snapshot(root)
    real_move = export_module.shutil.move
    calls = 0

    def fail_once(src, dst, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise OSError("injected finalization failure")
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(export_module.shutil, "move", fail_once)
    with pytest.raises(OSError, match="injected"):
        export_dataset(session, formats=("yolo",), split="none", in_place=True)

    assert _file_snapshot(root) == before
    assert not list(tmp_path.glob(".source-libreyolo-export-*"))


def test_in_place_export_rolls_back_when_yaml_commit_fails(tmp_path, monkeypatch):
    from libreyolo.label import export as export_module

    root = tmp_path / "source"
    session = _dataset(root)
    session.write_label(0, [{"cls": 0, "cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2}])
    before = _file_snapshot(root)

    def fail_yaml(*args, **kwargs):
        raise OSError("injected yaml failure")

    monkeypatch.setattr(export_module, "_write_yaml", fail_yaml)
    with pytest.raises(OSError, match="injected"):
        export_dataset(session, formats=("yolo",), split="none", in_place=True)

    assert _file_snapshot(root) == before


def test_in_place_export_rolls_back_when_reopened_session_is_invalid(tmp_path):
    root = tmp_path / "source"
    session = _dataset(root, task="detect")
    session.write_label(
        0, [{"cls": 0, "cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2}]
    )
    before = _file_snapshot(root)
    before_dirs = _directory_snapshot(root)

    def reject_reopen(_yaml_path):
        raise RuntimeError("injected reopen failure")

    with pytest.raises(RuntimeError, match="injected reopen"):
        export_dataset(
            session,
            formats=("yolo",),
            split="trainval",
            val_frac=0.5,
            in_place=True,
            _in_place_validator=reject_reopen,
        )

    assert _file_snapshot(root) == before
    assert _directory_snapshot(root) == before_dirs


def test_in_place_export_rewrites_only_the_opened_custom_yaml(tmp_path):
    root = tmp_path / "source"
    original = _dataset(root, task="detect")
    custom_yaml = root / "project.yml"
    Path(original.yaml_file).replace(custom_yaml)
    unrelated_yaml = root / "data.yaml"
    unrelated = b"path: unrelated\r\ntrain: elsewhere\r\n"
    unrelated_yaml.write_bytes(unrelated)
    session = DatasetSession(str(custom_yaml))

    result = export_dataset(
        session, formats=("yolo",), split="none", in_place=True
    )

    assert Path(result["yaml"]).resolve() == custom_yaml.resolve()
    assert unrelated_yaml.read_bytes() == unrelated
    reopened = DatasetSession(str(custom_yaml))
    assert len(reopened) == len(session)


def test_uploaded_project_rolls_back_validation_moves_if_yaml_fails(tmp_path, monkeypatch):
    from libreyolo.label import dataset as dataset_module

    root = tmp_path / "uploads"
    train = root / "images" / "train"
    train.mkdir(parents=True)
    for i in range(5):
        Image.new("RGB", (12, 8), (i * 30, 0, 0)).save(train / f"image-{i}.jpg")
    before = _file_snapshot(root)

    def fail_write(*args, **kwargs):
        raise OSError("injected yaml failure")

    monkeypatch.setattr(dataset_module, "_atomic_write_text", fail_write)
    with pytest.raises(OSError, match="injected"):
        create_uploaded_project(str(root), classes=["thing"], make_val=True, val_frac=0.4)

    assert _file_snapshot(root) == before
    assert not (root / "data.yaml").exists()


def test_uploaded_project_rollback_preserves_preexisting_empty_val_dir(
    tmp_path, monkeypatch
):
    from libreyolo.label import dataset as dataset_module

    root = tmp_path / "uploads"
    train = root / "images" / "train"
    val = root / "images" / "val"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    for i in range(5):
        Image.new("RGB", (12, 8), (i * 30, 0, 0)).save(train / f"image-{i}.jpg")

    monkeypatch.setattr(
        dataset_module,
        "_atomic_write_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected")),
    )
    with pytest.raises(OSError, match="injected"):
        create_uploaded_project(
            str(root), classes=["thing"], make_val=True, val_frac=0.4
        )

    assert val.is_dir() and not any(val.iterdir())
    assert len(list(train.glob("*.jpg"))) == 5


def test_uploaded_project_rejects_preexisting_nonempty_val(tmp_path):
    root = tmp_path / "uploads"
    train = root / "images" / "train"
    val = root / "images" / "val"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    for i in range(5):
        Image.new("RGB", (12, 8), "red").save(train / f"image-{i}.jpg")
    Image.new("RGB", (12, 8), "blue").save(val / "unrelated.png")

    with pytest.raises(ValueError, match="Validation directory is not empty"):
        create_uploaded_project(
            str(root), classes=["thing"], make_val=True, val_frac=0.4
        )

    assert not (root / "data.yaml").exists()


def test_uploaded_project_rejects_cross_suffix_label_collision_before_split(tmp_path):
    root = tmp_path / "uploads"
    train = root / "images" / "train"
    train.mkdir(parents=True)
    Image.new("RGB", (12, 8), "red").save(train / "same.jpg")
    Image.new("RGB", (12, 8), "blue").save(train / "same.png")

    with pytest.raises(ValueError, match="share the label basename"):
        create_uploaded_project(str(root), classes=["thing"], make_val=True)

    assert not (root / "data.yaml").exists()
    assert {path.name for path in train.iterdir()} == {"same.jpg", "same.png"}


def test_concurrent_uploads_publish_exactly_one_complete_file(tmp_path):
    root = tmp_path / "uploads"
    barrier = threading.Barrier(2)
    outcomes = []
    payloads = [b"first-complete-payload", b"second-complete-payload"]

    def upload(payload):
        barrier.wait()
        try:
            outcomes.append(("ok", save_uploaded_image(str(root), "same.jpg", payload)))
        except Exception as exc:  # noqa: BLE001 - record the racing result
            outcomes.append(("error", exc))

    threads = [threading.Thread(target=upload, args=(payload,)) for payload in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(kind for kind, _ in outcomes) == ["error", "ok"]
    error = next(value for kind, value in outcomes if kind == "error")
    assert isinstance(error, FileExistsError)
    assert (root / "images" / "train" / "same.jpg").read_bytes() in payloads
    assert not list((root / "images" / "train").glob(".librelabel-upload-*"))


def test_cross_process_uploads_cannot_share_label_stem(tmp_path):
    root = tmp_path / "uploads"
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_process_upload,
            args=(str(root), name, payload, ready, start, outcomes),
        )
        for name, payload in (("same.jpg", b"jpg"), ("same.png", b"png"))
    ]
    for process in processes:
        process.start()
    assert ready.get(timeout=10) is True
    assert ready.get(timeout=10) is True
    start.set()
    results = [outcomes.get(timeout=10), outcomes.get(timeout=10)]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert sorted(kind for kind, _value in results) == ["error", "ok"]
    files = list((root / "images" / "train").glob("same.*"))
    assert len(files) == 1


def test_cross_process_upload_lock_canonicalizes_directory_alias(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    alias = tmp_path / "uploads-alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_process_upload,
            args=(str(path), name, payload, ready, start, outcomes),
        )
        for path, name, payload in (
            (root, "same.jpg", b"jpg"),
            (alias, "same.png", b"png"),
        )
    ]
    for process in processes:
        process.start()
    assert ready.get(timeout=10) is True
    assert ready.get(timeout=10) is True
    start.set()
    results = [outcomes.get(timeout=10), outcomes.get(timeout=10)]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert sorted(kind for kind, _value in results) == ["error", "ok"]
    assert len(list((root / "images" / "train").glob("same.*"))) == 1


def test_cross_process_label_compare_and_swap_has_exactly_one_winner(tmp_path):
    from libreyolo.label.dataset import _interprocess_path_lock

    root = tmp_path / "dataset"
    image_dir = root / "images" / "train"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (20, 12), "red").save(image_dir / "a.jpg")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nnc: 1\nnames: [thing]\n",
        encoding="utf-8",
    )
    label_path = root / "labels" / "train" / "a.txt"
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    attempting = context.Queue()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_process_label_write,
            args=(str(yaml_path), cx, ready, start, attempting, outcomes),
        )
        for cx in (0.25, 0.75)
    ]

    try:
        for process in processes:
            process.start()
        assert [ready.get(timeout=10), ready.get(timeout=10)] == [0, 0]
        with _interprocess_path_lock(label_path):
            start.set()
            assert attempting.get(timeout=10) is True
            assert attempting.get(timeout=10) is True
            with pytest.raises(queue.Empty):
                outcomes.get(timeout=0.25)
        results = [outcomes.get(timeout=10), outcomes.get(timeout=10)]
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(result[0] for result in results) == ["error", "ok"]
    failure = next(result for result in results if result[0] == "error")
    assert failure[1] == "RuntimeError"
    assert "changed by someone else" in failure[2]
    saved = DatasetSession(str(yaml_path)).read_label(0)[0]
    assert saved[0]["cx"] in (0.25, 0.75)


def test_cross_process_sidecar_updates_preserve_distinct_fields(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "librelabel.json").write_text('{"base": true}', encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    loaded = context.Queue()
    release = context.Event()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_process_sidecar_update,
            args=(str(root), key, value, ready, start, loaded, release, outcomes),
        )
        for key, value in (("name", "Project"), ("description", "Details"))
    ]

    try:
        for process in processes:
            process.start()
        assert ready.get(timeout=10) is True
        assert ready.get(timeout=10) is True
        start.set()
        loaded.get(timeout=10)
        try:
            loaded.get(timeout=0.5)
        except queue.Empty:
            pass
        release.set()
        results = [outcomes.get(timeout=10), outcomes.get(timeout=10)]
    finally:
        release.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(result[0] for result in results) == ["ok", "ok"]
    assert json.loads((root / "librelabel.json").read_text(encoding="utf-8")) == {
        "base": True,
        "name": "Project",
        "description": "Details",
    }


def test_trash_project_rejects_directory_links_without_moving_target(tmp_path):
    from libreyolo.label.dataset import trash_project

    target = tmp_path / "real-project"
    target.mkdir()
    (target / "data.yaml").write_text("names: []\n", encoding="utf-8")
    alias = tmp_path / "project-alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory links unavailable: {exc}")

    with pytest.raises(ValueError, match="directory link"):
        trash_project(str(alias))

    assert target.is_dir()
    assert (target / "data.yaml").is_file()


def test_trash_project_rejects_missing_yaml_without_moving_its_parent(tmp_path):
    from libreyolo.label.dataset import trash_project

    project = tmp_path / "project"
    project.mkdir()
    marker = project / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="dataset YAML"):
        trash_project(str(project / "missing.yaml"))

    assert marker.read_text(encoding="utf-8") == "keep"


def test_trash_project_never_moves_the_user_home(tmp_path, monkeypatch):
    from libreyolo.label import dataset as dataset_module

    home = tmp_path / "home"
    home.mkdir()
    marker = home / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    with pytest.raises(ValueError, match="filesystem, home"):
        dataset_module.trash_project(str(home))

    assert marker.read_text(encoding="utf-8") == "keep"


def test_trash_project_allocates_distinct_paths_in_the_same_tick(
    tmp_path, monkeypatch
):
    from libreyolo.label import dataset as dataset_module

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(dataset_module.time, "time_ns", lambda: 123456789)
    roots = []
    for parent, value in (("first", "one"), ("second", "two")):
        root = tmp_path / parent / "project"
        root.mkdir(parents=True)
        (root / "marker.txt").write_text(value, encoding="utf-8")
        roots.append(root)

    destinations = [Path(dataset_module.trash_project(str(root))) for root in roots]

    assert destinations[0] != destinations[1]
    assert [
        (destination / "marker.txt").read_text(encoding="utf-8")
        for destination in destinations
    ] == ["one", "two"]


def test_upload_temp_cleanup_failure_does_not_hide_success(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    real_unlink = Path.unlink

    def fail_upload_temp_cleanup(path, *args, **kwargs):
        if path.name.startswith(".librelabel-upload-"):
            raise OSError("injected cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_upload_temp_cleanup)
    saved = save_uploaded_image(str(root), "image.jpg", b"complete")

    assert Path(saved).read_bytes() == b"complete"


def test_upload_publication_does_not_require_hard_links(tmp_path, monkeypatch):
    from libreyolo.label import dataset as dataset_module

    monkeypatch.setattr(
        dataset_module.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unsupported")),
    )

    saved = save_uploaded_image(str(tmp_path / "uploads"), "image.jpg", b"complete")

    assert Path(saved).read_bytes() == b"complete"


@pytest.mark.parametrize("task", ["semantic", "restore", "matte", "ocr"])
def test_non_annotation_tasks_are_view_only(tmp_path, task):
    session = _dataset(tmp_path / task, task=task)

    assert session.writable is False
    with pytest.raises(RuntimeError, match="view-only"):
        session.write_label(
            0,
            [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
        )


@pytest.mark.parametrize(
    "marker",
    [
        "target_dir: targets",
        "panoptic_dir: annotations/panoptic",
        "depth_scale: 256.0",
        "val_mattes: mattes/val",
        "images: images\nlabels: labels",
    ],
)
def test_taskless_task_specific_markers_are_view_only(tmp_path, marker):
    root = tmp_path / "marked"
    _dataset(root, task=None)
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        yaml_path.read_text(encoding="utf-8") + marker + "\n",
        encoding="utf-8",
    )

    session = DatasetSession(str(yaml_path))

    assert session.writable is False
    assert session._lossy_export is True


def test_markerless_restore_layout_cannot_use_generic_export(tmp_path):
    root = tmp_path / "restore"
    inputs = root / "inputs" / "train"
    targets = root / "targets" / "train"
    inputs.mkdir(parents=True)
    targets.mkdir(parents=True)
    Image.new("RGB", (12, 8), "red").save(inputs / "a.jpg")
    Image.new("RGB", (12, 8), "blue").save(targets / "a.jpg")
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: inputs/train\nnames: [image]\nnc: 1\n",
        encoding="utf-8",
    )
    session = DatasetSession(str(root / "data.yaml"))

    assert session.writable is False
    with pytest.raises(ValueError, match="cannot preserve"):
        export_dataset(
            session, dst=str(tmp_path / "out"), formats=("yolo",), split="none"
        )
    assert (targets / "a.jpg").exists()


def test_split_assignment_validates_fractions_and_keeps_train_image():
    with pytest.raises(ValueError, match="finite"):
        _assign_splits(5, "trainval", math.nan, 0.0, 1)
    with pytest.raises(ValueError, match="split must"):
        _assign_splits(5, "unexpected", 0.2, 0.0, 1)
    assigned = _assign_splits(5, "trainvaltest", 1.0, 1.0, 1)
    assert assigned.count("train") == 1
