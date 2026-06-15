"""Unit tests for LibreLabel data-quality features: Radar, geometry lint, dup fix.

All pure-logic / filesystem-only -- no model weights, no network. The model
forward pass is stubbed so the matching/classification logic is exercised in
isolation (and so the trust contract -- "Radar writes nothing" -- can be checked
by source inspection).
"""

import inspect

import pytest

from libreyolo.label import radar
from libreyolo.label.quality import lint_annotations

pytestmark = pytest.mark.unit


# --- IoU --------------------------------------------------------------------
def test_iou_identical_disjoint_and_partial():
    assert radar.iou((0.5, 0.5, 0.4, 0.4), (0.5, 0.5, 0.4, 0.4)) == pytest.approx(1.0)
    assert radar.iou((0.1, 0.1, 0.1, 0.1), (0.9, 0.9, 0.1, 0.1)) == 0.0
    # two equal boxes offset by half their width overlap on half their area
    v = radar.iou((0.25, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
    assert 0.3 < v < 0.34  # intersection 0.25*0.5 / union (0.5*0.5*2 - inter)


# --- classify ---------------------------------------------------------------
def _pred(cls, cx, cy, w, h, conf, mapped=True, name="x"):
    return {"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h,
            "conf": conf, "mapped": mapped, "name": name}


def test_classify_agreement_yields_nothing():
    labels = [(0, 0.5, 0.5, 0.2, 0.2)]
    preds = [_pred(0, 0.5, 0.5, 0.2, 0.2, 0.9)]
    findings, score = radar.classify(labels, preds)
    assert findings == [] and score == 0.0


def test_classify_class_slip():
    labels = [(0, 0.5, 0.5, 0.2, 0.2)]
    preds = [_pred(1, 0.5, 0.5, 0.2, 0.2, 0.88, name="dog")]
    findings, _ = radar.classify(labels, preds)
    assert len(findings) == 1
    assert findings[0]["type"] == "class"
    assert findings[0]["label_cls"] == 0 and findings[0]["pred_cls"] == 1


def test_classify_phantom_for_unmatched_label():
    findings, score = radar.classify([(0, 0.5, 0.5, 0.2, 0.2)], [])
    assert len(findings) == 1 and findings[0]["type"] == "phantom"
    assert score == pytest.approx(0.5)


def test_classify_miss_for_confident_unmatched_pred():
    labels = [(0, 0.1, 0.1, 0.1, 0.1)]
    preds = [_pred(0, 0.1, 0.1, 0.1, 0.1, 0.9),       # matches the label -> agreement
             _pred(0, 0.7, 0.7, 0.2, 0.2, 0.9)]        # nothing labelled here -> miss
    findings, _ = radar.classify(labels, preds, miss_conf=0.55)
    assert [f["type"] for f in findings] == ["miss"]


def test_classify_low_conf_pred_is_not_a_miss():
    labels = [(0, 0.1, 0.1, 0.1, 0.1)]
    preds = [_pred(0, 0.1, 0.1, 0.1, 0.1, 0.90),      # matches the label -> agreement
             _pred(0, 0.7, 0.7, 0.2, 0.2, 0.30)]       # unmatched but below miss_conf
    findings, _ = radar.classify(labels, preds, miss_conf=0.55)
    assert findings == []  # the low-conf stray pred is ignored -> no noise


def test_classify_ignores_unmapped_predictions():
    # an unmapped detection (a class the dataset doesn't have) must not create a
    # class-slip or a miss; the human's box just looks unmatched -> phantom only.
    labels = [(0, 0.5, 0.5, 0.2, 0.2)]
    preds = [_pred(None, 0.5, 0.5, 0.2, 0.2, 0.95, mapped=False)]
    findings, _ = radar.classify(labels, preds)
    assert [f["type"] for f in findings] == ["phantom"]


# --- geometry linter --------------------------------------------------------
def test_lint_flags_tiny_box():
    issues = lint_annotations([{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                                "w": 0.002, "h": 0.002}], imgsz=640)
    assert len(issues) == 1 and issues[0]["type"] == "tiny"


def test_lint_flags_sliver_but_not_tiny():
    issues = lint_annotations([{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                                "w": 0.9, "h": 0.01}], imgsz=640)  # 6.4px tall, ar 90:1
    assert len(issues) == 1 and issues[0]["type"] == "sliver"


def test_lint_flags_full_frame():
    issues = lint_annotations([{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                                "w": 0.99, "h": 0.99}], imgsz=640)
    assert len(issues) == 1 and issues[0]["type"] == "fullframe"


def test_lint_passes_a_normal_box():
    assert lint_annotations([{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                              "w": 0.3, "h": 0.4}], imgsz=640) == []


# --- trust contract ---------------------------------------------------------
def test_radar_module_never_writes_labels():
    assert "write_label" not in inspect.getsource(radar)


# --- dataset integration (filesystem only) ----------------------------------
def _make_split_dataset(root, *, leak=False):
    """Dataset with a train image; optionally a duplicate val image (leakage)."""
    from PIL import Image

    (root / "images" / "train").mkdir(parents=True)
    im = Image.new("RGB", (40, 30), (10, 120, 200))
    im.save(root / "images" / "train" / "a.jpg")
    val_line = "val: images/val\n"
    if leak:
        (root / "images" / "val").mkdir(parents=True)
        im.save(root / "images" / "val" / "a.jpg")  # identical bytes -> same dHash
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\n{val_line}"
        "nc: 2\nnames:\n  0: cat\n  1: dog\n",
        encoding="utf-8",
    )
    return yaml_path


def test_dataset_quality_surfaces_tiny_box(tmp_path):
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    ds.write_label(0, [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                        "w": 0.002, "h": 0.002}])
    q = ds.quality(imgsz=640)
    assert q["issues"] == 1
    assert q["counts"]["tiny"] == 1
    assert q["flagged"][0]["id"] == 0


def test_resolve_duplicates_keeps_train_deletes_val(tmp_path):
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path, leak=True)))
    assert len(ds) == 2
    # ids: 0 = train copy, 1 = val copy (same image -> leakage)
    train_path = ds.image_path(0)
    val_path = ds.image_path(1)
    assert train_path.parts[-2] == "train" and val_path.parts[-2] == "val"

    res = ds.resolve_duplicates([0, 1])
    assert res["kept"] == 0
    assert [r["id"] for r in res["removed"]] == [1]
    assert train_path.exists() and not val_path.exists()
    # id stays stable (tombstone), surfaced as "deleted" in the listing
    rows = {r["id"]: r["status"] for r in ds.list_images()}
    assert rows[1] == "deleted"
    # the live count drops the tombstone
    assert ds.stats()["total"] == 1


def test_resolve_duplicates_read_only_raises(tmp_path):
    from libreyolo.label.dataset import DatasetSession

    root = tmp_path / "my" / "images" / "proj"  # ambiguous 'images' -> read-only
    ds = DatasetSession(str(_make_split_dataset(root, leak=True)))
    assert ds.writable is False
    with pytest.raises(RuntimeError):
        ds.resolve_duplicates([0, 1])


def test_boost_agreement_resolves_class_names_to_dataset_space():
    # H1 regression: the base model predicts in its own class space (e.g. COCO);
    # agreement must be measured in the DATASET's space by resolving the predicted
    # class *name*, not the raw index. A detector emitting raw index 5 whose name
    # is "cat" must count as dataset class 0 ("cat"), never as index 5.
    import numpy as np

    from libreyolo.label.boost import BoostEngine

    class _Sess:
        names = ["cat", "dog"]

    class _Boxes:
        def __init__(self, cls, conf):
            self._c = np.array(cls)
            self._k = np.array(conf)

        def __len__(self):
            return len(self._c)

        def numpy(self):
            return self

        @property
        def cls(self):
            return self._c

        @property
        def conf(self):
            return self._k

    class _Res:
        def __init__(self, boxes, names):
            self.boxes = boxes
            self.names = names

    def model_for(name):
        return lambda img: _Res(_Boxes([5], [0.9]), {5: name})  # alien raw idx 5

    be = BoostEngine(_Sess(), model_name="x")
    imgs, want, names = ["a.jpg"], {"a.jpg": 0}, ["cat", "dog"]  # dataset idx 0 == "cat"
    assert be._agreement(model_for("cat"), imgs, want, names) == 1.0
    assert be._agreement(model_for("dog"), imgs, want, names) == 0.0


def test_resolve_duplicates_refuses_list_of_txt_split(tmp_path):
    # M1 regression: a split defined by a LIST of .txt manifests must not be pruned
    # (deleting a file there leaves a dangling manifest row).
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path, leak=True)))
    ds._split_sources["val"] = ["/data/train2017.txt", "/data/extra.txt"]
    with pytest.raises(RuntimeError):
        ds.resolve_duplicates([0, 1])
    assert ds.image_path(1).exists()  # guard fires before any file is touched


def test_radar_scan_end_to_end_with_stub_model(tmp_path):
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    ds.write_label(0, [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                        "w": 0.25, "h": 0.5}])  # human labelled it "cat"

    def stub_predict(image_path, names, model_name, conf):
        # the model is sure it's a "dog" in the same place -> a class slip
        return [_pred(1, 0.5, 0.5, 0.25, 0.5, 0.92, name="dog")]

    out = radar.scan_dataset(stub_predict, ds, conf=0.25)
    assert out["scanned"] == 1 and out["flagged"] == 1
    assert out["deck"][0]["id"] == 0
    assert out["deck"][0]["counts"].get("class") == 1
    assert out["findings"][0][0]["type"] == "class"


def test_has_unsupported_rows_flags_keypoints():
    # Codex P1: keypoint/pose rows must be detected so their files stay read-only.
    from libreyolo.label.labelio import has_unsupported_rows

    assert has_unsupported_rows("0 0.5 0.5 0.2 0.2\n") is False            # box
    assert has_unsupported_rows("1 0.1 0.1 0.4 0.1 0.25 0.48\n") is False  # polygon
    assert has_unsupported_rows("0 " + " ".join(["0.5"] * 55) + "\n") is True  # pose (cls+bbox+17*3)


def test_pose_dataset_is_view_only(tmp_path):
    from PIL import Image

    from libreyolo.label.dataset import DatasetSession

    (tmp_path / "images" / "train").mkdir(parents=True)
    Image.new("RGB", (20, 10)).save(tmp_path / "images" / "train" / "a.jpg")
    (tmp_path / "data.yaml").write_text(
        f"path: {tmp_path.as_posix()}\ntrain: images/train\n"
        "kpt_shape: [17, 3]\nnc: 1\nnames:\n  0: person\n", encoding="utf-8")
    ds = DatasetSession(str(tmp_path / "data.yaml"))
    assert ds.writable is False and "keypoint" in ds.reason.lower()


def test_write_label_rejects_unsupported_file(tmp_path):
    # Codex round 2: the write path must re-check the read-only contract.
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    lp = tmp_path / "labels" / "train"
    lp.mkdir(parents=True, exist_ok=True)
    (lp / "a.txt").write_text("0 " + " ".join(["0.5"] * 55) + "\n")  # pose row
    _anns, editable = ds.read_label(0)
    assert editable is False
    with pytest.raises(RuntimeError):
        ds.write_label(0, [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])


def test_images_substring_ancestor_is_read_only(tmp_path):
    # Codex round 2: an ancestor like "images_2026" mis-derives the label path.
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path / "images_2026" / "proj")))
    assert ds.writable is False


def test_mask_dataset_is_view_only(tmp_path):
    from PIL import Image

    from libreyolo.label.dataset import DatasetSession

    (tmp_path / "images" / "train").mkdir(parents=True)
    Image.new("RGB", (20, 10)).save(tmp_path / "images" / "train" / "a.jpg")
    (tmp_path / "data.yaml").write_text(
        f"path: {tmp_path.as_posix()}\ntrain: images/train\n"
        "masks_dir: masks\nnc: 1\nnames:\n  0: road\n", encoding="utf-8")
    ds = DatasetSession(str(tmp_path / "data.yaml"))
    assert ds.writable is False


def test_degenerate_polygon_dropped():
    # Codex P2: a collapsed (collinear) polygon would yield a zero-area box -> drop it.
    from libreyolo.label.labelio import sanitize_annotations

    flat = {"type": "poly", "cls": 0, "points": [0.1, 0.5, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5]}
    assert sanitize_annotations([flat], nc=2) == []
    good = {"type": "poly", "cls": 0, "points": [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]}
    assert len(sanitize_annotations([good], nc=2)) == 1


def test_write_label_epoch_guard_rejects_stale_save(tmp_path):
    # H1 regression: a save carrying an epoch from a since-switched project is
    # rejected, so an in-flight save can never land in the wrong dataset.
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    st = _LabelState(ds, assist=False)
    box = [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
    assert st.write_label(0, box, epoch=0) == 1   # matches the current epoch
    assert st.write_label(0, box) == 1            # no epoch -> always allowed (back-compat)
    with pytest.raises(RuntimeError):
        st.write_label(0, box, epoch=5)           # stale epoch -> rejected
