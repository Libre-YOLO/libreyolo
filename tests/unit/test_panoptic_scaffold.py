"""Tests for the panoptic-segmentation task surface (issue #555).

Task registration, the PanopticSegmentation result payload, and the
PanopticValidator's wiring and loud-failure paths. The PQ metric itself is
covered in test_panoptic_quality.py and the loader in test_panoptic_dataset.py.
"""

import numpy as np
import pytest

from libreyolo.tasks import (
    TASKS,
    normalize_task,
    suffix_to_task,
    task_suffix_pattern,
    task_to_suffix,
)

pytestmark = pytest.mark.unit


def test_panoptic_task_registered():
    assert "panoptic" in TASKS
    assert normalize_task("panoptic") == "panoptic"


@pytest.mark.parametrize(
    "alias",
    ["panoptic", "panoptic-segmentation", "panoptic_segmentation", "panseg", "pano"],
)
def test_panoptic_aliases(alias):
    assert normalize_task(alias) == "panoptic"


def test_panoptic_suffix_roundtrip():
    assert task_to_suffix("panoptic") == "panoptic"
    assert suffix_to_task("panoptic") == "panoptic"
    assert suffix_to_task("-panoptic") == "panoptic"


def test_panoptic_suffix_in_pattern():
    pattern = task_suffix_pattern(["panoptic"])
    assert pattern == "-panoptic"
    # Longest-first ordering keeps -panoptic from being masked by shorter suffixes.
    multi = task_suffix_pattern(["panoptic", "point", "pose"])
    assert multi.split("|")[0] == "-panoptic"


def test_panoptic_segmentation_payload_basics():
    from libreyolo.utils.results import PanopticSegmentation

    data = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
    segments_info = [
        {"id": 1, "category_id": 0, "isthing": True},
        {"id": 2, "category_id": 15, "isthing": False},
    ]
    pan = PanopticSegmentation(data, segments_info)

    assert pan.orig_shape == (2, 3)
    assert pan.segment_ids == [1, 2]  # 0 is void, excluded
    assert pan.segment_mask(2).tolist() == [[False, False, False], [True, True, False]]


def test_panoptic_payload_rejects_non_2d():
    from libreyolo.utils.results import PanopticSegmentation

    with pytest.raises(ValueError):
        PanopticSegmentation(np.zeros((2, 3, 4), dtype=np.int32))


def test_panoptic_payload_preserves_segments_info_across_moves():
    from libreyolo.utils.results import PanopticSegmentation

    data = np.zeros((4, 4), dtype=np.int32)
    data[1, 1] = 7
    info = [{"id": 7, "category_id": 3, "isthing": True}]
    pan = PanopticSegmentation(data, info)

    # segments_info is plain-Python metadata and must survive device/array moves
    # and whole-image slicing.
    assert pan.cpu().segments_info == info
    assert pan.numpy().segments_info == info
    assert pan[0].segments_info == info
    # Slicing must not collapse the dense (H, W) map.
    assert pan[0].data.shape == (4, 4)


def test_results_panoptic_slot_roundtrips():
    from libreyolo.utils.results import PanopticSegmentation, Results

    data = np.zeros((3, 3), dtype=np.int32)
    data[0, 0] = 5
    info = [{"id": 5, "category_id": 1, "isthing": True}]
    result = Results(
        boxes=None,
        orig_shape=(3, 3),
        panoptic=PanopticSegmentation(data, info),
    )
    assert result.panoptic is not None
    assert "panoptic" in repr(result)
    # cpu()/numpy() rebuild Results via _new; the slot and its metadata survive.
    assert result.cpu().panoptic.segments_info == info
    assert result[0].panoptic.data.shape == (3, 3)


def test_panoptic_validator_computes_pq():
    """The validator is no longer a stub: its metric hooks accumulate PQ."""
    import numpy as np

    from libreyolo.validation import PanopticValidator, ValidationConfig

    assert PanopticValidator.task == "panoptic"

    class _StubModel:
        task = "panoptic"
        nb_classes = 133

    config = ValidationConfig(data="dummy.yaml", device="cpu")
    validator = PanopticValidator(model=_StubModel(), config=config)

    validator._init_metrics()
    perfect = np.ones((2, 2), dtype=np.int64)
    validator._pq.update(
        perfect,
        [{"id": 1, "category_id": 0, "iscrowd": 0}],
        perfect,
        [{"id": 1, "category_id": 0}],
    )
    metrics = validator._compute_metrics()
    assert metrics["metrics/PQ"] == 1.0
    assert metrics["fitness"] == 1.0


def test_panoptic_postprocess_without_family_support_fails_loudly():
    """A family that does not implement panoptic must not silently score zero."""
    from libreyolo.validation import PanopticValidator, ValidationConfig

    class _NoPanopticModel:
        task = "panoptic"
        nb_classes = 133

        def _postprocess(self, *args, **kwargs):
            return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    validator = PanopticValidator(
        model=_NoPanopticModel(), config=ValidationConfig(data="d.yaml", device="cpu")
    )
    validator._original_size = (4, 4)
    with pytest.raises(ValueError, match="panoptic"):
        validator._postprocess_predictions(preds=None, batch=None)


def test_panoptic_validator_exported_from_package():
    import libreyolo

    assert hasattr(libreyolo, "PanopticValidator")
    assert hasattr(libreyolo, "PanopticSegmentation")


def test_empty_thing_class_ids_still_enforces_agreement(monkeypatch, tmp_path):
    """`thing_class_ids = set()` means 'all stuff', not 'no opinion'.

    A falsy guard would skip the check and quietly mis-split PQ_things/PQ_stuff.
    """
    import pytest as _pytest

    from libreyolo.validation import PanopticValidator, ValidationConfig
    from libreyolo.validation import panoptic_validator as pv

    class _AllStuffModel:
        task = "panoptic"
        nb_classes = 2
        thing_class_ids: set = set()

    class _Dataset:
        nc = 2
        thing_train_ids = {0}  # dataset says category 0 IS a thing
        names = {0: "person", 1: "sky"}

    monkeypatch.setattr(pv, "resolve_panoptic_data", lambda *a, **k: {})
    monkeypatch.setattr(pv, "PanopticDataset", lambda *a, **k: _Dataset())

    validator = PanopticValidator(
        model=_AllStuffModel(),
        config=ValidationConfig(data=str(tmp_path / "d.yaml"), device="cpu"),
    )
    with _pytest.raises(ValueError, match="thing_class_ids"):
        validator._setup_dataloader()


# ---------------------------------------------------------------------------
# Rendering + summary (predict --save and Results.summary)
# ---------------------------------------------------------------------------


def _quadrant_panoptic():
    """(H, W) map: segment 1 fills the top-left quadrant, rest is void."""
    data = np.zeros((40, 40), dtype=np.int32)
    data[:20, :20] = 1
    info = [{"id": 1, "category_id": 0, "isthing": True, "score": 0.9}]
    return data, info


def test_draw_panoptic_paints_segments_and_leaves_void():
    from PIL import Image

    from libreyolo.utils.drawing import draw_panoptic

    img = Image.new("RGB", (40, 40), (255, 255, 255))
    data, info = _quadrant_panoptic()
    out = draw_panoptic(img, data, info, class_names=None)

    assert out.size == img.size
    arr = np.asarray(out)
    # Segment region is tinted, void region is untouched.
    assert (arr[:20, :20] != 255).any()
    assert (arr[25:, 25:] == 255).all()


def test_draw_panoptic_with_labels_and_resize():
    """Class-name labels and a half-resolution map (NEAREST upsample) work."""
    from PIL import Image

    from libreyolo.utils.drawing import draw_panoptic

    img = Image.new("RGB", (80, 80), (255, 255, 255))
    data, info = _quadrant_panoptic()  # 40x40 map on an 80x80 image
    out = draw_panoptic(img, data, info, class_names={0: "person"})
    assert out.size == (80, 80)
    assert (np.asarray(out)[:30, :30] != 255).any()


def test_panoptic_summary_rows():
    from libreyolo.utils.results import PanopticSegmentation, Results

    data, info = _quadrant_panoptic()
    result = Results(
        boxes=None,
        orig_shape=(40, 40),
        names={0: "person"},
        panoptic=PanopticSegmentation(data, info),
    )
    rows = result.summary()
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "person"
    assert row["segment_id"] == 1
    assert row["isthing"] is True
    assert row["pixel_count"] == 400
    assert row["pixel_fraction"] == pytest.approx(0.25)
    assert row["confidence"] == pytest.approx(0.9)


def test_save_annotated_image_renders_panoptic(tmp_path):
    """The shared save path draws panoptic output instead of the raw source."""
    from PIL import Image

    from libreyolo.models.base.inference import InferenceRunner
    from libreyolo.utils.results import PanopticSegmentation, Results

    data, info = _quadrant_panoptic()
    result = Results(
        boxes=None,
        orig_shape=(40, 40),
        names={0: "person"},
        panoptic=PanopticSegmentation(data, info),
    )
    img = Image.new("RGB", (40, 40), (255, 255, 255))
    save_path = tmp_path / "pan.png"
    InferenceRunner._save_annotated_image(
        InferenceRunner.__new__(InferenceRunner), result, img, save_path
    )
    assert save_path.exists()
    arr = np.asarray(Image.open(save_path).convert("RGB"))
    assert (arr[:20, :20] != 255).any()
