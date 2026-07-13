"""Behavioural unit tests for the OC-SORT tracker and its integration.

Numeric parity against the original reference lives in
``test_ocsort_parity.py``; this file covers the LibreYOLO-facing behaviour:
the ``Results`` adapter, ID persistence, occlusion recovery, config
validation, and the ``model.track(tracker=...)`` selector.
"""

import numpy as np
import pytest
import torch

from libreyolo.models.base.model import BaseModel
from libreyolo.tracking import OCSortConfig, OCSortTracker
from libreyolo.utils.results import Boxes, Masks, Results

pytestmark = pytest.mark.unit


def _make_results(boxes_list, confs, classes, orig_shape=(720, 1280), backend="torch"):
    if backend == "torch":
        boxes = torch.tensor(boxes_list, dtype=torch.float32).reshape(-1, 4)
        conf = torch.tensor(confs, dtype=torch.float32).reshape(-1)
        cls = torch.tensor(classes, dtype=torch.float32).reshape(-1)
    else:
        boxes = np.asarray(boxes_list, dtype=np.float32).reshape(-1, 4)
        conf = np.asarray(confs, dtype=np.float32).reshape(-1)
        cls = np.asarray(classes, dtype=np.float32).reshape(-1)
    return Results(boxes=Boxes(boxes, conf, cls), orig_shape=orig_shape)


def _box(cx, cy, w=40, h=90):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def test_config_defaults():
    cfg = OCSortConfig()
    assert cfg.det_thresh == 0.25
    assert cfg.delta_t == 3
    assert cfg.use_byte is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"det_thresh": 1.5},
        {"det_thresh": -0.1},
        {"iou_threshold": 2.0},
        {"max_age": -1},
        {"min_hits": -1},
        {"delta_t": 0},
        {"delta_t": 1.5},
        {"delta_t": True},
        {"inertia": -0.5},
        {"max_age": float("nan")},
        {"inertia": float("nan")},
    ],
)
def test_config_validation_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        OCSortConfig(**kwargs)


def test_config_from_kwargs_warns_unknown():
    with pytest.warns(UserWarning, match="Unknown OC-SORT config keys"):
        cfg = OCSortConfig.from_kwargs(det_thresh=0.4, bogus_key=1)
    assert cfg.det_thresh == 0.4


# --------------------------------------------------------------------------
# Tracking behaviour via the Results adapter
# --------------------------------------------------------------------------


def test_ids_persist_for_moving_objects():
    tracker = OCSortTracker(det_thresh=0.3, min_hits=1)
    first_ids = {}
    for t in range(8):
        res = _make_results(
            [_box(100 + 10 * t, 200), _box(900 - 8 * t, 400, 50, 110)],
            [0.9, 0.85],
            [0, 0],
        )
        out = tracker.update(res)
        ids = out.track_id.tolist()
        assert len(ids) == 2
        assert len(set(ids)) == 2  # two distinct identities
        if t == 5:
            first_ids = set(ids)
        if t == 7:
            assert set(ids) == first_ids  # stable across frames


def test_track_id_attached_to_boxes_backend():
    tracker = OCSortTracker(det_thresh=0.3, min_hits=1)
    res = _make_results([_box(100, 200)], [0.9], [0])
    out = tracker.update(res)
    assert out.track_id is not None
    assert out.boxes.id is not None
    # Same backend as the source boxes (torch in -> torch out).
    assert isinstance(out.track_id, torch.Tensor)


def test_numpy_backend_supported():
    tracker = OCSortTracker(det_thresh=0.3, min_hits=1)
    res = _make_results([_box(100, 200)], [0.9], [0], backend="numpy")
    out = tracker.update(res)
    assert isinstance(out.track_id, np.ndarray)


def test_occlusion_recovery_keeps_id():
    tracker = OCSortTracker(det_thresh=0.3, min_hits=1, max_age=30)
    seen = []
    for t in range(20):
        present = not (8 <= t <= 13)  # object missing for 6 frames
        boxes = [_box(150 + 12 * t, 360, 48, 100)]
        confs = [0.9]
        if not present:
            boxes, confs = [], []
        res = _make_results(boxes, confs, [0] * len(boxes))
        out = tracker.update(res)
        ids = out.track_id.tolist()
        if present:
            seen.append(ids[0] if ids else None)
    # The same identity should bookend the occlusion gap.
    non_none = [s for s in seen if s is not None]
    assert non_none[0] == non_none[-1]


def test_empty_frame_returns_empty():
    tracker = OCSortTracker()
    out = tracker.update(_make_results([], [], []))
    assert len(out.track_id.tolist()) == 0


def test_reset_clears_state():
    tracker = OCSortTracker(det_thresh=0.3, min_hits=1)
    for _ in range(3):
        tracker.update(_make_results([_box(100, 200)], [0.9], [0]))
    assert tracker.trackers
    tracker.reset()
    assert tracker.trackers == []
    assert tracker.frame_count == 0
    assert tracker._id_count == 0


def test_update_tracks_numeric_entry():
    tracker = OCSortTracker(det_thresh=0.3, min_hits=1)
    out = tracker._update_tracks(np.array([_box(100, 200) + [0.9]], dtype=float))
    assert out.shape[1] == 5
    assert int(out[0, 4]) >= 1  # 1-based id


def test_masks_preserved_through_ocsort():
    """Segmentation masks survive OC-SORT.update() and stay sliced to matches."""
    h, w = 720, 1280
    boxes_list = [_box(200, 300, 60, 100), _box(900, 400, 60, 100)]
    boxes = torch.tensor(boxes_list, dtype=torch.float32)
    conf = torch.tensor([0.9, 0.85], dtype=torch.float32)
    cls = torch.tensor([0, 1], dtype=torch.float32)
    mask_data = torch.zeros((2, h, w), dtype=torch.uint8)
    for i, (x1, y1, x2, y2) in enumerate(boxes_list):
        mask_data[i, int(y1):int(y2), int(x1):int(x2)] = 1
    res = Results(
        boxes=Boxes(boxes, conf, cls),
        orig_shape=(h, w),
        masks=Masks(mask_data, (h, w)),
    )

    tracker = OCSortTracker(det_thresh=0.3, min_hits=1)
    tracked = tracker.update(res)
    assert tracked.masks is not None, "Masks should survive tracking"
    assert len(tracked.masks) == len(tracked), "One mask per tracked detection"
    for i in range(len(tracked.masks)):
        assert tracked.masks.data[i].sum() > 0


def test_nonfinite_prediction_drops_track_without_desync(monkeypatch):
    """A track that predicts a non-finite box is dropped, keeping indices aligned.

    Regression guard: ``masked_invalid`` removes inf rows too, so the delete
    predicate must catch inf (not just NaN) or self.trackers and trks desync
    and IDs corrupt silently.
    """
    tracker = OCSortTracker(det_thresh=0.3, min_hits=1)
    tracker._update_tracks(np.array([_box(100, 200) + [0.9]], dtype=float))
    bad = tracker.trackers[0]
    monkeypatch.setattr(bad, "predict", lambda: np.array([[np.inf, 0.0, np.inf, 0.0]]))

    out = tracker._update_tracks(np.array([_box(800, 400) + [0.9]], dtype=float))
    assert bad not in tracker.trackers  # diverged track removed
    assert np.all(np.isfinite(out[:, :4]))  # output stays finite, no crash


# --------------------------------------------------------------------------
# model.track(tracker=...) selector
# --------------------------------------------------------------------------


def test_track_selector_rejects_unknown_tracker():
    model = type("DetModel", (), {"task": "detect"})()
    with pytest.raises(ValueError, match="Unknown tracker"):
        next(BaseModel.track(model, "missing.mp4", tracker="bogus"))


def _spy_trackers(monkeypatch):
    """Patch the tracker classes with spies recording which one is built.

    Construction happens before track() validates the source path, so each
    spy records the choice and the call then fails on the missing video.
    Relies on track() importing the tracker classes in its function body
    (``from ...tracking import ...``), so patching the package attrs takes
    effect; hoisting that import to module scope would break these spies.
    """
    import libreyolo.tracking as tk

    built = {}

    class SpyByte(tk.ByteTracker):
        def __init__(self, *a, **k):
            built["cls"] = "bytetrack"
            super().__init__(*a, **k)

    class SpyOC(tk.OCSortTracker):
        def __init__(self, *a, **k):
            built["cls"] = "ocsort"
            super().__init__(*a, **k)

    monkeypatch.setattr(tk, "ByteTracker", SpyByte)
    monkeypatch.setattr(tk, "OCSortTracker", SpyOC)
    return built


def test_selector_string_builds_ocsort(monkeypatch):
    built = _spy_trackers(monkeypatch)
    model = type("DetModel", (), {"task": "detect"})()
    with pytest.raises(FileNotFoundError):
        next(BaseModel.track(model, "missing.mp4", tracker="ocsort"))
    assert built["cls"] == "ocsort"


def test_selector_default_builds_bytetrack(monkeypatch):
    built = _spy_trackers(monkeypatch)
    model = type("DetModel", (), {"task": "detect"})()
    with pytest.raises(FileNotFoundError):
        next(BaseModel.track(model, "missing.mp4"))
    assert built["cls"] == "bytetrack"


def test_ocsort_config_type_routes_to_ocsort(monkeypatch):
    built = _spy_trackers(monkeypatch)
    model = type("DetModel", (), {"task": "detect"})()
    with pytest.raises(FileNotFoundError):
        next(
            BaseModel.track(
                model, "missing.mp4", tracker_config=OCSortConfig(det_thresh=0.4)
            )
        )
    assert built["cls"] == "ocsort"
