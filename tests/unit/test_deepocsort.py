"""Unit tests for DeepOCSortTracker (libreyolo.tracking.deepocsort).

Numeric parity with the reference implementation is covered by
``test_deepocsort_parity.py``; these tests cover the Results-level contract,
the appearance machinery (EMA, adaptive weighting) and the ``model.track``
selector wiring. A fake embedder is injected throughout, so no weights are
needed: it derives a stable unit vector from each box's centre cell on a
coarse grid, giving identity-consistent appearance for smoothly-moving
objects.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.models.base.model import BaseModel
from libreyolo.tracking import DeepOCSortConfig, DeepOCSortTracker
from libreyolo.tracking.deepocsort import _adaptive_weights, _EmbTrack
from libreyolo.utils.results import Boxes, Results

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


IMAGE = np.zeros((720, 1280, 3), dtype=np.uint8)


class FakeEmbedder:
    """Deterministic embedder: unit vector from the box-centre grid cell."""

    def __init__(self, dim=32, cell=160):
        self.dim = dim
        self.cell = cell
        self.calls = 0

    def __call__(self, image, boxes):
        self.calls += 1
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        out = np.zeros((len(boxes), self.dim))
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cx = int((x1 + x2) / 2 / self.cell)
            cy = int((y1 + y2) / 2 / self.cell)
            rng = np.random.default_rng(1 + 31 * cx + 7919 * cy)
            v = rng.normal(size=self.dim)
            out[i] = v / np.linalg.norm(v)
        return out.astype(np.float32)


def _tracker(**params):
    return DeepOCSortTracker(
        config=DeepOCSortConfig(**params), embedder=FakeEmbedder()
    )


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def test_config_defaults():
    cfg = DeepOCSortConfig()
    assert cfg.det_thresh == 0.25
    assert cfg.w_association_emb == 0.75
    assert cfg.alpha_fixed_emb == 0.95
    assert cfg.aw_param == 0.5
    assert cfg.embedder == "osnet_ain_x0_25"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"det_thresh": 1.0},  # divides trust by (1 - det_thresh)
        {"det_thresh": -0.1},
        {"iou_threshold": 2.0},
        {"max_age": -1},
        {"min_hits": -1},
        {"delta_t": 0},
        {"inertia": -0.5},
        {"w_association_emb": -0.1},
        {"alpha_fixed_emb": 1.5},
        {"aw_param": -1.0},
    ],
)
def test_config_validation_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        DeepOCSortConfig(**kwargs)


def test_config_from_kwargs_warns_unknown():
    with pytest.warns(UserWarning, match="Unknown Deep OC-SORT config keys"):
        cfg = DeepOCSortConfig.from_kwargs(det_thresh=0.4, bogus=1)
    assert cfg.det_thresh == 0.4


# --------------------------------------------------------------------------
# Appearance machinery
# --------------------------------------------------------------------------


def test_emb_track_ema_update():
    emb0 = np.zeros(4)
    emb0[0] = 1.0
    trk = _EmbTrack(np.array(_box(100, 100) + [0.9]), 0, 0, 3, emb0.copy())
    new = np.zeros(4)
    new[1] = 1.0
    trk.update_emb(new, alpha=0.75)
    expected = 0.75 * emb0 + 0.25 * new
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(trk.emb, expected)
    np.testing.assert_allclose(np.linalg.norm(trk.emb), 1.0)


def test_adaptive_weights_boosts_discriminative_cells():
    emb_cost = np.array([[0.9, 0.1], [0.1, 0.9]])
    w = _adaptive_weights(emb_cost, w_assoc_emb=0.75, max_diff=0.5)
    # Every row and column has a 0.8 gap, capped at 0.5: bonus 0.25 + 0.25.
    np.testing.assert_allclose(w, 0.75 + 0.5)

    flat = np.array([[0.5, 0.5], [0.5, 0.5]])
    w = _adaptive_weights(flat, w_assoc_emb=0.75, max_diff=0.5)
    np.testing.assert_allclose(w, 0.75)  # no gap, no bonus


def test_adaptive_weights_single_row_and_column():
    one_cell = np.array([[0.7]])
    np.testing.assert_allclose(
        _adaptive_weights(one_cell, 0.75, 0.5), np.array([[0.75]])
    )


def test_ids_persist_for_moving_objects():
    tracker = _tracker(det_thresh=0.5)
    seen = []
    for t in range(8):
        res = _make_results(
            [_box(100 + 8 * t, 200), _box(900 - 6 * t, 400, 50, 110)],
            [0.9, 0.85],
            [0, 1],
        )
        out = tracker.update(res, IMAGE)
        if len(out) == 2:
            seen.append(tuple(out.track_id.tolist()))
    assert len(seen) >= 4
    assert len(set(seen)) == 1  # same two IDs every frame


def test_occlusion_recovery_keeps_id():
    tracker = _tracker(det_thresh=0.5, max_age=15)
    track_ids = {}
    for t in range(24):
        boxes = [_box(150 + 10 * t, 360)]
        confs = [0.9]
        if not (8 <= t <= 13):
            boxes.append(_box(1000, 250, 40, 88))
            confs.append(0.8)
        res = _make_results(boxes, confs, [0] * len(boxes))
        out = tracker.update(res, IMAGE)
        for i, tid in enumerate(out.track_id.tolist()):
            track_ids.setdefault(t, []).append(tid)
    before = set(track_ids[7])
    after = set(track_ids[20])
    assert before == after


# --------------------------------------------------------------------------
# Results-level contract
# --------------------------------------------------------------------------


def test_update_requires_image():
    tracker = _tracker()
    res = _make_results([_box(100, 200)], [0.9], [0])
    with pytest.raises(ValueError, match="needs the frame image"):
        tracker.update(res, None)


def test_track_id_attached_to_boxes_backend():
    tracker = _tracker(min_hits=1)
    res = _make_results([_box(100, 200)], [0.9], [0])
    out = tracker.update(res, IMAGE)
    assert isinstance(out.track_id, torch.Tensor)
    assert out.boxes._id is out.track_id


def test_numpy_backend_supported():
    tracker = _tracker(min_hits=1)
    res = _make_results([_box(100, 200)], [0.9], [0], backend="numpy")
    out = tracker.update(res, IMAGE)
    assert isinstance(out.track_id, np.ndarray)
    assert len(out.track_id) == 1


def test_empty_frame_returns_empty():
    tracker = _tracker()
    out = tracker.update(_make_results([], [], []), IMAGE)
    assert len(out) == 0
    assert len(out.track_id) == 0


def test_below_threshold_detections_are_discarded():
    tracker = _tracker(det_thresh=0.5, min_hits=1)
    res = _make_results([_box(100, 200), _box(500, 300)], [0.9, 0.3], [0, 0])
    out = tracker.update(res, IMAGE)
    assert len(out) == 1


def test_reset_clears_state():
    tracker = _tracker(min_hits=1)
    for _ in range(3):
        tracker.update(_make_results([_box(100, 200)], [0.9], [0]), IMAGE)
    tracker.reset()
    assert tracker.trackers == []
    assert tracker.frame_count == 0
    out = tracker.update(_make_results([_box(100, 200)], [0.9], [0]), IMAGE)
    assert out.track_id.tolist() == [1]  # IDs restart


def test_custom_embedder_skips_osnet_construction():
    """Injected embedders must not trigger the OSNet weight path."""
    fake = FakeEmbedder()
    tracker = DeepOCSortTracker(embedder=fake)
    tracker.update(_make_results([_box(100, 200)], [0.9], [0]), IMAGE)
    assert fake.calls == 1
    assert tracker._embedder is fake


def test_update_tracks_numeric_entry():
    tracker = _tracker(det_thresh=0.5, min_hits=1)
    dets = np.array([_box(100, 200) + [0.9]])
    embs = np.eye(1, 16)
    rows = tracker._update_tracks(dets, embs)
    assert rows.shape == (1, 5)
    assert rows[0, 4] == 1  # 1-based track id


# --------------------------------------------------------------------------
# model.track(tracker=...) selector
# --------------------------------------------------------------------------


def _spy_deepocsort(monkeypatch):
    import libreyolo.tracking as tk

    built = {}

    class Spy(tk.DeepOCSortTracker):
        def __init__(self, *a, **k):
            built["cls"] = "deepocsort"
            built["device"] = k.get("device")
            super().__init__(*a, **k)

    monkeypatch.setattr(tk, "DeepOCSortTracker", Spy)
    return built


def _det_model(device: str = "cpu"):
    return type("DetModel", (), {"task": "detect", "device": device})()


def test_selector_string_builds_deepocsort(monkeypatch):
    built = _spy_deepocsort(monkeypatch)
    with pytest.raises(FileNotFoundError):
        next(BaseModel.track(_det_model(), "missing.mp4", tracker="deepocsort"))
    assert built["cls"] == "deepocsort"


def test_deepocsort_config_type_routes(monkeypatch):
    built = _spy_deepocsort(monkeypatch)
    with pytest.raises(FileNotFoundError):
        next(
            BaseModel.track(
                _det_model(),
                "missing.mp4",
                tracker_config=DeepOCSortConfig(det_thresh=0.4),
            )
        )
    assert built["cls"] == "deepocsort"


def test_reid_embedder_follows_the_detector_device(monkeypatch):
    """ReID must not grab CUDA when the caller asked for CPU tracking."""
    built = _spy_deepocsort(monkeypatch)
    with pytest.raises(FileNotFoundError):
        next(BaseModel.track(_det_model("cpu"), "missing.mp4", tracker="deepocsort"))
    assert built["device"] == "cpu"

    seen = {}

    class FakeEmbedder:
        def __init__(self, variant, device=None):
            seen["device"] = device

    monkeypatch.setattr(
        "libreyolo.tracking.reid.OSNetEmbedder", FakeEmbedder, raising=True
    )
    DeepOCSortTracker(device="cpu")._get_embedder()
    assert seen["device"] == "cpu"
