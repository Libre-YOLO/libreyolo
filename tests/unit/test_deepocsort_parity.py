"""Numeric track-ID parity test for LibreYOLO's Deep OC-SORT port.

LibreYOLO's :class:`DeepOCSortTracker` ports the appearance modules of Deep
OC-SORT (Maggiolino et al., ICIP 2023). This test proves it reproduces the
*original authors'* reference implementation
(GerardMaggiolino/Deep-OC-SORT, MIT) frame-for-frame on identical detection
sequences with identical injected embeddings, in the reference's
``cmc_off / new_kf_off / grid_off`` configuration (the sub-algorithm this
port implements).

Embeddings are synthetic but semantically meaningful: each scenario object
has a persistent unit base vector with per-frame jitter, so the appearance
term genuinely drives association (and the adaptive-weighting and EMA paths
are exercised), while remaining fully deterministic for both trackers.

Two layers, mirroring ``test_ocsort_parity.py``:

* ``test_golden_parity`` (always runs) replays a committed golden fixture of
  the reference's per-frame output rows and asserts the port matches exactly.
* ``test_live_reference_parity`` (skipped unless the reference is importable)
  re-derives the reference outputs on the fly. Point it at a checkout with
  ``DEEPOCSORT_REF_PATH=/path/to/Deep-OC-SORT``.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from libreyolo.tracking.config import DeepOCSortConfig
from libreyolo.tracking.deepocsort import DeepOCSortTracker

pytestmark = pytest.mark.unit

GOLDEN_PATH = Path(__file__).parent / "data" / "deepocsort_parity_golden.json"
IMG_H, IMG_W = 720, 1280
EMB_DIM = 64

# Hyperparameter sets used by the curated sequences below.
DEFAULT = dict(
    det_thresh=0.5, max_age=30, min_hits=3, iou_threshold=0.3,
    delta_t=3, inertia=0.2,
    w_association_emb=0.75, alpha_fixed_emb=0.95, aw_param=0.5,
)
# The reference's DanceTrack-style setting: appearance weighted much higher.
HIGH_W = {**DEFAULT, "w_association_emb": 1.3, "alpha_fixed_emb": 0.9}
LOWCONF = {**DEFAULT, "det_thresh": 0.3, "min_hits": 1}


# --------------------------------------------------------------------------
# Deterministic detection+embedding generators (shared with the fixture maker)
# --------------------------------------------------------------------------


def _box(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _bank(n_ids, seed):
    """Persistent per-object unit appearance vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n_ids, EMB_DIM))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _emb(bank, obj_id, frame, jitter=0.05):
    """Object embedding at a frame: base identity vector + small jitter."""
    rng = np.random.default_rng(100_000 + obj_id * 1_000 + frame)
    v = bank[obj_id] + jitter * rng.normal(size=bank.shape[1])
    return v / np.linalg.norm(v)


def _seq(rows_per_frame):
    """Pack per-frame ``[(box5, obj_id), ...]`` rows into dets/embs arrays."""
    n_ids = 1 + max(
        (obj for frame in rows_per_frame for _, obj in frame), default=0
    )
    bank = _bank(n_ids, seed=1234)
    frames_dets, frames_embs = [], []
    for t, frame in enumerate(rows_per_frame):
        dets = np.array([r for r, _ in frame], dtype=float).reshape(-1, 5)
        embs = np.array(
            [_emb(bank, obj, t) for _, obj in frame], dtype=float
        ).reshape(-1, EMB_DIM)
        frames_dets.append(dets)
        frames_embs.append(embs)
    return frames_dets, frames_embs


def seq_linear(T=25):
    """Two objects, constant velocity, always detected."""
    return _seq([
        [
            (_box(100 + 8 * t, 200, 40, 90) + [0.9], 0),
            (_box(900 - 6 * t, 400, 50, 110) + [0.85], 1),
        ]
        for t in range(T)
    ])


def seq_crossing(T=30):
    """Two distinct-looking objects crossing paths (appearance disambiguates)."""
    return _seq([
        [
            (_box(100 + 20 * t, 300 + 6 * t, 45, 95) + [0.92], 0),
            (_box(700 - 20 * t, 300 + 6 * t, 45, 95) + [0.88], 1),
        ]
        for t in range(T)
    ])


def seq_crossing_twins(T=30):
    """Two *identically-looking* objects crossing (appearance uninformative)."""
    frames = [
        [
            (_box(100 + 20 * t, 300 + 6 * t, 45, 95) + [0.92], 0),
            (_box(700 - 20 * t, 300 + 6 * t, 45, 95) + [0.88], 0),
        ]
        for t in range(T)
    ]
    return _seq(frames)


def seq_occlusion(T=30):
    """One object vanishes for frames 10-17 then returns (EMA retention)."""
    rows = []
    for t in range(T):
        frame = [(_box(150 + 18 * t, 360, 48, 100) + [0.9], 0)]
        if not (10 <= t <= 17):
            frame.append((_box(1100 - 15 * t, 250 + 4 * t, 40, 88) + [0.8], 1))
        rows.append(frame)
    return _seq(rows)


def seq_occlusion_impostor(T=32):
    """Object 1 vanishes; a different-looking object 2 passes through its
    predicted position during the gap (appearance must reject the swap)."""
    rows = []
    for t in range(T):
        frame = [(_box(200 + 10 * t, 500, 50, 100) + [0.9], 0)]
        if t < 10 or t >= 22:
            frame.append((_box(600, 200 + 8 * t, 44, 92) + [0.85], 1))
        if 8 <= t < 24:
            # Impostor sweeping through object 1's neighbourhood.
            frame.append((_box(600, 200 + 8 * (t - 12), 44, 92) + [0.75], 2))
        rows.append(frame)
    return _seq(rows)


def seq_noisy(seed, T=40, n_obj=5):
    """Multiple objects with box jitter, score jitter and random dropouts."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform([100, 100], [1100, 600], size=(n_obj, 2))
    vel = rng.uniform(-12, 12, size=(n_obj, 2))
    size = rng.uniform([35, 80], [60, 130], size=(n_obj, 2))
    rows = []
    for _ in range(T):
        pos = pos + vel
        frame = []
        for i in range(n_obj):
            if rng.random() < 0.12:
                continue
            cx, cy = pos[i] + rng.normal(0, 1.5, size=2)
            w, h = size[i]
            score = float(np.clip(rng.normal(0.75, 0.18), 0.05, 0.99))
            frame.append((_box(cx, cy, w, h) + [score], i))
        rows.append(frame)
    return _seq(rows)


def seq_empty_and_sparse(T=20):
    """Frames with zero or one detection, with leading/trailing gaps."""
    rows = []
    for t in range(T):
        if t < 2 or t in (8, 9, 10) or t >= T - 2:
            rows.append([])
        else:
            rows.append([(_box(200 + 15 * t, 300, 50, 100) + [0.7], 0)])
    return _seq(rows)


def parity_sequences():
    """Ordered registry of ``(name, params, dets_frames, embs_frames)``."""
    return [
        ("linear", DEFAULT, *seq_linear()),
        ("crossing", DEFAULT, *seq_crossing()),
        ("crossing_twins", DEFAULT, *seq_crossing_twins()),
        ("crossing_high_w", HIGH_W, *seq_crossing()),
        ("occlusion", DEFAULT, *seq_occlusion()),
        ("occlusion_impostor", DEFAULT, *seq_occlusion_impostor()),
        ("occlusion_impostor_high_w", HIGH_W, *seq_occlusion_impostor()),
        ("empty_sparse", DEFAULT, *seq_empty_and_sparse()),
        ("noisy0", DEFAULT, *seq_noisy(0)),
        ("noisy1", HIGH_W, *seq_noisy(1)),
        ("noisy2_lowconf", LOWCONF, *seq_noisy(2)),
    ]


# --------------------------------------------------------------------------
# Comparison helpers
# --------------------------------------------------------------------------


def _by_id(rows):
    """Frame output as ``{track_id: box[x1,y1,x2,y2]}``."""
    out = {}
    for r in rows:
        out[int(round(float(r[4])))] = np.asarray(r[:4], dtype=float)
    return out


def _assert_frame_parity(port_rows, ref_rows, ctx, atol=1e-2):
    """Same reported track IDs, with boxes agreeing to within ``atol`` pixels."""
    port, ref = _by_id(port_rows), _by_id(ref_rows)
    assert set(port) == set(ref), f"{ctx}: track IDs differ {set(port)} vs {set(ref)}"
    for tid in port:
        assert np.allclose(port[tid], ref[tid], atol=atol), (
            f"{ctx}: box for id {tid} differs {port[tid]} vs {ref[tid]}"
        )


def _port_outputs(params, dets_frames, embs_frames):
    """Run DeepOCSortTracker over a sequence, returning per-frame output rows."""
    tracker = DeepOCSortTracker(config=DeepOCSortConfig(**params))
    return [
        tracker._update_tracks(d.copy(), e.copy())
        for d, e in zip(dets_frames, embs_frames)
    ]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_golden_parity():
    """The port reproduces the reference's recorded per-frame output exactly."""
    if not GOLDEN_PATH.exists():
        pytest.skip(f"golden fixture missing at {GOLDEN_PATH} (run make_fixture)")
    golden = json.loads(GOLDEN_PATH.read_text())
    by_name = {s["name"]: s for s in golden["sequences"]}

    seqs = parity_sequences()
    assert set(by_name) == {name for name, *_ in seqs}, "fixture/sequence drift"

    for name, params, dets_frames, embs_frames in seqs:
        outputs = _port_outputs(params, dets_frames, embs_frames)
        ref_frames = by_name[name]["frames"]
        assert len(outputs) == len(ref_frames), name
        for t, (port_rows, ref_rows) in enumerate(zip(outputs, ref_frames)):
            _assert_frame_parity(port_rows, ref_rows, f"{name} frame {t}")


def _import_reference():
    """Import the original Deep OC-SORT ``OCSort``, or skip.

    The reference's embedding module imports ``torchreid`` and its FastReID
    adaptor at module scope; both are stubbed since the parity harness
    injects embeddings and never runs a real ReID model.
    """
    ref_path = os.environ.get("DEEPOCSORT_REF_PATH")
    candidates = [ref_path] if ref_path else []
    candidates.append(str(Path(__file__).resolve().parents[2] / "parity_ref" / "Deep-OC-SORT"))
    for cand in candidates:
        if not cand:
            continue
        pkg = Path(cand) / "trackers" / "integrated_ocsort_embedding" / "ocsort.py"
        if not pkg.exists():
            continue
        def _reshape_z(z, dim_z, ndim):
            # Functional port of filterpy.common.reshape_z (MIT): normalize a
            # measurement to a (dim_z, 1) column vector. The reference filter
            # calls this on every update, so a bare stub is not enough.
            z = np.atleast_2d(z)
            if z.shape[1] == dim_z:
                z = z.T
            if z.shape != (dim_z, 1):
                raise ValueError(f"z must be convertible to shape ({dim_z}, 1)")
            if ndim == 1:
                z = z[:, 0]
            if ndim == 0:
                z = z[0, 0]
            return z

        for name in ("torchreid", "external", "external.adaptors",
                     "external.adaptors.fastreid_adaptor",
                     "filterpy", "filterpy.stats", "filterpy.common"):
            if name not in sys.modules:
                mod = types.ModuleType(name)
                if name.endswith("fastreid_adaptor"):
                    mod.FastReID = object
                if name.endswith("filterpy.stats"):
                    # Imported for log-likelihoods the harness never evaluates.
                    mod.logpdf = None
                if name.endswith("filterpy.common"):
                    mod.reshape_z = _reshape_z
                    mod.pretty_str = lambda label, arr: f"{label} = {arr}"
                sys.modules[name] = mod
        sys.path.insert(0, cand)
        try:
            from trackers.integrated_ocsort_embedding import ocsort as ref_ocsort
            return ref_ocsort
        except ImportError:
            continue
    pytest.skip("Deep OC-SORT reference not found (set DEEPOCSORT_REF_PATH)")


class _Shape:
    """Duck-typed stand-in for the reference's image arguments (scale == 1)."""

    def __init__(self, shape):
        self.shape = shape


def reference_outputs(ref_ocsort, params, dets_frames, embs_frames):
    """Drive the reference tracker with injected embeddings; returns rows."""
    # The reference constructor eagerly builds its embedder (which creates a
    # ./cache directory) and CMC helper; neither is used by this harness.
    ref_ocsort.EmbeddingComputer = lambda *a, **k: types.SimpleNamespace(
        compute_embedding=None
    )
    ref_ocsort.CMCComputer = lambda *a, **k: None
    tracker = ref_ocsort.OCSort(
        det_thresh=params["det_thresh"],
        max_age=params["max_age"],
        min_hits=params["min_hits"],
        iou_threshold=params["iou_threshold"],
        delta_t=params["delta_t"],
        asso_func="iou",
        inertia=params["inertia"],
        w_association_emb=params["w_association_emb"],
        alpha_fixed_emb=params["alpha_fixed_emb"],
        aw_param=params["aw_param"],
        embedding_off=False,
        cmc_off=True,
        aw_off=False,
        new_kf_off=True,
        grid_off=True,
        args=types.SimpleNamespace(dataset="mot17", test_dataset=False),
    )

    state = {"t": 0}

    def fake_compute_embedding(img, bbox, tag):
        dets = dets_frames[state["t"]]
        mask = dets[:, 4] > params["det_thresh"]
        expected = dets[mask, :4]
        assert bbox.shape == expected.shape, "harness/reference det drift"
        return embs_frames[state["t"]][mask]

    tracker.embedder = types.SimpleNamespace(compute_embedding=fake_compute_embedding)
    tracker.cmc = None  # cmc_off=True; never touched

    img_tensor = _Shape((1, 3, IMG_H, IMG_W))
    img_numpy = _Shape((IMG_H, IMG_W, 3))
    outputs = []
    for t, dets in enumerate(dets_frames):
        state["t"] = t
        outputs.append(
            tracker.update(dets.copy(), img_tensor, img_numpy, tag="parity:seq")
        )
    return outputs


def test_live_reference_parity():
    """Re-derive reference outputs live and compare to the port."""
    ref_ocsort = _import_reference()
    for name, params, dets_frames, embs_frames in parity_sequences():
        ref_rows = reference_outputs(ref_ocsort, params, dets_frames, embs_frames)
        port_rows = _port_outputs(params, dets_frames, embs_frames)
        for t in range(len(dets_frames)):
            _assert_frame_parity(port_rows[t], ref_rows[t], f"{name} frame {t}")
