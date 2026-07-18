"""Behavioral and numerical tests for LibreYOLO's BoT-SORT integration."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from libreyolo.models.base.model import BaseModel
from libreyolo.tracking import BoTSortConfig, BoTSortTracker
from libreyolo.tracking.botsort import _BoTSTrack
from libreyolo.tracking.kalman_filter import KalmanFilterXYWH
from libreyolo.utils.results import Boxes, Masks, Results

pytestmark = pytest.mark.unit


def _make_results(
    boxes_list,
    confs,
    classes,
    *,
    orig_shape=(360, 640),
    backend="torch",
    masks=False,
):
    if backend == "torch":
        boxes = torch.tensor(boxes_list, dtype=torch.float32).reshape(-1, 4)
        conf = torch.tensor(confs, dtype=torch.float32).reshape(-1)
        cls = torch.tensor(classes, dtype=torch.float32).reshape(-1)
    else:
        boxes = np.asarray(boxes_list, dtype=np.float32).reshape(-1, 4)
        conf = np.asarray(confs, dtype=np.float32).reshape(-1)
        cls = np.asarray(classes, dtype=np.float32).reshape(-1)

    mask_obj = None
    if masks:
        h, w = orig_shape
        data = torch.zeros((len(boxes_list), h, w), dtype=torch.uint8)
        for i, (x1, y1, x2, y2) in enumerate(boxes_list):
            data[i, int(y1) : int(y2), int(x1) : int(x2)] = 1
        mask_obj = Masks(data, orig_shape)
    return Results(boxes=Boxes(boxes, conf, cls), orig_shape=orig_shape, masks=mask_obj)


def _textured_frame(height=360, width=640, seed=7):
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _box(cx, cy, w=50, h=100):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# --------------------------------------------------------------------------
# Config and public API
# --------------------------------------------------------------------------


def test_config_defaults_and_inherited_association_contract():
    cfg = BoTSortConfig()
    assert cfg.track_high_thresh == 0.25
    assert cfg.track_low_thresh == 0.1
    assert cfg.match_thresh == 0.8
    assert cfg.enable_cmc is True
    assert cfg.cmc_method == "sparseOptFlow"
    assert cfg.cmc_downscale == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cmc_downscale": 0},
        {"cmc_method": "orb"},
        {"track_high_thresh": 1.1},
        {"track_low_thresh": -0.1},
    ],
)
def test_config_validation_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        BoTSortConfig(**kwargs)


def test_config_from_kwargs_warns_unknown():
    with pytest.warns(UserWarning, match="Unknown BoT-SORT config keys"):
        cfg = BoTSortConfig.from_kwargs(track_high_thresh=0.6, unknown=1)
    assert cfg.track_high_thresh == 0.6


def test_public_exports_are_available():
    from libreyolo import BoTSortConfig as TopLevelConfig
    from libreyolo import BoTSortTracker as TopLevelTracker

    assert TopLevelConfig is BoTSortConfig
    assert TopLevelTracker is BoTSortTracker


# --------------------------------------------------------------------------
# XYWH Kalman state and affine warping
# --------------------------------------------------------------------------


def test_xywh_kalman_tracks_width_and_height_independently():
    kf = KalmanFilterXYWH()
    mean, covariance = kf.initiate(np.array([100.0, 200.0, 40.0, 100.0]))
    assert mean.tolist() == [100.0, 200.0, 40.0, 100.0, 0.0, 0.0, 0.0, 0.0]
    assert covariance[0, 0] < covariance[1, 1]
    assert covariance[2, 2] < covariance[3, 3]

    mean[6] = 5.0
    mean[7] = -2.0
    predicted, covariance = kf.predict(mean, covariance)
    assert predicted[2] == pytest.approx(45.0)
    assert predicted[3] == pytest.approx(98.0)

    updated, covariance = kf.update(
        predicted, covariance, np.array([102.0, 201.0, 50.0, 94.0])
    )
    assert np.isfinite(updated).all()
    assert np.isfinite(covariance).all()
    assert updated[2] > 0 and updated[3] > 0


def test_xywh_multi_predict_matches_scalar_predict():
    kf = KalmanFilterXYWH()
    states, covariances = zip(
        *(
            kf.initiate(np.array(measurement, dtype=float))
            for measurement in ([100, 200, 40, 100], [300, 150, 80, 30])
        )
    )
    states = np.asarray(states)
    covariances = np.asarray(covariances)
    batch_mean, batch_cov = kf.multi_predict(states.copy(), covariances.copy())
    scalar = [kf.predict(m.copy(), c.copy()) for m, c in zip(states, covariances)]
    assert np.allclose(batch_mean, np.asarray([item[0] for item in scalar]))
    assert np.allclose(batch_cov, np.asarray([item[1] for item in scalar]))


def test_camera_motion_warps_center_velocity_and_covariance():
    kf = KalmanFilterXYWH()
    track = _BoTSTrack(np.array([80, 150, 120, 250]), 0.9, 0, 0)
    track.activate(kf, frame_id=1, track_id=1)
    track.mean[4:6] = [3.0, 4.0]
    before_size = track.mean[2:4].copy()
    before_cov = track.covariance.copy()
    affine = np.array([[0.0, -1.0, 10.0], [1.0, 0.0, -5.0]])

    track.apply_camera_motion(affine)

    assert np.allclose(track.mean[:2], [-190.0, 95.0])
    assert np.allclose(track.mean[4:6], [-4.0, 3.0])
    assert np.allclose(track.mean[2:4], before_size)
    # A 90-degree rotation swaps center x/y uncertainty.
    assert track.covariance[0, 0] == pytest.approx(before_cov[1, 1])
    assert track.covariance[1, 1] == pytest.approx(before_cov[0, 0])


# --------------------------------------------------------------------------
# Sparse optical-flow camera motion
# --------------------------------------------------------------------------


def test_sparse_optical_flow_recovers_known_translation():
    tracker = BoTSortTracker(cmc_downscale=1)
    first = _textured_frame()
    expected = np.array([[1.0, 0.0, 24.0], [0.0, 1.0, -13.0]], dtype=np.float32)
    second = cv2.warpAffine(
        first,
        expected,
        (first.shape[1], first.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )

    initial = tracker.camera_motion.estimate(first)
    estimated = tracker.camera_motion.estimate(second)

    assert np.allclose(initial, np.eye(2, 3), atol=1e-6)
    assert np.allclose(estimated[:, :2], expected[:, :2], atol=0.02)
    assert np.allclose(estimated[:, 2], expected[:, 2], atol=0.75)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((1, 1), dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((16, 16, 4), dtype=np.uint8),
    ],
)
def test_camera_motion_textureless_and_tiny_frames_fall_back_to_identity(frame):
    tracker = BoTSortTracker(cmc_downscale=2)
    first = tracker.camera_motion.estimate(frame)
    second = tracker.camera_motion.estimate(frame)
    assert np.allclose(first, np.eye(2, 3))
    assert np.allclose(second, np.eye(2, 3))


def test_camera_motion_rejects_invalid_frame_shape():
    tracker = BoTSortTracker()
    with pytest.raises(ValueError, match="RGB/RGBA or grayscale"):
        tracker.camera_motion.estimate(np.zeros((10,), dtype=np.uint8))


def test_camera_motion_accepts_normalized_float_frames():
    tracker = BoTSortTracker(cmc_downscale=1)
    frame = _textured_frame().astype(np.float32) / 255.0
    assert np.allclose(tracker.camera_motion.estimate(frame), np.eye(2, 3))


# --------------------------------------------------------------------------
# Results adapter and tracker lifecycle
# --------------------------------------------------------------------------


def test_image_is_required_only_when_camera_motion_is_enabled():
    results = _make_results([_box(100, 150)], [0.9], [0])
    tracker = BoTSortTracker()
    with pytest.raises(ValueError, match="needs the frame image"):
        tracker.update(results)
    assert tracker._frame_id == 0

    no_cmc = BoTSortTracker(enable_cmc=False)
    out = no_cmc.update(results)
    assert out.track_id.tolist() == [1]


def test_ids_persist_for_moving_objects():
    tracker = BoTSortTracker(enable_cmc=False)
    ids = []
    for t in range(10):
        result = _make_results(
            [_box(100 + 8 * t, 180), _box(500 - 6 * t, 240, 60, 80)],
            [0.9, 0.85],
            [0, 0],
        )
        out = tracker.update(result)
        ids.append(out.track_id.tolist())
    assert ids[0] == ids[-1]
    assert len(set(ids[-1])) == 2


def test_first_frame_is_instant_but_late_birth_obeys_confirmation_count():
    tracker = BoTSortTracker(enable_cmc=False, minimum_consecutive_frames=3)
    first = tracker.update(_make_results([_box(100, 150)], [0.9], [0]))
    assert first.track_id.tolist() == [1]

    # A new object appearing after frame one is tentative for its first two
    # detections, then receives the next public ID on its third hit.
    second = tracker.update(
        _make_results([_box(104, 150), _box(500, 250)], [0.9, 0.9], [0, 0])
    )
    third = tracker.update(
        _make_results([_box(108, 150), _box(504, 250)], [0.9, 0.9], [0, 0])
    )
    fourth = tracker.update(
        _make_results([_box(112, 150), _box(508, 250)], [0.9, 0.9], [0, 0])
    )
    assert second.track_id.tolist() == [1]
    assert third.track_id.tolist() == [1]
    assert fourth.track_id.tolist() == [1, 2]


def test_camera_motion_is_applied_before_association(monkeypatch):
    tracker = BoTSortTracker(cmc_downscale=1)
    frame = _textured_frame()
    first = tracker.update(_make_results([[100, 100, 150, 200]], [0.9], [0]), frame)
    first_id = first.track_id.item()
    monkeypatch.setattr(
        tracker.camera_motion,
        "estimate",
        lambda image: np.array([[1, 0, 40], [0, 1, 0]], dtype=np.float32),
    )

    shifted = tracker.update(_make_results([[140, 100, 190, 200]], [0.9], [0]), frame)
    assert shifted.track_id.item() == first_id


def test_low_confidence_recovery_keeps_id():
    tracker = BoTSortTracker(
        enable_cmc=False, track_high_thresh=0.5, track_low_thresh=0.1
    )
    first = tracker.update(_make_results([_box(100, 150)], [0.9], [0]))
    second = tracker.update(_make_results([_box(103, 151)], [0.3], [0]))
    assert second.track_id.item() == first.track_id.item()


def test_pinned_apache_reference_lifecycle_parity_golden():
    """Match the pinned Roboflow reference's IDs for every input detection.

    The golden rows were generated with Roboflow Trackers commit
    3b6d910df78a7ab48d5770b5bc86e043476d2e76. They cover first-frame
    activation, low-confidence association, occlusion recovery, a late birth
    that must be confirmed, and one-frame false births that must not consume a
    public ID. LibreYOLO IDs are normalized to its positive, one-based API.
    """
    frames = [
        ([_box(100, 100, 40, 80), _box(400, 200, 40, 80)], [0.9, 0.8]),
        ([_box(106, 100, 40, 80), _box(394, 200, 40, 80)], [0.9, 0.2]),
        ([_box(388, 200, 40, 80)], [0.2]),
        (
            [
                _box(118, 100, 40, 80),
                _box(382, 200, 40, 80),
                _box(600, 300, 40, 80),
            ],
            [0.9, 0.8, 0.9],
        ),
        (
            [
                _box(124, 100, 40, 80),
                _box(376, 200, 40, 80),
                _box(606, 300, 40, 80),
            ],
            [0.9, 0.8, 0.9],
        ),
        (
            [
                _box(130, 100, 40, 80),
                _box(370, 200, 40, 80),
                _box(612, 300, 40, 80),
                _box(800, 100, 40, 80),
            ],
            [0.9, 0.8, 0.9, 0.9],
        ),
        (
            [
                _box(136, 100, 40, 80),
                _box(364, 200, 40, 80),
                _box(618, 300, 40, 80),
            ],
            [0.9, 0.8, 0.9],
        ),
        (
            [
                _box(142, 100, 40, 80),
                _box(358, 200, 40, 80),
                _box(624, 300, 40, 80),
                _box(900, 400, 40, 80),
            ],
            [0.9, 0.8, 0.9, 0.9],
        ),
        (
            [
                _box(148, 100, 40, 80),
                _box(352, 200, 40, 80),
                _box(630, 300, 40, 80),
                _box(906, 400, 40, 80),
            ],
            [0.9, 0.8, 0.9, 0.9],
        ),
    ]
    expected = [
        [1, 2],
        [1, 2],
        [2],
        [1, 2, -1],
        [1, 2, 3],
        [1, 2, 3, -1],
        [1, 2, 3],
        [1, 2, 3, -1],
        [1, 2, 3, 4],
    ]
    tracker = BoTSortTracker(
        enable_cmc=False,
        track_buffer=10,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        minimum_consecutive_frames=1,
    )

    actual = []
    for (boxes, scores), expected_ids in zip(frames, expected):
        output = tracker.update(_make_results(boxes, scores, [0] * len(boxes)))
        frame_ids = [-1] * len(boxes)
        output_boxes = output.boxes.xyxy.cpu().numpy()
        for output_box, track_id in zip(output_boxes, output.track_id.tolist()):
            distances = np.max(
                np.abs(np.asarray(boxes, dtype=float) - output_box), axis=1
            )
            frame_ids[int(np.argmin(distances))] = int(track_id)
        actual.append(frame_ids)
        assert frame_ids == expected_ids
    assert actual == expected


def test_numpy_backend_and_empty_output_are_preserved():
    tracker = BoTSortTracker(enable_cmc=False)
    first = tracker.update(_make_results([_box(100, 150)], [0.9], [0], backend="numpy"))
    empty = tracker.update(_make_results([], [], [], backend="numpy"))
    assert isinstance(first.track_id, np.ndarray)
    assert first.track_id.dtype == np.int64
    assert isinstance(empty.track_id, np.ndarray)
    assert empty.track_id.shape == (0,)


def test_masks_are_sliced_with_tracked_detections():
    tracker = BoTSortTracker(enable_cmc=False)
    result = _make_results(
        [_box(100, 150), _box(450, 230)], [0.9, 0.8], [0, 1], masks=True
    )
    tracked = tracker.update(result)
    assert tracked.masks is not None
    assert len(tracked.masks) == len(tracked) == 2
    assert all(tracked.masks.data[i].sum() > 0 for i in range(2))


def test_reset_clears_tracks_ids_and_camera_history():
    tracker = BoTSortTracker()
    frame = _textured_frame()
    result = _make_results([_box(100, 150)], [0.9], [0])
    tracker.update(result, frame)
    assert tracker.camera_motion._initialized

    tracker.reset()

    assert tracker._frame_id == 0
    assert tracker.tracked_stracks == []
    assert not tracker.camera_motion._initialized
    assert tracker.update(result, frame).track_id.tolist() == [1]


# --------------------------------------------------------------------------
# BaseModel.track selector
# --------------------------------------------------------------------------


def _spy_botsort(monkeypatch):
    import libreyolo.tracking as tracking

    built = {}

    class SpyBoTSort(tracking.BoTSortTracker):
        def __init__(self, *args, **kwargs):
            built["tracker"] = "botsort"
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(tracking, "BoTSortTracker", SpyBoTSort)
    return built


def test_selector_string_builds_botsort(monkeypatch):
    built = _spy_botsort(monkeypatch)
    model = type("DetModel", (), {"task": "detect"})()
    with pytest.raises(FileNotFoundError):
        next(BaseModel.track(model, "missing.mp4", tracker="botsort"))
    assert built["tracker"] == "botsort"


def test_botsort_config_type_routes_before_trackconfig(monkeypatch):
    built = _spy_botsort(monkeypatch)
    model = type("DetModel", (), {"task": "detect"})()
    with pytest.raises(FileNotFoundError):
        next(
            BaseModel.track(
                model,
                "missing.mp4",
                tracker_config=BoTSortConfig(enable_cmc=False),
            )
        )
    assert built["tracker"] == "botsort"


def test_tracker_rejects_wrong_config_type():
    from libreyolo.tracking import TrackConfig

    with pytest.raises(TypeError, match="must be a BoTSortConfig"):
        BoTSortTracker(config=TrackConfig())
