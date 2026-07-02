"""OC-SORT multi-object tracker.

Implements Observation-Centric SORT from:
    OC-SORT: Observation-Centric SORT on video Multi-Object Tracking
    (Cao et al., CVPR 2023)

OC-SORT keeps SORT's Kalman-filter + IoU association but adds three
observation-centric ideas that make it robust to occlusion and non-linear
motion:

* **OCM** (Observation-Centric Momentum) — a velocity-direction consistency
  term added to the IoU cost during association.
* **OCR** (Observation-Centric Recovery) — a second association pass that
  matches leftover detections against each track's *last real observation*
  instead of its (possibly drifted) Kalman prediction.
* **ORU** (Observation-Centric Re-Update) — when a track is re-found after a
  gap, the Kalman state is re-estimated along a virtual linear trajectory
  between the last observation and the new one, undoing the drift accumulated
  while the track coasted unobserved.

The motion model is the classic SORT state ``[cx, cy, area, aspect]`` (plus
velocities), which differs from the ``[cx, cy, aspect, height]`` filter used by
:class:`~libreyolo.tracking.tracker.ByteTracker`; OC-SORT is defined against the
area form, so it is reproduced here rather than sharing the ByteTrack filter.

This is a clean-room implementation written against the OC-SORT paper and
validated for numeric track-ID parity against the original authors' reference
implementation (see ``tests/unit/test_ocsort_parity.py``).
"""

from __future__ import annotations

import numpy as np

from ..utils.results import Results
from ._helpers import _as_numpy, _make_track_ids
from .config import OCSortConfig

# ---------------------------------------------------------------------------
# Geometry / cost helpers (operate on plain numpy arrays)
# ---------------------------------------------------------------------------


def _convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert ``[x1, y1, x2, y2]`` to measurement ``[cx, cy, area, aspect]``."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([cx, cy, s, r], dtype=np.float64).reshape((4, 1))


def _convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    """Convert state ``[cx, cy, area, aspect, ...]`` to ``[[x1, y1, x2, y2]]``."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array(
        [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
    ).reshape((1, 4))


def _speed_direction(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Unit ``[dy, dx]`` direction from box1 centre to box2 centre."""
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def _iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Pairwise IoU; returns an ``(len(bboxes1), len(bboxes2))`` matrix."""
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    return wh / (area1 + area2 - wh)


def _speed_direction_batch(
    dets: np.ndarray, tracks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Direction from each track's previous observation to each detection.

    Returns ``(dy, dx)`` each shaped ``(num_tracks, num_dets)``.
    """
    tracks = tracks[..., np.newaxis]
    cx1, cy1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    cx2, cy2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = cx1 - cx2
    dy = cy1 - cy2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    return dy / norm, dx / norm


def _linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """Hungarian assignment via scipy, returned as ``(K, 2)`` (row, col) pairs."""
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost_matrix)
    # reshape keeps an empty result as (0, 2) rather than (0,) so downstream
    # ``matched[:, 0]`` indexing stays valid.
    return np.array(list(zip(rows, cols))).reshape(-1, 2)


def _k_previous_obs(observations: dict, cur_age: int, k: int) -> np.ndarray:
    """Return the observation ``k`` steps before ``cur_age`` (or the oldest)."""
    if len(observations) == 0:
        return np.array([-1, -1, -1, -1, -1], dtype=np.float64)
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def _filter_matches(
    matched_indices: np.ndarray,
    iou_matrix: np.ndarray,
    iou_threshold: float,
    num_dets: int,
    num_trks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split raw assignments into matches / unmatched by the IoU threshold."""
    if matched_indices.shape[0] > 0:
        unmatched_dets = np.setdiff1d(np.arange(num_dets), matched_indices[:, 0])
        unmatched_trks = np.setdiff1d(np.arange(num_trks), matched_indices[:, 1])
        iou_vals = iou_matrix[matched_indices[:, 0], matched_indices[:, 1]]
        low = iou_vals < iou_threshold
        unmatched_dets = np.concatenate([unmatched_dets, matched_indices[low, 0]])
        unmatched_trks = np.concatenate([unmatched_trks, matched_indices[low, 1]])
        matches = matched_indices[~low]
    else:
        unmatched_dets = np.arange(num_dets)
        unmatched_trks = np.arange(num_trks)
        matches = np.empty((0, 2), dtype=int)
    return matches, unmatched_dets.astype(int), unmatched_trks.astype(int)


def _associate(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float,
    velocities: np.ndarray,
    previous_obs: np.ndarray,
    vdc_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-round association: IoU fused with observation-centric momentum."""
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    y, x = _speed_direction_batch(detections, previous_obs)
    inertia_y = velocities[:, 0][:, np.newaxis]
    inertia_x = velocities[:, 1][:, np.newaxis]
    diff_angle_cos = np.clip(inertia_x * x + inertia_y * y, -1, 1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[previous_obs[:, 4] < 0] = 0
    valid_mask = valid_mask[:, np.newaxis]

    iou_matrix = _iou_batch(detections, trackers)
    scores = detections[:, -1][:, np.newaxis]

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T * scores

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = _linear_assignment(-(iou_matrix + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0, 2))

    return _filter_matches(
        matched_indices, iou_matrix, iou_threshold, len(detections), len(trackers)
    )


# ---------------------------------------------------------------------------
# Kalman filter (SORT area form) with observation-centric re-update
# ---------------------------------------------------------------------------


class _ObservationKalman:
    """7-dim constant-velocity Kalman filter on ``[cx, cy, area, aspect]``.

    Reproduces the SORT motion model plus OC-SORT's freeze/unfreeze re-update:
    when the track coasts unobserved and is later re-found, :meth:`unfreeze`
    rewinds to the pre-gap state and replays updates along a virtual linear
    trajectory so the recovered velocity reflects the true displacement.
    """

    # State transition (F) and measurement (H) matrices, shared by all filters.
    _F = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    _H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    def __init__(self):
        self.x = np.zeros((7, 1), dtype=np.float64)
        # Initial covariance: high uncertainty on the unobservable velocities.
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        # Process noise: small on velocities, smallest on the area velocity.
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-4])
        # Measurement noise: looser on area / aspect.
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self._I = np.eye(7)

        self.history_obs: list = []
        self.observed = False
        self._frozen: dict | None = None

    def predict(self):
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self.Q

    def freeze(self):
        """Snapshot the state at the moment a track first goes unobserved."""
        self._frozen = {
            "x": self.x.copy(),
            "P": self.P.copy(),
            "history_obs": list(self.history_obs),
        }

    def unfreeze(self):
        """Re-update along a virtual trajectory after a gap (the ORU step)."""
        if self._frozen is None:
            return
        new_history = list(self.history_obs)
        saved = self._frozen
        self.x = saved["x"].copy()
        self.P = saved["P"].copy()
        self.history_obs = saved["history_obs"][:-1]
        self.observed = True
        self._frozen = None

        non_null = [i for i, d in enumerate(new_history) if d is not None]
        if len(non_null) < 2:
            return
        index1, index2 = non_null[-2], non_null[-1]
        box1 = new_history[index1].flatten()
        box2 = new_history[index2].flatten()
        x1, y1, s1, r1 = box1
        w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
        x2, y2, s2, r2 = box2
        w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)
        time_gap = index2 - index1
        dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
        dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap
        for i in range(time_gap):
            x = x1 + (i + 1) * dx
            y = y1 + (i + 1) * dy
            w = w1 + (i + 1) * dw
            h = h1 + (i + 1) * dh
            s, r = w * h, w / float(h)
            self.update(np.array([x, y, s, r]).reshape((4, 1)))
            if i != time_gap - 1:
                self.predict()

    def update(self, z: np.ndarray | None):
        self.history_obs.append(z)

        if z is None:
            if self.observed:
                self.freeze()
            self.observed = False
            return

        if not self.observed:
            self.unfreeze()
        self.observed = True

        y = z - self._H @ self.x
        pht = self.P @ self._H.T
        s = self._H @ pht + self.R
        k = pht @ np.linalg.inv(s)
        self.x = self.x + k @ y
        i_kh = self._I - k @ self._H
        self.P = i_kh @ self.P @ i_kh.T + k @ self.R @ k.T


class _Track:
    """A single OC-SORT tracklet (the SORT ``KalmanBoxTracker`` role)."""

    def __init__(self, bbox: np.ndarray, det_index: int, track_id: int, delta_t: int):
        self.kf = _ObservationKalman()
        self.kf.x[:4] = _convert_bbox_to_z(bbox)
        self.id = track_id
        self.det_index = det_index
        self.delta_t = delta_t

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=np.float64)
        self.observations: dict = {}
        self.velocity: np.ndarray | None = None

    def update(self, bbox: np.ndarray | None, det_index: int = -1):
        if bbox is None:
            self.kf.update(None)
            return

        if self.last_observation.sum() >= 0:  # has a previous observation
            previous_box = None
            for i in range(self.delta_t):
                dt = self.delta_t - i
                if self.age - dt in self.observations:
                    previous_box = self.observations[self.age - dt]
                    break
            if previous_box is None:
                previous_box = self.last_observation
            self.velocity = _speed_direction(previous_box, bbox)

        self.last_observation = bbox
        self.observations[self.age] = bbox
        self.det_index = det_index
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(_convert_bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        # Guard against a negative predicted area.
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _convert_x_to_bbox(self.kf.x)

    def get_state(self) -> np.ndarray:
        return _convert_x_to_bbox(self.kf.x)


# ---------------------------------------------------------------------------
# Public tracker
# ---------------------------------------------------------------------------


class OCSortTracker:
    """Multi-object tracker using the OC-SORT algorithm.

    Drop-in alternative to :class:`~libreyolo.tracking.tracker.ByteTracker` with
    the same ``update(results) -> results`` contract, selected via
    ``model.track(..., tracker="ocsort")``.

    Args:
        config: Tracking configuration. If None, uses default OCSortConfig.
        **kwargs: Forwarded to ``OCSortConfig.from_kwargs`` when config is None.

    Example::

        tracker = OCSortTracker()
        for frame in frames:
            result = model(frame, conf=0.1)[0]
            tracked = tracker.update(result)
            print(tracked.track_id)
    """

    def __init__(self, config: OCSortConfig | None = None, **kwargs):
        self.config = config or OCSortConfig.from_kwargs(**kwargs)
        self.trackers: list[_Track] = []
        self.frame_count = 0
        self._id_count = 0

    def reset(self):
        """Clear all tracks and reset the ID counter."""
        self.trackers.clear()
        self.frame_count = 0
        self._id_count = 0

    def _next_id(self) -> int:
        track_id = self._id_count
        self._id_count += 1
        return track_id

    # -- numeric core -------------------------------------------------------

    def _step(self, dets_all: np.ndarray) -> list[_Track]:
        """Run one frame of association/lifecycle on ``(N, 6)`` detections.

        ``dets_all`` rows are ``[x1, y1, x2, y2, score, original_index]``.
        Returns the list of currently-reported tracks (those updated this frame
        and past the confirmation gate), in the algorithm's native order.
        """
        cfg = self.config
        self.frame_count += 1

        scores = dets_all[:, 4]
        orig = dets_all[:, 5].astype(int)

        # 0.1 is OC-SORT's fixed low-score floor for the BYTE band (kept for
        # parity with the reference); det_thresh < 0.1 leaves the band empty.
        inds_second = np.logical_and(scores > 0.1, scores < cfg.det_thresh)
        dets_second = dets_all[inds_second]
        orig_second = orig[inds_second]

        remain = scores > cfg.det_thresh
        dets = dets_all[remain]
        orig_high = orig[remain]

        # Predict existing tracks; drop any that diverged to NaN.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t in range(len(self.trackers)):
            pos = self.trackers[t].predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            # ``masked_invalid`` below drops both NaN *and* inf rows, so the
            # delete-list must match (a diverged track can predict an inf box
            # via the sqrt in _convert_x_to_bbox) — otherwise self.trackers and
            # trks desync and downstream IDs corrupt silently.
            if not np.all(np.isfinite(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        zero_vel = np.array((0, 0))
        velocities = np.array(
            [t.velocity if t.velocity is not None else zero_vel for t in self.trackers]
        )
        last_boxes = np.array([t.last_observation for t in self.trackers])
        k_obs = np.array(
            [_k_previous_obs(t.observations, t.age, cfg.delta_t) for t in self.trackers]
        )

        # First round: IoU + observation-centric momentum.
        matched, unmatched_dets, unmatched_trks = _associate(
            dets, trks, cfg.iou_threshold, velocities, k_obs, cfg.inertia
        )
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :5], int(orig_high[m[0]]))

        # Optional BYTE recovery against low-score detections.
        if cfg.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = np.array(_iou_batch(dets_second, u_trks))
            if iou_left.size > 0 and iou_left.max() > cfg.iou_threshold:
                matched_indices = _linear_assignment(-iou_left)
                to_remove = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < cfg.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(
                        dets_second[det_ind, :5], int(orig_second[det_ind])
                    )
                    to_remove.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove))

        # Observation-centric recovery against last real observations.
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = np.array(_iou_batch(left_dets, left_trks))
            if iou_left.size > 0 and iou_left.max() > cfg.iou_threshold:
                rematched = _linear_assignment(-iou_left)
                to_remove_det, to_remove_trk = [], []
                for m in rematched:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < cfg.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :5], int(orig_high[det_ind]))
                    to_remove_det.append(det_ind)
                    to_remove_trk.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # Spawn tracks for unmatched high-score detections.
        for i in unmatched_dets:
            trk = _Track(
                dets[i, :5], int(orig_high[i]), self._next_id(), cfg.delta_t
            )
            self.trackers.append(trk)

        # Collect reported tracks and prune dead ones.
        reported: list[_Track] = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= cfg.min_hits or self.frame_count <= cfg.min_hits
            ):
                reported.append(trk)
            i -= 1
            if trk.time_since_update > cfg.max_age:
                self.trackers.pop(i)
        return reported

    # -- adapters -----------------------------------------------------------

    def update(self, results: Results) -> Results:
        """Run one frame of tracking on detector ``Results``.

        Returns new ``Results`` carrying ``track_id``, sliced to the detections
        that are currently tracked (matching the ByteTracker contract).
        """
        boxes_np = _as_numpy(results.boxes.xyxy).astype(np.float64)
        scores_np = _as_numpy(results.boxes.conf).astype(np.float64)
        n = len(scores_np)
        dets_all = np.zeros((n, 6), dtype=np.float64)
        if n > 0:
            dets_all[:, :4] = boxes_np
            dets_all[:, 4] = scores_np
            dets_all[:, 5] = np.arange(n)

        reported = self._step(dets_all)
        # Each reported track was updated this frame, so det_index is valid.
        pairs = sorted(
            ((t.det_index, t.id + 1) for t in reported if t.det_index >= 0),
            key=lambda p: p[0],
        )
        if not pairs:
            tracked = results._select([])
            tracked.track_id = _make_track_ids([], results)
            tracked.boxes._id = tracked.track_id
            return tracked

        indices = [p[0] for p in pairs]
        track_ids = _make_track_ids([p[1] for p in pairs], results)
        tracked = results._select(indices)
        tracked.track_id = track_ids
        tracked.boxes._id = track_ids
        return tracked

    def _update_tracks(self, dets: np.ndarray) -> np.ndarray:
        """Internal numpy entry point mirroring the reference ``OCSort.update``.

        Public tracking goes through :meth:`update`; this method exists so the
        parity test can compare raw numeric output against the original
        implementation.

        Args:
            dets: ``(N, 5)`` array of ``[x1, y1, x2, y2, score]``.

        Returns:
            ``(M, 5)`` array of ``[x1, y1, x2, y2, track_id]`` for reported
            tracks (track_id is 1-based, as in the MOT benchmark protocol).
        """
        n = len(dets)
        dets_all = np.zeros((n, 6), dtype=np.float64)
        if n > 0:
            dets_all[:, :5] = dets
            dets_all[:, 5] = np.arange(n)
        reported = self._step(dets_all)

        rows = []
        for trk in reported:
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            rows.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
        if rows:
            return np.concatenate(rows)
        return np.empty((0, 5))
