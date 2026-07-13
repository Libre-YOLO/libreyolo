"""Deep OC-SORT multi-object tracker (OC-SORT + adaptive appearance ReID).

Implements the appearance components of:

    Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification
    (Maggiolino et al., ICIP 2023)

on top of the OC-SORT machinery in :mod:`libreyolo.tracking.ocsort`. Two ideas
are added to pure-motion OC-SORT:

* **Dynamic appearance** — each track keeps an exponential moving average of
  its detections' ReID embeddings, where the EMA factor adapts to detector
  confidence: confident detections update the track appearance strongly,
  uncertain ones barely at all.
* **Adaptive weighting (AW)** — the cosine-similarity term added to the
  association cost is boosted per det/track by how *discriminative* the
  embedding is (the gap between its best and second-best match), so appearance
  only dominates when it is actually informative.

Camera-motion compensation (the reference's third module) is out of scope, and
so is its alternative Kalman state: this port composes with the classic SORT
state already used by :class:`~libreyolo.tracking.ocsort.OCSortTracker`, which
matches the reference run with ``cmc_off --new_kf_off --grid_off``. That
configuration is what the golden parity fixture pins down
(``tests/unit/test_deepocsort_parity.py``).

Reference implementation: https://github.com/GerardMaggiolino/Deep-OC-SORT
(MIT license).
"""

from __future__ import annotations

import numpy as np

from ..utils.results import Results
from ._helpers import _as_numpy, _make_track_ids
from .config import DeepOCSortConfig
from .ocsort import (
    _filter_matches,
    _iou_batch,
    _k_previous_obs,
    _linear_assignment,
    _speed_direction_batch,
    _Track,
)

# ---------------------------------------------------------------------------
# Appearance-aware association
# ---------------------------------------------------------------------------


def _adaptive_weights(
    emb_cost: np.ndarray, w_assoc_emb: float, max_diff: float
) -> np.ndarray:
    """Per-cell weights for the appearance term (the AW module).

    Starts from the fixed ``w_assoc_emb`` and adds half the (capped) gap
    between the best and second-best similarity in each row and column, so
    discriminative embeddings count more. Mirrors the reference's
    ``compute_aw_new_metric`` exactly (loops kept: matrices are tiny).
    """
    weights = np.full_like(emb_cost, w_assoc_emb)
    bonus = np.zeros_like(emb_cost)
    if emb_cost.shape[1] >= 2:
        for i in range(emb_cost.shape[0]):
            inds = np.argsort(-emb_cost[i])
            row_gap = min(emb_cost[i, inds[0]] - emb_cost[i, inds[1]], max_diff)
            bonus[i] += row_gap / 2
    if emb_cost.shape[0] >= 2:
        for j in range(emb_cost.shape[1]):
            inds = np.argsort(-emb_cost[:, j])
            col_gap = min(emb_cost[inds[0], j] - emb_cost[inds[1], j], max_diff)
            bonus[:, j] += col_gap / 2
    return weights + bonus


def _associate_emb(
    detections: np.ndarray,
    trackers: np.ndarray,
    det_embs: np.ndarray,
    trk_embs: np.ndarray,
    iou_threshold: float,
    velocities: np.ndarray,
    previous_obs: np.ndarray,
    vdc_weight: float,
    w_assoc_emb: float,
    aw_param: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-round association: IoU + momentum + adaptively-weighted appearance.

    Identical to :func:`libreyolo.tracking.ocsort._associate` except that when
    the IoU alone is ambiguous (no one-to-one assignment above threshold), a
    cosine-similarity term scaled by :func:`_adaptive_weights` joins the cost.
    """
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
            # Unambiguous by IoU alone; appearance not consulted (reference
            # behaviour).
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            emb_cost = det_embs @ trk_embs.T
            emb_cost *= _adaptive_weights(emb_cost, w_assoc_emb, aw_param)
            final_cost = -(iou_matrix + angle_diff_cost + emb_cost)
            matched_indices = _linear_assignment(final_cost)
    else:
        matched_indices = np.empty(shape=(0, 2))

    return _filter_matches(
        matched_indices, iou_matrix, iou_threshold, len(detections), len(trackers)
    )


# ---------------------------------------------------------------------------
# Track with appearance state
# ---------------------------------------------------------------------------


class _EmbTrack(_Track):
    """OC-SORT tracklet carrying an EMA appearance embedding."""

    def __init__(
        self,
        bbox: np.ndarray,
        det_index: int,
        track_id: int,
        delta_t: int,
        emb: np.ndarray,
    ):
        super().__init__(bbox, det_index, track_id, delta_t)
        self.emb = emb

    def update_emb(self, emb: np.ndarray, alpha: float):
        """Blend a new detection embedding into the track EMA."""
        blended = alpha * self.emb + (1 - alpha) * emb
        norm = np.linalg.norm(blended)
        if not np.isfinite(norm) or norm <= np.finfo(blended.dtype).eps:
            # Exact cancellation can occur for opposite unit vectors at
            # alpha=0.5. Keeping the last valid appearance is safer than
            # injecting NaN into every later association matrix.
            return
        self.emb = blended / norm


def _validated_embeddings(embs: np.ndarray, count: int) -> np.ndarray:
    """Return finite, non-zero, row-normalized embeddings for ``count`` boxes."""
    embs = np.asarray(embs, dtype=np.float64)
    if count == 0:
        if embs.size == 0:
            return np.empty((0, 1), dtype=np.float64)
        if embs.ndim != 2 or embs.shape[0] != 0:
            raise ValueError(
                "appearance embeddings must have shape (N, D) aligned with detections"
            )
        return embs
    if embs.ndim != 2 or embs.shape[0] != count or embs.shape[1] == 0:
        raise ValueError(
            "appearance embeddings must have shape (N, D) aligned with detections; "
            f"got {embs.shape} for {count} detections"
        )
    if not np.all(np.isfinite(embs)):
        raise ValueError("appearance embeddings must contain only finite values")
    norms = np.linalg.norm(embs, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("appearance embeddings must have non-zero finite norms")
    return embs / norms[:, np.newaxis]


# ---------------------------------------------------------------------------
# Public tracker
# ---------------------------------------------------------------------------


class DeepOCSortTracker:
    """Multi-object tracker using Deep OC-SORT (appearance-aware OC-SORT).

    Selected via ``model.track(..., tracker="deepocsort")``. Unlike the
    pure-motion trackers, the update step needs the frame pixels to embed each
    detection crop, so the standalone contract is
    ``update(results, image) -> results``.

    Args:
        config: Tracking configuration. If None, uses default DeepOCSortConfig.
        embedder: Appearance embedder: any callable mapping
            ``(image_rgb_uint8, boxes_xyxy) -> (N, D) float32`` L2-normalized
            embeddings. When None, an
            :class:`~libreyolo.tracking.reid.OSNetEmbedder` for
            ``config.embedder`` is built lazily on the first frame.
        device: Torch device for the lazily built embedder. Defaults to the
            embedder's own auto-selection; ``model.track()`` passes the
            detector's device so ReID never lands on a device the caller did
            not ask for.
        **kwargs: Forwarded to ``DeepOCSortConfig.from_kwargs`` when config is
            None.

    Example::

        tracker = DeepOCSortTracker()
        for frame in frames:  # RGB numpy arrays
            result = model(frame, conf=0.1)
            tracked = tracker.update(result, frame)
            print(tracked.track_id)
    """

    def __init__(
        self,
        config: DeepOCSortConfig | None = None,
        embedder=None,
        device: str | None = None,
        **kwargs,
    ):
        self.config = config or DeepOCSortConfig.from_kwargs(**kwargs)
        self._embedder = embedder
        self.device = device
        self.trackers: list[_EmbTrack] = []
        self.frame_count = 0
        self._id_count = 0

    def reset(self):
        """Clear all tracks and reset the ID counter (embedder is kept)."""
        self.trackers.clear()
        self.frame_count = 0
        self._id_count = 0

    def _next_id(self) -> int:
        track_id = self._id_count
        self._id_count += 1
        return track_id

    def _get_embedder(self):
        if self._embedder is None:
            from .reid import OSNetEmbedder

            self._embedder = OSNetEmbedder(
                variant=self.config.embedder, device=self.device
            )
        return self._embedder

    # -- numeric core -------------------------------------------------------

    def _step(self, dets: np.ndarray, embs: np.ndarray, orig: np.ndarray) -> list:
        """One frame of association/lifecycle on high-score detections.

        Args:
            dets: ``(N, 5)`` rows of ``[x1, y1, x2, y2, score]``, already
                filtered to ``score > det_thresh`` (Deep OC-SORT has no
                low-score BYTE band).
            embs: ``(N, D)`` L2-normalized appearance embeddings, row-aligned
                with ``dets``.
            orig: ``(N,)`` original detection indices for Results slicing.
        """
        embs = _validated_embeddings(embs, len(dets))
        cfg = self.config
        self.frame_count += 1

        # Confidence-adaptive EMA factor: as detector confidence drops toward
        # det_thresh, alpha rises toward 1 (the new embedding barely counts).
        trust = (dets[:, 4] - cfg.det_thresh) / (1 - cfg.det_thresh)
        af = cfg.alpha_fixed_emb
        dets_alpha = af + (1 - af) * (1 - trust)

        # Predict existing tracks; drop any that diverged to NaN/inf (see the
        # matching guard in OCSortTracker._step for why isfinite, not isnan).
        trks = np.zeros((len(self.trackers), 5))
        trk_embs = []
        to_del = []
        for t in range(len(self.trackers)):
            pos = self.trackers[t].predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if not np.all(np.isfinite(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].emb)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        trk_embs = np.array(trk_embs)

        zero_vel = np.array((0, 0))
        velocities = np.array(
            [t.velocity if t.velocity is not None else zero_vel for t in self.trackers]
        )
        last_boxes = np.array([t.last_observation for t in self.trackers])
        k_obs = np.array(
            [_k_previous_obs(t.observations, t.age, cfg.delta_t) for t in self.trackers]
        )

        # First round: IoU + momentum + adaptively-weighted appearance.
        matched, unmatched_dets, unmatched_trks = _associate_emb(
            dets,
            trks,
            embs,
            trk_embs,
            cfg.iou_threshold,
            velocities,
            k_obs,
            cfg.inertia,
            cfg.w_association_emb,
            cfg.aw_param,
        )
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :5], int(orig[m[0]]))
            self.trackers[m[1]].update_emb(embs[m[0]], alpha=dets_alpha[m[0]])

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
                    self.trackers[trk_ind].update(dets[det_ind, :5], int(orig[det_ind]))
                    self.trackers[trk_ind].update_emb(
                        embs[det_ind], alpha=dets_alpha[det_ind]
                    )
                    to_remove_det.append(det_ind)
                    to_remove_trk.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # Spawn tracks for unmatched detections, seeded with their embedding.
        for i in unmatched_dets:
            trk = _EmbTrack(
                dets[i, :5], int(orig[i]), self._next_id(), cfg.delta_t, embs[i]
            )
            self.trackers.append(trk)

        # Collect reported tracks and prune dead ones.
        reported: list[_EmbTrack] = []
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

    def update(self, results: Results, image) -> Results:
        """Run one frame of tracking on detector ``Results``.

        Args:
            results: Detections for the frame (boxes in ``image`` pixel
                coordinates).
            image: The frame itself as an ``(H, W, 3)`` RGB uint8 array or a
                PIL image; needed to compute appearance embeddings.

        Returns:
            New ``Results`` carrying ``track_id``, sliced to the detections
            that are currently tracked (matching the ByteTracker contract).
        """
        if image is None:
            raise ValueError(
                "DeepOCSortTracker.update() needs the frame image to compute "
                "appearance embeddings; pass update(results, image)."
            )
        image = np.asarray(image)

        boxes_np = _as_numpy(results.boxes.xyxy).astype(np.float64)
        scores_np = _as_numpy(results.boxes.conf).astype(np.float64)
        keep = scores_np > self.config.det_thresh
        dets = np.concatenate(
            [boxes_np[keep], scores_np[keep, np.newaxis]], axis=1
        ) if keep.any() else np.empty((0, 5), dtype=np.float64)
        orig = np.flatnonzero(keep)

        if len(dets) > 0:
            embs = self._get_embedder()(image, dets[:, :4])
        else:
            embs = np.zeros((0, 1), dtype=np.float32)

        reported = self._step(dets, embs.astype(np.float64), orig)
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

    def _update_tracks(self, dets: np.ndarray, embs: np.ndarray) -> np.ndarray:
        """Internal numpy entry point mirroring the reference ``OCSort.update``.

        Used by the parity test to compare raw numeric output against the
        original implementation with injected (synthetic) embeddings.

        Args:
            dets: ``(N, 5)`` array of ``[x1, y1, x2, y2, score]`` (unfiltered).
            embs: ``(N, D)`` embeddings row-aligned with ``dets``; rows for
                sub-threshold detections are ignored.

        Returns:
            ``(M, 5)`` array of ``[x1, y1, x2, y2, track_id]`` for reported
            tracks (track_id is 1-based, as in the MOT benchmark protocol).
        """
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 5)
        keep = dets[:, 4] > self.config.det_thresh
        orig = np.flatnonzero(keep)
        reported = self._step(dets[keep], np.asarray(embs, dtype=np.float64)[keep], orig)

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
