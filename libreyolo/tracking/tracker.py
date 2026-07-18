"""ByteTrack multi-object tracker.

Implements the BYTE association method from:
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    (Zhang et al., ECCV 2022)
"""

from __future__ import annotations

import numpy as np

from ..utils.results import Results
from ._helpers import _as_numpy, _make_track_ids
from .config import TrackConfig
from .kalman_filter import KalmanFilterXYAH
from .matching import fuse_score, iou_distance, linear_assignment
from .strack import STrack, TrackState


class ByteTracker:
    """Multi-object tracker using the ByteTrack algorithm.

    Args:
        config: Tracking configuration. If None, uses default TrackConfig.
        **kwargs: Forwarded to TrackConfig.from_kwargs when config is None.

    Example::

        tracker = ByteTracker()
        for frame in frames:
            result = model(frame, conf=0.1)
            tracked = tracker.update(result)
            print(tracked.track_id)
    """

    def __init__(self, config: TrackConfig | None = None, **kwargs):
        self.config = config or TrackConfig.from_kwargs(**kwargs)
        self._id_count: int = 0
        self._frame_id: int = 0
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.kalman_filter = KalmanFilterXYAH()

    def _next_id(self) -> int:
        self._id_count += 1
        return self._id_count

    def reset(self):
        """Clear all tracks and reset the ID counter."""
        self._id_count = 0
        self._frame_id = 0
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()

    def _make_track(
        self,
        xyxy: np.ndarray,
        score: float,
        cls: int,
        detection_index: int,
    ) -> STrack:
        """Create one detection tracklet.

        Subclasses override this to select a different Kalman state
        representation while reusing ByteTrack's association lifecycle.
        """
        return STrack(xyxy, score, cls, detection_index)

    def _after_predict(self, image) -> None:
        """Hook for subclasses to transform predicted tracks before matching."""

    def _activate_new_track(self, track: STrack) -> None:
        """Activate a newly spawned track using ByteTrack's ID semantics."""
        track.activate(self.kalman_filter, self._frame_id, self._next_id())
        if self.config.minimum_consecutive_frames > 1:
            track.is_activated = False

    def _after_unconfirmed_match(self, track: STrack) -> None:
        """Hook called after an unconfirmed track receives a detection."""

    def _should_output(self, track: STrack) -> bool:
        """Return whether a currently tracked object is mature enough to emit."""
        return (
            track.is_activated
            and track._hits >= self.config.minimum_consecutive_frames
        )

    def update(self, results: Results, image=None) -> Results:
        """Run one frame of tracking.

        Takes detection results and returns new Results with track IDs
        assigned. Only confirmed, currently tracked objects are returned.

        Args:
            results: Detection results from any detector.
            image: Optional current frame. Ignored by ByteTrack; used by
                trackers that extend its lifecycle with image-based motion.

        Returns:
            New Results with ``track_id`` matching the input box backend.
        """
        self._frame_id += 1
        cfg = self.config

        # ------------------------------------------------------------------
        # 1. Extract detections (torch/NumPy -> NumPy boundary)
        # ------------------------------------------------------------------
        boxes_np = _as_numpy(results.boxes.xyxy).astype(np.float64)
        scores_np = _as_numpy(results.boxes.conf).astype(np.float64)
        classes_np = _as_numpy(results.boxes.cls).astype(np.float64)

        # Filter below track_low_thresh and build STrack candidates.
        keep = scores_np >= cfg.track_low_thresh
        boxes_np = boxes_np[keep]
        scores_np = scores_np[keep]
        classes_np = classes_np[keep]
        # Map back to original detection indices for result slicing.
        original_indices = np.where(keep)[0]

        # Split into high / low confidence.
        high_mask = scores_np >= cfg.track_high_thresh
        low_mask = ~high_mask

        high_dets = [
            self._make_track(
                boxes_np[i], scores_np[i], classes_np[i], int(original_indices[i])
            )
            for i in np.where(high_mask)[0]
        ]
        low_dets = [
            self._make_track(
                boxes_np[i], scores_np[i], classes_np[i], int(original_indices[i])
            )
            for i in np.where(low_mask)[0]
        ]

        high_bboxes = boxes_np[high_mask] if len(high_dets) > 0 else np.empty((0, 4))
        low_bboxes = boxes_np[low_mask] if len(low_dets) > 0 else np.empty((0, 4))
        high_scores = scores_np[high_mask] if len(high_dets) > 0 else np.empty(0)

        # ------------------------------------------------------------------
        # 2. Predict existing tracks
        # ------------------------------------------------------------------
        for t in self.tracked_stracks:
            t.predict(self.kalman_filter)
        for t in self.lost_stracks:
            t.predict(self.kalman_filter)
        self._after_predict(image)

        # Split tracked into confirmed and unconfirmed.
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked_stracks = [t for t in self.tracked_stracks if t.is_activated]

        # Pool for first association: confirmed tracked + lost.
        strack_pool = _joint_stracks(tracked_stracks, self.lost_stracks)

        # ------------------------------------------------------------------
        # 3. Stage 1: high-confidence detections ↔ track pool
        # ------------------------------------------------------------------
        cost = iou_distance(strack_pool, high_bboxes)
        if cfg.fuse_score and len(high_scores) > 0:
            cost = fuse_score(cost, high_scores)
        matches, u_track, u_det_high = linear_assignment(cost, cfg.match_thresh)

        for m in matches:
            track = strack_pool[m[0]]
            det = high_dets[m[1]]
            if track.state == TrackState.Tracked:
                track.update(self.kalman_filter, det, self._frame_id)
            else:
                track.re_activate(self.kalman_filter, det, self._frame_id)

        # ------------------------------------------------------------------
        # 4. Stage 2: low-confidence detections ↔ remaining tracked (NOT lost)
        # ------------------------------------------------------------------
        remaining_tracked = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        cost2 = iou_distance(remaining_tracked, low_bboxes)
        matches2, u_track2, _ = linear_assignment(cost2, cfg.match_thresh_low)

        for m in matches2:
            track = remaining_tracked[m[0]]
            det = low_dets[m[1]]
            track.update(self.kalman_filter, det, self._frame_id)

        # Mark unmatched tracked as lost.
        for i in u_track2:
            t = remaining_tracked[i]
            if t.state != TrackState.Lost:
                t.mark_lost()

        # ------------------------------------------------------------------
        # 5. Stage 3: remaining high-conf ↔ unconfirmed tracks
        # ------------------------------------------------------------------
        remaining_high_dets = [high_dets[i] for i in u_det_high]
        remaining_high_bboxes = (
            np.array([d._xyxy for d in remaining_high_dets])
            if remaining_high_dets
            else np.empty((0, 4))
        )
        remaining_high_scores = np.array([d.score for d in remaining_high_dets])
        cost3 = iou_distance(unconfirmed, remaining_high_bboxes)
        if cfg.fuse_score and len(remaining_high_scores) > 0:
            cost3 = fuse_score(cost3, remaining_high_scores)
        matches3, u_unconf, u_det_final = linear_assignment(
            cost3, cfg.match_thresh_unconfirmed
        )

        for m in matches3:
            track = unconfirmed[m[0]]
            det = remaining_high_dets[m[1]]
            track.update(self.kalman_filter, det, self._frame_id)
            self._after_unconfirmed_match(track)

        # Remove unmatched unconfirmed tracks.
        for i in u_unconf:
            unconfirmed[i].mark_removed()
            self.removed_stracks.append(unconfirmed[i])

        # ------------------------------------------------------------------
        # 6. Initialize new tracks from remaining unmatched high-conf detections
        # ------------------------------------------------------------------
        for i in u_det_final:
            det = remaining_high_dets[i]
            if det.score >= cfg.new_track_thresh:
                self._activate_new_track(det)

        # ------------------------------------------------------------------
        # 7. Handle lost tracks: mark expired as removed
        # ------------------------------------------------------------------
        max_time_lost = int(cfg.track_buffer * cfg.frame_rate / 30)
        for t in self.lost_stracks:
            if self._frame_id - t.frame_id > max_time_lost:
                t.mark_removed()
                self.removed_stracks.append(t)

        # ------------------------------------------------------------------
        # 8. Update track lists
        # ------------------------------------------------------------------
        # Collect all tracks that are now Tracked (from any source).
        all_candidates = strack_pool + unconfirmed + high_dets + low_dets
        new_tracked = [t for t in all_candidates if t.state == TrackState.Tracked]
        # Deduplicate by track_id (keep first seen = the updated one).
        seen_ids: set[int] = set()
        deduped_tracked: list[STrack] = []
        for t in new_tracked:
            if t.track_id not in seen_ids:
                seen_ids.add(t.track_id)
                deduped_tracked.append(t)
        self.tracked_stracks = deduped_tracked

        # Lost: anything from strack_pool or old lost list that is still Lost.
        tracked_ids = {t.track_id for t in self.tracked_stracks}
        self.lost_stracks = [
            t
            for t in strack_pool + self.lost_stracks
            if t.state == TrackState.Lost and t.track_id not in tracked_ids
        ]
        # Deduplicate lost list.
        seen_lost: set[int] = set()
        deduped_lost: list[STrack] = []
        for t in self.lost_stracks:
            if t.track_id not in seen_lost:
                seen_lost.add(t.track_id)
                deduped_lost.append(t)
        self.lost_stracks = deduped_lost

        # Remove duplicates between tracked and lost.
        self.tracked_stracks, self.lost_stracks = _remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # Prune removed list.
        self.removed_stracks = [
            t for t in self.removed_stracks if t.state == TrackState.Removed
        ]
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-500:]

        # ------------------------------------------------------------------
        # 9. Build output Results
        # ------------------------------------------------------------------
        output_stracks = [
            t for t in self.tracked_stracks if self._should_output(t)
        ]

        if len(output_stracks) == 0:
            tracked = results._select([])
            tracked.track_id = _make_track_ids([], results)
            tracked.boxes._id = tracked.track_id
            return tracked

        # Slice original detections by detection_index.
        indices = [t.detection_index for t in output_stracks]
        track_ids = _make_track_ids([t.track_id for t in output_stracks], results)
        tracked = results._select(indices)
        tracked.track_id = track_ids
        tracked.boxes._id = track_ids
        return tracked


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------


def _joint_stracks(a: list[STrack], b: list[STrack]) -> list[STrack]:
    """Merge two track lists, deduplicating by track_id."""
    seen = {}
    for t in a:
        seen[t.track_id] = t
    for t in b:
        if t.track_id not in seen:
            seen[t.track_id] = t
    return list(seen.values())


def _remove_duplicate_stracks(
    tracked: list[STrack], lost: list[STrack]
) -> tuple[list[STrack], list[STrack]]:
    """Remove duplicates between tracked and lost lists.

    When two tracks overlap (IoU > 0.85), keep the one with more frames.
    """
    if not tracked or not lost:
        return tracked, lost

    from .matching import bbox_iou_batch

    t_bboxes = np.array([t.xyxy for t in tracked], dtype=np.float64)
    l_bboxes = np.array([t.xyxy for t in lost], dtype=np.float64)
    iou = bbox_iou_batch(t_bboxes, l_bboxes)

    remove_tracked = set()
    remove_lost = set()
    for ti, li in zip(*np.where(iou > 0.85)):
        t_age = tracked[ti].frame_id - tracked[ti].start_frame
        l_age = lost[li].frame_id - lost[li].start_frame
        if t_age >= l_age:
            remove_lost.add(li)
        else:
            remove_tracked.add(ti)

    kept_tracked = [t for i, t in enumerate(tracked) if i not in remove_tracked]
    kept_lost = [t for i, t in enumerate(lost) if i not in remove_lost]
    return kept_tracked, kept_lost
