"""BoT-SORT multi-object tracking with camera-motion compensation.

The implementation is adapted to LibreYOLO's ``Results`` contract from the
Apache-2.0 clean-room implementation in Roboflow Trackers, pinned at commit
``3b6d910df78a7ab48d5770b5bc86e043476d2e76``. See
``THIRD_PARTY_NOTICES.txt`` for provenance and license details.

This module implements the motion-only BoT-SORT variant described by Aharon,
Orfaig, and Bobrovsky (2022): ByteTrack association with an independent
width/height Kalman state and camera-motion compensation. The paper names the
separate appearance-enabled variant ``BoT-SORT-ReID``; no ReID model is run
here.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..utils.results import Results
from .config import BoTSortConfig
from .kalman_filter import KalmanFilterXYAH, KalmanFilterXYWH
from .strack import STrack, TrackState
from .tracker import ByteTracker


def _identity_affine() -> np.ndarray:
    return np.eye(2, 3, dtype=np.float32)


class _SparseOptFlowCameraMotion:
    """Estimate previous-to-current affine motion from sparse optical flow."""

    def __init__(self, downscale: int = 2):
        self.downscale = int(downscale)
        self.frames_failed = 0
        self.reset()

    def reset(self) -> None:
        """Forget the previous frame and feature points."""
        self._initialized = False
        self.frames_failed = 0
        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None

    @staticmethod
    def _to_gray(image) -> np.ndarray:
        frame = np.asarray(image)
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] == 3:
            # BaseModel.track() and the public tracker API use RGB frames.
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError(
                "BoTSortTracker image must be an RGB/RGBA or grayscale frame; "
                f"got shape {frame.shape}"
            )
        if gray.dtype != np.uint8:
            if np.issubdtype(gray.dtype, np.floating) and gray.size:
                finite_max = np.nanmax(gray)
                if finite_max <= 1.0:
                    gray = gray * 255.0
            gray = np.nan_to_num(gray, nan=0.0, posinf=255.0, neginf=0.0)
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(gray)

    def _prepare(self, image) -> np.ndarray:
        gray = self._to_gray(image)
        if self.downscale > 1:
            height, width = gray.shape[:2]
            gray = cv2.resize(
                gray,
                (
                    max(1, width // self.downscale),
                    max(1, height // self.downscale),
                ),
            )
        return gray

    def _store(self, gray: np.ndarray, points: np.ndarray | None) -> None:
        self._prev_gray = gray.copy()
        self._prev_points = None if points is None else points.copy()
        self._initialized = True

    def estimate(self, image) -> np.ndarray:
        """Return a 2x3 affine transform mapping the last frame to ``image``.

        The first frame and all recoverable OpenCV failures return identity
        while refreshing the estimator state, so a bad or textureless frame
        cannot poison the tracking lifecycle.
        """
        gray = self._prepare(image)
        identity = _identity_affine()
        try:
            points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=1000,
                qualityLevel=0.01,
                minDistance=1,
                mask=None,
                blockSize=3,
                useHarrisDetector=False,
                k=0.04,
            )
        except cv2.error:
            self.frames_failed += 1
            self._store(gray, None)
            return identity

        if not self._initialized:
            self._store(gray, points)
            return identity

        if self._prev_gray is None or self._prev_points is None or points is None:
            self._store(gray, points)
            return identity

        try:
            matched, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_points, None
            )  # ty: ignore[no-matching-overload]
        except cv2.error:
            self.frames_failed += 1
            self._store(gray, points)
            return identity

        affine = identity
        if matched is not None and status is not None:
            valid = status.reshape(-1).astype(bool)
            previous = self._prev_points.reshape(-1, 2)[valid]
            current = matched.reshape(-1, 2)[valid]
            finite = np.isfinite(previous).all(axis=1) & np.isfinite(current).all(
                axis=1
            )
            previous = previous[finite]
            current = current[finite]
            if len(previous) >= 5:
                try:
                    estimated, _ = cv2.estimateAffinePartial2D(
                        previous,
                        current,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                    )
                except cv2.error:
                    estimated = None
                    self.frames_failed += 1
                if estimated is not None and np.isfinite(estimated).all():
                    affine = estimated.astype(np.float32)
                    if self.downscale > 1:
                        affine[0, 2] *= self.downscale
                        affine[1, 2] *= self.downscale
            else:
                self.frames_failed += 1

        self._store(gray, points)
        return affine


class _BoTSTrack(STrack):
    """Single BoT-SORT track using ``(cx, cy, w, h, velocities...)``."""

    @staticmethod
    def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
        width = max(float(xyxy[2] - xyxy[0]), 1e-3)
        height = max(float(xyxy[3] - xyxy[1]), 1e-3)
        return np.array(
            [xyxy[0] + width / 2, xyxy[1] + height / 2, width, height],
            dtype=np.float64,
        )

    def _measurement(self, xyxy: np.ndarray) -> np.ndarray:
        return self.xyxy_to_xywh(xyxy)

    @property
    def xywh(self) -> np.ndarray:
        """Current ``(center_x, center_y, width, height)`` state."""
        return self.mean[:4].copy()

    @property
    def tlwh(self) -> np.ndarray:
        """Current ``(top-left x, top-left y, width, height)`` state."""
        ret = self.mean[:4].copy()
        ret[2:] = np.maximum(ret[2:], 1e-3)
        ret[:2] -= ret[2:] / 2
        return ret

    def _clamp_size(self) -> None:
        self.mean[2:4] = np.maximum(self.mean[2:4], 1e-3)

    def activate(self, kf: KalmanFilterXYAH, frame_id: int, track_id: int):
        super().activate(kf, frame_id, track_id)
        self._clamp_size()

    def re_activate(
        self,
        kf: KalmanFilterXYAH,
        new_detection: STrack,
        frame_id: int,
        new_id: bool = False,
        new_track_id: int = 0,
    ):
        super().re_activate(kf, new_detection, frame_id, new_id, new_track_id)
        self._clamp_size()

    def update(self, kf: KalmanFilterXYAH, new_detection: STrack, frame_id: int):
        super().update(kf, new_detection, frame_id)
        self._clamp_size()

    def predict(self, kf: KalmanFilterXYAH):
        # Freeze both size velocities for lost tracks; center motion remains
        # available for short-gap recovery.
        if self.state != TrackState.Tracked:
            self.mean[6:8] = 0
        super().predict(kf)
        self._clamp_size()

    def apply_camera_motion(self, affine: np.ndarray) -> None:
        """Warp center state, center velocity, and their covariance in place."""
        rotation = np.asarray(affine[:2, :2], dtype=np.float64)
        translation = np.asarray(affine[:2, 2], dtype=np.float64)
        self.mean[:2] = rotation @ self.mean[:2] + translation
        self.mean[4:6] = rotation @ self.mean[4:6]

        transform = np.eye(8, dtype=np.float64)
        transform[:2, :2] = rotation
        transform[4:6, 4:6] = rotation
        self.covariance = transform @ self.covariance @ transform.T
        self._clamp_size()


class BoTSortTracker(ByteTracker):
    """BoT-SORT tracker with XYWH motion and sparse optical-flow CMC.

    Args:
        config: BoT-SORT configuration. If None, ``BoTSortConfig`` is built
            from ``kwargs``.
        **kwargs: Forwarded to :meth:`BoTSortConfig.from_kwargs` when config
            is None.

    Example::

        tracker = BoTSortTracker()
        for frame in frames:  # PIL images or RGB numpy arrays
            result = model(frame, conf=0.1)
            tracked = tracker.update(result, frame)
            print(tracked.track_id)
    """

    def __init__(self, config: BoTSortConfig | None = None, **kwargs):
        if config is not None and not isinstance(config, BoTSortConfig):
            raise TypeError(
                "BoTSortTracker config must be a BoTSortConfig instance; "
                f"got {type(config).__name__}"
            )
        resolved = config or BoTSortConfig.from_kwargs(**kwargs)
        super().__init__(config=resolved)
        self.config: BoTSortConfig = resolved
        self.kalman_filter = KalmanFilterXYWH()
        self.camera_motion = _SparseOptFlowCameraMotion(
            downscale=self.config.cmc_downscale
        )
        self._tentative_id_count = 0

    def _make_track(
        self,
        xyxy: np.ndarray,
        score: float,
        cls: int,
        detection_index: int,
    ) -> _BoTSTrack:
        return _BoTSTrack(xyxy, score, cls, detection_index)

    def _after_predict(self, image) -> None:
        if not self.config.enable_cmc:
            return
        affine = self.camera_motion.estimate(image)
        seen: set[int] = set()
        for track in self.tracked_stracks + self.lost_stracks:
            if id(track) in seen:
                continue
            seen.add(id(track))
            if not isinstance(track, _BoTSTrack):
                raise TypeError(
                    "BoTSortTracker encountered a non-BoT-SORT tracklet: "
                    f"{type(track).__name__}"
                )
            track.apply_camera_motion(affine)

    def _activate_new_track(self, track: STrack) -> None:
        # The reference lifecycle immediately confirms detections in frame 1,
        # but makes later births survive one association before consuming a
        # public ID. Unique negative IDs keep tentative tracks distinct inside
        # ByteTracker's list bookkeeping without leaking into Results.
        if self._frame_id == 1:
            track.activate(self.kalman_filter, self._frame_id, self._next_id())
            return
        self._tentative_id_count += 1
        track.activate(self.kalman_filter, self._frame_id, -self._tentative_id_count)
        track.is_activated = False

    def _after_unconfirmed_match(self, track: STrack) -> None:
        if track._hits >= self.config.minimum_consecutive_frames:
            track.track_id = self._next_id()
            track.is_activated = True
        else:
            track.is_activated = False

    def _should_output(self, track: STrack) -> bool:
        # Frame-one activation is intentionally immediate, matching the
        # BoT-SORT/ByteTrack reference lifecycle even when the configured
        # consecutive-frame requirement is greater than one.
        return track.is_activated and track.track_id > 0

    def update(self, results: Results, image=None) -> Results:
        """Track one frame and return confirmed detections with stable IDs."""
        if self.config.enable_cmc and image is None:
            raise ValueError(
                "BoTSortTracker.update() needs the frame image when camera-motion "
                "compensation is enabled; pass update(results, image), or set "
                "enable_cmc=False."
            )
        return super().update(results, image=image)

    def reset(self):
        """Clear tracks, IDs, and camera-motion history."""
        super().reset()
        self._tentative_id_count = 0
        if hasattr(self, "camera_motion"):
            self.camera_motion.reset()


__all__ = ["BoTSortTracker"]
