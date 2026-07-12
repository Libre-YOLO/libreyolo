"""Tracking configuration for ByteTrack."""

import math
import warnings
from dataclasses import dataclass, fields


def _require_finite(config, *names: str) -> None:
    """Reject NaN/Inf before one-sided range checks can accidentally accept them."""
    for name in names:
        value = getattr(config, name)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")


@dataclass(kw_only=True)
class TrackConfig:
    """Configuration for the ByteTrack multi-object tracker.

    Args:
        track_high_thresh: Minimum confidence for first association stage.
        track_low_thresh: Minimum confidence for second association (low-conf recovery).
        new_track_thresh: Minimum confidence to initialize a new track.
        match_thresh: IoU cost threshold for first association.
        match_thresh_low: IoU cost threshold for low-confidence association (stage 2).
        match_thresh_unconfirmed: IoU cost threshold for unconfirmed track matching (stage 3).
        track_buffer: Frames to keep lost tracks before removal.
        frame_rate: Video frame rate (used to scale track_buffer).
        fuse_score: Fuse detection score with IoU for first association.
        minimum_consecutive_frames: Total matched frames (hits) a track must
            accumulate before it is confirmed and emitted (not required to be
            consecutive).
    """

    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    match_thresh: float = 0.8
    match_thresh_low: float = 0.5
    match_thresh_unconfirmed: float = 0.7
    track_buffer: int = 30
    frame_rate: int = 30
    fuse_score: bool = True
    minimum_consecutive_frames: int = 1

    def __post_init__(self):
        _require_finite(
            self,
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "match_thresh",
            "match_thresh_low",
            "match_thresh_unconfirmed",
            "track_buffer",
            "frame_rate",
            "minimum_consecutive_frames",
        )
        if self.frame_rate <= 0:
            raise ValueError(f"frame_rate must be > 0, got {self.frame_rate}")
        if not (0 <= self.track_high_thresh <= 1):
            raise ValueError(
                f"track_high_thresh must be in [0, 1], got {self.track_high_thresh}"
            )
        if not (0 <= self.track_low_thresh <= 1):
            raise ValueError(
                f"track_low_thresh must be in [0, 1], got {self.track_low_thresh}"
            )
        if not (0 <= self.new_track_thresh <= 1):
            raise ValueError(
                f"new_track_thresh must be in [0, 1], got {self.new_track_thresh}"
            )
        if not (0 <= self.match_thresh <= 1):
            raise ValueError(f"match_thresh must be in [0, 1], got {self.match_thresh}")
        if not (0 <= self.match_thresh_low <= 1):
            raise ValueError(
                f"match_thresh_low must be in [0, 1], got {self.match_thresh_low}"
            )
        if not (0 <= self.match_thresh_unconfirmed <= 1):
            raise ValueError(
                f"match_thresh_unconfirmed must be in [0, 1], got "
                f"{self.match_thresh_unconfirmed}"
            )
        if self.track_buffer < 0:
            raise ValueError(f"track_buffer must be >= 0, got {self.track_buffer}")
        if self.minimum_consecutive_frames < 1:
            raise ValueError(
                f"minimum_consecutive_frames must be >= 1, got {self.minimum_consecutive_frames}"
            )
        if self.track_high_thresh < self.track_low_thresh:
            raise ValueError(
                f"track_high_thresh ({self.track_high_thresh}) must be >= "
                f"track_low_thresh ({self.track_low_thresh})"
            )

    @classmethod
    def from_kwargs(cls, **kwargs) -> "TrackConfig":
        """Construct config, warning on unknown keys."""
        valid = {f.name for f in fields(cls)}
        unknown = set(kwargs) - valid
        if unknown:
            warnings.warn(
                f"Unknown tracking config keys (ignored): {sorted(unknown)}",
                stacklevel=2,
            )
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**filtered)


@dataclass(kw_only=True)
class DeepOCSortConfig:
    """Configuration for the Deep OC-SORT multi-object tracker.

    Deep OC-SORT (Maggiolino et al., ICIP 2023) extends OC-SORT with
    appearance re-identification: each track keeps a confidence-adaptive
    moving average of ReID embeddings, and an adaptively-weighted cosine
    similarity term joins the association cost. Use it when identities must
    survive long occlusions or crossing targets; it costs one small embedding
    network forward per frame.

    Motion parameters mirror :class:`OCSortConfig` (Deep OC-SORT has no BYTE
    low-score band, so there is no ``use_byte``).

    Args:
        det_thresh: Minimum detection score; lower-score boxes are discarded.
        max_age: Frames a track survives without an observation before deletion.
        min_hits: Consecutive hits required before a track is reported (except
            in the first ``min_hits`` frames).
        iou_threshold: Minimum IoU for a valid association.
        delta_t: Time span (frames) used to estimate a track's velocity direction.
        inertia: Weight of the velocity-direction (momentum) term in the cost.
        w_association_emb: Base weight of the appearance term in the cost.
        alpha_fixed_emb: Base EMA factor for track appearance; the effective
            factor rises toward 1 (ignore new embedding) as detection
            confidence falls toward ``det_thresh``.
        aw_param: Cap on the per-row/column adaptive-weighting boost.
        embedder: Named OSNet variant used to compute embeddings when no
            custom embedder callable is supplied to the tracker.
    """

    det_thresh: float = 0.25
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    delta_t: int = 3
    inertia: float = 0.2
    w_association_emb: float = 0.75
    alpha_fixed_emb: float = 0.95
    aw_param: float = 0.5
    embedder: str = "osnet_ain_x0_25"

    def __post_init__(self):
        _require_finite(
            self,
            "det_thresh",
            "max_age",
            "min_hits",
            "iou_threshold",
            "delta_t",
            "inertia",
            "w_association_emb",
            "alpha_fixed_emb",
            "aw_param",
        )
        if not (0 <= self.det_thresh < 1):
            # < 1 because the appearance EMA trust term divides by
            # (1 - det_thresh).
            raise ValueError(f"det_thresh must be in [0, 1), got {self.det_thresh}")
        if not (0 <= self.iou_threshold <= 1):
            raise ValueError(
                f"iou_threshold must be in [0, 1], got {self.iou_threshold}"
            )
        if self.max_age < 0:
            raise ValueError(f"max_age must be >= 0, got {self.max_age}")
        if self.min_hits < 0:
            raise ValueError(f"min_hits must be >= 0, got {self.min_hits}")
        if self.delta_t < 1:
            raise ValueError(f"delta_t must be >= 1, got {self.delta_t}")
        if self.inertia < 0:
            raise ValueError(f"inertia must be >= 0, got {self.inertia}")
        if self.w_association_emb < 0:
            raise ValueError(
                f"w_association_emb must be >= 0, got {self.w_association_emb}"
            )
        if not (0 <= self.alpha_fixed_emb <= 1):
            raise ValueError(
                f"alpha_fixed_emb must be in [0, 1], got {self.alpha_fixed_emb}"
            )
        if self.aw_param < 0:
            raise ValueError(f"aw_param must be >= 0, got {self.aw_param}")

    @classmethod
    def from_kwargs(cls, **kwargs) -> "DeepOCSortConfig":
        """Construct config, warning on unknown keys."""
        valid = {f.name for f in fields(cls)}
        unknown = set(kwargs) - valid
        if unknown:
            warnings.warn(
                f"Unknown Deep OC-SORT config keys (ignored): {sorted(unknown)}",
                stacklevel=2,
            )
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**filtered)


@dataclass(kw_only=True)
class OCSortConfig:
    """Configuration for the OC-SORT multi-object tracker.

    OC-SORT (Cao et al., CVPR 2023) is a pure-motion tracker that improves on
    ByteTrack/SORT under occlusion and non-linear motion via three ideas:
    Observation-Centric Momentum (a velocity-direction term in the association
    cost), Observation-Centric Recovery (a second association pass against each
    track's last real observation), and Observation-Centric Re-Update (smoothing
    the Kalman state along a virtual trajectory when a track is re-found).

    Args:
        det_thresh: Detections with score above this drive first-round association
            and may spawn new tracks. Lower-score boxes (down to a fixed 0.1
            floor) are only used by the optional BYTE recovery pass, so a
            ``det_thresh`` at or below 0.1 disables that pass.
        max_age: Frames a track survives without an observation before deletion.
        min_hits: Consecutive hits required before a track is reported (except in
            the first ``min_hits`` frames, mirroring the MOT benchmark protocol).
        iou_threshold: Minimum IoU for a valid association.
        delta_t: Time span (frames) used to estimate a track's velocity direction.
        inertia: Weight of the velocity-direction (momentum) term in the cost.
        use_byte: Enable the BYTE low-score recovery pass before OCR.
    """

    det_thresh: float = 0.25
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    delta_t: int = 3
    inertia: float = 0.2
    use_byte: bool = False

    def __post_init__(self):
        _require_finite(
            self,
            "det_thresh",
            "max_age",
            "min_hits",
            "iou_threshold",
            "delta_t",
            "inertia",
        )
        if not (0 <= self.det_thresh <= 1):
            raise ValueError(f"det_thresh must be in [0, 1], got {self.det_thresh}")
        if not (0 <= self.iou_threshold <= 1):
            raise ValueError(
                f"iou_threshold must be in [0, 1], got {self.iou_threshold}"
            )
        if self.max_age < 0:
            raise ValueError(f"max_age must be >= 0, got {self.max_age}")
        if self.min_hits < 0:
            raise ValueError(f"min_hits must be >= 0, got {self.min_hits}")
        if self.delta_t < 1:
            raise ValueError(f"delta_t must be >= 1, got {self.delta_t}")
        if self.inertia < 0:
            raise ValueError(f"inertia must be >= 0, got {self.inertia}")

    @classmethod
    def from_kwargs(cls, **kwargs) -> "OCSortConfig":
        """Construct config, warning on unknown keys."""
        valid = {f.name for f in fields(cls)}
        unknown = set(kwargs) - valid
        if unknown:
            warnings.warn(
                f"Unknown OC-SORT config keys (ignored): {sorted(unknown)}",
                stacklevel=2,
            )
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**filtered)
