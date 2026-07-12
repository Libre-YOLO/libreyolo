"""Predict keyword compatibility policy."""

from __future__ import annotations

import math
import warnings
from numbers import Integral, Real
from typing import Any, Optional


NOOP_PREDICT_KWARGS = {
    "agnostic_nms",
    "boxes",
    "dnn",
    "half",
    "line_width",
    "retina_masks",
    "show_conf",
    "show_labels",
    "stream_buffer",
    "verbose",
}
REJECTED_PREDICT_KWARGS = {"visualize", "embed"}
ACCEPTED_PREDICT_KWARGS = {
    "classes",
    "conf",
    "device",
    "imgsz",
    "iou",
    "max_det",
    "augment",
    "save",
    "stream",
    "vid_stride",
}


def validate_predict_inputs(
    *,
    names: Any,
    conf: Real,
    iou: Real,
    classes: Any,
    max_det: int,
    batch: int,
    vid_stride: int,
    overlap_ratio: Real | None = None,
) -> Optional[list[int]]:
    """Validate shared native/exported public predict options."""

    def probability(name: str, value: Real, *, upper_exclusive: bool = False) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
        numeric = float(value)
        upper_ok = numeric < 1.0 if upper_exclusive else numeric <= 1.0
        interval = "[0, 1)" if upper_exclusive else "[0, 1]"
        if not math.isfinite(numeric) or numeric < 0.0 or not upper_ok:
            raise ValueError(f"{name} must be finite and in {interval}, got {value!r}.")

    def positive_int(name: str, value: int, *, allow_zero: bool = False) -> None:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
        minimum = 0 if allow_zero else 1
        if int(value) < minimum:
            relation = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {relation}, got {value!r}.")

    probability("conf", conf)
    probability("iou", iou)
    if overlap_ratio is not None:
        probability("overlap_ratio", overlap_ratio, upper_exclusive=True)
    positive_int("max_det", max_det, allow_zero=True)
    positive_int("batch", batch)
    positive_int("vid_stride", vid_stride)

    if classes is None:
        return None
    if isinstance(classes, Integral) and not isinstance(classes, bool):
        normalized = [int(classes)]
    elif isinstance(classes, (list, tuple)):
        normalized = []
        for class_id in classes:
            if isinstance(class_id, bool) or not isinstance(class_id, Integral):
                raise TypeError(
                    "classes must contain integer class IDs; "
                    f"got {class_id!r}."
                )
            normalized.append(int(class_id))
    else:
        raise TypeError(
            "classes must be an integer class ID or a list/tuple of integer IDs."
        )

    negative = [class_id for class_id in normalized if class_id < 0]
    if negative:
        raise ValueError(f"classes must contain non-negative IDs, got {negative}.")

    if isinstance(names, dict) and names:
        valid_ids = {key for key in names if isinstance(key, Integral)}
    elif isinstance(names, (list, tuple)) and names:
        valid_ids = set(range(len(names)))
    else:
        valid_ids = set()
    unknown = sorted(set(normalized) - valid_ids) if valid_ids else []
    if unknown:
        raise ValueError(
            f"classes contains unknown class IDs {unknown}; valid IDs are "
            f"{sorted(valid_ids)}."
        )
    return normalized


def normalize_predict_kwargs(kwargs: dict, passthrough: set[str] | None = None) -> dict:
    """Warn or fail for predict kwargs LibreYOLO does not implement."""
    passthrough = passthrough or set()
    remaining = dict(kwargs)

    rejected = sorted(k for k in remaining if k in REJECTED_PREDICT_KWARGS)
    if rejected:
        raise NotImplementedError(
            "LibreYOLO does not support these predict options: "
            f"{', '.join(rejected)}."
        )

    noops = sorted(k for k in remaining if k in NOOP_PREDICT_KWARGS)
    for key in noops:
        warnings.warn(
            f"Predict option {key!r} is accepted for CLI compatibility but is "
            "currently a no-op in LibreYOLO.",
            stacklevel=3,
        )
        remaining.pop(key, None)

    for key in ACCEPTED_PREDICT_KWARGS:
        remaining.pop(key, None)

    forwarded = {}
    for key in sorted(passthrough):
        if key in remaining:
            forwarded[key] = remaining.pop(key)

    if remaining:
        raise TypeError(
            "Unsupported predict option(s): "
            f"{', '.join(sorted(remaining))}. "
            "Supported options include conf, iou, imgsz, device, classes, "
            "max_det, save, stream, and vid_stride."
        )

    return forwarded
