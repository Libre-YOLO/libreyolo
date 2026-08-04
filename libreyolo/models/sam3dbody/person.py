"""Person-source layer for body-mesh inference.

Top-down mesh models need a person box before they can regress anything. Rather
than binding one detector in, this defines a small protocol plus adapters, so a
caller can hand over boxes directly, plug in a LibreYOLO detector, or pass any
callable. Mirrors the arrangement the gaze task uses for faces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol

import numpy as np


@dataclass
class PersonBox:
    """A single person detection in original-image pixels."""

    xyxy: tuple
    score: float = 1.0


class PersonDetector(Protocol):
    """Anything mapping a numpy RGB image to a list of ``PersonBox``."""

    def __call__(self, image_rgb: np.ndarray) -> List[PersonBox]:  # pragma: no cover
        ...


@dataclass
class CallablePersonDetector:
    """Adapt an arbitrary ``image -> boxes`` callable into the protocol."""

    fn: Callable[[np.ndarray], Any]
    min_score: float = 0.0

    def __call__(self, image_rgb: np.ndarray) -> List[PersonBox]:
        return normalize_person_boxes(self.fn(image_rgb), self.min_score)


@dataclass
class LibreYOLOPersonDetector:
    """Adapt a LibreYOLO detection model into a person detector.

    Defaults to COCO class 0 ("person"). Pass ``person_class=None`` to keep
    every detection, for a model that only ever emits people.
    """

    model: Any
    conf: float = 0.4
    person_class: Optional[int] = 0
    imgsz: Optional[int] = None

    def __call__(self, image_rgb: np.ndarray) -> List[PersonBox]:
        from PIL import Image

        kwargs: dict = {"conf": self.conf}
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        if self.person_class is not None:
            kwargs["classes"] = [self.person_class]
        result = self.model(Image.fromarray(image_rgb), **kwargs)
        if isinstance(result, list):
            result = result[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy
        conf = boxes.conf
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
        if hasattr(conf, "cpu"):
            conf = conf.cpu().numpy()
        return [
            PersonBox(
                xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                score=float(s),
            )
            for b, s in zip(xyxy, conf)
        ]


def normalize_person_boxes(raw: Any, min_score: float = 0.0) -> List[PersonBox]:
    """Coerce flexible box input into a list of ``PersonBox``."""
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        if raw.ndim == 1:
            raw = raw[None, :]
        if raw.ndim != 2 or raw.shape[1] not in (4, 5):
            raise ValueError(
                f"expected person boxes of shape (N, 4) or (N, 5), got {raw.shape}"
            )
        rows = raw.tolist()
    else:
        rows = list(raw)

    boxes: List[PersonBox] = []
    for item in rows:
        if isinstance(item, PersonBox):
            if item.score >= min_score:
                boxes.append(item)
            continue
        seq = list(item)
        if len(seq) == 4:
            score = 1.0
        elif len(seq) >= 5:
            score = float(seq[4])
        else:
            raise ValueError(
                f"unsupported person-box length {len(seq)}; expected 4 or 5+"
            )
        if score < min_score:
            continue
        boxes.append(
            PersonBox(
                xyxy=(float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3])),
                score=score,
            )
        )
    return boxes


def resolve_person_detector(spec: Any) -> Optional[PersonDetector]:
    """Coerce a user-supplied detector spec into a ``PersonDetector`` or None."""
    if spec is None:
        return None
    if isinstance(spec, (CallablePersonDetector, LibreYOLOPersonDetector)):
        return spec
    from ..base.model import BaseModel

    if isinstance(spec, BaseModel):
        return LibreYOLOPersonDetector(model=spec)
    if callable(spec):
        return CallablePersonDetector(fn=spec)
    raise TypeError(
        f"Unsupported person_detector spec: {type(spec).__name__}. "
        "Provide a callable, a LibreYOLO model, or a PersonDetector instance."
    )
