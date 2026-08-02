"""Person-detector protocol and adapters for top-down HRNet pose inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class PersonBox:
    """One person detection in source-image ``xyxy`` coordinates."""

    xyxy: tuple[float, float, float, float]
    score: float = 1.0


class PersonDetector(Protocol):
    """Anything callable that maps an RGB numpy image to person boxes."""

    def __call__(self, image_rgb: np.ndarray) -> list[PersonBox]: ...


def normalize_person_boxes(
    raw: Any,
    *,
    min_score: float = 0.0,
) -> list[PersonBox]:
    """Coerce flexible ``xyxy``/``xyxy+score`` inputs to ``PersonBox``."""
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return []
        if raw.ndim != 2 or raw.shape[1] < 4:
            raise ValueError(
                "person-box array must have shape (N, 4+) with xyxy first, "
                f"got {raw.shape}"
            )
        iterable = raw.tolist()
    else:
        iterable = raw

    boxes: list[PersonBox] = []
    for item in iterable:
        if isinstance(item, PersonBox):
            person = item
        else:
            values = list(item)
            if len(values) < 4:
                raise ValueError(
                    f"person box must contain at least four values, got {values!r}"
                )
            score = float(values[4]) if len(values) >= 5 else 1.0
            person = PersonBox(
                xyxy=tuple(float(value) for value in values[:4]),
                score=score,
            )
        if person.score >= min_score:
            boxes.append(person)
    return boxes


@dataclass
class CallablePersonDetector:
    """Adapt an arbitrary ``image -> boxes`` callable to the protocol."""

    function: Callable[[np.ndarray], Any]
    min_score: float = 0.0

    def __call__(self, image_rgb: np.ndarray) -> list[PersonBox]:
        return normalize_person_boxes(
            self.function(image_rgb),
            min_score=self.min_score,
        )


def _person_class_id(model: Any, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    names = getattr(model, "names", None)
    items = names.items() if isinstance(names, dict) else enumerate(names or [])
    for class_id, name in items:
        if str(name).strip().lower() == "person":
            return int(class_id)
    raise ValueError(
        "The paired detector has no class named 'person'. Pass person_class=<id> "
        "when constructing LibreYOLOPersonDetector."
    )


@dataclass
class LibreYOLOPersonDetector:
    """Adapt any LibreYOLO detector, including YOLO9 and RF-DETR."""

    model: Any
    conf: float = 0.05
    iou: float = 0.45
    max_det: int = 300
    person_class: int | None = None
    imgsz: int | None = None

    def __call__(self, image_rgb: np.ndarray) -> list[PersonBox]:
        from PIL import Image

        class_id = _person_class_id(self.model, self.person_class)
        kwargs: dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "classes": [class_id],
        }
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        result = self.model(Image.fromarray(image_rgb), **kwargs)
        if isinstance(result, list):
            result = result[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy
        confidence = boxes.conf
        classes = boxes.cls
        if hasattr(xyxy, "detach"):
            xyxy = xyxy.detach().cpu().numpy()
        if hasattr(confidence, "detach"):
            confidence = confidence.detach().cpu().numpy()
        if hasattr(classes, "detach"):
            classes = classes.detach().cpu().numpy()
        return [
            PersonBox(
                xyxy=tuple(float(value) for value in box[:4]),
                score=float(score),
            )
            for box, score, detected_class in zip(xyxy, confidence, classes)
            if int(detected_class) == class_id and float(score) >= self.conf
        ]


def default_person_detector(device: str = "auto") -> LibreYOLOPersonDetector:
    """Build the default lightweight LibreYOLO9 person detector lazily."""
    from ... import LibreYOLO

    return LibreYOLOPersonDetector(
        model=LibreYOLO("LibreYOLO9t.pt", device=device),
    )


def resolve_person_detector(
    specification: Any,
    *,
    device: str = "auto",
) -> PersonDetector | None:
    """Resolve a protocol instance, LibreYOLO model, callable, or named default."""
    if specification is None:
        return None
    if isinstance(specification, str):
        name = specification.strip().lower()
        if name in {"auto", "yolo9"}:
            return default_person_detector(device=device)
        if name == "rfdetr":
            from ... import LibreYOLO

            return LibreYOLOPersonDetector(
                model=LibreYOLO("LibreRFDETRn.pt", device=device),
            )
        raise ValueError(
            f"Unknown person_detector {specification!r}; use 'auto', 'yolo9', "
            "'rfdetr', a LibreYOLO detector, or a callable."
        )
    if isinstance(
        specification,
        (CallablePersonDetector, LibreYOLOPersonDetector),
    ):
        return specification
    if (
        hasattr(specification, "names")
        and hasattr(specification, "task")
        and callable(specification)
    ):
        task = str(getattr(specification, "task", "detect"))
        if task != "detect":
            raise ValueError(
                f"person_detector must be a detection model, got task={task!r}"
            )
        return LibreYOLOPersonDetector(model=specification)
    if callable(specification):
        return CallablePersonDetector(function=specification)
    raise TypeError(
        f"Unsupported person_detector type {type(specification).__name__}; "
        "provide a LibreYOLO detector, callable, or PersonDetector adapter."
    )
