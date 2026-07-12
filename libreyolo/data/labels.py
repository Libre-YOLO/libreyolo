"""Shared validation for normalized YOLO text-label rows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence


def label_row_error(
    message: str,
    *,
    label_path: str | Path | None = None,
    line_number: int | None = None,
) -> ValueError:
    """Build a label error whose source can be located without a second scan."""
    if label_path is None:
        return ValueError(message)
    location = str(label_path)
    if line_number is not None:
        location = f"{location}:{line_number}"
    return ValueError(f"{location}: {message}")


def parse_yolo_class_id(
    token: str,
    *,
    num_classes: int | None = None,
    label_path: str | Path | None = None,
    line_number: int | None = None,
    task: str = "YOLO",
) -> int:
    """Parse an integral, non-negative class id with an optional upper bound."""
    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise label_row_error(
            f"{task} class id must be numeric, got {token!r}",
            label_path=label_path,
            line_number=line_number,
        ) from exc
    if not math.isfinite(value) or not value.is_integer():
        raise label_row_error(
            f"{task} class id must be an integer, got {token!r}",
            label_path=label_path,
            line_number=line_number,
        )

    class_id = int(value)
    if class_id < 0:
        raise label_row_error(
            f"{task} class id must be non-negative, got {class_id}",
            label_path=label_path,
            line_number=line_number,
        )
    if num_classes is not None:
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise ValueError(
                f"num_classes must be a positive integer, got {num_classes!r}"
            )
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if class_id >= num_classes:
            raise label_row_error(
                f"{task} class id {class_id} out of range [0, {num_classes - 1}]",
                label_path=label_path,
                line_number=line_number,
            )
    return class_id


def _parse_normalized_values(
    tokens: Sequence[str],
    *,
    field_name: str,
    label_path: str | Path | None,
    line_number: int | None,
) -> list[float]:
    try:
        values = [float(token) for token in tokens]
    except (TypeError, ValueError) as exc:
        raise label_row_error(
            f"{field_name} must be numeric",
            label_path=label_path,
            line_number=line_number,
        ) from exc
    if not all(math.isfinite(value) for value in values):
        raise label_row_error(
            f"{field_name} must be finite",
            label_path=label_path,
            line_number=line_number,
        )
    if any(value < 0.0 or value > 1.0 for value in values):
        raise label_row_error(
            f"{field_name} must be normalized to [0, 1]",
            label_path=label_path,
            line_number=line_number,
        )
    return values


def parse_yolo_box_or_segment_label_line(
    line: str | Sequence[str],
    *,
    num_classes: int | None = None,
    label_path: str | Path | None = None,
    line_number: int | None = None,
) -> tuple[int, tuple[float, float, float, float], list[float] | None]:
    """Parse one detection row or polygon row in normalized YOLO format.

    Returns ``(class_id, (cx, cy, width, height), polygon)``. ``polygon`` is
    ``None`` for a five-field detection row. Polygon rows contain at least
    three vertices and must have non-zero shoelace area.
    """
    parts = line.split() if isinstance(line, str) else list(line)
    if len(parts) == 5:
        class_id = parse_yolo_class_id(
            parts[0],
            num_classes=num_classes,
            label_path=label_path,
            line_number=line_number,
        )
        cx, cy, width, height = _parse_normalized_values(
            parts[1:],
            field_name="YOLO box coordinates",
            label_path=label_path,
            line_number=line_number,
        )
        if width <= 0.0 or height <= 0.0:
            raise label_row_error(
                "YOLO box width and height must be positive",
                label_path=label_path,
                line_number=line_number,
            )
        return class_id, (cx, cy, width, height), None

    coordinate_count = len(parts) - 1
    if coordinate_count < 6 or coordinate_count % 2:
        raise label_row_error(
            "YOLO label rows must contain 5 box fields or an even number "
            "of polygon coordinates for at least 3 vertices",
            label_path=label_path,
            line_number=line_number,
        )

    class_id = parse_yolo_class_id(
        parts[0],
        num_classes=num_classes,
        label_path=label_path,
        line_number=line_number,
    )
    coordinates = _parse_normalized_values(
        parts[1:],
        field_name="YOLO polygon coordinates",
        label_path=label_path,
        line_number=line_number,
    )
    xs = coordinates[0::2]
    ys = coordinates[1::2]
    twice_area = abs(
        sum(
            xs[index] * ys[(index + 1) % len(xs)]
            - ys[index] * xs[(index + 1) % len(xs)]
            for index in range(len(xs))
        )
    )
    if twice_area <= 0.0:
        raise label_row_error(
            "YOLO polygon must be non-degenerate",
            label_path=label_path,
            line_number=line_number,
        )

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0.0 or height <= 0.0:
        raise label_row_error(
            "YOLO polygon must have positive width and height",
            label_path=label_path,
            line_number=line_number,
        )
    return (
        class_id,
        ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, width, height),
        coordinates,
    )


__all__ = [
    "label_row_error",
    "parse_yolo_box_or_segment_label_line",
    "parse_yolo_class_id",
]
