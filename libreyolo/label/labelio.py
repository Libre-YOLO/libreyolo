"""Pure parse/serialize helpers for YOLO detection (bounding-box) labels.

This module is LibreLabel's *format oracle* for v1 (boxes only). It is
deliberately dependency-free and side-effect-free so it can be unit-tested in
isolation, and it mirrors LibreYOLO's own on-disk contract (see
``libreyolo/data/dataset.py`` and ``libreyolo/data/utils.py``)::

    <cls> <cx> <cy> <w> <h>

one box per line, ``cls`` an **integer** class index and ``cx cy w h`` the box
centre and size normalised to ``[0, 1]``.

A line with anything other than exactly five whitespace-separated fields is a
polygon / OBB annotation. Those are reported back via ``has_non_box`` so callers
can refuse to overwrite them in box-only mode rather than silently destroy them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import List, Optional, Tuple, TypedDict


class Box(TypedDict):
    """A single axis-aligned box, normalised to ``[0, 1]``."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float


def parse_label_text(text: str) -> Tuple[List[Box], bool]:
    """Parse YOLO label text into boxes.

    Returns ``(boxes, has_non_box)`` where ``has_non_box`` is ``True`` if any
    non-empty line was not a 5-field box (i.e. a polygon/OBB row, or malformed),
    so the caller can treat the file as not box-editable.
    """
    boxes: List[Box] = []
    has_non_box = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            has_non_box = True
            continue
        try:
            # Integer-class contract (matches parse_annotations + the training
            # loader): a fractional token like "0.5" is non-box/unsupported, never
            # coerced to 0.
            cls = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:5])
        except (ValueError, OverflowError):
            has_non_box = True
            continue
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            has_non_box = True   # nan/inf coords aren't a valid box -> don't overwrite
            continue
        boxes.append(Box(cls=cls, cx=cx, cy=cy, w=w, h=h))
    return boxes, has_non_box


def format_label_text(boxes: List[Box]) -> str:
    """Serialize boxes to YOLO label text.

    ``cls`` is written as a **bare integer** (``"0"``, never ``"0.0"``): the
    LibreYOLO loader does ``int(parts[0])`` and a float token would raise and
    abort the whole image's label load. Coordinates use ``%.6f``. An empty list
    serialises to ``""`` (an empty ``.txt`` == a valid background image).
    """
    clean = sanitize_boxes(boxes)
    lines = [
        f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
        for b in clean
    ]
    return ("\n".join(lines) + "\n") if lines else ""


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _class_id(value, nc: Optional[int], where: str) -> int:
    """Return an exact integer class id, never a truncation of a float."""
    if isinstance(value, bool):
        raise ValueError(f"{where}: class id must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{where}: class id must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{where}: class id must be a finite integer")
    cls = int(numeric)
    if cls < 0 or (nc is not None and cls >= nc):
        limit = f"0..{nc - 1}" if nc is not None else "zero or greater"
        raise ValueError(f"{where}: class id {cls} is outside {limit}")
    return cls


def _finite_float(value, field: str, where: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{where}: {field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{where}: {field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{where}: {field} must be finite")
    return result


def _disk_float(value: float) -> float:
    """Canonical value represented by the six-decimal on-disk format."""
    return float(f"{value:.6f}")


def _polygon_area2(points: Sequence[float]) -> float:
    xs, ys = points[0::2], points[1::2]
    n = len(xs)
    return abs(
        sum(
            xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
            for i in range(n)
        )
    )


def _clip_obb(points: List[float], where: str) -> List[float]:
    """Translate an out-of-canvas OBB into bounds without shearing it."""
    xs, ys = points[0::2], points[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if not _obb_overlaps_unit_square(points):
        raise ValueError(f"{where}: oriented box does not overlap the image canvas")
    if max_x - min_x > 1.0 + 1e-9 or max_y - min_y > 1.0 + 1e-9:
        raise ValueError(f"{where}: oriented box is larger than the image canvas")
    dx = max(0.0, -min_x) - max(0.0, max_x - 1.0)
    dy = max(0.0, -min_y) - max(0.0, max_y - 1.0)
    shifted = [
        _clamp01(value + (dx if i % 2 == 0 else dy))
        for i, value in enumerate(points)
    ]
    return shifted


def _obb_overlaps_unit_square(points: Sequence[float]) -> bool:
    """Strict positive-area intersection using the separating-axis theorem."""
    polygon = [(points[index], points[index + 1]) for index in range(0, 8, 2)]
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    axes = [(1.0, 0.0), (0.0, 1.0)]
    for index in range(4):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % 4]
        axes.append((-(y2 - y1), x2 - x1))
    for axis_x, axis_y in axes:
        norm = math.hypot(axis_x, axis_y)
        if norm <= 1e-12:
            continue
        obb_projection = [x * axis_x + y * axis_y for x, y in polygon]
        square_projection = [x * axis_x + y * axis_y for x, y in square]
        overlap = min(max(obb_projection), max(square_projection)) - max(
            min(obb_projection), min(square_projection)
        )
        if overlap <= 1e-12 * norm:
            return False
    return True


def _clip_polygon_to_unit_square(points: Sequence[float]) -> List[float]:
    """Clip a polygon to ``[0, 1]²`` with Sutherland-Hodgman clipping."""
    vertices = [
        (float(points[index]), float(points[index + 1]))
        for index in range(0, len(points), 2)
    ]

    def clip(vertices, inside, intersection):
        if not vertices:
            return []
        output = []
        start = vertices[-1]
        start_inside = inside(start)
        for end in vertices:
            end_inside = inside(end)
            if end_inside:
                if not start_inside:
                    output.append(intersection(start, end))
                output.append(end)
            elif start_inside:
                output.append(intersection(start, end))
            start, start_inside = end, end_inside
        return output

    def at_x(start, end, boundary):
        ratio = (boundary - start[0]) / (end[0] - start[0])
        return boundary, start[1] + ratio * (end[1] - start[1])

    def at_y(start, end, boundary):
        ratio = (boundary - start[1]) / (end[1] - start[1])
        return start[0] + ratio * (end[0] - start[0]), boundary

    vertices = clip(vertices, lambda point: point[0] >= 0.0,
                    lambda start, end: at_x(start, end, 0.0))
    vertices = clip(vertices, lambda point: point[0] <= 1.0,
                    lambda start, end: at_x(start, end, 1.0))
    vertices = clip(vertices, lambda point: point[1] >= 0.0,
                    lambda start, end: at_y(start, end, 0.0))
    vertices = clip(vertices, lambda point: point[1] <= 1.0,
                    lambda start, end: at_y(start, end, 1.0))

    deduplicated = []
    for vertex in vertices:
        if not deduplicated or any(
            abs(vertex[axis] - deduplicated[-1][axis]) > 1e-12
            for axis in (0, 1)
        ):
            deduplicated.append(vertex)
    if len(deduplicated) > 1 and all(
        abs(deduplicated[0][axis] - deduplicated[-1][axis]) <= 1e-12
        for axis in (0, 1)
    ):
        deduplicated.pop()
    return [coordinate for vertex in deduplicated for coordinate in vertex]


def _validate_obb_rectangle(points: Sequence[float], where: str) -> None:
    """Require four cyclic corners of a rectangle, within text-rounding tolerance."""
    vertices = [
        (points[index], points[index + 1]) for index in range(0, 8, 2)
    ]
    edges = [
        (
            vertices[(index + 1) % 4][0] - vertices[index][0],
            vertices[(index + 1) % 4][1] - vertices[index][1],
        )
        for index in range(4)
    ]
    lengths = [math.hypot(x, y) for x, y in edges]
    if min(lengths) <= 1e-8:
        raise ValueError(f"{where}: oriented box corners must be distinct")
    span = max(
        max(points[0::2]) - min(points[0::2]),
        max(points[1::2]) - min(points[1::2]),
        1e-4,
    )
    coordinate_tolerance = 2e-5 * span + 2e-6
    # Diagonals of a parallelogram bisect each other. Combined with one right
    # angle this is exactly a rectangle and rejects trapezoids/bow-ties.
    midpoint_error = max(
        abs(vertices[0][axis] + vertices[2][axis] - vertices[1][axis] - vertices[3][axis])
        for axis in (0, 1)
    )
    right_angle_error = abs(edges[0][0] * edges[1][0] + edges[0][1] * edges[1][1])
    # Each serialized coordinate is rounded to six decimals, so an edge component
    # can accumulate ~1e-6 error at both ends. Include that absolute dot-product
    # error as well as the scale-relative angular tolerance; otherwise legitimate
    # thin rotated rectangles are rejected merely because their short edge is tiny.
    quantization_tolerance = 2e-6 * (lengths[0] + lengths[1]) + 4e-12
    angular_tolerance = max(
        2e-5 * lengths[0] * lengths[1] + 2e-10,
        quantization_tolerance,
    )
    if midpoint_error > coordinate_tolerance or right_angle_error > angular_tolerance:
        raise ValueError(
            f"{where}: four OBB corners must form an oriented rectangle in cyclic order"
        )


def sanitize_boxes(boxes: List[Box], nc: Optional[int] = None) -> List[Box]:
    """Validate boxes before writing, raising instead of silently changing them."""
    if not isinstance(boxes, (list, tuple)):
        raise ValueError("boxes must be a list")
    out: List[Box] = []
    for i, b in enumerate(boxes):
        where = f"box {i}"
        if not isinstance(b, Mapping):
            raise ValueError(f"{where}: expected an object")
        cls = _class_id(b.get("cls"), nc, where)
        cx = _finite_float(b.get("cx"), "cx", where)
        cy = _finite_float(b.get("cy"), "cy", where)
        w = _finite_float(b.get("w"), "w", where)
        h = _finite_float(b.get("h"), "h", where)
        if any(v < 0.0 or v > 1.0 for v in (cx, cy, w, h)):
            raise ValueError(f"{where}: coordinates must be normalized to [0, 1]")
        cx, cy, w, h = (_disk_float(value) for value in (cx, cy, w, h))
        if w <= 0.0 or h <= 0.0:
            raise ValueError(
                f"{where}: width and height must remain positive at six-decimal precision"
            )
        out.append(Box(cls=cls, cx=cx, cy=cy, w=w, h=h))
    return out


# --- Annotations: boxes (5 fields) OR polygons (cls + >=6 even coords) ---------
# A polygon line is ``cls x1 y1 x2 y2 ... xn yn`` (normalised), matching
# LibreYOLO's segmentation/OBB on-disk format (``data/dataset.py``,
# ``data/obb.py``). A 4-vertex polygon round-trips an OBB's corner format
# byte-identically, so reading/writing OBB rows as polygons never corrupts them.


def parse_annotations(text: str) -> List[dict]:
    """Parse a YOLO label file into mixed box/polygon annotations.

    Each box: ``{"type": "box", "cls", "cx", "cy", "w", "h"}``.
    Each polygon: ``{"type": "poly", "cls", "points": [x1, y1, ...]}`` (normalised).
    Malformed lines are skipped.
    """
    anns: List[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls = int(parts[0])   # integer class contract: never coerce "0.5" -> 0
        except (ValueError, IndexError, OverflowError):
            continue
        nums = parts[1:]
        if len(nums) == 4:
            try:
                cx, cy, w, h = (float(p) for p in nums)
            except ValueError:
                continue
            if not all(math.isfinite(v) for v in (cx, cy, w, h)):
                continue   # nan/inf -> skip so NaN never reaches stats/Radar/JSON
            anns.append({"type": "box", "cls": cls, "cx": cx, "cy": cy, "w": w, "h": h})
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            try:
                pts = [float(p) for p in nums]
            except ValueError:
                continue
            if not all(math.isfinite(v) for v in pts):
                continue
            anns.append({"type": "poly", "cls": cls, "points": pts})
    return anns


def has_unsupported_rows(text: str) -> bool:
    """True if any non-empty line is not a 5-field box or a valid polygon row.

    Keypoint/pose rows (``cls bbox kpt...``) and malformed rows fall here. A file
    with such rows must stay **read-only** so saving (which keeps only the parsed
    boxes/polygons) can't silently drop the unparsed fields.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            int(parts[0])   # class must be an integer token; "0.5" is unsupported
            nums = [float(p) for p in parts[1:]]
        except (ValueError, IndexError, OverflowError):
            return True
        if not (len(nums) == 4 or (len(nums) >= 6 and len(nums) % 2 == 0)):
            return True
        if not all(math.isfinite(n) for n in nums):
            return True   # nan/inf coords -> malformed; keep the file read-only
    return False


def has_out_of_bounds_coords(text: str) -> bool:
    """True if any normalized coordinate is outside ``[0, 1]`` (beyond float-format
    tolerance) or non-finite.

    Such rows violate the normalized file contract. Keep the file read-only so an
    unrelated save cannot rewrite geometry the user has not explicitly repaired.
    """
    tol = 1e-6
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            nums = [float(p) for p in parts[1:]]
        except (ValueError, IndexError):
            continue   # malformed numbers handled by has_unsupported_rows
        if not (len(nums) == 4 or (len(nums) >= 6 and len(nums) % 2 == 0)):
            continue
        if any((not math.isfinite(n)) or n < -tol or n > 1.0 + tol for n in nums):
            return True
    return False


def has_obb_shaped_rows(text: str) -> bool:
    """True if any row is exactly ``cls + 8 coords`` (a 4-point quad).

    Per ``docs/dataset_schema.md`` a 9-field row is an oriented box in ``obb`` mode
    but a 4-vertex polygon in ``segment`` mode -- byte-identical, so when the dataset
    declares no task we can't tell them apart and must keep them read-only rather than
    risk rewriting an OBB as an arbitrary polygon. (Real segmentation masks use many
    vertices, so the cost of protecting genuine quad polygons is small.)
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            int(parts[0])
            nums = [float(p) for p in parts[1:]]
        except (ValueError, IndexError, OverflowError):
            continue
        if len(nums) == 8:
            return True
    return False


def has_zero_area_box(text: str) -> bool:
    """True if any box row (``cls cx cy w h``) has ``w <= 0`` or ``h <= 0``.

    The strict write validator rejects such boxes. Keep an existing malformed file
    read-only so an unrelated edit cannot implicitly repair or delete the row.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            int(parts[0])
            nums = [float(p) for p in parts[1:]]
        except (ValueError, IndexError, OverflowError):
            continue
        if len(nums) == 4 and (nums[2] <= 0.0 or nums[3] <= 0.0):
            return True
    return False


def has_degenerate_polygon(text: str) -> bool:
    """True if any polygon row has ~zero clamped shoelace area (``area2 <= 1e-8``).

    A file containing one stays read-only so an unrelated edit/save cannot rewrite the
    malformed row before the user repairs it. Mirrors the sanitizer's
    clamp-then-area sequence exactly so read/write stay in lockstep.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            int(parts[0])
            nums = [float(p) for p in parts[1:]]
        except (ValueError, IndexError, OverflowError):
            continue   # malformed -> has_unsupported_rows
        if not (len(nums) >= 6 and len(nums) % 2 == 0):
            continue   # only polygon rows are area-checked
        if not all(math.isfinite(v) for v in nums):
            continue   # nan/inf -> has_unsupported_rows
        pts = [_clamp01(v) for v in nums]   # clamp BEFORE the area, exactly like sanitize
        xs, ys = pts[0::2], pts[1::2]
        n = len(xs)
        area2 = abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n)))
        if area2 <= 1e-8:
            return True
    return False


def has_out_of_range_rows(text: str, nc: Optional[int]) -> bool:
    """True if any row's integer class is outside ``[0, nc)``.

    Such a file must stay read-only so an unrelated edit/save cannot rewrite the
    invalid annotation before the user repairs it. (Malformed/non-integer rows are
    handled by :func:`has_unsupported_rows`.)
    """
    if not nc or nc <= 0:
        return False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            c = int(parts[0])
        except (ValueError, IndexError, OverflowError):
            continue
        if c < 0 or c >= nc:
            return True
    return False


def format_annotations(anns: List[dict]) -> str:
    """Serialize mixed box/polygon annotations to YOLO label text."""
    lines: List[str] = []
    for a in anns:
        cls = _class_id(a.get("cls"), None, "annotation")
        if a.get("type") == "poly":
            pts = a.get("points") or []
            if len(pts) >= 6:
                lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in pts))
        else:
            lines.append(
                f"{cls} {a['cx']:.6f} {a['cy']:.6f} {a['w']:.6f} {a['h']:.6f}"
            )
    return ("\n".join(lines) + "\n") if lines else ""


def sanitize_annotations(
    anns: List[dict], nc: Optional[int] = None, task: Optional[str] = None
) -> List[dict]:
    """Validate mixed annotations and return their canonical persisted geometry.

    Boxes must already satisfy the normalized on-disk contract. Polygon vertices
    are clipped to the image canvas because interactive vertex moves may cross an
    edge. OBB quads are translated as a whole when possible so clipping does not
    shear a rectangle. Invalid input raises ``ValueError``; a successful save never
    reports fewer annotations than the caller supplied.
    """
    if not isinstance(anns, (list, tuple)):
        raise ValueError("annotations must be a list")
    normalized_task = str(task or "").strip().lower()
    out: List[dict] = []
    for i, a in enumerate(anns):
        where = f"annotation {i}"
        if not isinstance(a, Mapping):
            raise ValueError(f"{where}: expected an object")
        cls = _class_id(a.get("cls"), nc, where)
        kind = a.get("type") or "box"
        if kind not in ("box", "poly"):
            raise ValueError(f"{where}: type must be 'box' or 'poly'")
        if normalized_task == "detect" and kind != "box":
            raise ValueError(f"{where}: detection annotations require boxes")
        if normalized_task == "obb" and kind != "poly":
            raise ValueError(f"{where}: OBB annotations require exactly 4 corners")
        if kind == "poly":
            values = a.get("points")
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"{where}: polygon points must be a list")
            raw_pts = [
                _finite_float(value, f"point {j}", where)
                for j, value in enumerate(values)
            ]
            if len(raw_pts) < 6 or len(raw_pts) % 2 != 0:
                raise ValueError(f"{where}: polygon needs at least 3 coordinate pairs")
            if normalized_task == "obb":
                if len(raw_pts) != 8:
                    raise ValueError(f"{where}: OBB annotations require exactly 4 corners")
                if _polygon_area2(raw_pts) <= 1e-8:
                    raise ValueError(f"{where}: polygon must have non-zero area")
                _validate_obb_rectangle(raw_pts, where)
                pts = _clip_obb(raw_pts, where)
            else:
                pts = _clip_polygon_to_unit_square(raw_pts)
                if len(pts) < 6:
                    raise ValueError(
                        f"{where}: polygon does not overlap the image canvas"
                    )
            pts = [_disk_float(value) for value in pts]
            if any(
                not math.isfinite(value) or value < 0.0 or value > 1.0
                for value in pts
            ):
                raise ValueError(
                    f"{where}: clipped polygon points must remain finite and normalized"
                )
            if len(set(zip(pts[0::2], pts[1::2], strict=True))) < 3:
                raise ValueError(
                    f"{where}: polygon needs 3 distinct points at six-decimal precision"
                )
            # Shoelace area catches collinear/collapsed polygons after clipping.
            if _polygon_area2(pts) <= 1e-8:
                raise ValueError(f"{where}: polygon must have non-zero area")
            if normalized_task == "obb":
                _validate_obb_rectangle(pts, where)
            out.append({"type": "poly", "cls": cls, "points": pts})
        else:
            cx = _finite_float(a.get("cx"), "cx", where)
            cy = _finite_float(a.get("cy"), "cy", where)
            w = _finite_float(a.get("w"), "w", where)
            h = _finite_float(a.get("h"), "h", where)
            if any(v < 0.0 or v > 1.0 for v in (cx, cy, w, h)):
                raise ValueError(f"{where}: box coordinates must be normalized to [0, 1]")
            cx, cy, w, h = (_disk_float(value) for value in (cx, cy, w, h))
            if w <= 0.0 or h <= 0.0:
                raise ValueError(
                    f"{where}: box width and height must remain positive at six-decimal precision"
                )
            out.append({"type": "box", "cls": cls, "cx": cx, "cy": cy, "w": w, "h": h})
    return out
