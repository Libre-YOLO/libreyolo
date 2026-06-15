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
            # Accept "0" or "0.0"; our own writer always emits a bare int.
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(p) for p in parts[1:5])
        except ValueError:
            has_non_box = True
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
    lines = [
        f"{int(b['cls'])} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
        for b in boxes
    ]
    return ("\n".join(lines) + "\n") if lines else ""


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def sanitize_boxes(boxes: List[Box], nc: Optional[int] = None) -> List[Box]:
    """Validate/clamp boxes before writing.

    Drops boxes with a negative class, a class ``>= nc`` (when ``nc`` is known),
    or zero/negative area; clamps every coordinate into ``[0, 1]``.
    """
    out: List[Box] = []
    for b in boxes:
        try:
            cls = int(b["cls"])
        except (KeyError, TypeError, ValueError):
            continue
        if cls < 0 or (nc is not None and cls >= nc):
            continue
        cx, cy = _clamp01(float(b["cx"])), _clamp01(float(b["cy"]))
        w, h = _clamp01(float(b["w"])), _clamp01(float(b["h"]))
        if w <= 0.0 or h <= 0.0:
            continue
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
            cls = int(float(parts[0]))
        except (ValueError, IndexError):
            continue
        nums = parts[1:]
        if len(nums) == 4:
            try:
                cx, cy, w, h = (float(p) for p in nums)
            except ValueError:
                continue
            anns.append({"type": "box", "cls": cls, "cx": cx, "cy": cy, "w": w, "h": h})
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            try:
                pts = [float(p) for p in nums]
            except ValueError:
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
    return False


def format_annotations(anns: List[dict]) -> str:
    """Serialize mixed box/polygon annotations to YOLO label text."""
    lines: List[str] = []
    for a in anns:
        cls = int(a["cls"])
        if a.get("type") == "poly":
            pts = a.get("points") or []
            if len(pts) >= 6:
                lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in pts))
        else:
            lines.append(
                f"{cls} {a['cx']:.6f} {a['cy']:.6f} {a['w']:.6f} {a['h']:.6f}"
            )
    return ("\n".join(lines) + "\n") if lines else ""


def sanitize_annotations(anns: List[dict], nc: Optional[int] = None) -> List[dict]:
    """Validate/clamp mixed annotations before writing."""
    out: List[dict] = []
    for a in anns:
        try:
            cls = int(a["cls"])
        except (KeyError, TypeError, ValueError):
            continue
        if cls < 0 or (nc is not None and cls >= nc):
            continue
        if a.get("type") == "poly":
            pts = [_clamp01(float(v)) for v in (a.get("points") or [])]
            if len(pts) < 6 or len(pts) % 2 != 0:
                continue
            xs, ys = pts[0::2], pts[1::2]
            # Shoelace area: a width/height (bbox) check misses *diagonal* collinear
            # polygons -- e.g. (0,0),(0.5,0.5),(1,1) has positive bbox extents but
            # zero area. Drop any polygon whose actual area is ~0 so we never write a
            # degenerate segment that breaks training after vertex edits.
            n = len(xs)
            area2 = abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                            for i in range(n)))
            if area2 <= 1e-8:   # 2*area in normalised coords; ~0 -> collinear/collapsed
                continue
            out.append({"type": "poly", "cls": cls, "points": pts})
        else:
            cx, cy = _clamp01(float(a["cx"])), _clamp01(float(a["cy"]))
            w, h = _clamp01(float(a["w"])), _clamp01(float(a["h"]))
            if w <= 0.0 or h <= 0.0:
                continue
            out.append({"type": "box", "cls": cls, "cx": cx, "cy": cy, "w": w, "h": h})
    return out
