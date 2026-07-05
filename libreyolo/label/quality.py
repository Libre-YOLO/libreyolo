"""Geometry linter for LibreLabel: catch labels a detector can't learn from.

Pure, dependency-free checks over normalised annotations. The headline check is
the one people forget: a box only a few pixels across *at the training image
size* carries almost no signal -- the model literally cannot learn it -- yet it
inflates your object count and your confidence. We also flag absurd aspect
ratios and whole-frame boxes, which are nearly always slips.

Like the rest of the assist layer, this only *reports*; it never edits a label.
"""

from __future__ import annotations

from typing import List, Optional


def _box_wh(a) -> Optional[tuple]:
    """Normalised ``(w, h)`` of a box annotation, or a polygon's bbox size."""
    if a.get("type") == "poly":
        pts = a.get("points") or []
        xs, ys = pts[0::2], pts[1::2]
        if not xs or not ys:
            return None
        return (max(xs) - min(xs), max(ys) - min(ys))
    try:
        return (float(a["w"]), float(a["h"]))
    except (KeyError, TypeError, ValueError):
        return None


def lint_annotations(anns, *, imgsz: int = 640, min_px: float = 4.0,
                     max_ar: float = 20.0, full_frac: float = 0.985) -> List[dict]:
    """Return geometry issues for one image's normalised annotations.

    Each issue: ``{"i": annotation_index, "type": ..., "msg": ...}``. ``type`` is
    one of ``tiny`` (smaller than ``min_px`` at ``imgsz``), ``fullframe``
    (covers the whole image), or ``sliver`` (aspect ratio beyond ``max_ar``).
    """
    issues: List[dict] = []
    for i, a in enumerate(anns):
        wh = _box_wh(a)
        if wh is None:
            continue
        nw, nh = wh
        pw, ph = nw * imgsz, nh * imgsz
        if pw < min_px or ph < min_px:
            issues.append({"i": i, "type": "tiny",
                           "msg": f"~{min(pw, ph):.1f}px at imgsz {imgsz} - too small to learn"})
            continue
        if nw >= full_frac and nh >= full_frac:
            issues.append({"i": i, "type": "fullframe",
                           "msg": "covers the whole frame - likely a slip"})
            continue
        ar = max(nw / nh, nh / nw) if nw > 0 and nh > 0 else 999.0
        if ar > max_ar:
            issues.append({"i": i, "type": "sliver",
                           "msg": f"aspect ratio {ar:.0f}:1 - extreme sliver"})
    return issues
