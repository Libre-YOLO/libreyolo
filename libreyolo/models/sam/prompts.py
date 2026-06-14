"""Pure prompt-normalization for the LibreSAM tier.

The promptable-segmentation API accepts points and boxes in the flexible,
user-friendly shapes documented for the de-facto standard interface, then
coerces them into the single canonical nesting the underlying processor
expects:

    points -> [point_batch, num_points, 2]   (one entry per object/mask)
    labels -> [point_batch, num_points]
    boxes  -> [num_boxes, 4]

Nesting depth carries meaning, matching the documented public surface:

    points=[x, y]                      one object, one point
    points=[[x, y], [x, y]]            two objects, one point each
    points=[[[x, y], [x, y]]]          one object, two points (grouped)

Everything here is pure (lists in, lists out) so it can be unit-tested fully
offline without torch, transformers, or model weights.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def _to_list(x):
    """Recursively convert array-likes/tuples to plain nested lists.

    numpy arrays and torch tensors (the natural carriers of click/box
    coordinates in CV code) are coerced via ``tolist()`` so a valid ``(N, 2)``
    array is accepted rather than rejected as "nesting depth 0".
    """
    if hasattr(x, "tolist") and not isinstance(x, (list, tuple, str, bytes)):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_list(v) for v in x]
    return x


def _depth(x) -> int:
    """Nesting depth of a (non-empty) nested list; 0 for a scalar."""
    d = 0
    while isinstance(x, (list, tuple)) and len(x) > 0:
        d += 1
        x = x[0]
    return d


def normalize_points(points) -> Optional[List[List[List[float]]]]:
    """Coerce user points into ``[point_batch, num_points, 2]``.

    Returns ``None`` when ``points`` is ``None`` so callers can omit the key.
    """
    if points is None:
        return None
    pts = _to_list(points)
    d = _depth(pts)
    if d == 1:  # [x, y] -> one object, one point
        canonical = [[list(pts)]]
    elif d == 2:  # [[x, y], ...] -> N objects, one point each
        canonical = [[list(p)] for p in pts]
    elif d == 3:  # [[[x, y], ...], ...] -> objects x points, as given
        canonical = [[list(p) for p in obj] for obj in pts]
    else:
        raise ValueError(
            "points must be [x, y], [[x, y], ...], or [[[x, y], ...], ...]; "
            f"got nesting depth {d}."
        )
    for obj in canonical:
        for p in obj:
            if len(p) != 2:
                raise ValueError(f"each point must be [x, y]; got {p!r}.")
    return canonical


def normalize_labels(
    labels, canonical_points: Optional[Sequence[Sequence[Sequence[float]]]]
) -> Optional[List[List[int]]]:
    """Coerce point labels to ``[point_batch, num_points]`` matching points.

    ``1`` = positive (object), ``0`` = negative (background). When ``labels`` is
    ``None`` every point defaults to positive. The shape must line up with the
    normalized points or a ``ValueError`` is raised.
    """
    if canonical_points is None:
        if labels is not None:
            raise ValueError("labels given without points.")
        return None
    if labels is None:
        return [[1] * len(obj) for obj in canonical_points]

    lbls = _to_list(labels)
    d = _depth(lbls)
    if d == 1:  # [l, ...] -> one label per object (one point each)
        out = [[int(v)] for v in lbls]
    elif d == 2:  # [[l, ...], ...] -> labels per object, as given
        out = [[int(v) for v in obj] for obj in lbls]
    else:
        raise ValueError(
            f"labels must be [l, ...] or [[l, ...], ...]; got nesting depth {d}."
        )

    if len(out) != len(canonical_points):
        raise ValueError(
            f"labels has {len(out)} objects but points has "
            f"{len(canonical_points)}."
        )
    for lab, obj in zip(out, canonical_points):
        if len(lab) != len(obj):
            raise ValueError(
                "labels and points disagree on number of points per object: "
                f"{len(lab)} vs {len(obj)}."
            )
    return out


def normalize_boxes(boxes) -> Optional[List[List[float]]]:
    """Coerce user boxes into ``[num_boxes, 4]`` xyxy. ``None`` passes through."""
    if boxes is None:
        return None
    bx = _to_list(boxes)
    d = _depth(bx)
    if d == 1:  # [x1, y1, x2, y2] -> single box
        out = [list(bx)]
    elif d == 2:  # [[x1, y1, x2, y2], ...] -> as given
        out = [list(b) for b in bx]
    else:
        raise ValueError(
            "boxes must be [x1, y1, x2, y2] or [[x1, y1, x2, y2], ...]; "
            f"got nesting depth {d}."
        )
    for b in out:
        if len(b) != 4:
            raise ValueError(f"each box must be [x1, y1, x2, y2]; got {b!r}.")
    return out


def build_point_grid(points_per_side: int) -> List[List[float]]:
    """A uniform grid of ``points_per_side**2`` points, normalized to [0, 1].

    Used to drive "segment everything" by prompting the model on a dense grid,
    the standard automatic-mask-generation strategy.
    """
    if points_per_side < 1:
        raise ValueError("points_per_side must be >= 1.")
    offset = 1.0 / (2 * points_per_side)
    grid = []
    for j in range(points_per_side):
        for i in range(points_per_side):
            grid.append([offset + i / points_per_side, offset + j / points_per_side])
    return grid
