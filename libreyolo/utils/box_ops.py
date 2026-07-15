"""Unified bounding box IoU operations for LibreYOLO."""

import math

import numpy as np
import torch
from torch import Tensor


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) boxes in xyxy format.
        boxes2: (M, 4) boxes in xyxy format.

    Returns:
        IoU matrix of shape (N, M).
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


def compute_iou(
    bbox1: Tensor,
    bbox2: Tensor,
    mode: str = "iou",
    *,
    pairwise: bool = True,
) -> Tensor:
    """
    Calculate IoU, DIoU, or CIoU between two sets of bounding boxes.

    Args:
        bbox1: Bounding boxes in xyxy format. Shape: (A, 4) or (B, A, 4)
        bbox2: Bounding boxes in xyxy format. Shape: (B, 4) or (B, B, 4)
        mode: IoU variant - "iou", "diou", or "ciou"
        pairwise: Whether to compute all pairwise IoUs for 2D inputs.

    Returns:
        IoU tensor. Shape depends on input dimensions.
    """
    mode = mode.lower()
    EPS = 1e-7
    dtype = bbox1.dtype
    bbox1 = bbox1.to(torch.float32)
    bbox2 = bbox2.to(torch.float32)

    if bbox1.ndim == 2 and bbox2.ndim == 2:
        if pairwise:
            bbox1 = bbox1.unsqueeze(1)  # (A, 4) -> (A, 1, 4)
            bbox2 = bbox2.unsqueeze(0)  # (B, 4) -> (1, B, 4)
        else:
            if bbox1.shape != bbox2.shape:
                raise ValueError(
                    "bbox1 and bbox2 must have the same shape for elementwise IoU"
                )
    elif bbox1.ndim == 3 and bbox2.ndim == 3:
        bbox1 = bbox1.unsqueeze(2)  # (BZ, A, 4) -> (BZ, A, 1, 4)
        bbox2 = bbox2.unsqueeze(1)  # (BZ, B, 4) -> (BZ, 1, B, 4)

    xmin_inter = torch.max(bbox1[..., 0], bbox2[..., 0])
    ymin_inter = torch.max(bbox1[..., 1], bbox2[..., 1])
    xmax_inter = torch.min(bbox1[..., 2], bbox2[..., 2])
    ymax_inter = torch.min(bbox1[..., 3], bbox2[..., 3])

    intersection_area = torch.clamp(xmax_inter - xmin_inter, min=0) * torch.clamp(
        ymax_inter - ymin_inter, min=0
    )

    area_bbox1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area_bbox2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    union_area = area_bbox1 + area_bbox2 - intersection_area

    iou = intersection_area / (union_area + EPS)
    if mode == "iou":
        return iou.to(dtype)

    # Centroid distance
    cx1 = (bbox1[..., 2] + bbox1[..., 0]) / 2
    cy1 = (bbox1[..., 3] + bbox1[..., 1]) / 2
    cx2 = (bbox2[..., 2] + bbox2[..., 0]) / 2
    cy2 = (bbox2[..., 3] + bbox2[..., 1]) / 2
    cent_dis = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Diagonal of smallest enclosing box
    c_x = torch.max(bbox1[..., 2], bbox2[..., 2]) - torch.min(
        bbox1[..., 0], bbox2[..., 0]
    )
    c_y = torch.max(bbox1[..., 3], bbox2[..., 3]) - torch.min(
        bbox1[..., 1], bbox2[..., 1]
    )
    diag_dis = c_x**2 + c_y**2 + EPS

    diou = iou - (cent_dis / diag_dis)
    if mode == "diou":
        return diou.to(dtype)

    # Aspect ratio penalty (CIoU)
    arctan = torch.atan(
        (bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + EPS)
    ) - torch.atan(
        (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + EPS)
    )
    v = (4 / (math.pi**2)) * (arctan**2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + EPS)
    ciou = diou - alpha * v
    return ciou.to(dtype)


# =============================================================================
# Oriented (rotated) boxes
# =============================================================================


def rbox_corners(boxes: Tensor) -> Tensor:
    """Convert ``xywhr`` rotated boxes to their four corners.

    Matches the corner order of :func:`libreyolo.data.obb.xywhr_to_corners`.

    Args:
        boxes: ``(..., 5)`` as ``(cx, cy, w, h, angle)``, angle in radians.

    Returns:
        ``(..., 4, 2)`` corners.
    """
    cx, cy, w, h, angle = boxes.unbind(-1)
    cos, sin = angle.cos(), angle.sin()
    wx, wy = cos * w / 2, sin * w / 2
    hx, hy = -sin * h / 2, cos * h / 2
    return torch.stack(
        (
            torch.stack((cx - wx - hx, cy - wy - hy), -1),
            torch.stack((cx + wx - hx, cy + wy - hy), -1),
            torch.stack((cx + wx + hx, cy + wy + hy), -1),
            torch.stack((cx - wx + hx, cy - wy + hy), -1),
        ),
        dim=-2,
    )


def _convex_quad_intersection_area(
    poly_a: Tensor, poly_b: Tensor, eps: float = 1e-7
) -> Tensor:
    """Exact intersection area of paired convex quadrilaterals.

    The intersection of two convex polygons is the convex hull of: each
    polygon's vertices that lie inside the other, plus every edge-edge
    crossing. Both inputs are rectangles, so at most eight of the twenty-four
    candidate points survive. They are ordered by angle about their centroid
    and the area follows from the shoelace formula; rejected slots collapse
    onto the first surviving vertex, where they enclose no area.

    Args:
        poly_a: ``(K, 4, 2)`` corners.
        poly_b: ``(K, 4, 2)`` corners.

    Returns:
        ``(K,)`` intersection areas.
    """

    def _vertices_inside(points: Tensor, polygon: Tensor) -> Tensor:
        edge_start = polygon[:, None, :, :]
        edge_end = polygon.roll(-1, dims=-2)[:, None, :, :]
        probe = points[:, :, None, :]
        edge = edge_end - edge_start
        offset = probe - edge_start
        cross = edge[..., 0] * offset[..., 1] - edge[..., 1] * offset[..., 0]
        # Convex and consistently wound: inside means every cross product agrees.
        return (cross >= -eps).all(-1) | (cross <= eps).all(-1)

    inside_a = _vertices_inside(poly_a, poly_b)
    inside_b = _vertices_inside(poly_b, poly_a)

    a_start = poly_a[:, :, None, :]
    a_dir = (poly_a.roll(-1, dims=-2) - poly_a)[:, :, None, :]
    b_start = poly_b[:, None, :, :]
    b_dir = (poly_b.roll(-1, dims=-2) - poly_b)[:, None, :, :]

    denom = a_dir[..., 0] * b_dir[..., 1] - a_dir[..., 1] * b_dir[..., 0]
    parallel = denom.abs() < eps
    safe = torch.where(parallel, torch.ones_like(denom), denom)
    delta = b_start - a_start
    t = (delta[..., 0] * b_dir[..., 1] - delta[..., 1] * b_dir[..., 0]) / safe
    u = (delta[..., 0] * a_dir[..., 1] - delta[..., 1] * a_dir[..., 0]) / safe
    hit = ~parallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    crossings = a_start + t[..., None] * a_dir

    points = torch.cat((poly_a, poly_b, crossings.flatten(1, 2)), dim=1)
    valid = torch.cat((inside_a, inside_b, hit.flatten(1, 2)), dim=1)

    count = valid.sum(-1, keepdim=True)
    centroid = (points * valid[..., None]).sum(-2, keepdim=True) / count.clamp_min(
        1
    ).unsqueeze(-1)
    offset = points - centroid
    angle = torch.atan2(offset[..., 1], offset[..., 0]).masked_fill(
        ~valid, float("inf")
    )

    order = angle.argsort(-1)
    points = points.gather(-2, order[..., None].expand_as(points))
    valid = valid.gather(-1, order)
    points = torch.where(valid[..., None], points, points[:, 0:1, :])

    following = points.roll(-1, dims=-2)
    area = 0.5 * (
        points[..., 0] * following[..., 1] - following[..., 0] * points[..., 1]
    ).sum(-1).abs()
    return torch.where(count.squeeze(-1) >= 3, area, torch.zeros_like(area))


def rotated_iou_pairwise(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Exact IoU of rotated boxes, one pair per row.

    Args:
        boxes1: ``(K, 5)`` ``xywhr`` boxes.
        boxes2: ``(K, 5)`` ``xywhr`` boxes.

    Returns:
        ``(K,)`` IoU values.
    """
    if boxes1.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0],))
    intersection = _convex_quad_intersection_area(
        rbox_corners(boxes1), rbox_corners(boxes2), eps
    )
    area1 = (boxes1[:, 2] * boxes1[:, 3]).clamp_min(0)
    area2 = (boxes2[:, 2] * boxes2[:, 3]).clamp_min(0)
    union = area1 + area2 - intersection
    return (intersection / union.clamp_min(eps)).clamp(0.0, 1.0)


def _aabb_overlap(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Which ``(i, j)`` pairs have overlapping axis-aligned envelopes.

    Rotated boxes can only intersect when their axis-aligned envelopes do, so
    this cheap test discards the overwhelming majority of pairs before any
    polygon work is attempted.
    """
    corners1 = rbox_corners(boxes1)
    corners2 = rbox_corners(boxes2)
    lo1, hi1 = corners1.amin(-2), corners1.amax(-2)
    lo2, hi2 = corners2.amin(-2), corners2.amax(-2)
    return (
        (lo1[:, None, 0] < hi2[None, :, 0])
        & (lo2[None, :, 0] < hi1[:, None, 0])
        & (lo1[:, None, 1] < hi2[None, :, 1])
        & (lo2[None, :, 1] < hi1[:, None, 1])
    )


def rotated_iou_matrix(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Exact pairwise IoU between two sets of rotated boxes.

    Vectorized equivalent of :func:`libreyolo.data.obb.xywhr_iou`, which
    evaluates a single pair at a time through OpenCV. Pairs whose axis-aligned
    envelopes miss are known to have zero IoU, so only the surviving pairs
    reach the polygon intersection; on dense aerial imagery that is a small
    fraction of the matrix.

    Args:
        boxes1: ``(N, 5)`` ``xywhr`` boxes.
        boxes2: ``(M, 5)`` ``xywhr`` boxes.

    Returns:
        ``(N, M)`` IoU matrix.
    """
    n, m = boxes1.shape[0], boxes2.shape[0]
    iou = boxes1.new_zeros((n, m))
    if n == 0 or m == 0:
        return iou

    rows, cols = _aabb_overlap(boxes1, boxes2).nonzero(as_tuple=True)
    if rows.numel() == 0:
        return iou
    iou[rows, cols] = rotated_iou_pairwise(boxes1[rows], boxes2[cols], eps)
    return iou


def rotated_nms(
    xywhr: Tensor,
    scores: Tensor,
    class_ids: Tensor,
    iou_thres: float,
    max_det: int,
) -> Tensor:
    """Class-aware greedy NMS over rotated ``xywhr`` boxes.

    The suppression order and threshold semantics are the textbook greedy
    ones; the geometry is the exact vectorized rotated IoU
    (:func:`rotated_iou_pairwise`). Ranking by score means a suppressor
    always precedes its victim, so only the upper triangle can suppress, and
    same-class plus axis-aligned-envelope overlap gate which pairs reach the
    polygon intersection at all. Shared by the YOLO9 postprocess and the
    oriented-box TTA merge so both de-duplicate identically.
    """
    if xywhr.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=xywhr.device)

    finite_mask = torch.isfinite(xywhr).all(dim=1) & torch.isfinite(scores)
    if not finite_mask.all():
        valid_indices = torch.where(finite_mask)[0]
        if len(valid_indices) == 0:
            return torch.zeros(0, dtype=torch.long, device=xywhr.device)
        xywhr = xywhr[finite_mask]
        scores = scores[finite_mask]
        class_ids = class_ids[finite_mask]
    else:
        valid_indices = None

    order = torch.argsort(scores, descending=True)
    ranked = xywhr[order]
    ranked_classes = class_ids[order]
    count = ranked.shape[0]

    candidates = (
        _aabb_overlap(ranked, ranked)
        & (ranked_classes[:, None] == ranked_classes[None, :])
    ).triu(diagonal=1)
    rows, cols = candidates.nonzero(as_tuple=True)

    suppresses = np.zeros((count, count), dtype=bool)
    if rows.numel():
        overlapping = rotated_iou_pairwise(ranked[rows], ranked[cols]) > iou_thres
        suppresses[
            rows[overlapping].cpu().numpy(), cols[overlapping].cpu().numpy()
        ] = True

    alive = np.ones(count, dtype=bool)
    keep_local: list[int] = []
    for candidate in range(count):
        if not alive[candidate]:
            continue
        keep_local.append(candidate)
        if len(keep_local) >= max_det:
            break
        alive &= ~suppresses[candidate]

    keep = order[torch.as_tensor(keep_local, dtype=torch.long, device=xywhr.device)]
    if valid_indices is not None:
        keep = valid_indices[keep]
    return keep
