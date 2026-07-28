"""Metrics for human body mesh recovery.

The field reports three numbers, all in millimeters and all lower-is-better:

* **MPJPE**, mean per-joint position error after aligning the root joint.
  Measures pose accuracy while ignoring where the person is in the scene.
* **PA-MPJPE** (also written MPJPE-PA or "reconstruction error"), the same
  quantity after a full Procrustes alignment (rotation, uniform scale and
  translation). Removes global orientation and body-size error, isolating the
  articulated pose itself.
* **PVE** (also called MPVPE), mean per-vertex position error after root
  alignment, which unlike the joint metrics is sensitive to body shape.

Kept as free functions rather than folded into a validator class so the math
can be tested directly against hand-computed cases.
"""

from __future__ import annotations

from typing import Optional

import torch


def _root_align(
    points: torch.Tensor, root_index: int | None, root: Optional[torch.Tensor]
) -> torch.Tensor:
    if root is not None:
        return points - root[:, None, :]
    if root_index is None:
        return points - points.mean(dim=1, keepdim=True)
    return points - points[:, root_index : root_index + 1, :]


def mpjpe(
    pred: torch.Tensor,
    target: torch.Tensor,
    root_index: int | None = 0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean per-joint position error after root alignment.

    Args:
        pred: ``(N, J, 3)`` predicted joints.
        target: ``(N, J, 3)`` ground-truth joints, same units as ``pred``.
        root_index: Joint index to align on. ``None`` centers on the centroid.
        mask: Optional ``(N, J)`` boolean mask of valid joints.

    Returns:
        Scalar tensor: the mean error over all valid joints.
    """
    _check_pair(pred, target)
    pred = _root_align(pred, root_index, None)
    target = _root_align(target, root_index, None)
    error = torch.linalg.norm(pred - target, dim=-1)
    return _masked_mean(error, mask)


def pve(
    pred: torch.Tensor,
    target: torch.Tensor,
    root: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean per-vertex position error after root alignment.

    Args:
        pred: ``(N, V, 3)`` predicted vertices.
        target: ``(N, V, 3)`` ground-truth vertices.
        root: Optional ``(N, 3)`` alignment point per sample. Defaults to the
            vertex centroid, since a mesh has no canonical root vertex.
    """
    _check_pair(pred, target)
    pred = _root_align(pred, None, root)
    target = _root_align(target, None, root)
    return torch.linalg.norm(pred - target, dim=-1).mean()


def procrustes_align(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Similarity-align ``pred`` onto ``target`` per sample.

    Solves the orthogonal Procrustes problem for rotation, uniform scale and
    translation (the classic Umeyama solution) and returns the transformed
    prediction. Reflections are excluded: a mirrored body is a different pose,
    not an aligned one, so the sign of the smallest singular value is flipped
    when the naive solution would reflect.

    Args:
        pred: ``(N, P, 3)`` points to align.
        target: ``(N, P, 3)`` reference points.
    """
    _check_pair(pred, target)
    # Solved in double precision: the SVD of a near-degenerate configuration
    # is where this silently loses accuracy in float32.
    pred64, target64 = pred.double(), target.double()

    pred_mean = pred64.mean(dim=1, keepdim=True)
    target_mean = target64.mean(dim=1, keepdim=True)
    pred_c = pred64 - pred_mean
    target_c = target64 - target_mean

    variance = (pred_c**2).sum(dim=(1, 2), keepdim=True)
    covariance = pred_c.transpose(1, 2) @ target_c
    u, s, vh = torch.linalg.svd(covariance)

    # Force a proper rotation (det = +1) rather than a reflection.
    det = torch.linalg.det(u @ vh)
    sign = torch.ones_like(s)
    sign[:, -1] = torch.sign(det)
    rotation = (u * sign[:, None, :]) @ vh
    trace = (s * sign).sum(dim=1).reshape(-1, 1, 1)

    scale = trace / variance.clamp(min=1e-12)
    aligned = scale * (pred_c @ rotation) + target_mean
    return aligned.to(pred.dtype)


def pa_mpjpe(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean per-joint position error after full Procrustes alignment."""
    aligned = procrustes_align(pred, target)
    error = torch.linalg.norm(aligned - target, dim=-1)
    return _masked_mean(error, mask)


def _check_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"prediction and target must match, got {tuple(pred.shape)} "
            f"and {tuple(target.shape)}"
        )
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(
            f"expected (N, P, 3) point sets, got {tuple(pred.shape)}"
        )


def _masked_mean(error: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return error.mean()
    mask = mask.to(error.dtype)
    total = mask.sum()
    if total == 0:
        return torch.zeros((), dtype=error.dtype, device=error.device)
    return (error * mask).sum() / total


def mesh_metrics(
    pred_joints: torch.Tensor,
    target_joints: torch.Tensor,
    pred_vertices: Optional[torch.Tensor] = None,
    target_vertices: Optional[torch.Tensor] = None,
    root_index: int | None = 0,
    mask: Optional[torch.Tensor] = None,
    scale_to_mm: float = 1000.0,
) -> dict:
    """Compute the standard mesh-recovery metric set.

    Inputs are assumed metric (meters); ``scale_to_mm`` converts to the
    millimeters the literature reports in.
    """
    metrics = {
        "metrics/mpjpe": float(mpjpe(pred_joints, target_joints, root_index, mask))
        * scale_to_mm,
        "metrics/pa_mpjpe": float(pa_mpjpe(pred_joints, target_joints, mask))
        * scale_to_mm,
    }
    if pred_vertices is not None and target_vertices is not None:
        metrics["metrics/pve"] = float(pve(pred_vertices, target_vertices)) * scale_to_mm
    return metrics
