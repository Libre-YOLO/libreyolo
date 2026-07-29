"""Camera helpers shared by body-mesh models.

Top-down mesh regressors see a square crop around one person, so everything
they predict is expressed relative to that crop. The mesh task contract is
stated on the original image instead: ``transl`` is metric in the full-image
camera frame and ``joints2d`` is in original-image pixels. These helpers are
the bridge, kept separate from any one model so the conversion is testable on
its own.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def default_focal_length(image_height: int, image_width: int) -> float:
    """Focal length guess for an image with unknown intrinsics.

    Uses the image diagonal, which corresponds to roughly a 55 degree
    horizontal field of view on a 4:3 frame. This is only a fallback: pass real
    intrinsics, or a predicted focal length, whenever they are available, since
    metric depth scales directly with this number.
    """
    return float(math.sqrt(image_height**2 + image_width**2))


def crop_cam_to_full_image(
    crop_cam: torch.Tensor,
    box_center: torch.Tensor,
    box_size: torch.Tensor,
    image_size: Tuple[int, int],
    focal_length: torch.Tensor | float,
) -> torch.Tensor:
    """Lift a weak-perspective crop camera to a full-image translation.

    Args:
        crop_cam: ``(N, 3)`` weak-perspective ``(s, tx, ty)`` predicted on the
            crop, where ``s`` is scale and ``tx, ty`` are crop-normalized
            offsets.
        box_center: ``(N, 2)`` person box centers in original-image pixels.
        box_size: ``(N,)`` side length of the square crop in original-image
            pixels.
        image_size: ``(height, width)`` of the original image.
        focal_length: scalar or ``(N,)`` focal length in pixels.

    Returns:
        ``(N, 3)`` metric translation in the full-image camera frame.
    """
    height, width = image_size
    scale, tx, ty = crop_cam[:, 0], crop_cam[:, 1], crop_cam[:, 2]

    if not isinstance(focal_length, torch.Tensor):
        focal_length = torch.full_like(scale, float(focal_length))
    focal_length = focal_length.reshape(-1).to(scale)

    # Depth follows from how large the person appears inside the crop: a
    # smaller predicted scale means the person is further away.
    box_size = box_size.reshape(-1).to(scale)
    z = 2.0 * focal_length / (box_size * scale + 1e-9)

    # Re-center from the crop's own frame onto the image principal point.
    cx = box_center[:, 0] - width * 0.5
    cy = box_center[:, 1] - height * 0.5
    x = tx + cx * z / focal_length
    y = ty + cy * z / focal_length
    return torch.stack([x, y, z], dim=-1)


def perspective_project(
    points3d: torch.Tensor,
    focal_length: torch.Tensor | float,
    principal_point: Optional[torch.Tensor] = None,
    image_size: Optional[Tuple[int, int]] = None,
    translation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project metric camera-frame points to pixels.

    Args:
        points3d: ``(N, P, 3)`` points in the camera frame.
        focal_length: scalar or ``(N,)`` focal length in pixels.
        principal_point: ``(N, 2)`` principal point in pixels. Defaults to the
            image center when ``image_size`` is given.
        image_size: ``(height, width)``, used only to derive a default
            principal point.
        translation: optional ``(N, 3)`` translation added before projecting,
            for points that are still root-relative.

    Returns:
        ``(N, P, 2)`` pixel coordinates.
    """
    if points3d.ndim != 3:
        raise ValueError(
            f"expected (N, P, 3) points, got shape {tuple(points3d.shape)}"
        )
    points = points3d if translation is None else points3d + translation[:, None, :]

    batch = points.shape[0]
    if not isinstance(focal_length, torch.Tensor):
        focal_length = torch.full((batch,), float(focal_length), device=points.device)
    focal_length = focal_length.reshape(-1).to(points).reshape(batch, 1)

    if principal_point is None:
        if image_size is None:
            raise ValueError(
                "perspective_project needs principal_point or image_size"
            )
        height, width = image_size
        principal_point = torch.tensor(
            [width * 0.5, height * 0.5], device=points.device, dtype=points.dtype
        ).expand(batch, 2)
    principal_point = principal_point.to(points).reshape(batch, 1, 2)

    # Guard against division by a vanishing or negative depth for points that
    # land behind the camera; those pixels are meaningless but must not be inf.
    z = points[..., 2:3].clamp(min=1e-6)
    return focal_length[..., None] * points[..., :2] / z + principal_point
