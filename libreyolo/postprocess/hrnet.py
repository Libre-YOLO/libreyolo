"""HRNet heatmap decoding and flip-test restoration.

Adapted from ``lib/core/inference.py`` and ``lib/utils/transforms.py`` in
``leoxiaobin/deep-high-resolution-net.pytorch`` at commit
``6f69e4676ad8d43d0d61b64b1b9726f0c369e7b1`` (MIT License).

Copyright (c) Microsoft. Written by Bin Xiao.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from ..data.pose_metadata import COCO17_FLIP_IDX
from ..data.pose_metadata import COCO17_OKS_SIGMAS


def get_max_preds(batch_heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract integer heatmap maxima exactly as the upstream decoder does."""
    if not isinstance(batch_heatmaps, np.ndarray) or batch_heatmaps.ndim != 4:
        raise ValueError("batch_heatmaps must be a 4D numpy array")
    batch_size, num_keypoints, _height, width = batch_heatmaps.shape
    flattened = batch_heatmaps.reshape((batch_size, num_keypoints, -1))
    indices = np.argmax(flattened, axis=2).reshape((batch_size, num_keypoints, 1))
    max_values = np.amax(flattened, axis=2).reshape(
        (batch_size, num_keypoints, 1)
    )

    predictions = np.tile(indices, (1, 1, 2)).astype(np.float32)
    predictions[:, :, 0] %= width
    predictions[:, :, 1] = np.floor(predictions[:, :, 1] / width)
    positive = np.tile(np.greater(max_values, 0.0), (1, 1, 2)).astype(np.float32)
    return predictions * positive, max_values


def transform_preds(
    coordinates: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    output_size_wh: Sequence[int],
) -> np.ndarray:
    """Map heatmap coordinates back into the source image."""
    # Lazy import preserves the postprocess -> models direction rule in ADR 0005.
    from ..models.hrnet.utils import affine_transform, get_affine_transform

    target = np.zeros(coordinates.shape)
    transform = get_affine_transform(
        center,
        scale,
        0,
        output_size_wh,
        inverse=True,
    )
    for index in range(coordinates.shape[0]):
        target[index, 0:2] = affine_transform(coordinates[index, 0:2], transform)
    return target


def decode_heatmaps(
    batch_heatmaps: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    *,
    post_process: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode heatmaps to source-image coordinates and peak responses."""
    coordinates, max_values = get_max_preds(batch_heatmaps)
    heatmap_height, heatmap_width = batch_heatmaps.shape[2:]

    if post_process:
        for batch_index in range(coordinates.shape[0]):
            for keypoint_index in range(coordinates.shape[1]):
                heatmap = batch_heatmaps[batch_index, keypoint_index]
                px = int(math.floor(coordinates[batch_index, keypoint_index, 0] + 0.5))
                py = int(math.floor(coordinates[batch_index, keypoint_index, 1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    difference = np.asarray(
                        [
                            heatmap[py, px + 1] - heatmap[py, px - 1],
                            heatmap[py + 1, px] - heatmap[py - 1, px],
                        ]
                    )
                    coordinates[batch_index, keypoint_index] += (
                        np.sign(difference) * 0.25
                    )

    predictions = coordinates.copy()
    for batch_index in range(coordinates.shape[0]):
        predictions[batch_index] = transform_preds(
            coordinates[batch_index],
            centers[batch_index],
            scales[batch_index],
            (heatmap_width, heatmap_height),
        )
    return predictions, max_values


def flip_back(
    output_flipped: np.ndarray,
    flip_index: Sequence[int] = COCO17_FLIP_IDX,
) -> np.ndarray:
    """Horizontally restore flipped heatmaps and swap left/right keypoints."""
    if output_flipped.ndim != 4:
        raise ValueError("output_flipped must have shape [batch, keypoints, height, width]")
    return output_flipped[:, list(flip_index), :, ::-1].copy()


def flip_back_tensor(
    output_flipped: torch.Tensor,
    flip_index: Sequence[int] = COCO17_FLIP_IDX,
    *,
    shift: bool = True,
) -> torch.Tensor:
    """Torch-native flip restoration with the official optional one-pixel shift."""
    if output_flipped.ndim != 4:
        raise ValueError("output_flipped must have shape [batch, keypoints, height, width]")
    restored = output_flipped[:, list(flip_index)].flip(-1)
    if shift:
        shifted = restored.clone()
        restored[:, :, :, 1:] = shifted[:, :, :, :-1]
    return restored


def oks_iou(
    reference_keypoints: np.ndarray,
    candidate_keypoints: np.ndarray,
    reference_area: float,
    candidate_areas: np.ndarray,
    sigmas: Sequence[float] = COCO17_OKS_SIGMAS,
) -> np.ndarray:
    """Compute upstream-style object keypoint similarity for one-to-many poses."""
    if candidate_keypoints.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    variances = (np.asarray(sigmas, dtype=np.float32) * 2.0) ** 2
    reference = reference_keypoints.reshape(-1, 3)
    candidates = candidate_keypoints.reshape(candidate_keypoints.shape[0], -1, 3)
    dx = candidates[:, :, 0] - reference[None, :, 0]
    dy = candidates[:, :, 1] - reference[None, :, 1]
    average_areas = (float(reference_area) + candidate_areas[:, None]) / 2.0
    exponent = (dx**2 + dy**2) / variances[None, :]
    exponent = exponent / (average_areas + np.spacing(1)) / 2.0
    return np.exp(-exponent).mean(axis=1)


def oks_nms(
    keypoints: np.ndarray,
    scores: np.ndarray,
    areas: np.ndarray,
    threshold: float = 0.9,
) -> np.ndarray:
    """Greedily suppress poses whose OKS exceeds ``threshold``."""
    if keypoints.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        index = int(order[0])
        keep.append(index)
        overlaps = oks_iou(
            keypoints[index],
            keypoints[order[1:]],
            float(areas[index]),
            areas[order[1:]],
        )
        remaining = np.where(overlaps <= threshold)[0]
        order = order[remaining + 1]
    return np.asarray(keep, dtype=np.int64)


def _empty_pose_result(num_keypoints: int = 17) -> dict[str, np.ndarray | int]:
    return {
        "num_detections": 0,
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int64),
        "keypoints": np.zeros((0, num_keypoints, 3), dtype=np.float32),
    }


def postprocess_hrnet(
    heatmaps: torch.Tensor | np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    boxes: np.ndarray,
    box_scores: np.ndarray,
    *,
    keypoint_threshold: float = 0.2,
    oks_threshold: float = 0.9,
    max_det: int = 300,
) -> dict[str, np.ndarray | int]:
    """Decode, rescore, and OKS-suppress one batch of HRNet person poses."""
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().float().cpu().numpy()
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    if heatmaps.ndim != 4:
        raise ValueError(f"heatmaps must have shape (N, K, H, W), got {heatmaps.shape}")
    num_instances, num_keypoints = heatmaps.shape[:2]
    if num_instances == 0:
        return _empty_pose_result(num_keypoints)

    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    scales = np.asarray(scales, dtype=np.float32).reshape(-1, 2)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    box_scores = np.asarray(box_scores, dtype=np.float32).reshape(-1)
    lengths = {num_instances, len(centers), len(scales), len(boxes), len(box_scores)}
    if len(lengths) != 1:
        raise ValueError(
            "heatmaps, centers, scales, boxes, and box_scores must have the same "
            f"instance count, got {sorted(lengths)}"
        )

    points, peak_responses = decode_heatmaps(
        heatmaps,
        centers,
        scales,
        post_process=True,
    )
    keypoints = np.concatenate((points, peak_responses), axis=2).astype(
        np.float32,
        copy=False,
    )

    responses = peak_responses[:, :, 0]
    visible = responses > float(keypoint_threshold)
    visible_counts = visible.sum(axis=1)
    mean_responses = np.divide(
        (responses * visible).sum(axis=1),
        visible_counts,
        out=np.zeros((num_instances,), dtype=np.float32),
        where=visible_counts != 0,
    )
    scores = (box_scores * mean_responses).astype(np.float32, copy=False)
    areas = np.prod(scales * 200.0, axis=1)
    keep = oks_nms(keypoints, scores, areas, threshold=float(oks_threshold))
    keep = keep[: max(0, int(max_det))]

    return {
        "num_detections": int(len(keep)),
        "boxes": boxes[keep],
        "scores": scores[keep],
        "classes": np.zeros((len(keep),), dtype=np.int64),
        "keypoints": keypoints[keep],
    }
