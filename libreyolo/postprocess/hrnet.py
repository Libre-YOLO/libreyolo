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
