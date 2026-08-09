"""BSDS-style edge validation with in-process non-maximum thinning.

The validator reports optimal-dataset-scale (ODS) and optimal-image-scale
(OIS) F-measures. Pixel correspondence uses one-to-one bipartite matching
within ``max_dist * image_diagonal`` so a thick prediction cannot claim the
same ground-truth edge more than once.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

from ..data.edge_dataset import EdgeDataset, edge_collate_fn, resolve_edge_data
from .base import BaseValidator

logger = logging.getLogger(__name__)


def thin_edge_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Thin edge probabilities with four-direction gradient NMS.

    Accepts ``(H, W)``, ``(B, H, W)``, or ``(B, 1, H, W)`` tensors and returns
    the same shape. Suppressed pixels are zero and retained pixels preserve
    their original probability.
    """
    probabilities = torch.as_tensor(probabilities)
    original_shape = probabilities.shape
    if probabilities.ndim == 2:
        work = probabilities[None, None]
    elif probabilities.ndim == 3:
        work = probabilities[:, None]
    elif probabilities.ndim == 4 and probabilities.shape[1] == 1:
        work = probabilities
    else:
        raise ValueError(
            "edge thinning expects (H,W), (B,H,W), or (B,1,H,W), got "
            f"{tuple(probabilities.shape)}"
        )

    work = work.float()
    sobel_x = work.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])[
        None, None
    ]
    sobel_y = work.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])[
        None, None
    ]
    grad_x = F.conv2d(work, sobel_x, padding=1)
    grad_y = F.conv2d(work, sobel_y, padding=1)
    angle = torch.rad2deg(torch.atan2(grad_y, grad_x)).remainder(180.0)

    padded = F.pad(work, (1, 1, 1, 1), mode="constant", value=0.0)
    left = padded[..., 1:-1, :-2]
    right = padded[..., 1:-1, 2:]
    up = padded[..., :-2, 1:-1]
    down = padded[..., 2:, 1:-1]
    up_left = padded[..., :-2, :-2]
    down_right = padded[..., 2:, 2:]
    up_right = padded[..., :-2, 2:]
    down_left = padded[..., 2:, :-2]

    direction_0 = (angle < 22.5) | (angle >= 157.5)
    direction_45 = (angle >= 22.5) & (angle < 67.5)
    direction_90 = (angle >= 67.5) & (angle < 112.5)
    direction_135 = (angle >= 112.5) & (angle < 157.5)

    keep = (
        direction_0 & (work >= left) & (work >= right)
        | direction_45 & (work >= up_left) & (work >= down_right)
        | direction_90 & (work >= up) & (work >= down)
        | direction_135 & (work >= up_right) & (work >= down_left)
    )
    thinned = torch.where(keep & (work > 0.0), work, torch.zeros_like(work))
    return thinned.reshape(original_shape)


def match_edge_pixels(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    max_dist: float = 0.0075,
    valid: np.ndarray | None = None,
) -> Tuple[int, int, int]:
    """Return one-to-one matches, predicted pixels, and target pixels."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            "edge matching expects equal (H, W) arrays, got "
            f"{prediction.shape} and {target.shape}"
        )
    if not 0.0 <= max_dist <= 1.0:
        raise ValueError(f"max_dist must be in [0, 1], got {max_dist}")
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != prediction.shape:
            raise ValueError(
                f"valid mask shape {valid.shape} does not match {prediction.shape}"
            )
        prediction = prediction & valid
        target = target & valid

    predicted_points = np.argwhere(prediction)
    target_points = np.argwhere(target)
    predicted_count = int(predicted_points.shape[0])
    target_count = int(target_points.shape[0])
    if predicted_count == 0 or target_count == 0:
        return 0, predicted_count, target_count

    height, width = prediction.shape
    radius = float(max_dist) * float(np.hypot(height, width))
    if radius == 0.0:
        matches = int(np.count_nonzero(prediction & target))
        return matches, predicted_count, target_count

    neighbors = cKDTree(target_points).query_ball_point(
        predicted_points,
        r=radius,
    )
    row_indices = []
    column_indices = []
    for row_index, columns in enumerate(neighbors):
        row_indices.extend([row_index] * len(columns))
        column_indices.extend(columns)
    if not row_indices:
        return 0, predicted_count, target_count

    graph = csr_matrix(
        (
            np.ones(len(row_indices), dtype=np.uint8),
            (row_indices, column_indices),
        ),
        shape=(predicted_count, target_count),
    )
    assignment = maximum_bipartite_matching(graph, perm_type="column")
    matches = int(np.count_nonzero(assignment >= 0))
    return matches, predicted_count, target_count


def edge_f_measure(matches: int, predicted: int, target: int) -> float:
    """Compute F-measure from one-to-one correspondence counts."""
    if predicted == 0 and target == 0:
        return 1.0
    if matches == 0 or predicted == 0 or target == 0:
        return 0.0
    precision = matches / predicted
    recall = matches / target
    return 2.0 * precision * recall / (precision + recall)


class EdgeValidator(BaseValidator):
    """ODS/OIS validator for the canonical ``edge`` task."""

    task = "edge"

    def _setup_dataloader(self) -> DataLoader:
        if not self.config.data:
            raise ValueError("Edge validation requires data= (a dataset YAML).")
        data_config = resolve_edge_data(
            self.config.data,
            allow_scripts=getattr(self.config, "allow_download_scripts", False),
        )
        divisor = int(getattr(self.model, "edge_imgsz_divisor", 1) or 1)
        imgsz = int(self.config.imgsz)
        if imgsz % divisor:
            raise ValueError(
                f"Edge validation imgsz={imgsz} must be divisible by {divisor} "
                "for this model family."
            )
        dataset = EdgeDataset(
            data_config,
            split=self.config.split or "val",
            imgsz=imgsz,
            augment=False,
            resize_mode=getattr(self.model, "edge_resize_mode", "letterbox"),
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=edge_collate_fn,
        )

    def _init_metrics(self) -> None:
        self._thresholds = np.asarray(
            tuple(self.config.edge_thresholds), dtype=np.float64
        )
        count = len(self._thresholds)
        self._matches = np.zeros(count, dtype=np.int64)
        self._predicted = np.zeros(count, dtype=np.int64)
        self._targets = np.zeros(count, dtype=np.int64)
        self._ois_sum = 0.0
        self._image_count = 0

    def _preprocess_batch(self, batch: Any) -> tuple:
        images, targets, image_info, image_ids = batch
        return images, targets, image_info, image_ids

    def _postprocess_predictions(self, preds: Any, batch: Any) -> torch.Tensor:
        output = preds
        if isinstance(output, dict):
            output = output.get(
                "edges",
                output.get("edge", output.get("predictions")),
            )
        if isinstance(output, (list, tuple)):
            output = output[-1]
        if output is None:
            raise ValueError("Edge validation model output did not contain edges.")
        output = torch.as_tensor(output)

        expected_batch = int(batch[1]["edges"].shape[0])
        if output.ndim == 2:
            if expected_batch != 1:
                raise ValueError(
                    "Unbatched edge output is only valid for batch size 1."
                )
            output = output[None, None]
        elif output.ndim == 3:
            if int(output.shape[0]) == expected_batch:
                output = output[:, None]
            elif expected_batch == 1 and int(output.shape[0]) == 1:
                output = output[None]
            else:
                raise ValueError(
                    f"Cannot interpret edge output shape {tuple(output.shape)}."
                )
        elif output.ndim == 4 and output.shape[1] == 1:
            pass
        elif output.ndim == 4 and output.shape[-1] == 1:
            output = output.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "Edge validation expects [B,1,H,W], [B,H,W], or [H,W], got "
                f"{tuple(output.shape)}."
            )
        if int(output.shape[0]) != expected_batch:
            raise ValueError(
                f"Edge output batch {int(output.shape[0])} does not match "
                f"target batch {expected_batch}."
            )

        output = output.float()
        if bool(getattr(self.model, "edge_output_logits", False)):
            output = output.sigmoid()
        target_hw = tuple(batch[1]["edges"].shape[-2:])
        if tuple(output.shape[-2:]) != target_hw:
            output = F.interpolate(
                output,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        return thin_edge_probabilities(output.clamp(0.0, 1.0))

    def _update_metrics(
        self,
        preds: Any,
        targets: Any,
        img_info: Any,
        img_ids: Any = None,
    ) -> None:
        probabilities = preds.detach().cpu().numpy()[:, 0]
        target_edges = targets["edges"].detach().cpu().numpy()[:, 0] >= 0.5
        valid_masks = targets["valid"].detach().cpu().numpy()[:, 0].astype(bool)

        for probability, target, valid in zip(probabilities, target_edges, valid_masks):
            image_f_measures = []
            for index, threshold in enumerate(self._thresholds):
                matches, predicted_count, target_count = match_edge_pixels(
                    probability >= threshold,
                    target,
                    max_dist=float(self.config.edge_max_dist),
                    valid=valid,
                )
                self._matches[index] += matches
                self._predicted[index] += predicted_count
                self._targets[index] += target_count
                image_f_measures.append(
                    edge_f_measure(matches, predicted_count, target_count)
                )
            self._ois_sum += max(image_f_measures)
            self._image_count += 1

    def _compute_metrics(self) -> Dict[str, float]:
        if self._image_count == 0:
            raise ValueError("Edge validation processed no images.")
        if not bool(np.any(self._targets)):
            raise ValueError("Edge validation found no ground-truth edge pixels.")

        f_measures = np.asarray(
            [
                edge_f_measure(int(matches), int(predicted), int(target))
                for matches, predicted, target in zip(
                    self._matches,
                    self._predicted,
                    self._targets,
                )
            ],
            dtype=np.float64,
        )
        best_index = int(np.argmax(f_measures))
        ods = float(f_measures[best_index])
        return {
            "metrics/ODS": ods,
            "metrics/OIS": float(self._ois_sum / self._image_count),
            "metrics/best_threshold": float(self._thresholds[best_index]),
            "fitness": ods,
        }

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("Edge Validation Results")
        logger.info("=" * 50)
        logger.info("  ODS F-measure: %.4f", metrics["metrics/ODS"])
        logger.info("  OIS F-measure: %.4f", metrics["metrics/OIS"])
        logger.info("  ODS threshold: %.3f", metrics["metrics/best_threshold"])
        logger.info("=" * 50)


__all__ = [
    "EdgeValidator",
    "edge_f_measure",
    "match_edge_pixels",
    "thin_edge_probabilities",
]
