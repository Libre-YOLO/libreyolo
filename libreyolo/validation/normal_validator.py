"""Surface-normal validator for LibreYOLO.

Metrics are computed over valid ground-truth pixels in degrees: mean and
median angular error, plus the percentage within 11.25, 22.5, and 30 degrees.
Predictions are renormalized after output resizing. A non-finite or zero-length
prediction at a valid ground-truth pixel receives the maximum 180-degree error.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.normal_dataset import (
    NormalDataset,
    normal_collate_fn,
    resolve_normal_data,
)
from .base import BaseValidator

logger = logging.getLogger(__name__)

_NORM_EPS = 1e-12
_HISTOGRAM_SCALE = 1000
_HISTOGRAM_BINS = 180 * _HISTOGRAM_SCALE + 1

_METRIC_KEYS = (
    "metrics/mean_angular_error",
    "metrics/median_angular_error",
    "metrics/within_11_25",
    "metrics/within_22_5",
    "metrics/within_30",
)


def angular_errors(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return angular errors in degrees for valid target pixels.

    Inputs must have matching ``[..., 3]`` shapes. Target vectors that are
    zero-length or non-finite are excluded. Invalid predictions at otherwise
    valid target pixels are retained with a 180-degree error.
    """
    predictions = torch.as_tensor(predictions).double()
    targets = torch.as_tensor(targets).double()
    if predictions.shape != targets.shape or predictions.shape[-1] != 3:
        raise ValueError(
            "angular_errors expects matching [..., 3] tensors, got "
            f"{tuple(predictions.shape)} and {tuple(targets.shape)}"
        )

    target_finite = torch.isfinite(targets).all(dim=-1)
    target_norms = torch.linalg.vector_norm(
        torch.where(target_finite[..., None], targets, 0.0), dim=-1
    )
    valid_targets = target_finite & (target_norms > _NORM_EPS)
    if not bool(valid_targets.any()):
        return torch.empty(0, dtype=torch.float64, device=targets.device)

    selected_targets = targets[valid_targets]
    selected_predictions = predictions[valid_targets]
    selected_target_norms = target_norms[valid_targets]

    prediction_finite = torch.isfinite(selected_predictions).all(dim=-1)
    prediction_norms = torch.linalg.vector_norm(
        torch.where(
            prediction_finite[..., None],
            selected_predictions,
            0.0,
        ),
        dim=-1,
    )
    valid_predictions = prediction_finite & (prediction_norms > _NORM_EPS)

    errors = torch.full(
        (selected_targets.shape[0],),
        180.0,
        dtype=torch.float64,
        device=targets.device,
    )
    if bool(valid_predictions.any()):
        pred_unit = (
            selected_predictions[valid_predictions]
            / prediction_norms[valid_predictions, None]
        )
        target_unit = (
            selected_targets[valid_predictions]
            / selected_target_norms[valid_predictions, None]
        )
        cosine = (pred_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
        errors[valid_predictions] = torch.rad2deg(torch.acos(cosine))
    return errors


class NormalValidator(BaseValidator):
    """Angular-error validator for the ``normal`` task."""

    task = "normal"

    def _setup_dataloader(self) -> DataLoader:
        if not self.config.data:
            raise ValueError("Normal validation requires data= (a dataset YAML).")
        data_config = resolve_normal_data(
            self.config.data,
            allow_scripts=getattr(self.config, "allow_download_scripts", False),
        )
        split = self.config.split or "val"

        divisor = getattr(self.model, "normal_imgsz_divisor", None)
        if divisor and self.config.imgsz % int(divisor):
            raise ValueError(
                f"Normal validation imgsz={self.config.imgsz} must be "
                f"divisible by {int(divisor)} for this model family."
            )
        resize_mode = getattr(self.model, "normal_resize_mode", "letterbox")
        dataset = NormalDataset(
            data_config,
            split=split,
            imgsz=self.config.imgsz,
            augment=False,
            resize_mode=resize_mode,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=normal_collate_fn,
        )

    def _init_metrics(self) -> None:
        self._error_sum = 0.0
        self._valid_pixel_count = 0
        self._threshold_counts = {
            "metrics/within_11_25": 0,
            "metrics/within_22_5": 0,
            "metrics/within_30": 0,
        }
        self._error_histogram = torch.zeros(_HISTOGRAM_BINS, dtype=torch.int64)

    def _preprocess_batch(self, batch: Any) -> tuple:
        images, targets, image_info, image_ids = batch
        return images, targets, image_info, image_ids

    def _postprocess_predictions(self, preds: Any, batch: Any) -> torch.Tensor:
        """Decode model output into normalized ``[B, 3, H, W]`` vectors."""
        output = preds
        if isinstance(output, dict):
            output = output.get(
                "normal",
                output.get("normals", output.get("predictions")),
            )
        if isinstance(output, (list, tuple)):
            output = output[0]
        if output is None:
            raise ValueError("Normal validation model output did not contain normals.")
        output = torch.as_tensor(output)

        targets = batch[1]
        expected_batch = int(targets.shape[0])
        if output.ndim == 3:
            if expected_batch != 1:
                raise ValueError(
                    "Unbatched normal output is only valid for batch size 1."
                )
            if output.shape[0] == 3:
                output = output.unsqueeze(0)
            elif output.shape[-1] == 3:
                output = output.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(
                    "Normal validation expects CHW or HWC output, got shape "
                    f"{tuple(output.shape)}."
                )
        elif output.ndim == 4 and output.shape[1] == 3:
            pass
        elif output.ndim == 4 and output.shape[-1] == 3:
            output = output.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "Normal validation expects [B, 3, H, W] or [B, H, W, 3] "
                f"output, got shape {tuple(output.shape)}."
            )
        if int(output.shape[0]) != expected_batch:
            raise ValueError(
                f"Normal output batch {int(output.shape[0])} does not match "
                f"target batch {expected_batch}."
            )

        target_hw = tuple(targets.shape[-2:])
        output = output.float()
        if tuple(output.shape[-2:]) != target_hw:
            output = F.interpolate(
                output,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        finite = torch.isfinite(output).all(dim=1, keepdim=True)
        safe = torch.where(finite, output, 0.0)
        norms = torch.linalg.vector_norm(safe, dim=1, keepdim=True)
        valid = finite & (norms > _NORM_EPS)
        return torch.where(valid, safe / norms.clamp_min(_NORM_EPS), 0.0)

    def _update_metrics(
        self,
        preds: Any,
        targets: Any,
        img_info: Any,
        img_ids: Any = None,
    ) -> None:
        predictions = preds.detach().cpu().permute(0, 2, 3, 1)
        target_vectors = targets.detach().cpu().permute(0, 2, 3, 1)
        errors = angular_errors(predictions, target_vectors)
        if errors.numel() == 0:
            return

        self._error_sum += float(errors.sum())
        self._valid_pixel_count += int(errors.numel())
        self._threshold_counts["metrics/within_11_25"] += int((errors <= 11.25).sum())
        self._threshold_counts["metrics/within_22_5"] += int((errors <= 22.5).sum())
        self._threshold_counts["metrics/within_30"] += int((errors <= 30.0).sum())

        bins = (
            (errors * _HISTOGRAM_SCALE)
            .round()
            .clamp(0, _HISTOGRAM_BINS - 1)
            .to(torch.int64)
        )
        self._error_histogram += torch.bincount(
            bins,
            minlength=_HISTOGRAM_BINS,
        )

    def _histogram_value_at(self, index: int) -> float:
        cumulative = torch.cumsum(self._error_histogram, dim=0)
        bin_index = int(
            torch.searchsorted(
                cumulative,
                torch.tensor(index + 1, dtype=cumulative.dtype),
            )
        )
        return bin_index / _HISTOGRAM_SCALE

    def _compute_metrics(self) -> Dict[str, float]:
        if self._valid_pixel_count == 0:
            raise ValueError("Normal validation found no valid ground-truth pixels.")

        count = self._valid_pixel_count
        lower = self._histogram_value_at((count - 1) // 2)
        upper = self._histogram_value_at(count // 2)
        metrics = {
            "metrics/mean_angular_error": self._error_sum / count,
            "metrics/median_angular_error": (lower + upper) / 2.0,
            "metrics/within_11_25": (
                self._threshold_counts["metrics/within_11_25"] / count * 100.0
            ),
            "metrics/within_22_5": (
                self._threshold_counts["metrics/within_22_5"] / count * 100.0
            ),
            "metrics/within_30": (
                self._threshold_counts["metrics/within_30"] / count * 100.0
            ),
        }
        metrics["fitness"] = metrics["metrics/within_11_25"] / 100.0
        return metrics

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("Surface Normal Validation Results")
        logger.info("=" * 50)
        logger.info(
            "  Mean angular error:   %.3f deg",
            metrics.get("metrics/mean_angular_error", 0.0),
        )
        logger.info(
            "  Median angular error: %.3f deg",
            metrics.get("metrics/median_angular_error", 0.0),
        )
        logger.info(
            "  Within 11.25 deg:      %.2f%%",
            metrics.get("metrics/within_11_25", 0.0),
        )
        logger.info(
            "  Within 22.5 deg:       %.2f%%",
            metrics.get("metrics/within_22_5", 0.0),
        )
        logger.info(
            "  Within 30 deg:         %.2f%%",
            metrics.get("metrics/within_30", 0.0),
        )
        logger.info("=" * 50)


__all__ = ["NormalValidator", "angular_errors"]
