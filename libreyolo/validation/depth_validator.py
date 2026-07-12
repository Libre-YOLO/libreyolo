"""Monocular-depth validator for LibreYOLO.

Computes the standard relative-depth metrics (AbsRel, RMSE, delta thresholds)
over a dense depth validation split, reusing the :class:`BaseValidator`
template (setup -> iterate -> finalize).

Predictions are relative inverse depth; ground truth is plain depth. Each
prediction is aligned to its ground truth with a per-image least-squares
scale and shift in inverse-depth space before metrics are computed, the
standard protocol for affine-invariant depth models. Pixels with invalid
ground truth (``<= 0``) are excluded.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.depth_dataset import (
    DepthDataset,
    depth_collate_fn,
    resolve_depth_data,
)
from .base import BaseValidator
from .contracts import require_finite, require_matching_batch_sizes

logger = logging.getLogger(__name__)

_EPS = 1e-6

_METRIC_KEYS = (
    "metrics/abs_rel",
    "metrics/rmse",
    "metrics/delta1",
    "metrics/delta2",
    "metrics/delta3",
)


def align_inverse_depth(
    pred: torch.Tensor, gt_inverse: torch.Tensor
) -> torch.Tensor:
    """Least-squares scale-and-shift alignment of a prediction to a target.

    Solves ``min_{s,t} sum((s * pred + t - gt_inverse)^2)`` over the given
    flat tensors (valid pixels only) and returns the aligned prediction.
    Falls back to a median shift when the system is degenerate (e.g. a
    constant prediction) or when the fitted scale is non-positive. Depth
    predictions are relative inverse depth where larger means closer, so a
    negative scale would reward an inverted depth ordering.
    """
    pred = pred.double()
    gt_inverse = gt_inverse.double()
    n = pred.numel()
    sum_p = pred.sum()
    sum_pp = (pred * pred).sum()
    sum_g = gt_inverse.sum()
    sum_pg = (pred * gt_inverse).sum()
    det = sum_pp * n - sum_p * sum_p
    if float(det.abs()) < _EPS:
        return pred + (gt_inverse.median() - pred.median())
    scale = (sum_pg * n - sum_p * sum_g) / det
    shift = (sum_pp * sum_g - sum_p * sum_pg) / det
    if float(scale) <= _EPS:
        return pred + (gt_inverse.median() - pred.median())
    return scale * pred + shift


class DepthValidator(BaseValidator):
    """AbsRel / RMSE / delta-threshold validator for the depth task."""

    task = "depth"

    def _setup_dataloader(self) -> DataLoader:
        if not self.config.data:
            raise ValueError("Depth validation requires data= (a dataset YAML).")
        data_config = resolve_depth_data(
            self.config.data,
            allow_scripts=getattr(self.config, "allow_download_scripts", False),
        )
        split = self.config.split or "val"

        divisor = getattr(self.model, "depth_imgsz_divisor", None)
        if divisor and self.config.imgsz % int(divisor):
            raise ValueError(
                f"Depth validation imgsz={self.config.imgsz} must be "
                f"divisible by {int(divisor)} for this model family."
            )
        resize_mode = getattr(self.model, "depth_resize_mode", "letterbox")
        self._native_preprocess = resize_mode == "native"
        dataset = DepthDataset(
            data_config,
            split=split,
            imgsz=self.config.imgsz,
            augment=False,
            resize_mode=resize_mode,
        )
        batch_size = self.config.batch_size
        if self._native_preprocess:
            if batch_size != 1:
                logger.info(
                    "Native-aspect depth validation runs one image at a time "
                    "(batch_size=%d ignored).",
                    batch_size,
                )
            batch_size = 1
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=depth_collate_fn,
        )

    def _init_metrics(self) -> None:
        # Per-image accumulation (Eigen/MiDaS protocol): each metric is the
        # mean over an image's valid pixels, then averaged over images. RMSE
        # keeps its own running sum since it is not linear in the pixel terms.
        self._metric_sums = {
            "metrics/abs_rel": 0.0,
            "metrics/delta1": 0.0,
            "metrics/delta2": 0.0,
            "metrics/delta3": 0.0,
        }
        self._rmse_sum = 0.0
        self._image_count = 0

    def _preprocess_batch(self, batch: Any) -> tuple:
        images, targets, img_info, img_ids = batch
        if getattr(self, "_native_preprocess", False):
            if len(img_info) != 1:
                raise ValueError(
                    "Native-aspect depth validation requires exactly one image "
                    "per batch."
                )
            info = img_info[0]
            tensor, _, original_size, _ = self.model._preprocess(
                info["img_path"],
                color_format="rgb",
                input_size=self.config.imgsz,
            )
            expected_size = (int(info["orig_shape"][1]), int(info["orig_shape"][0]))
            if tuple(original_size) != expected_size:
                raise ValueError(
                    "Depth native preprocessing changed the source canvas: "
                    f"reported {tuple(original_size)}, expected {expected_size}."
                )
            self._native_original_size = tuple(original_size)
            images = tensor
        return images, targets, img_info, img_ids

    def _postprocess_predictions(self, preds: Any, batch: Any) -> Any:
        """Decode raw model output into ``[B, H, W]`` inverse-depth maps."""
        if getattr(self, "_native_preprocess", False):
            decoded = self.model._postprocess(
                preds,
                conf_thres=self.config.conf_thres,
                iou_thres=self.config.iou_thres,
                original_size=self._native_original_size,
                input_size=self.config.imgsz,
            )
            output = decoded.get("depth") if isinstance(decoded, dict) else decoded
            output = torch.as_tensor(output)
            if output.ndim == 2:
                output = output.unsqueeze(0)
            targets = torch.as_tensor(batch[1])
            if output.ndim != 3:
                raise ValueError(
                    "Native depth postprocessing must return [H, W] or "
                    f"[B, H, W], got {tuple(output.shape)}."
                )
            require_matching_batch_sizes(
                "Depth validation", predictions=output, targets=targets
            )
            if output.shape != targets.shape:
                raise ValueError(
                    "Native depth prediction and target canvases must match, got "
                    f"{tuple(output.shape)} and {tuple(targets.shape)}."
                )
            return output

        output = preds
        if isinstance(output, dict):
            output = output.get("depth", output.get("predictions"))
        if isinstance(output, (list, tuple)):
            output = output[0]
        output = torch.as_tensor(output)

        targets = torch.as_tensor(batch[1])
        if targets.ndim != 3:
            raise ValueError(
                "Depth validation expects [B, H, W] targets, got shape "
                f"{tuple(targets.shape)}."
            )
        target_hw = tuple(targets.shape[-2:])
        if output.ndim == 3:
            output = output.unsqueeze(1)
        if output.ndim != 4 or output.shape[1] != 1:
            raise ValueError(
                f"Depth validation expects [B, 1, H, W] output, got shape "
                f"{tuple(output.shape)}."
            )
        require_matching_batch_sizes(
            "Depth validation", predictions=output, targets=targets
        )
        if tuple(output.shape[-2:]) != target_hw:
            output = F.interpolate(
                output.float(), size=target_hw, mode="bilinear", align_corners=False
            )
        return output[:, 0]

    def _update_metrics(
        self, preds: Any, targets: Any, img_info: Any, img_ids: Any = None
    ) -> None:
        preds = torch.as_tensor(preds).detach().cpu().float()
        targets = torch.as_tensor(targets).detach().cpu().float()
        if preds.ndim != 3 or targets.ndim != 3:
            raise ValueError(
                "Depth validation metrics expect [B, H, W] predictions and "
                f"targets, got {tuple(preds.shape)} and {tuple(targets.shape)}."
            )
        require_matching_batch_sizes(
            "Depth validation", predictions=preds, targets=targets
        )
        if preds.shape != targets.shape:
            raise ValueError(
                "Depth validation prediction and target shapes must match, got "
                f"{tuple(preds.shape)} and {tuple(targets.shape)}."
            )
        valid_pixels = torch.isfinite(targets) & (targets > 0)
        require_finite(
            preds,
            "Depth validation predictions",
            where=valid_pixels,
        )
        for pred, gt_depth in zip(preds, targets):
            valid = torch.isfinite(gt_depth) & (gt_depth > 0)
            valid_count = int(valid.sum())
            if valid_count < 2:
                continue
            pred_valid = pred[valid]
            gt_inverse = 1.0 / gt_depth[valid].double()

            aligned = align_inverse_depth(pred_valid, gt_inverse)
            depth = 1.0 / aligned.clamp_min(_EPS)
            gt = gt_depth[valid].double()

            residual = depth - gt
            ratio = torch.maximum(depth / gt, gt / depth)
            # Per-image means, accumulated for a final average over images.
            self._metric_sums["metrics/abs_rel"] += float((residual.abs() / gt).mean())
            self._rmse_sum += float(torch.sqrt((residual * residual).mean()))
            self._metric_sums["metrics/delta1"] += float((ratio < 1.25).double().mean())
            self._metric_sums["metrics/delta2"] += float((ratio < 1.25**2).double().mean())
            self._metric_sums["metrics/delta3"] += float((ratio < 1.25**3).double().mean())
            self._image_count += 1

    def _compute_metrics(self) -> Dict[str, float]:
        if self._image_count == 0:
            raise ValueError(
                "Depth validation found no images with at least two valid pixels."
            )
        count = float(self._image_count)
        metrics = {
            "metrics/abs_rel": self._metric_sums["metrics/abs_rel"] / count,
            "metrics/rmse": self._rmse_sum / count,
            "metrics/delta1": self._metric_sums["metrics/delta1"] / count,
            "metrics/delta2": self._metric_sums["metrics/delta2"] / count,
            "metrics/delta3": self._metric_sums["metrics/delta3"] / count,
        }
        metrics["fitness"] = metrics["metrics/delta1"]
        return metrics

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("Depth Validation Results")
        logger.info("=" * 50)
        logger.info("  AbsRel: %.4f", metrics.get("metrics/abs_rel", 0.0))
        logger.info("  RMSE:   %.4f", metrics.get("metrics/rmse", 0.0))
        logger.info("  delta1: %.4f", metrics.get("metrics/delta1", 0.0))
        logger.info("  delta2: %.4f", metrics.get("metrics/delta2", 0.0))
        logger.info("  delta3: %.4f", metrics.get("metrics/delta3", 0.0))
        logger.info("=" * 50)


__all__ = ["DepthValidator", "align_inverse_depth"]
