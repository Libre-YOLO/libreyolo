"""Matte (background-removal / dichotomous segmentation) validator.

Two metrics on paired image/matte folders, both standard for dichotomous image
segmentation and salient-object detection, both in ``[0, 1]`` and independent of
resolution:

- **MAE** (mean absolute error): ``mean(|pred - gt|)`` over all pixels, with the
  predicted and ground-truth alpha in ``[0, 1]``. Lower is better.
- **S-measure** (structure measure, Fan et al., ICCV 2017): a structural
  similarity combining an object-aware term ``S_o`` and a region-aware term
  ``S_r`` as ``S = 0.5 * S_o + 0.5 * S_r``. It rewards preserving the subject's
  structure (edges, holes), which pixel MAE alone misses. Higher is better.

This validator drives the model's own ``predict`` (so the family's exact
preprocessing is used) over paired data resolved by
``libreyolo.data.matte_dataset.resolve_matte_pairs``, and is deterministic.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from ..data.matte_dataset import resolve_matte_pairs
from .contracts import require_finite

logger = logging.getLogger(__name__)

_EPS = np.finfo(np.float64).eps
_METRIC_KEYS = ("metrics/MAE", "metrics/Smeasure")


def _load_matte_gt(path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def matte_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute error between two ``[0, 1]`` alpha maps of equal shape."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    _validate_matte_pair(pred, gt, "Matte MAE")
    return float(np.abs(pred - gt).mean())


def _validate_matte_pair(pred: np.ndarray, gt: np.ndarray, context: str) -> None:
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError(
            f"{context} expects [H, W] maps, got {pred.shape} and {gt.shape}."
        )
    if pred.shape != gt.shape:
        raise ValueError(
            f"{context} prediction and target shapes must match, got "
            f"{pred.shape} and {gt.shape}."
        )
    require_finite(pred, f"{context} prediction")
    require_finite(gt, f"{context} target")
    if pred.size and ((pred < 0.0).any() or (pred > 1.0).any()):
        raise ValueError(f"{context} prediction values must lie in [0, 1].")
    if gt.size and ((gt < 0.0).any() or (gt > 1.0).any()):
        raise ValueError(f"{context} target values must lie in [0, 1].")


def _object(pred_region: np.ndarray) -> float:
    if pred_region.size == 0:
        return 0.0
    x = float(pred_region.mean())
    sigma = float(pred_region.std())
    return 2.0 * x / (x * x + 1.0 + sigma + _EPS)


def _s_object(pred: np.ndarray, gt: np.ndarray) -> float:
    fg = pred[gt > 0.5]
    bg = pred[gt <= 0.5]
    u = float((gt > 0.5).mean())
    o_fg = _object(fg)
    o_bg = _object(1.0 - bg)
    return u * o_fg + (1.0 - u) * o_bg


def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    n = pred.size
    if n == 0:
        return 1.0
    x = float(gt.mean())
    y = float(pred.mean())
    sigma_x2 = float(((gt - x) ** 2).sum() / (n - 1 + _EPS))
    sigma_y2 = float(((pred - y) ** 2).sum() / (n - 1 + _EPS))
    sigma_xy = float(((gt - x) * (pred - y)).sum() / (n - 1 + _EPS))
    alpha = 4.0 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)
    if alpha != 0:
        return alpha / (beta + _EPS)
    if alpha == 0 and beta == 0:
        return 1.0
    return 0.0


def _s_region(pred: np.ndarray, gt: np.ndarray) -> float:
    h, w = gt.shape
    total = float(gt.sum())
    if total == 0:
        cx, cy = w // 2, h // 2
    else:
        ys, xs = np.mgrid[0:h, 0:w]
        cx = int(round(float((xs * gt).sum() / total)))
        cy = int(round(float((ys * gt).sum() / total)))
        cx = min(max(cx, 0), w - 1)
        cy = min(max(cy, 0), h - 1)

    # Four quadrants around the GT centroid, weighted by GT area fraction.
    quads = [
        (gt[: cy + 1, : cx + 1], pred[: cy + 1, : cx + 1]),
        (gt[: cy + 1, cx + 1:], pred[: cy + 1, cx + 1:]),
        (gt[cy + 1:, : cx + 1], pred[cy + 1:, : cx + 1]),
        (gt[cy + 1:, cx + 1:], pred[cy + 1:, cx + 1:]),
    ]
    area = float(h * w)
    score = 0.0
    for g_q, p_q in quads:
        if g_q.size == 0:
            continue
        w_q = float(g_q.sum() / (total + _EPS)) if total > 0 else g_q.size / area
        score += w_q * _ssim(p_q, g_q)
    return score


def s_measure(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
    """Structure measure (Fan et al., ICCV 2017) for two ``[0, 1]`` alpha maps."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    _validate_matte_pair(pred, gt, "S-measure")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"S-measure alpha must lie in [0, 1], got {alpha!r}.")
    y = float(gt.mean())
    if y == 0.0:  # all background
        return float(1.0 - pred.mean())
    if y == 1.0:  # all foreground
        return float(pred.mean())
    s = alpha * _s_object(pred, gt) + (1.0 - alpha) * _s_region(pred, gt)
    return float(max(s, 0.0))


class MatteValidator:
    """MAE / S-measure validator for the matte task (paired image/matte folders)."""

    task = "matte"

    def __init__(self, model, config, **kwargs):
        self.model = model
        self.config = config

    def __call__(self, **kwargs) -> Dict[str, float]:
        if not self.config.data:
            raise ValueError(
                "Matte validation requires data= (a paired image/matte directory "
                "or a dataset YAML; see docs/dataset_schema.md)."
            )
        pairs = resolve_matte_pairs(self.config.data, split=self.config.split or "val")

        maes: List[float] = []
        sms: List[float] = []
        for image_path, matte_path in pairs:
            result = self.model.predict(str(image_path), verbose=False)[0]
            if result.matte is None:
                raise ValueError(f"Model returned no matte for {image_path}.")
            pred = np.asarray(result.matte.array, dtype=np.float32)
            gt = _load_matte_gt(matte_path)
            if gt.shape != pred.shape:
                from PIL import Image

                # Resize in float32 ("F" mode) so the soft alpha values are not
                # quantized to 8 bits before the metric is computed.
                gt = np.asarray(
                    Image.fromarray(gt, mode="F").resize(
                        (pred.shape[1], pred.shape[0]), Image.BILINEAR
                    ),
                    dtype=np.float32,
                )
            maes.append(matte_mae(pred, gt))
            sms.append(s_measure(pred, gt))

        metrics = {
            "metrics/MAE": float(np.mean(maes)),
            "metrics/Smeasure": float(np.mean(sms)),
        }
        require_finite(list(metrics.values()), "Matte validation metrics")
        metrics["fitness"] = metrics["metrics/Smeasure"]
        if getattr(self.config, "verbose", True):
            self._print_results(metrics, len(pairs))
        return metrics

    @staticmethod
    def _print_results(metrics: Dict[str, float], n: int) -> None:
        logger.info("=" * 50)
        logger.info("Matte Validation Results (%d image/matte pairs)", n)
        logger.info("=" * 50)
        logger.info("  MAE:        %.4f", metrics.get("metrics/MAE", 0.0))
        logger.info("  S-measure:  %.4f", metrics.get("metrics/Smeasure", 0.0))
        logger.info("=" * 50)


__all__ = ["MatteValidator", "matte_mae", "s_measure"]
