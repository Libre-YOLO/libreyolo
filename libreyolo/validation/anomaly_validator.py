"""Image- and pixel-level validation for visual anomaly detection."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from PIL import Image

from ..data.anomaly_dataset import resolve_anomaly_test_samples

logger = logging.getLogger(__name__)


def binary_auroc(labels, scores) -> float:
    """Compute binary AUROC with average ranks for tied scores."""
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have identical shapes.")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires at least one normal and one anomalous sample.")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def best_f1(labels, scores) -> tuple[float, float]:
    """Return maximum binary F1 and the corresponding inclusive threshold."""
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(labels) or labels.sum() == 0:
        return 0.0, float("inf")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    fn = int(labels.sum()) - tp
    f1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1)
    best = int(np.argmax(f1))
    return float(f1[best]), float(sorted_scores[best])


def _load_mask(path, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != (shape[1], shape[0]):
        mask = mask.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


class AnomalyValidator:
    """AUROC and best-F1 validator for an MVTec-style test split."""

    task = "anomaly"

    def __init__(self, model, config, **kwargs):
        self.model = model
        self.config = config

    def __call__(self, **kwargs) -> Dict[str, float]:
        if not self.config.data:
            raise ValueError("Anomaly validation requires data= pointing to a category root or YAML.")
        samples = resolve_anomaly_test_samples(self.config.data)
        image_labels: list[int] = []
        image_scores: list[float] = []
        pixel_labels: list[np.ndarray] = []
        pixel_scores: list[np.ndarray] = []
        have_defect_mask = False

        for image_path, label, mask_path in samples:
            result = self.model.predict(str(image_path), verbose=False)[0]
            if result.anomaly_map is None or result.anomaly_score is None:
                raise ValueError(f"Model returned no anomaly result for {image_path}.")
            score_map = result.anomaly_map.array
            image_labels.append(label)
            image_scores.append(float(result.anomaly_score))
            if label == 0:
                pixel_labels.append(np.zeros(score_map.shape, dtype=np.uint8).reshape(-1))
                pixel_scores.append(score_map.reshape(-1))
            elif mask_path is not None:
                have_defect_mask = True
                pixel_labels.append(_load_mask(mask_path, score_map.shape).astype(np.uint8).reshape(-1))
                pixel_scores.append(score_map.reshape(-1))

        image_auc = binary_auroc(image_labels, image_scores)
        image_f1, image_threshold = best_f1(image_labels, image_scores)
        metrics: Dict[str, float] = {
            "metrics/image_AUROC": float(image_auc),
            "metrics/image_F1_max": float(image_f1),
            "metrics/image_F1_threshold": float(image_threshold),
            "fitness": float(image_auc),
        }
        if have_defect_mask and pixel_labels:
            labels = np.concatenate(pixel_labels)
            scores = np.concatenate(pixel_scores)
            metrics["metrics/pixel_AUROC"] = float(binary_auroc(labels, scores))
            pixel_f1, pixel_threshold = best_f1(labels, scores)
            metrics["metrics/pixel_F1_max"] = float(pixel_f1)
            metrics["metrics/pixel_F1_threshold"] = float(pixel_threshold)
        if getattr(self.config, "verbose", True):
            logger.info(
                "Anomaly validation: image AUROC %.4f, image F1-max %.4f",
                image_auc,
                image_f1,
            )
        return metrics


__all__ = ["AnomalyValidator", "best_f1", "binary_auroc"]
