"""OCR validator: detection hmean, end-to-end F1, and recognition 1-NED.

Metrics on the OCR dataset schema (``docs/dataset_schema.md``):

- **Detection hmean**: ICDAR-style precision/recall/F1 with one-to-one
  polygon matching at IoU > 0.5. Ground-truth regions marked ``"###"``
  (unreadable) are don't-care: they are not counted as targets and any
  prediction overlapping one is dropped from the precision denominator.
- **End-to-end F1** (headline): a match additionally requires the exact
  transcript after normalization. Normalization is Unicode NFKC plus removal
  of all whitespace; matching stays case-sensitive (case is part of what an
  OCR engine must read).
- **Recognition quality**: mean ``1 - NED`` (normalized Levenshtein edit
  distance) over detection-matched pairs.

Polygon IoU is computed with Sutherland-Hodgman clipping; DB predictions are
min-area rectangles (always convex) and the schema's ground-truth quads are
expected to be convex as well.

The validator drives the model's own ``predict`` (so the family's exact
two-stage pipeline is used) and is deterministic.
"""

from __future__ import annotations

import logging
import unicodedata
from numbers import Integral
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..data.ocr_dataset import DONT_CARE, resolve_ocr_samples

logger = logging.getLogger(__name__)

_IOU_THRESHOLD = 0.5


# =============================================================================
# Geometry
# =============================================================================


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar 2D cross product (np.cross on 2D vectors is deprecated)."""
    return float(a[0] * b[1] - a[1] * b[0])


def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: clip ``subject`` against convex ``clip``."""
    # Ensure the clip polygon is counter-clockwise for a consistent inside test.
    if _cross2(clip[1] - clip[0], clip[2] - clip[1]) < 0:
        clip = clip[::-1]

    output = list(subject)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a
        input_list, output = output, []
        if not input_list:
            break
        prev = input_list[-1]
        prev_inside = _cross2(edge, prev - a) >= 0
        for point in input_list:
            inside = _cross2(edge, point - a) >= 0
            if inside != prev_inside:
                # Intersection of clip edge (a, b) with segment (prev, point):
                # signed distances to the edge line interpolate linearly.
                d1 = _cross2(edge, prev - a)
                d2 = _cross2(edge, point - a)
                t = d1 / (d1 - d2)
                output.append(prev + t * (point - prev))
            if inside:
                output.append(point)
            prev, prev_inside = point, inside
    return np.asarray(output, dtype=np.float64) if output else np.zeros((0, 2))


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two convex polygons."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    area_a = _polygon_area(a)
    area_b = _polygon_area(b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter_poly = _clip_polygon(a, b)
    if inter_poly.shape[0] < 3:
        return 0.0
    inter = _polygon_area(inter_poly)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# =============================================================================
# Text metrics
# =============================================================================


def normalize_text(text: str) -> str:
    """NFKC-normalize and remove all whitespace; case-sensitive by design."""
    return "".join(unicodedata.normalize("NFKC", text).split())


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def one_minus_ned(pred: str, gt: str) -> float:
    denom = max(len(pred), len(gt))
    if denom == 0:
        return 1.0
    return 1.0 - edit_distance(pred, gt) / denom


# =============================================================================
# Per-image matching
# =============================================================================


def match_image(
    pred_polys: Sequence[np.ndarray],
    pred_texts: Sequence[str],
    gt_regions: Sequence[dict],
) -> Dict[str, float]:
    """Optimal one-to-one IoU matching for one image.

    Uses a linear-sum assignment over the IoU > 0.5 candidate pairs so the
    maximum number of valid pairs is matched (a greedy best-IoU-first pass
    can strand a prediction in crowded layouts when its only ground truth
    was taken by a prediction that had another option).

    Returns the raw counts the corpus metrics aggregate over.
    """
    care_gt = [g for g in gt_regions if g["text"] != DONT_CARE]
    dont_care_gt = [g for g in gt_regions if g["text"] == DONT_CARE]

    # Drop predictions that sit on don't-care regions (not penalized).
    kept_idx: List[int] = []
    for i, poly in enumerate(pred_polys):
        ignored = any(
            polygon_iou(poly, g["polygon"]) > _IOU_THRESHOLD for g in dont_care_gt
        )
        if not ignored:
            kept_idx.append(i)

    # Reward matrix over valid pairs: 1 per matchable pair (cardinality
    # dominates) plus the IoU as a tie-break so equal-cardinality solutions
    # prefer better-aligned pairs.
    reward = np.zeros((len(kept_idx), len(care_gt)), dtype=np.float64)
    for pi_pos, pi in enumerate(kept_idx):
        for gi, gt in enumerate(care_gt):
            iou = polygon_iou(pred_polys[pi], gt["polygon"])
            if iou > _IOU_THRESHOLD:
                reward[pi_pos, gi] = 1.0 + iou

    det_matches = 0
    e2e_matches = 0
    ned_scores: List[float] = []
    if reward.size and reward.any():
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(reward, maximize=True)
        for pi_pos, gi in zip(rows, cols):
            if reward[pi_pos, gi] <= 0:
                continue  # assignment filler below the IoU threshold
            det_matches += 1
            pred_text = pred_texts[kept_idx[pi_pos]]
            gt_text = care_gt[gi]["text"]
            ned_scores.append(
                one_minus_ned(normalize_text(pred_text), normalize_text(gt_text))
            )
            if normalize_text(pred_text) == normalize_text(gt_text):
                e2e_matches += 1

    return {
        "num_pred": len(kept_idx),
        "num_gt": len(care_gt),
        "det_matches": det_matches,
        "e2e_matches": e2e_matches,
        "ned_scores": ned_scores,
    }


def _prf(matches: int, num_pred: int, num_gt: int) -> Tuple[float, float, float]:
    precision = matches / num_pred if num_pred else 0.0
    recall = matches / num_gt if num_gt else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f1


class OCRValidator:
    """Detection hmean / end-to-end F1 / 1-NED validator for the ocr task."""

    task = "ocr"

    def __init__(self, model, config, **kwargs):
        self.model = model
        self.config = config

    def __call__(self, **kwargs) -> Dict[str, float]:
        if not self.config.data:
            raise ValueError(
                "OCR validation requires data= (a dataset root or YAML with "
                "images/<split> and labels/<split>.jsonl; see docs/dataset_schema.md)."
            )
        samples = resolve_ocr_samples(self.config.data, split=self.config.split or "val")

        totals = {"num_pred": 0, "num_gt": 0, "det_matches": 0, "e2e_matches": 0}
        ned_scores: List[float] = []
        conf = float(getattr(self.config, "conf_thres", 0.0) or 0.0)
        raw_imgsz = getattr(self.config, "imgsz", None)
        if (
            isinstance(raw_imgsz, bool)
            or not isinstance(raw_imgsz, Integral)
            or int(raw_imgsz) <= 0
        ):
            raise ValueError(
                "OCR validation imgsz must be one positive integer: it is the "
                "detector long-side limit, not a (height, width) canvas."
            )
        imgsz = int(raw_imgsz)

        for image_path, gt_regions in samples:
            result = self.model.predict(str(image_path), conf=conf, imgsz=imgsz)
            if isinstance(result, list):
                result = result[0]
            if result.ocr is None:
                raise ValueError(f"Model returned no OCR payload for {image_path}.")
            ocr_np = result.ocr.numpy()
            pred_polys = [np.asarray(ocr_np.data[i], dtype=np.float64) for i in range(len(ocr_np))]
            counts = match_image(pred_polys, ocr_np.texts, gt_regions)
            for key in totals:
                totals[key] += counts[key]
            ned_scores.extend(counts["ned_scores"])

        det_p, det_r, det_f = _prf(totals["det_matches"], totals["num_pred"], totals["num_gt"])
        e2e_p, e2e_r, e2e_f = _prf(totals["e2e_matches"], totals["num_pred"], totals["num_gt"])

        metrics = {
            "metrics/det_precision": det_p,
            "metrics/det_recall": det_r,
            "metrics/det_hmean": det_f,
            "metrics/e2e_precision": e2e_p,
            "metrics/e2e_recall": e2e_r,
            "metrics/e2e_f1": e2e_f,
            "metrics/rec_1-NED": float(np.mean(ned_scores)) if ned_scores else 0.0,
        }
        metrics["fitness"] = metrics["metrics/e2e_f1"]
        if getattr(self.config, "verbose", True):
            self._print_results(metrics, len(samples))
        return metrics

    @staticmethod
    def _print_results(metrics: Dict[str, float], n: int) -> None:
        logger.info("=" * 50)
        logger.info("OCR Validation Results (%d images)", n)
        logger.info("=" * 50)
        logger.info(
            "  Detection   P: %.4f  R: %.4f  hmean: %.4f",
            metrics["metrics/det_precision"],
            metrics["metrics/det_recall"],
            metrics["metrics/det_hmean"],
        )
        logger.info(
            "  End-to-end  P: %.4f  R: %.4f  F1:    %.4f",
            metrics["metrics/e2e_precision"],
            metrics["metrics/e2e_recall"],
            metrics["metrics/e2e_f1"],
        )
        logger.info("  Recognition 1-NED (matched pairs): %.4f", metrics["metrics/rec_1-NED"])
        logger.info("=" * 50)


__all__ = ["OCRValidator", "polygon_iou", "normalize_text", "one_minus_ned", "match_image"]
