"""COCO evaluator for LibreYOLO."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Env override for the COCO eval backend: "1"/"true"/"yes" forces the
# faster-coco-eval backend on, "0"/"false"/"no" forces it off, unset defers
# to the faster_coco_eval config flag. Useful for benchmark harnesses that
# cannot touch per-run configs.
FASTER_COCO_EVAL_ENV_VAR = "LIBREYOLO_FASTER_COCO_EVAL"

_warned_faster_unavailable = False


def _faster_coco_eval_env_override() -> Optional[bool]:
    value = os.environ.get(FASTER_COCO_EVAL_ENV_VAR)
    if value is None:
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


def resolve_faster_coco_eval(requested: bool) -> bool:
    """Decide whether to use the faster-coco-eval backend.

    The LIBREYOLO_FASTER_COCO_EVAL env var, when set, overrides `requested`.
    Returns False (stock pycocotools) if the package is not importable.
    """
    override = _faster_coco_eval_env_override()
    enabled = requested if override is None else override
    if not enabled:
        return False
    try:
        import faster_coco_eval  # noqa: F401
    except ImportError:
        global _warned_faster_unavailable
        if not _warned_faster_unavailable:
            logger.warning(
                "faster_coco_eval requested but not installed; falling back to "
                "pycocotools. Install with: pip install faster-coco-eval"
            )
            _warned_faster_unavailable = True
        return False
    return True


class COCOEvaluator:
    """
    COCO evaluation wrapper.

    Computes standard COCO metrics: AP (mAP@[0.5:0.95]), AP50, AP75,
    AP/AR by object size, and AR at different maxDets.
    """

    def __init__(
        self,
        coco_gt,
        iou_type: str = "bbox",
        label_to_category_id: Optional[Mapping[int, int]] = None,
        max_det: int = 100,
        faster_coco_eval: bool = False,
    ):
        if max_det < 1:
            raise ValueError(f"max_det must be >= 1, got {max_det}")
        self.coco_gt = coco_gt
        self.iou_type = iou_type
        self.max_det = int(max_det)
        self.faster_coco_eval = faster_coco_eval
        self.label_to_category_id = (
            {int(k): int(v) for k, v in label_to_category_id.items()}
            if label_to_category_id is not None
            else None
        )
        self.results = []
        self._img_ids = set()
        self._last_coco_eval = None
        # Provenance: backend actually used by the last compute() call.
        self.last_backend: Optional[str] = None

    def update(self, predictions: Dict, image_id: int):
        """
        Add predictions for an image.

        Args:
            predictions: Dict with boxes (xyxy), scores, classes.
            image_id: Image ID matching COCO API.
        """
        boxes = predictions["boxes"]
        scores = predictions["scores"]
        classes = predictions["classes"]
        masks = predictions.get("masks")

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(classes, torch.Tensor):
            classes = classes.cpu().numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        boxes = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes
        scores = np.array(scores) if not isinstance(scores, np.ndarray) else scores
        classes = np.array(classes) if not isinstance(classes, np.ndarray) else classes
        masks = np.array(masks) if masks is not None and not isinstance(masks, np.ndarray) else masks

        if self.iou_type == "segm" and masks is None:
            self._img_ids.add(image_id)
            return

        for idx, (box, score, label) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            label = int(label)
            category_id = (
                self.label_to_category_id.get(label, label)
                if self.label_to_category_id is not None
                else label
            )

            result = {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [float(x1), float(y1), float(w), float(h)],  # COCO xywh
                "score": float(score),
            }
            if self.iou_type == "segm":
                mask = masks[idx]
                result["segmentation"] = self._encode_mask(mask)
                result["area"] = float((mask > 0).sum())
            self.results.append(result)

        self._img_ids.add(image_id)

    @staticmethod
    def _encode_mask(mask: np.ndarray) -> dict:
        """Encode a binary mask to JSON-safe COCO RLE."""
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            raise ImportError(
                "pycocotools not installed. Install with: pip install pycocotools"
            )

        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask for COCO RLE, got shape {mask.shape}")
        mask = (mask > 0).astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mask))
        counts = rle.get("counts")
        if isinstance(counts, bytes):
            rle["counts"] = counts.decode("ascii")
        rle["size"] = [int(mask.shape[0]), int(mask.shape[1])]
        return rle

    def compute(self, save_json: Optional[str] = None) -> Dict[str, float]:
        """
        Run COCO evaluation and return 12 standard metrics.

        Args:
            save_json: Optional path to save predictions in COCO JSON format.
                Written even when no predictions were accumulated.
        """
        if save_json:
            # Written first: an opted-in run must produce the file even when
            # there are no predictions or evaluation fails below, and loadRes
            # mutates the result dicts in place (adds id/area/segmentation).
            with open(save_json, "w") as f:
                json.dump(self.results, f, indent=2)
            logger.info("Saved predictions to %s", Path(save_json).resolve())

        if len(self.results) == 0:
            logger.warning("No predictions to evaluate")
            return self._empty_metrics()

        coco_eval = self._build_coco_eval()
        if self._img_ids:
            coco_eval.params.imgIds = sorted(self._img_ids)
        # Retain a real AR@100 compatibility slot while adding the requested
        # protocol cap. COCOeval supports an arbitrary maxDets axis, but its
        # stock summarize() hard-codes overall AP to maxDets=100. Metrics below
        # are therefore read directly from the accumulated arrays.
        coco_eval.params.maxDets = sorted({1, 10, 100, self.max_det})
        coco_eval.evaluate()
        coco_eval.accumulate()
        if self.max_det == 100:
            # Preserve the historical/default path literally. This makes the
            # default output subject to pycocotools' own summarize semantics.
            coco_eval.summarize()
        else:
            coco_eval.stats = self._standard_stats(coco_eval)
        self._last_coco_eval = coco_eval  # kept for per-class AP access

        # stats layout: [mAP, mAP50, mAP75, AP_s, AP_m, AP_l,
        #                AR1, AR10, AR@max_det, AR_s, AR_m, AR_l]
        #
        # NOTE: these are NOT a precision/recall pair at a fixed operating
        # point. ``precision`` here is the mean of the precision array at the
        # last maxDet over all IoU/recall/class bins == mAP@[.5:.95] (stats[0]),
        # and ``recall`` remains the historical AR@100 value. They are emitted
        # under the honest ``map_5095`` / ``ar_100`` keys below; the legacy
        # ``precision`` / ``recall`` keys are kept as aliases for backward
        # compatibility and must not be plotted as a distinct P/R.
        map_5095 = self._summarize_metric(
            coco_eval, ap=True, max_det=self.max_det, empty=0.0
        )
        ar_100 = self._summarize_metric(
            coco_eval, ap=False, max_det=100, empty=0.0
        )
        ar_max_det = float(coco_eval.stats[8])
        return {
            "max_det": float(self.max_det),
            "map_5095": map_5095,
            "ar_100": ar_100,
            "ar_max_det": ar_max_det,
            "precision": map_5095,  # alias (deprecated): == map_5095, not real P
            "recall": ar_100,  # alias (deprecated): == ar_100, not real R
            "mAP": float(coco_eval.stats[0]),
            "mAP50": float(coco_eval.stats[1]),
            "mAP75": float(coco_eval.stats[2]),
            "mAP_small": float(coco_eval.stats[3]),
            "mAP_medium": float(coco_eval.stats[4]),
            "mAP_large": float(coco_eval.stats[5]),
            "AR1": float(coco_eval.stats[6]),
            "AR10": float(coco_eval.stats[7]),
            "AR100": ar_100,
            "AR_max_det": ar_max_det,
            "AR_small": float(coco_eval.stats[9]),
            "AR_medium": float(coco_eval.stats[10]),
            "AR_large": float(coco_eval.stats[11]),
        }

    def _build_coco_eval(self):
        """Construct a COCOeval instance using the configured backend.

        With faster_coco_eval=True (or the LIBREYOLO_FASTER_COCO_EVAL env
        override) and the faster-coco-eval package installed, evaluation runs
        through its C++ backend, which is 10-50x faster on detection-dense
        datasets while producing metrics identical to pycocotools within
        float64 summation order (<= 1 ULP).
        """
        if resolve_faster_coco_eval(self.faster_coco_eval):
            import faster_coco_eval
            from faster_coco_eval import COCO as FasterCOCO
            from faster_coco_eval import COCOeval_faster

            gt_dataset = getattr(self.coco_gt, "dataset", None)
            if not gt_dataset or not gt_dataset.get("images"):
                # COCO-like GT objects (e.g. YOLOCocoAPI) that don't carry a
                # raw dataset dict: synthesize one from their index maps.
                gt_dataset = {
                    "images": list(self.coco_gt.imgs.values()),
                    "annotations": list(self.coco_gt.anns.values()),
                    "categories": list(self.coco_gt.cats.values()),
                }
            # use_deepcopy so backend-side mutations (e.g. segm polygon->RLE
            # conversion) never leak back into self.coco_gt.
            coco_gt = FasterCOCO(gt_dataset, use_deepcopy=True)
            coco_dt = coco_gt.loadRes(self.results)
            fce_version = getattr(
                getattr(faster_coco_eval, "version", None), "__version__", "?"
            )
            self.last_backend = f"faster-coco-eval {fce_version}"
            logger.info("COCO eval backend: %s", self.last_backend)
            return COCOeval_faster(coco_gt, coco_dt, self.iou_type)

        try:
            import pycocotools
            from pycocotools.coco import COCO  # noqa: F401
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            raise ImportError(
                "pycocotools not installed. Install with: pip install pycocotools"
            )

        self.last_backend = (
            f"pycocotools {getattr(pycocotools, '__version__', '?')}"
        )
        logger.info("COCO eval backend: %s", self.last_backend)
        coco_dt = self.coco_gt.loadRes(self.results)
        return COCOeval(self.coco_gt, coco_dt, self.iou_type)

    def _empty_metrics(self) -> Dict[str, float]:
        """Return all-zero metrics dict."""
        return {
            "max_det": float(self.max_det),
            "map_5095": 0.0,
            "ar_100": 0.0,
            "ar_max_det": 0.0,
            "precision": 0.0,  # alias (deprecated): == map_5095, not real P
            "recall": 0.0,  # alias (deprecated): == ar_100, not real R
            "mAP": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
            "AR1": 0.0,
            "AR10": 0.0,
            "AR100": 0.0,
            "AR_max_det": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0,
        }

    def reset(self):
        """Clear all accumulated results."""
        self.results = []
        self._img_ids = set()

    @staticmethod
    def _mean_valid(values: np.ndarray, *, empty: float = 0.0) -> float:
        """Mean over COCOeval arrays while ignoring absent -1 entries."""
        valid = values[values > -1]
        if valid.size == 0:
            return empty
        return float(valid.mean())

    def _summarize_metric(
        self,
        coco_eval,
        *,
        ap: bool,
        max_det: int,
        iou_thr: Optional[float] = None,
        area: str = "all",
        empty: float = -1.0,
    ) -> float:
        """Read one metric from COCOeval's accumulated precision/recall arrays."""
        params = coco_eval.params
        area_indices = [
            index for index, label in enumerate(params.areaRngLbl) if label == area
        ]
        max_det_indices = [
            index for index, value in enumerate(params.maxDets) if value == max_det
        ]
        if not area_indices or not max_det_indices:
            return -1.0

        if ap:
            values = coco_eval.eval["precision"]
            if iou_thr is not None:
                iou_indices = np.flatnonzero(np.isclose(params.iouThrs, iou_thr))
                values = values[iou_indices]
            values = values[:, :, :, area_indices, max_det_indices]
        else:
            values = coco_eval.eval["recall"]
            if iou_thr is not None:
                iou_indices = np.flatnonzero(np.isclose(params.iouThrs, iou_thr))
                values = values[iou_indices]
            values = values[:, :, area_indices, max_det_indices]
        return self._mean_valid(values, empty=empty)

    def _standard_stats(self, coco_eval) -> np.ndarray:
        """Build COCO's 12 detection metrics at the configured maximum."""
        max_det = self.max_det
        return np.asarray(
            [
                self._summarize_metric(coco_eval, ap=True, max_det=max_det),
                self._summarize_metric(
                    coco_eval, ap=True, max_det=max_det, iou_thr=0.5
                ),
                self._summarize_metric(
                    coco_eval, ap=True, max_det=max_det, iou_thr=0.75
                ),
                self._summarize_metric(
                    coco_eval, ap=True, max_det=max_det, area="small"
                ),
                self._summarize_metric(
                    coco_eval, ap=True, max_det=max_det, area="medium"
                ),
                self._summarize_metric(
                    coco_eval, ap=True, max_det=max_det, area="large"
                ),
                self._summarize_metric(coco_eval, ap=False, max_det=1),
                self._summarize_metric(coco_eval, ap=False, max_det=10),
                self._summarize_metric(coco_eval, ap=False, max_det=max_det),
                self._summarize_metric(
                    coco_eval, ap=False, max_det=max_det, area="small"
                ),
                self._summarize_metric(
                    coco_eval, ap=False, max_det=max_det, area="medium"
                ),
                self._summarize_metric(
                    coco_eval, ap=False, max_det=max_det, area="large"
                ),
            ],
            dtype=np.float64,
        )
