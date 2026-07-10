"""Panoptic Quality (PQ) metric.

Implemented from the metric definition in *Panoptic Segmentation*
(Kirillov, He, Girshick, Rother, Dollar; CVPR 2019, arXiv:1801.00868), which
specifies both the formula and the evaluation rules for void and group regions.

PQ, for one category::

    PQ = sum_{(p,g) in TP} IoU(p, g) / (|TP| + 0.5 * |FP| + 0.5 * |FN|)
       = SQ * RQ

where ``SQ = sum IoU / |TP|`` (the average IoU of matched segments) and
``RQ = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)`` (the F1 score).

Matching rule (paper, Theorem 1): a predicted and a ground-truth segment of the
**same category** match iff their IoU is strictly greater than 0.5. Because
panoptic segments do not overlap, such a match is provably unique, so no greedy
tie-breaking is needed.

Void handling (paper, "void labels"):

1. During matching, predicted pixels that are void in the ground truth are
   removed from the prediction and do not affect the IoU.
2. After matching, an unmatched predicted segment whose void fraction exceeds
   the matching threshold is removed and does not count as a false positive.

Group (crowd) handling (paper, "group labels"):

1. During matching, group regions are not used.
2. After matching, an unmatched predicted segment whose overlap with a group of
   the same class exceeds the matching threshold is removed and does not count
   as a false positive.

Ground-truth crowd segments therefore never become false negatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np

# Segments match iff IoU strictly exceeds this. Fixed by the metric definition:
# above 0.5 the match is unique (Theorem 1).
PQ_IOU_THRESHOLD = 0.5

# Void/unlabeled segment id, shared with the COCO-panoptic PNG encoding.
VOID_SEGMENT_ID = 0

# Pairs (gt_id, pred_id) are packed into one integer for a single bincount.
_OFFSET = 1 << 32


@dataclass
class CategoryStat:
    """Accumulators for one category, summed over images."""

    iou: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def pq_sq_rq(self) -> tuple[float, float, float]:
        denominator = self.tp + 0.5 * self.fp + 0.5 * self.fn
        if denominator == 0:
            return 0.0, 0.0, 0.0
        sq = self.iou / self.tp if self.tp else 0.0
        rq = self.tp / denominator
        return sq * rq, sq, rq

    @property
    def seen(self) -> bool:
        """Whether the category appeared in the ground truth or a prediction."""
        return (self.tp + self.fp + self.fn) > 0


@dataclass
class PanopticQuality:
    """Streaming Panoptic Quality accumulator.

    ``thing_class_ids`` splits the per-category results into PQ_things and
    PQ_stuff. It is a property of the label set, not of individual segments.
    """

    thing_class_ids: Set[int] = field(default_factory=set)
    stats: Dict[int, CategoryStat] = field(default_factory=dict)

    def _stat(self, category_id: int) -> CategoryStat:
        return self.stats.setdefault(category_id, CategoryStat())

    def update(
        self,
        gt_map: np.ndarray,
        gt_segments: Sequence[dict],
        pred_map: np.ndarray,
        pred_segments: Sequence[dict],
    ) -> None:
        """Accumulate one image.

        ``gt_map`` / ``pred_map`` are ``(H, W)`` segment-id maps; id
        ``VOID_SEGMENT_ID`` is void. Each entry of ``gt_segments`` needs
        ``id``, ``category_id`` and ``iscrowd``; ``pred_segments`` needs ``id``
        and ``category_id``.
        """
        if gt_map.shape != pred_map.shape:
            raise ValueError(
                f"gt_map {gt_map.shape} and pred_map {pred_map.shape} must match. "
                "Panoptic Quality is computed at the ground-truth resolution."
            )

        gt_by_id = {int(s["id"]): s for s in gt_segments}
        pred_by_id = {int(s["id"]): s for s in pred_segments}

        gt_flat = gt_map.reshape(-1).astype(np.int64)
        pred_flat = pred_map.reshape(-1).astype(np.int64)

        # Pairs are packed as gt * _OFFSET + pred for a single bincount, so an id
        # at or above _OFFSET would unpack as a different segment and silently
        # misattribute intersections. COCO ids fit in 24 bits; model output is
        # not bounds-checked anywhere else.
        self._check_id_range(gt_flat, "ground-truth")
        self._check_id_range(pred_flat, "predicted")

        gt_areas = self._areas(gt_flat)
        pred_areas = self._areas(pred_flat)
        intersections = self._pair_areas(gt_flat, pred_flat)

        matched_gt: Set[int] = set()
        matched_pred: Set[int] = set()

        # --- matching: same category, IoU > 0.5, crowd excluded ---------------
        for (gt_id, pred_id), inter in intersections.items():
            if gt_id == VOID_SEGMENT_ID or pred_id == VOID_SEGMENT_ID:
                continue
            gt_segment = gt_by_id.get(gt_id)
            pred_segment = pred_by_id.get(pred_id)
            if gt_segment is None or pred_segment is None:
                continue
            if int(gt_segment.get("iscrowd", 0)) == 1:
                continue  # rule: group regions are not used during matching
            if int(gt_segment["category_id"]) != int(pred_segment["category_id"]):
                continue

            # Void GT pixels inside the prediction are removed from it, so they
            # neither count toward the union nor penalize the IoU.
            void_inside_pred = intersections.get((VOID_SEGMENT_ID, pred_id), 0)
            union = (
                gt_areas.get(gt_id, 0)
                + pred_areas.get(pred_id, 0)
                - inter
                - void_inside_pred
            )
            if union <= 0:
                continue
            iou = inter / union
            if iou > PQ_IOU_THRESHOLD:
                stat = self._stat(int(gt_segment["category_id"]))
                stat.tp += 1
                stat.iou += iou
                matched_gt.add(gt_id)
                matched_pred.add(pred_id)

        # --- false negatives: unmatched GT, crowd never counts ----------------
        # A category may carry more than one crowd region in an image, so collect
        # them all: keeping only the last would under-count the ignored area and
        # turn a legitimately-excused prediction into a false positive.
        crowd_by_category: Dict[int, List[int]] = {}
        for gt_id, gt_segment in gt_by_id.items():
            if int(gt_segment.get("iscrowd", 0)) == 1:
                crowd_by_category.setdefault(
                    int(gt_segment["category_id"]), []
                ).append(gt_id)
                continue
            if gt_id in matched_gt:
                continue
            self._stat(int(gt_segment["category_id"])).fn += 1

        # --- false positives: unmatched predictions, minus void/crowd ---------
        for pred_id, pred_segment in pred_by_id.items():
            if pred_id in matched_pred:
                continue
            pred_area = pred_areas.get(pred_id, 0)
            if pred_area == 0:
                continue
            ignored = intersections.get((VOID_SEGMENT_ID, pred_id), 0)
            for crowd_id in crowd_by_category.get(int(pred_segment["category_id"]), ()):
                ignored += intersections.get((crowd_id, pred_id), 0)
            if ignored / pred_area > PQ_IOU_THRESHOLD:
                continue  # removed; does not count as a false positive
            self._stat(int(pred_segment["category_id"])).fp += 1

    @staticmethod
    def _check_id_range(flat: np.ndarray, which: str) -> None:
        if flat.size == 0:
            return
        if flat.min() < 0 or flat.max() >= _OFFSET:
            raise ValueError(
                f"{which} segment ids must lie in [0, {_OFFSET}); got "
                f"[{int(flat.min())}, {int(flat.max())}]. Ids are packed pairwise "
                "for the intersection histogram."
            )

    @staticmethod
    def _areas(flat: np.ndarray) -> Dict[int, int]:
        ids, counts = np.unique(flat, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, counts)}

    @staticmethod
    def _pair_areas(gt_flat: np.ndarray, pred_flat: np.ndarray) -> Dict[tuple, int]:
        packed = gt_flat * _OFFSET + pred_flat
        keys, counts = np.unique(packed, return_counts=True)
        return {
            (int(k // _OFFSET), int(k % _OFFSET)): int(c) for k, c in zip(keys, counts)
        }

    def compute(self) -> Dict[str, float]:
        """Average per-category PQ/SQ/RQ over categories that were seen."""
        overall = self._average(self.stats.keys())
        things = self._average(i for i in self.stats if i in self.thing_class_ids)
        stuff = self._average(i for i in self.stats if i not in self.thing_class_ids)
        return {
            "metrics/PQ": overall["pq"],
            "metrics/SQ": overall["sq"],
            "metrics/RQ": overall["rq"],
            "metrics/PQ_things": things["pq"],
            "metrics/PQ_stuff": stuff["pq"],
            "metrics/categories": overall["n"],
            "fitness": overall["pq"],
        }

    def _average(self, category_ids: Iterable[int]) -> Dict[str, float]:
        seen: List[CategoryStat] = [
            self.stats[i] for i in category_ids if self.stats[i].seen
        ]
        if not seen:
            return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "n": 0}
        triples = [stat.pq_sq_rq() for stat in seen]
        count = len(triples)
        return {
            "pq": sum(t[0] for t in triples) / count,
            "sq": sum(t[1] for t in triples) / count,
            "rq": sum(t[2] for t in triples) / count,
            "n": count,
        }


__all__ = ["CategoryStat", "PanopticQuality", "PQ_IOU_THRESHOLD", "VOID_SEGMENT_ID"]
