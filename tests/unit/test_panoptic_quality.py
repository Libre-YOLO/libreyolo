"""Tests for the Panoptic Quality metric (issue #555).

The rules under test come from the metric definition in *Panoptic Segmentation*
(Kirillov et al., CVPR 2019): unique matching at IoU > 0.5, void pixels removed
from predictions before the IoU, crowd regions never counted as false
negatives, and predictions mostly covering void/crowd removed before counting
false positives.
"""

import numpy as np
import pytest

from libreyolo.validation.panoptic_quality import PanopticQuality

pytestmark = pytest.mark.unit


def _gt(id_, cat, crowd=0):
    return {"id": id_, "category_id": cat, "iscrowd": crowd}


def _pred(id_, cat):
    return {"id": id_, "category_id": cat}


def test_perfect_prediction_scores_pq_one():
    gt = np.array([[1, 1], [2, 2]])
    pred = np.array([[1, 1], [2, 2]])
    pq = PanopticQuality(thing_class_ids={0})
    pq.update(gt, [_gt(1, 0), _gt(2, 5)], pred, [_pred(1, 0), _pred(2, 5)])
    m = pq.compute()
    assert m["metrics/PQ"] == pytest.approx(1.0)
    assert m["metrics/SQ"] == pytest.approx(1.0)
    assert m["metrics/RQ"] == pytest.approx(1.0)
    assert m["metrics/PQ_things"] == pytest.approx(1.0)  # category 0
    assert m["metrics/PQ_stuff"] == pytest.approx(1.0)   # category 5
    assert m["fitness"] == pytest.approx(1.0)


def test_segment_ids_need_not_agree_only_categories_and_overlap():
    """Predicted ids are independent of GT ids; matching is by category + IoU."""
    gt = np.array([[7, 7], [7, 7]])
    pred = np.array([[99, 99], [99, 99]])
    pq = PanopticQuality()
    pq.update(gt, [_gt(7, 3)], pred, [_pred(99, 3)])
    assert pq.compute()["metrics/PQ"] == pytest.approx(1.0)


def test_wrong_category_is_fp_plus_fn_not_a_match():
    gt = np.ones((2, 2), dtype=np.int64)
    pred = np.ones((2, 2), dtype=np.int64)
    pq = PanopticQuality()
    pq.update(gt, [_gt(1, 0)], pred, [_pred(1, 1)])
    m = pq.compute()
    assert pq.stats[0].fn == 1 and pq.stats[1].fp == 1
    assert pq.stats[0].tp == 0 and pq.stats[1].tp == 0
    assert m["metrics/PQ"] == pytest.approx(0.0)


def test_iou_at_or_below_half_does_not_match():
    """IoU exactly 0.5 must NOT match; the rule is strictly greater than 0.5."""
    gt = np.array([[1, 1, 0, 0]])          # gt area 2
    pred = np.array([[2, 0, 0, 0]])        # pred area 1, intersection 1
    # union = 2 + 1 - 1 = 2 -> IoU = 0.5 exactly
    pq = PanopticQuality()
    pq.update(gt, [_gt(1, 0)], pred, [_pred(2, 0)])
    assert pq.stats[0].tp == 0
    assert pq.stats[0].fn == 1
    assert pq.stats[0].fp == 1


def test_iou_above_half_matches_and_sq_is_the_iou():
    # No void anywhere: pixel 4 belongs to a different-category gt segment, so
    # the void rule cannot inflate the IoU.
    gt = np.array([[1, 1, 1, 9]])            # seg 1 (cat 0) area 3, seg 9 (cat 1)
    pred = np.array([[2, 2, 2, 2]])          # pred area 4, intersection 3
    # union = 3 + 4 - 3 = 4 -> IoU = 0.75 > 0.5
    pq = PanopticQuality()
    pq.update(gt, [_gt(1, 0), _gt(9, 1)], pred, [_pred(2, 0)])
    assert pq.stats[0].tp == 1
    assert pq.stats[0].iou == pytest.approx(0.75)
    # SQ for the matched category is its mean matched IoU.
    assert pq.stats[0].pq_sq_rq()[1] == pytest.approx(0.75)
    assert pq.stats[0].pq_sq_rq()[2] == pytest.approx(1.0)  # RQ: 1 tp, no fp/fn
    assert pq.stats[1].fn == 1  # the other gt segment was missed


def test_void_pixels_are_removed_from_the_prediction_before_iou():
    """Void GT under a prediction must not shrink its IoU (paper, void rule 1)."""
    gt = np.array([[1, 1, 0, 0]])           # ids 1,1,void,void ; gt area 2
    pred = np.array([[2, 2, 2, 2]])         # covers everything; pred area 4
    # Without the void rule: union = 2 + 4 - 2 = 4 -> IoU 0.5 -> no match.
    # With it: subtract the 2 void px inside pred -> union 2 -> IoU 1.0 -> match.
    pq = PanopticQuality()
    pq.update(gt, [_gt(1, 0)], pred, [_pred(2, 0)])
    assert pq.stats[0].tp == 1
    assert pq.compute()["metrics/SQ"] == pytest.approx(1.0)


def test_crowd_gt_is_never_a_false_negative():
    gt = np.array([[1, 1]])
    pred = np.zeros((1, 2), dtype=np.int64)  # predicted nothing
    pq = PanopticQuality()
    pq.update(gt, [_gt(1, 0, crowd=1)], pred, [])
    assert pq.stats == {}  # category never even instantiated
    assert pq.compute()["metrics/categories"] == 0


def test_unmatched_prediction_over_void_is_not_a_false_positive():
    """Paper, void rule 2: >50% void overlap removes the prediction."""
    gt = np.array([[0, 0, 0, 1]])            # 3 void px, 1 gt px
    pred = np.array([[5, 5, 5, 0]])          # pred area 3, all on void
    pq = PanopticQuality()
    pq.update(gt, [_gt(1, 0)], pred, [_pred(5, 1)])
    assert 1 not in pq.stats or pq.stats[1].fp == 0   # removed, not a FP
    assert pq.stats[0].fn == 1                        # gt segment still missed


def test_unmatched_prediction_over_crowd_of_same_class_is_not_a_false_positive():
    """Paper, group rule 2: same-class crowd overlap removes the prediction."""
    gt = np.array([[1, 1, 1, 1]])
    pred = np.array([[5, 5, 5, 5]])
    pq = PanopticQuality()
    pq.update(gt, [_gt(1, 0, crowd=1)], pred, [_pred(5, 0)])
    assert 0 not in pq.stats or pq.stats[0].fp == 0


def test_crowd_of_a_different_class_does_not_excuse_a_false_positive():
    gt = np.array([[1, 1, 1, 1]])
    pred = np.array([[5, 5, 5, 5]])
    pq = PanopticQuality()
    pq.update(gt, [_gt(1, 0, crowd=1)], pred, [_pred(5, 7)])  # class 7 != 0
    assert pq.stats[7].fp == 1


def test_things_and_stuff_are_averaged_separately():
    gt = np.array([[1, 1], [2, 2]])
    pred = np.array([[1, 1], [0, 0]])  # thing matched, stuff missed
    pq = PanopticQuality(thing_class_ids={0})
    pq.update(gt, [_gt(1, 0), _gt(2, 5)], pred, [_pred(1, 0)])
    m = pq.compute()
    assert m["metrics/PQ_things"] == pytest.approx(1.0)
    assert m["metrics/PQ_stuff"] == pytest.approx(0.0)
    assert m["metrics/PQ"] == pytest.approx(0.5)  # mean over both categories
    assert m["metrics/categories"] == 2


def test_only_seen_categories_are_averaged():
    """A category absent from both GT and prediction must not drag PQ down."""
    gt = np.ones((2, 2), dtype=np.int64)
    pred = np.ones((2, 2), dtype=np.int64)
    pq = PanopticQuality(thing_class_ids={0, 1, 2, 3})
    pq.update(gt, [_gt(1, 0)], pred, [_pred(1, 0)])
    m = pq.compute()
    assert m["metrics/categories"] == 1
    assert m["metrics/PQ"] == pytest.approx(1.0)


def test_accumulates_across_images():
    pq = PanopticQuality()
    perfect = np.ones((2, 2), dtype=np.int64)
    pq.update(perfect, [_gt(1, 0)], perfect, [_pred(1, 0)])
    pq.update(perfect, [_gt(1, 0)], np.zeros((2, 2), dtype=np.int64), [])
    assert pq.stats[0].tp == 1 and pq.stats[0].fn == 1
    # PQ = 1.0 / (1 + 0.5*0 + 0.5*1) = 0.6667
    assert pq.compute()["metrics/PQ"] == pytest.approx(2 / 3)


def test_shape_mismatch_is_rejected():
    pq = PanopticQuality()
    with pytest.raises(ValueError, match="ground-truth resolution"):
        pq.update(np.zeros((2, 2)), [], np.zeros((3, 3)), [])


def test_empty_prediction_and_empty_gt_gives_zero_not_nan():
    pq = PanopticQuality()
    pq.update(np.zeros((2, 2), dtype=np.int64), [], np.zeros((2, 2), dtype=np.int64), [])
    m = pq.compute()
    assert m["metrics/PQ"] == 0.0 and m["metrics/categories"] == 0
