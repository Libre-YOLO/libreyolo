"""Postprocessing contracts for Deformable DETR."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.postprocess.deformable_detr import postprocess
from libreyolo.utils.coco import COCO91_TO_COCO80

pytestmark = pytest.mark.unit


def test_postprocess_maps_coco_ids_scales_boxes_and_skips_nms():
    outputs = {
        "pred_logits": torch.full((1, 3, 91), -10.0),
        "pred_boxes": torch.tensor(
            [[[0.5, 0.5, 0.2, 0.4], [0.5, 0.5, 0.2, 0.4], [0.2, 0.2, 0.1, 0.1]]]
        ),
    }
    # Two identical boxes survive because set prediction is NMS-free. COCO id
    # 1 maps to contiguous class 0; sparse id 12 must never appear.
    outputs["pred_logits"][0, 0, 1] = 10.0
    outputs["pred_logits"][0, 1, 1] = 9.0
    outputs["pred_logits"][0, 2, 12] = 20.0

    result = postprocess(
        outputs,
        conf_thres=0.5,
        iou_thres=0.0,
        original_size=(100, 200),
        max_det=3,
        class_map=COCO91_TO_COCO80,
    )

    assert set(result) == {"num_detections", "boxes", "scores", "classes"}
    assert result["num_detections"] == 2
    assert result["classes"].tolist() == [0, 0]
    np.testing.assert_allclose(
        result["boxes"],
        [[40.0, 60.0, 60.0, 140.0], [40.0, 60.0, 60.0, 140.0]],
        rtol=0,
        atol=1e-5,
    )


def test_unmapped_columns_do_not_consume_topk_budget():
    unmapped = [index for index in range(91) if index not in COCO91_TO_COCO80]
    queries = 8
    logits = torch.full((1, queries, 91), -10.0)
    for query, column in enumerate(unmapped[:queries]):
        logits[0, query, column] = 20.0
        logits[0, query, query + 1] = 5.0

    result = postprocess(
        {
            "pred_logits": logits,
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]] * queries]),
        },
        conf_thres=0.01,
        original_size=(100, 100),
        max_det=5,
        class_map=COCO91_TO_COCO80,
    )

    assert result["num_detections"] == 5
    assert set(result["classes"].tolist()) <= set(COCO91_TO_COCO80.values())
    assert np.allclose(result["scores"], torch.sigmoid(torch.tensor(5.0)).item())


def test_wrapper_uses_coco_mapping_but_custom_heads_stay_contiguous():
    from libreyolo import LibreDeformableDETR

    coco = LibreDeformableDETR(None, size="r50ss", device="cpu")
    logits = torch.full((1, 1, 91), -10.0)
    logits[0, 0, 1] = 10.0
    coco_result = coco._postprocess(
        (logits, torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])),
        conf_thres=0.5,
        iou_thres=0.5,
        original_size=(100, 100),
    )
    assert coco_result["classes"].tolist() == [0]

    custom = LibreDeformableDETR(None, size="r50ss", nb_classes=3, device="cpu")
    custom_logits = torch.tensor([[[-10.0, -10.0, 10.0]]])
    custom_result = custom._postprocess(
        {"pred_logits": custom_logits, "pred_boxes": torch.ones(1, 1, 4) * 0.5},
        conf_thres=0.5,
        iou_thres=0.5,
        original_size=(10, 10),
    )
    assert custom_result["classes"].tolist() == [2]


def test_empty_postprocess_has_stable_shapes_and_dtypes():
    result = postprocess(
        {
            "pred_logits": torch.full((1, 2, 3), -100.0),
            "pred_boxes": torch.zeros(1, 2, 4),
        },
        conf_thres=0.5,
        original_size=(20, 10),
    )

    assert result["num_detections"] == 0
    assert result["boxes"].shape == (0, 4)
    assert result["scores"].shape == (0,)
    assert result["classes"].shape == (0,)
    assert result["classes"].dtype == np.int64
