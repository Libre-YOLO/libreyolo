"""NMS-free DINO-DETR postprocessing contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libreyolo.postprocess.dinodetr import postprocess
from libreyolo.utils.coco import COCO91_TO_COCO80

pytestmark = [pytest.mark.unit, pytest.mark.dinodetr]


def test_sparse_coco_columns_are_removed_before_topk_and_nms_is_skipped():
    logits = torch.full((1, 8, 91), -10.0)
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4]] * 8])
    unused = sorted(set(range(91)) - set(COCO91_TO_COCO80))
    for query, column in enumerate(unused[:8]):
        logits[0, query, column] = 20.0
        logits[0, query, query + 1] = 5.0

    result = postprocess(
        {"pred_logits": logits, "pred_boxes": boxes},
        conf_thres=0.01,
        iou_thres=0.0,
        original_size=(100, 200),
        max_det=5,
        class_map=COCO91_TO_COCO80,
    )

    assert result["num_detections"] == 5
    assert set(result["classes"].tolist()) <= set(range(80))
    np.testing.assert_allclose(
        result["scores"], torch.sigmoid(torch.tensor(5.0)).item()
    )
    np.testing.assert_allclose(result["boxes"][0], [40, 60, 60, 140], atol=1e-5)


def test_wrapper_maps_coco91_but_leaves_custom_heads_contiguous():
    from libreyolo import LibreDINODETR

    wrapper = object.__new__(LibreDINODETR)
    wrapper.model = SimpleNamespace(num_select=300)
    wrapper._arch_num_classes = 91
    wrapper.nb_classes = 80
    logits = torch.full((1, 1, 91), -10.0)
    logits[0, 0, 1] = 10.0
    coco = wrapper._postprocess(
        (logits, torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])),
        conf_thres=0.5,
        iou_thres=0.5,
        original_size=(100, 100),
    )
    assert coco["classes"].tolist() == [0]

    wrapper._arch_num_classes = wrapper.nb_classes = 3
    custom = wrapper._postprocess(
        {
            "pred_logits": torch.tensor([[[-10.0, -10.0, 10.0]]]),
            "pred_boxes": torch.ones(1, 1, 4) * 0.5,
        },
        conf_thres=0.5,
        iou_thres=0.5,
        original_size=(10, 10),
    )
    assert custom["classes"].tolist() == [2]


def test_empty_result_shapes_and_dtypes_are_stable():
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
