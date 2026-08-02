"""Strict Faster R-CNN checkpoint recognition and sibling rejection."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def _faster_rcnn_state() -> dict[str, torch.Tensor]:
    return {
        "backbone.body.0.0.weight": torch.zeros(1),
        "rpn.head.conv.0.0.weight": torch.zeros(1),
        "roi_heads.box_predictor.cls_score.weight": torch.zeros(91, 1),
        "roi_heads.box_predictor.bbox_pred.weight": torch.zeros(364, 1),
    }


def _resnet18_state() -> dict[str, torch.Tensor]:
    state = {
        "conv1.weight": torch.zeros(1),
        "fc.weight": torch.zeros(1000, 1),
    }
    for stage in range(1, 5):
        for block in range(2):
            state[f"layer{stage}.{block}.conv1.weight"] = torch.zeros(1)
    return state


def _rtdetr_state() -> dict[str, torch.Tensor]:
    return {
        "backbone.res_layers.0.blocks.0.conv1.weight": torch.zeros(1),
        "encoder.input_proj.0.conv.weight": torch.zeros(1),
        "decoder.input_proj.0.conv.weight": torch.zeros(1),
        "decoder.dec_score_head.0.weight": torch.zeros(80, 1),
    }


def _l2cs_state() -> dict[str, torch.Tensor]:
    return {
        "fc_yaw_gaze.weight": torch.zeros(90, 1),
        "fc_pitch_gaze.weight": torch.zeros(90, 1),
    }


@pytest.mark.parametrize(
    ("class_name", "state_factory"),
    [
        ("LibreResNet", _resnet18_state),
        ("LibreRTDETR", _rtdetr_state),
        ("LibreRTDETRv2", _rtdetr_state),
        ("LibreL2CS", _l2cs_state),
    ],
)
def test_faster_rcnn_and_sibling_reject_each_other(class_name, state_factory):
    import libreyolo
    from libreyolo import LibreFasterRCNN

    sibling = getattr(libreyolo, class_name)
    sibling_state = state_factory()
    faster_state = _faster_rcnn_state()
    assert sibling.can_load(sibling_state) is True
    assert LibreFasterRCNN.can_load(sibling_state) is False
    assert LibreFasterRCNN.can_load(faster_state) is True
    assert sibling.can_load(faster_state) is False


@pytest.mark.parametrize(
    "extra_key",
    [
        "roi_heads.mask_predictor.mask_fcn_logits.weight",
        "roi_heads.keypoint_predictor.kps_score_lowres.weight",
    ],
)
def test_faster_rcnn_rejects_mask_and_keypoint_heads(extra_key):
    from libreyolo import LibreFasterRCNN

    state = _faster_rcnn_state()
    state[extra_key] = torch.zeros(1)
    assert LibreFasterRCNN.can_load(state) is False
