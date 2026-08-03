"""Mask R-CNN checkpoint recognition and sibling rejection."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def _faster_rcnn_state() -> dict[str, torch.Tensor]:
    return {
        "rpn.head.conv.0.0.weight": torch.zeros(1),
        "roi_heads.box_predictor.cls_score.weight": torch.zeros(91, 1),
        "roi_heads.box_predictor.bbox_pred.weight": torch.zeros(364, 1),
    }


def _mask_rcnn_state() -> dict[str, torch.Tensor]:
    state = _faster_rcnn_state()
    state["roi_heads.mask_predictor.mask_fcn_logits.weight"] = torch.zeros(
        91, 1, 1, 1
    )
    return state


def test_family_metadata_filename_and_download_contract():
    from libreyolo import LibreMaskRCNN

    state = _mask_rcnn_state()
    assert LibreMaskRCNN.FAMILY == "mask_rcnn"
    assert LibreMaskRCNN.FILENAME_PREFIX == "LibreMaskRCNN"
    assert LibreMaskRCNN.INPUT_SIZES == {"r50": 800}
    assert LibreMaskRCNN.SUPPORTED_TASKS == ("detect", "segment")
    assert LibreMaskRCNN.DEFAULT_TASK == "segment"
    assert LibreMaskRCNN.detect_size(state) == "r50"
    assert LibreMaskRCNN.detect_checkpoint_task(state) == "segment"
    assert LibreMaskRCNN.detect_nb_classes(state) == 80
    assert (
        LibreMaskRCNN.detect_size_from_filename("LibreMaskRCNNr50.pt")
        == "r50"
    )
    assert (
        LibreMaskRCNN.detect_size_from_filename(
            "maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
        )
        == "r50"
    )
    assert LibreMaskRCNN.get_download_url("LibreMaskRCNNr50.pt") == (
        "https://huggingface.co/LibreYOLO/LibreMaskRCNNr50/resolve/main/"
        "LibreMaskRCNNr50.pt"
    )


def test_mask_and_faster_rcnn_reject_each_other():
    from libreyolo import LibreFasterRCNN, LibreMaskRCNN

    faster_state = _faster_rcnn_state()
    mask_state = _mask_rcnn_state()
    assert LibreMaskRCNN.can_load(mask_state) is True
    assert LibreFasterRCNN.can_load(mask_state) is False
    assert LibreFasterRCNN.can_load(faster_state) is True
    assert LibreMaskRCNN.can_load(faster_state) is False


@pytest.mark.parametrize(
    "state",
    [
        {
            "backbone.backbone.register_token": torch.zeros(1),
            "decoder.decoder.segmentation_head.mask_embed.layers.0.weight": (
                torch.zeros(1)
            ),
        },
        {
            "class_embed.weight": torch.zeros(91, 256),
            "segmentation_head.layers.0.weight": torch.zeros(1),
            "backbone.0.encoder.encoder.embeddings.position_embeddings": (
                torch.zeros(1, 677, 384)
            ),
        },
    ],
    ids=["ec-segment", "rfdetr-segment"],
)
def test_mask_rcnn_rejects_other_segment_families(state):
    from libreyolo import LibreMaskRCNN

    assert LibreMaskRCNN.can_load(state) is False
