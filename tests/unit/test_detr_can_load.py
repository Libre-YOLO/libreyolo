"""Bidirectional checkpoint-discriminator tests for vanilla DETR siblings."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def _detr_state() -> dict[str, torch.Tensor]:
    return {
        "query_embed.weight": torch.zeros(100, 256),
        "transformer.decoder.layers.0.multihead_attn.in_proj_weight": torch.zeros(
            768, 256
        ),
        "backbone.0.body.conv1.weight": torch.zeros(64, 3, 7, 7),
        "class_embed.weight": torch.zeros(92, 256),
    }


def _dfine_deim_state() -> dict[str, torch.Tensor]:
    return {"decoder.pre_bbox_head.layers.0.weight": torch.zeros(4, 4)}


def _rtdetr_state() -> dict[str, torch.Tensor]:
    return {
        "backbone.res_layers.0.blocks.0.conv1.weight": torch.zeros(4, 4, 3, 3),
        "encoder.input_proj.0.0.weight": torch.zeros(4, 4, 1, 1),
        "decoder.input_proj.0.conv.weight": torch.zeros(4, 4, 1, 1),
        "decoder.dec_score_head.0.weight": torch.zeros(80, 4),
    }


def _ec_state() -> dict[str, torch.Tensor]:
    return {"backbone.backbone.register_token": torch.zeros(1, 1, 192)}


def _rfdetr_state() -> dict[str, torch.Tensor]:
    return {
        "backbone.0.encoder.encoder.embeddings.cls_token": torch.zeros(1, 1, 256),
        "transformer.decoder.query_embed.weight": torch.zeros(300, 256),
        "class_embed.weight": torch.zeros(91, 256),
        "bbox_embed.layers.0.weight": torch.zeros(256, 256),
    }


@pytest.mark.parametrize(
    ("class_name", "state_factory"),
    (
        ("LibreDFINE", _dfine_deim_state),
        ("LibreDEIM", _dfine_deim_state),
        ("LibreRTDETR", _rtdetr_state),
        ("LibreEC", _ec_state),
    ),
)
def test_detr_and_core_siblings_reject_each_other(class_name, state_factory):
    import libreyolo

    from libreyolo import LibreDETR

    sibling = getattr(libreyolo, class_name)
    sibling_state = state_factory()
    assert sibling.can_load(sibling_state) is True
    assert LibreDETR.can_load(sibling_state) is False
    assert LibreDETR.can_load(_detr_state()) is True
    assert sibling.can_load(_detr_state()) is False


def test_detr_and_rfdetr_reject_each_other():
    pytest.importorskip("transformers")
    from libreyolo import LibreDETR
    from libreyolo.models.rfdetr.model import LibreRFDETR

    assert LibreRFDETR.can_load(_rfdetr_state()) is True
    assert LibreDETR.can_load(_rfdetr_state()) is False
    assert LibreDETR.can_load(_detr_state()) is True
    assert LibreRFDETR.can_load(_detr_state()) is False


def test_detr_does_not_trigger_optional_rfdetr_registration():
    from libreyolo.models import _needs_rfdetr_registration

    assert _needs_rfdetr_registration(_detr_state()) is False
    assert _needs_rfdetr_registration(_rfdetr_state()) is True
