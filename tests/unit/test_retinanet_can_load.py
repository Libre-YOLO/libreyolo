"""Strict RetinaNet checkpoint recognition and sibling rejection."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.retinanet]


def _retinanet_state(*, v2: bool = False) -> dict[str, torch.Tensor]:
    state = {
        "backbone.body.layer2.0.conv1.weight": torch.zeros(1),
        "backbone.fpn.extra_blocks.p6.weight": torch.zeros(
            256, 2048 if v2 else 256, 3, 3
        ),
        "head.classification_head.cls_logits.weight": torch.zeros(9 * 91, 256, 3, 3),
        "head.regression_head.bbox_reg.weight": torch.zeros(9 * 4, 256, 3, 3),
    }
    if v2:
        state["head.classification_head.conv.0.1.weight"] = torch.zeros(256)
    return state


def _rtdetr_state() -> dict[str, torch.Tensor]:
    return {
        "backbone.res_layers.0.blocks.0.conv1.weight": torch.zeros(1),
        "encoder.input_proj.0.conv.weight": torch.zeros(1),
        "decoder.input_proj.0.conv.weight": torch.zeros(1),
        "decoder.dec_score_head.0.weight": torch.zeros(80, 1),
    }


def _resnet_state() -> dict[str, torch.Tensor]:
    return {
        "conv1.weight": torch.zeros(64, 3, 7, 7),
        **{
            f"layer{stage}.{block}.conv1.weight": torch.zeros(1)
            for stage, count in enumerate((3, 4, 6, 3), start=1)
            for block in range(count)
        },
        "layer1.0.conv3.weight": torch.zeros(1),
        "fc.weight": torch.zeros(1000, 1),
    }


@pytest.mark.parametrize("v2", [False, True])
def test_retinanet_detects_variant_and_class_count(v2):
    from libreyolo import LibreRetinaNet

    state = _retinanet_state(v2=v2)
    assert LibreRetinaNet.can_load(state)
    assert LibreRetinaNet.detect_size(state) == ("r50v2" if v2 else "r50")
    assert LibreRetinaNet.detect_nb_classes(state) == 80


@pytest.mark.parametrize(
    ("class_name", "state_factory"),
    [("LibreRTDETR", _rtdetr_state), ("LibreResNet", _resnet_state)],
)
def test_retinanet_and_sibling_reject_each_other(class_name, state_factory):
    import libreyolo
    from libreyolo import LibreRetinaNet

    sibling = getattr(libreyolo, class_name)
    sibling_state = state_factory()
    retina_state = _retinanet_state()
    assert sibling.can_load(sibling_state)
    assert not LibreRetinaNet.can_load(sibling_state)
    assert LibreRetinaNet.can_load(retina_state)
    assert not sibling.can_load(retina_state)


def test_retinanet_rejects_fcos_centerness_head():
    from libreyolo import LibreRetinaNet

    state = _retinanet_state()
    state["head.classification_head.bbox_ctrness.weight"] = torch.zeros(1)
    assert not LibreRetinaNet.can_load(state)


def test_default_runtime_autoconvert_claim_is_strict():
    from libreyolo import LibreRetinaNet

    retina_state = _retinanet_state()
    converted = LibreRetinaNet.convert_upstream_state_dict(retina_state)
    assert converted is not None
    assert set(converted) == set(retina_state)
    assert all(torch.equal(converted[key], value) for key, value in retina_state.items())
    assert LibreRetinaNet.convert_upstream_state_dict(_rtdetr_state()) is None
    assert LibreRetinaNet.convert_upstream_state_dict(_resnet_state()) is None
