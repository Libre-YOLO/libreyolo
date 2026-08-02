"""Registry and factory contracts for the Deformable DETR family."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def _signature(*, levels: int = 4, refined: bool = False, two_stage: bool = False):
    first = torch.zeros(91, 256)
    second = torch.ones(91, 256) if refined else first.clone()
    state = {
        "backbone.0.body.conv1.weight": torch.zeros(64, 3, 7, 7),
        "transformer.encoder.layers.0.self_attn.sampling_offsets.weight": torch.zeros(
            256, 256
        ),
        "transformer.level_embed": torch.zeros(levels, 256),
        "input_proj.0.0.weight": torch.zeros(256, 64, 1, 1),
        "class_embed.0.weight": first,
        "class_embed.1.weight": second,
        "bbox_embed.0.layers.0.weight": torch.zeros(256, 256),
    }
    if two_stage:
        state["transformer.enc_output.weight"] = torch.zeros(256, 256)
    else:
        state["query_embed.weight"] = torch.zeros(300, 512)
    return state


def test_family_is_public_and_registered():
    from libreyolo import LibreDeformableDETR
    from libreyolo.models.base.model import BaseModel

    assert LibreDeformableDETR.FAMILY == "deformable_detr"
    assert LibreDeformableDETR.FILENAME_PREFIX == "LibreDeformableDETR"
    assert LibreDeformableDETR.SUPPORTED_TASKS == ("detect",)
    assert LibreDeformableDETR.TRAIN_CONFIG is None
    assert LibreDeformableDETR in BaseModel._registry


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("LibreDeformableDETRr50ss.pt", "r50ss"),
        ("LibreDeformableDETRr50ssdc5.pt", "r50ssdc5"),
        ("LibreDeformableDETRr50.pt", "r50"),
        ("LibreDeformableDETRr50refine.pt", "r50refine"),
        ("LibreDeformableDETRr50twostage.pt", "r50twostage"),
        ("deformable-detr-single-scale/model.safetensors", "r50ss"),
        ("deformable-detr-single-scale-dc5/model.safetensors", "r50ssdc5"),
        ("deformable-detr-with-box-refine/model.safetensors", "r50refine"),
        (
            "deformable-detr-with-box-refine-two-stage/model.safetensors",
            "r50twostage",
        ),
        ("deformable-detr/model.safetensors", "r50"),
    ],
)
def test_filename_detection(filename, expected):
    from libreyolo import LibreDeformableDETR

    assert LibreDeformableDETR.detect_size_from_filename(filename) == expected


def test_state_dict_discriminator_and_size_detection():
    from libreyolo import LibreDeformableDETR

    assert LibreDeformableDETR.can_load(_signature()) is True
    assert LibreDeformableDETR.detect_size(_signature()) == "r50"
    assert LibreDeformableDETR.detect_size(_signature(refined=True)) == "r50refine"
    assert (
        LibreDeformableDETR.detect_size(_signature(refined=True, two_stage=True))
        == "r50twostage"
    )
    assert LibreDeformableDETR.detect_size(_signature(levels=1)) is None
    assert LibreDeformableDETR.detect_nb_classes(_signature()) == 80


def test_discriminator_requires_the_complete_signature():
    from libreyolo import LibreDeformableDETR

    state = _signature()
    for required in (
        "backbone.0.body.conv1.weight",
        "transformer.encoder.layers.0.self_attn.sampling_offsets.weight",
        "input_proj.0.0.weight",
        "class_embed.0.weight",
        "bbox_embed.0.layers.0.weight",
    ):
        incomplete = dict(state)
        incomplete.pop(required)
        assert LibreDeformableDETR.can_load(incomplete) is False


def test_original_family_does_not_trigger_lazy_rfdetr_registration():
    from libreyolo.models import _needs_rfdetr_registration

    assert _needs_rfdetr_registration(_signature()) is False


def test_rfdetr_explicitly_rejects_original_signature():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    assert LibreRFDETR.can_load(_signature()) is False


@pytest.mark.parametrize(
    "family",
    ("dfine", "deim", "deimv2", "rtdetr", "rfdetr", "ec"),
)
def test_confusable_detr_families_reject_each_other_bidirectionally(family):
    """Keep the original architecture isolated from all DETR descendants."""
    from libreyolo import LibreDeformableDETR
    from libreyolo.models.deim.model import LibreDEIM
    from libreyolo.models.deimv2.model import LibreDEIMv2
    from libreyolo.models.dfine.model import LibreDFINE
    from libreyolo.models.ec.model import LibreEC
    from libreyolo.models.rfdetr.model import LibreRFDETR
    from libreyolo.models.rtdetr.model import LibreRTDETR

    sibling_cases = {
        "dfine": (
            LibreDFINE,
            {"decoder.pre_bbox_head.0.layers.0.weight": torch.zeros(1)},
        ),
        "deim": (
            LibreDEIM,
            {"decoder.pre_bbox_head.0.layers.0.weight": torch.zeros(1)},
        ),
        "deimv2": (
            LibreDEIMv2,
            {"decoder.layers.0.swish_ffn.fc1.weight": torch.zeros(1)},
        ),
        "rtdetr": (
            LibreRTDETR,
            {
                "backbone.res_layers.0.weight": torch.zeros(1),
                "encoder.input_proj.0.0.weight": torch.zeros(1),
                "decoder.input_proj.0.weight": torch.zeros(1),
                "decoder.dec_score_head.0.weight": torch.zeros(1),
            },
        ),
        "rfdetr": (
            LibreRFDETR,
            {"transformer.decoder.layers.0.weight": torch.zeros(1)},
        ),
        "ec": (
            LibreEC,
            {"backbone.backbone.register_token": torch.zeros(1)},
        ),
    }
    sibling_class, sibling_state = sibling_cases[family]
    original_state = _signature()

    assert sibling_class.can_load(sibling_state) is True
    assert LibreDeformableDETR.can_load(sibling_state) is False
    assert LibreDeformableDETR.can_load(original_state) is True
    assert sibling_class.can_load(original_state) is False


@pytest.mark.parametrize(
    ("size", "side", "state_keys"),
    (
        ("r50ss", 64, 549),
        ("r50ssdc5", 64, 549),
        ("r50", 64, 561),
        ("r50refine", 64, 597),
        ("r50twostage", 128, 630),
    ),
)
def test_native_architecture_builds_and_runs(size, side, state_keys):
    from libreyolo import LibreDeformableDETR

    model = LibreDeformableDETR(None, size=size, device="cpu")
    assert model.size == size
    assert model.input_size == 800
    assert model.nb_classes == 80
    assert model._arch_num_classes == 91
    assert len(model.model.state_dict()) == state_keys

    model.model.eval()
    with torch.inference_mode():
        output = model.model(torch.zeros(1, 3, side, side))

    assert output["pred_logits"].shape == (1, 300, 91)
    assert output["pred_boxes"].shape == (1, 300, 4)
    assert len(output["aux_outputs"]) == 5


def test_variant_structure_matches_upstream_contracts():
    from libreyolo.models.deformable_detr.nn import LibreDeformableDETRModel

    single_scale = LibreDeformableDETRModel("r50ss")
    assert single_scale.num_feature_levels == 1
    assert single_scale.backbone.strides == [32]
    del single_scale

    dc5 = LibreDeformableDETRModel("r50ssdc5")
    assert dc5.backbone.strides == [16]
    del dc5

    base = LibreDeformableDETRModel("r50")
    assert base.num_feature_levels == 4
    assert base.class_embed[0] is base.class_embed[1]
    del base

    refined = LibreDeformableDETRModel("r50refine")
    assert refined.class_embed[0] is not refined.class_embed[1]
    assert hasattr(refined, "query_embed")
    del refined

    two_stage = LibreDeformableDETRModel("r50twostage")
    assert not hasattr(two_stage, "query_embed")
    assert hasattr(two_stage.transformer, "enc_output")
    assert len(two_stage.class_embed) == 7


def test_pure_pytorch_deformable_attention_core_has_gradients():
    from libreyolo.models.deformable_detr.ms_deform_attn import (
        ms_deform_attn_core_pytorch,
    )

    value = torch.arange(8, dtype=torch.float32).view(1, 4, 1, 2)
    value.requires_grad_(True)
    output = ms_deform_attn_core_pytorch(
        value,
        torch.tensor([[2, 2]], dtype=torch.long),
        torch.tensor([[[[[[0.5, 0.5]]]]]], dtype=torch.float32),
        torch.ones(1, 1, 1, 1, 1),
    )

    assert output.shape == (1, 1, 2)
    torch.testing.assert_close(output, value.detach().mean(dim=1))
    output.sum().backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
