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
    "size", ("r50ss", "r50ssdc5", "r50", "r50refine", "r50twostage")
)
def test_family_skeleton_builds(size):
    from libreyolo import LibreDeformableDETR

    model = LibreDeformableDETR(None, size=size, device="cpu")
    assert model.size == size
    assert model.input_size == 800
    assert model.nb_classes == 80
    assert model._arch_num_classes == 91
