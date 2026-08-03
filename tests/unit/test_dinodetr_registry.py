"""Registry and checkpoint-identification contracts for DINO-DETR."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.dinodetr]


def _signature(size: str = "r50", classes: int = 91) -> dict:
    levels = 4 if size == "r50" else 5
    state = {
        "label_enc.weight": torch.zeros(classes + 1, 256),
        "transformer.tgt_embed.weight": torch.zeros(900, 256),
        "transformer.level_embed": torch.zeros(levels, 256),
        "transformer.enc_out_class_embed.weight": torch.zeros(classes, 256),
        "transformer.enc_out_bbox_embed.layers.2.weight": torch.zeros(4, 256),
        "transformer.decoder.ref_point_head.layers.0.weight": torch.zeros(256, 512),
        "class_embed.0.weight": torch.zeros(classes, 256),
    }
    if size == "swinl":
        state["backbone.0.patch_embed.proj.weight"] = torch.zeros(192, 3, 4, 4)
    else:
        state["backbone.0.body.conv1.weight"] = torch.zeros(64, 3, 7, 7)
    return state


def test_family_identity_and_group_registration():
    from libreyolo import LibreDINODETR
    from libreyolo.models.base import BaseModel
    from libreyolo.models.registry import MODEL_GROUPS

    assert LibreDINODETR.FAMILY == "dinodetr"
    assert LibreDINODETR.FILENAME_PREFIX == "LibreDINODETR"
    assert LibreDINODETR.INPUT_SIZES == {"r50": 800, "r50s5": 800, "swinl": 800}
    assert LibreDINODETR.SUPPORTED_TASKS == ("detect",)
    assert LibreDINODETR.TRAIN_CONFIG is None
    assert LibreDINODETR in BaseModel._registry
    assert MODEL_GROUPS["dinodetr"] == "g3"


@pytest.mark.parametrize("size", ["r50", "r50s5", "swinl"])
def test_filename_cli_and_download_contract(size):
    from libreyolo import LibreDINODETR
    from libreyolo.cli.config import get_all_cli_names, resolve_model_name
    from libreyolo.ui.server import _resolve_download_url

    filename = f"LibreDINODETR{size}.pt"
    cli_name = f"dinodetr-{size}"
    assert LibreDINODETR.detect_size_from_filename(filename) == size
    assert cli_name in set(get_all_cli_names())
    assert Path(resolve_model_name(cli_name)).name == filename
    expected = (
        f"https://huggingface.co/LibreYOLO/LibreDINODETR{size}/resolve/main/{filename}"
    )
    assert LibreDINODETR.get_download_url(filename) == expected
    assert _resolve_download_url(cli_name) == expected


@pytest.mark.parametrize("size", ["r50", "r50s5", "swinl"])
def test_native_checkpoint_signature_detects_size_and_public_classes(size):
    from libreyolo import LibreDINODETR

    state = _signature(size)
    assert LibreDINODETR.can_load(state) is True
    assert LibreDINODETR.detect_size(state) == size
    assert LibreDINODETR.detect_nb_classes(state) == 80


def test_signature_does_not_claim_neighboring_dino_or_detr_families():
    from libreyolo import LibreDINODETR

    for state in (
        {"backbone.embeddings.cls_token": torch.zeros(1, 1, 768)},
        {"transformer.level_embed": torch.zeros(4, 256)},
        {
            "label_enc.weight": torch.zeros(92, 256),
            "transformer.tgt_embed.weight": torch.zeros(300, 256),
        },
    ):
        assert LibreDINODETR.can_load(state) is False


def test_training_is_explicitly_out_of_scope():
    from libreyolo import LibreDINODETR

    with pytest.raises(NotImplementedError, match="inference-only"):
        LibreDINODETR.train(None)
