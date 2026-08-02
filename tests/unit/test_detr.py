"""Unit contract for the standalone DETR family skeleton."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def _official_signature() -> dict[str, torch.Tensor]:
    return {
        "query_embed.weight": torch.zeros(100, 256),
        "transformer.decoder.layers.0.multihead_attn.in_proj_weight": torch.zeros(
            768, 256
        ),
        "backbone.0.body.conv1.weight": torch.zeros(64, 3, 7, 7),
        "class_embed.weight": torch.zeros(92, 256),
    }


def test_detr_registration_and_filename_contract():
    from libreyolo import LibreDETR
    from libreyolo.models.base.model import BaseModel

    assert any(cls is LibreDETR for cls in BaseModel._registry)
    assert LibreDETR.FAMILY == "detr"
    assert LibreDETR.FILENAME_PREFIX == "LibreDETR"
    assert LibreDETR.SUPPORTED_TASKS == ("detect",)
    assert LibreDETR.DEFAULT_TASK == "detect"
    assert LibreDETR.TRAIN_CONFIG is None

    for size in ("r50", "r50dc5", "r101", "r101dc5"):
        assert LibreDETR.detect_size_from_filename(f"LibreDETR{size}.pt") == size

    assert LibreDETR.detect_size_from_filename("detr-r50-e632da11.pth") == "r50"
    assert LibreDETR.detect_size_from_filename("detr-r50-dc5-f0fb7ef5.pth") == "r50dc5"
    assert LibreDETR.detect_size_from_filename("detr-r101-2c7b67e5.pth") == "r101"
    assert (
        LibreDETR.detect_size_from_filename("detr-r101-dc5-a2e86def.pth") == "r101dc5"
    )
    assert LibreDETR.detect_size_from_filename("LibreRTDETRr50.pt") is None


def test_detr_official_signature_and_class_count():
    from libreyolo import LibreDETR

    state = _official_signature()
    assert LibreDETR.can_load(state) is True
    assert LibreDETR.detect_nb_classes(state) == 80
    # Dilation is not serialized, so raw state alone cannot honestly choose
    # between r50/r50dc5 or r101/r101dc5.
    assert LibreDETR.detect_size(state) is None


def test_detr_raw_checkpoint_requires_an_unambiguous_filename(tmp_path):
    from libreyolo.models.autoconvert import autoconvert_upstream_checkpoint

    state = _official_signature()
    ambiguous = tmp_path / "renamed.pth"
    torch.save({"model": state}, ambiguous)
    assert autoconvert_upstream_checkpoint(str(ambiguous)) is None

    official = tmp_path / "detr-r50-e632da11.pth"
    torch.save({"model": state}, official)
    converted_path = autoconvert_upstream_checkpoint(str(official))
    assert converted_path is not None
    converted = torch.load(converted_path, map_location="cpu", weights_only=True)
    assert converted["model_family"] == "detr"
    assert converted["size"] == "r50"
    assert converted["task"] == "detect"
    assert converted["nc"] == 80


@pytest.mark.parametrize("size", ("r50", "r50dc5", "r101", "r101dc5"))
def test_detr_native_model_builds_and_forwards(size):
    from libreyolo import LibreDETR

    model = LibreDETR(None, size=size, nb_classes=3, device="cpu")
    assert model.family == "detr"
    assert model.task == "detect"
    model.model.eval()
    with torch.no_grad():
        output = model.model(torch.zeros(1, 3, 64, 64))
    assert output["pred_logits"].shape == (1, 100, 4)
    assert output["pred_boxes"].shape == (1, 100, 4)


def test_detr_native_model_is_inference_only():
    from libreyolo import LibreDETR

    model = LibreDETR(None, size="r50", nb_classes=3, device="cpu")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="coco128.yaml")
