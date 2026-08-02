"""LibreDeiT registry, architecture, inference, and discriminator tests."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit

from libreyolo.models.deit.model import LibreDeiT  # noqa: E402
from libreyolo.models.deit.nn import DeiT  # noqa: E402
from libreyolo.postprocess.deit import postprocess  # noqa: E402


def _minimal_state(size: str, nc: int = 1000) -> dict[str, torch.Tensor]:
    dim = {"t": 192, "s": 384, "b": 768}[size]
    return {
        "patch_embed.proj.weight": torch.zeros(dim, 3, 16, 16),
        "cls_token": torch.zeros(1, 1, dim),
        "pos_embed": torch.zeros(1, 197, dim),
        "blocks.11.mlp.fc2.weight": torch.zeros(dim, 4 * dim),
        "head.weight": torch.zeros(nc, dim),
    }


def test_registered_classify_contract_and_download_url():
    from libreyolo.models.base import BaseModel

    assert any(cls is LibreDeiT for cls in BaseModel._registry)
    model = LibreDeiT(size="t", nb_classes=7, device="cpu")
    assert model.family == "deit"
    assert model.task == "classify"
    assert model.input_size == 224
    assert model.crop_pct == 0.9
    assert model.interpolation == "bicubic"
    assert model.TRAIN_CONFIG is None
    assert LibreDeiT.get_download_url("LibreDeiTt-cls.pt") == (
        "https://huggingface.co/LibreYOLO/LibreDeiTt-cls/resolve/main/"
        "LibreDeiTt-cls.pt"
    )


def test_filename_contract_requires_cls_suffix():
    for size in ("t", "s", "b"):
        canonical = f"LibreDeiT{size}-cls.pt"
        assert LibreDeiT.detect_size_from_filename(canonical) == size
        assert LibreDeiT.detect_task_from_filename(canonical) == "classify"
        assert LibreDeiT.detect_size_from_filename(f"LibreDeiT{size}.pt") is None


@pytest.mark.parametrize("size", ["t", "s", "b"])
def test_detect_size_classes_and_plain_variant_gate(size):
    state = _minimal_state(size, nc=37)
    assert LibreDeiT.can_load(state)
    assert LibreDeiT.detect_size(state) == size
    assert LibreDeiT.detect_nb_classes(state) == 37

    distilled = dict(state)
    dim = state["cls_token"].shape[-1]
    distilled["dist_token"] = torch.zeros(1, 1, dim)
    distilled["head_dist.weight"] = torch.zeros(37, dim)
    distilled["pos_embed"] = torch.zeros(1, 198, dim)
    assert LibreDeiT.can_load(distilled) is False

    high_resolution = dict(state)
    high_resolution["pos_embed"] = torch.zeros(1, 577, dim)
    assert LibreDeiT.can_load(high_resolution) is False


def test_neighboring_families_reject_deit_bidirectionally():
    timm = pytest.importorskip("timm")
    from libreyolo import (
        LibreCLIP,
        LibreConvNeXt,
        LibreEfficientNetV2,
        LibreMobileNetV4,
        LibreResNet,
    )
    from libreyolo.models.dinov2.model import LibreDINOv2
    from libreyolo.models.lwdetr.model import LibreLWDETR

    deit_state = DeiT(size="t", num_classes=1000).state_dict()
    for sibling in (
        LibreMobileNetV4,
        LibreConvNeXt,
        LibreEfficientNetV2,
        LibreResNet,
        LibreCLIP,
        LibreDINOv2,
        LibreLWDETR,
    ):
        assert sibling.can_load(deit_state) is False

    for tag in (
        "mobilenetv4_conv_small",
        "convnext_tiny",
        "tf_efficientnetv2_b0",
        "resnet18",
    ):
        assert LibreDeiT.can_load(timm.create_model(tag, pretrained=False).state_dict()) is False

    assert LibreDeiT.can_load(
        {
            "visual.conv1.weight": torch.zeros(192, 3, 16, 16),
            "text_projection": torch.zeros(512, 512),
            "logit_scale": torch.zeros(()),
        }
    ) is False
    assert LibreDeiT.can_load(
        {
            "backbone.encoder.encoder.embeddings.position_embeddings": torch.zeros(
                1, 257, 384
            ),
            "linear.weight": torch.zeros(1000, 384),
        }
    ) is False


@pytest.mark.parametrize("size", ["t", "s", "b"])
def test_forward_shape(size):
    model = DeiT(size=size, num_classes=5).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 5)


def test_reset_classifier_preserves_runtime_contract():
    model = DeiT(size="t", num_classes=1000)
    model.reset_classifier(7)
    model.eval()
    assert model.head.out_features == 7
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 7)


def test_wrong_resolution_is_rejected():
    model = DeiT(size="t", num_classes=5).eval()
    with pytest.raises(AssertionError, match="height"):
        model(torch.zeros(1, 3, 192, 192))


def test_postprocess_and_public_predict_populate_probs():
    logits = torch.randn(1, 5)
    output = postprocess(logits)
    assert set(output) == {"probs"}
    assert output["probs"].shape == (5,)
    assert torch.isclose(output["probs"].sum(), torch.tensor(1.0), atol=1e-6)
    assert int(output["probs"].argmax()) == int(logits.argmax())

    model = LibreDeiT(size="t", nb_classes=5, device="cpu")
    result = model(Image.new("RGB", (320, 240), color=(90, 120, 180)))
    assert result.probs is not None
    assert result.probs.data.shape == (5,)
    assert result.boxes is None


def test_training_is_explicitly_out_of_scope():
    model = LibreDeiT(size="t", nb_classes=5, device="cpu")
    with pytest.raises(NotImplementedError, match="inference-only museum"):
        model.train(data="unused")
