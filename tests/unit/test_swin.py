"""LibreSwin registry, graph, conversion, and result-contract tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.swin]

from libreyolo.models.swin.classifier import SwinClassifier  # noqa: E402
from libreyolo.models.swin.config import SWIN_CONFIGS  # noqa: E402
from libreyolo.models.swin.convert import convert_upstream  # noqa: E402
from libreyolo.models.swin.model import LibreSwin  # noqa: E402
from libreyolo.models.swin.nn import SwinBackbone, SwinDims  # noqa: E402
from libreyolo.postprocess.swin import postprocess  # noqa: E402


def _signature_state_dict(size: str, nc: int = 7) -> dict[str, torch.Tensor]:
    spec = SWIN_CONFIGS[size]
    embed_dim = spec["embed_dim"]
    return {
        "patch_embed.proj.weight": torch.empty(embed_dim, 3, 4, 4),
        "layers.0.blocks.0.attn.relative_position_bias_table": torch.empty(
            169, spec["num_heads"][0]
        ),
        **{
            f"layers.2.blocks.{index}.norm1.weight": torch.empty(embed_dim * 4)
            for index in range(spec["depths"][2])
        },
        "norm.weight": torch.empty(embed_dim * 8),
        "head.fc.weight": torch.empty(nc, embed_dim * 8),
    }


def test_registered_and_shared_backbone_exports_remain_available():
    from libreyolo import LibreSwin as PublicLibreSwin
    from libreyolo.models.base import BaseModel
    from libreyolo.models.swin import SwinBackbone as PublicSwinBackbone

    assert PublicLibreSwin is LibreSwin
    assert PublicSwinBackbone is SwinBackbone
    assert SwinDims().tf_order is False
    assert any(model_class is LibreSwin for model_class in BaseModel._registry)


def test_classify_task_and_filename_detection():
    model = LibreSwin(size="t", nb_classes=7, device="cpu")
    assert model.family == "swin"
    assert model.task == "classify"
    assert model.input_size == 224
    assert model.crop_pct == 0.9
    for size in SWIN_CONFIGS:
        filename = f"LibreSwin{size}-cls.pt"
        assert LibreSwin.detect_size_from_filename(filename) == size
        assert LibreSwin.detect_task_from_filename(filename) == "classify"
        assert LibreSwin.detect_size_from_filename(f"LibreSwin{size}.pt") is None


@pytest.mark.parametrize("size", list(SWIN_CONFIGS))
def test_size_and_class_detection(size):
    state_dict = _signature_state_dict(size, nc=13)
    assert LibreSwin.can_load(state_dict) is True
    assert LibreSwin.detect_size(state_dict) == size
    assert LibreSwin.detect_nb_classes(state_dict) == 13


def test_swin_v2_window12_and_backbone_only_are_rejected():
    v2 = _signature_state_dict("t")
    v2["layers.0.blocks.0.attn.cpb_mlp.0.weight"] = torch.empty(512, 2)
    assert LibreSwin.can_load(v2) is False

    window12 = _signature_state_dict("b")
    window12["layers.0.blocks.0.attn.relative_position_bias_table"] = torch.empty(
        529, 4
    )
    assert LibreSwin.can_load(window12) is False

    backbone_only = _signature_state_dict("t")
    del backbone_only["head.fc.weight"]
    assert LibreSwin.can_load(backbone_only) is False


def test_official_layout_is_remapped_without_generated_buffers():
    native = _signature_state_dict("t")
    official = {
        (key.replace("head.fc.", "head.") if key.startswith("head.fc.") else key): value
        for key, value in native.items()
    }
    official["layers.0.downsample.norm.weight"] = torch.empty(384)
    official[
        "layers.0.blocks.0.attn.relative_position_index"
    ] = torch.empty(49, 49, dtype=torch.long)

    converted = convert_upstream(official)
    assert "head.fc.weight" in converted
    assert "head.weight" not in converted
    assert "layers.1.downsample.norm.weight" in converted
    assert not any("relative_position_index" in key for key in converted)
    assert LibreSwin.convert_upstream_state_dict(official) is not None


def test_forward_reset_and_postprocess_contract():
    network = SwinClassifier(size="t", num_classes=7).eval()
    with torch.no_grad():
        logits = network(torch.zeros(1, 3, 224, 224))
    assert logits.shape == (1, 7)

    network.reset_classifier(3)
    assert network.head.fc.out_features == 3
    result = postprocess(torch.tensor([[0.0, 1.0, 2.0]]))
    assert set(result) == {"probs"}
    assert result["probs"].shape == (3,)
    assert torch.isclose(result["probs"].sum(), torch.tensor(1.0))
    assert int(result["probs"].argmax()) == 2


def test_training_is_explicitly_out_of_scope():
    model = LibreSwin(size="t", nb_classes=7, device="cpu")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="unused")


def test_non_native_imgsz_is_rejected_before_the_attention_graph():
    model = LibreSwin(size="t", nb_classes=7, device="cpu")
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="prediction imgsz.*native imgsz=224"):
        model.predict(image, imgsz=256)
    with pytest.raises(ValueError, match="validation imgsz.*native imgsz=224"):
        model.val(data="unused", imgsz=256, workers=1)
    with pytest.raises(ValueError, match="export imgsz.*native imgsz=224"):
        model.export(format="onnx", imgsz=(224, 256))
