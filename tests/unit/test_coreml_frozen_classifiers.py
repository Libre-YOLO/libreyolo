"""Core ML contracts for frozen-class CLIP and SigLIP2 exports."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from libreyolo.export import coreml
from libreyolo.export.exporter import CoreMLExporter
from libreyolo.models.clip.model import LibreCLIP
from libreyolo.models.siglip2.model import LibreSigLIP2
from libreyolo.validation import CLIPClassifyValidator, SigLIP2ClassifyValidator

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("validator_cls", "family"),
    [
        (CLIPClassifyValidator, "CLIP"),
        (SigLIP2ClassifyValidator, "SigLIP2"),
    ],
)
def test_frozen_classifier_validation_requires_exact_humanized_class_order(
    validator_cls,
    family,
):
    validator = object.__new__(validator_cls)
    validator.model = SimpleNamespace(
        frozen_classes=True,
        names={0: "dog", 1: "cat"},
    )

    with pytest.raises(ValueError, match=rf"Frozen {family} class order"):
        validator._resolve_class_names(["cat", "dog"])


@pytest.mark.parametrize(
    ("validator_cls", "wnid", "expected"),
    [
        (CLIPClassifyValidator, "n01440764", "tench"),
        (SigLIP2ClassifyValidator, "n01440764", "tench"),
    ],
)
def test_frozen_classifier_validation_accepts_matching_humanized_wnid_order(
    validator_cls,
    wnid,
    expected,
):
    validator = object.__new__(validator_cls)
    validator.model = SimpleNamespace(
        frozen_classes=True,
        names={0: expected},
    )

    assert validator._resolve_class_names([wnid]) == [wnid]


def test_clip_coreml_photometric_and_spatial_contract_is_exact():
    x = torch.tensor(
        [
            [
                [[0.0, 0.25], [0.5, 1.0]],
                [[0.1, 0.3], [0.7, 0.9]],
                [[0.2, 0.4], [0.6, 0.8]],
            ]
        ],
        dtype=torch.float32,
    )
    wrapped = coreml._wrap_coreml_contract(nn.Identity(), "clip", "classify")
    mean = torch.tensor(coreml._CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(coreml._CLIP_STD).view(1, 3, 1, 1)
    torch.testing.assert_close(wrapped(x), (x - mean) / std)

    input_contract = coreml._input_contract("clip", "classify", "b32")
    assert input_contract == {
        "name": "image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "center_crop",
        "interpolation": "bicubic",
        "resize_backend": "pillow",
        "pad_value": 0,
        "crop_pct": 1.0,
    }
    assert coreml._validation_contract("clip", "classify") == {
        "color": "rgb",
        "range": "standardized",
        "mean": list(coreml._CLIP_MEAN),
        "std": list(coreml._CLIP_STD),
    }


def test_siglip2_coreml_photometric_and_spatial_contract_is_exact():
    x = torch.tensor(
        [
            [
                [[0.0, 0.25], [0.5, 1.0]],
                [[0.1, 0.3], [0.7, 0.9]],
                [[0.2, 0.4], [0.6, 0.8]],
            ]
        ],
        dtype=torch.float32,
    )
    wrapped = coreml._wrap_coreml_contract(nn.Identity(), "siglip2", "classify")
    torch.testing.assert_close(wrapped(x), (x - 0.5) / 0.5)

    input_contract = coreml._input_contract("siglip2", "classify", "b16")
    assert input_contract == {
        "name": "image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "stretch",
        "interpolation": "bilinear",
        "resize_backend": "pillow",
        "pad_value": 0,
        "crop_pct": 1.0,
    }
    assert coreml._validation_contract("siglip2", "classify") == {
        "color": "rgb",
        "range": "minus_1_1",
    }


def test_frozen_classifier_pairs_have_semantic_logits_contract():
    supported = coreml.supported_coreml_exports()
    assert ("clip", "classify") in supported
    assert ("siglip2", "classify") in supported
    assert coreml._output_contract("clip", "classify", nms=False) == [
        {"name": "class_logits", "role": "class_logits"}
    ]
    assert coreml._output_contract("siglip2", "classify", nms=False) == [
        {"name": "class_logits", "role": "class_logits"}
    ]


@pytest.mark.parametrize("family", ["clip", "siglip2"])
def test_real_frozen_vision_graph_is_traceable_and_input_sensitive(family):
    torch.manual_seed(20260729)
    if family == "clip":
        from libreyolo.models.clip.export import _FrozenCLIPClassifier
        from libreyolo.models.clip.nn import CLIPConfig, CLIPModel

        config = CLIPConfig(
            embed_dim=16,
            image_size=32,
            patch_size=16,
            vision_width=64,
            vision_layers=1,
            text_width=32,
            text_heads=2,
            text_layers=1,
        )
        tower = CLIPModel(config).visual.eval()
        frozen = _FrozenCLIPClassifier(tower, torch.randn(3, 16)).eval()
    else:
        from libreyolo.models.siglip2.export import _FrozenSigLIP2Classifier
        from libreyolo.models.siglip2.nn import SigLIP2Config, SiglipVisionModel

        config = SigLIP2Config(
            vision_width=32,
            vision_layers=1,
            vision_heads=2,
            vision_intermediate=64,
            image_size=32,
            patch_size=16,
            text_width=16,
            text_layers=1,
            text_heads=2,
            text_intermediate=32,
            vocab_size=32,
            max_position_embeddings=8,
            projection_size=32,
        )
        tower = SiglipVisionModel(config).eval()
        frozen = _FrozenSigLIP2Classifier(
            tower,
            torch.randn(3, 32),
            torch.tensor(-1.75),
        ).eval()

    wrapped = coreml._wrap_coreml_contract(frozen, family, "classify")
    probe_a = coreml._canonical_trace_probe(torch.zeros(1, 3, 32, 32))
    probe_b = 1.0 - probe_a
    traced = torch.jit.trace(
        wrapped,
        probe_a,
        check_trace=True,
        check_inputs=[(probe_b,)],
    )
    with torch.no_grad():
        eager_a = wrapped(probe_a)
        eager_b = wrapped(probe_b)
        traced_a = traced(probe_a)
        traced_b = traced(probe_b)
    torch.testing.assert_close(traced_a, eager_a)
    torch.testing.assert_close(traced_b, eager_b)
    assert eager_a.shape == (1, 3)
    assert not torch.allclose(eager_a, eager_b)


def test_frozen_classifier_helper_runs_shared_coreml_preflight(monkeypatch, tmp_path):
    calls = []

    def fake_validate(self, half, int8, data):
        calls.append(("validate", half, int8, data))
        return half, int8

    def fake_preflight(self, **kwargs):
        calls.append(("preflight", kwargs))

    def fake_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        calls.append(("metadata", precision, dynamic, onnx_path, imgsz))
        return {"precision": precision, "imgsz": imgsz}

    monkeypatch.setattr(CoreMLExporter, "_validate", fake_validate)
    monkeypatch.setattr(CoreMLExporter, "_preflight", fake_preflight)
    monkeypatch.setattr(CoreMLExporter, "_build_metadata", fake_metadata)
    owner = SimpleNamespace(
        input_size=32,
        multi_label=True,
        task="classify",
        size="b32",
        _get_model_name=lambda: "clip",
    )

    size, output, metadata, precision, compute_units = (
        coreml.prepare_frozen_classifier_coreml_export(
            owner,
            {
                "imgsz": (32, 32),
                "output": str(tmp_path / "frozen"),
                "device": "auto",
                "half": True,
                "compute_units": "CPU_ONLY",
            },
            default_output="unused",
        )
    )

    assert size == 32
    assert output == str(tmp_path / "frozen.mlpackage")
    assert precision == "fp16"
    assert compute_units == "cpu_only"
    assert metadata == {
        "precision": "fp16",
        "imgsz": (32, 32),
        "frozen_classes": True,
        "classification_activation": "sigmoid",
    }
    assert calls == [
        ("validate", True, False, None),
        (
            "preflight",
            {
                "half": True,
                "int8": False,
                "data": None,
                "nms": False,
                "compute_units": "cpu_only",
                "conf": 0.25,
                "iou": 0.45,
                "max_det": 300,
            },
        ),
        ("metadata", "fp16", False, None, (32, 32)),
    ]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"dynamic": True}, NotImplementedError, "fixed input shape"),
        ({"batch": 2}, ValueError, "batch=1"),
        ({"imgsz": (32, 24)}, NotImplementedError, "square input"),
        ({"imgsz": 64}, NotImplementedError, "native 32x32"),
        ({"device": "cuda:0"}, NotImplementedError, "traces on CPU"),
        ({"unknown": True}, TypeError, "Unsupported Core ML"),
        (
            {"output": "one", "output_path": "two"},
            ValueError,
            "only one Core ML destination",
        ),
    ],
)
def test_frozen_classifier_helper_rejects_invalid_request_before_preflight(
    monkeypatch, kwargs, error, match
):
    def preflight_must_not_run(self, **unused):
        raise AssertionError("preflight must not run")

    monkeypatch.setattr(CoreMLExporter, "_preflight", preflight_must_not_run)
    owner = SimpleNamespace(input_size=32, multi_label=False)
    with pytest.raises(error, match=match):
        coreml.prepare_frozen_classifier_coreml_export(
            owner,
            kwargs,
            default_output="unused",
        )


class _TinyVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = x[:, 0].mean(dim=(1, 2))
        second = x[:, 1:].mean(dim=(1, 2, 3))
        return torch.stack((first, second), dim=1) * self.gain


def test_clip_public_coreml_route_freezes_classes_and_restores_tower(
    monkeypatch, tmp_path
):
    visual = _TinyVisionTower().train()
    text_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    owner = SimpleNamespace(
        _text_embeds=text_embeds,
        model=SimpleNamespace(
            visual=visual,
            logit_scale=torch.tensor(0.25),
        ),
        size="tiny",
    )
    destination = str(tmp_path / "clip.mlpackage")
    captured = {}

    def fake_prepare(model, kwargs, *, default_output):
        assert model is owner
        assert kwargs == {"output_path": destination, "compute_units": "cpu_only"}
        assert default_output == "clip_coreml"
        assert visual.training is True
        return 8, destination, {"frozen_classes": True}, "fp32", "cpu_only"

    def fake_export(model, dummy, **kwargs):
        captured["weight"] = model.weight.detach().clone()
        captured["shape"] = tuple(dummy.shape)
        captured["kwargs"] = kwargs
        return destination

    monkeypatch.setattr(coreml, "prepare_frozen_classifier_coreml_export", fake_prepare)
    monkeypatch.setattr(coreml, "export_coreml", fake_export)
    embed_identity = owner._text_embeds

    result = LibreCLIP.export(
        owner,
        format="coreml",
        output_path=destination,
        compute_units="cpu_only",
    )

    assert result == destination
    torch.testing.assert_close(
        captured["weight"],
        torch.exp(torch.tensor(0.25)) * text_embeds,
    )
    assert captured["shape"] == (1, 3, 8, 8)
    assert captured["kwargs"] == {
        "output_path": destination,
        "precision": "fp32",
        "compute_units": "cpu_only",
        "metadata": {"frozen_classes": True},
        "model_family": "clip",
        "model_task": "classify",
        "model_size": "tiny",
    }
    assert owner._text_embeds is embed_identity
    assert visual.training is True


def test_siglip2_public_coreml_route_keeps_bias_and_restores_on_failure(
    monkeypatch, tmp_path
):
    vision = _TinyVisionTower().train()
    text_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bias = torch.tensor(-2.75)
    owner = SimpleNamespace(
        _text_embeds=text_embeds,
        model=SimpleNamespace(
            vision_model=vision,
            logit_scale=torch.tensor(-0.5),
            logit_bias=bias,
        ),
        size="tiny",
    )
    destination = str(tmp_path / "siglip2.mlpackage")
    captured = {}

    def fake_prepare(model, kwargs, *, default_output):
        assert model is owner
        assert kwargs == {"output_path": destination}
        assert default_output == "siglip2_coreml"
        assert vision.training is True
        return 8, destination, {"frozen_classes": True}, "fp32", "all"

    def fail_export(model, dummy, **kwargs):
        captured["weight"] = model.weight.detach().clone()
        captured["bias"] = model.bias.detach().clone()
        captured["shape"] = tuple(dummy.shape)
        raise RuntimeError("synthetic conversion failure")

    monkeypatch.setattr(coreml, "prepare_frozen_classifier_coreml_export", fake_prepare)
    monkeypatch.setattr(coreml, "export_coreml", fail_export)
    embed_identity = owner._text_embeds

    with pytest.raises(RuntimeError, match="synthetic conversion failure"):
        LibreSigLIP2.export(
            owner,
            format="coreml",
            output_path=destination,
        )

    torch.testing.assert_close(
        captured["weight"],
        torch.exp(torch.tensor(-0.5)) * text_embeds,
    )
    torch.testing.assert_close(captured["bias"], bias)
    assert captured["shape"] == (1, 3, 8, 8)
    assert owner._text_embeds is embed_identity
    assert vision.training is True
