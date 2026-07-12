"""Offline export-contract tests for LibreSigLIP2."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from libreyolo.models.siglip2.model import LibreSigLIP2, siglip2_logits

pytestmark = [pytest.mark.unit, pytest.mark.siglip2]


class _TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images).mean(dim=(2, 3))


class _TinySigLIP2Core(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = _TinyVision()
        self.logit_scale = nn.Parameter(torch.tensor(0.25))
        self.logit_bias = nn.Parameter(torch.tensor(-0.1))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.vision_model(images)


@pytest.fixture
def tiny_siglip2_export_model() -> LibreSigLIP2:
    torch.manual_seed(7)
    model = object.__new__(LibreSigLIP2)
    model.model = _TinySigLIP2Core().eval()
    model.device = torch.device("cpu")
    model.size = "tiny"
    model.input_size = 8
    model.nb_classes = 2
    model.names = {0: "cat", 1: "dog"}
    model.task = "classify"
    model._text_embeds = F.normalize(torch.randn(2, 4), dim=-1)
    model._multi_label = False
    return model


def test_export_accepts_standard_cli_arguments(
    tiny_siglip2_export_model, monkeypatch, tmp_path
):
    captured = {}

    def fake_export(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(
        "libreyolo.models.siglip2.export.export_frozen_onnx", fake_export
    )
    output = tmp_path / "siglip2.onnx"

    result = tiny_siglip2_export_model.export(
        output_path=output,
        opset=None,
        simplify=False,
        dynamic=False,
        half=False,
        int8=False,
        imgsz=(8, 8),
        batch=2,
        device="cpu",
        verbose=True,
    )

    assert result == str(output)
    assert captured == {
        "model": tiny_siglip2_export_model,
        "imgsz": 8,
        "opset": None,
        "output": str(output),
        "batch": 2,
        "dynamic": False,
        "device": "cpu",
        "simplify": False,
        "verbose": True,
    }


def test_export_helper_honors_graph_and_finalization_options(
    tiny_siglip2_export_model, monkeypatch, tmp_path
):
    captured = {}

    def fake_torch_export(module, dummy, output, **kwargs):
        captured["dummy_shape"] = tuple(dummy.shape)
        captured["torch_output"] = output
        captured["torch_kwargs"] = kwargs

    def fake_finalize(path, **kwargs):
        captured["finalize_path"] = path
        captured["finalize_kwargs"] = kwargs

    monkeypatch.setattr(torch.onnx, "export", fake_torch_export)
    monkeypatch.setattr("libreyolo.export.onnx.finalize_onnx_artifact", fake_finalize)
    output = tmp_path / "options.onnx"

    tiny_siglip2_export_model.export(
        output_path=output,
        opset=None,
        batch=3,
        dynamic=True,
        device="auto",
        simplify=True,
        verbose=True,
    )

    assert captured["dummy_shape"] == (3, 3, 8, 8)
    assert captured["torch_output"] == str(output)
    assert captured["torch_kwargs"]["opset_version"] == 14
    assert captured["torch_kwargs"]["dynamic_axes"] == {
        "images": {0: "batch"},
        "logits": {0: "batch"},
    }
    assert captured["torch_kwargs"]["verbose"] is True
    assert captured["finalize_path"] == str(output)
    assert captured["finalize_kwargs"]["simplify"] is True
    assert captured["finalize_kwargs"]["dynamic"] is True


def test_export_restores_mixed_module_modes_after_trace_failure(
    tiny_siglip2_export_model, monkeypatch, tmp_path
):
    vision = tiny_siglip2_export_model.model.vision_model
    vision.train()
    vision.projection.eval()
    original_modes = [module.training for module in vision.modules()]

    def fail_export(*args, **kwargs):
        raise RuntimeError("synthetic trace failure")

    monkeypatch.setattr(torch.onnx, "export", fail_export)

    with pytest.raises(RuntimeError, match="synthetic trace failure"):
        tiny_siglip2_export_model.export(
            output_path=tmp_path / "failed.onnx",
            simplify=False,
        )

    assert [module.training for module in vision.modules()] == original_modes


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"half": True}, NotImplementedError, "half=True"),
        ({"int8": True}, NotImplementedError, "int8=True"),
        (
            {"int8": True, "fraction": 0.5, "allow_download_scripts": False},
            NotImplementedError,
            "int8=True",
        ),
        ({"batch": 0}, ValueError, "positive integer"),
        ({"batch": True}, ValueError, "positive integer"),
        ({"imgsz": 16}, ValueError, "only supports imgsz=8"),
        ({"imgsz": (8, 7)}, ValueError, "only supports imgsz=8"),
    ],
)
def test_export_rejects_unsupported_requests(
    tiny_siglip2_export_model, tmp_path, kwargs, error, match
):
    output = tmp_path / "should-not-exist.onnx"
    with pytest.raises(error, match=match):
        tiny_siglip2_export_model.export(output_path=output, **kwargs)
    assert not output.exists()


def test_frozen_onnx_roundtrip_records_multilabel_contract(
    tiny_siglip2_export_model, tmp_path
):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    tiny_siglip2_export_model._multi_label = True
    output = tmp_path / "siglip2.onnx"

    exported = tiny_siglip2_export_model.export(
        output_path=output,
        opset=None,
        batch=2,
        dynamic=False,
        device="cpu",
        simplify=False,
    )

    proto = onnx.load(exported)
    metadata = {entry.key: entry.value for entry in proto.metadata_props}
    assert proto.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 2
    assert metadata["model_family"] == "siglip2"
    assert metadata["task"] == "classify"
    assert json.loads(metadata["names"]) == {"0": "cat", "1": "dog"}
    assert json.loads(metadata["classification_mean"]) == [0.5, 0.5, 0.5]
    assert json.loads(metadata["classification_std"]) == [0.5, 0.5, 0.5]
    assert metadata["classification_crop_pct"] == "1.0"
    assert metadata["classification_interpolation"] == "bilinear"
    assert metadata["classification_square_resize"] == "true"
    assert metadata["classification_activation"] == "sigmoid"
    assert metadata["crop_pct"] == "1.0"
    assert metadata["interpolation"] == "bilinear"

    inputs = np.random.default_rng(3).normal(size=(2, 3, 8, 8)).astype(np.float32)
    session = ort.InferenceSession(exported, providers=["CPUExecutionProvider"])
    (actual,) = session.run(None, {"images": inputs})
    with torch.no_grad():
        image_features = tiny_siglip2_export_model.model.encode_image(
            torch.from_numpy(inputs)
        )
        expected = siglip2_logits(
            image_features,
            tiny_siglip2_export_model._text_embeds,
            tiny_siglip2_export_model.model.logit_scale,
            tiny_siglip2_export_model.model.logit_bias,
        ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
