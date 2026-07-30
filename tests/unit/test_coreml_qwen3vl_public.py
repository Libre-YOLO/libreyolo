"""Public Qwen3-VL Core ML routing and adapter tests."""

from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

pytestmark = pytest.mark.unit


def _bundle(path):
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"bundle_format": "libreyolo_coreml_qwen3vl_bundle"}),
        encoding="utf-8",
    )
    return path


class _FakeProcessor:
    def batch_decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens is True
        assert np.asarray(tokens).shape == (1, 3)
        return ['[{"bbox_2d":[100,200,800,900],"label":"bus"}]']


class _FakeQwenRuntime:
    calls: ClassVar[list] = []

    def __init__(self, bundle_path, *, compute_units):
        self.bundle_path = str(bundle_path)
        self.max_new_tokens = 48
        self.processor = _FakeProcessor()
        self.closed = False
        type(self).calls.append(("init", self.bundle_path, compute_units))

    def generate(self, image, prompt, *, max_new_tokens):
        type(self).calls.append(
            ("generate", image.size, prompt, max_new_tokens)
        )
        return np.asarray([[7, 8, 9]], dtype=np.int64)

    def close(self):
        self.closed = True
        type(self).calls.append(("close",))


def test_librevlm_routes_qwen_bundle_and_predicts(monkeypatch, tmp_path):
    from libreyolo.models.vlm import CoreMLQwen3VL, LibreVLM

    bundle = _bundle(tmp_path / "qwen.coremlvlm")
    _FakeQwenRuntime.calls = []
    monkeypatch.setattr(
        "libreyolo.backends.coreml_qwen3vl.CoreMLQwen3VLRuntime",
        _FakeQwenRuntime,
    )

    model = LibreVLM(
        str(bundle),
        names=["bus"],
        compute_units="cpu_only",
        max_new_tokens=12,
    )
    assert isinstance(model, CoreMLQwen3VL)
    assert model.family == "qwen3vl"
    assert model.size == "2b"
    assert model.input_size == 448
    assert model.MAX_NEW_TOKENS == 12

    image = Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8))
    result = model.predict(image)
    np.testing.assert_allclose(
        result.boxes.xyxy.cpu().numpy(),
        np.asarray([[10.0, 10.0, 80.0, 45.0]], dtype=np.float32),
    )
    assert result.boxes.cls.tolist() == [0.0]
    assert result.names == {0: "bus"}
    call = _FakeQwenRuntime.calls[-1]
    assert call[0:2] == ("generate", (100, 50))
    assert "Detect all instances of: bus." in call[2]
    assert call[3] == 12

    assert model.chat(image, "Count buses.", max_new_tokens=4).startswith("[")
    model.close()
    assert model._coreml_runtime.closed is True
    assert model.processor is None


def test_qwen_bundle_rejects_invalid_public_options(monkeypatch, tmp_path):
    from libreyolo.models.vlm import LibreVLM

    bundle = _bundle(tmp_path / "qwen.coremlvlm")
    monkeypatch.setattr(
        "libreyolo.backends.coreml_qwen3vl.CoreMLQwen3VLRuntime",
        _FakeQwenRuntime,
    )
    with pytest.raises(ValueError, match="device"):
        LibreVLM(str(bundle), device="cuda")
    with pytest.raises(ValueError, match="between 1 and 48"):
        LibreVLM(str(bundle), max_new_tokens=49)


def test_qwen_bundle_dispatch_constant_matches_manifest():
    from libreyolo.backends.coreml_qwen3vl import (
        COREML_QWEN3VL_BUNDLE_FORMAT,
    )
    from libreyolo.export.coreml_qwen3vl import qwen3vl_bundle_manifest

    assert (
        COREML_QWEN3VL_BUNDLE_FORMAT
        == qwen3vl_bundle_manifest()["bundle_format"]
    )


def _native_qwen_for_export(tmp_path):
    from libreyolo.models.vlm.qwen3vl import LibreQwen3VL

    model = object.__new__(LibreQwen3VL)
    model.size = "2b"
    model.device = torch.device("cpu")
    model.model = nn.Linear(2, 2)
    processor = tmp_path / "processor"
    processor.mkdir()
    model._ensure_weights = lambda: str(processor)
    return model, processor


def test_native_qwen_export_builds_portable_bundle(monkeypatch, tmp_path):
    from libreyolo.export import coreml_qwen3vl

    model, processor = _native_qwen_for_export(tmp_path)
    output = tmp_path / "qwen.coremlvlm"
    captured = {}
    monkeypatch.setattr(
        coreml_qwen3vl,
        "validate_qwen3vl_processor_assets",
        lambda path: captured.setdefault("processor", path),
    )
    monkeypatch.setattr(
        coreml_qwen3vl,
        "validate_qwen3vl_weight_asset",
        lambda path: captured.setdefault("weights", path),
    )
    monkeypatch.setattr(
        coreml_qwen3vl,
        "validate_qwen3vl_source_model",
        lambda source: captured.setdefault("source", source),
    )

    def fake_export(source, **kwargs):
        captured["export"] = (source, kwargs)
        kwargs["output_dir"].mkdir()
        return {}

    def fake_bundle(component_dir, **kwargs):
        captured["bundle"] = (component_dir, kwargs)
        kwargs["output_path"].mkdir()
        return str(kwargs["output_path"])

    monkeypatch.setattr(
        coreml_qwen3vl,
        "export_qwen3vl_coreml_components",
        fake_export,
    )
    monkeypatch.setattr(
        coreml_qwen3vl,
        "build_qwen3vl_coreml_bundle",
        fake_bundle,
    )
    result = model.export(
        format="coreml",
        output_path=output,
        compute_units="cpu_only",
    )
    assert result == str(output)
    assert captured["processor"] == processor
    assert captured["weights"] == processor
    assert captured["source"] is model.model
    assert captured["export"][0] is model.model
    assert captured["export"][1]["checkpoint_dir"] == processor
    assert captured["export"][1]["compute_units"] == "cpu_only"
    assert captured["bundle"][1]["processor_dir"] == processor
    assert captured["bundle"][1]["output_path"] == output


def test_native_qwen_export_default_fails_before_weight_access(tmp_path):
    model, _processor = _native_qwen_for_export(tmp_path)
    calls = []
    model._ensure_weights = lambda: calls.append("weights")
    with pytest.raises(ValueError, match="cpu_only"):
        model.export(
            format="coreml",
            output_path=tmp_path / "qwen.coremlvlm",
        )
    assert calls == []
