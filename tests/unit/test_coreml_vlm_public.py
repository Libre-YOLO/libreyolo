"""Public LibreVLM routing and export tests for the Core ML VLM bundle."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

pytestmark = pytest.mark.unit


def _portable_bundle(path, *, bundle_format="libreyolo_coreml_vlm_bundle"):
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"bundle_format": bundle_format}),
        encoding="utf-8",
    )
    return path


class _FakeCoreMLVLMRuntime:
    calls: ClassVar[list] = []

    def __init__(self, bundle_path, *, compute_units):
        self.bundle_path = str(bundle_path)
        self.compute_units = compute_units
        self.profile = SimpleNamespace(max_new_tokens=37)
        self.processor = object()
        self.closed = False
        type(self).calls.append(("init", self.bundle_path, compute_units))

    def chat(
        self,
        image,
        prompt,
        *,
        max_new_tokens,
        color_format,
    ):
        type(self).calls.append(
            (
                "chat",
                image.size,
                prompt,
                max_new_tokens,
                color_format,
            )
        )
        return '[{"label":"cat","bbox":[0.1,0.2,0.5,0.6]}]'

    def close(self):
        self.closed = True
        type(self).calls.append(("close",))


def test_librevlm_routes_bundle_through_smol_runtime(
    monkeypatch,
    tmp_path,
):
    from libreyolo.models.vlm import CoreMLSmolVLM2, LibreVLM

    bundle = _portable_bundle(tmp_path / "renamed.coremlvlm")
    _FakeCoreMLVLMRuntime.calls = []
    monkeypatch.setattr(
        "libreyolo.backends.coreml_vlm.CoreMLVLMRuntime",
        _FakeCoreMLVLMRuntime,
    )

    model = LibreVLM(
        str(bundle),
        names=["cat"],
        compute_units="cpu_only",
    )
    assert isinstance(model, CoreMLSmolVLM2)
    assert model.family == "smolvlm2"
    assert model.size == "500m"
    assert model.task == "detect"
    assert model.input_size == 2048
    assert model.MAX_NEW_TOKENS == 37
    assert _FakeCoreMLVLMRuntime.calls[0] == (
        "init",
        str(bundle),
        "cpu_only",
    )

    image = Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8))
    result = model.predict(image)
    np.testing.assert_allclose(
        result.boxes.xyxy.cpu().numpy(),
        np.asarray([[10.0, 10.0, 50.0, 30.0]], dtype=np.float32),
    )
    assert result.boxes.cls.tolist() == [0.0]
    assert result.names == {0: "cat"}
    assert "Detect all instances of: cat." in _FakeCoreMLVLMRuntime.calls[1][2]

    text = model.chat(image, "describe", max_new_tokens=5)
    assert '"cat"' in text
    assert _FakeCoreMLVLMRuntime.calls[-1][3] == 5

    model.close()
    assert model._coreml_runtime.closed is True
    assert model.processor is None


def test_librevlm_bundle_path_and_runtime_options_fail_closed(
    monkeypatch,
    tmp_path,
):
    from libreyolo.models.vlm import LibreVLM

    missing = tmp_path / "missing.coremlvlm"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LibreVLM(str(missing))

    bundle = _portable_bundle(tmp_path / "model.coremlvlm")
    _FakeCoreMLVLMRuntime.calls = []
    monkeypatch.setattr(
        "libreyolo.backends.coreml_vlm.CoreMLVLMRuntime",
        _FakeCoreMLVLMRuntime,
    )
    with pytest.raises(ValueError, match="compute_units"):
        LibreVLM(str(bundle), device="cuda")
    assert _FakeCoreMLVLMRuntime.calls == []

    with pytest.raises(ValueError, match="reviewed limit"):
        LibreVLM(str(bundle), max_new_tokens=38)
    assert _FakeCoreMLVLMRuntime.calls[-1] == ("close",)


def test_coreml_smol_predict_rejects_imgsz_override(
    monkeypatch,
    tmp_path,
):
    from libreyolo.models.vlm import LibreVLM

    bundle = _portable_bundle(tmp_path / "model.coremlvlm")
    monkeypatch.setattr(
        "libreyolo.backends.coreml_vlm.CoreMLVLMRuntime",
        _FakeCoreMLVLMRuntime,
    )
    model = LibreVLM(str(bundle), names=["cat"])
    assert _FakeCoreMLVLMRuntime.calls[-1] == (
        "init",
        str(bundle),
        "validated",
    )
    image = Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="fixed 2048x2048"):
        model.predict(image, imgsz=512)
    model.close()


class _TinySmolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))


def _native_smol_for_export(tmp_path):
    from libreyolo.models.vlm.smolvlm import LibreSmolVLM2

    model = object.__new__(LibreSmolVLM2)
    model.size = "500m"
    model.device = torch.device("cpu")
    model.model = _TinySmolModel()
    model._ensure_weights = lambda: str(tmp_path / "processor")
    (tmp_path / "processor").mkdir()
    return model


def test_native_smol_export_builds_portable_bundle(
    monkeypatch,
    tmp_path,
):
    from libreyolo.backends import coreml_vlm as backend
    from libreyolo.export import coreml_vlm

    model = _native_smol_for_export(tmp_path)
    output = tmp_path / "smol.coremlvlm"
    captured = {}

    def fake_export(source, **kwargs):
        captured["source"] = source
        captured["export"] = kwargs
        package = kwargs["output_path"]
        package.mkdir()
        return str(package)

    def fake_bundle(package, **kwargs):
        captured["package"] = package
        captured["bundle"] = kwargs
        kwargs["output_path"].mkdir()
        return str(kwargs["output_path"])

    monkeypatch.setattr(
        coreml_vlm,
        "export_smolvlm2_500m_coreml_package",
        fake_export,
    )
    monkeypatch.setattr(backend, "build_coreml_vlm_bundle", fake_bundle)

    result = model.export(
        format="coreml",
        output_path=output,
        context_length=2048,
        compute_units="cpu_only",
    )
    assert result == str(output)
    assert captured["source"] is model.model
    assert captured["export"]["processor_revision"] == (
        coreml_vlm.SMOLVLM2_500M_REVISION
    )
    assert captured["export"]["context_length"] == 2048
    assert captured["export"]["compute_units"] == "cpu_only"
    assert captured["bundle"]["move_model"] is True
    assert captured["bundle"]["processor_dir"] == tmp_path / "processor"
    assert captured["bundle"]["output_path"] == output


def test_native_smol_export_default_rejects_before_weight_access(
    tmp_path,
):
    model = _native_smol_for_export(tmp_path)
    weight_calls = []
    model._ensure_weights = lambda: weight_calls.append("weights")

    with pytest.raises(NotImplementedError, match="exact Apple-M4"):
        model.export(
            format="coreml",
            output_path=tmp_path / "smol.coremlvlm",
            context_length=2048,
        )

    assert weight_calls == []


@pytest.mark.parametrize("context_length", [8192, 1024, True])
def test_native_smol_export_rejects_unreviewed_context(
    tmp_path,
    context_length,
):
    model = _native_smol_for_export(tmp_path)
    with pytest.raises((TypeError, ValueError), match="context"):
        model.export(
            format="coreml",
            output_path=tmp_path / "smol.coremlvlm",
            context_length=context_length,
        )


def test_native_smol_export_rejects_size_device_options_and_overwrite(
    tmp_path,
):
    model = _native_smol_for_export(tmp_path)
    output = tmp_path / "smol.coremlvlm"

    model.size = "2.2b"
    with pytest.raises(NotImplementedError, match="only.*500M"):
        model.export(format="coreml", output_path=output)
    model.size = "500m"

    model.device = torch.device("cuda")
    with pytest.raises(NotImplementedError, match="CPU-loaded"):
        model.export(format="coreml", output_path=output)
    model.device = torch.device("cpu")

    with pytest.raises(TypeError, match="irrelevant"):
        model.export(
            format="coreml",
            output_path=output,
            dynamic=False,
        )
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        model.export(
            format="coreml",
            output_path=output,
            compute_units="cpu_only",
        )


def test_smolvlm_coreml_support_is_experimental():
    from libreyolo.export.support import get_support

    entry = get_support("smolvlm2", "detect", "coreml")
    assert entry.tier == "experimental"
    assert ".coremlvlm" in str(entry.constraint)
