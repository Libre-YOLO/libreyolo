"""Public Florence-2 Core ML routing and export tests."""

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


def _bundle(path):
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"bundle_format": "libreyolo_coreml_florence_bundle"}),
        encoding="utf-8",
    )
    return path


class _FakeFlorenceRuntime:
    calls: ClassVar[list] = []

    def __init__(
        self,
        bundle_path,
        *,
        names,
        compute_units,
    ):
        self.bundle_path = str(bundle_path)
        self.profile = SimpleNamespace(image_size=768, max_new_tokens=23)
        self.processor = object()
        self.closed = False
        self.names = dict(enumerate(names))
        type(self).calls.append(("init", self.bundle_path, tuple(names), compute_units))

    def set_classes(self, names):
        values = [str(value).strip() for value in names]
        if "reject" in values:
            raise ValueError("rejected test vocabulary")
        self.names = dict(enumerate(values))
        type(self).calls.append(("set_classes", tuple(values)))

    def generate(
        self,
        image,
        *,
        max_new_tokens,
        color_format,
    ):
        type(self).calls.append(
            (
                "generate",
                image.size,
                max_new_tokens,
                color_format,
                tuple(self.names.values()),
            )
        )
        return {
            "parsed": {
                "<OPEN_VOCABULARY_DETECTION>": {
                    "bboxes": [[2, 3, 40, 30]],
                    "bboxes_labels": [self.names[0]],
                }
            }
        }

    def close(self):
        self.closed = True
        type(self).calls.append(("close",))


def test_librevlm_routes_florence_bundle_and_predicts(
    monkeypatch,
    tmp_path,
):
    from libreyolo.models.vlm import CoreMLFlorence2, LibreVLM

    bundle = _bundle(tmp_path / "florence.coremlvlm")
    _FakeFlorenceRuntime.calls = []
    monkeypatch.setattr(
        "libreyolo.backends.coreml_florence.CoreMLFlorenceRuntime",
        _FakeFlorenceRuntime,
    )

    model = LibreVLM(
        str(bundle),
        names=["cat"],
        compute_units="cpu_only",
        max_new_tokens=7,
    )
    assert isinstance(model, CoreMLFlorence2)
    assert model.family == "florence2"
    assert model.size == "base"
    assert model.input_size == 768
    assert model.MAX_NEW_TOKENS == 7

    image = Image.fromarray(np.zeros((36, 64, 3), dtype=np.uint8))
    result = model.predict(image)
    np.testing.assert_allclose(
        result.boxes.xyxy.cpu().numpy(),
        np.asarray([[2.0, 3.0, 40.0, 30.0]], dtype=np.float32),
    )
    assert result.boxes.cls.tolist() == [0.0]
    assert result.names == {0: "cat"}
    assert _FakeFlorenceRuntime.calls[-1] == (
        "generate",
        (64, 36),
        7,
        "rgb",
        ("cat",),
    )

    previous = dict(model.names)
    with pytest.raises(ValueError, match="rejected"):
        model.set_classes(["reject"])
    assert model.names == previous

    model.close()
    assert model._coreml_runtime.closed is True
    assert model.processor is None


def test_florence_bundle_public_default_requests_validated_policy(
    monkeypatch,
    tmp_path,
):
    from libreyolo.models.vlm import LibreVLM

    bundle = _bundle(tmp_path / "florence.coremlvlm")
    _FakeFlorenceRuntime.calls = []
    monkeypatch.setattr(
        "libreyolo.backends.coreml_florence.CoreMLFlorenceRuntime",
        _FakeFlorenceRuntime,
    )

    model = LibreVLM(str(bundle), names=["cat"])
    assert _FakeFlorenceRuntime.calls[0] == (
        "init",
        str(bundle),
        ("cat",),
        "validated",
    )
    model.close()


class _TinyFlorenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))


def _native_florence_for_export(tmp_path):
    from libreyolo.models.vlm.florence2 import LibreFlorence2

    model = object.__new__(LibreFlorence2)
    model.size = "base"
    model.device = torch.device("cpu")
    model.model = _TinyFlorenceModel()
    model._ensure_weights = lambda: str(tmp_path / "processor")
    (tmp_path / "processor").mkdir()
    return model


def test_native_florence_export_builds_portable_bundle(
    monkeypatch,
    tmp_path,
):
    from libreyolo.backends import coreml_florence as backend
    from libreyolo.export import coreml_florence

    model = _native_florence_for_export(tmp_path)
    output = tmp_path / "florence.coremlvlm"
    captured = {}

    def fake_export(source, **kwargs):
        captured["source"] = source
        captured["export"] = kwargs
        kwargs["output_path"].mkdir()
        return str(kwargs["output_path"])

    def fake_bundle(package, **kwargs):
        captured["package"] = package
        captured["bundle"] = kwargs
        kwargs["output_path"].mkdir()
        return str(kwargs["output_path"])

    monkeypatch.setattr(
        coreml_florence,
        "export_florence2_base_coreml_package",
        fake_export,
    )
    monkeypatch.setattr(
        backend,
        "build_coreml_florence_bundle",
        fake_bundle,
    )

    result = model.export(
        format="coreml",
        output_path=output,
        compute_units="cpu_only",
    )
    assert result == str(output)
    assert captured["source"] is model.model
    assert captured["export"]["processor_revision"] == (
        coreml_florence.FLORENCE2_BASE_REVISION
    )
    assert captured["export"]["compute_units"] == "cpu_only"
    assert captured["bundle"]["processor_dir"] == tmp_path / "processor"
    assert captured["bundle"]["output_path"] == output
    assert captured["bundle"]["move_model"] is True


def test_native_florence_export_default_rejects_before_weight_access(
    tmp_path,
):
    model = _native_florence_for_export(tmp_path)
    weight_calls = []
    model._ensure_weights = lambda: weight_calls.append("weights")

    with pytest.raises(NotImplementedError, match="exact Apple-M4"):
        model.export(
            format="coreml",
            output_path=tmp_path / "florence.coremlvlm",
        )

    assert weight_calls == []


def test_native_florence_export_rejects_unreviewed_variants(
    tmp_path,
):
    model = _native_florence_for_export(tmp_path)
    output = tmp_path / "florence.coremlvlm"

    model.size = "large"
    with pytest.raises(NotImplementedError, match="only.*base"):
        model.export(format="coreml", output_path=output)
    model.size = "base"

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


def test_coreml_vlm_dispatch_rejects_unknown_or_duplicate_format(
    tmp_path,
):
    from libreyolo.models.vlm import LibreVLM

    unknown = _bundle(tmp_path / "unknown.coremlvlm")
    (unknown / "manifest.json").write_text(
        '{"bundle_format":"not_libreyolo"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported"):
        LibreVLM(str(unknown))

    duplicate = tmp_path / "duplicate.coremlvlm"
    duplicate.mkdir()
    (duplicate / "manifest.json").write_text(
        '{"bundle_format":"one","bundle_format":"two"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="repeats key"):
        LibreVLM(str(duplicate))


def test_florence_coreml_support_is_experimental():
    from libreyolo.export.support import get_support

    entry = get_support("florence2", "detect", "coreml")
    assert entry.tier == "experimental"
    assert "Florence-2-base" in str(entry.constraint)
    assert ".coremlvlm" in str(entry.constraint)
