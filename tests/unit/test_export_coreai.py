"""Core AI exporter contracts that do not require Apple hardware."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from libreyolo.export import coreai, coreai_compat
from libreyolo.export.exporter import BaseExporter, CoreAIExporter

pytestmark = pytest.mark.unit


class _Wrapper:
    task = "detect"

    def __init__(self, family="yolo9"):
        self.family = family

    def _get_model_name(self):
        return self.family


def test_coreai_format_defaults_static_and_rejects_dynamic(monkeypatch):
    calls = []

    def fake_call(self, **kwargs):
        calls.append(kwargs)
        return "model.aimodel"

    monkeypatch.setattr(BaseExporter, "__call__", fake_call)
    exporter = CoreAIExporter(_Wrapper())

    assert exporter() == "model.aimodel"
    assert calls == [{"dynamic": False}]
    with pytest.raises(NotImplementedError, match="dynamic=True"):
        exporter(dynamic=True)


def test_coreai_rejects_unsupported_precision():
    exporter = CoreAIExporter(_Wrapper())
    with pytest.raises(NotImplementedError, match="FP16"):
        exporter._validate(True, False, None)
    with pytest.raises(NotImplementedError, match="INT8"):
        exporter._validate(False, True, None)


def test_low_level_coreai_rejects_dynamic_and_precision_before_import():
    model = nn.Identity()
    dummy = torch.zeros(1, 3, 8, 8)
    with pytest.raises(NotImplementedError, match="dynamic=True"):
        coreai.export_coreai(model, dummy, output_path="unused", dynamic=True)
    with pytest.raises(NotImplementedError, match="FP32 only"):
        coreai.export_coreai(model, dummy, output_path="unused", precision="fp16")


def test_blocked_support_is_checked_before_dependency(monkeypatch):
    def dependency_must_not_run():
        raise AssertionError("dependency check ran before support policy")

    monkeypatch.setattr(coreai, "_require_coreai", dependency_must_not_run)
    exporter = CoreAIExporter(_Wrapper("unwired_family"))
    with pytest.raises(NotImplementedError, match="coreai"):
        exporter._preflight(half=False, int8=False, data=None)


def test_structured_metadata_uses_json():
    assert coreai._stringify_metadata_value(["a", "b"]) == '["a","b"]'
    assert coreai._stringify_metadata_value({1: "one"}) == '{"1":"one"}'
    assert coreai._stringify_metadata_value(("x", 2)) == '["x",2]'


def test_frozen_classifier_helper_runs_shared_validation(monkeypatch):
    calls = []

    def fake_validate(self, half, int8, data):
        calls.append(("validate", half, int8, data))
        return half, int8

    def fake_preflight(self, **kwargs):
        calls.append(("preflight", kwargs))

    def fake_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        calls.append(("metadata", precision, dynamic, onnx_path, imgsz))
        return {"precision": precision, "imgsz": imgsz}

    monkeypatch.setattr(CoreAIExporter, "_validate", fake_validate)
    monkeypatch.setattr(CoreAIExporter, "_preflight", fake_preflight)
    monkeypatch.setattr(CoreAIExporter, "_build_metadata", fake_metadata)
    owner = SimpleNamespace(input_size=224)

    size, output, metadata = coreai.prepare_frozen_classifier_export(
        owner,
        {"imgsz": (192, 192), "output_path": "frozen", "device": "cpu"},
        default_output="unused",
    )
    assert (size, output) == (192, "frozen")
    assert metadata == {"precision": "fp32", "imgsz": (192, 192)}
    assert calls == [
        ("validate", False, False, None),
        (
            "preflight",
            {"half": False, "int8": False, "data": None, "nms": False},
        ),
        ("metadata", "fp32", False, None, (192, 192)),
    ]


def test_compat_shim_declines_unverified_toolchain_version(monkeypatch):
    monkeypatch.setattr(coreai_compat, "version", lambda package: "0.4.2")
    assert coreai_compat._patch_avg_pool2d() is False


def test_manual_grid_sample_flag_is_nestable():
    from libreyolo.models.dfine.ms_deform import _FORCE_MANUAL_GRID_SAMPLE

    assert _FORCE_MANUAL_GRID_SAMPLE.get() is False
    restore_outer = coreai._force_manual_grid_sample()
    assert _FORCE_MANUAL_GRID_SAMPLE.get() is True
    restore_inner = coreai._force_manual_grid_sample()
    restore_inner()
    assert _FORCE_MANUAL_GRID_SAMPLE.get() is True
    restore_outer()
    assert _FORCE_MANUAL_GRID_SAMPLE.get() is False


def test_graph_preparation_restores_snapshot_when_prepare_fails(monkeypatch):
    restored = []
    monkeypatch.setattr(
        coreai,
        "_snapshot_rtdetr_static_eval",
        lambda model: lambda: restored.append(model),
    )

    def fail_prepare(*args):
        raise RuntimeError("preparation failed")

    monkeypatch.setattr(coreai, "_prepare_rtdetr_static_eval", fail_prepare)
    model = nn.Identity()
    dummy = torch.zeros(1, 3, 8, 8)
    with (
        pytest.raises(RuntimeError, match="preparation failed"),
        coreai._prepare_coreai_graph(model, dummy, "dfine"),
    ):
        pass
    assert restored == [model]


def test_graph_preparation_restores_replaced_pool_after_capture_error():
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1))
    original = model[0]
    dummy = torch.zeros(1, 3, 8, 8)
    with (
        pytest.raises(RuntimeError, match="capture failed"),
        coreai._prepare_coreai_graph(model, dummy, "resnet"),
    ):
        assert model[0] is not original
        raise RuntimeError("capture failed")
    assert model[0] is original


def test_anchor_freeze_reaches_wrapped_model_and_restores_cache():
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.export = True
            self.anchors = torch.tensor([])
            self.strides = torch.tensor([])
            self.shape = None

        def _anchor_grid(self, features):
            return self.anchors, self.strides

    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = Head()

        def forward(self, tensor):
            self.head.anchors = torch.ones(2, 4)
            self.head.strides = torch.full((2, 4), 8.0)
            self.head.shape = tuple(tensor.shape)
            return tensor

    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, tensor):
            return self.model(tensor)

    detector = Detector()
    wrapped = Wrapper(detector)
    old_anchors = detector.head.anchors
    old_strides = detector.head.strides
    restore = coreai._freeze_anchor_grid(wrapped, torch.zeros(1, 3, 8, 8))
    assert "_anchor_grid" in detector.head.__dict__
    restore()

    assert detector.head.anchors is old_anchors
    assert detector.head.strides is old_strides
    assert detector.head.shape is None
    assert "_anchor_grid" not in detector.head.__dict__
    assert detector.head.export is True


def test_rtdetr_snapshot_removes_new_attributes_and_buffers():
    class Encoder(nn.Module):
        use_encoder_idx = (0,)

        def build_2d_sincos_position_embedding(self):
            pass

    class Decoder(nn.Module):
        def _generate_anchors(self):
            pass

    model = SimpleNamespace(encoder=Encoder(), decoder=Decoder())
    restore = coreai._snapshot_rtdetr_static_eval(model)
    model.encoder.eval_spatial_size = (8, 8)
    model.encoder.pos_embed0 = torch.ones(1)
    model.decoder.eval_spatial_size = (8, 8)
    model.decoder.register_buffer("anchors", torch.ones(1))
    model.decoder.register_buffer("valid_mask", torch.ones(1))
    restore()

    assert not hasattr(model.encoder, "eval_spatial_size")
    assert not hasattr(model.encoder, "pos_embed0")
    assert not hasattr(model.decoder, "eval_spatial_size")
    assert "anchors" not in model.decoder._buffers
    assert "valid_mask" not in model.decoder._buffers


def test_rtdetr_snapshot_preserves_existing_none_buffer():
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("anchors", None)

        def _generate_anchors(self):
            pass

    model = SimpleNamespace(encoder=None, decoder=Decoder())
    restore = coreai._snapshot_rtdetr_static_eval(model)
    model.decoder.anchors = torch.ones(1)
    restore()
    assert "anchors" in model.decoder._buffers
    assert model.decoder.anchors is None
