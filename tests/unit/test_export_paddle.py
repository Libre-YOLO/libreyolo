"""Unit tests for Paddle export and runtime integration."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from libreyolo.export.exporter import PaddleExporter

pytestmark = pytest.mark.unit


def _wrapper(family: str = "yolo9", task: str = "detect") -> MagicMock:
    model = MagicMock()
    model._get_model_name.return_value = family
    model.task = task
    return model


def test_paddle_exporter_rejects_dynamic_batch_and_unsimplified_graph():
    exporter = PaddleExporter(_wrapper())
    with pytest.raises(ValueError, match="dynamic=False"):
        exporter(dynamic=True)
    with pytest.raises(ValueError, match="batch=1"):
        exporter(batch=2)
    with pytest.raises(ValueError, match="simplify=True"):
        exporter(simplify=False)


def test_rfdetr_block_happens_before_dependency_check(monkeypatch):
    from libreyolo.export import paddle as paddle_export

    dependency_check = MagicMock(side_effect=AssertionError("must not run"))
    monkeypatch.setattr(
        paddle_export, "check_paddle_export_available", dependency_check
    )
    exporter = PaddleExporter(_wrapper("rfdetr"))
    with pytest.raises(NotImplementedError, match="GridSample"):
        exporter._preflight(half=False, int8=False, data=None)
    dependency_check.assert_not_called()


def test_export_paddle_keeps_only_runtime_artifacts_and_metadata(monkeypatch, tmp_path):
    from libreyolo.export import paddle as paddle_export

    calls = {}

    def fake_convert(model_path, save_dir, **kwargs):
        calls.update(model_path=model_path, save_dir=save_dir, kwargs=kwargs)
        inference = Path(save_dir) / "inference_model"
        inference.mkdir(parents=True)
        (inference / "model.pdmodel").write_bytes(b"model")
        (inference / "model.pdiparams").write_bytes(b"parameters")
        (inference / "model.pdiparams.info").write_bytes(b"info")
        (Path(save_dir) / "x2paddle_code.py").write_text("generated")

    package = types.ModuleType("x2paddle")
    convert = types.ModuleType("x2paddle.convert")
    convert.onnx2paddle = fake_convert
    package.convert = convert
    monkeypatch.setitem(sys.modules, "x2paddle", package)
    monkeypatch.setitem(sys.modules, "x2paddle.convert", convert)
    monkeypatch.setattr(paddle_export, "check_paddle_export_available", lambda: None)
    monkeypatch.setattr(
        paddle_export, "_normalize_onnx_for_x2paddle", lambda path: None
    )

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    output = tmp_path / "model_paddle"
    result = paddle_export.export_paddle(
        str(onnx_path),
        str(output),
        metadata={"model_family": "yolo9", "task": "detect"},
    )

    assert result == str(output)
    assert {path.name for path in output.iterdir()} == {
        "metadata.yaml",
        "model.pdiparams",
        "model.pdiparams.info",
        "model.pdmodel",
    }
    metadata = yaml.safe_load((output / "metadata.yaml").read_text())
    assert metadata == {"model_family": "yolo9", "task": "detect"}
    assert calls["kwargs"] == {
        "enable_optim": False,
        "disable_feedback": True,
        "enable_onnx_checker": True,
    }


def test_x2paddle_onnx_normalization_removes_only_default_dilation(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from libreyolo.export.paddle import _normalize_onnx_for_x2paddle

    nodes = [
        helper.make_node(
            "MaxPool", ["input"], ["middle"], kernel_shape=[3, 3], dilations=[1, 1]
        ),
        helper.make_node(
            "MaxPool", ["middle"], ["output"], kernel_shape=[3, 3], dilations=[2, 1]
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "maxpool",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 10, 12])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=7
    )
    path = tmp_path / "model.onnx"
    onnx.save(model, path)

    _normalize_onnx_for_x2paddle(path)

    normalized = onnx.load(path)
    first_names = {attribute.name for attribute in normalized.graph.node[0].attribute}
    second = {
        attribute.name: list(attribute.ints)
        for attribute in normalized.graph.node[1].attribute
    }
    assert "dilations" not in first_names
    assert second["dilations"] == [2, 1]


def test_x2paddle_onnx_normalization_rejects_newer_opset(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from libreyolo.export.paddle import _normalize_onnx_for_x2paddle

    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "newer",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    path = tmp_path / "model.onnx"
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=7
        ),
        path,
    )
    with pytest.raises(NotImplementedError, match="opset 15 or lower"):
        _normalize_onnx_for_x2paddle(path)


class _InputHandle:
    def __init__(self):
        self.value = None
        self.shape = None

    def reshape(self, shape):
        self.shape = tuple(shape)

    def copy_from_cpu(self, value):
        self.value = value.copy()


class _OutputHandle:
    def __init__(self, value):
        self.value = value

    def copy_to_cpu(self):
        return self.value.copy()


class _Predictor:
    def __init__(self):
        self.input = _InputHandle()
        self.output = np.ones((1, 84, 4), dtype=np.float32)

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["output"]

    def get_input_handle(self, name):
        assert name == "images"
        return self.input

    def get_output_handle(self, name):
        assert name == "output"
        return _OutputHandle(self.output)

    def run(self):
        return True


def _install_fake_paddle(monkeypatch):
    predictor = _Predictor()

    class Config:
        def __init__(self, model, parameters):
            self.model = model
            self.parameters = parameters
            self.cpu = False
            self.memory_optimized = False

        def disable_gpu(self):
            self.cpu = True

        def enable_memory_optim(self):
            self.memory_optimized = True

    inference = types.ModuleType("paddle.inference")
    inference.Config = Config
    inference.create_predictor = lambda config: predictor
    paddle = types.ModuleType("paddle")
    paddle.__path__ = []
    paddle.inference = inference
    monkeypatch.setitem(sys.modules, "paddle", paddle)
    monkeypatch.setitem(sys.modules, "paddle.inference", inference)
    return predictor


def _paddle_artifact(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.pdmodel").write_bytes(b"model")
    (artifact / "model.pdiparams").write_bytes(b"parameters")
    (artifact / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "model_family": "yolo9",
                "size": "t",
                "task": "detect",
                "supported_tasks": ["detect"],
                "default_task": "detect",
                "nc": 2,
                "names": {"0": "first", "1": "second"},
                "imgsz": 320,
                "imgsz_h": 320,
                "imgsz_w": 320,
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_paddle_backend_reads_metadata_and_runs_cpu(monkeypatch, tmp_path):
    predictor = _install_fake_paddle(monkeypatch)
    from libreyolo.backends.paddle import PaddleBackend

    backend = PaddleBackend(_paddle_artifact(tmp_path), device="cpu")
    blob = np.zeros((1, 3, 320, 320), dtype=np.float32)
    outputs = backend._run_inference(blob)

    assert backend.model_family == "yolo9"
    assert backend.model_size == "t"
    assert backend.task == "detect"
    assert backend.imgsz == 320
    assert backend.names == {0: "first", 1: "second"}
    assert predictor.input.shape == blob.shape
    assert np.array_equal(predictor.input.value, blob)
    assert np.array_equal(outputs[0], predictor.output)


def test_factory_routes_paddle_artifact(monkeypatch, tmp_path):
    _install_fake_paddle(monkeypatch)
    import libreyolo.backends.paddle as paddle_backend
    from libreyolo.models import LibreYOLO

    artifact = _paddle_artifact(tmp_path)
    sentinel = object()
    factory = MagicMock(return_value=sentinel)
    monkeypatch.setattr(paddle_backend, "PaddleBackend", factory)

    assert LibreYOLO(str(artifact), device="cpu") is sentinel
    factory.assert_called_once_with(
        str(artifact), nb_classes=None, device="cpu", task=None
    )
