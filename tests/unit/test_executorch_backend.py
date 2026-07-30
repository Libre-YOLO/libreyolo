"""Unit tests for ExecuTorch sidecar loading and runtime execution."""

from __future__ import annotations

import json
import sys
import types
from typing import ClassVar

import numpy as np
import pytest

from libreyolo.backends.executorch import ExecuTorchBackend

pytestmark = pytest.mark.unit


class _FakeRegistry:
    def __init__(self, available: bool = True):
        self.available = available

    def is_available(self, name):
        return self.available and name == "XnnpackBackend"


class _FakeMethod:
    def execute(self, inputs):
        return [inputs[0] + 1, inputs[0].mean(dim=(2, 3))]


class _FakeProgram:
    method_names: ClassVar = {"forward"}

    def load_method(self, name):
        assert name == "forward"
        return _FakeMethod()


def _install_fake_runtime(monkeypatch, *, backend_available=True):
    runtime = types.ModuleType("executorch.runtime")

    class Runtime:
        backend_registry = _FakeRegistry(backend_available)

        @classmethod
        def get(cls):
            return cls

        @classmethod
        def load_program(cls, data):
            assert isinstance(data, bytes)
            return _FakeProgram()

    runtime.Runtime = Runtime
    package = types.ModuleType("executorch")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "executorch", package)
    monkeypatch.setitem(sys.modules, "executorch.runtime", runtime)


def _write_artifact(tmp_path):
    path = tmp_path / "model.pte"
    path.write_bytes(b"fake-pte")
    metadata = {
        "schema_version": "1.0",
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "supported_tasks": ["detect"],
        "default_task": "detect",
        "nc": 2,
        "names": {"0": "a", "1": "b"},
        "imgsz": 64,
        "imgsz_h": 64,
        "imgsz_w": 64,
        "precision": "fp32",
        "dynamic": False,
        "executorch_version": "1.3.0",
        "executorch_delegate": "xnnpack",
    }
    (tmp_path / "model.pte.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return path


def test_backend_loads_sidecar_and_returns_contiguous_numpy(tmp_path, monkeypatch):
    _install_fake_runtime(monkeypatch)
    path = _write_artifact(tmp_path)

    backend = ExecuTorchBackend(str(path))
    outputs = backend._run_inference(np.zeros((1, 3, 64, 64), dtype=np.float64))

    assert backend.model_family == "yolo9"
    assert backend.task == "detect"
    assert backend.names == {0: "a", 1: "b"}
    assert [output.shape for output in outputs] == [(1, 3, 64, 64), (1, 3)]
    assert all(output.flags.c_contiguous for output in outputs)
    assert all(output.dtype == np.float32 for output in outputs)


def test_backend_requires_xnnpack_runtime(tmp_path, monkeypatch):
    _install_fake_runtime(monkeypatch, backend_available=False)
    path = _write_artifact(tmp_path)

    with pytest.raises(RuntimeError, match="XnnpackBackend"):
        ExecuTorchBackend(str(path))


def test_backend_rejects_missing_or_wrong_sidecar(tmp_path, monkeypatch):
    _install_fake_runtime(monkeypatch)
    path = tmp_path / "model.pte"
    path.write_bytes(b"fake-pte")

    with pytest.raises(FileNotFoundError, match="sidecar"):
        ExecuTorchBackend(str(path))

    (tmp_path / "model.pte.json").write_text(
        json.dumps({"executorch_delegate": "vulkan"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="XNNPACK"):
        ExecuTorchBackend(str(path))
