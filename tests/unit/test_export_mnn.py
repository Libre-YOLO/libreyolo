"""Unit tests for the clean-room MNN exporter."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

import libreyolo.export.mnn as mnn_export
from libreyolo.export.exporter import BaseExporter, MNNExporter
from libreyolo.export.support import EXPORT_FORMATS, get_support
from libreyolo.models.registry import families_in

pytestmark = pytest.mark.unit


def _wrapper(family: str = "yolo9") -> MagicMock:
    wrapper = MagicMock()
    wrapper._get_model_name.return_value = family
    wrapper._get_input_size.return_value = 64
    wrapper.task = "detect"
    wrapper.SUPPORTED_TASKS = ("detect",)
    wrapper.DEFAULT_TASK = "detect"
    wrapper.size = "t" if family == "yolo9" else "n"
    wrapper.nb_classes = 2
    wrapper.names = {0: "a", 1: "b"}
    wrapper.device = torch.device("cpu")
    return wrapper


def test_mnn_registry_and_support_contract():
    assert "mnn" in EXPORT_FORMATS
    assert BaseExporter._registry["mnn"] is MNNExporter
    assert MNNExporter.suffix == ".mnn"
    assert MNNExporter.requires_onnx is True
    assert MNNExporter.supports_fp16 is False
    assert MNNExporter.supports_int8 is False
    assert get_support("yolo9", "detect", "mnn").tier == "validated"
    assert get_support("rfdetr", "detect", "mnn").tier == "validated"
    assert get_support("deimv2", "detect", "mnn").tier == "experimental"
    assert get_support("yolox", "detect", "mnn").tier == "blocked"


def test_mnn_covers_every_g0_g1_detection_family():
    expected = set(families_in("g0") + families_in("g1"))
    covered = {
        family
        for family in expected
        if get_support(family, "detect", "mnn").tier != "blocked"
    }

    assert covered == expected
    assert get_support("ec", "pose", "mnn").tier == "blocked"
    assert get_support("rfdetr", "segment", "mnn").tier == "blocked"


def test_mnn_rejects_dynamic_before_base_export(monkeypatch):
    base_call = MagicMock(side_effect=AssertionError("base export entered"))
    monkeypatch.setattr(BaseExporter, "__call__", base_call)

    with pytest.raises(ValueError, match="dynamic=False"):
        MNNExporter(_wrapper())(dynamic=True)

    base_call.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"half": True}, NotImplementedError, "MNN FP16"),
        ({"int8": True}, NotImplementedError, "MNN INT8"),
        ({"nms": True}, NotImplementedError, "MNN embedded NMS"),
    ],
)
def test_mnn_rejects_unsupported_precision_and_nms(monkeypatch, kwargs, error, message):
    monkeypatch.setattr(mnn_export, "check_mnn_available", lambda: Path("tool"))
    with pytest.raises(error, match=message):
        MNNExporter(_wrapper())(**kwargs)


def test_mnn_metadata_is_fixed_fp32_contract():
    metadata = MNNExporter(_wrapper())._build_metadata(
        "fp32", True, None, imgsz=(64, 64)
    )
    assert metadata["precision"] == "fp32"
    assert metadata["dynamic"] is False
    assert metadata["model_family"] == "yolo9"
    assert metadata["task"] == "detect"


def test_mnn_preflight_reports_optional_dependency_install_hint(monkeypatch):
    def missing_dependency():
        raise ImportError("Install with: pip install libreyolo[mnn]")

    monkeypatch.setattr(mnn_export, "check_mnn_available", missing_dependency)
    with pytest.raises(ImportError, match=r"pip install libreyolo\[mnn\]"):
        MNNExporter(_wrapper())._preflight(half=False, int8=False, data=None)


def test_export_mnn_invokes_converter_and_writes_ordered_sidecar(tmp_path, monkeypatch):
    source = tmp_path / "intermediate.onnx"
    source.write_bytes(b"onnx")
    converter = tmp_path / "mnnconvert.exe"
    converter.write_bytes(b"tool")
    destination = tmp_path / "model.mnn"
    calls = []

    monkeypatch.setattr(mnn_export, "check_mnn_available", lambda: converter)
    monkeypatch.setattr(
        mnn_export,
        "_onnx_io_contract",
        lambda path: (["images"], ["boxes", "logits"], [2, 3, 64, 64]),
    )
    monkeypatch.setattr(
        mnn_export.importlib.metadata,
        "version",
        lambda package: "3.6.1",
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--MNNModel") + 1])
        output.write_bytes(b"mnn-model")
        return subprocess.CompletedProcess(command, 0, "converted", "")

    monkeypatch.setattr(mnn_export.subprocess, "run", fake_run)
    result = mnn_export.export_mnn(
        str(source),
        str(destination),
        metadata={"precision": "fp32", "model_family": "yolo9"},
        batch=2,
        verbose=True,
    )

    assert result == str(destination)
    assert destination.read_bytes() == b"mnn-model"
    sidecar = json.loads(Path(f"{destination}.json").read_text(encoding="utf-8"))
    assert sidecar["format"] == "mnn"
    assert sidecar["dynamic"] is False
    assert sidecar["mnn_version"] == "3.6.1"
    assert sidecar["mnn_backend"] == "cpu"
    assert sidecar["mnn_input_names"] == ["images"]
    assert sidecar["mnn_output_names"] == ["boxes", "logits"]
    assert sidecar["mnn_input_shape"] == [2, 3, 64, 64]
    assert sidecar["mnn_batch"] == 2

    command, run_kwargs = calls[0]
    assert command == [
        str(converter),
        "-f",
        "ONNX",
        "--modelFile",
        str(source),
        "--MNNModel",
        command[6],
        "--batch",
        "2",
        "--bizCode",
        "LibreYOLO",
    ]
    assert run_kwargs["capture_output"] is True
    assert run_kwargs["text"] is True
    assert run_kwargs["check"] is False
    assert run_kwargs["env"]["PATH"].split(mnn_export.os.pathsep)[0] == str(
        converter.parent
    )


def test_export_mnn_failure_preserves_previous_pair(tmp_path, monkeypatch):
    source = tmp_path / "intermediate.onnx"
    source.write_bytes(b"onnx")
    converter = tmp_path / "mnnconvert.exe"
    converter.write_bytes(b"tool")
    destination = tmp_path / "model.mnn"
    sidecar = Path(f"{destination}.json")
    destination.write_bytes(b"old-model")
    sidecar.write_text('{"old": true}', encoding="utf-8")

    monkeypatch.setattr(mnn_export, "check_mnn_available", lambda: converter)
    monkeypatch.setattr(
        mnn_export,
        "_onnx_io_contract",
        lambda path: (["images"], ["output"], [1, 3, 64, 64]),
    )
    monkeypatch.setattr(
        mnn_export.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 9, "converter stdout", "converter stderr"
        ),
    )

    with pytest.raises(RuntimeError, match="exit code 9") as raised:
        mnn_export.export_mnn(
            str(source),
            str(destination),
            metadata={"precision": "fp32"},
            batch=1,
        )

    assert "converter stdout" in str(raised.value)
    assert "converter stderr" in str(raised.value)
    assert destination.read_bytes() == b"old-model"
    assert sidecar.read_text(encoding="utf-8") == '{"old": true}'


@pytest.mark.parametrize(
    "returncode",
    [3221225477, -1073741819, 3221226505, -1073740791],
)
def test_windows_teardown_failure_requires_independent_artifact_validation(
    tmp_path, monkeypatch, returncode
):
    source = tmp_path / "intermediate.onnx"
    source.write_bytes(b"onnx")
    converter = tmp_path / "mnnconvert.exe"
    converter.write_bytes(b"tool")
    destination = tmp_path / "model.mnn"
    validated = []

    monkeypatch.setattr(mnn_export, "check_mnn_available", lambda: converter)
    monkeypatch.setattr(
        mnn_export,
        "_onnx_io_contract",
        lambda path: (["images"], ["output"], [1, 3, 64, 64]),
    )
    monkeypatch.setattr(
        mnn_export.importlib.metadata,
        "version",
        lambda package: "3.6.1",
    )
    monkeypatch.setattr(mnn_export.os, "name", "nt")

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--MNNModel") + 1])
        output.write_bytes(b"mnn-model")
        return subprocess.CompletedProcess(
            command, returncode, "Converted Success!", ""
        )

    monkeypatch.setattr(mnn_export.subprocess, "run", fake_run)
    monkeypatch.setattr(
        mnn_export,
        "_validate_mnn_artifact",
        lambda path, inputs, outputs: validated.append((path, inputs, outputs)),
    )

    result = mnn_export.export_mnn(
        str(source),
        str(destination),
        metadata={"precision": "fp32"},
        batch=1,
    )

    assert result == str(destination)
    assert destination.read_bytes() == b"mnn-model"
    assert len(validated) == 1
    assert validated[0][1:] == (["images"], ["output"])


def test_artifact_pair_restores_previous_files_on_commit_failure(tmp_path, monkeypatch):
    destination = tmp_path / "model.mnn"
    sidecar = Path(f"{destination}.json")
    staged_model = tmp_path / "new.mnn"
    staged_sidecar = tmp_path / "new.mnn.json"
    destination.write_bytes(b"old-model")
    sidecar.write_bytes(b"old-sidecar")
    staged_model.write_bytes(b"new-model")
    staged_sidecar.write_bytes(b"new-sidecar")
    real_replace = mnn_export.os.replace
    failed = False

    def fail_once(source, target):
        nonlocal failed
        if Path(source) == staged_sidecar and Path(target) == sidecar and not failed:
            failed = True
            raise OSError("simulated sidecar failure")
        return real_replace(source, target)

    monkeypatch.setattr(mnn_export.os, "replace", fail_once)
    with pytest.raises(OSError, match="simulated"):
        mnn_export._commit_artifact_pair(
            staged_model, staged_sidecar, destination, sidecar
        )

    assert destination.read_bytes() == b"old-model"
    assert sidecar.read_bytes() == b"old-sidecar"
