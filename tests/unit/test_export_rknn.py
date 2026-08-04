
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch

from libreyolo.export import rknn as rknn_module
from libreyolo.export.exporter import BaseExporter, RknnExporter

pytestmark = pytest.mark.unit


class _FakeRKNN:
    instances: ClassVar[list[_FakeRKNN]] = []
    failures: ClassVar[dict[str, int]] = {}

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.calls = []
        self.calibration_arrays = []
        self.released = False
        self.__class__.instances.append(self)

    def _status(self, name):
        return self.__class__.failures.get(name, 0)

    def config(self, **kwargs):
        self.calls.append(("config", kwargs))
        return self._status("config")

    def load_onnx(self, **kwargs):
        self.calls.append(("load_onnx", kwargs))
        return self._status("load_onnx")

    def build(self, **kwargs):
        self.calls.append(("build", kwargs))
        dataset = kwargs.get("dataset")
        if dataset:
            for entry in Path(dataset).read_text(encoding="utf-8").splitlines():
                self.calibration_arrays.append(np.load(entry))
        return self._status("build")

    def export_rknn(self, output_path):
        self.calls.append(("export_rknn", output_path))
        if self._status("export_rknn") == 0:
            Path(output_path).write_bytes(b"rknn")
        return self._status("export_rknn")

    def load_rknn(self, model_path):
        self.calls.append(("load_rknn", model_path))
        return self._status("load_rknn")

    def init_runtime(self):
        self.calls.append(("init_runtime", {}))
        return self._status("init_runtime")

    def inference(self, *, inputs, data_format=None):
        self.calls.append(("inference", {"inputs": inputs, "data_format": data_format}))
        return [np.asarray(inputs[0]) + 1]

    def release(self):
        self.released = True
        self.calls.append(("release", {}))


class _FakeModel:
    def __init__(self, family="yolo9", size="t", task="detect"):
        self.family = family
        self.size = size
        self.task = task

    def _get_model_name(self):
        return self.family


@pytest.fixture(autouse=True)
def _reset_fake():
    _FakeRKNN.instances.clear()
    _FakeRKNN.failures.clear()


def test_rknn_exporter_is_registered():
    assert BaseExporter._registry["rknn"] is RknnExporter


def test_rknn_exporter_rejects_dynamic_and_batch_before_sdk_lookup():
    exporter = RknnExporter(object())

    with pytest.raises(ValueError, match="static input shapes"):
        exporter(dynamic=True)
    with pytest.raises(NotImplementedError, match="batch=1"):
        exporter(batch=2)


def test_rknn_exporter_uses_opset_19_when_cli_passes_none(monkeypatch):
    monkeypatch.setattr(
        BaseExporter,
        "__call__",
        lambda self, *args, **kwargs: kwargs,
    )

    kwargs = RknnExporter(_FakeModel())(opset=None)

    assert kwargs["opset"] == 19
    assert kwargs["dynamic"] is False
    assert kwargs["batch"] == 1
    assert kwargs["imgsz"] == 640


def test_rknn_exporter_rejects_untested_opset_and_imgsz():
    exporter = RknnExporter(_FakeModel())

    with pytest.raises(NotImplementedError, match="opset 19"):
        exporter(opset=18)
    with pytest.raises(NotImplementedError, match="imgsz=640"):
        exporter(imgsz=1280)

    picodet = RknnExporter(_FakeModel("picodet", "s"))
    with pytest.raises(NotImplementedError, match="imgsz=320"):
        picodet(imgsz=(640, 640))


def test_rknn_exporter_blocks_unvalidated_int8():
    with pytest.raises(NotImplementedError, match="RKNN INT8 export is not supported"):
        RknnExporter(object())._validate(False, True, "coco.yaml")


@pytest.mark.parametrize(
    ("family", "size"),
    [
        ("yolo9", "t"),
        ("yolo9_e2e", "t"),
        ("yolonas", "s"),
        ("picodet", "s"),
    ],
)
def test_rknn_validated_model_allowlist(family, size):
    rknn_module.validate_rknn_export_request(
        model_family=family,
        model_size=size,
        task="detect",
        target_platform="rk3588",
    )


def test_rknn_allowlist_rejects_compile_only_models_and_unchecked_targets():
    with pytest.raises(NotImplementedError, match="Compile-only"):
        rknn_module.validate_rknn_export_request(
            model_family="rfdetr",
            model_size="n",
            task="detect",
            target_platform="rk3588",
        )
    with pytest.raises(NotImplementedError, match="target 'rk3576'"):
        rknn_module.validate_rknn_export_request(
            model_family="yolo9",
            model_size="t",
            task="detect",
            target_platform="rk3576",
        )


def test_rknn_preflight_checks_scope_before_vendor_sdk(monkeypatch):
    sdk_checked = False

    def fake_check():
        nonlocal sdk_checked
        sdk_checked = True

    monkeypatch.setattr(rknn_module, "check_rknn_available", fake_check)
    with pytest.raises(NotImplementedError, match="rfdetr-n/detect"):
        RknnExporter(_FakeModel("rfdetr", "n"))._preflight(
            half=False,
            int8=False,
            data=None,
            name="rk3588",
        )
    assert sdk_checked is False

    with pytest.warns(RuntimeWarning, match="experimental"):
        RknnExporter(_FakeModel())._preflight(
            half=False,
            int8=False,
            data=None,
            name="rk3588",
        )
    assert sdk_checked is True


def test_resolve_rknn_target_aliases():
    assert rknn_module.resolve_rknn_target() == "rk3588"
    assert rknn_module.resolve_rknn_target(name="RK3576") == "rk3576"
    assert (
        rknn_module.resolve_rknn_target(name="rk3588", target="RK3588")
        == "rk3588"
    )
    with pytest.raises(ValueError, match="Conflicting RKNN targets"):
        rknn_module.resolve_rknn_target(name="rk3588", target="rk3576")


def test_rknn_exporter_verify_writes_parity_report(monkeypatch, tmp_path):
    captured = {}

    def fake_export_with_simulator(**kwargs):
        captured["export"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"rknn")
        Path(f"{kwargs['output_path']}.metadata.json").write_text(
            "{}\n", encoding="utf-8"
        )
        captured["verify_input"] = kwargs["simulator_inputs"]
        return kwargs["output_path"], [np.zeros((1, 2), dtype=np.float32)]

    monkeypatch.setattr(
        rknn_module, "export_rknn_with_simulator", fake_export_with_simulator
    )
    monkeypatch.setattr(
        rknn_module,
        "_run_onnx_reference",
        lambda onnx_path, input_data: [np.zeros((1, 2), dtype=np.float32)],
    )
    output_path = tmp_path / "model.rknn"
    onnx_path = tmp_path / "model.onnx"

    result = RknnExporter(object())._export(
        None,
        torch.zeros((1, 3, 2, 2)),
        output_path=str(output_path),
        metadata={"model_family": "yolo9"},
        calibration_data=None,
        onnx_path=str(onnx_path),
        int8=False,
        opset=19,
        verbose=False,
        name="RK3588",
        verify=True,
    )

    assert result == str(output_path)
    assert captured["export"]["target_platform"] == "rk3588"
    assert captured["export"]["metadata"]["onnx_opset"] == 19
    verify_input = captured["verify_input"]
    assert verify_input.shape == (1, 3, 2, 2)
    assert np.all((verify_input >= 0.0) & (verify_input <= 1.0))
    report = json.loads(
        Path(f"{output_path}.parity.json").read_text(encoding="utf-8")
    )
    assert report["target"] == "rk3588"
    assert report["passed"] is True
    assert report["outputs"][0]["within_tolerance"] is True


def test_rknn_exporter_cleanup_failure_keeps_success(
    monkeypatch, tmp_path, caplog
):
    def fake_export_with_simulator(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"new-rknn")
        Path(f"{kwargs['output_path']}.metadata.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return kwargs["output_path"], [np.zeros((1, 2), dtype=np.float32)]

    monkeypatch.setattr(
        rknn_module, "export_rknn_with_simulator", fake_export_with_simulator
    )
    monkeypatch.setattr(
        rknn_module,
        "_run_onnx_reference",
        lambda onnx_path, input_data: [np.zeros((1, 2), dtype=np.float32)],
    )
    output_path = tmp_path / "model.rknn"
    failed_report = Path(f"{output_path}.failed.parity.json")
    failed_report.write_text("stale failure\n", encoding="utf-8")

    original_unlink = Path.unlink

    def fail_stale_backup_cleanup(self, *args, **kwargs):
        if (
            ".failed.parity.json.backup." in self.name
            and self.is_file()
            and self.read_text(encoding="utf-8") == "stale failure\n"
        ):
            raise OSError("injected cleanup failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_stale_backup_cleanup)
    with caplog.at_level("WARNING", logger="libreyolo.export.rknn"):
        result = RknnExporter(object())._export(
            None,
            torch.zeros((1, 3, 2, 2)),
            output_path=str(output_path),
            metadata={"model_family": "yolo9"},
            calibration_data=None,
            onnx_path=str(tmp_path / "model.onnx"),
            int8=False,
            opset=19,
            verbose=False,
            verify=True,
        )

    assert result == str(output_path)
    assert output_path.read_bytes() == b"new-rknn"
    assert Path(f"{output_path}.metadata.json").is_file()
    assert Path(f"{output_path}.parity.json").is_file()
    assert not failed_report.exists()
    assert "Could not remove RKNN publication backup" in caplog.text


def test_rknn_exporter_preserves_failed_parity_report(monkeypatch, tmp_path):
    def fake_export_with_simulator(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"rknn")
        Path(f"{kwargs['output_path']}.metadata.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return kwargs["output_path"], [np.ones((1, 2), dtype=np.float32)]

    monkeypatch.setattr(
        rknn_module, "export_rknn_with_simulator", fake_export_with_simulator
    )
    monkeypatch.setattr(
        rknn_module,
        "_run_onnx_reference",
        lambda onnx_path, input_data: [np.zeros((1, 2), dtype=np.float32)],
    )
    output_path = tmp_path / "model.rknn"

    with pytest.raises(AssertionError, match="Metrics:"):
        RknnExporter(object())._export(
            None,
            torch.zeros((1, 3, 2, 2)),
            output_path=str(output_path),
            metadata={"model_family": "yolo9"},
            calibration_data=None,
            onnx_path=str(tmp_path / "model.onnx"),
            int8=False,
            opset=19,
            verbose=False,
            verify=True,
        )

    report = json.loads(
        Path(f"{output_path}.failed.parity.json").read_text(encoding="utf-8")
    )
    assert report["passed"] is False
    assert report["outputs"][0]["within_tolerance"] is False
    assert not output_path.exists()
    assert not Path(f"{output_path}.metadata.json").exists()


def test_rknn_exporter_reference_failure_preserves_prior_artifacts(
    monkeypatch, tmp_path
):
    def fake_export_with_simulator(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"rknn")
        Path(f"{kwargs['output_path']}.metadata.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return kwargs["output_path"], [np.zeros((1, 2), dtype=np.float32)]

    monkeypatch.setattr(
        rknn_module, "export_rknn_with_simulator", fake_export_with_simulator
    )

    def fail_reference(onnx_path, input_data):
        raise RuntimeError("ONNX Runtime failed")

    monkeypatch.setattr(rknn_module, "_run_onnx_reference", fail_reference)
    output_path = tmp_path / "model.rknn"
    output_path.write_bytes(b"previous-rknn")
    metadata_path = Path(f"{output_path}.metadata.json")
    metadata_path.write_text('{"previous": true}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="ONNX Runtime failed"):
        RknnExporter(object())._export(
            None,
            torch.zeros((1, 3, 2, 2)),
            output_path=str(output_path),
            metadata={"model_family": "yolo9"},
            calibration_data=None,
            onnx_path=str(tmp_path / "model.onnx"),
            int8=False,
            opset=19,
            verbose=False,
            verify=True,
        )

    assert output_path.read_bytes() == b"previous-rknn"
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {
        "previous": True
    }
    assert not Path(f"{output_path}.parity.json").exists()


def test_export_rknn_float_writes_artifact_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(rknn_module, "_load_rknn_class", lambda: _FakeRKNN)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    output_path = tmp_path / "model.rknn"

    result = rknn_module.export_rknn(
        onnx_path=str(onnx_path),
        output_path=str(output_path),
        target_platform="RK3588",
        metadata={"model_family": "yolo9", "dynamic": False},
        config={"optimization_level": 2},
        build={"auto_hybrid_cos_thresh": 0.99},
    )

    assert result == str(output_path)
    assert output_path.read_bytes() == b"rknn"
    metadata = json.loads(
        Path(f"{output_path}.metadata.json").read_text(encoding="utf-8")
    )
    assert metadata == {"dynamic": False, "model_family": "yolo9"}

    instance = _FakeRKNN.instances[-1]
    assert instance.calls[0] == (
        "config",
        {"target_platform": "rk3588", "optimization_level": 2},
    )
    assert instance.calls[2] == (
        "build",
        {"do_quantization": False, "auto_hybrid_cos_thresh": 0.99},
    )
    assert instance.released is True


def test_export_rknn_metadata_failure_preserves_prior_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(rknn_module, "_load_rknn_class", lambda: _FakeRKNN)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    output_path = tmp_path / "model.rknn"
    metadata_path = Path(f"{output_path}.metadata.json")
    output_path.write_bytes(b"previous-model")
    metadata_path.write_text("previous-metadata\n", encoding="utf-8")

    def fail_metadata_write(output_path, metadata):
        raise OSError("injected metadata failure")

    monkeypatch.setattr(rknn_module, "_write_metadata_sidecar", fail_metadata_write)
    with pytest.raises(OSError, match="injected metadata failure"):
        rknn_module.export_rknn(
            onnx_path=str(onnx_path),
            output_path=str(output_path),
            metadata={"model_family": "yolo9"},
        )

    assert output_path.read_bytes() == b"previous-model"
    assert metadata_path.read_text(encoding="utf-8") == "previous-metadata\n"


def test_unverified_export_removes_stale_sidecars(monkeypatch, tmp_path):
    monkeypatch.setattr(rknn_module, "_load_rknn_class", lambda: _FakeRKNN)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    output_path = tmp_path / "model.rknn"
    output_path.write_bytes(b"previous-model")
    stale_sidecars = [
        Path(f"{output_path}.metadata.json"),
        Path(f"{output_path}.parity.json"),
        Path(f"{output_path}.failed.parity.json"),
    ]
    for sidecar in stale_sidecars:
        sidecar.write_text("stale\n", encoding="utf-8")

    result = rknn_module.export_rknn(
        onnx_path=str(onnx_path),
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.read_bytes() == b"rknn"
    assert all(not sidecar.exists() for sidecar in stale_sidecars)


def test_rknn_bundle_publication_rolls_back_sidecar_failure(monkeypatch, tmp_path):
    destination = tmp_path / "model.rknn"
    destination_metadata = Path(f"{destination}.metadata.json")
    destination_report = Path(f"{destination}.parity.json")
    destination.write_bytes(b"previous-model")
    destination_metadata.write_text("previous-metadata\n", encoding="utf-8")
    destination_report.write_text("previous-report\n", encoding="utf-8")

    staged_model = tmp_path / "staged.rknn"
    staged_metadata = Path(f"{staged_model}.metadata.json")
    staged_report = Path(f"{staged_model}.parity.json")
    staged_model.write_bytes(b"new-model")
    staged_metadata.write_text("new-metadata\n", encoding="utf-8")
    staged_report.write_text("new-report\n", encoding="utf-8")

    original_replace = Path.replace

    def fail_metadata_replace(self, target):
        if self == staged_metadata and Path(target) == destination_metadata:
            raise OSError("injected sidecar failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_metadata_replace)
    with pytest.raises(OSError, match="injected sidecar failure"):
        rknn_module._publish_rknn_artifacts(
            staged_model=staged_model,
            staged_metadata=staged_metadata,
            staged_report=staged_report,
            destination=destination,
        )

    assert destination.read_bytes() == b"previous-model"
    assert destination_metadata.read_text(encoding="utf-8") == "previous-metadata\n"
    assert destination_report.read_text(encoding="utf-8") == "previous-report\n"


def test_export_rknn_int8_materializes_nchw_calibration(monkeypatch, tmp_path):
    monkeypatch.setattr(rknn_module, "_load_rknn_class", lambda: _FakeRKNN)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    calibration = [np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)]

    rknn_module.export_rknn(
        onnx_path=str(onnx_path),
        output_path=str(tmp_path / "model_int8.rknn"),
        int8=True,
        calibration_data=calibration,
    )

    instance = _FakeRKNN.instances[-1]
    assert len(instance.calibration_arrays) == 1
    np.testing.assert_array_equal(instance.calibration_arrays[0], calibration[0])
    build_call = next(call for call in instance.calls if call[0] == "build")
    assert build_call[1]["do_quantization"] is True
    assert build_call[1]["dataset"].endswith("dataset.txt")


def test_export_rknn_releases_sdk_after_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(rknn_module, "_load_rknn_class", lambda: _FakeRKNN)
    _FakeRKNN.failures["build"] = 7
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    with pytest.raises(RuntimeError, match="build failed with status 7"):
        rknn_module.export_rknn(
            onnx_path=str(onnx_path),
            output_path=str(tmp_path / "model.rknn"),
        )

    assert _FakeRKNN.instances[-1].released is True


def test_simulator_failure_removes_exported_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(rknn_module, "_load_rknn_class", lambda: _FakeRKNN)
    _FakeRKNN.failures["init_runtime"] = 9
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    output_path = tmp_path / "model.rknn"

    with pytest.raises(RuntimeError, match="simulator initialization failed"):
        rknn_module.export_rknn_with_simulator(
            onnx_path=str(onnx_path),
            output_path=str(output_path),
            simulator_inputs=np.zeros((1, 3, 2, 2), dtype=np.float32),
            metadata={"model_family": "yolo9"},
        )

    assert _FakeRKNN.instances[-1].released is True
    assert not output_path.exists()
    assert not Path(f"{output_path}.metadata.json").exists()


def test_run_rknn_simulator_uses_board_free_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(rknn_module, "_load_rknn_class", lambda: _FakeRKNN)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    input_tensor = np.zeros((1, 3, 2, 2), dtype=np.float32)

    outputs = rknn_module.run_rknn_simulator(str(onnx_path), input_tensor)

    np.testing.assert_array_equal(outputs[0], input_tensor + 1)
    instance = _FakeRKNN.instances[-1]
    assert any(call[0] == "load_onnx" for call in instance.calls)
    assert not any(call[0] == "load_rknn" for call in instance.calls)
    assert ("init_runtime", {}) in instance.calls
    inference_call = next(call for call in instance.calls if call[0] == "inference")
    assert inference_call[1]["data_format"] == "nchw"
    assert instance.released is True


def test_compare_rknn_outputs_reports_metrics_and_failures():
    reference = [np.array([1.0, 2.0], dtype=np.float32)]
    close = [np.array([1.00001, 2.00001], dtype=np.float32)]

    metrics = rknn_module.compare_rknn_outputs(
        reference, close, rtol=1e-3, atol=1e-4
    )
    assert metrics[0]["within_tolerance"] is True
    assert metrics[0]["max_abs_error"] > 0
    assert metrics[0]["cosine_similarity"] > 0.999
    assert metrics[0]["normalized_rmse"] < 1e-3

    failed_metrics = rknn_module.compare_rknn_outputs(
        reference,
        [np.array([5.0, 6.0], dtype=np.float32)],
        rtol=1e-6,
        atol=1e-6,
        raise_on_failure=False,
    )
    assert failed_metrics[0]["within_tolerance"] is False

    with pytest.raises(AssertionError, match="parity failed"):
        rknn_module.compare_rknn_outputs(
            reference,
            [np.array([5.0, 6.0], dtype=np.float32)],
            rtol=1e-6,
            atol=1e-6,
        )


def test_rknn_scale_independent_acceptance_retains_strict_result():
    passed, evaluated = rknn_module.evaluate_rknn_metrics(
        [
            {
                "index": 0,
                "within_tolerance": False,
                "cosine_similarity": 0.99999,
                "normalized_rmse": 0.005,
            }
        ]
    )
    assert passed is True
    assert evaluated[0]["within_tolerance"] is False
    assert evaluated[0]["scale_independent_pass"] is True
    assert evaluated[0]["accepted"] is True

    passed, evaluated = rknn_module.evaluate_rknn_metrics(
        [
            {
                "index": 0,
                "within_tolerance": False,
                "cosine_similarity": 0.9997,
                "normalized_rmse": 0.023,
            }
        ]
    )
    assert passed is False
    assert evaluated[0]["accepted"] is False


def test_verify_rknn_parity_can_return_failed_diagnostics(monkeypatch):
    monkeypatch.setattr(
        rknn_module,
        "_run_onnx_reference",
        lambda onnx_path, input_data: [np.zeros((1, 2), dtype=np.float32)],
    )
    monkeypatch.setattr(
        rknn_module,
        "run_rknn_simulator",
        lambda onnx_path, input_data, **kwargs: [
            np.ones((1, 2), dtype=np.float32)
        ],
    )

    metrics = rknn_module.verify_rknn_simulator_parity(
        "model.onnx",
        np.zeros((1, 3, 2, 2), dtype=np.float32),
        raise_on_failure=False,
    )
    assert metrics[0]["within_tolerance"] is False


def test_rknn_dependency_error_points_windows_to_wsl(monkeypatch):
    monkeypatch.setattr(rknn_module.sys, "platform", "win32")
    with pytest.raises(ImportError, match="WSL2"):
        rknn_module.check_rknn_available()


def test_rknn_dependency_error_rejects_linux_arm64(monkeypatch):
    monkeypatch.setattr(rknn_module.sys, "platform", "linux")
    monkeypatch.setattr(rknn_module.platform, "machine", lambda: "aarch64")
    with pytest.raises(ImportError, match="requires Linux x86_64"):
        rknn_module.check_rknn_available()
