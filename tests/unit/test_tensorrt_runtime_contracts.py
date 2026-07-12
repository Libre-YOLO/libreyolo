from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from libreyolo.backends.base import BaseBackend
from libreyolo.backends.tensorrt import TensorRTBackend, _resolve_tensorrt_device
from libreyolo.export.config import TensorRTExportConfig
from libreyolo.export.exporter import BaseExporter, TensorRTExporter
from libreyolo.export.tensorrt import (
    _calibration_cache_path,
    _create_calibrator_class,
    _profile_shape,
    _publish_tensorrt_artifacts,
    _resolve_hardware_compatibility_level,
    _select_tensorrt_device,
    _validate_batch_profile,
    _validate_dynamic_calibration_contract,
    _validate_static_calibration_contract,
    export_tensorrt,
)


pytestmark = pytest.mark.unit


def _bare_backend() -> TensorRTBackend:
    backend = TensorRTBackend.__new__(TensorRTBackend)
    backend.input_name = "images"
    backend.input_shape = (-1, 3, -1, -1)
    backend._dynamic_input = True
    backend.device = torch.device("cuda")
    return backend


def test_profile_shape_preserves_dynamic_spatial_dimensions():
    traced = (1, 3, 320, 640)

    assert _profile_shape((-1, 3, -1, -1), traced, batch=8) == (
        8,
        3,
        320,
        640,
    )


def test_same_compute_capability_uses_available_tensorrt_enum():
    expected = object()
    fake_trt = SimpleNamespace(
        HardwareCompatibilityLevel=SimpleNamespace(SAME_COMPUTE_CAPABILITY=expected)
    )

    assert (
        _resolve_hardware_compatibility_level(fake_trt, "same_compute_capability")
        is expected
    )


def test_same_compute_capability_is_unavailable_on_older_tensorrt_api():
    fake_trt = SimpleNamespace(
        HardwareCompatibilityLevel=SimpleNamespace(NONE=object())
    )

    assert (
        _resolve_hardware_compatibility_level(fake_trt, "same_compute_capability")
        is None
    )


def test_tensorrt_export_selects_device_zero_explicitly(monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"gpu-{index}")

    assert _select_tensorrt_device(0) == 0
    assert selected == [0]


@pytest.mark.parametrize("device", [True, False, 0.5, "0.5"])
def test_tensorrt_export_rejects_non_integer_devices(device):
    with pytest.raises(ValueError, match="non-negative integer"):
        _select_tensorrt_device(device)


def test_tensorrt_unified_export_builds_on_traced_cuda_device(monkeypatch, tmp_path):
    captured = {}

    def fake_export_tensorrt(**kwargs):
        captured.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(
        "libreyolo.export.tensorrt.export_tensorrt", fake_export_tensorrt
    )
    dummy = SimpleNamespace(device=torch.device("cuda:1"), shape=(1, 3, 32, 32))
    exporter = TensorRTExporter(SimpleNamespace())

    exporter._export(
        None,
        dummy,
        output_path=str(tmp_path / "model.engine"),
        precision="fp16",
        metadata={},
        calibration_data=None,
        onnx_path=str(tmp_path / "model.onnx"),
        half=True,
        int8=False,
        dynamic=False,
        verbose=False,
    )

    assert captured["device"] == 1


def test_tensorrt_unified_export_rejects_trace_build_device_mismatch(tmp_path):
    dummy = SimpleNamespace(device=torch.device("cuda:1"), shape=(1, 3, 32, 32))
    exporter = TensorRTExporter(SimpleNamespace())

    with pytest.raises(ValueError, match="trace and build devices must match"):
        exporter._export(
            None,
            dummy,
            output_path=str(tmp_path / "model.engine"),
            precision="fp16",
            metadata={},
            calibration_data=None,
            onnx_path=str(tmp_path / "model.onnx"),
            half=True,
            int8=False,
            dynamic=False,
            verbose=False,
            gpu_device=0,
        )


def test_tensorrt_config_device_is_used_for_auto_trace(monkeypatch):
    captured = {}

    def fake_base_call(self, *args, **kwargs):
        captured.update(kwargs)
        return "model.engine"

    monkeypatch.setattr(BaseExporter, "__call__", fake_base_call)
    exporter = TensorRTExporter(SimpleNamespace())

    assert (
        exporter(trt_config=TensorRTExportConfig(device=2), device="auto")
        == "model.engine"
    )
    assert captured["device"] == 2


def test_tensorrt_artifact_publication_is_atomic_and_returns_requested_path(tmp_path):
    output = tmp_path / "model.engine"

    result = _publish_tensorrt_artifacts(
        b"new-engine",
        output,
        {"precision": "fp16"},
    )

    assert result == str(output)
    assert output.read_bytes() == b"new-engine"
    assert json.loads(Path(f"{output}.json").read_text()) == {"precision": "fp16"}
    assert not list(tmp_path.glob("libreyolo-tensorrt-publish-*"))


def test_tensorrt_artifact_staging_failure_preserves_existing_files(tmp_path):
    output = tmp_path / "model.engine"
    sidecar = Path(f"{output}.json")
    output.write_bytes(b"old-engine")
    sidecar.write_text('{"old": true}')

    with pytest.raises(TypeError):
        _publish_tensorrt_artifacts(b"new-engine", output, {"bad": {object()}})

    assert output.read_bytes() == b"old-engine"
    assert sidecar.read_text() == '{"old": true}'
    assert not list(tmp_path.glob("libreyolo-tensorrt-publish-*"))


def test_tensorrt_metadata_free_publication_removes_stale_sidecar(tmp_path):
    output = tmp_path / "model.engine"
    sidecar = Path(f"{output}.json")
    output.write_bytes(b"old-engine")
    sidecar.write_text('{"stale": true}')

    _publish_tensorrt_artifacts(b"new-engine", output, None)

    assert output.read_bytes() == b"new-engine"
    assert not sidecar.exists()
    assert not list(tmp_path.glob(".*.bak"))


def test_tensorrt_stale_sidecar_removal_failure_rolls_back_engine(
    monkeypatch, tmp_path
):
    output = tmp_path / "model.engine"
    sidecar = Path(f"{output}.json")
    output.write_bytes(b"old-engine")
    sidecar.write_text('{"old": true}')
    real_unlink = Path.unlink
    failed = False

    def fail_sidecar_once(path, *args, **kwargs):
        nonlocal failed
        if path == sidecar and not failed:
            failed = True
            raise OSError("synthetic sidecar removal failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_sidecar_once)

    with pytest.raises(OSError, match="synthetic sidecar removal failure"):
        _publish_tensorrt_artifacts(b"new-engine", output, None)

    assert output.read_bytes() == b"old-engine"
    assert sidecar.read_text() == '{"old": true}'
    assert not list(tmp_path.glob(".*.bak"))


def test_tensorrt_backup_failure_cleans_prior_and_partial_backups(
    monkeypatch, tmp_path
):
    output = tmp_path / "model.engine"
    sidecar = Path(f"{output}.json")
    output.write_bytes(b"old-engine")
    sidecar.write_text('{"old": true}')
    real_link = os.link

    def fail_sidecar_link(source, destination):
        if Path(source) == sidecar:
            raise OSError("links unavailable")
        return real_link(source, destination)

    def fail_sidecar_copy(source, destination):
        Path(destination).write_bytes(b"partial-backup")
        raise PermissionError("backup denied")

    monkeypatch.setattr(os, "link", fail_sidecar_link)
    monkeypatch.setattr("libreyolo.export.tensorrt.shutil.copy2", fail_sidecar_copy)

    with pytest.raises(PermissionError, match="backup denied"):
        _publish_tensorrt_artifacts(b"new-engine", output, {"new": True})

    assert output.read_bytes() == b"old-engine"
    assert sidecar.read_text() == '{"old": true}'
    assert not list(tmp_path.glob(".*.bak"))
    assert not list(tmp_path.glob("libreyolo-tensorrt-publish-*"))


def test_tensorrt_backup_copy_preserves_restrictive_source_mode(monkeypatch, tmp_path):
    output = tmp_path / "model.engine"
    output.write_bytes(b"old-engine")
    output.chmod(0o600)
    original_mode = stat.S_IMODE(output.stat().st_mode)
    real_copy2 = shutil.copy2
    copied_modes = []

    def no_hardlinks(source, destination):
        raise OSError("hardlinks unavailable")

    def checked_copy2(source, destination):
        result = real_copy2(source, destination)
        copied_modes.append(
            (
                stat.S_IMODE(Path(source).stat().st_mode),
                stat.S_IMODE(Path(destination).stat().st_mode),
            )
        )
        return result

    monkeypatch.setattr(os, "link", no_hardlinks)
    monkeypatch.setattr("libreyolo.export.tensorrt.shutil.copy2", checked_copy2)

    _publish_tensorrt_artifacts(b"new-engine", output, None)

    assert copied_modes
    assert all(source_mode == backup_mode for source_mode, backup_mode in copied_modes)
    assert stat.S_IMODE(output.stat().st_mode) == original_mode


def test_tensorrt_artifact_promotion_failure_rolls_back(monkeypatch, tmp_path):
    output = tmp_path / "model.engine"
    sidecar = Path(f"{output}.json")
    output.write_bytes(b"old-engine")
    sidecar.write_text('{"old": true}')
    real_replace = os.replace
    failed = False

    def fail_sidecar_once(source, destination):
        nonlocal failed
        if Path(destination) == sidecar and not failed:
            failed = True
            raise OSError("synthetic sidecar promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_sidecar_once)

    with pytest.raises(OSError, match="synthetic sidecar promotion failure"):
        _publish_tensorrt_artifacts(b"new-engine", output, {"new": True})

    assert output.read_bytes() == b"old-engine"
    assert sidecar.read_text() == '{"old": true}'
    assert not list(tmp_path.glob("libreyolo-tensorrt-publish-*"))


def test_tensorrt_calibration_cache_tracks_exact_preprocessed_batches(tmp_path):
    image = tmp_path / "sample.bin"
    image.write_bytes(b"first")

    class _Loader:
        def __iter__(self):
            values = np.frombuffer(image.read_bytes(), dtype=np.uint8)
            yield values.reshape(1, 1, 1, -1)

    loader = _Loader()

    first = _calibration_cache_path(
        tmp_path / "model.engine", b"onnx", loader, enabled=True
    )
    image.write_bytes(b"second")
    second = _calibration_cache_path(
        tmp_path / "model.engine", b"onnx", loader, enabled=True
    )
    image.write_bytes(b"third-value")
    third = _calibration_cache_path(
        tmp_path / "model.engine", b"onnx", loader, enabled=True
    )

    assert first is not None
    assert len({first, second, third}) == 3
    assert (
        _calibration_cache_path(
            tmp_path / "model.engine", b"onnx", loader, enabled=False
        )
        is None
    )


def test_tensorrt_calibration_cache_disables_unstable_preprocessing(tmp_path):
    class _UnstableLoader:
        def __init__(self):
            self.value = 0

        def __iter__(self):
            self.value += 1
            yield np.array([self.value], dtype=np.float32)

    assert (
        _calibration_cache_path(
            tmp_path / "model.engine",
            b"onnx",
            _UnstableLoader(),
            enabled=True,
        )
        is None
    )


def test_tensorrt_calibrator_creates_nested_cache_parent(monkeypatch, tmp_path):
    fake_trt = SimpleNamespace(IInt8EntropyCalibrator2=object)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    loader = SimpleNamespace(batch=1)
    cache_path = tmp_path / "new" / "nested" / "model.cache"

    calibrator = _create_calibrator_class()(loader, cache_file=cache_path)
    calibrator.write_calibration_cache(b"cache")

    assert cache_path.read_bytes() == b"cache"


def test_tensorrt_calibrator_cuda_bindings_reselects_requested_device(monkeypatch):
    fake_trt = SimpleNamespace(IInt8EntropyCalibrator2=object)
    events = []
    success = 0
    cudart = SimpleNamespace(
        cudaError_t=SimpleNamespace(cudaSuccess=success),
        cudaMemcpyKind=SimpleNamespace(cudaMemcpyHostToDevice=1),
        cudaSetDevice=lambda index: events.append(("device", index)) or (success,),
        cudaMalloc=lambda size: events.append(("malloc", size)) or (success, 123),
        cudaMemcpy=lambda pointer, source, size, kind: (
            events.append(("copy", pointer, size, kind)) or (success,)
        ),
        cudaFree=lambda pointer: events.append(("free", pointer)) or (success,),
    )
    cuda_package = ModuleType("cuda")
    cuda_package.__path__ = []
    bindings_package = ModuleType("cuda.bindings")
    bindings_package.__path__ = []
    bindings_package.runtime = cudart
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setitem(sys.modules, "cuda", cuda_package)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings_package)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", cudart)
    loader = SimpleNamespace(batch=1)
    calibrator = _create_calibrator_class()(loader, device_index=2)

    assert calibrator._ensure_cuda_memory(np.zeros((1, 3, 2, 2))) == 123
    calibrator._release_cuda_memory()

    assert events[0] == ("device", 2)
    assert events[1][0] == "malloc"
    assert events[2][0] == "copy"
    assert events[-2:] == [("device", 2), ("free", 123)]


def test_tensorrt_calibrator_pycuda_uses_requested_primary_context(monkeypatch):
    fake_trt = SimpleNamespace(IInt8EntropyCalibrator2=object)
    events = []

    class _Allocation:
        def __int__(self):
            return 456

        def free(self):
            events.append("free")

    class _Context:
        def push(self):
            events.append("push")

        def pop(self):
            events.append("pop")

    context = _Context()

    class _Device:
        def __init__(self, index):
            events.append(("device", index))

        def retain_primary_context(self):
            events.append("retain")
            return context

    driver = ModuleType("pycuda.driver")
    driver.init = lambda: events.append("init")
    driver.Device = _Device
    driver.mem_alloc = lambda size: events.append(("alloc", size)) or _Allocation()
    driver.memcpy_htod = lambda allocation, batch: events.append(("copy", batch.nbytes))
    cuda_package = ModuleType("cuda")
    cuda_package.__path__ = []
    pycuda_package = ModuleType("pycuda")
    pycuda_package.__path__ = []
    pycuda_package.driver = driver
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setitem(sys.modules, "cuda", cuda_package)
    monkeypatch.setitem(sys.modules, "pycuda", pycuda_package)
    monkeypatch.setitem(sys.modules, "pycuda.driver", driver)
    monkeypatch.delitem(sys.modules, "pycuda.autoinit", raising=False)
    loader = SimpleNamespace(batch=1)
    calibrator = _create_calibrator_class()(loader, device_index=3)

    assert calibrator._ensure_cuda_memory(np.zeros((1, 3, 2, 2))) == 456
    calibrator._release_cuda_memory()

    assert events[:4] == ["init", ("device", 3), "retain", "push"]
    assert events[-3:] == ["push", "free", "pop"]
    assert "pycuda.autoinit" not in sys.modules


@pytest.mark.parametrize(
    "bounds",
    [
        (0, 1, 1),
        (2, 1, 4),
        (1, 5, 4),
    ],
)
def test_dynamic_batch_profile_bounds_are_ordered_and_positive(bounds):
    with pytest.raises(ValueError, match="1 <= min_batch"):
        _validate_batch_profile(*bounds)


def test_invalid_batch_profile_fails_before_dependency_check(monkeypatch, tmp_path):
    dependency_checked = False

    def check_dependency():
        nonlocal dependency_checked
        dependency_checked = True

    monkeypatch.setattr(
        "libreyolo.export.tensorrt.check_tensorrt_available",
        check_dependency,
    )

    with pytest.raises(ValueError, match="1 <= min_batch"):
        export_tensorrt(
            str(tmp_path / "missing.onnx"),
            str(tmp_path / "model.engine"),
            min_batch=2,
            opt_batch=1,
            max_batch=4,
        )

    assert dependency_checked is False


def test_profile_shape_requires_trace_shape_for_dynamic_non_batch_axis():
    with pytest.raises(ValueError, match="traced input_shape"):
        _profile_shape((-1, 3, -1, -1), None, batch=1)


def test_dynamic_int8_calibration_accepts_batch_one_opt_shape():
    loader = SimpleNamespace(batch=1, shape=(1, 3, 320, 640))

    _validate_dynamic_calibration_contract(
        loader,
        int8=True,
        dynamic=True,
        opt_batch=1,
        traced_shape=(4, 3, 320, 640),
    )


@pytest.mark.parametrize(
    ("loader", "opt_batch", "match"),
    [
        (SimpleNamespace(batch=2, shape=(2, 3, 320, 640)), 1, "batch-1"),
        (SimpleNamespace(batch=1, shape=(1, 3, 320, 640)), 2, "opt_batch=1"),
        (SimpleNamespace(batch=1, shape=(1, 3, 224, 224)), 1, "match the traced"),
    ],
)
def test_dynamic_int8_calibration_rejects_profile_mismatch(loader, opt_batch, match):
    with pytest.raises(ValueError, match=match):
        _validate_dynamic_calibration_contract(
            loader,
            int8=True,
            dynamic=True,
            opt_batch=opt_batch,
            traced_shape=(1, 3, 320, 640),
        )


def test_static_int8_calibration_accepts_matching_trace_and_network_shape():
    loader = SimpleNamespace(batch=4, shape=(4, 3, 320, 640))

    _validate_static_calibration_contract(
        loader,
        int8=True,
        dynamic=False,
        traced_shape=(4, 3, 320, 640),
        network_shape=(-1, 3, 320, 640),
    )


@pytest.mark.parametrize(
    ("loader", "traced_shape", "network_shape", "match"),
    [
        (
            SimpleNamespace(batch=2, shape=(1, 3, 320, 640)),
            None,
            (1, 3, 320, 640),
            "first axis equals",
        ),
        (
            SimpleNamespace(batch=1, shape=(1, 3, 224, 224)),
            (1, 3, 320, 640),
            None,
            "match the traced",
        ),
        (
            SimpleNamespace(batch=1, shape=(1, 3, 320, 640)),
            None,
            (1, 3, 640, 640),
            "parsed network input",
        ),
    ],
)
def test_static_int8_calibration_rejects_loader_contract_mismatch(
    loader, traced_shape, network_shape, match
):
    with pytest.raises(ValueError, match=match):
        _validate_static_calibration_contract(
            loader,
            int8=True,
            dynamic=False,
            traced_shape=traced_shape,
            network_shape=network_shape,
        )


def test_binding_dtype_uses_tensorrt_declared_numpy_type():
    fake_trt = SimpleNamespace(nptype=lambda dtype: np.float16)

    numpy_dtype, torch_dtype = TensorRTBackend._binding_dtypes(
        fake_trt,
        "HALF",
        tensor_name="images",
    )

    assert numpy_dtype == np.dtype(np.float16)
    assert torch_dtype == torch.float16


@pytest.mark.parametrize(
    "device",
    [0, "0", "cuda:0", torch.device("cuda:0")],
)
def test_tensorrt_runtime_resolves_explicit_device_zero(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    assert _resolve_tensorrt_device(device) == torch.device("cuda:0")


@pytest.mark.parametrize("device", [True, False])
def test_tensorrt_runtime_rejects_boolean_device(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(ValueError, match="Invalid TensorRT CUDA device"):
        _resolve_tensorrt_device(device)


def test_tensorrt_runtime_resolves_indexed_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    assert _resolve_tensorrt_device("cuda:2") == torch.device("cuda:2")


def test_tensorrt_runtime_rejects_cpu_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="requires device"):
        _resolve_tensorrt_device("cpu")


def test_tensorrt_inference_reenters_engine_cuda_device(monkeypatch):
    backend = _bare_backend()
    backend.device = torch.device("cuda:2")
    events = []

    class _DeviceContext:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, traceback):
            events.append("exit")

    monkeypatch.setattr(
        torch.cuda,
        "device",
        lambda device: events.append(device) or _DeviceContext(),
    )
    backend._infer_current_device = lambda array: "result"

    assert backend._infer(np.zeros((1, 3, 1, 1), dtype=np.float32)) == "result"
    assert events == [torch.device("cuda:2"), "enter", "exit"]


def test_dynamic_shape_rejection_is_not_ignored():
    backend = _bare_backend()
    backend.context = SimpleNamespace(set_input_shape=lambda name, shape: False)

    with pytest.raises(ValueError, match="rejected input shape"):
        backend._set_runtime_input_shape((1, 3, 320, 640))


def test_output_shape_is_read_from_context_after_input_resolution():
    backend = _bare_backend()
    backend.output_shapes = {"scores": (-1, 100, 80)}
    backend.context = SimpleNamespace(
        get_tensor_shape=lambda name: (4, 100, 80),
    )

    assert backend._runtime_output_shape("scores") == (4, 100, 80)


def test_tensorrt_fixed_spatial_profile_uses_export_canvas_for_realesrgan():
    backend = _bare_backend()
    backend.engine = SimpleNamespace(
        get_tensor_profile_shape=lambda name, profile: (
            (1, 3, 8, 8),
            (1, 3, 8, 8),
            (4, 3, 8, 8),
        )
    )
    backend.model_family = "realesrgan"
    backend.task = "restore"
    backend.imgsz = 8
    backend._dynamic_spatial = backend._detect_dynamic_spatial_profile()

    blob, _, original_size, _ = backend._preprocess(
        np.zeros((5, 7, 3), dtype=np.uint8), 8, "rgb"
    )

    assert backend._dynamic_spatial is False
    assert original_size == (7, 5)
    assert blob.shape == (1, 3, 8, 8)


def test_buffer_allocation_sets_input_shape_before_reading_outputs(monkeypatch):
    backend = _bare_backend()
    backend.input_torch_dtype = torch.float16
    backend.output_names = ["scores"]
    backend.output_torch_dtypes = {"scores": torch.int32}
    events = []

    backend._set_runtime_input_shape = lambda shape: events.append(("input", shape))

    def output_shape(name):
        events.append(("output", name))
        return (2, 10)

    backend._runtime_output_shape = output_shape
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: SimpleNamespace())

    allocations = []

    def fake_empty(size, *, dtype, device):
        allocations.append((size, dtype, device))
        return SimpleNamespace()

    monkeypatch.setattr(torch, "empty", fake_empty)

    backend._allocate_buffers((2, 3, 32, 64))

    assert events == [("input", (2, 3, 32, 64)), ("output", "scores")]
    assert allocations == [
        (2 * 3 * 32 * 64, torch.float16, torch.device("cuda")),
        (20, torch.int32, torch.device("cuda")),
    ]


def test_execute_async_false_is_a_hard_failure():
    backend = _bare_backend()
    backend.context = SimpleNamespace(execute_async_v3=lambda stream: False)
    backend.stream = SimpleNamespace(cuda_stream=123)

    with pytest.raises(RuntimeError, match="execute_async_v3 returned False"):
        backend._execute()


def test_rejected_tensor_address_is_a_hard_failure():
    backend = _bare_backend()
    backend.context = SimpleNamespace(
        set_tensor_address=lambda name, address: name != "scores"
    )

    with pytest.raises(RuntimeError, match="tensor 'scores'"):
        backend._set_tensor_address("scores", torch.empty(1))


def test_tensorrt_stream_waits_for_pytorch_input_copy(monkeypatch):
    backend = _bare_backend()
    producer = object()
    seen = []
    backend.stream = SimpleNamespace(wait_stream=lambda stream: seen.append(stream))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: producer)

    backend._wait_for_input_copy()

    assert seen == [producer]


def test_tensorrt_restore_batch_uses_shared_task_dispatch(monkeypatch):
    backend = _bare_backend()
    backend._dynamic_batch = True
    backend._max_batch = 2
    backend.task = "restore"
    captured = {}

    def shared(self, images, **kwargs):
        captured["task"] = self.task
        captured["batch"] = kwargs["batch"]
        return ["shared"]

    monkeypatch.setattr(BaseBackend, "_process_in_batches", shared)

    result = backend._process_in_batches([object()] * 3, batch=8)

    assert result == ["shared"]
    assert captured == {"task": "restore", "batch": 2}


def test_tensorrt_static_batch_uses_engine_batch_and_allows_a_short_tail(monkeypatch):
    backend = _bare_backend()
    backend._dynamic_batch = False
    backend._max_batch = 4
    backend.task = "restore"
    captured = {}

    def shared(self, images, **kwargs):
        captured["count"] = len(images)
        captured["batch"] = kwargs["batch"]
        return ["shared"]

    monkeypatch.setattr(BaseBackend, "_process_in_batches", shared)

    result = backend._process_in_batches([object()] * 5, batch=1)

    assert result == ["shared"]
    assert captured == {"count": 5, "batch": 4}


def test_tensorrt_inference_pads_below_profile_minimum_and_slices_outputs():
    backend = _bare_backend()
    backend.input_shape = (-1, 1, 1, 1)
    backend._min_batch = 2
    backend._max_batch = 4
    backend.input_numpy_dtype = np.dtype(np.float32)
    backend.input_torch_dtype = torch.float32
    backend.device = torch.device("cpu")
    backend._current_input_shape = (2, 1, 1, 1)
    backend.inputs = {"images": torch.empty(2)}
    backend.output_names = ["scores"]
    backend.outputs = {"scores": torch.tensor([[10.0], [20.0]])}
    backend._runtime_output_shapes = {"scores": (2, 1)}
    backend.context = SimpleNamespace(set_tensor_address=lambda name, address: None)
    backend.stream = SimpleNamespace(synchronize=lambda: None)
    backend._wait_for_input_copy = lambda: None
    backend._execute = lambda: None

    outputs = backend._infer_current_device(np.ones((1, 1, 1, 1), dtype=np.float32))

    assert backend.inputs["images"].tolist() == [1.0, 1.0]
    assert outputs["scores"].shape == (1, 1)
    assert outputs["scores"].tolist() == [[10.0]]
