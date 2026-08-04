from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from libreyolo.backends.base import BaseBackend
from libreyolo.backends.metadata import ExportMetadataError
from libreyolo.backends.triton import (
    TRITON_METADATA_PARAMETER,
    TritonBackend,
    TritonBackendError,
    create_triton_config,
    parse_triton_model_url,
)

pytestmark = [pytest.mark.unit, pytest.mark.triton]


def _runtime_metadata(*, family="yolo9", size="t", task="detect", **updates):
    metadata = {
        "schema_version": "1.0",
        "libreyolo_version": "1.4.0",
        "model_family": family,
        "size": size,
        "model_size": size,
        "task": task,
        "supported_tasks": [task],
        "default_task": task,
        "nc": 2,
        "names": {"0": "cat", "1": "dog"},
        "imgsz": 8,
        "imgsz_h": 8,
        "imgsz_w": 8,
        "nms": "false",
    }
    metadata.update(updates)
    return metadata


def _model_responses(*, runtime_metadata=None, max_batch_size=8, dtype="FP32"):
    runtime_metadata = (
        _runtime_metadata() if runtime_metadata is None else runtime_metadata
    )
    config = {
        "name": "detector",
        "max_batch_size": max_batch_size,
        "input": [{"name": "images", "data_type": f"TYPE_{dtype}", "dims": [3, 8, 8]}],
        "output": [
            {"name": "scores", "data_type": "TYPE_FP32", "dims": [-1, 2]},
            {"name": "boxes", "data_type": "TYPE_FP32", "dims": [-1, 4]},
        ],
        "parameters": {
            TRITON_METADATA_PARAMETER: {
                "string_value": json.dumps(runtime_metadata),
            }
        },
    }
    metadata = {
        "name": "detector",
        "versions": ["1"],
        "inputs": [{"name": "images", "datatype": dtype, "shape": [-1, 3, 8, 8]}],
        # Deliberately opposite to config order. Model metadata preserves the
        # graph output order required by positional backend postprocessors.
        "outputs": [
            {"name": "boxes", "datatype": "FP32", "shape": [-1, -1, 4]},
            {"name": "scores", "datatype": "FP32", "shape": [-1, -1, 2]},
        ],
    }
    return config, metadata


class _FakeInferInput:
    instances = []

    def __init__(self, name, shape, datatype):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.array = None
        self.binary_data = None
        self.__class__.instances.append(self)

    def set_data_from_numpy(self, array, binary_data=True):
        self.array = array
        self.binary_data = binary_data


class _FakeRequestedOutput:
    def __init__(self, name, binary_data=True):
        self.name = name
        self.binary_data = binary_data


class _FakeResult:
    def __init__(self, outputs):
        self.outputs = outputs

    def as_numpy(self, name):
        return self.outputs.get(name)


def _install_fake_client(
    monkeypatch,
    *,
    config=None,
    metadata=None,
    server_ready=True,
    model_ready=True,
    init_error=None,
    infer_error=None,
):
    default_config, default_metadata = _model_responses()
    config = default_config if config is None else config
    metadata = default_metadata if metadata is None else metadata
    _FakeInferInput.instances = []

    class FakeClient:
        instances = []

        def __init__(self, **kwargs):
            if init_error is not None:
                raise init_error
            self.kwargs = kwargs
            self.calls = []
            self.__class__.instances.append(self)

        def is_server_ready(self):
            return server_ready

        def is_model_ready(self, model_name, model_version=""):
            self.calls.append(("ready", model_name, model_version))
            return model_ready

        def get_model_config(self, model_name, model_version=""):
            self.calls.append(("config", model_name, model_version))
            return config

        def get_model_metadata(self, model_name, model_version=""):
            self.calls.append(("metadata", model_name, model_version))
            return metadata

        def infer(self, model_name, **kwargs):
            self.calls.append(("infer", model_name, kwargs))
            if infer_error is not None:
                raise infer_error
            batch = kwargs["inputs"][0].array.shape[0]
            return _FakeResult(
                {
                    "boxes": np.full((batch, 1, 4), 2.0, dtype=np.float32),
                    "scores": np.full((batch, 1, 2), 1.0, dtype=np.float32),
                }
            )

    package = types.ModuleType("tritonclient")
    package.__path__ = []
    http = types.ModuleType("tritonclient.http")
    http.InferenceServerClient = FakeClient
    http.InferInput = _FakeInferInput
    http.InferRequestedOutput = _FakeRequestedOutput
    utils = types.ModuleType("tritonclient.utils")
    utils.triton_to_np_dtype = {
        "FP16": np.float16,
        "FP32": np.float32,
        "FP64": np.float64,
        "INT32": np.int32,
        "BYTES": np.object_,
    }.get
    package.http = http
    package.utils = utils
    monkeypatch.setitem(sys.modules, "tritonclient", package)
    monkeypatch.setitem(sys.modules, "tritonclient.http", http)
    monkeypatch.setitem(sys.modules, "tritonclient.utils", utils)
    return FakeClient


@pytest.mark.parametrize(
    ("url", "scheme", "server", "model", "version"),
    [
        ("http://127.0.0.1:8000/detector", "http", "127.0.0.1:8000", "detector", ""),
        (
            "https://example.test:8443/detector/7",
            "https",
            "example.test:8443",
            "detector",
            "7",
        ),
        ("http://[::1]:8000/model", "http", "[::1]:8000", "model", ""),
    ],
)
def test_parse_triton_model_url(url, scheme, server, model, version):
    parsed = parse_triton_model_url(url)
    assert (
        parsed.scheme,
        parsed.server_url,
        parsed.model_name,
        parsed.model_version,
    ) == (
        scheme,
        server,
        model,
        version,
    )


@pytest.mark.parametrize(
    "url",
    [
        "grpc://localhost:8001/model",
        "http://localhost/model",
        "http://localhost:8000/",
        "http://localhost:8000/model/",
        "http://localhost:8000/a/b/c",
        "http://user:pass@localhost:8000/model",
        "http://localhost:8000/model?version=1",
        "http://localhost:8000/model/latest",
        "http://localhost:8000/model/0",
        "http://localhost:8000/model%2Fother",
    ],
)
def test_parse_triton_model_url_rejects_unsupported_forms(url):
    with pytest.raises((TypeError, ValueError)):
        parse_triton_model_url(url)


def test_factory_routes_url_before_local_path_resolution(monkeypatch):
    import libreyolo.backends.triton as triton_module
    import libreyolo.models as models

    sentinel = object()
    monkeypatch.setattr(
        triton_module, "TritonBackend", lambda *args, **kwargs: sentinel
    )
    monkeypatch.setattr(
        models,
        "_resolve_weights_path",
        lambda path: pytest.fail("URL must not enter local path resolution"),
    )

    assert models.LibreYOLO("http://127.0.0.1:8000/detector") is sentinel


def test_cli_resolution_accepts_triton_url():
    from libreyolo.cli.command_utils import resolve_model_or_exit

    url = "http://127.0.0.1:8000/detector"
    assert resolve_model_or_exit(object(), url) == url


def test_triton_public_lazy_exports():
    import libreyolo

    assert libreyolo.TritonBackend is TritonBackend
    assert libreyolo.create_triton_config is create_triton_config


def test_create_triton_config_preserves_metadata_and_graph_order(monkeypatch, tmp_path):
    def value(name, dtype, dims):
        shape = types.SimpleNamespace(
            dim=[types.SimpleNamespace(dim_value=max(0, dim)) for dim in dims]
        )
        tensor_type = types.SimpleNamespace(elem_type=dtype, shape=shape)
        return types.SimpleNamespace(
            name=name,
            type=types.SimpleNamespace(tensor_type=tensor_type),
        )

    runtime_metadata = _runtime_metadata()
    model = types.SimpleNamespace(
        metadata_props=[
            types.SimpleNamespace(key=key, value=value)
            for key, value in runtime_metadata.items()
        ],
        graph=types.SimpleNamespace(
            initializer=[],
            input=[value("images", 1, [-1, 3, 8, 8])],
            output=[
                value("scores", 1, [-1, -1, 2]),
                value("boxes", 1, [-1, -1, 4]),
            ],
        ),
    )
    fake_onnx = types.ModuleType("onnx")
    fake_onnx.load = lambda *args, **kwargs: model
    fake_onnx.shape_inference = types.SimpleNamespace(infer_shapes=lambda value: value)
    fake_onnx.TensorProto = types.SimpleNamespace(
        DataType=types.SimpleNamespace(Name=lambda dtype: {1: "FLOAT"}[dtype])
    )
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.touch()
    config_path = tmp_path / "detector" / "config.pbtxt"

    written = create_triton_config(
        onnx_path,
        config_path,
        model_name="detector",
        max_batch_size=4,
    )

    config = config_path.read_text(encoding="utf-8")
    assert written == str(config_path)
    assert 'name: "detector"' in config
    assert "max_batch_size: 4" in config
    assert "dims: [ 3, 8, 8 ]" in config
    assert config.index('name: "scores"') < config.index('name: "boxes"')
    assert f'key: "{TRITON_METADATA_PARAMETER}"' in config
    assert '\\"model_family\\":\\"yolo9\\"' in config
    assert "KIND_CPU" in config


def test_create_triton_config_rejects_static_batch_for_server_batching(
    monkeypatch, tmp_path
):
    tensor_type = types.SimpleNamespace(
        elem_type=1,
        shape=types.SimpleNamespace(
            dim=[
                types.SimpleNamespace(dim_value=1),
                types.SimpleNamespace(dim_value=3),
                types.SimpleNamespace(dim_value=8),
                types.SimpleNamespace(dim_value=8),
            ]
        ),
    )
    graph_input = types.SimpleNamespace(
        name="images", type=types.SimpleNamespace(tensor_type=tensor_type)
    )
    model = types.SimpleNamespace(
        metadata_props=[
            types.SimpleNamespace(key=key, value=value)
            for key, value in _runtime_metadata().items()
        ],
        graph=types.SimpleNamespace(
            initializer=[], input=[graph_input], output=[graph_input]
        ),
    )
    fake_onnx = types.ModuleType("onnx")
    fake_onnx.load = lambda *args, **kwargs: model
    fake_onnx.shape_inference = types.SimpleNamespace(infer_shapes=lambda value: value)
    fake_onnx.TensorProto = types.SimpleNamespace(
        DataType=types.SimpleNamespace(Name=lambda dtype: "FLOAT")
    )
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.touch()

    with pytest.raises(ValueError, match="dynamic batch axis"):
        create_triton_config(
            onnx_path,
            tmp_path / "config.pbtxt",
            model_name="detector",
            max_batch_size=4,
        )


def test_triton_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "tritonclient", None)
    monkeypatch.delitem(sys.modules, "tritonclient.http", raising=False)
    monkeypatch.delitem(sys.modules, "tritonclient.utils", raising=False)

    with pytest.raises(ImportError, match=r"tritonclient\[http\]"):
        TritonBackend("http://127.0.0.1:8000/detector")


def test_triton_initializes_metadata_and_https_client(monkeypatch):
    client_cls = _install_fake_client(monkeypatch)

    backend = TritonBackend("https://example.test:8443/detector/1", timeout=4.5)

    assert backend.model_family == "yolo9"
    assert backend.model_size == "t"
    assert backend.task == "detect"
    assert backend.names == {0: "cat", 1: "dog"}
    assert backend.imgsz == 8
    assert backend.output_names == ["boxes", "scores"]
    assert client_cls.instances[0].kwargs == {
        "url": "example.test:8443",
        "ssl": True,
        "connection_timeout": 4.5,
        "network_timeout": 4.5,
    }
    assert ("ready", "detector", "1") in client_cls.instances[0].calls


@pytest.mark.parametrize(("family", "size"), [("yolo9", "t"), ("rfdetr", "n")])
def test_triton_preserves_flagship_detection_metadata(monkeypatch, family, size):
    config, metadata = _model_responses(
        runtime_metadata=_runtime_metadata(family=family, size=size)
    )
    _install_fake_client(monkeypatch, config=config, metadata=metadata)

    backend = TritonBackend("http://127.0.0.1:8000/detector")

    assert backend.FAMILY == family
    assert backend.size == size
    assert backend.task == "detect"
    assert backend.nb_classes == 2


def test_triton_casts_input_and_returns_metadata_order(monkeypatch):
    config, metadata = _model_responses(dtype="FP16")
    client_cls = _install_fake_client(monkeypatch, config=config, metadata=metadata)
    backend = TritonBackend("http://127.0.0.1:8000/detector")

    outputs = backend._run_inference(np.ones((2, 3, 8, 8), dtype=np.float32))

    infer_input = _FakeInferInput.instances[-1]
    assert infer_input.datatype == "FP16"
    assert infer_input.array.dtype == np.float16
    assert infer_input.binary_data is True
    assert [output.shape[-1] for output in outputs] == [4, 2]
    assert [float(output[0, 0, 0]) for output in outputs] == [2.0, 1.0]
    infer_call = next(
        call for call in client_cls.instances[0].calls if call[0] == "infer"
    )
    assert [output.name for output in infer_call[2]["outputs"]] == ["boxes", "scores"]
    assert infer_call[2]["timeout"] == 30_000_000


@pytest.mark.parametrize(
    ("server_ready", "model_ready", "message"),
    [(False, True, "server.*not ready"), (True, False, "model.*not ready")],
)
def test_triton_reports_readiness_failures(
    monkeypatch, server_ready, model_ready, message
):
    _install_fake_client(
        monkeypatch,
        server_ready=server_ready,
        model_ready=model_ready,
    )
    with pytest.raises(TritonBackendError, match=message):
        TritonBackend("http://127.0.0.1:8000/detector")


def test_triton_reports_connection_failure(monkeypatch):
    _install_fake_client(monkeypatch, init_error=OSError("connection refused"))
    with pytest.raises(TritonBackendError, match="connection refused"):
        TritonBackend("http://127.0.0.1:8000/detector")


def test_triton_reports_inference_failure(monkeypatch):
    _install_fake_client(monkeypatch, infer_error=TimeoutError("timed out"))
    backend = TritonBackend("http://127.0.0.1:8000/detector")

    with pytest.raises(TritonBackendError, match="inference failed.*timed out"):
        backend._run_inference(np.ones((1, 3, 8, 8), dtype=np.float32))


@pytest.mark.parametrize(
    "runtime_metadata",
    [
        {},
        _runtime_metadata(model_family=None),
        _runtime_metadata(names="not-json"),
        _runtime_metadata(nc=0),
        _runtime_metadata(task="unknown"),
        _runtime_metadata(schema_version="2.0"),
        _runtime_metadata(libreyolo_version=None),
    ],
)
def test_triton_rejects_missing_or_malformed_runtime_metadata(
    monkeypatch, runtime_metadata
):
    config, metadata = _model_responses(runtime_metadata=runtime_metadata)
    _install_fake_client(monkeypatch, config=config, metadata=metadata)

    with pytest.raises((ExportMetadataError, ValueError)):
        TritonBackend("http://127.0.0.1:8000/detector")


def test_triton_rejects_missing_metadata_parameter(monkeypatch):
    config, metadata = _model_responses()
    config["parameters"] = {}
    _install_fake_client(monkeypatch, config=config, metadata=metadata)

    with pytest.raises(ExportMetadataError, match=TRITON_METADATA_PARAMETER):
        TritonBackend("http://127.0.0.1:8000/detector")


def test_triton_requires_exactly_one_input(monkeypatch):
    config, metadata = _model_responses()
    config["input"].append(dict(config["input"][0], name="other"))
    _install_fake_client(monkeypatch, config=config, metadata=metadata)

    with pytest.raises(TritonBackendError, match="exactly one input"):
        TritonBackend("http://127.0.0.1:8000/detector")


def test_triton_rejects_config_metadata_output_mismatch(monkeypatch):
    config, metadata = _model_responses()
    metadata["outputs"].pop()
    _install_fake_client(monkeypatch, config=config, metadata=metadata)

    with pytest.raises(TritonBackendError, match="output names differ"):
        TritonBackend("http://127.0.0.1:8000/detector")


@pytest.mark.parametrize(
    ("max_batch_size", "supported"),
    [(0, False), (1, False), (4, True)],
)
def test_triton_batch_capability(monkeypatch, max_batch_size, supported):
    config, metadata = _model_responses(max_batch_size=max_batch_size)
    _install_fake_client(monkeypatch, config=config, metadata=metadata)
    backend = TritonBackend("http://127.0.0.1:8000/detector")

    assert backend._supports_batched_inference() is supported


def test_triton_caps_requested_batch_to_server_limit(monkeypatch):
    config, metadata = _model_responses(max_batch_size=4)
    _install_fake_client(monkeypatch, config=config, metadata=metadata)
    backend = TritonBackend("http://127.0.0.1:8000/detector")
    captured = {}

    def fake_process(self, images, batch=1, **kwargs):
        captured["batch"] = batch
        return []

    monkeypatch.setattr(BaseBackend, "_process_in_batches", fake_process)
    backend._process_in_batches([], batch=12)

    assert captured["batch"] == 4
