"""Unit tests for CoreML export. Mocks coremltools so it runs on every platform."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Install a fake `coremltools` module so the import inside libreyolo.export.coreml
# succeeds even on machines without coremltools installed. Only do this if the
# real coremltools is genuinely unavailable, so we don't pollute sys.modules
# for any e2e test that runs in the same pytest session.
try:  # pragma: no cover - environment-dependent
    import coremltools  # noqa: F401
except ImportError:
    _fake_ct = MagicMock()
    _fake_ct.ComputeUnit.ALL = "ALL"
    _fake_ct.ComputeUnit.CPU_AND_GPU = "CPU_AND_GPU"
    _fake_ct.ComputeUnit.CPU_AND_NE = "CPU_AND_NE"
    _fake_ct.ComputeUnit.CPU_ONLY = "CPU_ONLY"
    _fake_ct.precision.FLOAT32 = "FLOAT32"
    _fake_ct.precision.FLOAT16 = "FLOAT16"
    _fake_ct.target.iOS15 = "iOS15"
    sys.modules["coremltools"] = _fake_ct

from libreyolo.export.coreml import _stringify_metadata, _to_compute_unit  # noqa: E402


pytestmark = pytest.mark.unit


class TestToComputeUnit:
    def test_all(self):
        import coremltools as ct
        assert _to_compute_unit("all") == ct.ComputeUnit.ALL

    def test_cpu_and_gpu(self):
        import coremltools as ct
        assert _to_compute_unit("cpu_and_gpu") == ct.ComputeUnit.CPU_AND_GPU

    def test_cpu_and_ne(self):
        import coremltools as ct
        assert _to_compute_unit("cpu_and_ne") == ct.ComputeUnit.CPU_AND_NE

    def test_cpu_only(self):
        import coremltools as ct
        assert _to_compute_unit("cpu_only") == ct.ComputeUnit.CPU_ONLY

    def test_case_insensitive(self):
        import coremltools as ct
        assert _to_compute_unit("ALL") == ct.ComputeUnit.ALL
        assert _to_compute_unit("Cpu_And_Ne") == ct.ComputeUnit.CPU_AND_NE

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="compute_units"):
            _to_compute_unit("tpu")


class _DummyModel(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


class _DummyYoloxExportModel(torch.nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        return torch.zeros(batch, 10, 85, dtype=x.dtype, device=x.device)


class _DummyRtdetrExportModel(torch.nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        return {
            "pred_logits": torch.zeros(batch, 300, 80, dtype=x.dtype, device=x.device),
            "pred_boxes": torch.zeros(batch, 300, 4, dtype=x.dtype, device=x.device),
        }


class _DummyRtdetrStaticEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eval_spatial_size = [96, 80]
        self.use_encoder_idx = [0, 1]
        self.feat_strides = [8, 16]
        self.hidden_dim = 4
        self.pe_temperature = 10_000
        self.register_buffer(
            "pos_embed0",
            torch.full((1, 1, self.hidden_dim), -1.0),
            persistent=False,
        )

    @staticmethod
    def build_2d_sincos_position_embedding(width, height, hidden_dim, temperature):
        del temperature
        return torch.full((1, width * height, hidden_dim), 7.0)


class _DummyRtdetrStaticDecoder(torch.nn.Module):
    def __init__(self, *, fail_preparation=False):
        super().__init__()
        self.eval_spatial_size = [96, 80]
        self.fail_preparation = fail_preparation
        self.register_buffer("anchors", torch.full((1, 4), -2.0), persistent=False)

    def _generate_anchors(self, *, device):
        if self.fail_preparation:
            raise RuntimeError("anchor preparation failed")
        return torch.full((1, 4), 3.0, device=device), torch.ones(
            (1, 1), dtype=torch.bool, device=device
        )


class _DummyRtdetrStaticModel(torch.nn.Module):
    def __init__(self, *, fail_preparation=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.encoder = _DummyRtdetrStaticEncoder()
        self.decoder = _DummyRtdetrStaticDecoder(fail_preparation=fail_preparation)

    def forward(self, x):
        batch = x.shape[0]
        return {
            "pred_logits": self.scale
            * torch.ones(batch, 2, 3, dtype=x.dtype, device=x.device),
            "pred_boxes": self.scale
            * torch.ones(batch, 2, 4, dtype=x.dtype, device=x.device),
        }


def _snapshot_module_tensors(model):
    return {
        "parameters": {
            name: (tensor, tensor.detach().clone(), tensor.device, tensor.dtype)
            for name, tensor in model.named_parameters()
        },
        "buffers": {
            name: (tensor, tensor.detach().clone(), tensor.device, tensor.dtype)
            for name, tensor in model.named_buffers()
        },
        "modes": [(module, module.training) for module in model.modules()],
        "attribute_keys": [
            (module, frozenset(module.__dict__)) for module in model.modules()
        ],
    }


def _assert_module_tensors_unchanged(model, snapshot):
    for kind in ("parameters", "buffers"):
        current = dict(
            model.named_parameters() if kind == "parameters" else model.named_buffers()
        )
        assert current.keys() == snapshot[kind].keys()
        for name, (original, value, device, dtype) in snapshot[kind].items():
            assert current[name] is original
            assert current[name].device == device
            assert current[name].dtype == dtype
            assert torch.equal(current[name], value)
    assert [(module, module.training) for module in model.modules()] == snapshot[
        "modes"
    ]
    assert [
        (module, frozenset(module.__dict__)) for module in model.modules()
    ] == snapshot["attribute_keys"]


def _patch_ct(monkeypatch):
    """Reset the fake coremltools module and return the mock for assertions."""
    # Create the main coremltools mock
    fake = MagicMock()
    fake.ComputeUnit.ALL = "ALL"
    fake.ComputeUnit.CPU_AND_GPU = "CPU_AND_GPU"
    fake.ComputeUnit.CPU_AND_NE = "CPU_AND_NE"
    fake.ComputeUnit.CPU_ONLY = "CPU_ONLY"
    fake.precision.FLOAT32 = "FLOAT32"
    fake.precision.FLOAT16 = "FLOAT16"
    fake.target.iOS15 = "iOS15"
    fake.ImageType = MagicMock(side_effect=lambda **kw: ("ImageType", kw))
    fake.TensorType = MagicMock(side_effect=lambda **kw: ("TensorType", kw))

    # Create models submodule mock
    fake_models = MagicMock()
    fake_models.pipeline = MagicMock()
    fake.models = fake_models
    # MLModel is in the models submodule
    fake.models.MLModel = MagicMock()

    # Create the MLModel mock that gets returned by convert
    mlmodel = MagicMock()
    mlmodel.user_defined_metadata = {}
    fake.convert = MagicMock(return_value=mlmodel)

    # Patch the module and submodules
    monkeypatch.setitem(sys.modules, "coremltools", fake)
    monkeypatch.setitem(sys.modules, "coremltools.models", fake_models)
    monkeypatch.setitem(
        sys.modules, "coremltools.models.pipeline", fake_models.pipeline
    )

    return fake, mlmodel


class TestExportCoreML:
    def test_fp32_basic_call(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        nn_model = _DummyModel().eval()
        dummy = torch.randn(1, 3, 640, 640)
        out = tmp_path / "model.mlpackage"

        result = export_coreml(
            nn_model, dummy,
            output_path=str(out),
            precision="fp32",
            compute_units="all",
            nms=False,
            metadata={"libreyolo_version": "0.0.1", "model_family": "yolox",
                      "names": {"0": "person"}, "imgsz": 640},
            model_family="yolox",
        )

        assert result == str(out)
        # ct.convert called with mlprogram + FLOAT32 + iOS15 + ImageType input
        kwargs = fake.convert.call_args.kwargs
        assert kwargs["convert_to"] == "mlprogram"
        assert kwargs["compute_precision"] == "FLOAT32"
        assert kwargs["minimum_deployment_target"] == "iOS15"
        # ImageType called with scale=1/255 and image input name 'image'
        img_kwargs = fake.ImageType.call_args.kwargs
        assert img_kwargs["name"] == "image"
        assert img_kwargs["scale"] == pytest.approx(1.0 / 255.0)
        assert img_kwargs["bias"] == [0.0, 0.0, 0.0]
        # Compute unit set
        assert mlmodel.compute_unit == "ALL"
        # Metadata was stringified and stored
        assert all(isinstance(v, str) for v in mlmodel.user_defined_metadata.values())
        assert mlmodel.user_defined_metadata["model_family"] == "yolox"
        # Save called
        mlmodel.save.assert_called_once_with(str(out))

    def test_fp16_uses_float16_precision(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        export_coreml(
            _DummyModel().eval(),
            torch.randn(1, 3, 640, 640),
            output_path=str(tmp_path / "m.mlpackage"),
            precision="fp16",
            compute_units="cpu_and_ne",
            nms=False,
            metadata={"model_family": "yolox"},
            model_family="yolox",
        )
        assert fake.convert.call_args.kwargs["compute_precision"] == "FLOAT16"
        assert mlmodel.compute_unit == "CPU_AND_NE"

    def test_metadata_names_json_encoded(self, tmp_path, monkeypatch):
        import json
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        export_coreml(
            _DummyModel().eval(),
            torch.randn(1, 3, 640, 640),
            output_path=str(tmp_path / "m.mlpackage"),
            precision="fp32",
            compute_units="all",
            nms=False,
            metadata={"names": {"0": "person", "1": "cat"}, "imgsz": 640},
            model_family="yolox",
        )
        decoded = json.loads(mlmodel.user_defined_metadata["names"])
        assert decoded == {"0": "person", "1": "cat"}

    def test_supported_tasks_json_encoded(self):
        import json

        metadata = _stringify_metadata(
            {
                "supported_tasks": ["detect", "segment"],
                "default_task": "detect",
            }
        )

        assert json.loads(metadata["supported_tasks"]) == ["detect", "segment"]

    def test_rtdetr_dict_output_is_flattened_for_trace(self):
        from libreyolo.export.coreml import _wrap_for_family

        wrapped = _wrap_for_family(_DummyRtdetrExportModel().eval(), "rtdetr")
        logits, boxes = wrapped(torch.randn(1, 3, 640, 640))

        assert logits.shape == (1, 300, 80)
        assert boxes.shape == (1, 300, 4)

    @pytest.mark.parametrize("failure_stage", [None, "trace", "preparation"])
    def test_rtdetr_static_preparation_does_not_mutate_live_model(
        self, tmp_path, monkeypatch, failure_stage
    ):
        fake, _ = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        model = _DummyRtdetrStaticModel(fail_preparation=failure_stage == "preparation")
        model.train()
        model.encoder.eval()  # Exercise restoration of a mixed-mode module tree.
        encoder_size = model.encoder.eval_spatial_size
        decoder_size = model.decoder.eval_spatial_size
        snapshot = _snapshot_module_tensors(model)

        def trace(prepared, dummy):
            inner = prepared.model
            assert inner is not model
            assert inner.encoder.eval_spatial_size == (32, 48)
            assert inner.decoder.eval_spatial_size == (32, 48)
            assert torch.all(inner.encoder.pos_embed0 == 7)
            assert hasattr(inner.encoder, "pos_embed1")
            assert torch.all(inner.decoder.anchors == 3)
            assert torch.all(inner.decoder.valid_mask)
            if failure_stage == "trace":
                raise RuntimeError("trace failed")
            return MagicMock(name="traced")

        monkeypatch.setattr(torch.jit, "trace", trace)

        def call():
            return export_coreml(
                model,
                torch.randn(1, 3, 32, 48),
                output_path=str(tmp_path / "rtdetr.mlpackage"),
                metadata={"model_family": "rtdetr"},
                model_family="rtdetr",
            )

        if failure_stage is None:
            call()
            fake.convert.assert_called_once()
        else:
            with pytest.raises(RuntimeError):
                call()

        _assert_module_tensors_unchanged(model, snapshot)
        assert model.encoder.eval_spatial_size is encoder_size
        assert model.decoder.eval_spatial_size is decoder_size
        assert not hasattr(model.encoder, "pos_embed1")
        assert "valid_mask" not in model.decoder._buffers


class TestUnsupportedFamily:
    def test_unknown_family_raises(self, tmp_path, monkeypatch):
        _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        with pytest.raises(NotImplementedError, match="not supported"):
            export_coreml(
                _DummyModel().eval(),
                torch.randn(1, 3, 640, 640),
                output_path=str(tmp_path / "m.mlpackage"),
                precision="fp32",
                compute_units="all",
                nms=False,
                metadata={"model_family": "yolonas"},
                model_family="yolonas",
            )


class TestNMSWrap:
    def test_rfdetr_raises(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        with pytest.raises(NotImplementedError, match="RF-DETR"):
            export_coreml(
                _DummyModel().eval(),
                torch.randn(1, 3, 640, 640),
                output_path=str(tmp_path / "m.mlpackage"),
                precision="fp32",
                compute_units="all",
                nms=True,
                metadata={"model_family": "rfdetr"},
                model_family="rfdetr",
            )

    def test_rtdetr_raises(self, tmp_path, monkeypatch):
        _patch_ct(monkeypatch)
        from libreyolo.export.coreml import export_coreml

        with pytest.raises(NotImplementedError, match="RT-DETR"):
            export_coreml(
                _DummyModel().eval(),
                torch.randn(1, 3, 640, 640),
                output_path=str(tmp_path / "m.mlpackage"),
                precision="fp32",
                compute_units="all",
                nms=True,
                metadata={"model_family": "rtdetr"},
                model_family="rtdetr",
            )

    def test_yolox_calls_pipeline(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)

        from libreyolo.export import coreml as coreml_mod

        wrap = MagicMock(return_value=mlmodel)
        monkeypatch.setattr(coreml_mod, "_wrap_with_nms", wrap)

        coreml_mod.export_coreml(
            _DummyYoloxExportModel().eval(),
            torch.randn(1, 3, 640, 640),
            output_path=str(tmp_path / "m.mlpackage"),
            precision="fp32",
            compute_units="all",
            nms=True,
            metadata={"model_family": "yolox", "nb_classes": 80},
            model_family="yolox",
        )
        kwargs = fake.convert.call_args.kwargs
        assert kwargs["outputs"] == [
            ("TensorType", {"name": "confidence"}),
            ("TensorType", {"name": "coordinates"}),
        ]
        wrap.assert_called_once_with(mlmodel, model_family="yolox", iou=0.45, conf=0.25)
        assert mlmodel.user_defined_metadata["nms"] == "True"


class TestCoreMLExporterRegistry:
    def test_format_registered(self):
        from libreyolo.export.exporter import BaseExporter, CoreMLExporter
        assert "coreml" in BaseExporter._registry
        assert BaseExporter._registry["coreml"] is CoreMLExporter

    def test_class_attrs(self):
        from libreyolo.export.exporter import CoreMLExporter
        assert CoreMLExporter.format_name == "coreml"
        assert CoreMLExporter.suffix == ".mlpackage"
        assert CoreMLExporter.requires_onnx is False
        assert CoreMLExporter.supports_int8 is False
        assert CoreMLExporter.apply_model_half is False


class TestCoreMLBackendModule:
    def test_backend_class_importable(self):
        # On non-macOS, importing the class itself must succeed (only
        # instantiation should refuse). Use the lazy import path.
        import libreyolo
        assert hasattr(libreyolo, "CoreMLBackend")
        cls = libreyolo.CoreMLBackend
        assert cls.__name__ == "CoreMLBackend"

    def test_dispatch_mlpackage(self, tmp_path, monkeypatch):
        # Create a fake .mlpackage directory and ensure the model factory
        # routes it to CoreMLBackend (we patch the class to a sentinel).
        pkg = tmp_path / "fake.mlpackage"
        pkg.mkdir()

        sentinel = MagicMock(name="CoreMLBackendSentinel")
        import libreyolo.backends.coreml as coreml_mod
        monkeypatch.setattr(coreml_mod, "CoreMLBackend", sentinel)

        from libreyolo.models import LibreYOLO
        LibreYOLO(str(pkg), nb_classes=80, device="cpu")
        sentinel.assert_called_once()

    def test_backend_preserves_task_metadata(self, tmp_path, monkeypatch):
        fake, mlmodel = _patch_ct(monkeypatch)
        monkeypatch.setattr(sys, "platform", "darwin")

        pkg = tmp_path / "fake.mlpackage"
        pkg.mkdir()

        mlmodel.user_defined_metadata = {
            "model_family": "rfdetr",
            "model_size": "n",
            "task": "segment",
            "default_task": "detect",
            "supported_tasks": '["detect", "segment"]',
            "names": '{"0": "person"}',
            "imgsz": "512",
        }
        mlmodel.get_spec.return_value = SimpleNamespace(
            description=SimpleNamespace(
                output=[
                    SimpleNamespace(name="boxes"),
                    SimpleNamespace(name="logits"),
                    SimpleNamespace(name="masks"),
                ]
            )
        )
        fake.models.MLModel.return_value = mlmodel

        from libreyolo.backends.coreml import CoreMLBackend

        backend = CoreMLBackend(str(pkg), nb_classes=80)

        assert backend.model_family == "rfdetr"
        assert backend.model_size == "n"
        assert backend.size == "n"
        assert backend.task == "segment"
        assert backend.DEFAULT_TASK == "detect"
        assert backend.SUPPORTED_TASKS == ("detect", "segment")
        assert backend.imgsz == 512
        assert backend.names == {0: "person"}

    def test_backend_parses_legacy_supported_tasks_repr(self):
        from libreyolo.backends.coreml import CoreMLBackend

        parsed = CoreMLBackend._parse_metadata(
            {
                "task": "segment",
                "default_task": "detect",
                "supported_tasks": "['detect', 'segment']",
            },
            default_nb_classes=80,
        )

        assert parsed[2] == "segment"
        assert parsed[3] == ("detect", "segment")
        assert parsed[4] == "detect"

    def test_backend_parses_rectangular_metadata(self):
        from libreyolo.backends.coreml import CoreMLBackend

        parsed = CoreMLBackend._parse_metadata(
            {
                "model_family": "yolo9",
                "imgsz": "640",
                "imgsz_h": "320",
                "imgsz_w": "640",
            },
            default_nb_classes=80,
        )

        assert parsed[6] == (320, 640)

    def test_backend_preprocess_accepts_rectangular_yolo9_imgsz(self):
        from libreyolo.backends.coreml import CoreMLBackend

        backend = CoreMLBackend.__new__(CoreMLBackend)
        backend.model_family = "yolo9"

        tensor, _original_img, original_size, ratio = backend._preprocess(
            np.zeros((8, 16, 3), dtype=np.uint8),
            (32, 64),
            "rgb",
        )

        assert tuple(tensor.shape) == (1, 3, 32, 64)
        assert original_size == (16, 8)
        assert ratio == 1.0
