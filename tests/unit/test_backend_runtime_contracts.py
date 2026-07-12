from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libreyolo.backends.onnx import OnnxBackend, _resolve_onnx_providers
from libreyolo.backends.coreml import CoreMLBackend
from libreyolo.backends.torchscript import TorchScriptBackend
from libreyolo.validation.base import BaseValidator, validation_model_state
from libreyolo.validation.detection_validator import DetectionValidator


pytestmark = pytest.mark.unit


def test_onnx_runtime_input_is_cast_to_declared_dtype():
    captured = {}

    class _Session:
        def run(self, output_names, feed):
            captured["output_names"] = output_names
            captured["blob"] = feed["images"]
            return [np.zeros((1,), dtype=np.float16)]

    backend = OnnxBackend.__new__(OnnxBackend)
    backend.session = _Session()
    backend.input_name = "images"
    backend.input_dtype = np.dtype(np.float16)

    outputs = backend._run_inference(np.ones((1, 3, 4, 4), dtype=np.float32))

    assert outputs[0].dtype == np.float16
    assert captured["output_names"] is None
    assert captured["blob"].dtype == np.float16
    assert captured["blob"].flags.c_contiguous


@pytest.mark.parametrize("device", ["cpu", "CPU", torch.device("cpu")])
def test_onnx_explicit_cpu_uses_cpu_provider_without_warning(device, caplog):
    providers, resolved = _resolve_onnx_providers(
        device,
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert providers == ["CPUExecutionProvider"]
    assert resolved == "cpu"
    assert not caplog.records


def test_onnx_indexed_cuda_passes_device_id_to_execution_provider():
    providers, resolved = _resolve_onnx_providers(
        "cuda:2",
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert providers == [
        ("CUDAExecutionProvider", {"device_id": 2}),
        "CPUExecutionProvider",
    ]
    assert resolved == "cuda:2"


def test_onnx_runtime_rejects_unrepresentable_input_type():
    with pytest.raises(TypeError, match=r"tensor\(bfloat16\)"):
        OnnxBackend._numpy_dtype_for_onnx_type("tensor(bfloat16)")


def test_onnx_embedded_metadata_fallback_is_used_for_all_runtime_contracts(
    monkeypatch, tmp_path
):
    metadata = {
        "model_family": "clip",
        "model_size": "tiny",
        "task": "classify",
        "supported_tasks": '["classify"]',
        "default_task": "classify",
        "names": '{"0": "cat", "1": "dog"}',
        "imgsz": "8",
        "classification_mean": "[0.1, 0.2, 0.3]",
        "classification_std": "[0.4, 0.5, 0.6]",
        "classification_crop_pct": "1.0",
        "classification_interpolation": "bicubic",
        "classification_square_resize": "true",
        "classification_activation": "sigmoid",
        "num_bins": "12",
        "bin_width_deg": "5.0",
        "offset_deg": "-30.0",
    }

    class _Session:
        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="images", type="tensor(float)", shape=[1, 3, 8, 8]
                )
            ]

        def get_outputs(self):
            return [SimpleNamespace(name="scores")]

        def get_modelmeta(self):
            raise RuntimeError("runtime metadata unavailable")

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda path, providers: _Session(),
    )
    fake_onnx = SimpleNamespace(
        load=lambda path: SimpleNamespace(
            metadata_props=[
                SimpleNamespace(key=key, value=value)
                for key, value in metadata.items()
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    model_path = tmp_path / "classifier.onnx"
    model_path.write_bytes(b"placeholder")

    backend = OnnxBackend(str(model_path), device="cpu")

    assert backend.task == "classify"
    assert backend.classification_mean == (0.1, 0.2, 0.3)
    assert backend.classification_std == (0.4, 0.5, 0.6)
    assert backend.crop_pct == 1.0
    assert backend.interpolation == "bicubic"
    assert backend.classification_square_resize is True
    assert backend.classification_activation == "sigmoid"
    assert backend.num_bins == 12
    assert backend.bin_width_deg == 5.0
    assert backend.offset_deg == -30.0


def test_torchscript_uses_first_floating_parameter_dtype():
    model = torch.nn.Conv2d(3, 2, 1).half()

    assert TorchScriptBackend._floating_model_dtype(model) == torch.float16


def test_torchscript_uses_floating_buffer_for_parameterless_module():
    class _Buffered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.ones(1, dtype=torch.float64))

    assert TorchScriptBackend._floating_model_dtype(_Buffered()) == torch.float64


def test_torchscript_runtime_casts_input_to_loaded_model_dtype():
    class _Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_dtype = None

        def forward(self, tensor):
            self.seen_dtype = tensor.dtype
            return tensor

    backend = TorchScriptBackend.__new__(TorchScriptBackend)
    backend.model = _Capture()
    backend.device = torch.device("cpu")
    backend.input_dtype = torch.float16

    outputs = backend._run_inference(np.ones((1, 3, 2, 2), dtype=np.float32))

    assert backend.model.seen_dtype == torch.float16
    assert outputs[0].dtype == np.float16


def test_validation_state_allows_runtime_without_inner_torch_model():
    runtime = SimpleNamespace(device="cpu")

    with validation_model_state(runtime, torch.device("cpu")):
        pass


def test_validator_eval_hook_is_noop_for_non_pytorch_runtime():
    validator = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace()))

    BaseValidator._set_model_eval(validator)


def test_coreml_validation_forward_restores_canonical_rgb_pixels():
    backend = CoreMLBackend.__new__(CoreMLBackend)
    backend.model_family = "yolo9"
    backend._has_embedded_nms = False
    captured = {}

    def run(blob):
        captured["blob"] = blob.copy()
        return [np.zeros((1, 1), dtype=np.float32)]

    backend._run_inference = run
    backend._forward(torch.full((1, 3, 2, 2), 0.5))

    np.testing.assert_allclose(captured["blob"], 128.0)


def test_coreml_embedded_nms_forward_restores_batch_axis():
    backend = CoreMLBackend.__new__(CoreMLBackend)
    backend.model_family = "yolo9"
    backend._has_embedded_nms = True

    def run(blob):
        marker = float(blob[0, 0, 0, 0])
        return [
            np.full((3, 2), marker, dtype=np.float32),
            np.full((3, 4), marker, dtype=np.float32),
        ]

    backend._run_inference = run
    confidence, coordinates = backend._forward(
        torch.tensor([0.25, 0.75]).view(2, 1, 1, 1).expand(2, 3, 1, 1)
    )

    assert confidence.shape == (2, 3, 2)
    assert coordinates.shape == (2, 3, 4)
    assert confidence[0, 0, 0].item() == 64.0
    assert confidence[1, 0, 0].item() == 191.0


def test_coreml_validation_inverts_rfdetr_imagenet_normalization():
    backend = CoreMLBackend.__new__(CoreMLBackend)
    backend.model_family = "rfdetr"
    canonical = torch.tensor([0.2, 0.4, 0.6]).view(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    restored = backend._validation_tensor_to_canonical_rgb(
        (canonical - mean) / std
    )

    torch.testing.assert_close(restored, canonical * 255.0)


def test_coreml_validation_converts_yolox_bgr_to_rgb():
    backend = CoreMLBackend.__new__(CoreMLBackend)
    backend.model_family = "yolox"
    bgr = torch.tensor([10.0, 20.0, 30.0]).view(1, 3, 1, 1)

    restored = backend._validation_tensor_to_canonical_rgb(bgr)

    torch.testing.assert_close(
        restored,
        torch.tensor([30.0, 20.0, 10.0]).view(1, 3, 1, 1),
    )


class _RuntimeValidator(BaseValidator):
    def _setup_dataloader(self):
        return []

    def _init_metrics(self):
        pass

    def _preprocess_batch(self, batch):
        return batch, None, None, None

    def _postprocess_predictions(self, preds, batch):
        return preds

    def _update_metrics(self, detections, targets, img_info, img_ids):
        pass

    def _compute_metrics(self):
        return {}


class _Runtime:
    def __init__(self):
        self.forward_calls = 0

    def _forward(self, images):
        self.forward_calls += 1
        return []


def test_validator_normal_loop_allows_runtime_without_model_eval():
    validator = object.__new__(_RuntimeValidator)
    validator.model = _Runtime()
    validator.config = SimpleNamespace(augment=False, verbose=False, half=False)
    validator.device = torch.device("cpu")
    validator.dataloader = [torch.zeros(1, 3, 2, 2)]
    validator.speed = {
        "preprocess": 0.0,
        "inference": 0.0,
        "postprocess": 0.0,
        "total": 0.0,
    }
    validator.seen = 0

    validator._run_validation()

    assert validator.model.forward_calls == 1


def test_validator_warmup_allows_runtime_without_model_eval():
    validator = object.__new__(_RuntimeValidator)
    validator.model = _Runtime()
    validator.config = SimpleNamespace(
        verbose=False,
        imgsz=4,
        batch_size=1,
        half=False,
    )
    validator.device = torch.device("cpu")
    validator.dataloader = SimpleNamespace(batch_size=1)

    validator._warmup_model(n_warmup=1)

    assert validator.model.forward_calls == 1


def test_augmented_loop_allows_runtime_without_model_eval():
    class _EmptyLoader(list):
        dataset = []

    validator = object.__new__(DetectionValidator)
    validator.model = SimpleNamespace(model=SimpleNamespace(), model_family="coreml")
    validator.config = SimpleNamespace(verbose=False, conf_thres=0.25)
    validator.device = torch.device("cpu")
    validator.dataloader = _EmptyLoader()
    validator.speed = {"inference": 0.0, "total": 0.0}
    validator.seen = 0

    validator._run_validation_augmented()

    assert validator.seen == 0
