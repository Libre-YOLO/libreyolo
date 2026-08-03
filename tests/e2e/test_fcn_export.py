"""FCN trained-checkpoint export and runtime parity gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from .conftest import cuda_cleanup, require_test_weights

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.fcn,
]


@dataclass(frozen=True)
class FCNExportCase:
    size: str
    format: str
    imgsz: int


def _case(size: str, format: str):
    marks = []
    if format == "onnx":
        marks.extend((pytest.mark.onnx, pytest.mark.supported_backend))
    elif format == "torchscript":
        marks.extend((pytest.mark.torchscript, pytest.mark.supported_backend))
    elif format == "openvino":
        marks.extend((pytest.mark.openvino, pytest.mark.supported_backend))
    elif format == "tensorrt":
        marks.extend(
            (
                pytest.mark.tensorrt,
                pytest.mark.trt,
                pytest.mark.supported_backend,
            )
        )
    return pytest.param(
        FCNExportCase(size, format, 520 if format == "onnx" else 64),
        marks=marks,
        id=f"{size}-{format}",
    )


FCN_EXPORT_CASES = tuple(
    _case(size, format)
    for size in ("r50", "r101")
    for format in ("onnx", "torchscript", "openvino", "tensorrt")
)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value.astype(np.float64)))))


def _cosine(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = expected.astype(np.float64).ravel()
    actual = actual.astype(np.float64).ravel()
    denominator = float(np.linalg.norm(expected) * np.linalg.norm(actual))
    return float(np.dot(expected, actual) / denominator) if denominator else 1.0


@pytest.mark.parametrize("case", FCN_EXPORT_CASES)
def test_fcn_trained_checkpoint_export_parity(tmp_path, case):
    if case.format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    elif case.format == "openvino":
        pytest.importorskip("openvino")
    elif case.format == "tensorrt":
        pytest.importorskip("tensorrt")
        if not torch.cuda.is_available():
            pytest.skip("TensorRT parity requires CUDA")

    from libreyolo import LibreYOLO

    device = "cuda" if case.format == "tensorrt" else "cpu"
    weights = require_test_weights(f"LibreFCN{case.size}.pt", expected_family="fcn")
    model = LibreYOLO(weights, size=case.size, device=device)
    generator = torch.Generator().manual_seed(637)
    first = torch.rand(1, 3, case.imgsz, case.imgsz, generator=generator)
    second = 1.0 - first

    with torch.inference_mode():
        expected_first = model.model(first.to(device))["out"].float().cpu().numpy()
        expected_second = model.model(second.to(device))["out"].float().cpu().numpy()

    suffix = {
        "onnx": ".onnx",
        "torchscript": ".torchscript",
        "openvino": "_openvino",
        "tensorrt": ".engine",
    }[case.format]
    artifact = model.export(
        case.format,
        output_path=str(tmp_path / f"LibreFCN{case.size}{suffix}"),
        imgsz=case.imgsz,
        batch=1,
        dynamic=False,
        half=False,
        simplify=False,
    )
    runtime = LibreYOLO(artifact, device=device)
    actual_first = np.asarray(runtime._run_inference(first.numpy())[0])
    actual_second = np.asarray(runtime._run_inference(second.numpy())[0])

    tolerance = 1e-4 if case.format in ("onnx", "torchscript") else 5e-3
    np.testing.assert_allclose(
        actual_first, expected_first, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        actual_second, expected_second, rtol=tolerance, atol=tolerance
    )
    error = max(
        _rms(actual_first - expected_first),
        _rms(actual_second - expected_second),
    )
    signal = max(
        _rms(expected_first - expected_second),
        _rms(actual_first - actual_second),
    )
    assert signal > max(20.0 * error, 1e-5)
    assert _cosine(expected_first, actual_first) > 0.999
    assert _cosine(expected_second, actual_second) > 0.999

    image = np.random.default_rng(637).integers(
        0, 256, size=(case.imgsz, case.imgsz, 3), dtype=np.uint8
    )
    expected_mask = (
        model.predict(image, imgsz=case.imgsz).semantic_mask.data.cpu().numpy()
    )
    actual_mask = runtime.predict(image).semantic_mask.data.cpu().numpy()
    assert actual_mask.shape == expected_mask.shape
    assert float(np.mean(actual_mask == expected_mask)) > 0.99
    assert runtime.model_family == "fcn"
    assert runtime.model_size == case.size
    assert runtime.task == "semantic"
    assert runtime.imgsz == case.imgsz
    if case.format == "onnx":
        assert runtime.output_names == ["semantic_logits"]

    del runtime, model
    cuda_cleanup()
