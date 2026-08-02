"""Round 23 parity probes for normal and edge export gaps."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
]


@dataclass(frozen=True)
class Round23Case:
    family: str
    task: str
    format: str
    imgsz: int


def _case(family: str, task: str, format: str, imgsz: int):
    marks = []
    if format == "tensorrt":
        marks.extend((pytest.mark.tensorrt, pytest.mark.trt))
    elif format == "openvino":
        marks.append(pytest.mark.openvino)
    elif format == "torchscript":
        marks.append(pytest.mark.torchscript)
    elif format == "tflite":
        marks.append(pytest.mark.slow)
    case = Round23Case(family, task, format, imgsz)
    return pytest.param(case, marks=marks, id=f"{family}-{task}-{format}")


ROUND23_CONVERTER_CASES = (
    _case("moge2", "normal", "torchscript", 28),
    _case("moge2", "normal", "openvino", 28),
    _case("moge2", "normal", "tensorrt", 28),
    _case("segformer", "semantic", "tensorrt", 64),
    _case("teed", "edge", "tflite", 64),
    _case("dexined", "edge", "tflite", 64),
)


def _build_model(case: Round23Case, device: str):
    from libreyolo import LibreDexiNed, LibreMoGe2, LibreSegformer, LibreTEED

    if case.family == "moge2":
        return LibreMoGe2(None, size="s", task="normal", device=device)
    if case.family == "teed":
        return LibreTEED(None, size="t", device=device)
    if case.family == "segformer":
        return LibreSegformer(
            None,
            size="b0",
            nb_classes=3,
            device=device,
        )
    return LibreDexiNed(None, size="b", device=device)


def _exporter(case: Round23Case, model):
    from libreyolo.export.exporter import (
        OpenVINOExporter,
        TensorRTExporter,
        TFLiteExporter,
        TorchScriptExporter,
    )

    exporters = {
        "torchscript": TorchScriptExporter,
        "openvino": OpenVINOExporter,
        "tensorrt": TensorRTExporter,
        "tflite": TFLiteExporter,
    }
    return exporters[case.format](model)


def _outputs(value) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, dict):
        value = value.values()
    tensors = []
    for item in value:
        tensors.extend(_outputs(item))
    return tuple(tensors)


def _native_outputs(
    case: Round23Case,
    model,
    tensor: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, ...]:
    with (
        _exporter(case, model)._model_context(
            device,
            False,
            False,
            1,
            (case.imgsz, case.imgsz),
        ) as (prepared, _),
        torch.inference_mode(),
    ):
        output = prepared(tensor.to(device))
    return tuple(value.detach().float().cpu().numpy() for value in _outputs(output))


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value.astype(np.float64)))))


def _assert_raw_parity(expected_a, expected_b, actual_a, actual_b):
    assert len(actual_a) == len(expected_a)
    assert len(actual_b) == len(expected_b)
    for reference, changed_reference, actual, changed_actual in zip(
        expected_a,
        expected_b,
        actual_a,
        actual_b,
    ):
        assert actual.shape == reference.shape
        assert changed_actual.shape == changed_reference.shape
        np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=5e-3)
        np.testing.assert_allclose(
            changed_actual,
            changed_reference,
            rtol=2e-3,
            atol=5e-3,
        )
        error = max(_rms(actual - reference), _rms(changed_actual - changed_reference))
        signal = max(_rms(reference - changed_reference), _rms(actual - changed_actual))
        assert signal > max(20.0 * error, 1e-5)


def _assert_public_parity(case: Round23Case, native, runtime):
    image = np.random.default_rng(23).integers(
        0,
        256,
        size=(case.imgsz, case.imgsz, 3),
        dtype=np.uint8,
    )
    expected = native.predict(image, imgsz=case.imgsz)
    actual = runtime.predict(image)
    if case.task == "semantic":
        expected_mask = expected.semantic_mask.data.cpu().numpy()
        actual_mask = actual.semantic_mask.data.cpu().numpy()
        assert actual_mask.shape == expected_mask.shape
        assert float(np.mean(actual_mask == expected_mask)) > 0.95
        return
    if case.task == "edge":
        expected_edges = expected.edges.data.cpu().numpy()
        actual_edges = actual.edges.data.cpu().numpy()
        mse = float(
            np.mean(
                np.square(
                    expected_edges.astype(np.float64)
                    - actual_edges.astype(np.float64)
                )
            )
        )
        psnr = float("inf") if mse == 0.0 else -10.0 * np.log10(mse)
        assert psnr > 40.0
        return

    expected_normal = expected.normal_map.data.cpu().numpy().astype(np.float64)
    actual_normal = actual.normal_map.data.cpu().numpy().astype(np.float64)
    assert actual_normal.shape == expected_normal.shape
    dots = np.sum(expected_normal * actual_normal, axis=-1)
    angular = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))
    assert float(np.mean(angular)) < 0.1
    np.testing.assert_allclose(
        np.linalg.norm(actual_normal, axis=-1),
        1.0,
        atol=1e-5,
    )


@pytest.mark.parametrize("case", ROUND23_CONVERTER_CASES)
def test_round23_converter_export_predict_parity(tmp_path, case):
    if case.format == "openvino":
        pytest.importorskip("openvino")
    elif case.format == "tensorrt":
        pytest.importorskip("tensorrt")
        if not torch.cuda.is_available():
            pytest.skip("TensorRT parity requires CUDA")
    elif case.format == "tflite" and (
        importlib.util.find_spec("onnx2tf") is None
        or importlib.util.find_spec("ai_edge_litert") is None
    ):
        pytest.skip("onnx2tf and ai-edge-litert are required")

    from libreyolo import LibreYOLO
    torch.manual_seed(23)
    device = torch.device("cuda" if case.format == "tensorrt" else "cpu")
    model = _build_model(case, str(device))
    model.model.eval()
    first = torch.rand(
        1,
        3,
        case.imgsz,
        case.imgsz,
        generator=torch.Generator().manual_seed(23),
    )
    second = 1.0 - first
    expected_first = _native_outputs(case, model, first, device)
    expected_second = _native_outputs(case, model, second, device)

    suffix = {
        "torchscript": ".torchscript",
        "tensorrt": ".engine",
        "openvino": "_openvino",
        "tflite": ".tflite",
    }[case.format]
    artifact = model.export(
        case.format,
        output_path=str(tmp_path / f"{case.family}-{case.task}{suffix}"),
        imgsz=case.imgsz,
        batch=1,
        dynamic=False,
        half=False,
        simplify=False,
    )
    runtime = LibreYOLO(artifact, device=str(device))
    actual_first = runtime._run_inference(first.numpy())
    actual_second = runtime._run_inference(second.numpy())
    _assert_raw_parity(
        expected_first,
        expected_second,
        actual_first,
        actual_second,
    )
    _assert_public_parity(case, model, runtime)
    assert runtime.model_family == case.family
    assert runtime.task == case.task
    assert runtime.imgsz == case.imgsz
