"""Trained-weight ONNX parity for RT-DETRv2 and RT-DETRv4.

Both published checkpoints are Apache-2.0 and are loaded through LibreYOLO's
normal auto-download route. No checkpoint is committed. Each case verifies two
input-sensitive raw probes after unordered-query alignment, factory reload,
metadata, and public top-100 ``predict()`` parity on a non-square image.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
    pytest.mark.onnx,
    pytest.mark.network,
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("onnx") is None
        or importlib.util.find_spec("onnxruntime") is None,
        reason="onnx and onnxruntime are required",
    ),
]


@dataclass(frozen=True)
class ONNXRound16Case:
    weights: str
    family: str


_RTDETRV2 = ONNXRound16Case("LibreRTDETRv2r18.pt", "rtdetrv2")
_RTDETRV4 = ONNXRound16Case("LibreRTDETRv4s.pt", "rtdetrv4")


def _tensor_outputs(output) -> tuple[torch.Tensor, ...]:
    if isinstance(output, torch.Tensor):
        return (output,)
    if isinstance(output, dict):
        return tuple(output[key] for key in ("pred_logits", "pred_boxes"))
    return tuple(output)


def _native_outputs(model, tensor: torch.Tensor) -> tuple[np.ndarray, ...]:
    from libreyolo.export.exporter import OnnxExporter

    with (
        OnnxExporter(model)._model_context(
            tensor.device,
            False,
            False,
            1,
            (640, 640),
        ) as (wrapped, _),
        torch.inference_mode(),
    ):
        output = wrapped(tensor)
    return tuple(
        value.detach().float().cpu().numpy() for value in _tensor_outputs(output)
    )


def _align_queries(
    actual: list[np.ndarray],
    expected: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    aligned = [np.empty_like(output) for output in actual]
    for batch_index in range(expected[0].shape[0]):
        cost = None
        for actual_output, expected_output in zip(actual, expected):
            actual_vectors = actual_output[batch_index].reshape(
                actual_output.shape[1],
                -1,
            )
            expected_vectors = expected_output[batch_index].reshape(
                expected_output.shape[1],
                -1,
            )
            scale = max(float(np.std(expected_vectors)), 1e-6)
            output_cost = np.square(
                (actual_vectors[:, None] - expected_vectors[None, :]) / scale
            ).mean(axis=-1)
            cost = output_cost if cost is None else cost + output_cost
        assert cost is not None
        actual_indices, expected_indices = linear_sum_assignment(cost)
        actual_order = actual_indices[np.argsort(expected_indices)]
        for aligned_output, actual_output in zip(aligned, actual):
            aligned_output[batch_index] = actual_output[batch_index, actual_order]
    return tuple(aligned)


def _assert_raw_parity(
    expected_first,
    expected_second,
    actual_first,
    actual_second,
) -> None:
    for expected_a, expected_b, actual_a, actual_b in zip(
        expected_first,
        expected_second,
        actual_first,
        actual_second,
    ):
        matches = np.isclose(actual_a, expected_a, rtol=2e-3, atol=2e-2)
        assert float(matches.mean()) > 0.95
        expected_signal = float(
            np.sqrt(np.mean((expected_a.astype(np.float64) - expected_b) ** 2))
        )
        parity_error = float(
            np.sqrt(np.mean((expected_a.astype(np.float64) - actual_a) ** 2))
        )
        actual_signal = float(
            np.sqrt(np.mean((actual_a.astype(np.float64) - actual_b) ** 2))
        )
        assert expected_signal > 1e-6
        assert actual_signal > 20.0 * max(parity_error, 1e-12)


def _assert_predict_parity(native_result, converted_result) -> None:
    native = native_result.boxes.data.cpu().numpy()
    converted = converted_result.boxes.data.cpu().numpy()
    assert converted.shape == native.shape
    cost = np.square(converted[:, None, :4] - native[None, :, :4]).sum(axis=-1)
    converted_indices, native_indices = linear_sum_assignment(cost)
    converted = converted[converted_indices[np.argsort(native_indices)]]
    box_match = np.isclose(
        converted[:, :4],
        native[:, :4],
        rtol=2e-3,
        atol=1.0,
    ).all(axis=-1)
    score_match = np.isclose(
        converted[:, 4],
        native[:, 4],
        rtol=2e-3,
        atol=2e-2,
    )
    class_match = converted[:, 5] == native[:, 5]
    assert float(box_match.mean()) > 0.95
    assert float(score_match.mean()) > 0.95
    assert float(class_match.mean()) > 0.95


def _run_case(tmp_path, case: ONNXRound16Case) -> None:
    from libreyolo import LibreYOLO

    model = LibreYOLO(case.weights, device="cpu")
    model.model.eval()
    assert model.FAMILY == case.family

    torch.manual_seed(16)
    first = torch.rand(1, 3, 640, 640)
    second = 1.0 - first
    expected_first = _native_outputs(model, first)
    expected_second = _native_outputs(model, second)

    artifact = model.export(
        format="onnx",
        output_path=str(tmp_path / f"{case.family}.onnx"),
        imgsz=640,
        dynamic=False,
        half=False,
        simplify=False,
    )
    backend = LibreYOLO(artifact, device="cpu")
    actual_first = _align_queries(
        backend._run_inference(first.cpu().numpy()),
        expected_first,
    )
    actual_second = _align_queries(
        backend._run_inference(second.cpu().numpy()),
        expected_second,
    )
    _assert_raw_parity(
        expected_first,
        expected_second,
        actual_first,
        actual_second,
    )

    assert backend.model_family == case.family
    assert backend.task == "detect"
    assert backend.imgsz == 640
    image = np.random.default_rng(51).integers(
        0,
        256,
        size=(72, 96, 3),
        dtype=np.uint8,
    )
    native_result = model.predict(image, imgsz=640, conf=0.0, max_det=100)
    converted_result = backend.predict(image, conf=0.0, max_det=100)
    _assert_predict_parity(native_result, converted_result)


def test_onnx_round16_rtdetrv2_trained_parity(tmp_path):
    _run_case(tmp_path, _RTDETRV2)


def test_onnx_round16_rtdetrv4_trained_parity(tmp_path):
    _run_case(tmp_path, _RTDETRV4)
