"""Trained or input-sensitive OpenVINO parity for Round 21 DETR task cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
    pytest.mark.network,
    pytest.mark.slow,
]


@dataclass(frozen=True)
class OpenVINORound21Case:
    family: str
    task: str
    imgsz: int
    weights: str | None = None


ROUND21_OPENVINO_CASES = (
    OpenVINORound21Case("deim", "detect", 640, "LibreDEIMn.pt"),
    OpenVINORound21Case("rtdetrv2", "detect", 640, "LibreRTDETRv2r18.pt"),
    OpenVINORound21Case("ec", "pose", 640, "LibreECs-pose.pt"),
    OpenVINORound21Case("rfdetr", "segment", 312, "LibreRFDETRn-seg.pt"),
    OpenVINORound21Case("rfdetr", "pose", 576, "LibreRFDETRx-pose.pt"),
    OpenVINORound21Case("rfdetr", "obb", 384),
)

ROUND21_OPENVINO_HOLDS = {
    ("deim", "detect"): (
        "Trained outputs carry 17.9x more input signal than conversion error; "
        "validation requires more than 20x."
    ),
    ("rtdetrv2", "detect"): (
        "Only 93.94% of trained raw elements meet the converted-runtime tolerance."
    ),
    ("ec", "pose"): (
        "Raw parity passes, but trained public boxes fall to 0.916 matched IoU."
    ),
    ("rfdetr", "segment"): (
        "Only 69.0% of trained raw elements meet the converted-runtime tolerance."
    ),
    ("rfdetr", "pose"): (
        "Only 72.75% of trained raw elements meet the converted-runtime tolerance."
    ),
    ("rfdetr", "obb"): (
        "Only 91.25% of input-sensitive raw elements meet the converted-runtime tolerance."
    ),
}

ROUND21_OPENVINO_PARAMS = tuple(
    pytest.param(
        case,
        marks=pytest.mark.xfail(
            strict=True,
            reason=ROUND21_OPENVINO_HOLDS[(case.family, case.task)],
        ),
        id=f"{case.family}-{case.task}",
    )
    for case in ROUND21_OPENVINO_CASES
)


def _tensor_outputs(output) -> tuple[torch.Tensor, ...]:
    if isinstance(output, torch.Tensor):
        return (output,)
    if isinstance(output, dict):
        ordered = ("pred_logits", "pred_boxes", "pred_masks", "pred_keypoints", "pred_angles")
        return tuple(output[key] for key in ordered if key in output)
    return tuple(output)


def _native_outputs(model, tensor: torch.Tensor, imgsz: int) -> tuple[np.ndarray, ...]:
    from libreyolo.export.exporter import OnnxExporter

    with (
        OnnxExporter(model)._model_context(
            torch.device("cpu"),
            False,
            False,
            1,
            (imgsz, imgsz),
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
    """Align unordered query rows using all outputs after scale normalization."""
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


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value.astype(np.float64)))))


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
        match_rate = float(matches.mean())
        assert match_rate > 0.95, f"element match rate: {match_rate:.4f}"
        expected_signal = _rms(expected_a - expected_b)
        parity_error = _rms(expected_a - actual_a)
        actual_signal = _rms(actual_a - actual_b)
        assert expected_signal > 1e-6
        assert actual_signal > 20.0 * max(parity_error, 1e-12)


def _build_model(case: OpenVINORound21Case):
    from libreyolo import LibreRFDETR, LibreYOLO

    if case.weights:
        return LibreYOLO(case.weights, device="cpu")
    model = LibreRFDETR(
        {},
        size="n",
        task="obb",
        nb_classes=2,
        device="cpu",
    )
    with torch.no_grad():
        angle_head = model.model.model.angle_embed.layers[-1]
        torch.nn.init.uniform_(angle_head.weight, -0.02, 0.02)
        torch.nn.init.uniform_(angle_head.bias, -0.02, 0.02)
    return model


def _assert_public_parity(case, native, runtime):
    from .test_onnx_round16 import _assert_predict_parity
    from .test_tensorrt_round11 import _assert_predict_parity as _assert_pose_parity
    from .test_tensorrt_round12 import _assert_predict_parity as _assert_rf_parity

    image = np.random.default_rng(61).integers(
        0,
        256,
        size=(case.imgsz, case.imgsz, 3),
        dtype=np.uint8,
    )
    if case.task == "detect":
        expected = native.predict(
            image,
            imgsz=case.imgsz,
            conf=0.0,
            max_det=100,
        )
        actual = runtime.predict(image, conf=0.0, max_det=100)
        _assert_predict_parity(expected, actual)
        return
    if case.family == "ec":
        _assert_pose_parity(case, native, runtime, image)
        return
    _assert_rf_parity(case, native, runtime)


@pytest.mark.parametrize(
    "case",
    ROUND21_OPENVINO_PARAMS,
)
def test_openvino_round21_parity(tmp_path, case):
    pytest.importorskip("openvino")
    from libreyolo import LibreYOLO

    torch.manual_seed(21)
    model = _build_model(case)
    model.model.eval()
    assert model.FAMILY == case.family
    assert model.task == case.task

    first = torch.rand(
        1,
        3,
        case.imgsz,
        case.imgsz,
        generator=torch.Generator().manual_seed(21),
    )
    second = 1.0 - first
    expected_first = _native_outputs(model, first, case.imgsz)
    expected_second = _native_outputs(model, second, case.imgsz)

    artifact = model.export(
        "openvino",
        output_path=str(tmp_path / f"{case.family}-{case.task}_openvino"),
        imgsz=case.imgsz,
        dynamic=False,
        half=False,
        simplify=False,
    )
    runtime = LibreYOLO(artifact, device="cpu")
    actual_first = _align_queries(
        runtime._run_inference(first.numpy()),
        expected_first,
    )
    actual_second = _align_queries(
        runtime._run_inference(second.numpy()),
        expected_second,
    )
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
