"""TensorRT FP32 measurements for ten harder DETR-family task cells.

Eight cases use published Apache-2.0 checkpoints and two use deterministic
LibreYOLO initialization.  The suite verifies two image-sensitive raw probes
after unordered-query alignment, factory reload and metadata, and task-aware
public ``predict()`` parity.  All ten currently remain strict measured holds.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from .conftest import requires_tensorrt
from .test_tensorrt_round10 import _image, _raw_outputs
from .test_tensorrt_round11 import _align_outputs, _match_detection_rows, _rms

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.extended_backend,
    pytest.mark.tensorrt,
    pytest.mark.trt,
]


@dataclass(frozen=True)
class TensorRTRound12Case:
    class_name: str
    family: str
    size: str
    task: str
    imgsz: int
    nb_classes: int = 3
    weights: str | None = None
    empty_checkpoint: bool = False


_ROUND12_CASES = (
    TensorRTRound12Case(
        "LibreDFINE",
        "dfine",
        "n",
        "detect",
        256,
        weights="LibreDFINEn.pt",
    ),
    TensorRTRound12Case(
        "LibreDFINE",
        "dfine",
        "n",
        "segment",
        256,
        weights="LibreDFINEn-seg.pt",
    ),
    TensorRTRound12Case(
        "LibreDEIM",
        "deim",
        "n",
        "detect",
        256,
        weights="LibreDEIMn.pt",
    ),
    TensorRTRound12Case("LibreRTDETRv2", "rtdetrv2", "r18", "detect", 256),
    TensorRTRound12Case(
        "LibreEC",
        "ec",
        "s",
        "detect",
        256,
        weights="LibreECs.pt",
    ),
    TensorRTRound12Case(
        "LibreEC",
        "ec",
        "s",
        "pose",
        256,
        weights="LibreECs-pose.pt",
    ),
    TensorRTRound12Case(
        "LibreEC",
        "ec",
        "s",
        "segment",
        256,
        weights="LibreECs-seg.pt",
    ),
    TensorRTRound12Case(
        "LibreRFDETR",
        "rfdetr",
        "n",
        "segment",
        312,
        weights="LibreRFDETRn-seg.pt",
    ),
    TensorRTRound12Case(
        "LibreRFDETR",
        "rfdetr",
        "x",
        "pose",
        576,
        nb_classes=1,
        weights="LibreRFDETRx-pose.pt",
    ),
    TensorRTRound12Case(
        "LibreRFDETR",
        "rfdetr",
        "n",
        "obb",
        384,
        empty_checkpoint=True,
    ),
)

_HOLD_REASONS = {
    ("dfine", "detect"): "Trained public top-k class membership drifts.",
    ("dfine", "segment"): "Trained public top-k class membership drifts.",
    ("deim", "detect"): "Trained raw output error is 0.41%.",
    ("rtdetrv2", "detect"): "Synthetic public boxes drift by at least 8 px.",
    ("ec", "detect"): "Trained raw output error is 1.2%.",
    ("ec", "pose"): "Trained public boxes fall to 0.920 IoU.",
    ("ec", "segment"): "Trained public top-k class membership drifts.",
    ("rfdetr", "segment"): "Trained public top-k class membership drifts.",
    ("rfdetr", "pose"): "Trained public boxes fall to 0.704 IoU.",
    ("rfdetr", "obb"): "Synthetic public top-k class membership drifts.",
}

ROUND12_CASES = tuple(
    pytest.param(
        case,
        marks=pytest.mark.xfail(
            strict=True,
            reason=_HOLD_REASONS[(case.family, case.task)],
        ),
    )
    for case in _ROUND12_CASES
)


def _build_model(case: TensorRTRound12Case):
    import libreyolo

    if case.weights:
        return libreyolo.LibreYOLO(case.weights, device="cuda")
    model_cls = getattr(libreyolo, case.class_name)
    model_path = {} if case.empty_checkpoint else None
    return model_cls(
        model_path,
        size=case.size,
        task=case.task,
        nb_classes=case.nb_classes,
        device="cuda",
    )


def _align_queries(
    actual_outputs: list[np.ndarray],
    expected_outputs: list[np.ndarray],
) -> list[np.ndarray]:
    """Align unordered DETR query rows using the geometric second output."""
    assert len(actual_outputs) == len(expected_outputs)
    assert len(actual_outputs) > 1
    actual_geometry = actual_outputs[1]
    expected_geometry = expected_outputs[1]
    assert actual_geometry.shape == expected_geometry.shape
    aligned = [np.empty_like(output) for output in actual_outputs]
    for batch_index, (actual_batch, expected_batch) in enumerate(
        zip(actual_geometry, expected_geometry)
    ):
        actual_vectors = actual_batch.reshape(actual_batch.shape[0], -1)
        expected_vectors = expected_batch.reshape(expected_batch.shape[0], -1)
        cost = np.square(
            actual_vectors[:, None] - expected_vectors[None, :]
        ).sum(axis=-1)
        actual_indices, expected_indices = linear_sum_assignment(cost)
        actual_order = actual_indices[np.argsort(expected_indices)]
        for aligned_output, actual_output in zip(aligned, actual_outputs):
            aligned_output[batch_index] = actual_output[batch_index, actual_order]
    return aligned


def _assert_mask_parity(expected, actual, matches) -> None:
    assert expected.masks is not None
    assert actual.masks is not None
    expected_masks = expected.masks.data.detach().float().cpu().numpy()
    actual_masks = actual.masks.data.detach().float().cpu().numpy()
    for expected_index, actual_index in matches:
        expected_mask = expected_masks[expected_index] > 0.5
        actual_mask = actual_masks[actual_index] > 0.5
        union = np.logical_or(expected_mask, actual_mask).sum()
        if not union:
            continue
        intersection = np.logical_and(expected_mask, actual_mask).sum()
        assert float(intersection / union) > 0.95


def _assert_keypoint_parity(expected, actual, matches) -> None:
    assert expected.keypoints is not None
    assert actual.keypoints is not None
    expected_data = expected.keypoints.data.detach().float().cpu().numpy()
    actual_data = actual.keypoints.data.detach().float().cpu().numpy()
    for expected_index, actual_index in matches:
        coordinate_l2 = np.linalg.norm(
            expected_data[expected_index, :, :2]
            - actual_data[actual_index, :, :2],
            axis=1,
        )
        assert float(np.max(coordinate_l2)) < 2.0
        if expected_data.shape[-1] > 2:
            confidence_error = np.max(
                np.abs(
                    expected_data[expected_index, :, 2]
                    - actual_data[actual_index, :, 2]
                )
            )
            assert float(confidence_error) < 0.01


def _assert_obb_parity(expected, actual, matches) -> None:
    assert expected.obb is not None
    assert actual.obb is not None
    expected_data = expected.obb.data.detach().float().cpu().numpy()
    actual_data = actual.obb.data.detach().float().cpu().numpy()
    for expected_index, actual_index in matches:
        np.testing.assert_allclose(
            actual_data[actual_index, :5],
            expected_data[expected_index, :5],
            rtol=2e-3,
            atol=2e-2,
        )
        assert (
            abs(
                float(actual_data[actual_index, 5])
                - float(expected_data[expected_index, 5])
            )
            < 0.01
        )
        assert actual_data[actual_index, 6] == expected_data[expected_index, 6]


def _assert_predict_parity(case, native_model, backend) -> None:
    image = _image(case.imgsz)
    expected = native_model.predict(
        image,
        imgsz=case.imgsz,
        conf=0.0,
        max_det=25,
    )
    actual = backend.predict(image, conf=0.0, max_det=25)
    assert expected.boxes is not None
    assert actual.boxes is not None
    expected_boxes = expected.boxes.data.detach().float().cpu().numpy()
    actual_boxes = actual.boxes.data.detach().float().cpu().numpy()
    assert len(expected_boxes) == len(actual_boxes)
    assert len(expected_boxes) > 0
    matches = _match_detection_rows(expected_boxes, actual_boxes)

    if case.task == "segment":
        _assert_mask_parity(expected, actual, matches)
    elif case.task == "pose":
        _assert_keypoint_parity(expected, actual, matches)
    elif case.task == "obb":
        _assert_obb_parity(expected, actual, matches)


def _run_tensorrt_case(tmp_path, case: TensorRTRound12Case) -> None:
    from libreyolo import LibreYOLO

    torch.manual_seed(12)
    model = _build_model(case)
    model.model.eval()
    assert model.FAMILY == case.family
    assert model.task == case.task

    first = torch.zeros(1, 3, case.imgsz, case.imgsz, device="cuda")
    second = torch.rand(1, 3, case.imgsz, case.imgsz, device="cuda")
    expected_first = _raw_outputs(model, first, case.imgsz)
    expected_second = _raw_outputs(model, second, case.imgsz)

    engine_path = tmp_path / f"{case.family}-{case.task}.engine"
    artifact = model.export(
        format="tensorrt",
        output_path=str(engine_path),
        imgsz=case.imgsz,
        dynamic=False,
        half=False,
        simplify=False,
    )
    backend = LibreYOLO(artifact, device="cuda")

    actual_first = _align_outputs(
        expected_first,
        backend._run_inference(first.cpu().numpy()),
    )
    actual_second = _align_outputs(
        expected_second,
        backend._run_inference(second.cpu().numpy()),
    )
    actual_first = _align_queries(actual_first, expected_first)
    actual_second = _align_queries(actual_second, expected_second)

    assert backend.model_family == case.family
    assert backend.task == case.task
    assert backend.imgsz == case.imgsz
    _assert_predict_parity(case, model, backend)

    for expected_a, expected_b, actual_a, actual_b in zip(
        expected_first,
        expected_second,
        actual_first,
        actual_second,
    ):
        parity_error = _rms(expected_a - actual_a)
        reference_scale = max(_rms(expected_a), 1e-6)
        expected_signal = _rms(expected_a - expected_b)
        actual_signal = _rms(actual_a - actual_b)
        assert parity_error / reference_scale < 1e-3
        assert expected_signal / reference_scale > 1e-6
        assert actual_signal / max(parity_error, 1e-12) > 20.0

    del backend, model
    gc.collect()
    torch.cuda.empty_cache()


@requires_tensorrt
@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    ROUND12_CASES,
    ids=lambda case: f"{case.family}-{case.task}",
)
def test_tensorrt_round12_measured_available(tmp_path, case):
    _run_tensorrt_case(tmp_path, case)
