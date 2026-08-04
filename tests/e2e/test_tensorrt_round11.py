"""TensorRT FP32 parity for the Round 11 trained detector and pose batch.

The seven published checkpoints used here are permissively licensed and loaded
through LibreYOLO's normal factory.  YOLO9-P2 uses the pinned MIT YOLO9 transfer
fixture; YOLO-NAS detect and pose use deterministic synthetic training because
the published YOLO-NAS artifacts are proprietary.  Every case verifies two
input-sensitive raw probes, factory reload, metadata, and public ``predict()``
parity.  Synthetic fixtures validate conversion behavior, not task accuracy.
"""

from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from .conftest import requires_tensorrt
from .test_tensorrt_round10 import _image, _pairwise_box_iou, _raw_outputs

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
    pytest.mark.tensorrt,
    pytest.mark.trt,
]


@dataclass(frozen=True)
class TensorRTRound11Case:
    family: str
    task: str
    imgsz: int
    weights: str | None = None
    fixture: str | None = None
    repeatability_hold: bool = False


_YOLO1 = TensorRTRound11Case("yolo1", "detect", 448, weights="LibreYOLO1b.pt")
_YOLO7 = TensorRTRound11Case("yolo7", "detect", 640, weights="LibreYOLO7b.pt")
_YOLO9_E2E = TensorRTRound11Case(
    "yolo9_e2e",
    "detect",
    640,
    weights="LibreYOLO9E2Et.pt",
    repeatability_hold=True,
)
_YOLO9_P2 = TensorRTRound11Case(
    "yolo9_p2",
    "detect",
    640,
    fixture="yolo9_transfer",
)
_YOLOX = TensorRTRound11Case("yolox", "detect", 416, weights="LibreYOLOXn.pt")
_PICODET = TensorRTRound11Case(
    "picodet",
    "detect",
    320,
    weights="LibrePICODETs.pt",
)
_RTMDET = TensorRTRound11Case(
    "rtmdet",
    "detect",
    640,
    weights="LibreRTMDett.pt",
)
_RTDETR = TensorRTRound11Case(
    "rtdetr",
    "detect",
    640,
    weights="LibreRTDETRr18.pt",
)
_YOLONAS_DETECT = TensorRTRound11Case(
    "yolonas",
    "detect",
    96,
    fixture="synthetic_detect",
)
_YOLONAS_POSE = TensorRTRound11Case(
    "yolonas",
    "pose",
    96,
    fixture="synthetic_pose",
)

ROUND11_VALIDATED_CASES = (
    _YOLO1,
    _PICODET,
    _RTMDET,
)

ROUND11_EXPERIMENTAL_CASES = (
    pytest.param(
        _YOLO7,
        marks=pytest.mark.xfail(
            strict=True,
            reason="Repeated engine builds alternate between top-k drift and parity.",
        ),
    ),
    pytest.param(
        _YOLO9_E2E,
        marks=pytest.mark.xfail(
            strict=True,
            reason="Trained-weight public top-k class membership drifts.",
        ),
    ),
    pytest.param(
        _YOLO9_P2,
        marks=pytest.mark.xfail(
            strict=True,
            reason="The permissive transfer fixture changes public top-k classes.",
        ),
    ),
    pytest.param(
        _YOLOX,
        marks=pytest.mark.xfail(
            strict=True,
            reason="Trained raw error is 1.6% and signal is only 2.1x the error.",
        ),
    ),
    pytest.param(
        _RTDETR,
        marks=pytest.mark.xfail(
            strict=True,
            reason="Trained raw outputs drift 17-38% after TensorRT conversion.",
        ),
    ),
    pytest.param(
        _YOLONAS_DETECT,
        marks=pytest.mark.xfail(
            strict=True,
            reason="Synthetic output signal is only 4-5x conversion error.",
        ),
    ),
    pytest.param(
        _YOLONAS_POSE,
        marks=pytest.mark.xfail(
            strict=True,
            reason="Synthetic output signal is only 2-6x conversion error.",
        ),
    ),
)


def _build_yolo9_p2_transfer():
    from libreyolo import LibreYOLO9P2
    from libreyolo.utils.download import download_weights

    model = LibreYOLO9P2(None, size="t", device="cuda")
    weights_path = Path(model._resolve_weights_path("LibreYOLO9t.pt"))
    if not weights_path.exists():
        download_weights(str(weights_path), model.size)
    with weights_path.open("rb") as weights_file:
        digest = hashlib.file_digest(weights_file, "sha256").hexdigest()
    assert digest == "b4d7e93f9e0393830fb42e6135c0e3464b2673b05e5ecf4b7f2374ec18e39eb2"
    model._load_transfer_weights(weights_path)
    return model


def _build_synthetic_yolonas_detect(imgsz: int):
    from libreyolo import LibreYOLONAS
    from libreyolo.models.yolonas.loss import PPYoloELoss

    model = LibreYOLONAS(None, size="s", nb_classes=2, device="cuda")
    network = model.model.train()
    loss_fn = PPYoloELoss(num_classes=2).cuda()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

    for step in range(12):
        images = torch.rand(2, 3, imgsz, imgsz, device="cuda")
        targets = torch.zeros(2, 10, 5, device="cuda")
        targets[0, 0] = torch.tensor(
            [float(step % 2), 36.0 + step, 42.0, 24.0, 30.0],
            device="cuda",
        )
        targets[1, 0] = torch.tensor(
            [float((step + 1) % 2), 64.0, 52.0, 20.0, 26.0],
            device="cuda",
        )
        loss, _ = loss_fn(network(images), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    network.eval()
    with torch.no_grad():
        for head in (
            network.heads.head1,
            network.heads.head2,
            network.heads.head3,
        ):
            head.reg_pred.weight.mul_(20.0)
    return model


def _build_synthetic_yolonas_pose(imgsz: int):
    from libreyolo import LibreYOLONAS
    from libreyolo.data import default_oks_sigmas
    from libreyolo.models.yolonas.loss import YoloNASPoseLoss

    model = LibreYOLONAS(None, size="n", task="pose", device="cuda")
    network = model.model.train()
    keypoints = model.num_keypoints
    loss_fn = YoloNASPoseLoss(
        oks_sigmas=default_oks_sigmas(keypoints),
    ).cuda()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

    for step in range(8):
        images = torch.rand(2, 3, imgsz, imgsz, device="cuda")
        targets = torch.zeros(
            2,
            4,
            5 + 3 * keypoints,
            device="cuda",
        )
        for batch_index in range(2):
            cx = 40.0 + 8.0 * batch_index + float(step % 3)
            cy = 44.0 + 4.0 * batch_index
            targets[batch_index, 0, 1:5] = torch.tensor(
                [cx, cy, 30.0, 36.0],
                device="cuda",
            )
            for keypoint_index in range(keypoints):
                offset = (keypoint_index - keypoints / 2.0) * 0.8
                start = 5 + 3 * keypoint_index
                targets[batch_index, 0, start : start + 3] = torch.tensor(
                    [cx + offset, cy + offset, 2.0],
                    device="cuda",
                )
        loss, _ = loss_fn(network(images), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    network.eval()
    with torch.no_grad():
        for head in (
            network.heads.head1,
            network.heads.head2,
            network.heads.head3,
        ):
            head.reg_pred.weight.mul_(20.0)
            head.pose_pred.weight.mul_(20.0)
    return model


def _build_model(case: TensorRTRound11Case):
    if case.weights:
        from libreyolo import LibreYOLO

        return LibreYOLO(case.weights, device="cuda")
    if case.fixture == "yolo9_transfer":
        return _build_yolo9_p2_transfer()
    if case.fixture == "synthetic_detect":
        return _build_synthetic_yolonas_detect(case.imgsz)
    if case.fixture == "synthetic_pose":
        return _build_synthetic_yolonas_pose(case.imgsz)
    raise AssertionError(f"Round 11 case has no fixture: {case}")


def _align_outputs(
    expected_outputs: list[np.ndarray],
    actual_outputs: list[np.ndarray],
) -> list[np.ndarray]:
    """Restore backend binding order using the unique output shapes."""
    assert len(actual_outputs) == len(expected_outputs)
    remaining = list(range(len(actual_outputs)))
    aligned = []
    for expected_index, expected in enumerate(expected_outputs):
        candidates = [
            index for index in remaining if actual_outputs[index].shape == expected.shape
        ]
        if len(candidates) == 1:
            selected = candidates[0]
        else:
            assert expected_index in candidates
            selected = expected_index
        aligned.append(actual_outputs[selected])
        remaining.remove(selected)
    return aligned


def _match_detection_rows(expected_data, actual_data):
    expected_classes = expected_data[:, 5].astype(np.int64)
    actual_classes = actual_data[:, 5].astype(np.int64)
    assert sorted(expected_classes.tolist()) == sorted(actual_classes.tolist())
    matches = []
    for class_id in np.unique(expected_classes):
        expected_indices = np.flatnonzero(expected_classes == class_id)
        actual_indices = np.flatnonzero(actual_classes == class_id)
        expected_rows = expected_data[expected_indices]
        actual_rows = actual_data[actual_indices]
        coordinate_cost = np.max(
            np.abs(expected_rows[:, None, :4] - actual_rows[None, :, :4]),
            axis=2,
        )
        row_indices, column_indices = linear_sum_assignment(coordinate_cost)
        ious = _pairwise_box_iou(expected_rows[:, :4], actual_rows[:, :4])
        for row_index, column_index in zip(row_indices, column_indices):
            expected_global = int(expected_indices[row_index])
            actual_global = int(actual_indices[column_index])
            coordinate_error = float(coordinate_cost[row_index, column_index])
            assert (
                float(ious[row_index, column_index]) > 0.95
                or coordinate_error < 1e-2
            )
            assert (
                abs(
                    float(expected_rows[row_index, 4])
                    - float(actual_rows[column_index, 4])
                )
                < 0.01
            )
            matches.append((expected_global, actual_global))
    return matches


def _assert_predict_parity(case, native_model, backend, image) -> None:
    expected = native_model.predict(
        image,
        imgsz=case.imgsz,
        conf=0.0,
        max_det=25,
    )
    actual = backend.predict(image, conf=0.0, max_det=25)
    assert expected.boxes is not None
    assert actual.boxes is not None
    expected_data = expected.boxes.data.detach().float().cpu().numpy()
    actual_data = actual.boxes.data.detach().float().cpu().numpy()
    assert len(expected_data) == len(actual_data)
    assert len(expected_data) > 0
    matches = _match_detection_rows(expected_data, actual_data)

    if case.task != "pose":
        return
    assert expected.keypoints is not None
    assert actual.keypoints is not None
    expected_keypoints = expected.keypoints.data.detach().float().cpu().numpy()
    actual_keypoints = actual.keypoints.data.detach().float().cpu().numpy()
    for expected_index, actual_index in matches:
        coordinate_l2 = np.linalg.norm(
            expected_keypoints[expected_index, :, :2]
            - actual_keypoints[actual_index, :, :2],
            axis=1,
        )
        assert float(np.max(coordinate_l2)) < 2.0
        if expected_keypoints.shape[-1] > 2:
            confidence_error = np.max(
                np.abs(
                    expected_keypoints[expected_index, :, 2]
                    - actual_keypoints[actual_index, :, 2]
                )
            )
            assert float(confidence_error) < 0.01


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(value.astype(np.float64) ** 2)))


def _native_probe_outputs(model, first, second, imgsz):
    if model.FAMILY != "yolonas":
        return (
            _raw_outputs(model, first, imgsz),
            _raw_outputs(model, second, imgsz),
        )

    traced = torch.jit.trace(model.model, first)
    with torch.inference_mode():
        first_output = traced(first)
        second_output = traced(second)
    if isinstance(first_output, torch.Tensor):
        first_output = (first_output,)
        second_output = (second_output,)
    return (
        [value.detach().float().cpu().numpy() for value in first_output],
        [value.detach().float().cpu().numpy() for value in second_output],
    )


def _run_tensorrt_case(tmp_path, case: TensorRTRound11Case) -> None:
    from libreyolo import LibreYOLO

    torch.manual_seed(11)
    model = _build_model(case)
    model.model.eval()
    assert model.FAMILY == case.family
    assert model.task == case.task

    first = torch.zeros(1, 3, case.imgsz, case.imgsz, device="cuda")
    second = (
        torch.rand(1, 3, case.imgsz, case.imgsz, device="cuda")
        if case.family == "yolonas"
        else torch.ones(1, 3, case.imgsz, case.imgsz, device="cuda")
    )
    expected_first, expected_second = _native_probe_outputs(
        model,
        first,
        second,
        case.imgsz,
    )

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
    raw_metrics = []
    for expected_a, expected_b, actual_a, actual_b in zip(
        expected_first,
        expected_second,
        actual_first,
        actual_second,
    ):
        expected_signal = _rms(expected_a - expected_b)
        actual_signal = _rms(actual_a - actual_b)
        parity_error = _rms(expected_a - actual_a)
        reference_scale = max(_rms(expected_a), 1e-6)
        raw_metrics.append(
            (
                parity_error / reference_scale,
                expected_signal / reference_scale,
                actual_signal / max(parity_error, 1e-12),
            )
        )

    assert backend.model_family == case.family
    assert backend.task == case.task
    assert backend.imgsz == case.imgsz
    _assert_predict_parity(case, model, backend, _image(case.imgsz))
    for relative_error, relative_signal, signal_to_error in raw_metrics:
        assert relative_error < 1e-3
        assert relative_signal > 1e-6
        assert signal_to_error > 20.0
    if case.repeatability_hold:
        pytest.fail(
            "Repeated TensorRT builds alternate between public top-k drift and parity."
        )

    del backend, model
    gc.collect()
    torch.cuda.empty_cache()


@requires_tensorrt
@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    ROUND11_VALIDATED_CASES,
    ids=lambda case: f"{case.family}-{case.task}",
)
def test_tensorrt_round11_raw_and_predict_parity(tmp_path, case):
    _run_tensorrt_case(tmp_path, case)


@requires_tensorrt
@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    ROUND11_EXPERIMENTAL_CASES,
    ids=lambda case: f"{case.family}-{case.task}",
)
def test_tensorrt_round11_measured_experimental(tmp_path, case):
    _run_tensorrt_case(tmp_path, case)
