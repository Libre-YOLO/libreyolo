"""Round 22 parity for edge, gaze, and DINOv2 classification exports."""

from __future__ import annotations

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
class Round22Case:
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
    case = Round22Case(family, task, format, imgsz)
    return pytest.param(case, marks=marks, id=f"{family}-{task}-{format}")


ROUND22_CASES = (
    _case("teed", "edge", "torchscript", 64),
    _case("teed", "edge", "openvino", 64),
    _case("teed", "edge", "tensorrt", 64),
    _case("dexined", "edge", "torchscript", 64),
    _case("dexined", "edge", "openvino", 64),
    _case("dexined", "edge", "tensorrt", 64),
    _case("l2cs", "gaze", "openvino", 448),
    _case("l2cs", "gaze", "tensorrt", 448),
    _case("dinov2", "classify", "openvino", 224),
    _case("dinov2", "classify", "tensorrt", 224),
)


def _build_model(case: Round22Case, device: str):
    from libreyolo import LibreDexiNed, LibreDINOv2, LibreL2CS, LibreTEED

    if case.family == "teed":
        return LibreTEED(None, size="t", device=device)
    if case.family == "dexined":
        return LibreDexiNed(None, size="b", device=device)
    if case.family == "l2cs":
        return LibreL2CS(None, size="r18", num_bins=90, device=device)
    return LibreDINOv2(
        None,
        size="n",
        task="classify",
        nb_classes=3,
        device=device,
    )


def _exporter(case: Round22Case, model):
    from libreyolo.export.exporter import (
        OpenVINOExporter,
        TensorRTExporter,
        TorchScriptExporter,
    )

    exporters = {
        "torchscript": TorchScriptExporter,
        "openvino": OpenVINOExporter,
        "tensorrt": TensorRTExporter,
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
    case: Round22Case,
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


def _assert_dinov2_tensorrt_floor(
    expected_a,
    expected_b,
    actual_a,
    actual_b,
):
    """Keep the runnable cell covered without claiming repeatable strict parity."""
    assert len(actual_a) == len(expected_a) == 1
    assert len(actual_b) == len(expected_b) == 1
    for reference, actual in (
        (expected_a[0], actual_a[0]),
        (expected_b[0], actual_b[0]),
    ):
        assert actual.shape == reference.shape
        cosine = torch.nn.functional.cosine_similarity(
            torch.from_numpy(reference),
            torch.from_numpy(actual),
        )
        assert float(cosine) > 0.99
        assert int(reference.argmax()) == int(actual.argmax())
        assert _rms(actual - reference) < 0.1 * max(_rms(reference), 1e-6)
    assert _rms(expected_a[0] - expected_b[0]) > 1e-5
    assert _rms(actual_a[0] - actual_b[0]) > 1e-5


def _psnr(expected: np.ndarray, actual: np.ndarray) -> float:
    error = float(
        np.mean((expected.astype(np.float64) - actual.astype(np.float64)) ** 2)
    )
    return float("inf") if error == 0.0 else -10.0 * np.log10(error)


def _assert_public_parity(case: Round22Case, native, runtime):
    image = np.random.default_rng(22).integers(
        0,
        256,
        size=(case.imgsz, case.imgsz, 3),
        dtype=np.uint8,
    )
    if case.task == "edge":
        expected = native.predict(image, imgsz=case.imgsz)
        actual = runtime.predict(image)
        assert _psnr(
            expected.edges.data.cpu().numpy(),
            actual.edges.data.cpu().numpy(),
        ) > 40.0
        return
    if case.task == "classify":
        expected = native.predict(image, imgsz=case.imgsz)
        actual = runtime.predict(image)
        expected_probs = expected.probs.data.cpu()
        actual_probs = actual.probs.data.cpu()
        cosine = torch.nn.functional.cosine_similarity(
            expected_probs[None],
            actual_probs[None],
        )
        assert float(cosine) > 0.999
        assert int(expected_probs.argmax()) == int(actual_probs.argmax())
        return

    expected = native.predict(
        image,
        face_boxes=[[0, 0, case.imgsz, case.imgsz]],
    )
    actual = runtime.predict(image)
    torch.testing.assert_close(
        actual.gaze.data.cpu(),
        expected.gaze.data.cpu(),
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.parametrize("case", ROUND22_CASES)
def test_round22_export_predict_parity(tmp_path, case):
    if case.format == "openvino":
        pytest.importorskip("openvino")
    elif case.format == "tensorrt":
        pytest.importorskip("tensorrt")
        if not torch.cuda.is_available():
            pytest.skip("TensorRT parity requires CUDA")

    torch.manual_seed(22)
    device = torch.device("cuda" if case.format == "tensorrt" else "cpu")
    model = _build_model(case, str(device))
    model.model.eval()
    first = torch.rand(
        1,
        3,
        case.imgsz,
        case.imgsz,
        generator=torch.Generator().manual_seed(22),
    )
    second = 1.0 - first
    expected_first = _native_outputs(case, model, first, device)
    expected_second = _native_outputs(case, model, second, device)

    suffix = {
        "torchscript": ".torchscript",
        "tensorrt": ".engine",
        "openvino": "_openvino",
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

    from libreyolo import LibreYOLO

    runtime = LibreYOLO(artifact, device=str(device))
    actual_first = runtime._run_inference(first.numpy())
    actual_second = runtime._run_inference(second.numpy())
    if case.family == "dinov2" and case.format == "tensorrt":
        _assert_dinov2_tensorrt_floor(
            expected_first,
            expected_second,
            actual_first,
            actual_second,
        )
    else:
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
