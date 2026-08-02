"""Round 24 parity probes for embedding, depth, and normal export gaps."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from .test_export_round23 import _assert_raw_parity, _outputs
from .test_tensorrt_round11 import _align_outputs

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
]


@dataclass(frozen=True)
class Round24Case:
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
    elif format in {"ncnn", "tflite"}:
        marks.append(pytest.mark.slow)
    hold_reasons = {
        ("dinov2", "openvino"): "embedding elements miss strict tolerance",
        ("dinov2", "tensorrt"): "embedding elements miss strict tolerance",
        ("moge2", "ncnn"): "raw signal is only 4.5x conversion error",
        ("moge2", "tflite"): "cubic Resize retains dynamic C/H/W",
    }
    if (family, format) in hold_reasons:
        marks.append(
            pytest.mark.xfail(
                strict=True,
                reason=hold_reasons[(family, format)],
            )
        )
    case = Round24Case(family, task, format, imgsz)
    return pytest.param(case, marks=marks, id=f"{family}-{task}-{format}")


ROUND24_CASES = (
    _case("dinov2", "embed", "onnx", 224),
    _case("dinov2", "embed", "torchscript", 224),
    _case("dinov2", "embed", "openvino", 224),
    _case("dinov2", "embed", "tensorrt", 224),
    _case("depth_anything3", "depth", "onnx", 56),
    _case("depth_anything3", "depth", "torchscript", 56),
    _case("depth_anything3", "depth", "openvino", 56),
    _case("depth_anything3", "depth", "tensorrt", 56),
    _case("moge2", "normal", "ncnn", 28),
    _case("moge2", "normal", "tflite", 28),
)


def _build_model(case: Round24Case, device: str):
    from libreyolo import LibreDepthAnything3, LibreDINOv2, LibreMoGe2

    if case.family == "dinov2":
        return LibreDINOv2(None, size="n", task="embed", device=device)
    if case.family == "depth_anything3":
        return LibreDepthAnything3(None, size="l", device=device)
    return LibreMoGe2(None, size="s", task="normal", device=device)


def _native_outputs(case, model, tensor, device):
    from libreyolo.export.exporter import BaseExporter

    with (
        BaseExporter.create(case.format, model)._model_context(
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


def _assert_public_parity(case: Round24Case, native, runtime):
    image = np.random.default_rng(24).integers(
        0,
        256,
        size=(case.imgsz, case.imgsz, 3),
        dtype=np.uint8,
    )
    expected = native.predict(image, imgsz=case.imgsz)
    actual = runtime.predict(image)
    if case.task == "embed":
        torch.testing.assert_close(
            actual.embeddings.data.cpu(),
            expected.embeddings.data.cpu(),
            rtol=2e-3,
            atol=5e-3,
        )
        torch.testing.assert_close(
            actual.embeddings.data.norm(dim=1).cpu(),
            torch.ones(1),
            rtol=0.0,
            atol=1e-5,
        )
        return
    if case.task == "depth":
        expected_depth = expected.depth_map.data.cpu().numpy()
        actual_depth = actual.depth_map.data.cpu().numpy()
        peak = max(float(np.max(np.abs(expected_depth))), 1e-6)
        mse = float(
            np.mean(
                np.square(
                    expected_depth.astype(np.float64)
                    - actual_depth.astype(np.float64)
                )
            )
        )
        psnr = float("inf") if mse == 0.0 else 20.0 * np.log10(peak / np.sqrt(mse))
        assert psnr > 40.0
        return

    expected_normal = expected.normal_map.data.cpu().numpy().astype(np.float64)
    actual_normal = actual.normal_map.data.cpu().numpy().astype(np.float64)
    dots = np.sum(expected_normal * actual_normal, axis=-1)
    angular = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))
    assert float(np.mean(angular)) < 0.1


@pytest.mark.parametrize("case", ROUND24_CASES)
def test_round24_export_predict_parity(tmp_path, monkeypatch, case):
    if case.format == "openvino":
        pytest.importorskip("openvino")
    elif case.format == "tensorrt":
        pytest.importorskip("tensorrt")
        if not torch.cuda.is_available():
            pytest.skip("TensorRT parity requires CUDA")
    elif case.format == "ncnn":
        pytest.importorskip("ncnn")
    elif case.format == "tflite" and (
        importlib.util.find_spec("onnx2tf") is None
        or importlib.util.find_spec("ai_edge_litert") is None
    ):
        pytest.skip("onnx2tf and ai-edge-litert are required")

    from libreyolo import LibreYOLO
    from libreyolo.export.support import SUPPORT, SupportEntry, get_support

    if get_support(case.family, case.task, case.format).tier == "blocked":
        monkeypatch.setitem(
            SUPPORT,
            (case.family, case.task, case.format),
            SupportEntry("experimental", "Round 24 measured block probe."),
        )
    torch.manual_seed(21 if case.family == "depth_anything3" else 24)
    device = torch.device("cuda" if case.format == "tensorrt" else "cpu")
    model = _build_model(case, str(device))
    model.model.eval()
    if case.family == "depth_anything3":
        first = torch.rand(
            1,
            3,
            case.imgsz,
            case.imgsz,
            generator=torch.Generator().manual_seed(21),
        )
        second = torch.rand(
            1,
            3,
            case.imgsz,
            case.imgsz,
            generator=torch.Generator().manual_seed(22),
        )
    else:
        first = torch.rand(
            1,
            3,
            case.imgsz,
            case.imgsz,
            generator=torch.Generator().manual_seed(24),
        )
        second = 1.0 - first
    expected_first = _native_outputs(case, model, first, device)
    expected_second = _native_outputs(case, model, second, device)

    suffix = {
        "onnx": ".onnx",
        "torchscript": ".torchscript",
        "openvino": "_openvino",
        "tensorrt": ".engine",
        "ncnn": "_ncnn",
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
    actual_first = _align_outputs(
        expected_first,
        runtime._run_inference(first.numpy()),
    )
    actual_second = _align_outputs(
        expected_second,
        runtime._run_inference(second.numpy()),
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
