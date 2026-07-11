"""Deterministic raw-output parity for CPU DETR-family exports."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

_DETR_CASES = (
    ("LibreDFINE", "n", 256),
    ("LibreDEIM", "n", 640),
    ("LibreDEIMv2", "atto", 320),
    ("LibreEC", "s", 640),
    ("LibreRTDETR", "r18", 640),
    ("LibreRTDETRv2", "r18", 640),
    ("LibreRTDETRv4", "s", 640),
)
_ONNX_PARITY_GAPS = {
    "LibreDEIM": "8.7% of selected boxes exceed the ONNX tolerance",
    "LibreDEIMv2": "ONNX top-k selection causes score and box drift",
    "LibreRTDETRv2": "9% of selected boxes exceed the ONNX tolerance",
    "LibreRTDETRv4": "7.3% of selected boxes exceed the ONNX tolerance",
}


def _export_cases():
    for format in ("onnx", "torchscript"):
        for class_name, size, imgsz in _DETR_CASES:
            marks = ()
            if format == "onnx" and class_name in _ONNX_PARITY_GAPS:
                marks = pytest.mark.xfail(
                    strict=True, reason=_ONNX_PARITY_GAPS[class_name]
                )
            yield pytest.param(
                class_name,
                size,
                imgsz,
                format,
                marks=marks,
                id=f"{format}-{class_name}",
            )


@pytest.mark.parametrize(
    ("class_name", "size", "imgsz", "format"),
    _export_cases(),
)
def test_detr_detect_raw_parity(tmp_path, class_name, size, imgsz, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    import libreyolo
    from libreyolo.export.exporter import OnnxExporter

    torch.manual_seed(0)
    model = getattr(libreyolo, class_name)(None, size=size, nb_classes=3, device="cpu")
    model.model.eval()
    tensor = torch.rand(1, 3, imgsz, imgsz)
    exporter = OnnxExporter(model)
    with exporter._model_context("cpu", False, False, 1, (imgsz, imgsz)) as (
        wrapped,
        _,
    ):
        with torch.no_grad():
            native = wrapped(tensor)
    if isinstance(native, torch.Tensor):
        native = (native,)

    artifact = model.export(
        format=format,
        imgsz=imgsz,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / f"{class_name}.{format}"),
    )
    actual = libreyolo.LibreYOLO(artifact, device="cpu")._run_inference(tensor.numpy())

    assert len(actual) == len(native)
    rtol, atol = (2e-3, 2e-2) if format == "onnx" else (1e-3, 1e-3)
    for actual_output, native_output in zip(actual, native):
        expected = native_output.detach().cpu().numpy()
        if format == "onnx" and expected.shape[-1] == 4:
            row_match = np.isclose(actual_output, expected, rtol=rtol, atol=atol).all(
                axis=-1
            )
            assert float(row_match.mean()) > 0.95
            continue
        if format == "onnx":
            element_match = np.isclose(actual_output, expected, rtol=rtol, atol=atol)
            assert float(element_match.mean()) > 0.95
            continue
        np.testing.assert_allclose(
            actual_output,
            expected,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    ("class_name", "size", "task", "imgsz"),
    [
        ("LibreDFINE", "n", "segment", 256),
        ("LibreEC", "s", "pose", 640),
        ("LibreEC", "s", "segment", 640),
    ],
)
@pytest.mark.parametrize("format", ["onnx", "torchscript"])
def test_detr_task_head_raw_parity(tmp_path, class_name, size, task, imgsz, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    import libreyolo
    from libreyolo.export.exporter import OnnxExporter

    torch.manual_seed(0)
    model = getattr(libreyolo, class_name)(
        None, size=size, nb_classes=3, device="cpu", task=task
    )
    model.model.eval()
    tensor = torch.rand(1, 3, imgsz, imgsz)
    exporter = OnnxExporter(model)
    with exporter._model_context("cpu", False, False, 1, (imgsz, imgsz)) as (
        wrapped,
        _,
    ):
        with torch.no_grad():
            native = wrapped(tensor)
    if isinstance(native, torch.Tensor):
        native = (native,)

    artifact = model.export(
        format=format,
        imgsz=imgsz,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / f"{class_name}-{task}.{format}"),
    )
    actual = libreyolo.LibreYOLO(artifact, device="cpu")._run_inference(tensor.numpy())

    assert len(actual) == len(native)
    rtol, atol = (2e-3, 2e-2) if format == "onnx" else (1e-3, 1e-3)
    for actual_output, native_output in zip(actual, native):
        expected = native_output.detach().cpu().numpy()
        if format == "onnx":
            element_match = np.isclose(actual_output, expected, rtol=rtol, atol=atol)
            assert float(element_match.mean()) > 0.95
            continue
        np.testing.assert_allclose(
            actual_output,
            expected,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    ("size", "task", "imgsz", "nb_classes"),
    [
        ("n", "segment", 312, 3),
        ("n", "obb", 384, 3),
        ("x", "pose", 576, 1),
    ],
)
@pytest.mark.parametrize("format", ["onnx", "torchscript"])
def test_rfdetr_task_raw_parity(tmp_path, size, task, imgsz, nb_classes, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    import libreyolo
    from libreyolo.export.exporter import OnnxExporter

    torch.manual_seed(0)
    model = libreyolo.LibreRFDETR(
        {}, size=size, nb_classes=nb_classes, device="cpu", task=task
    )
    model.model.eval()
    tensor = torch.rand(1, 3, imgsz, imgsz)
    exporter = OnnxExporter(model)
    with exporter._model_context("cpu", False, False, 1, (imgsz, imgsz)) as (
        wrapped,
        _,
    ):
        with torch.no_grad():
            native = wrapped(tensor)
    if isinstance(native, torch.Tensor):
        native = (native,)

    artifact = model.export(
        format=format,
        imgsz=imgsz,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / f"LibreRFDETR-{task}.{format}"),
    )
    actual = libreyolo.LibreYOLO(artifact, device="cpu")._run_inference(tensor.numpy())

    assert len(actual) == len(native)
    rtol, atol = (2e-3, 2e-2) if format == "onnx" else (1e-3, 1e-3)
    for actual_output, native_output in zip(actual, native):
        expected = native_output.detach().cpu().numpy()
        if format == "onnx":
            element_match = np.isclose(actual_output, expected, rtol=rtol, atol=atol)
            assert float(element_match.mean()) > 0.95
            continue
        np.testing.assert_allclose(
            actual_output,
            expected,
            rtol=rtol,
            atol=atol,
        )
