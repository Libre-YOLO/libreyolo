"""Deterministic raw-output parity for CPU DETR-family exports."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

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

# LibreDFINE ONNX parity lands at 284/300 matching rows (0.9467) on the newer
# Apple-silicon host class behind macos-latest while the older class stays
# above the 0.95 bar: the same SHA failed (attempt 1) and passed (attempt 2)
# in run 29708218768. Host-dependent kernel dispatch reorders near-tied
# top-300-of-320 queries; not a code regression. LibreEC hit the same mode
# (21d8a16e), durably fixed later with query alignment (db877031). Non-strict
# because the outcome depends on which host class the job draws.
_MACOS_DFINE_ONNX_DRIFT = pytest.mark.xfail(
    sys.platform == "darwin",
    strict=False,
    reason="macOS host-class kernel dispatch drifts LibreDFINE ONNX row match below 0.95",
)


def _export_cases():
    for format in ("onnx", "torchscript"):
        for class_name, size, imgsz in _DETR_CASES:
            marks = ()
            if format == "onnx" and class_name in _ONNX_PARITY_GAPS:
                marks = pytest.mark.xfail(
                    strict=True, reason=_ONNX_PARITY_GAPS[class_name]
                )
            elif format == "onnx" and class_name == "LibreDFINE":
                marks = _MACOS_DFINE_ONNX_DRIFT
            yield pytest.param(
                class_name,
                size,
                imgsz,
                format,
                marks=marks,
                id=f"{format}-{class_name}",
            )


def _align_query_outputs(actual, expected):
    """Align unordered DETR queries using their predicted geometry."""
    assert len(actual) > 1, "query alignment requires a geometric output"
    assert actual[1].shape == expected[1].shape
    aligned = [np.empty_like(output) for output in actual]
    for batch_index, (actual_geometry, expected_geometry) in enumerate(
        zip(actual[1], expected[1])
    ):
        actual_vectors = actual_geometry.reshape(actual_geometry.shape[0], -1)
        expected_vectors = expected_geometry.reshape(expected_geometry.shape[0], -1)
        cost = np.square(actual_vectors[:, None] - expected_vectors[None, :]).sum(
            axis=-1
        )
        actual_indices, expected_indices = linear_sum_assignment(cost)
        actual_order = actual_indices[np.argsort(expected_indices)]
        for aligned_output, actual_output in zip(aligned, actual):
            aligned_output[batch_index] = actual_output[batch_index, actual_order]
    return tuple(aligned)


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
    expected_outputs = tuple(output.detach().cpu().numpy() for output in native)
    if format == "onnx" and class_name == "LibreEC":
        actual = _align_query_outputs(actual, expected_outputs)
    rtol, atol = (2e-3, 2e-2) if format == "onnx" else (1e-3, 1e-3)
    for actual_output, expected in zip(actual, expected_outputs):
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


_TASK_HEAD_CASES = (
    ("LibreDFINE", "n", "segment", 256),
    ("LibreEC", "s", "pose", 640),
    ("LibreEC", "s", "segment", 640),
)


@pytest.mark.parametrize(
    ("class_name", "size", "task", "imgsz"),
    _TASK_HEAD_CASES,
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
    expected_outputs = tuple(output.detach().cpu().numpy() for output in native)
    if format == "onnx" and class_name == "LibreEC":
        actual = _align_query_outputs(actual, expected_outputs)
    rtol, atol = (2e-3, 2e-2) if format == "onnx" else (1e-3, 1e-3)
    for actual_output, expected in zip(actual, expected_outputs):
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
