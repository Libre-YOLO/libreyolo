"""Deterministic raw-output parity for CPU DETR-family exports."""

from __future__ import annotations

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
    "LibreDEIMv2": "only 43.7% of aligned score values meet tolerance",
}
_OPENVINO_PARITY_GAPS = {
    "LibreDEIM": "exactly 95% of aligned boxes meet tolerance; validation requires more than 95%",
    "LibreDEIMv2": "42.3% of aligned scores meet the converted-runtime tolerance",
    "LibreRTDETRv2": "92.3% of aligned boxes meet the converted-runtime tolerance",
}
# Families whose ONNX raw outputs are compared after Hungarian query
# alignment. Their graphs select queries with an in-graph top-k over the
# near-uniform scores of this test's random-init weights, so tiny host-class
# float drift (macOS CI vs Linux) reorders a handful of near-tied queries.
# Alignment removes that ordering sensitivity while keeping the numeric
# tolerance intact — a genuinely wrong box still fails after alignment.
_ONNX_QUERY_ALIGNED = {
    "LibreDFINE",
    "LibreDEIM",
    "LibreDEIMv2",
    "LibreEC",
    "LibreRTDETRv2",
    "LibreRTDETRv4",
}
_ONNX_PREDICT_PARITY = {
    "LibreDEIM",
    "LibreRTDETRv2",
    "LibreRTDETRv4",
}


def _export_cases():
    for format in ("onnx", "torchscript", "openvino"):
        for class_name, size, imgsz in _DETR_CASES:
            marks = ()
            if format == "onnx" and class_name in _ONNX_PARITY_GAPS:
                marks = pytest.mark.xfail(
                    strict=True, reason=_ONNX_PARITY_GAPS[class_name]
                )
            if format == "openvino" and class_name in _OPENVINO_PARITY_GAPS:
                marks = pytest.mark.xfail(
                    strict=True, reason=_OPENVINO_PARITY_GAPS[class_name]
                )
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


def _assert_detect_predict_parity(native_result, converted_result):
    native = native_result.boxes.data.cpu().numpy()
    converted = converted_result.boxes.data.cpu().numpy()
    assert converted.shape == native.shape
    if native.shape[0] == 0:
        return

    cost = np.square(
        converted[:, None, :4] - native[None, :, :4]
    ).sum(axis=-1)
    converted_indices, native_indices = linear_sum_assignment(cost)
    converted_order = converted_indices[np.argsort(native_indices)]
    converted = converted[converted_order]
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


@pytest.mark.parametrize(
    ("class_name", "size", "imgsz", "format"),
    _export_cases(),
)
def test_detr_detect_raw_parity(tmp_path, class_name, size, imgsz, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "openvino":
        pytest.importorskip("openvino")

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
    ), torch.no_grad():
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
    backend = libreyolo.LibreYOLO(artifact, device="cpu")
    actual = backend._run_inference(tensor.numpy())

    assert len(actual) == len(native)
    expected_outputs = tuple(output.detach().cpu().numpy() for output in native)
    if format == "openvino" or (
        format == "onnx" and class_name in _ONNX_QUERY_ALIGNED
    ):
        actual = _align_query_outputs(actual, expected_outputs)
    converted = format in {"onnx", "openvino"}
    rtol, atol = (2e-3, 2e-2) if converted else (1e-3, 1e-3)
    for actual_output, expected in zip(actual, expected_outputs):
        if converted and expected.shape[-1] == 4:
            row_match = np.isclose(actual_output, expected, rtol=rtol, atol=atol).all(
                axis=-1
            )
            match_rate = float(row_match.mean())
            assert match_rate > 0.95, f"box row match rate: {match_rate:.4f}"
            continue
        if converted:
            element_match = np.isclose(actual_output, expected, rtol=rtol, atol=atol)
            match_rate = float(element_match.mean())
            assert match_rate > 0.95, f"element match rate: {match_rate:.4f}"
            continue
        np.testing.assert_allclose(
            actual_output,
            expected,
            rtol=rtol,
            atol=atol,
        )
    if format == "openvino" or (
        format == "onnx" and class_name in _ONNX_PREDICT_PARITY
    ):
        image = np.random.default_rng(51).integers(
            0, 256, size=(72, 96, 3), dtype=np.uint8
        )
        result = backend.predict(image, conf=0.0, max_det=100)
        assert result.boxes is not None and result.orig_shape == (72, 96)
        if format == "onnx":
            native_result = model.predict(
                image,
                imgsz=imgsz,
                conf=0.0,
                max_det=100,
            )
            _assert_detect_predict_parity(native_result, result)


_TASK_HEAD_CASES = (
    ("LibreDFINE", "n", "segment", 256),
    ("LibreEC", "s", "pose", 640),
    ("LibreEC", "s", "segment", 640),
)


def _task_head_export_cases():
    for format in ("onnx", "torchscript", "openvino"):
        for class_name, size, task, imgsz in _TASK_HEAD_CASES:
            marks = ()
            if format == "openvino" and class_name == "LibreEC" and task == "pose":
                marks = pytest.mark.xfail(
                    strict=True,
                    reason="93.92% of aligned pose values meet tolerance",
                )
            yield pytest.param(
                class_name,
                size,
                task,
                imgsz,
                format,
                marks=marks,
                id=f"{format}-{class_name}-{task}",
            )


@pytest.mark.parametrize(
    ("class_name", "size", "task", "imgsz", "format"),
    _task_head_export_cases(),
)
def test_detr_task_head_raw_parity(tmp_path, class_name, size, task, imgsz, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "openvino":
        pytest.importorskip("openvino")

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
    ), torch.no_grad():
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
    backend = libreyolo.LibreYOLO(artifact, device="cpu")
    actual = backend._run_inference(tensor.numpy())

    assert len(actual) == len(native)
    expected_outputs = tuple(output.detach().cpu().numpy() for output in native)
    if format in {"onnx", "openvino"} and class_name == "LibreEC":
        actual = _align_query_outputs(actual, expected_outputs)
    converted = format in {"onnx", "openvino"}
    rtol, atol = (2e-3, 2e-2) if converted else (1e-3, 1e-3)
    for actual_output, expected in zip(actual, expected_outputs):
        if converted:
            element_match = np.isclose(actual_output, expected, rtol=rtol, atol=atol)
            assert float(element_match.mean()) > 0.95
            continue
        np.testing.assert_allclose(
            actual_output,
            expected,
            rtol=rtol,
            atol=atol,
        )
    if format == "openvino":
        image = np.random.default_rng(52).integers(
            0, 256, size=(72, 96, 3), dtype=np.uint8
        )
        result = backend.predict(image, conf=0.99)
        assert result.orig_shape == (72, 96)


_RFDETR_TASK_CASES = (
    ("n", "segment", 312, 3),
    ("n", "obb", 384, 3),
    ("x", "pose", 576, 1),
)
_RFDETR_OPENVINO_GAPS = {
    "segment": "87% of aligned box values meet tolerance",
    "obb": "86.17% of aligned box values meet tolerance",
    "pose": "79% of aligned box values meet tolerance",
}


def _rfdetr_task_export_cases():
    for format in ("onnx", "torchscript", "openvino"):
        for size, task, imgsz, nb_classes in _RFDETR_TASK_CASES:
            marks = ()
            if format == "openvino":
                marks = pytest.mark.xfail(
                    strict=True,
                    reason=_RFDETR_OPENVINO_GAPS[task],
                )
            yield pytest.param(
                size,
                task,
                imgsz,
                nb_classes,
                format,
                marks=marks,
                id=f"{format}-{size}-{task}",
            )


@pytest.mark.parametrize(
    ("size", "task", "imgsz", "nb_classes", "format"),
    _rfdetr_task_export_cases(),
)
def test_rfdetr_task_raw_parity(tmp_path, size, task, imgsz, nb_classes, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "openvino":
        pytest.importorskip("openvino")

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
    ), torch.no_grad():
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
    backend = libreyolo.LibreYOLO(artifact, device="cpu")
    actual = backend._run_inference(tensor.numpy())

    assert len(actual) == len(native)
    expected_outputs = tuple(output.detach().cpu().numpy() for output in native)
    if format == "openvino":
        actual = _align_query_outputs(actual, expected_outputs)
    converted = format in {"onnx", "openvino"}
    rtol, atol = (2e-3, 2e-2) if converted else (1e-3, 1e-3)
    for actual_output, expected in zip(actual, expected_outputs):
        if converted:
            element_match = np.isclose(actual_output, expected, rtol=rtol, atol=atol)
            assert float(element_match.mean()) > 0.95
            continue
        np.testing.assert_allclose(
            actual_output,
            expected,
            rtol=rtol,
            atol=atol,
        )
    if format == "openvino":
        image = np.random.default_rng(53).integers(
            0, 256, size=(72, 96, 3), dtype=np.uint8
        )
        result = backend.predict(image, conf=0.99)
        assert result.orig_shape == (72, 96)
