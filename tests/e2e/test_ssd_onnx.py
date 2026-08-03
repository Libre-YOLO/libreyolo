"""Official SSD300 checkpoint parity through the ONNX Runtime backend."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.supported_backend,
    pytest.mark.onnx,
    pytest.mark.ssd,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("onnx") is None
        or importlib.util.find_spec("onnxruntime") is None,
        reason="onnx and onnxruntime are required",
    ),
]


def test_official_ssd300_onnx_raw_and_predict_parity(tmp_path):
    import onnx

    from libreyolo import LibreYOLO
    from libreyolo.models.ssd.nn import SSDExportWrapper

    model = LibreYOLO("LibreSSD300.pt", device="cpu")
    image = "tests/fixtures/dog.jpg"
    input_tensor = model._preprocess(image)[0]
    with torch.inference_mode():
        expected_raw = SSDExportWrapper(model.model)(input_tensor).numpy()

    artifact = model.export(
        format="onnx",
        output_path=str(tmp_path / "LibreSSD300.onnx"),
        imgsz=300,
        dynamic=False,
        simplify=False,
        device="cpu",
        opset=13,
    )
    graph = onnx.load(artifact)
    onnx.checker.check_model(graph)
    assert [output.name for output in graph.graph.output] == ["output"]

    backend = LibreYOLO(artifact, device="cpu")
    actual_raw = backend._run_inference(input_tensor.numpy())[0]
    assert actual_raw.shape == expected_raw.shape == (1, 84, 8732)
    raw_error = np.abs(actual_raw - expected_raw)
    assert float(raw_error.max()) < 1e-3
    assert float(raw_error.mean()) < 1e-6

    native = model.predict(image, conf=0.01, iou=0.45, max_det=200)
    converted = backend.predict(image, conf=0.01, iou=0.45, max_det=200)
    native_boxes = native.boxes.data.numpy()
    converted_boxes = converted.boxes.data.numpy()
    assert native_boxes.shape == converted_boxes.shape
    assert len(native_boxes) > 0
    np.testing.assert_array_equal(converted_boxes[:, 5], native_boxes[:, 5])
    np.testing.assert_allclose(
        converted_boxes[:, :4], native_boxes[:, :4], rtol=1e-5, atol=1e-3
    )
    np.testing.assert_allclose(
        converted_boxes[:, 4], native_boxes[:, 4], rtol=1e-5, atol=1e-5
    )
