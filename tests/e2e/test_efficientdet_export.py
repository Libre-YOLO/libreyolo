"""Trained-checkpoint EfficientDet export/runtime parity.

Set ``LIBREYOLO_EFFICIENTDET_CONVERTED_DIR`` to a directory containing
``LibreEfficientDetd0.pt``. These probes build real artifacts and are kept out
of the PR gate by the external-data marker.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from libreyolo import LibreYOLO, SAMPLE_IMAGE

CONVERTED_DIR = os.environ.get("LIBREYOLO_EFFICIENTDET_CONVERTED_DIR")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.external_data,
    pytest.mark.efficientdet,
    pytest.mark.export_backend,
    pytest.mark.supported_backend,
    pytest.mark.skipif(
        not CONVERTED_DIR,
        reason="set LIBREYOLO_EFFICIENTDET_CONVERTED_DIR for export parity",
    ),
]


def _case(format_name: str):
    marks = [getattr(pytest.mark, format_name)]
    if format_name == "tensorrt":
        marks.append(pytest.mark.trt)
    return pytest.param(format_name, marks=marks, id=format_name)


@pytest.mark.parametrize(
    "format_name",
    tuple(_case(name) for name in ("onnx", "torchscript", "openvino", "tensorrt")),
)
def test_efficientdet_d0_export_predict_parity(tmp_path: Path, format_name: str):
    if format_name == "openvino":
        pytest.importorskip("openvino")
    elif format_name == "tensorrt":
        pytest.importorskip("tensorrt")
        if not torch.cuda.is_available():
            pytest.skip("TensorRT parity requires CUDA")
    elif format_name == "onnx" and importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("ONNX Runtime is required")

    suffix = {
        "onnx": ".onnx",
        "torchscript": ".torchscript",
        "openvino": "_openvino",
        "tensorrt": ".engine",
    }[format_name]
    device = "cuda" if format_name == "tensorrt" else "cpu"
    checkpoint = Path(CONVERTED_DIR) / "LibreEfficientDetd0.pt"
    model = LibreYOLO(str(checkpoint), device=device)
    artifact = model.export(
        format=format_name,
        output_path=str(tmp_path / f"LibreEfficientDetd0{suffix}"),
        imgsz=512,
        half=False,
        dynamic=False,
        simplify=False,
        opset=17,
        workspace=1.0,
    )
    runtime = LibreYOLO(artifact, device=device)

    input_tensor, _, _, _ = runtime._preprocess(SAMPLE_IMAGE, 512, "auto")
    raw_outputs = runtime._run_inference(input_tensor.numpy())
    assert len(raw_outputs) == 1
    expected_candidates = 3840 if format_name == "tensorrt" else 5000
    assert tuple(raw_outputs[0].shape) == (1, expected_candidates, 6)

    expected = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.5, max_det=100)
    actual = runtime.predict(SAMPLE_IMAGE, conf=0.3, iou=0.5, max_det=100)
    assert runtime.model_family == "efficientdet"
    assert runtime.model_size == "d0"
    assert runtime.task == "detect"
    assert runtime.nb_classes == 80
    assert len(actual.boxes) == len(expected.boxes)
    assert torch.equal(actual.boxes.cls.cpu(), expected.boxes.cls.cpu())

    box_atol = {
        "torchscript": 0.0,
        "onnx": 1e-3,
        "openvino": 0.5,
        "tensorrt": 0.1,
    }[format_name]
    score_atol = {
        "torchscript": 0.0,
        "onnx": 2e-5,
        "openvino": 1e-3,
        "tensorrt": 5e-4,
    }[format_name]
    torch.testing.assert_close(
        actual.boxes.xyxy.cpu(), expected.boxes.xyxy.cpu(), rtol=0, atol=box_atol
    )
    torch.testing.assert_close(
        actual.boxes.conf.cpu(), expected.boxes.conf.cpu(), rtol=0, atol=score_atol
    )
