"""Trained LibreSwin export parity through each advertised runtime."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.external_data,
    pytest.mark.export_backend,
    pytest.mark.slow,
    pytest.mark.swin,
]


def _case(format: str):
    marks = []
    if format == "onnx":
        marks.append(pytest.mark.onnx)
    elif format == "torchscript":
        marks.append(pytest.mark.torchscript)
    elif format == "openvino":
        marks.append(pytest.mark.openvino)
    elif format == "tensorrt":
        marks.extend((pytest.mark.tensorrt, pytest.mark.trt))
    return pytest.param(format, marks=marks, id=format)


@pytest.mark.parametrize(
    "format",
    [_case(name) for name in ("onnx", "torchscript", "openvino", "tensorrt")],
)
def test_trained_swin_t_export_predict_parity(tmp_path, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    elif format == "openvino":
        pytest.importorskip("openvino")
    elif format == "tensorrt":
        pytest.importorskip("tensorrt")
        if not torch.cuda.is_available():
            pytest.skip("TensorRT parity requires CUDA")
    elif importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is required")

    from libreyolo import LibreSwin, LibreYOLO, SAMPLE_IMAGE

    device = "cuda" if format == "tensorrt" else "cpu"
    local_weight = Path("weights/LibreSwint-cls.pt")
    weight = str(local_weight) if local_weight.exists() else "LibreSwint-cls.pt"
    model = LibreSwin(weight, device=device)
    expected = model.predict(SAMPLE_IMAGE).probs.data.cpu()
    suffix = {
        "onnx": ".onnx",
        "torchscript": ".torchscript",
        "openvino": "_openvino",
        "tensorrt": ".engine",
    }[format]
    artifact = model.export(
        format=format,
        output_path=str(tmp_path / f"LibreSwint-cls{suffix}"),
        imgsz=224,
        batch=1,
        dynamic=False,
        half=False,
        simplify=False,
    )

    runtime = LibreYOLO(artifact, device=device)
    actual = runtime.predict(SAMPLE_IMAGE).probs.data.cpu()
    cosine = torch.nn.functional.cosine_similarity(expected[None], actual[None])
    assert float(cosine) > 0.99999
    assert int(actual.argmax()) == int(expected.argmax())
    assert runtime.model_family == "swin"
    assert runtime.task == "classify"
    assert runtime.imgsz == 224
    assert runtime.names == model.names
