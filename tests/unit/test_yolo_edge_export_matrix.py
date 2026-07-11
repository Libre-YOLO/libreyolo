"""Raw-output parity for YOLO-family NCNN exports."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("class_name", "size"),
    [
        ("LibreYOLO9E2E", "t"),
        ("LibreYOLO9P2", "t"),
        ("LibreYOLOX", "n"),
    ],
)
@pytest.mark.parametrize("format", ["onnx", "torchscript", "ncnn"])
def test_yolo_raw_parity(tmp_path, class_name, size, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "ncnn" and (
        importlib.util.find_spec("pnnx") is None
        or importlib.util.find_spec("ncnn") is None
    ):
        pytest.skip("PNNX and NCNN are required")

    import libreyolo

    torch.manual_seed(0)
    model = getattr(libreyolo, class_name)(None, size=size, nb_classes=3, device="cpu")
    model.model.eval()
    tensor = torch.rand(1, 3, 64, 64)
    model.model.head.export = True
    with torch.no_grad():
        native = model.model(tensor)
    model.model.head.export = False
    if isinstance(native, (tuple, list)):
        native = native[0]
    native = native.numpy()

    artifact = model.export(
        format=format,
        imgsz=64,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / f"{class_name}.{format}"),
    )
    actual = libreyolo.LibreYOLO(artifact, device="cpu")._run_inference(tensor.numpy())[
        0
    ]
    np.testing.assert_allclose(actual, native, rtol=1e-3, atol=1e-3)
