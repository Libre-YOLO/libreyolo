"""Raw-output parity for Darknet-lineage edge exports."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("class_name", ["LibreYOLO2", "LibreYOLO3", "LibreYOLO4"])
@pytest.mark.parametrize("format", ["onnx", "torchscript", "ncnn"])
def test_darknet_edge_raw_parity(tmp_path, class_name, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "ncnn" and (
        importlib.util.find_spec("pnnx") is None
        or importlib.util.find_spec("ncnn") is None
    ):
        pytest.skip("PNNX and NCNN are required")
    if format == "tflite" and (
        importlib.util.find_spec("onnx2tf") is None
        or importlib.util.find_spec("ai_edge_litert") is None
    ):
        pytest.skip("onnx2tf and ai-edge-litert are required")

    import libreyolo
    from libreyolo.models.darknet.export import DarknetExportWrapper

    torch.manual_seed(0)
    model_cls = getattr(libreyolo, class_name)
    model = model_cls(size="t", nb_classes=3, device="cpu")
    model.model.eval()
    tensor = torch.rand(1, 3, 96, 96)
    wrapper = DarknetExportWrapper(model.model).eval()
    with torch.no_grad():
        native = wrapper(tensor).numpy()

    artifact = model.export(
        format=format,
        imgsz=96,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / f"{class_name}.{format}"),
    )
    actual = libreyolo.LibreYOLO(artifact, device="cpu")._run_inference(tensor.numpy())[
        0
    ]
    np.testing.assert_allclose(actual, native, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("format", ["onnx", "torchscript", "ncnn"])
def test_yolo7_edge_raw_parity(tmp_path, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "ncnn" and (
        importlib.util.find_spec("pnnx") is None
        or importlib.util.find_spec("ncnn") is None
    ):
        pytest.skip("PNNX and NCNN are required")
    if format == "tflite" and (
        importlib.util.find_spec("onnx2tf") is None
        or importlib.util.find_spec("ai_edge_litert") is None
    ):
        pytest.skip("onnx2tf and ai-edge-litert are required")

    import libreyolo
    from libreyolo.models.yolo7.export import YOLO7ExportWrapper

    torch.manual_seed(0)
    model = libreyolo.LibreYOLO7(size="b", nb_classes=3, device="cpu")
    model.model.eval()
    tensor = torch.rand(1, 3, 96, 96)
    wrapper = YOLO7ExportWrapper(model.model).eval()
    with torch.no_grad():
        native = wrapper(tensor).numpy()

    artifact = model.export(
        format=format,
        imgsz=96,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / f"yolo7.{format}"),
    )
    actual = libreyolo.LibreYOLO(artifact, device="cpu")._run_inference(tensor.numpy())[
        0
    ]
    np.testing.assert_allclose(actual, native, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("format", ["onnx", "torchscript", "ncnn"])
def test_yolo1_raw_parity(tmp_path, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "ncnn" and (
        importlib.util.find_spec("pnnx") is None
        or importlib.util.find_spec("ncnn") is None
    ):
        pytest.skip("PNNX and NCNN are required")

    import libreyolo
    from libreyolo.models.darknet.export import DarknetExportWrapper

    torch.manual_seed(0)
    model = libreyolo.LibreYOLO1(None, size="t", nb_classes=20, device="cpu")
    model.model.eval()
    tensor = torch.rand(1, 3, 448, 448)
    wrapper = DarknetExportWrapper(model.model).eval()
    with torch.no_grad():
        native = wrapper(tensor).numpy()

    artifact = model.export(
        format=format,
        imgsz=448,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / f"yolo1.{format}"),
    )
    actual = libreyolo.LibreYOLO(artifact, device="cpu")._run_inference(tensor.numpy())[
        0
    ]
    np.testing.assert_allclose(actual, native, rtol=1e-3, atol=1e-3)
