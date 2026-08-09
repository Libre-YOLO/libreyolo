"""Classifier parity across CPU-runnable edge export formats."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("class_name", "size"),
    [
        ("LibreMobileNetV4", "s"),
        ("LibreConvNeXt", "t"),
        ("LibreEfficientNetV2", "b0"),
        ("LibreResNet", "18"),
    ],
)
@pytest.mark.parametrize("format", ["torchscript", "openvino", "ncnn", "tflite"])
def test_classifier_edge_export_parity(tmp_path, class_name, size, format):
    if format == "openvino":
        pytest.importorskip("openvino")
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

    torch.manual_seed(0)
    model_cls = getattr(libreyolo, class_name)
    model = model_cls(size=size, nb_classes=5, device="cpu")
    model.model.eval()
    imgsz = model._get_input_size()
    image = np.random.default_rng(23).integers(
        0, 256, size=(imgsz + 16, imgsz + 40, 3), dtype=np.uint8
    )
    native = model.predict(image).probs.data

    output = (
        tmp_path
        / {
            "torchscript": "model.torchscript",
            "openvino": "model_openvino",
            "ncnn": "model_ncnn",
            "tflite": "model.tflite",
        }[format]
    )
    artifact = model.export(
        format=format,
        output_path=str(output),
        imgsz=imgsz,
        dynamic=False,
        simplify=False,
    )
    exported = libreyolo.LibreYOLO(artifact, device="cpu").predict(image).probs.data

    cosine = torch.nn.functional.cosine_similarity(native[None], exported[None])
    assert float(cosine) > 0.999
    assert int(native.argmax()) == int(exported.argmax())
