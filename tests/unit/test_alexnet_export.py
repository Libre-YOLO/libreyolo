"""AlexNet export runtime parity for classification outputs."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from PIL import Image


pytestmark = [pytest.mark.unit, pytest.mark.alexnet, pytest.mark.export_backend]


def _fixture(device: str = "cpu"):
    from libreyolo import LibreAlexNet

    torch.manual_seed(637)
    model = LibreAlexNet(size="b", nb_classes=7, device=device)
    model.model.eval()
    array = np.random.default_rng(637).integers(
        0, 256, size=(257, 301, 3), dtype=np.uint8
    )
    return model, Image.fromarray(array)


def _assert_public_classifier_parity(native_result, exported_result):
    native = native_result.probs.data.float().cpu()
    exported = exported_result.probs.data.float().cpu()
    cosine = torch.nn.functional.cosine_similarity(native[None], exported[None])
    assert float(cosine) > 0.999
    assert exported_result.probs.top1 == native_result.probs.top1


@pytest.mark.onnx
@pytest.mark.supported_backend
def test_onnx_backend_predict_and_raw_logits_match_native(tmp_path):
    """ONNX preserves both raw logits and the public probabilities contract."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from libreyolo.backends.onnx import OnnxBackend

    model, image = _fixture()
    tensor, *_ = model._preprocess(image)
    with torch.inference_mode():
        native_logits = model.model(tensor).cpu()
    native_result = model.predict(image)

    path = model.export(
        format="onnx",
        output_path=str(tmp_path / "LibreAlexNetb-cls.onnx"),
        imgsz=224,
        half=False,
        dynamic=True,
        simplify=False,
        opset=17,
    )
    backend = OnnxBackend(path, device="cpu")
    exported_result = backend.predict(image)
    exported_logits = torch.from_numpy(backend._run_inference(tensor.numpy())[0])

    assert backend.model_family == "alexnet"
    assert backend.model_size == "b"
    assert backend.task == "classify"
    assert backend.imgsz == 224
    torch.testing.assert_close(exported_logits, native_logits, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        exported_result.probs.data,
        native_result.probs.data,
        rtol=1e-4,
        atol=1e-5,
    )
    assert exported_result.probs.top1 == native_result.probs.top1


@pytest.mark.torchscript
@pytest.mark.supported_backend
def test_torchscript_public_predict_matches_native(tmp_path):
    from libreyolo import LibreYOLO

    model, image = _fixture()
    native_result = model.predict(image)

    path = model.export(
        format="torchscript",
        output_path=str(tmp_path / "LibreAlexNetb-cls.torchscript"),
        imgsz=224,
        half=False,
    )
    exported_result = LibreYOLO(path, device="cpu").predict(image)

    torch.testing.assert_close(
        exported_result.probs.data,
        native_result.probs.data,
        rtol=1e-5,
        atol=1e-6,
    )
    assert exported_result.probs.top1 == native_result.probs.top1


@pytest.mark.openvino
@pytest.mark.supported_backend
def test_openvino_public_predict_matches_native(tmp_path):
    pytest.importorskip("openvino")
    from libreyolo import LibreYOLO

    model, image = _fixture()
    native_result = model.predict(image)

    path = model.export(
        format="openvino",
        output_path=str(tmp_path / "LibreAlexNetb-cls_openvino"),
        imgsz=224,
        half=False,
        dynamic=False,
        simplify=False,
        opset=17,
    )
    exported_result = LibreYOLO(path, device="cpu").predict(image)

    _assert_public_classifier_parity(native_result, exported_result)


@pytest.mark.tensorrt
@pytest.mark.trt
@pytest.mark.supported_backend
def test_tensorrt_fp32_public_predict_matches_native(tmp_path):
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT runtime parity requires CUDA")
    from libreyolo import LibreYOLO

    model, image = _fixture(device="cuda")
    native_result = model.predict(image)

    path = model.export(
        format="tensorrt",
        output_path=str(tmp_path / "LibreAlexNetb-cls.engine"),
        imgsz=224,
        half=False,
        int8=False,
        dynamic=False,
        simplify=False,
        opset=17,
    )
    exported_result = LibreYOLO(path, device="cuda").predict(image)

    _assert_public_classifier_parity(native_result, exported_result)


@pytest.mark.external_data
@pytest.mark.slow
@pytest.mark.supported_backend
@pytest.mark.skipif(
    not os.environ.get("LIBREYOLO_ALEXNET_CKPT"),
    reason="Set LIBREYOLO_ALEXNET_CKPT to LibreAlexNetb-cls.pt.",
)
@pytest.mark.parametrize(
    "format",
    [
        pytest.param("onnx", marks=pytest.mark.onnx),
        pytest.param("torchscript", marks=pytest.mark.torchscript),
        pytest.param("openvino", marks=pytest.mark.openvino),
        pytest.param("tensorrt", marks=[pytest.mark.tensorrt, pytest.mark.trt]),
    ],
)
def test_official_checkpoint_export_preserves_ordered_top5(tmp_path, format):
    """Reproduce the trained-checkpoint acceptance gate for every claimed runtime."""
    if format == "openvino":
        pytest.importorskip("openvino")
    elif format == "tensorrt":
        pytest.importorskip("tensorrt")
        if not torch.cuda.is_available():
            pytest.skip("TensorRT runtime parity requires CUDA")

    from libreyolo import LibreYOLO, SAMPLE_IMAGE

    device = "cuda" if format == "tensorrt" else "cpu"
    model = LibreYOLO(os.environ["LIBREYOLO_ALEXNET_CKPT"], device=device)
    native_result = model.predict(SAMPLE_IMAGE)
    output = {
        "onnx": tmp_path / "LibreAlexNetb-cls.onnx",
        "torchscript": tmp_path / "LibreAlexNetb-cls.torchscript",
        "openvino": tmp_path / "LibreAlexNetb-cls_openvino",
        "tensorrt": tmp_path / "LibreAlexNetb-cls.engine",
    }[format]
    export_kwargs = {
        "format": format,
        "output_path": str(output),
        "imgsz": 224,
        "half": False,
    }
    if format in {"onnx", "openvino", "tensorrt"}:
        export_kwargs.update(dynamic=False, simplify=False, opset=17)
    if format == "tensorrt":
        export_kwargs["int8"] = False

    path = model.export(**export_kwargs)
    exported_result = LibreYOLO(path, device=device).predict(SAMPLE_IMAGE)

    _assert_public_classifier_parity(native_result, exported_result)
    assert exported_result.probs.top5 == native_result.probs.top5
