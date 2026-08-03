"""Full trained-checkpoint VGG export-backend acceptance gates."""

from __future__ import annotations

import gc

import pytest
import torch

pytestmark = [
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.vgg,
]


@pytest.fixture(autouse=True)
def _release_export_runtime_memory():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _official_vgg16():
    from torchvision.models import VGG16_Weights

    from libreyolo import LibreVGG

    model = LibreVGG(size="16", nb_classes=1000, device="cpu")
    model.model.load_state_dict(
        VGG16_Weights.IMAGENET1K_V1.get_state_dict(
            progress=True,
            check_hash=True,
        ),
        strict=True,
    )
    model.model.eval()
    return model


def _assert_public_probability_parity(expected, actual):
    assert expected.probs is not None
    assert actual.probs is not None
    assert actual.probs.top1 == expected.probs.top1
    cosine = torch.nn.functional.cosine_similarity(
        expected.probs.data,
        actual.probs.data,
        dim=0,
    )
    assert float(cosine) > 0.999


@pytest.mark.onnx
def test_official_vgg16_onnx_backend_matches_native(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from libreyolo import SAMPLE_IMAGE
    from libreyolo.backends.onnx import OnnxBackend

    model = _official_vgg16()
    expected = model.predict(SAMPLE_IMAGE)[0]

    exported = model.export(
        format="onnx",
        imgsz=224,
        dynamic=False,
        simplify=False,
        half=False,
        output_path=str(tmp_path / "LibreVGG16-cls.onnx"),
    )
    backend = OnnxBackend(exported, device="cpu")
    actual = backend.predict(SAMPLE_IMAGE)[0]
    _assert_public_probability_parity(expected, actual)


@pytest.mark.torchscript
def test_official_vgg16_torchscript_backend_matches_native(tmp_path):
    from libreyolo import LibreYOLO, SAMPLE_IMAGE

    model = _official_vgg16()
    expected = model.predict(SAMPLE_IMAGE)[0]
    exported = model.export(
        format="torchscript",
        imgsz=224,
        half=False,
        output_path=str(tmp_path / "LibreVGG16-cls.torchscript"),
    )
    backend = LibreYOLO(exported, device="cpu")
    actual = backend.predict(SAMPLE_IMAGE)[0]
    _assert_public_probability_parity(expected, actual)


@pytest.mark.openvino
def test_official_vgg16_openvino_backend_matches_native(tmp_path):
    pytest.importorskip("openvino")

    from libreyolo import LibreYOLO, SAMPLE_IMAGE

    model = _official_vgg16()
    expected = model.predict(SAMPLE_IMAGE)[0]
    exported = model.export(
        format="openvino",
        imgsz=224,
        half=False,
        output_path=str(tmp_path / "LibreVGG16-cls_openvino"),
    )
    backend = LibreYOLO(exported, device="cpu")
    actual = backend.predict(SAMPLE_IMAGE)[0]
    _assert_public_probability_parity(expected, actual)


@pytest.mark.tensorrt
def test_official_vgg16_tensorrt_backend_matches_native(tmp_path):
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT parity requires CUDA")

    from libreyolo import LibreYOLO, SAMPLE_IMAGE

    model = _official_vgg16()
    expected = model.predict(SAMPLE_IMAGE)[0]
    exported = model.export(
        format="tensorrt",
        imgsz=224,
        dynamic=False,
        simplify=False,
        half=False,
        output_path=str(tmp_path / "LibreVGG16-cls.engine"),
    )
    backend = LibreYOLO(exported, device="cuda")
    actual = backend.predict(SAMPLE_IMAGE)[0]
    _assert_public_probability_parity(expected, actual)
