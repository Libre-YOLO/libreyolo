"""CenterNet export wrapper and exported-backend contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


def _backend_stub():
    from libreyolo.backends.base import BaseBackend

    class StubBackend(BaseBackend):
        def _run_inference(self, blob):
            del blob
            return []

    backend = object.__new__(StubBackend)
    backend.model_family = "centernet"
    backend.model_size = "resdcn18"
    backend.imgsz = 128
    backend.input_size = 128
    backend.nb_classes = 2
    backend.task = "detect"
    return backend


def test_export_wrapper_bakes_decode_and_stride_scaling():
    from libreyolo.models.centernet.nn import CenterNetExportWrapper

    class FakeCenterNet(torch.nn.Module):
        def forward(self, images):
            batch = images.shape[0]
            heatmap = images.new_full((batch, 2, 32, 32), -20.0)
            width_height = images.new_zeros((batch, 2, 32, 32))
            regression = images.new_zeros((batch, 2, 32, 32))
            heatmap[:, 1, 16, 10] = 20.0
            width_height[:, :, 16, 10] = images.new_tensor([4.0, 6.0])
            regression[:, :, 16, 10] = images.new_tensor([0.25, 0.5])
            return {"hm": heatmap, "wh": width_height, "reg": regression}

    output = CenterNetExportWrapper(FakeCenterNet(), topk=10)(
        torch.zeros(1, 3, 128, 128)
    )
    assert output.shape == (1, 10, 6)
    torch.testing.assert_close(
        output[0, 0], torch.tensor([33.0, 54.0, 49.0, 78.0, 1.0, 1.0])
    )


def test_backend_preprocess_and_parser_match_native_helpers():
    from libreyolo.models.centernet.utils import preprocess_numpy
    from libreyolo.validation.preprocessors import CenterNetValPreprocessor

    backend = _backend_stub()
    image = np.random.default_rng(637).integers(0, 256, (50, 100, 3), dtype=np.uint8)
    tensor, original, original_size, ratio = backend._preprocess_centernet(
        Image.fromarray(image), 128, "rgb"
    )
    expected, expected_ratio = preprocess_numpy(image, 128)
    np.testing.assert_array_equal(tensor.numpy()[0], expected)
    assert original.size == original_size == (100, 50)
    assert ratio == expected_ratio
    assert isinstance(backend._get_val_preprocessor(), CenterNetValPreprocessor)

    decoded = np.array(
        [[[33.0, 54.0, 49.0, 78.0, 0.9, 1.0], [0, 0, 0, 0, 0.1, 0]]],
        dtype=np.float32,
    )
    boxes, scores, classes = backend._parse_centernet(
        [decoded], 128, 100, 50, conf=0.5, max_det=100
    )
    np.testing.assert_allclose(boxes[0], [25.78125, 17.1875, 38.28125, 35.9375])
    np.testing.assert_allclose(scores, [0.9])
    np.testing.assert_array_equal(classes, [1])


def test_export_context_uses_a_portable_copy_and_restores_live_model():
    from libreyolo import LibreCenterNet
    from libreyolo.export.exporter import OnnxExporter
    from libreyolo.models.centernet.nn import DCN

    model = LibreCenterNet(None, size="resdcn18", nb_classes=3, device="cpu")
    live_dcns = [module for module in model.model.modules() if isinstance(module, DCN)]
    assert live_dcns and not any(module.portable for module in live_dcns)
    with OnnxExporter(model)._model_context("cpu", False, False, 1, (64, 64)) as (
        wrapped,
        dummy,
    ):
        wrapped_dcns = [
            module for module in wrapped.modules() if isinstance(module, DCN)
        ]
        assert wrapped_dcns and all(module.portable for module in wrapped_dcns)
        with torch.no_grad():
            assert wrapped(dummy).shape == (1, 100, 6)
    assert not any(module.portable for module in live_dcns)


def test_export_support_and_early_ncnn_block():
    from libreyolo import LibreCenterNet
    from libreyolo.backends.base import _is_nms_free_family
    from libreyolo.export.support import get_support

    assert _is_nms_free_family("centernet") is True
    assert get_support("centernet", "detect", "onnx").tier == "validated"
    assert get_support("centernet", "detect", "torchscript").tier == "validated"
    ncnn = get_support("centernet", "detect", "ncnn")
    assert ncnn.tier == "blocked"
    assert "NCNN" in ncnn.reason

    model = LibreCenterNet(None, size="resdcn18", nb_classes=3, device="cpu")
    with pytest.raises(NotImplementedError, match="NCNN"):
        model.export(format="ncnn", imgsz=64)
    with pytest.raises(NotImplementedError, match="opset 16"):
        model.export(format="onnx", imgsz=64, opset=15)
