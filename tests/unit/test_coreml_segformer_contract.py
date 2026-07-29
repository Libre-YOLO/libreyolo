from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image


pytestmark = pytest.mark.unit


def _parsed_input_contract():
    from libreyolo.backends.coreml import _parse_io_contract
    from libreyolo.export.coreml_segformer import (
        segformer_coreml_input_contract,
        segformer_coreml_output_contract,
        segformer_coreml_validation_contract,
    )

    return _parse_io_contract(
        {
            "coreml_io": {
                "input": segformer_coreml_input_contract(),
                "outputs": segformer_coreml_output_contract(),
                "validation": segformer_coreml_validation_contract(),
            }
        }
    ).input


def _probe(height: int = 64, width: int = 64) -> torch.Tensor:
    ys = torch.linspace(0.05, 0.95, height).view(1, 1, height, 1)
    xs = torch.linspace(0.1, 0.9, width).view(1, 1, 1, width)
    red = xs.expand(1, 1, height, width)
    green = ys.expand(1, 1, height, width)
    blue = (0.65 * red + 0.35 * green).clamp(0.0, 1.0)
    return torch.cat((red, green, blue), dim=1).contiguous()


def test_segformer_coreml_contract_is_exact_and_dense():
    from libreyolo.export import coreml
    from libreyolo.export.coreml_segformer import (
        SEGFORMER_COREML_ALIGN_CORNERS,
        segformer_coreml_input_contract,
        segformer_coreml_output_contract,
        segformer_coreml_validation_contract,
    )

    assert segformer_coreml_input_contract() == {
        "name": "image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "letterbox_top_left",
        "interpolation": "bilinear",
        "resize_backend": "opencv",
        "resize_rounding": "floor",
        "pad_value": 114,
    }
    assert segformer_coreml_output_contract() == [
        {"name": "semantic_logits", "role": "semantic_logits"}
    ]
    assert segformer_coreml_validation_contract() == {
        "color": "rgb",
        "range": "0_1",
    }
    assert SEGFORMER_COREML_ALIGN_CORNERS is False
    assert ("segformer", "semantic") in coreml.supported_coreml_exports()
    assert coreml._input_contract("segformer", "semantic", "b0") == (
        segformer_coreml_input_contract()
    )
    assert coreml._output_contract("segformer", "semantic", nms=False) == (
        segformer_coreml_output_contract()
    )
    assert coreml._validation_contract("segformer", "semantic") == (
        segformer_coreml_validation_contract()
    )


@pytest.mark.parametrize("original_hw", [(37, 91), (91, 37), (63, 65), (2, 3)])
def test_declared_geometry_matches_native_preprocess_pixel_exact(original_hw):
    from libreyolo.backends.coreml import _apply_geometry
    from libreyolo.models.segformer.model import preprocess_numpy

    original_h, original_w = original_hw
    values = np.arange(original_h * original_w * 3, dtype=np.uint32)
    rgb = (values.reshape(original_h, original_w, 3) * 37 % 256).astype(np.uint8)

    native_chw, native_ratio = preprocess_numpy(rgb, (64, 96))
    transformed = _apply_geometry(
        Image.fromarray(rgb, mode="RGB"),
        input_h=64,
        input_w=96,
        contract=_parsed_input_contract(),
    )
    runtime_chw = (
        np.asarray(transformed.image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    )

    np.testing.assert_array_equal(runtime_chw, native_chw)
    assert transformed.offset_x == 0.0
    assert transformed.offset_y == 0.0
    assert transformed.ratio == pytest.approx(native_ratio, rel=0.0, abs=0.0)


def test_output_inversion_preserves_native_round_after_input_floor():
    from libreyolo.export.coreml_segformer import (
        segformer_letterbox_geometry,
        segformer_valid_logits_hw,
    )

    # Native preprocessing truncates 2 * (64 / 3) to 42 image rows. Native
    # postprocessing deliberately rounds the valid logit window to 43 rows.
    # Preserve the mismatch: changing either side makes exported prediction
    # differ at the image/padding boundary.
    geometry = segformer_letterbox_geometry((2, 3), (64, 64))
    assert geometry.resized_height == 42
    assert geometry.resized_width == 64
    assert segformer_valid_logits_hw((2, 3), (64, 64), (64, 64)) == (43, 64)


def test_backend_output_inversion_matches_native_segformer_postprocess():
    from libreyolo.backends.coreml import CoreMLBackend
    from libreyolo.models.segformer.model import LibreSegformer

    logits = torch.zeros(1, 2, 64, 64)
    logits[:, 0] = 1.0
    logits[:, 1, 42] = 3.0

    native = LibreSegformer.__new__(LibreSegformer)
    native._get_input_size = lambda: 64
    expected = native._postprocess_semantic_logits(
        logits,
        original_size=(3, 2),
        ratio=64 / 3,
        input_size=64,
    ).argmax(dim=1)[0]

    backend = CoreMLBackend.__new__(CoreMLBackend)
    backend.model_family = "segformer"
    actual = backend._parse_semantic_output(
        [logits.numpy()],
        original_size=(3, 2),
        effective_imgsz=64,
        ratio=64 / 3,
    )

    torch.testing.assert_close(actual, expected)


def test_real_b0_eval_graph_traces_with_two_nonconstant_probes():
    from libreyolo.export.coreml_segformer import (
        SEGFORMER_COREML_OUTPUT_NAME,
        segformer_coreml_output_contract,
    )
    from libreyolo.models.segformer.nn import LibreSegformerNet

    torch.manual_seed(19)
    model = LibreSegformerNet(size="b0", num_classes=3).eval()
    first = _probe()
    second = 1.0 - first

    with torch.no_grad():
        eager_first = model(first)
        eager_second = model(second)
        traced = torch.jit.trace(
            model,
            first,
            check_trace=True,
            check_inputs=[(second,)],
        )
        traced_first = traced(first)
        traced_second = traced(second)

    assert segformer_coreml_output_contract()[0]["name"] == SEGFORMER_COREML_OUTPUT_NAME
    assert eager_first.shape == (1, 3, 64, 64)
    assert eager_second.shape == (1, 3, 64, 64)
    torch.testing.assert_close(traced_first, eager_first, rtol=0.0, atol=0.0)
    torch.testing.assert_close(traced_second, eager_second, rtol=0.0, atol=0.0)
    assert float((eager_first - eager_second).abs().max()) > 1e-4


def test_model_exposes_export_route_and_stride(monkeypatch):
    from libreyolo.models.base.model import BaseModel
    from libreyolo.models.segformer.model import LibreSegformer

    calls = []

    def fake_export(self, format="onnx", **kwargs):
        calls.append((self, format, kwargs))
        return "segformer.mlpackage"

    monkeypatch.setattr(BaseModel, "export", fake_export)
    model = LibreSegformer(size="b0", nb_classes=3, device="cpu")

    assert model.IMGSZ_DIVISOR == model.semantic_imgsz_divisor == 32
    assert model.export(format="coreml", dynamic=False) == "segformer.mlpackage"
    assert calls == [(model, "coreml", {"dynamic": False})]
