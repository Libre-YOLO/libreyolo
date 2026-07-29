"""Focused contract and graph-capture tests for SwinIR Core ML export."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from libreyolo import LibreSwinIR
from libreyolo.export.coreml_swinir import (
    SWINIR_COREML_CANVAS,
    SWINIR_COREML_SCALE,
    SWINIR_COREML_SIZES,
    swinir_coreml_input_contract,
    swinir_coreml_output_contract,
    swinir_coreml_validation_contract,
    validate_swinir_coreml_profile,
    wrap_swinir_coreml_contract,
)

pytestmark = pytest.mark.unit


class _BadOutput(nn.Module):
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"restored": image}


def _probes() -> tuple[torch.Tensor, torch.Tensor]:
    count = 3 * SWINIR_COREML_CANVAS * SWINIR_COREML_CANVAS
    values = torch.arange(count, dtype=torch.float32).reshape(
        1,
        3,
        SWINIR_COREML_CANVAS,
        SWINIR_COREML_CANVAS,
    )
    first = values.remainder(251).div(250.0)
    return first, 1.0 - first


def test_swinir_coreml_metadata_contract_is_exact_and_fail_closed():
    from libreyolo.export import coreml

    assert SWINIR_COREML_SIZES == {"s", "m", "l"}
    assert SWINIR_COREML_SCALE == 4
    assert swinir_coreml_input_contract() == {
        "name": "image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "native",
        "interpolation": "bilinear",
        "resize_backend": "pillow",
        "pad_value": 0,
    }
    assert swinir_coreml_output_contract() == [
        {"name": "restored", "role": "restored"}
    ]
    assert swinir_coreml_validation_contract() == {
        "color": "rgb",
        "range": "0_1",
    }
    assert ("swinir", "restore") in coreml.supported_coreml_exports()
    assert coreml._input_contract("swinir", "restore", "s") == (
        swinir_coreml_input_contract()
    )
    assert coreml._output_contract("swinir", "restore", nms=False) == (
        swinir_coreml_output_contract()
    )
    assert coreml._validation_contract("swinir", "restore") == (
        swinir_coreml_validation_contract()
    )


@pytest.mark.parametrize("size", ["s", "m", "l"])
@pytest.mark.parametrize("canvas", [64, (64, 64), [64, 64]])
def test_swinir_coreml_profile_accepts_all_sizes_at_native_64(size, canvas):
    assert validate_swinir_coreml_profile(size=size.upper(), canvas_hw=canvas) == (
        64,
        64,
    )


@pytest.mark.parametrize("size", [None, "", "x"])
def test_swinir_coreml_profile_rejects_unconverted_sizes(size):
    with pytest.raises(NotImplementedError, match="'s', 'm', and 'l'"):
        validate_swinir_coreml_profile(size=size, canvas_hw=64)


@pytest.mark.parametrize("canvas", [0, -1, (0, 64), (64, -1)])
def test_swinir_coreml_profile_rejects_nonpositive_canvas(canvas):
    with pytest.raises(ValueError, match="positive"):
        validate_swinir_coreml_profile(size="s", canvas_hw=canvas)


@pytest.mark.parametrize("canvas", [32, 128, (64, 72), (72, 64)])
def test_swinir_coreml_profile_rejects_unchecked_canvas(canvas):
    with pytest.raises(NotImplementedError, match="64x64"):
        validate_swinir_coreml_profile(size="s", canvas_hw=canvas)


def test_swinir_coreml_profile_rejects_malformed_canvas():
    with pytest.raises(ValueError, match=r"int or \(height, width\)"):
        validate_swinir_coreml_profile(size="s", canvas_hw=(64, 64, 64))


def test_swinir_coreml_adapter_rejects_structured_output():
    adapter = wrap_swinir_coreml_contract(_BadOutput())
    with pytest.raises(RuntimeError, match="one restored tensor"):
        adapter(torch.zeros(1, 3, 64, 64))


def test_swinir_image_type_floats_match_native_photometric_input():
    model = LibreSwinIR(size="s", device="cpu")
    height = width = SWINIR_COREML_CANVAS
    rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    native, _image, original_size, ratio = model._preprocess(Image.fromarray(rgb))
    image_type = (
        torch.from_numpy(rgb.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .div(255.0)
    )

    assert original_size == (width, height)
    assert ratio == 1.0
    torch.testing.assert_close(native, image_type, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_real_swinir_graph_has_exact_two_probe_torchscript_parity(size):
    torch.manual_seed(0)
    network = LibreSwinIR(size=size, device="cpu").model.eval()
    adapter = wrap_swinir_coreml_contract(network)
    first, second = _probes()

    with torch.inference_mode():
        eager_first = adapter(first)
        eager_second = adapter(second)
        traced = torch.jit.trace(
            adapter,
            first,
            check_trace=True,
            check_inputs=[(second,)],
            strict=True,
        )
        traced_first = traced(first)
        traced_second = traced(second)

    expected_shape = (
        1,
        3,
        SWINIR_COREML_CANVAS * SWINIR_COREML_SCALE,
        SWINIR_COREML_CANVAS * SWINIR_COREML_SCALE,
    )
    assert tuple(traced_first.shape) == expected_shape
    assert traced_first.dtype == torch.float32
    torch.testing.assert_close(traced_first, eager_first, rtol=0.0, atol=0.0)
    torch.testing.assert_close(traced_second, eager_second, rtol=0.0, atol=0.0)

    conversion_error = max(
        float((traced_first - eager_first).abs().max()),
        float((traced_second - eager_second).abs().max()),
    )
    sensitivity = float((eager_first - eager_second).abs().max())
    scale = max(float(eager_first.abs().max()), 1e-8)
    assert sensitivity / scale > 1e-6
    assert sensitivity > 100.0 * conversion_error

    operators = set(torch.jit.export_opnames(traced))
    assert "aten::roll" in operators
    assert "aten::index.Tensor" in operators or "aten::index" in operators
    if size == "s":
        assert "aten::pixel_shuffle" in operators
    else:
        assert "aten::upsample_nearest2d.vec" in operators
