"""Focused contract tests for BiRefNet Core ML export."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from libreyolo.export.coreml_birefnet import (
    BIREFNET_COREML_CANVAS,
    BIREFNET_COREML_DEFORM_CONV_MERGE,
    BIREFNET_COREML_SIZES,
    _has_deform_conv_lowering,
    birefnet_coreml_input_contract,
    birefnet_coreml_output_contract,
    birefnet_coreml_validation_contract,
    require_birefnet_coreml_lowering,
    validate_birefnet_coreml_profile,
)
from libreyolo.models.birefnet.utils import preprocess_numpy

pytestmark = pytest.mark.unit


class _Registry:
    def __init__(self, value) -> None:
        self.value = value

    def get_func(self, name: str):
        assert name == "torchvision::deform_conv2d"
        return self.value


class _Identity(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image


def test_birefnet_coreml_contract_is_exact_and_registered():
    from libreyolo.export import coreml

    assert BIREFNET_COREML_CANVAS == 1024
    assert BIREFNET_COREML_SIZES == {"t", "l"}
    assert birefnet_coreml_input_contract() == {
        "name": "image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "stretch",
        "interpolation": "bilinear",
        "resize_backend": "pillow",
        "pad_value": 0,
    }
    assert birefnet_coreml_output_contract() == [
        {"name": "matte", "role": "matte_logits"}
    ]
    assert birefnet_coreml_validation_contract() == {
        "color": "rgb",
        "range": "imagenet",
    }
    assert ("birefnet", "matte") in coreml.supported_coreml_exports()
    assert coreml._input_contract("birefnet", "matte", "l") == (
        birefnet_coreml_input_contract()
    )
    assert coreml._output_contract("birefnet", "matte", nms=False) == (
        birefnet_coreml_output_contract()
    )
    assert coreml._validation_contract("birefnet", "matte") == (
        birefnet_coreml_validation_contract()
    )


@pytest.mark.parametrize("size", ["t", "l", " T ", " L "])
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_birefnet_profile_accepts_only_conversion_proven_shapes(size, precision):
    assert validate_birefnet_coreml_profile(
        size=size,
        precision=precision,
        canvas_hw=(1024, 1024),
    ) == (1024, 1024)


@pytest.mark.parametrize("size", [None, "", "s", "m", "x"])
def test_birefnet_profile_rejects_unknown_sizes(size):
    with pytest.raises(NotImplementedError, match="only size='t' or size='l'"):
        validate_birefnet_coreml_profile(
            size=size,
            precision="fp32",
            canvas_hw=1024,
        )


@pytest.mark.parametrize("canvas", [512, 1000, (1024, 1000), (1000, 1024)])
def test_birefnet_profile_rejects_non_native_canvas(canvas):
    with pytest.raises(NotImplementedError, match="1024x1024"):
        validate_birefnet_coreml_profile(
            size="l",
            precision="fp32",
            canvas_hw=canvas,
        )


def test_birefnet_feature_gate_checks_function_and_registry_entry():
    absent_ops = SimpleNamespace()
    present_ops = SimpleNamespace(torchvision_deform_conv2d=object())

    assert not _has_deform_conv_lowering(absent_ops, _Registry(object()))
    assert not _has_deform_conv_lowering(present_ops, _Registry(None))
    assert _has_deform_conv_lowering(present_ops, _Registry(object()))


def test_birefnet_feature_gate_error_is_actionable(monkeypatch):
    from libreyolo.export import coreml_birefnet

    monkeypatch.setattr(
        coreml_birefnet,
        "has_birefnet_coreml_lowering",
        lambda: False,
    )
    with pytest.raises(NotImplementedError) as caught:
        require_birefnet_coreml_lowering(SimpleNamespace(__version__="9.0"))

    message = str(caught.value)
    assert "torchvision::deform_conv2d" in message
    assert "coremltools 9.0" in message
    assert BIREFNET_COREML_DEFORM_CONV_MERGE in message


def test_birefnet_feature_gate_accepts_registered_lowering(monkeypatch):
    from libreyolo.export import coreml_birefnet

    monkeypatch.setattr(
        coreml_birefnet,
        "has_birefnet_coreml_lowering",
        lambda: True,
    )
    require_birefnet_coreml_lowering(SimpleNamespace(__version__="9.0"))


def test_birefnet_image_type_and_graph_normalization_match_native_exactly():
    from libreyolo.export.coreml import _ImageNetPreprocess

    height, width = 37, 53
    rgb = (
        np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3)
        % 256
    ).astype(np.uint8)
    native, ratio = preprocess_numpy(rgb, BIREFNET_COREML_CANVAS)
    resized = np.asarray(
        Image.fromarray(rgb, mode="RGB").resize(
            (BIREFNET_COREML_CANVAS, BIREFNET_COREML_CANVAS),
            Image.BILINEAR,
        ),
        dtype=np.uint8,
    )
    image_type = (
        torch.from_numpy(resized.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
    )
    wrapped = _ImageNetPreprocess(_Identity()).eval()

    with torch.inference_mode():
        prepared = wrapped(image_type)[0].numpy()

    assert ratio == 1.0
    np.testing.assert_array_equal(prepared, native)


def test_birefnet_output_semantics_pin_raw_full_canvas_logits():
    from libreyolo.export.coreml import (
        _output_contract,
        _validate_output_semantics,
    )

    outputs = _output_contract("birefnet", "matte", nms=False)
    valid = torch.zeros(1, 1, 1024, 1024)
    _validate_output_semantics(
        outputs,
        [valid],
        family="birefnet",
        task="matte",
        nc=1,
        input_hw=(1024, 1024),
        size="l",
        nms=False,
        metadata={},
    )

    for invalid in (
        torch.zeros(1, 2, 1024, 1024),
        torch.zeros(1, 1, 512, 512),
        torch.full((1, 1, 1024, 1024), float("nan")),
    ):
        with pytest.raises(RuntimeError, match="Core ML output contract violation"):
            _validate_output_semantics(
                outputs,
                [invalid],
                family="birefnet",
                task="matte",
                nc=1,
                input_hw=(1024, 1024),
                size="l",
                nms=False,
                metadata={},
            )


def test_birefnet_strict_loader_rejects_tampered_canvas_or_matte_shape():
    from libreyolo.backends.coreml import CoreMLBackend, _parse_io_contract

    def parsed(shape):
        return _parse_io_contract(
            {
                "coreml_io": {
                    "input": birefnet_coreml_input_contract(),
                    "validation": birefnet_coreml_validation_contract(),
                    "outputs": [
                        {
                            **birefnet_coreml_output_contract()[0],
                            "rank": 4,
                            "dtype": "float32",
                            "shape": list(shape),
                        }
                    ],
                }
            }
        )

    CoreMLBackend._validate_strict_profile(
        parsed((1, 1, 1024, 1024)),
        family="birefnet",
        task="matte",
        size="l",
        imgsz=(1024, 1024),
        has_embedded_nms=False,
        io_schema_version="2",
        nc=1,
    )
    with pytest.raises(ValueError, match="matte output must match"):
        CoreMLBackend._validate_strict_profile(
            parsed((1, 1, 512, 512)),
            family="birefnet",
            task="matte",
            size="l",
            imgsz=(1024, 1024),
            has_embedded_nms=False,
            io_schema_version="2",
            nc=1,
        )
    with pytest.raises(NotImplementedError, match="1024x1024"):
        CoreMLBackend._validate_strict_profile(
            parsed((1, 1, 512, 512)),
            family="birefnet",
            task="matte",
            size="l",
            imgsz=(512, 512),
            has_embedded_nms=False,
            io_schema_version="2",
            nc=1,
        )
