"""RTMDet-Ins Core ML raw-output and host-decoding contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.backends.base import BaseBackend
from libreyolo.export.coreml import (
    _RTMDetPreprocess,
    _input_contract,
    _output_contract,
    _validate_output_semantics,
    _validation_contract,
)
from libreyolo.export.coreml_rtmdet_ins import (
    RTMDET_INS_COREML_CONTRACT,
    expected_rtmdet_ins_coreml_shapes,
    rtmdet_ins_coreml_metadata,
    validate_rtmdet_ins_coreml_profile,
)
from libreyolo.models.rtmdet.nn import LibreRTMDetModel
from libreyolo.models.rtmdet.utils import preprocess_numpy
from libreyolo.postprocess.rtmdet import postprocess


pytestmark = pytest.mark.unit


def _gradient_image(width: int, height: int) -> Image.Image:
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (7 * xx + 3 * yy + 11) % 256,
            (5 * xx + 13 * yy + 47) % 256,
            (17 * xx + 19 * yy + 89) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def test_rtmdet_ins_contract_pins_names_shapes_and_host_constants():
    names = [
        output["name"]
        for output in _output_contract("rtmdet", "segment", nms=False)
    ]
    assert names == [
        "class_logits_s8",
        "class_logits_s16",
        "class_logits_s32",
        "box_distances_s8",
        "box_distances_s16",
        "box_distances_s32",
        "dynamic_kernels_s8",
        "dynamic_kernels_s16",
        "dynamic_kernels_s32",
        "mask_features",
    ]
    assert expected_rtmdet_ins_coreml_shapes(nc=3, canvas_hw=(64, 96)) == {
        "class_logits_s8": (1, 3, 8, 12),
        "class_logits_s16": (1, 3, 4, 6),
        "class_logits_s32": (1, 3, 2, 3),
        "box_distances_s8": (1, 4, 8, 12),
        "box_distances_s16": (1, 4, 4, 6),
        "box_distances_s32": (1, 4, 2, 3),
        "dynamic_kernels_s8": (1, 169, 8, 12),
        "dynamic_kernels_s16": (1, 169, 4, 6),
        "dynamic_kernels_s32": (1, 169, 2, 3),
        "mask_features": (1, 8, 8, 12),
    }
    metadata = rtmdet_ins_coreml_metadata()
    assert metadata["rtmdet_ins_contract"] == RTMDET_INS_COREML_CONTRACT
    assert metadata["rtmdet_ins_strides"] == [8, 16, 32]
    assert metadata["rtmdet_ins_num_gen_params"] == 169
    assert metadata["rtmdet_ins_num_prototypes"] == 8
    assert metadata["rtmdet_ins_mask_stride"] == 8
    assert metadata["rtmdet_ins_nms_pre"] == 1000
    assert metadata["rtmdet_ins_max_masks"] == 100
    assert metadata["rtmdet_ins_prior_offset"] == 0
    assert metadata["rtmdet_ins_dynamic_weight_nums"] == [80, 64, 8]
    assert metadata["rtmdet_ins_dynamic_bias_nums"] == [8, 8, 1]
    assert metadata["rtmdet_ins_dyconv_channels"] == 8
    assert metadata["rtmdet_ins_mask_threshold"] == 0.5


def test_rtmdet_ins_uses_canonical_rgb_image_boundary():
    input_contract = _input_contract("rtmdet", "segment", "t")
    assert input_contract == {
        "name": "image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "letterbox_top_left",
        "interpolation": "bilinear",
        "resize_backend": "pillow",
        "pad_value": 114,
    }
    assert _validation_contract("rtmdet", "segment") == {
        "color": "rgb",
        "range": "0_255",
    }


def test_rtmdet_ins_graph_adapter_matches_native_preprocess():
    source = _gradient_image(37, 23)
    native, _ = preprocess_numpy(np.asarray(source), input_size=64)

    ratio = min(64 / source.height, 64 / source.width)
    resized = source.resize(
        (int(source.width * ratio), int(source.height * ratio)),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (64, 64), (114, 114, 114))
    canvas.paste(resized, (0, 0))
    canonical = (
        torch.from_numpy(np.asarray(canvas).copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
    )
    actual = _RTMDetPreprocess(torch.nn.Identity())(canonical)
    torch.testing.assert_close(
        actual,
        torch.from_numpy(native).unsqueeze(0),
        rtol=0.0,
        atol=2e-6,
    )


@pytest.mark.parametrize("size", ["t", "s", "m", "l", "x"])
def test_rtmdet_ins_profile_accepts_all_native_sizes(size):
    validate_rtmdet_ins_coreml_profile(size=size, canvas_hw=(640, 640))


@pytest.mark.parametrize(
    ("size", "canvas", "error"),
    [
        ("unknown", (640, 640), "sizes t/s/m/l/x"),
        ("t", (65, 64), "divisible by 32"),
        ("t", (64, 63), "divisible by 32"),
    ],
)
def test_rtmdet_ins_profile_rejects_unsupported_graph_geometry(
    size,
    canvas,
    error,
):
    with pytest.raises(NotImplementedError, match=error):
        validate_rtmdet_ins_coreml_profile(size=size, canvas_hw=canvas)


def test_rtmdet_ins_real_graph_has_exact_two_probe_trace_parity():
    torch.manual_seed(20260729)
    model = LibreRTMDetModel(
        size="t",
        nc=2,
        enable_mask_head=True,
    ).eval()
    # RTMDet's deliberately tiny fresh-model initialization makes the class
    # and box towers numerically bias-only after a deep random backbone. Use a
    # deterministic non-degenerate fixture so every declared output must react
    # to both trace probes; trained-checkpoint sensitivity is covered on macOS.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.1)
    model.head.export = True
    wrapped = _RTMDetPreprocess(model).eval()

    first = torch.linspace(0.0, 1.0, 64 * 64).reshape(1, 1, 64, 64)
    first = torch.cat((first, first.flip(-1), first.flip(-2)), dim=1)
    second = 1.0 - first
    with torch.no_grad():
        expected_first = wrapped(first)
        expected_second = wrapped(second)
        traced = torch.jit.trace(
            wrapped,
            first,
            check_trace=True,
            check_inputs=[(second,)],
        )
        actual_first = traced(first)
        actual_second = traced(second)

    assert len(actual_first) == len(actual_second) == 10
    sensitivities = []
    for expected, actual, expected_alt, actual_alt in zip(
        expected_first,
        actual_first,
        expected_second,
        actual_second,
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_alt, expected_alt, rtol=0.0, atol=0.0)
        sensitivities.append(float((expected_alt - expected).abs().max()))
    assert all(value > 1e-8 for value in sensitivities)


def test_rtmdet_ins_output_semantics_reject_one_wrong_level_shape():
    outputs = _output_contract("rtmdet", "segment", nms=False)
    shapes = expected_rtmdet_ins_coreml_shapes(nc=2, canvas_hw=(64, 64))
    tensors = [torch.zeros(shapes[output["name"]]) for output in outputs]
    metadata = {"nc": 2, **rtmdet_ins_coreml_metadata()}
    _validate_output_semantics(
        outputs,
        tensors,
        family="rtmdet",
        task="segment",
        nc=2,
        input_hw=(64, 64),
        size="t",
        nms=False,
        metadata=metadata,
    )

    tensors[7] = torch.zeros(1, 169, 5, 4)
    with pytest.raises(RuntimeError, match="dynamic_kernels_s16"):
        _validate_output_semantics(
            outputs,
            tensors,
            family="rtmdet",
            task="segment",
            nc=2,
            input_hw=(64, 64),
            size="t",
            nms=False,
            metadata=metadata,
        )


def test_rtmdet_ins_exported_parser_matches_native_mask_postprocess():
    cls = tuple(torch.full((1, 1, size, size), -20.0) for size in (2, 1, 1))
    reg = tuple(torch.full((1, 4, size, size), 4.0) for size in (2, 1, 1))
    kernels = tuple(torch.zeros(1, 169, size, size) for size in (2, 1, 1))
    cls[0][0, 0, 1, 1] = 10.0
    kernels[0][0, -1, 1, 1] = 10.0
    mask_features = torch.zeros(1, 8, 2, 2)
    nested = (cls, reg, kernels, mask_features)
    native = postprocess(
        nested,
        conf_thres=0.25,
        iou_thres=0.6,
        input_size=16,
        original_size=(24, 12),
        ratio=2 / 3,
        max_det=100,
    )
    flat = [
        value.numpy()
        for value in (*cls, *reg, *kernels, mask_features)
    ]
    parsed = BaseBackend._parse_rtmdet_segment(
        flat,
        16,
        24,
        12,
        0.25,
        0.6,
        100,
        2 / 3,
    )

    np.testing.assert_allclose(parsed[0], native["boxes"].numpy())
    np.testing.assert_allclose(parsed[1], native["scores"].numpy())
    np.testing.assert_array_equal(parsed[2], native["classes"].numpy())
    np.testing.assert_array_equal(parsed[3], native["masks"].numpy())
