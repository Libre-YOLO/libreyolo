"""Unit coverage for the YOLO-NAS Core ML contract fragments."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from libreyolo.export.coreml_yolonas import (
    resolve_yolonas_coreml_geometry,
    wrap_yolonas_coreml_contract,
    yolonas_coreml_input_contract,
    yolonas_coreml_output_contract,
    yolonas_coreml_validation_contract,
)
from libreyolo.models.yolonas.utils import preprocess_numpy
from libreyolo.postprocess.yolonas import (
    YOLO_NAS_POSE_RESIZE_SIZE,
    YOLO_NAS_RESIZE_SIZE,
)

pytestmark = pytest.mark.unit


class _EagerDecodedModel(nn.Module):
    def __init__(self, output_count: int):
        super().__init__()
        self.output_count = output_count
        self.seen = None

    def forward(self, image):
        self.seen = image.detach().clone()
        decoded = tuple(
            image.mean(dim=(-2, -1)) + index for index in range(self.output_count)
        )
        raw = (image[:, :, :1, :1],)
        return decoded, raw


class _TraceDecodedModel(nn.Module):
    def __init__(self, output_count: int):
        super().__init__()
        self.output_count = output_count

    def forward(self, image):
        return tuple(
            image.mean(dim=(-2, -1)) + index for index in range(self.output_count)
        )


def test_detect_adapter_preserves_rgb_and_drops_raw_outputs():
    model = _EagerDecodedModel(2)
    adapter = wrap_yolonas_coreml_contract(model, "detect")
    image = torch.tensor([[[[0.1]], [[0.2]], [[0.3]]]])

    output = adapter(image)

    assert len(output) == 2
    torch.testing.assert_close(model.seen, image)


def test_pose_adapter_converts_canonical_rgb_to_native_bgr():
    model = _EagerDecodedModel(4)
    adapter = wrap_yolonas_coreml_contract(model, "pose")
    image = torch.tensor([[[[0.1]], [[0.2]], [[0.3]]]])

    output = adapter(image)

    assert len(output) == 4
    torch.testing.assert_close(model.seen, image[:, [2, 1, 0]])


@pytest.mark.parametrize(("task", "count"), [("detect", 2), ("pose", 4)])
def test_adapter_traces_with_flat_decoded_output(task, count):
    adapter = wrap_yolonas_coreml_contract(_TraceDecodedModel(count), task)
    probe = torch.rand(1, 3, 8, 8)

    traced = torch.jit.trace(adapter, probe, check_trace=True)
    output = traced(probe)

    assert isinstance(output, tuple)
    assert len(output) == count
    assert all(torch.is_tensor(item) for item in output)


def test_detect_contract_declares_native_636_centered_letterbox():
    contract = yolonas_coreml_input_contract("detect")

    assert contract == {
        "name": "image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "letterbox_center",
        "interpolation": "bilinear",
        "resize_backend": "pillow",
        "resize_long_side": YOLO_NAS_RESIZE_SIZE,
        "resize_rounding": "round",
        "pad_value": 114,
    }
    assert yolonas_coreml_validation_contract("detect") == {
        "color": "rgb",
        "range": "0_255",
    }


def test_pose_contract_declares_native_bgr_graph_and_bottom_right_padding():
    contract = yolonas_coreml_input_contract("pose")

    assert contract["geometry"] == "letterbox_top_left"
    assert contract["resize_long_side"] == YOLO_NAS_POSE_RESIZE_SIZE
    assert contract["resize_rounding"] == "round"
    assert contract["pad_value"] == 127
    assert [item["name"] for item in yolonas_coreml_output_contract("pose")] == [
        "boxes",
        "scores",
        "keypoints_xy",
        "keypoints_conf",
    ]


def test_detect_geometry_matches_native_preprocess_extent():
    original = np.full((720, 1280, 3), 255, dtype=np.uint8)
    native, native_ratio = preprocess_numpy(original, input_size=640)
    geometry = resolve_yolonas_coreml_geometry(
        task="detect",
        original_size=(1280, 720),
        canvas_size=(640, 640),
    )

    assert geometry.ratio == native_ratio
    assert geometry == type(geometry)(
        ratio=636 / 1280,
        resized_width=636,
        resized_height=358,
        offset_x=2,
        offset_y=141,
    )
    # A solid white source makes its exact native extent observable against
    # the 114-valued padding without relying on model weights.
    foreground = np.all(native == 1.0, axis=0)
    ys, xs = np.where(foreground)
    assert (xs.min(), xs.max() + 1) == (2, 638)
    assert (ys.min(), ys.max() + 1) == (141, 499)


def test_pose_geometry_uses_rounding_and_top_left_placement():
    geometry = resolve_yolonas_coreml_geometry(
        task="pose",
        original_size=(1280, 719),
        canvas_size=(640, 640),
    )

    assert geometry.ratio == 0.5
    assert geometry.resized_width == 640
    assert geometry.resized_height == 360
    assert geometry.offset_x == 0
    assert geometry.offset_y == 0


def test_geometry_rejects_native_zero_sized_rounded_dimension():
    with pytest.raises(ValueError, match="zero-sized dimension"):
        resolve_yolonas_coreml_geometry(
            task="detect",
            original_size=(100000, 1),
            canvas_size=(640, 640),
        )


def test_contract_normalizes_task_case_and_whitespace():
    assert yolonas_coreml_output_contract("DETECT ") == [
        {
            "name": "boxes",
            "role": "boxes",
            "encoding": "xyxy_pixels",
            "rank": 3,
        },
        {"name": "scores", "role": "class_scores", "rank": 3},
    ]


@pytest.mark.parametrize("task", ["segment", ""])
def test_contract_rejects_unknown_task(task):
    with pytest.raises(NotImplementedError, match="detect and pose"):
        yolonas_coreml_output_contract(task)


def test_adapter_fails_closed_on_wrong_decoded_arity():
    adapter = wrap_yolonas_coreml_contract(_TraceDecodedModel(3), "pose")

    with pytest.raises(RuntimeError, match="requires 4 decoded tensor outputs"):
        adapter(torch.rand(1, 3, 8, 8))
