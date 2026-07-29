"""YOLO-NAS contract fragments for the Core ML exporter.

This module contains only repository-owned adapter and metadata logic.  It
does not import ``coremltools`` and its presence is not a hardware-validation
claim.  The shared Core ML exporter and backend consume these fragments when
YOLO-NAS support is enabled.

YOLO-NAS has two distinct public image contracts:

* detection resizes the longest side to 636 pixels, rounds both resized
  dimensions, and centers the image in a square canvas filled with RGB 114;
* pose resizes the longest side to 640 pixels, rounds both resized dimensions,
  places the image at the top-left (padding only the bottom/right), and feeds
  the network BGR values with a pad value of 127.

The Core ML ImageType boundary is canonical uint8 RGB.  Consequently only the
pose channel reversal belongs in the traced graph; geometry stays on the host
and is described by the returned input contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..postprocess.yolonas import (
    YOLO_NAS_POSE_RESIZE_SIZE,
    YOLO_NAS_RESIZE_SIZE,
)
from ..models.yolonas.utils import YOLO_NAS_POSE_PAD_VALUE

_YOLO_NAS_TASKS = frozenset({"detect", "pose"})


def _canonical_task(task: str) -> str:
    value = str(task).strip().lower()
    if value not in _YOLO_NAS_TASKS:
        raise NotImplementedError(
            f"YOLO-NAS Core ML export supports only detect and pose; got task={task!r}."
        )
    return value


class YoloNASCoreMLAdapter(nn.Module):
    """Expose decoded YOLO-NAS tensors from canonical Core ML RGB input.

    Eager YOLO-NAS inference returns ``(decoded, raw)`` while the tracing
    branch returns ``decoded`` directly.  The adapter makes both paths expose
    the same flat tensor tuple used by ONNX and every exported backend.
    """

    def __init__(self, model: nn.Module, *, task: str) -> None:
        super().__init__()
        self.model = model
        self.task = _canonical_task(task)

    def forward(self, image: torch.Tensor):
        if self.task == "pose":
            # Core ML ImageType supplies RGB floats in [0, 1]; native
            # YOLO-NAS pose inference consumes BGR floats in [0, 1].
            image = image[:, [2, 1, 0], :, :]

        output = self.model(image)
        if (
            isinstance(output, (tuple, list))
            and len(output) == 2
            and isinstance(output[0], (tuple, list))
        ):
            output = output[0]
        if isinstance(output, (tuple, list)):
            output = tuple(output)

        expected = 4 if self.task == "pose" else 2
        if not isinstance(output, tuple) or len(output) != expected:
            raise RuntimeError(
                f"YOLO-NAS {self.task} Core ML export requires {expected} "
                "decoded tensor outputs; the model returned an incompatible "
                f"{type(output).__name__} contract."
            )
        if not all(torch.is_tensor(item) for item in output):
            raise RuntimeError(
                f"YOLO-NAS {self.task} Core ML outputs must all be tensors."
            )
        return output


def wrap_yolonas_coreml_contract(nn_model: nn.Module, task: str) -> nn.Module:
    """Return the task-specific, decoded-only Core ML graph adapter."""
    return YoloNASCoreMLAdapter(nn_model, task=task).eval()


def yolonas_coreml_input_contract(task: str) -> dict[str, Any]:
    """Describe exact host geometry at the canonical Core ML image boundary.

    ``resize_long_side`` and ``resize_rounding`` are deliberate schema fields.
    A normal canvas-sized letterbox would change YOLO-NAS detection predictions
    because its trained preprocessing target is 636 rather than 640.
    """
    task = _canonical_task(task)
    if task == "pose":
        geometry = "letterbox_top_left"
        resize_long_side = YOLO_NAS_POSE_RESIZE_SIZE
        pad_value = YOLO_NAS_POSE_PAD_VALUE
    else:
        geometry = "letterbox_center"
        resize_long_side = YOLO_NAS_RESIZE_SIZE
        pad_value = 114

    return {
        "name": "image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": geometry,
        "interpolation": "bilinear",
        "resize_backend": "pillow",
        "resize_long_side": int(resize_long_side),
        "resize_rounding": "round",
        "pad_value": int(pad_value),
    }


def yolonas_coreml_validation_contract(task: str) -> dict[str, str]:
    """Declare the canonical tensor supplied by detection-style validators."""
    _canonical_task(task)
    return {"color": "rgb", "range": "0_255"}


def yolonas_coreml_output_contract(task: str) -> list[dict[str, Any]]:
    """Return semantic output names in exact decoded graph order.

    For a batch-one canvas the shapes are ``boxes=(1, A, 4)`` and
    ``scores=(1, A, nc)``.  Pose additionally emits
    ``keypoints_xy=(1, A, K, 2)`` and ``keypoints_conf=(1, A, K)``.  At the
    native 640 canvas, ``A = 80*80 + 40*40 + 20*20 = 8400``.
    """
    task = _canonical_task(task)
    outputs = [
        {
            "name": "boxes",
            "role": "boxes",
            "encoding": "xyxy_pixels",
            "rank": 3,
        },
        {"name": "scores", "role": "class_scores", "rank": 3},
    ]
    if task == "pose":
        outputs.extend(
            [
                {
                    "name": "keypoints_xy",
                    "role": "keypoints_xy",
                    "encoding": "xy_pixels",
                    "rank": 4,
                },
                {
                    "name": "keypoints_conf",
                    "role": "keypoints_conf",
                    "encoding": "probabilities",
                    "rank": 3,
                },
            ]
        )
    return outputs


@dataclass(frozen=True)
class YoloNASGeometry:
    """Resolved native resize/pad transform for one original image."""

    ratio: float
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int


def resolve_yolonas_coreml_geometry(
    *,
    task: str,
    original_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> YoloNASGeometry:
    """Resolve the metadata geometry using the native preprocessing formula.

    Sizes are ``(width, height)``.  This pure helper is useful to keep backend
    integration tests tied to the same public contract without loading a model
    or the Apple runtime.
    """
    contract = yolonas_coreml_input_contract(task)
    original_width, original_height = (int(value) for value in original_size)
    canvas_width, canvas_height = (int(value) for value in canvas_size)
    if min(original_width, original_height, canvas_width, canvas_height) <= 0:
        raise ValueError("YOLO-NAS image and canvas dimensions must be positive.")
    if canvas_width != canvas_height:
        raise ValueError("YOLO-NAS Core ML export requires a square canvas.")

    resize_long_side = min(
        int(contract["resize_long_side"]),
        canvas_width,
        canvas_height,
    )
    ratio = min(
        resize_long_side / original_height,
        resize_long_side / original_width,
    )
    resized_width = int(round(original_width * ratio))
    resized_height = int(round(original_height * ratio))
    if resized_width <= 0 or resized_height <= 0:
        raise ValueError(
            "YOLO-NAS native rounded resize produced a zero-sized dimension; "
            "the source image aspect ratio is too extreme for this canvas."
        )
    if contract["geometry"] == "letterbox_center":
        offset_x = (canvas_width - resized_width) // 2
        offset_y = (canvas_height - resized_height) // 2
    else:
        offset_x = offset_y = 0
    return YoloNASGeometry(
        ratio=ratio,
        resized_width=resized_width,
        resized_height=resized_height,
        offset_x=offset_x,
        offset_y=offset_y,
    )


__all__ = [
    "YoloNASCoreMLAdapter",
    "YoloNASGeometry",
    "resolve_yolonas_coreml_geometry",
    "wrap_yolonas_coreml_contract",
    "yolonas_coreml_input_contract",
    "yolonas_coreml_output_contract",
    "yolonas_coreml_validation_contract",
]
