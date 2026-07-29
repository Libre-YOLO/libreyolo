"""SegFormer-specific pieces of the fixed-canvas Core ML contract.

This module contains no conversion code.  The shared Core ML exporter owns
capture, conversion, metadata serialization, and package saving; these helpers
only describe the native LibreSegformer image geometry and dense-output
inversion precisely enough for that shared path to consume.

The contract is derived solely from LibreYOLO's permissively licensed
``libreyolo.models.segformer`` implementation.  Published ADE20K weights have
separate non-commercial terms, but a model trained from scratch does not.
"""

from __future__ import annotations

from dataclasses import dataclass


SEGFORMER_COREML_OUTPUT_NAME = "semantic_logits"
SEGFORMER_COREML_ALIGN_CORNERS = False


@dataclass(frozen=True)
class SegformerLetterboxGeometry:
    """Exact spatial result of native SegFormer preprocessing."""

    ratio: float
    resized_height: int
    resized_width: int


def _positive_hw(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a (height, width) tuple, got {value!r}.")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} values must be positive, got {value!r}.")
    return height, width


def segformer_letterbox_geometry(
    original_hw: tuple[int, int],
    canvas_hw: tuple[int, int],
) -> SegformerLetterboxGeometry:
    """Return native top-left letterbox geometry.

    LibreSegformer scales by the smaller canvas ratio, truncates both resized
    dimensions with Python ``int`` (floor for positive values), resizes through
    OpenCV ``INTER_LINEAR``, and pads only the bottom/right with RGB value 114.
    """

    original_h, original_w = _positive_hw(original_hw, name="original_hw")
    canvas_h, canvas_w = _positive_hw(canvas_hw, name="canvas_hw")
    ratio = min(canvas_h / original_h, canvas_w / original_w)
    return SegformerLetterboxGeometry(
        ratio=ratio,
        resized_height=max(int(original_h * ratio), 1),
        resized_width=max(int(original_w * ratio), 1),
    )


def segformer_valid_logits_hw(
    original_hw: tuple[int, int],
    canvas_hw: tuple[int, int],
    logits_hw: tuple[int, int],
) -> tuple[int, int]:
    """Return the native valid top-left logit window before output resize.

    The native postprocessor intentionally uses ``round(original * ratio *
    output_scale)`` even though input resize dimensions are truncated.  That
    can retain one boundary padding row/column for some aspect ratios.  Core ML
    inversion must preserve this behavior for exact native parity.
    """

    original_h, original_w = _positive_hw(original_hw, name="original_hw")
    canvas_h, canvas_w = _positive_hw(canvas_hw, name="canvas_hw")
    logits_h, logits_w = _positive_hw(logits_hw, name="logits_hw")
    ratio = min(canvas_h / original_h, canvas_w / original_w)
    scale_y = logits_h / canvas_h
    scale_x = logits_w / canvas_w
    valid_h = min(logits_h, max(int(round(original_h * ratio * scale_y)), 1))
    valid_w = min(logits_w, max(int(round(original_w * ratio * scale_x)), 1))
    return valid_h, valid_w


def segformer_coreml_input_contract() -> dict[str, object]:
    """Return the exact host-side image contract for a SegFormer package."""

    return {
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


def segformer_coreml_output_contract() -> list[dict[str, str]]:
    """Return the single dense tensor emitted by the eval graph."""

    return [{"name": SEGFORMER_COREML_OUTPUT_NAME, "role": "semantic_logits"}]


def segformer_coreml_validation_contract() -> dict[str, str]:
    """Describe tensors produced by LibreYOLO's semantic validator.

    The Core ML ImageType boundary receives canonical RGB bytes.  Validator
    batches are RGB floats in ``[0, 1]`` and are inverted to those bytes by the
    backend.  ImageNet normalization must not be added here: it already lives
    inside ``LibreSegformerNet.forward``.
    """

    return {"color": "rgb", "range": "0_1"}


__all__ = [
    "SEGFORMER_COREML_ALIGN_CORNERS",
    "SEGFORMER_COREML_OUTPUT_NAME",
    "SegformerLetterboxGeometry",
    "segformer_coreml_input_contract",
    "segformer_coreml_output_contract",
    "segformer_coreml_validation_contract",
    "segformer_letterbox_geometry",
    "segformer_valid_logits_hw",
]
