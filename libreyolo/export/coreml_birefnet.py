"""BiRefNet-specific pieces of LibreYOLO's fixed-canvas Core ML contract.

BiRefNet's decoder uses ``torchvision::deform_conv2d``.  Apple added an
ML Program lowering for that operator after the Core ML Tools 9.0 release,
while development builds still report ``__version__ == "9.0"``.  Version
comparison is therefore not a reliable feature gate: this module checks the
converter's operator registry directly and fails before graph capture when the
lowering is absent.

The graph boundary stays deliberately small.  Core ML receives an RGB
``ImageType`` at BiRefNet's native 1024-square canvas, the shared exporter
performs ImageNet normalization in-graph, and the package emits the raw matte
logits.  Sigmoid, bilinear resize to the source canvas, and clamping remain the
existing LibreYOLO host contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

BIREFNET_COREML_CANVAS = 1024
BIREFNET_COREML_SIZES = frozenset({"t", "l"})
BIREFNET_COREML_DEFORM_CONV_MERGE = (
    "d5d4267a8849cd39367e17a2978629d3b341d973"
)


def _normalize_hw(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        height = width = value
    elif isinstance(value, Sequence) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
    else:
        raise ValueError(
            "BiRefNet Core ML canvas must be an int or (height, width), "
            f"got {value!r}."
        )
    if height <= 0 or width <= 0:
        raise ValueError(
            "BiRefNet Core ML canvas dimensions must be positive, "
            f"got {(height, width)}."
        )
    return height, width


def validate_birefnet_coreml_profile(
    *,
    size: str | None,
    precision: str | None,
    canvas_hw: int | Sequence[int],
) -> tuple[int, int]:
    """Validate the conversion-proven BiRefNet profile."""

    normalized_size = str(size or "").strip().lower()
    if normalized_size not in BIREFNET_COREML_SIZES:
        raise NotImplementedError(
            "BiRefNet Core ML export supports only size='t' or size='l'; "
            f"got size={size!r}."
        )
    if precision is not None and precision not in {"fp32", "fp16"}:
        raise ValueError(
            "BiRefNet Core ML precision must be 'fp32' or 'fp16'; "
            f"got {precision!r}."
        )
    height, width = _normalize_hw(canvas_hw)
    expected = (BIREFNET_COREML_CANVAS, BIREFNET_COREML_CANVAS)
    if (height, width) != expected:
        raise NotImplementedError(
            "BiRefNet Core ML export requires its fixed native "
            f"{expected[0]}x{expected[1]} canvas; got {height}x{width}. "
            "The Swin relative-position tables are resolution-tied."
        )
    return height, width


def _has_deform_conv_lowering(torch_ops: Any, registry: Any) -> bool:
    """Return whether a loaded converter exposes Apple's exact lowering."""

    if getattr(torch_ops, "torchvision_deform_conv2d", None) is None:
        return False
    try:
        lowering = registry.get_func("torchvision::deform_conv2d")
    except (AttributeError, KeyError, TypeError):
        return False
    return lowering is not None


def has_birefnet_coreml_lowering() -> bool:
    """Probe the installed Core ML Tools operator registry."""

    try:
        from coremltools.converters.mil.frontend.torch import ops as torch_ops
        from coremltools.converters.mil.frontend.torch.torch_op_registry import (
            _TORCH_OPS_REGISTRY,
        )
    except (ImportError, AttributeError):
        return False
    return _has_deform_conv_lowering(torch_ops, _TORCH_OPS_REGISTRY)


def require_birefnet_coreml_lowering(coremltools_module: Any) -> None:
    """Fail before tracing when the installed converter predates the lowering."""

    if has_birefnet_coreml_lowering():
        return
    version = str(getattr(coremltools_module, "__version__", "unknown"))
    raise NotImplementedError(
        "BiRefNet Core ML export requires Apple's torchvision::deform_conv2d "
        "ML Program lowering, which is absent from the installed "
        f"coremltools {version}. Stable coremltools 9.0 predates the lowering; "
        "use a released Core ML Tools build containing Apple merge "
        f"{BIREFNET_COREML_DEFORM_CONV_MERGE}. LibreYOLO does not vendor the "
        "lowering or pin mutable development branches."
    )


def birefnet_coreml_input_contract() -> dict[str, object]:
    """Describe exact host geometry and the RGB ImageType boundary."""

    return {
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


def birefnet_coreml_output_contract() -> list[dict[str, str]]:
    """Return BiRefNet's one raw-logit output."""

    return [{"name": "matte", "role": "matte_logits"}]


def birefnet_coreml_validation_contract() -> dict[str, str]:
    """Describe tensors produced by the native exported-backend preprocessor."""

    return {"color": "rgb", "range": "imagenet"}


__all__ = [
    "BIREFNET_COREML_CANVAS",
    "BIREFNET_COREML_DEFORM_CONV_MERGE",
    "BIREFNET_COREML_SIZES",
    "birefnet_coreml_input_contract",
    "birefnet_coreml_output_contract",
    "birefnet_coreml_validation_contract",
    "has_birefnet_coreml_lowering",
    "require_birefnet_coreml_lowering",
    "validate_birefnet_coreml_profile",
]
