"""SwinIR-specific pieces of LibreYOLO's fixed-canvas Core ML contract.

This module intentionally contains no ``coremltools`` import or runtime claim.
The shared exporter owns graph capture, conversion, metadata, and package
saving; these helpers define the fixed 64x64 profile shared by all three
SwinIR sizes.

The contract is derived solely from LibreYOLO's Apache-2.0-attributed SwinIR
implementation under ``libreyolo.models.swinir``.  Core ML receives canonical
RGB bytes and exposes them to the graph as RGB floats in ``[0, 1]``, which is
already the native SwinIR photometric input.  A fixed graph cannot reproduce
native arbitrary-resolution, pad-to-multiple-of-eight inference by padding a
smaller source all the way to its canvas: that changes the transformer's
context.  The host must therefore supply an image that exactly matches the
exported canvas.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


SWINIR_COREML_CANVAS = 64
SWINIR_COREML_SCALE = 4
SWINIR_COREML_SIZES = frozenset({"s", "m", "l"})


def _normalize_hw(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        height = width = value
    elif isinstance(value, Sequence) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
    else:
        raise ValueError(
            f"SwinIR Core ML canvas must be an int or (height, width), got {value!r}."
        )
    if height <= 0 or width <= 0:
        raise ValueError(
            f"SwinIR Core ML canvas dimensions must be positive, got {(height, width)}."
        )
    return height, width


def validate_swinir_coreml_profile(
    *,
    size: str | None,
    canvas_hw: int | Sequence[int],
) -> tuple[int, int]:
    """Validate the bounded SwinIR profile currently ready for conversion.

    The small, medium, and large 64x64 graphs have each passed strict
    two-input TorchScript parity and Core ML Tools 9 ML Program conversion.
    This is also the resolution for which every model owns precomputed
    shifted-window attention masks. Other canvases trace mask construction
    into the graph and remain outside this fixed-profile contract. Apple
    runtime parity is queued separately and is not claimed here.
    """

    normalized_size = str(size or "").strip().lower()
    if normalized_size not in SWINIR_COREML_SIZES:
        raise NotImplementedError(
            "SwinIR Core ML export supports sizes 's', 'm', and 'l' at their "
            f"fixed 64x64 profile; got size={size!r}."
        )
    height, width = _normalize_hw(canvas_hw)
    expected = (SWINIR_COREML_CANVAS, SWINIR_COREML_CANVAS)
    if (height, width) != expected:
        raise NotImplementedError(
            f"SwinIR-{normalized_size} Core ML export requires the bounded native "
            f"{expected[0]}x{expected[1]} canvas; got {height}x{width}. "
            "Other canvases materialize shifted-window mask construction in "
            "the trace and must first pass macOS conversion/runtime parity."
        )
    return height, width


class SwinIRCoreMLAdapter(nn.Module):
    """Expose SwinIR's restored tensor from canonical RGB ``[0, 1]`` input."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        if not torch.is_tensor(output):
            raise RuntimeError(
                "SwinIR Core ML export requires one restored tensor output."
            )
        return output


def wrap_swinir_coreml_contract(nn_model: nn.Module) -> nn.Module:
    """Return the graph adapter for the fixed SwinIR Core ML profile."""

    return SwinIRCoreMLAdapter(nn_model).eval()


def swinir_coreml_input_contract() -> dict[str, object]:
    """Describe the exact host-side image contract.

    ``geometry='native'`` is fail-closed: a fixed package may only receive a
    source whose dimensions equal its canvas.  Native arbitrary-size SwinIR
    remains available through PyTorch inference and tiling.
    """

    return {
        "name": "image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "native",
        # Required schema fields; no resize occurs for native geometry.
        "interpolation": "bilinear",
        "resize_backend": "pillow",
        "pad_value": 0,
    }


def swinir_coreml_output_contract() -> list[dict[str, str]]:
    """Return the single raw float restoration output."""

    return [{"name": "restored", "role": "restored"}]


def swinir_coreml_validation_contract() -> dict[str, str]:
    """Describe native restore-validator tensors before ImageType inversion."""

    return {"color": "rgb", "range": "0_1"}


__all__ = [
    "SWINIR_COREML_CANVAS",
    "SWINIR_COREML_SCALE",
    "SWINIR_COREML_SIZES",
    "SwinIRCoreMLAdapter",
    "swinir_coreml_input_contract",
    "swinir_coreml_output_contract",
    "swinir_coreml_validation_contract",
    "validate_swinir_coreml_profile",
    "wrap_swinir_coreml_contract",
]
