"""PicoSAM3's fixed-ROI Core ML component contract.

The exported graph is intentionally not a full promptable-segmentation
pipeline.  It consumes one already-cropped 96x96 RGB ROI and returns one mask
logit map.  Box expansion/cropping and placement into the source image remain
host operations, matching PicoSAM3's existing ONNX boundary.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

PICOSAM3_COREML_COMPONENT_CONTRACT = "picosam3_roi_v1"
PICOSAM3_COREML_INPUT_SIZE = 96
PICOSAM3_COREML_ROI_PADDING = 0.1
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class PicoSAM3CoreMLAdapter(nn.Module):
    """Normalize canonical Core ML RGB input for the native ROI network."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, roi_image: torch.Tensor) -> torch.Tensor:
        return self.model((roi_image - self._mean) / self._std)


def wrap_picosam3_coreml_contract(model: nn.Module) -> nn.Module:
    """Return the canonical-RGB adapter used by the shared exporter."""
    return PicoSAM3CoreMLAdapter(model).eval()


def validate_picosam3_coreml_profile(
    *,
    size: str | None,
    canvas_hw: tuple[int, int],
) -> None:
    """Reject profiles that cannot match the trained ROI component."""
    if size != "pico":
        raise NotImplementedError(
            "PicoSAM3 Core ML export supports only size='pico'; "
            f"got size={size!r}."
        )
    if tuple(int(value) for value in canvas_hw) != (
        PICOSAM3_COREML_INPUT_SIZE,
        PICOSAM3_COREML_INPUT_SIZE,
    ):
        raise NotImplementedError(
            "PicoSAM3 Core ML export requires its fixed 96x96 ROI canvas; "
            f"got {canvas_hw[0]}x{canvas_hw[1]}."
        )


def picosam3_coreml_input_contract() -> dict[str, Any]:
    """Describe the artifact boundary: one host-prepared RGB ROI."""
    return {
        "name": "roi_image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "native",
        "interpolation": "bilinear",
        "resize_backend": "pillow",
        "pad_value": 0,
    }


def picosam3_coreml_validation_contract() -> dict[str, Any]:
    """Describe the tensor domain consumed by the unwrapped PyTorch network."""
    # ``imagenet`` is a named schema domain whose exact mean/std are supplied
    # by the backend contract parser; explicit constants are reserved for the
    # generic ``standardized`` domain.
    return {"color": "rgb", "range": "imagenet"}


def picosam3_coreml_output_contract() -> list[dict[str, Any]]:
    """Return the exact raw ROI output contract."""
    return [
        {
            "name": "mask_logits",
            "role": "mask_logits",
            "encoding": "raw_logits",
            "rank": 4,
        }
    ]


def picosam3_coreml_component_metadata() -> dict[str, Any]:
    """Return orchestration metadata required by the prompt-aware runtime."""
    return {
        "artifact_scope": "roi_component",
        "component_contract": PICOSAM3_COREML_COMPONENT_CONTRACT,
        "roi_input_size": PICOSAM3_COREML_INPUT_SIZE,
        "roi_padding": PICOSAM3_COREML_ROI_PADDING,
        "roi_batch": 1,
        "prompt_type": "boxes",
    }


__all__ = [
    "PICOSAM3_COREML_COMPONENT_CONTRACT",
    "PICOSAM3_COREML_INPUT_SIZE",
    "PICOSAM3_COREML_ROI_PADDING",
    "PicoSAM3CoreMLAdapter",
    "picosam3_coreml_component_metadata",
    "picosam3_coreml_input_contract",
    "picosam3_coreml_output_contract",
    "picosam3_coreml_validation_contract",
    "validate_picosam3_coreml_profile",
    "wrap_picosam3_coreml_contract",
]
