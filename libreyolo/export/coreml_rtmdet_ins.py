"""RTMDet-Ins fixed-canvas raw-output contract for Core ML.

The exported graph contains the complete image backbone, neck, and prediction
head.  It deliberately stops before candidate filtering, NMS, and the
per-instance dynamic mask network.  Those operations depend on a variable
number of surviving instances and remain in LibreYOLO's existing host
postprocessor.
"""

from __future__ import annotations

import math
from typing import Any

RTMDET_INS_COREML_CONTRACT = "rtmdet_ins_raw_v1"
RTMDET_INS_COREML_STRIDES = (8, 16, 32)
RTMDET_INS_COREML_NUM_GEN_PARAMS = 169
RTMDET_INS_COREML_NUM_PROTOTYPES = 8
RTMDET_INS_COREML_MASK_STRIDE = 8
RTMDET_INS_COREML_NMS_PRE = 1000
RTMDET_INS_COREML_MAX_MASKS = 100
RTMDET_INS_COREML_PRIOR_OFFSET = 0
RTMDET_INS_COREML_DYNAMIC_WEIGHT_NUMS = (80, 64, 8)
RTMDET_INS_COREML_DYNAMIC_BIAS_NUMS = (8, 8, 1)
RTMDET_INS_COREML_DYCONV_CHANNELS = 8
RTMDET_INS_COREML_MASK_THRESHOLD = 0.5
_RTMDET_INS_SIZES = frozenset({"t", "s", "m", "l", "x"})


def validate_rtmdet_ins_coreml_profile(
    *,
    size: str | None,
    canvas_hw: tuple[int, int],
) -> None:
    """Reject graph shapes that cannot preserve the native mask geometry."""
    if size not in _RTMDET_INS_SIZES:
        raise NotImplementedError(
            "RTMDet-Ins Core ML export supports sizes t/s/m/l/x; "
            f"got size={size!r}."
        )
    height, width = (int(value) for value in canvas_hw)
    if height <= 0 or width <= 0:
        raise ValueError(
            f"RTMDet-Ins Core ML canvas must be positive; got {canvas_hw}."
        )
    if height % 32 or width % 32:
        raise NotImplementedError(
            "RTMDet-Ins Core ML export requires each canvas dimension to be "
            "divisible by 32 so the three FPN levels and stride-8 mask feature "
            f"have an exact fixed geometry; got {height}x{width}."
        )


def rtmdet_ins_coreml_output_contract() -> list[dict[str, Any]]:
    """Return the ten tensors in the exact order emitted by export mode."""
    outputs: list[dict[str, Any]] = []
    for stride in RTMDET_INS_COREML_STRIDES:
        outputs.append(
            {
                "name": f"class_logits_s{stride}",
                "role": f"class_logits_s{stride}",
                "encoding": "raw_logits",
                "rank": 4,
            }
        )
    for stride in RTMDET_INS_COREML_STRIDES:
        outputs.append(
            {
                "name": f"box_distances_s{stride}",
                "role": f"box_distances_s{stride}",
                "encoding": "ltrb_pixels",
                "rank": 4,
            }
        )
    for stride in RTMDET_INS_COREML_STRIDES:
        outputs.append(
            {
                "name": f"dynamic_kernels_s{stride}",
                "role": f"dynamic_kernels_s{stride}",
                "encoding": RTMDET_INS_COREML_CONTRACT,
                "rank": 4,
            }
        )
    outputs.append(
        {
            "name": "mask_features",
            "role": "mask_features",
            "encoding": "stride8_features",
            "rank": 4,
        }
    )
    return outputs


def expected_rtmdet_ins_coreml_shapes(
    *,
    nc: int,
    canvas_hw: tuple[int, int],
) -> dict[str, tuple[int, ...]]:
    """Return the exact schema-v2 output shapes for one fixed canvas."""
    height, width = (int(value) for value in canvas_hw)
    shapes: dict[str, tuple[int, ...]] = {}
    for stride in RTMDET_INS_COREML_STRIDES:
        spatial = (math.ceil(height / stride), math.ceil(width / stride))
        shapes[f"class_logits_s{stride}"] = (1, int(nc), *spatial)
        shapes[f"box_distances_s{stride}"] = (1, 4, *spatial)
        shapes[f"dynamic_kernels_s{stride}"] = (
            1,
            RTMDET_INS_COREML_NUM_GEN_PARAMS,
            *spatial,
        )
    shapes["mask_features"] = (
        1,
        RTMDET_INS_COREML_NUM_PROTOTYPES,
        height // RTMDET_INS_COREML_MASK_STRIDE,
        width // RTMDET_INS_COREML_MASK_STRIDE,
    )
    return shapes


def rtmdet_ins_coreml_metadata() -> dict[str, Any]:
    """Pin every host-side decoding constant used by the raw-output ABI."""
    return {
        "rtmdet_ins_contract": RTMDET_INS_COREML_CONTRACT,
        "rtmdet_ins_strides": list(RTMDET_INS_COREML_STRIDES),
        "rtmdet_ins_num_gen_params": RTMDET_INS_COREML_NUM_GEN_PARAMS,
        "rtmdet_ins_num_prototypes": RTMDET_INS_COREML_NUM_PROTOTYPES,
        "rtmdet_ins_mask_stride": RTMDET_INS_COREML_MASK_STRIDE,
        "rtmdet_ins_nms_pre": RTMDET_INS_COREML_NMS_PRE,
        "rtmdet_ins_max_masks": RTMDET_INS_COREML_MAX_MASKS,
        "rtmdet_ins_prior_offset": RTMDET_INS_COREML_PRIOR_OFFSET,
        "rtmdet_ins_dynamic_weight_nums": list(
            RTMDET_INS_COREML_DYNAMIC_WEIGHT_NUMS
        ),
        "rtmdet_ins_dynamic_bias_nums": list(
            RTMDET_INS_COREML_DYNAMIC_BIAS_NUMS
        ),
        "rtmdet_ins_dyconv_channels": RTMDET_INS_COREML_DYCONV_CHANNELS,
        "rtmdet_ins_mask_threshold": RTMDET_INS_COREML_MASK_THRESHOLD,
    }


__all__ = [
    "RTMDET_INS_COREML_CONTRACT",
    "RTMDET_INS_COREML_DYNAMIC_BIAS_NUMS",
    "RTMDET_INS_COREML_DYNAMIC_WEIGHT_NUMS",
    "RTMDET_INS_COREML_DYCONV_CHANNELS",
    "RTMDET_INS_COREML_MASK_STRIDE",
    "RTMDET_INS_COREML_MASK_THRESHOLD",
    "RTMDET_INS_COREML_MAX_MASKS",
    "RTMDET_INS_COREML_NMS_PRE",
    "RTMDET_INS_COREML_NUM_GEN_PARAMS",
    "RTMDET_INS_COREML_NUM_PROTOTYPES",
    "RTMDET_INS_COREML_PRIOR_OFFSET",
    "RTMDET_INS_COREML_STRIDES",
    "expected_rtmdet_ins_coreml_shapes",
    "rtmdet_ins_coreml_metadata",
    "rtmdet_ins_coreml_output_contract",
    "validate_rtmdet_ins_coreml_profile",
]
