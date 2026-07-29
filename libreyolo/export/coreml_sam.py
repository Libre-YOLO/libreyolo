"""Strict split-component Core ML contract for promptable SAM families.

Promptable segmentation is an interactive pipeline, not one image-to-results
graph.  The heavy image encoder runs once, then a cheap decoder is invoked for
each prompt query.  This module defines that boundary without importing
``coremltools``:

* one fixed-shape, model-ready FP32 image encoder;
* six decoder functions covering points, boxes, and points+box prompts in
  single-mask and multimask modes;
* fixed batch/query dimensions (``N=1``, ``Q=1``) with a genuinely dynamic,
  finitely bounded point dimension ``P``;
* exact named FP32/INT32 tensor interfaces and integrity-checked metadata.

The host owns raw-image preprocessing, prompt-coordinate transforms, the
``Q>1`` loop, low-resolution mask upscaling/cropping, thresholding, and
``Results`` assembly.  Decoder functions return raw mask logits and native
predicted-IoU values.

Two conversion-only rewrites are included because the unmodified graphs do not
lower faithfully with a symbolic point count:

* MobileSAM prompt label/corner updates are expressed functionally with
  ``torch.where`` and additions instead of boolean/slice assignment.
* SAM2's fixed 1024-pixel encoder freezes the model's own already-evaluated
  Hiera positional tensor, avoiding a conversion-time bicubic resize whose
  result is constant for this contract.

Both rewrites compose LibreYOLO's existing Apache-2.0 model modules and learned
parameters.  They do not introduce model code or weights from a new upstream.
SAM3 support is visual-prompt-only.  Its gated custom-license weights make any
converted artifact local-user-only and non-redistributable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from numbers import Integral
from types import MethodType
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

SAM_COREML_COMPONENT_CONTRACT = "sam_split_promptable_v1"
SAM_COREML_PIPELINE_SCHEMA_VERSION = 1
SAM_COREML_ARTIFACT_SCOPE = "host_orchestrated_promptable_component_bundle"
# Core ML multifunction ML Programs are available from iOS 18 / macOS 15.
# Individual component graphs could target iOS 15, but LibreYOLO deliberately
# emits one native package with seven named functions rather than inventing a
# directory-of-packages container that Core ML itself cannot load.
SAM_COREML_MINIMUM_DEPLOYMENT_TARGET = "iOS18"

SAM_COREML_ENCODER_FUNCTION = "encode_image"
SAM_COREML_PROMPT_MODES = ("points", "boxes", "points_boxes")
SAM_COREML_MASK_MODES = ("single", "multimask")
SAM_COREML_DECODER_FUNCTIONS = tuple(
    f"decode_{prompt_mode}_{mask_mode}"
    for prompt_mode in SAM_COREML_PROMPT_MODES
    for mask_mode in SAM_COREML_MASK_MODES
)
SAM_COREML_FUNCTION_NAMES = (
    SAM_COREML_ENCODER_FUNCTION,
    *SAM_COREML_DECODER_FUNCTIONS,
)

SAM_COREML_ENCODER_INPUT = "pixel_values"
SAM_COREML_POINT_COORDS_INPUT = "point_coords"
SAM_COREML_POINT_LABELS_INPUT = "point_labels"
SAM_COREML_BOXES_INPUT = "boxes"
SAM_COREML_MASKS_OUTPUT = "low_res_masks"
SAM_COREML_IOU_OUTPUT = "iou_scores"

SAM_COREML_IMAGE_BATCH = 1
SAM_COREML_QUERY_BATCH = 1
SAM_COREML_DEFAULT_MAX_POINTS = 16
SAM_COREML_MAX_POINTS_LIMIT = 64
SAM_COREML_NUM_MULTIMASK_OUTPUTS = 3

SAM_COREML_HOST_OPERATIONS = (
    "raw_image_preprocess",
    "prompt_coordinate_transform",
    "query_loop",
    "decoder_function_selection",
    "mask_resize_crop_and_threshold",
    "results_assembly",
)

_SAM2_FIXED_POSITION_BUFFER = "_libreyolo_coreml_fixed_position_embedding"
_SAM2_ORIGINAL_GET_POS_EMBED = "_libreyolo_coreml_original_get_pos_embed"


@dataclass(frozen=True)
class _SAMFamilySpec:
    sizes: tuple[str, ...]
    image_size: int
    embedding_names: tuple[str, ...]
    embedding_shapes: tuple[tuple[int, int, int, int], ...]
    low_res_mask_size: int
    preprocess_contract: str
    postprocess_size_metadata: tuple[str, ...]
    iou_encoding: str
    model_types: tuple[str, ...]
    encoder_rewrite: str
    weights_license: str
    redistributable: bool
    native_outputs_omitted: tuple[str, ...] = ()
    release_notice_gap: str | None = None
    visual_only: bool = False


_FAMILY_SPECS: dict[str, _SAMFamilySpec] = {
    "edgetam": _SAMFamilySpec(
        sizes=("edge",),
        image_size=1024,
        embedding_names=(
            "image_embedding_s4",
            "image_embedding_s8",
            "image_embedding_s16",
        ),
        embedding_shapes=(
            (1, 32, 256, 256),
            (1, 64, 128, 128),
            (1, 256, 64, 64),
        ),
        low_res_mask_size=256,
        preprocess_contract="edgetam_square_imagenet_v1",
        postprocess_size_metadata=("original_size",),
        iou_encoding="sigmoid_probability",
        model_types=("edgetam", "edgetam_video"),
        encoder_rewrite="none",
        weights_license="Apache-2.0",
        redistributable=True,
        native_outputs_omitted=("object_score_logits",),
    ),
    "mobilesam": _SAMFamilySpec(
        sizes=("tiny",),
        image_size=1024,
        embedding_names=("image_embedding",),
        embedding_shapes=((1, 256, 64, 64),),
        low_res_mask_size=256,
        preprocess_contract="mobilesam_longest_side_normalize_pad_v1",
        postprocess_size_metadata=("original_size", "reshaped_input_size"),
        iou_encoding="raw_unbounded",
        model_types=(),
        encoder_rewrite="model_ready_image_encoder",
        weights_license="Apache-2.0",
        redistributable=True,
        release_notice_gap=(
            "MobileSAM upstream source commit/checkpoint revision is not "
            "pinned; see weights/LICENSE_NOTICE.txt"
        ),
    ),
    "sam": _SAMFamilySpec(
        sizes=("base", "large", "huge"),
        image_size=1024,
        embedding_names=("image_embedding",),
        embedding_shapes=((1, 256, 64, 64),),
        low_res_mask_size=256,
        preprocess_contract="sam1_processor_model_ready_v1",
        postprocess_size_metadata=("original_size", "reshaped_input_size"),
        iou_encoding="raw_unbounded",
        model_types=("sam",),
        encoder_rewrite="none",
        weights_license="Apache-2.0",
        redistributable=True,
    ),
    "sam2": _SAMFamilySpec(
        sizes=("tiny", "small", "base-plus", "large"),
        image_size=1024,
        embedding_names=(
            "image_embedding_s4",
            "image_embedding_s8",
            "image_embedding_s16",
        ),
        embedding_shapes=(
            (1, 32, 256, 256),
            (1, 64, 128, 128),
            (1, 256, 64, 64),
        ),
        low_res_mask_size=256,
        preprocess_contract="sam2_processor_model_ready_v1",
        postprocess_size_metadata=("original_size",),
        iou_encoding="sigmoid_probability",
        model_types=("sam2",),
        encoder_rewrite="freeze_native_hiera_position_embedding",
        weights_license="Apache-2.0",
        redistributable=True,
        native_outputs_omitted=("object_score_logits",),
    ),
    "sam3": _SAMFamilySpec(
        sizes=("large",),
        image_size=1008,
        embedding_names=(
            "image_embedding_288",
            "image_embedding_144",
            "image_embedding_72",
        ),
        embedding_shapes=(
            (1, 32, 288, 288),
            (1, 64, 144, 144),
            (1, 256, 72, 72),
        ),
        low_res_mask_size=288,
        preprocess_contract="sam3_tracker_processor_model_ready_v1",
        postprocess_size_metadata=("original_size",),
        iou_encoding="sigmoid_probability",
        model_types=("sam3_tracker",),
        encoder_rewrite="none",
        weights_license="Meta SAM License (custom, gated)",
        redistributable=False,
        native_outputs_omitted=("object_score_logits",),
        visual_only=True,
    ),
}

# These strings are deliberately verbose. They are part of the signed profile
# persisted into the package, so changing any resize domain, arithmetic order,
# padding domain, or mask-resize stage requires a new versioned contract rather
# than silently changing how an old artifact is interpreted.
_SAM_COREML_HOST_CONTRACTS: dict[str, dict[str, str]] = {
    "edgetam": {
        "image_resize": (
            "torchvision_v1_to_tensor_float01_square_bilinear_antialias_true"
        ),
        "normalization": "fp32_imagenet_mean_std_after_resize",
        "padding": "none",
        "coordinates": "fp32_divide_xy_by_wh_then_multiply_1024",
        "mask_postprocess": (
            "bilinear_align_corners_false_low256_to_original_then_strict_gt_zero"
        ),
    },
    "mobilesam": {
        "image_resize": "pillow_uint8_longest_side_half_up_bilinear",
        "normalization": "fp32_raw255_mobile_mean_std_after_resize",
        "padding": "normalized_zero_right_bottom_to_1024",
        "coordinates": (
            "numpy_fp32_then_float64_half_up_resized_wh_scale_then_fp32"
        ),
        "mask_postprocess": (
            "bilinear_align_corners_false_low256_to_1024_crop_reshaped_"
            "then_original_then_strict_gt_zero"
        ),
    },
    "sam": {
        "image_resize": "pillow_uint8_longest_side_half_up_bilinear",
        "normalization": (
            "numpy_float64_div255_then_fp32_imagenet_mean_std_after_resize"
        ),
        "padding": "normalized_zero_right_bottom_to_1024",
        "coordinates": (
            "points_float64_boxes_float32_then_float64_half_up_resized_wh_"
            "scale_then_fp32"
        ),
        "mask_postprocess": (
            "bilinear_align_corners_false_low256_to_1024_crop_reshaped_"
            "then_original_then_strict_gt_zero"
        ),
    },
    "sam2": {
        "image_resize": (
            "torchvision_v2_uint8_square_bilinear_antialias_true_before_float"
        ),
        "normalization": "fp32_raw255_fused_imagenet_mean_std_after_resize",
        "padding": "none",
        "coordinates": "fp32_multiply_xy_by_1024_over_wh",
        "mask_postprocess": (
            "bilinear_align_corners_false_low256_to_original_then_strict_gt_zero"
        ),
    },
    "sam3": {
        "image_resize": (
            "torchvision_v2_uint8_square_bilinear_antialias_true_before_float"
        ),
        "normalization": "fp32_raw255_fused_half_mean_std_after_resize",
        "padding": "none",
        "coordinates": "fp32_multiply_xy_by_1008_over_wh",
        "mask_postprocess": (
            "bilinear_align_corners_false_low288_to_original_then_strict_gt_zero"
        ),
    },
}


def _strict_int(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {result}.")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_field(metadata: Mapping[str, Any], name: str, expected_type: type) -> Any:
    value = metadata.get(name)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} must contain valid JSON.") from exc
    if not isinstance(value, expected_type):
        raise ValueError(
            f"{name} must be a {expected_type.__name__}, got "
            f"{type(value).__name__}."
        )
    return value


@dataclass(frozen=True)
class SAMCoreMLProfile:
    """Resolved finite Core ML profile for one LibreSAM family and size."""

    family: str
    size: str
    precision: str
    prompt_max_points: int

    def __post_init__(self) -> None:
        family = str(self.family).strip().lower()
        spec = _FAMILY_SPECS.get(family)
        if spec is None:
            raise ValueError(
                f"Invalid LibreSAM Core ML family {self.family!r}; expected one "
                f"of {', '.join(_FAMILY_SPECS)}."
            )
        size = str(self.size).strip().lower()
        if size not in spec.sizes:
            raise ValueError(
                f"Invalid {family} Core ML size {self.size!r}; expected one of "
                f"{', '.join(spec.sizes)}."
            )
        if self.precision != "fp32":
            raise ValueError("LibreSAM Core ML profiles must use FP32.")
        point_max = _strict_int(
            self.prompt_max_points,
            name="prompt_max_points",
            minimum=2,
            maximum=SAM_COREML_MAX_POINTS_LIMIT,
        )
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "prompt_max_points", point_max)

    @property
    def spec(self) -> _SAMFamilySpec:
        return _FAMILY_SPECS[self.family]

    @property
    def image_size(self) -> int:
        return self.spec.image_size

    @property
    def embedding_names(self) -> tuple[str, ...]:
        return self.spec.embedding_names

    @property
    def embedding_shapes(self) -> tuple[tuple[int, int, int, int], ...]:
        return self.spec.embedding_shapes

    @property
    def low_res_mask_size(self) -> int:
        return self.spec.low_res_mask_size

    def as_dict(self) -> dict[str, Any]:
        spec = self.spec
        return {
            "family": self.family,
            "size": self.size,
            "precision": self.precision,
            "image_batch": SAM_COREML_IMAGE_BATCH,
            "query_batch": SAM_COREML_QUERY_BATCH,
            "prompt_min_points": 1,
            "prompt_max_points": self.prompt_max_points,
            "image_size": spec.image_size,
            "embedding_names": list(spec.embedding_names),
            "embedding_shapes": [list(shape) for shape in spec.embedding_shapes],
            "low_res_mask_size": spec.low_res_mask_size,
            "preprocess_contract": spec.preprocess_contract,
            "host_contract": dict(_SAM_COREML_HOST_CONTRACTS[self.family]),
            "postprocess_size_metadata": list(spec.postprocess_size_metadata),
            "iou_encoding": spec.iou_encoding,
            "encoder_rewrite": spec.encoder_rewrite,
            "visual_only": spec.visual_only,
            "weights_license": spec.weights_license,
            "redistributable": spec.redistributable,
            "native_outputs_omitted": list(spec.native_outputs_omitted),
            "release_notice_gap": spec.release_notice_gap,
        }


def validate_sam_coreml_profile(
    *,
    family: str | None,
    size: str | None,
    prompt_max_points: int,
    precision: str = "fp32",
) -> SAMCoreMLProfile:
    """Resolve a strict profile and reject unsupported or lossy variants."""

    resolved_family = "" if family is None else str(family).strip().lower()
    if resolved_family not in _FAMILY_SPECS:
        raise NotImplementedError(
            "LibreSAM Core ML split components support EdgeTAM, MobileSAM, "
            f"SAM-1, SAM-2, and SAM3 visual; got family={family!r}."
        )
    resolved_size = "" if size is None else str(size).strip().lower()
    if resolved_size not in _FAMILY_SPECS[resolved_family].sizes:
        raise NotImplementedError(
            f"{resolved_family} Core ML export does not support size={size!r}; "
            f"expected one of {', '.join(_FAMILY_SPECS[resolved_family].sizes)}."
        )
    if precision != "fp32":
        raise NotImplementedError(
            "LibreSAM Core ML export is FP32-only; prompt coordinates, boundary "
            f"logits, and predicted-IoU behavior are not qualified for {precision!r}."
        )
    return SAMCoreMLProfile(
        family=resolved_family,
        size=resolved_size,
        precision=precision,
        prompt_max_points=prompt_max_points,
    )


def _fixed_axis(axis: str, value: int) -> dict[str, Any]:
    return {"axis": axis, "kind": "fixed", "value": value}


def _shape_axes(shape: Sequence[int]) -> list[dict[str, Any]]:
    names = ("N", "C", "H", "W")
    return [_fixed_axis(name, int(value)) for name, value in zip(names, shape)]


def _point_axis(profile: SAMCoreMLProfile) -> dict[str, Any]:
    return {
        "axis": "P",
        "kind": "range",
        "lower_bound": 1,
        "upper_bound": profile.prompt_max_points,
        "default": 1,
        "padding": "forbidden",
    }


def sam_coreml_encoder_input_contract(profile: SAMCoreMLProfile) -> dict[str, Any]:
    """Return the fixed, host-preprocessed encoder input boundary."""

    size = profile.image_size
    return {
        "name": SAM_COREML_ENCODER_INPUT,
        "kind": "tensor",
        "dtype": "float32",
        "layout": "NCHW",
        "color": "rgb",
        "range": "family_native_standardized",
        "shape": [
            _fixed_axis("N", 1),
            _fixed_axis("C", 3),
            _fixed_axis("H", size),
            _fixed_axis("W", size),
        ],
        "preprocess_owner": "host",
        "preprocess_contract": profile.spec.preprocess_contract,
    }


def _embedding_contracts(profile: SAMCoreMLProfile) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "kind": "tensor",
            "dtype": "float32",
            "role": "image_embedding",
            "encoding": "raw_features",
            "shape": _shape_axes(shape),
        }
        for name, shape in zip(profile.embedding_names, profile.embedding_shapes)
    ]


def _decoder_function_parts(function_name: str) -> tuple[str, str]:
    if function_name not in SAM_COREML_DECODER_FUNCTIONS:
        raise ValueError(f"Unknown LibreSAM Core ML decoder {function_name!r}.")
    suffix = function_name.removeprefix("decode_")
    for mask_mode in SAM_COREML_MASK_MODES:
        marker = f"_{mask_mode}"
        if suffix.endswith(marker):
            prompt_mode = suffix[: -len(marker)]
            if prompt_mode in SAM_COREML_PROMPT_MODES:
                return prompt_mode, mask_mode
    raise ValueError(f"Malformed LibreSAM Core ML decoder name {function_name!r}.")


def _decoder_inputs(
    profile: SAMCoreMLProfile,
    prompt_mode: str,
) -> list[dict[str, Any]]:
    result = _embedding_contracts(profile)
    if prompt_mode in ("points", "points_boxes"):
        result.extend(
            [
                {
                    "name": SAM_COREML_POINT_COORDS_INPUT,
                    "kind": "tensor",
                    "dtype": "float32",
                    "role": "point_coordinates",
                    "coordinate_space": "model_input_xy",
                    "coordinate_min": 0.0,
                    "coordinate_max": float(profile.image_size),
                    "shape": [
                        _fixed_axis("N", 1),
                        _fixed_axis("Q", 1),
                        _point_axis(profile),
                        _fixed_axis("XY", 2),
                    ],
                },
                {
                    "name": SAM_COREML_POINT_LABELS_INPUT,
                    "kind": "tensor",
                    "dtype": "int32",
                    "role": "point_labels",
                    "allowed_values": [0, 1],
                    "shape": [
                        _fixed_axis("N", 1),
                        _fixed_axis("Q", 1),
                        _point_axis(profile),
                    ],
                },
            ]
        )
    if prompt_mode in ("boxes", "points_boxes"):
        result.append(
            {
                "name": SAM_COREML_BOXES_INPUT,
                "kind": "tensor",
                "dtype": "float32",
                "role": "box_coordinates",
                "coordinate_space": "model_input_xyxy",
                "coordinate_min": 0.0,
                "coordinate_max": float(profile.image_size),
                "ordering": "x1_y1_x2_y2",
                "shape": [
                    _fixed_axis("N", 1),
                    _fixed_axis("Q", 1),
                    _fixed_axis("XYXY", 4),
                ],
            }
        )
    return result


def _decoder_outputs(
    profile: SAMCoreMLProfile,
    mask_mode: str,
) -> list[dict[str, Any]]:
    masks = 1 if mask_mode == "single" else SAM_COREML_NUM_MULTIMASK_OUTPUTS
    size = profile.low_res_mask_size
    return [
        {
            "name": SAM_COREML_MASKS_OUTPUT,
            "kind": "tensor",
            "dtype": "float32",
            "role": "mask_logits",
            "encoding": "raw_logits",
            "threshold": 0.0,
            "shape": [
                _fixed_axis("N", 1),
                _fixed_axis("Q", 1),
                _fixed_axis("M", masks),
                _fixed_axis("H", size),
                _fixed_axis("W", size),
            ],
        },
        {
            "name": SAM_COREML_IOU_OUTPUT,
            "kind": "tensor",
            "dtype": "float32",
            "role": "predicted_iou",
            "encoding": profile.spec.iou_encoding,
            "shape": [
                _fixed_axis("N", 1),
                _fixed_axis("Q", 1),
                _fixed_axis("M", masks),
            ],
        },
    ]


def sam_coreml_function_contracts(
    profile: SAMCoreMLProfile,
) -> dict[str, dict[str, Any]]:
    """Return all seven exact component function descriptors."""

    functions: dict[str, dict[str, Any]] = {
        SAM_COREML_ENCODER_FUNCTION: {
            "component": "encoder",
            "inputs": [sam_coreml_encoder_input_contract(profile)],
            "outputs": _embedding_contracts(profile),
            "capture": "torch_jit_trace_fixed",
        }
    }
    for function_name in SAM_COREML_DECODER_FUNCTIONS:
        prompt_mode, mask_mode = _decoder_function_parts(function_name)
        functions[function_name] = {
            "component": "decoder",
            "prompt_mode": prompt_mode,
            "mask_mode": mask_mode,
            "inputs": _decoder_inputs(profile, prompt_mode),
            "outputs": _decoder_outputs(profile, mask_mode),
            "capture": "torch_export_dynamic_points",
        }
    return functions


def sam_coreml_decoder_dynamic_shapes(
    profile: SAMCoreMLProfile,
    function_name: str,
) -> tuple[tuple[dict[int, Any], ...]]:
    """Return the positional ``torch.export`` dynamic-shape tuple.

    Point coordinates and labels share one symbolic ``P``.  Box-only decoders
    are still captured separately but have no dynamic axes.  Callers should
    pass this tuple to ``torch.export.export(..., dynamic_shapes=...)`` and then
    run ``ExportedProgram.run_decompositions({})`` before Core ML conversion.
    """

    prompt_mode, _ = _decoder_function_parts(function_name)
    inputs = _decoder_inputs(profile, prompt_mode)
    point_dim = torch.export.Dim(
        "P",
        min=1,
        max=profile.prompt_max_points,
    )
    dynamic_shapes: list[dict[int, Any]] = []
    for item in inputs:
        if item["name"] in (
            SAM_COREML_POINT_COORDS_INPUT,
            SAM_COREML_POINT_LABELS_INPUT,
        ):
            dynamic_shapes.append({2: point_dim})
        else:
            dynamic_shapes.append({})
    # ``SAMCoreMLDecoder.forward`` uses ``*inputs``.  torch.export represents
    # that variadic parameter as one tuple pytree, so dynamic_shapes needs the
    # matching one-element outer tuple.
    return (tuple(dynamic_shapes),)


def sam_coreml_metadata(profile: SAMCoreMLProfile) -> dict[str, Any]:
    """Build the strict bundle manifest persisted into every component."""

    functions = sam_coreml_function_contracts(profile)
    spec = profile.spec
    return {
        "artifact_scope": SAM_COREML_ARTIFACT_SCOPE,
        "component_contract": SAM_COREML_COMPONENT_CONTRACT,
        "sam_coreml_schema_version": SAM_COREML_PIPELINE_SCHEMA_VERSION,
        "coreml_multifunction": True,
        "coreml_default_function": SAM_COREML_ENCODER_FUNCTION,
        "model_family": profile.family,
        "size": profile.size,
        "task": "segment",
        "precision": "fp32",
        "coreml_minimum_deployment_target": SAM_COREML_MINIMUM_DEPLOYMENT_TARGET,
        "coreml_function_names": list(SAM_COREML_FUNCTION_NAMES),
        "prompt_modes": list(SAM_COREML_PROMPT_MODES),
        "mask_modes": list(SAM_COREML_MASK_MODES),
        "sam_coreml_profile": profile.as_dict(),
        "sam_coreml_functions": functions,
        "sam_coreml_functions_sha256": _canonical_sha256(functions),
        "host_operations": list(SAM_COREML_HOST_OPERATIONS),
        "native_outputs_omitted": list(spec.native_outputs_omitted),
        "mask_encoding": "raw_logits",
        "mask_threshold": 0.0,
        "weights_license": spec.weights_license,
        "artifact_redistributable": spec.redistributable,
        "sam3_visual_only": spec.visual_only,
        "sam3_pcs_included": False,
        "release_notice_gap": spec.release_notice_gap,
    }


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def validate_sam_coreml_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate native or Core-ML-stringified bundle metadata."""

    if not isinstance(metadata, Mapping):
        raise ValueError("LibreSAM Core ML metadata must be a mapping.")
    profile_data = _json_field(metadata, "sam_coreml_profile", dict)
    try:
        profile = SAMCoreMLProfile(
            family=profile_data["family"],
            size=profile_data["size"],
            precision=profile_data["precision"],
            prompt_max_points=profile_data["prompt_max_points"],
        )
    except KeyError as exc:
        raise ValueError(
            f"sam_coreml_profile is missing required field {exc.args[0]!r}."
        ) from exc
    expected = sam_coreml_metadata(profile)

    scalar_fields = (
        "artifact_scope",
        "component_contract",
        "sam_coreml_schema_version",
        "model_family",
        "size",
        "task",
        "precision",
        "coreml_minimum_deployment_target",
        "sam_coreml_functions_sha256",
        "mask_encoding",
        "mask_threshold",
        "weights_license",
        "coreml_default_function",
    )
    for name in scalar_fields:
        value = metadata.get(name)
        expected_value = expected[name]
        if isinstance(expected_value, int) and isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                pass
        elif isinstance(expected_value, float) and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        if value != expected_value:
            raise ValueError(
                f"{name} conflicts with the strict LibreSAM Core ML contract: "
                f"expected {expected_value!r}, got {metadata.get(name)!r}."
            )

    for name in (
        "coreml_function_names",
        "prompt_modes",
        "mask_modes",
        "host_operations",
        "native_outputs_omitted",
    ):
        value = _json_field(metadata, name, list)
        if value != expected[name]:
            raise ValueError(f"{name} conflicts with the strict component manifest.")

    actual_profile = _json_field(metadata, "sam_coreml_profile", dict)
    if actual_profile != expected["sam_coreml_profile"]:
        raise ValueError("sam_coreml_profile contains inconsistent derived fields.")
    actual_functions = _json_field(metadata, "sam_coreml_functions", dict)
    if actual_functions != expected["sam_coreml_functions"]:
        raise ValueError("sam_coreml_functions conflicts with the exact graph ABI.")
    actual_hash = str(metadata.get("sam_coreml_functions_sha256", ""))
    computed_hash = _canonical_sha256(actual_functions)
    if not hmac.compare_digest(actual_hash, computed_hash):
        raise ValueError("sam_coreml_functions_sha256 does not match the manifest.")

    for name in (
        "artifact_redistributable",
        "coreml_multifunction",
        "sam3_visual_only",
        "sam3_pcs_included",
    ):
        if _strict_bool(metadata.get(name), name=name) is not expected[name]:
            raise ValueError(f"{name} conflicts with the family license contract.")
    release_gap = metadata.get("release_notice_gap")
    if release_gap in ("null", "None", ""):
        release_gap = None
    if release_gap != expected["release_notice_gap"]:
        raise ValueError("release_notice_gap conflicts with the family profile.")
    return expected


def _unwrap_model(model: nn.Module) -> nn.Module:
    candidate = getattr(model, "model", None)
    if isinstance(candidate, nn.Module):
        return candidate
    if not isinstance(model, nn.Module):
        raise ValueError("LibreSAM Core ML conversion requires a torch.nn.Module.")
    return model


@dataclass(frozen=True)
class SAMCoreMLModelSignature:
    """Static graph facts checked before any expensive conversion."""

    family: str
    model_type: str | None
    embedding_shapes: tuple[tuple[int, int, int, int], ...]
    low_res_mask_size: int
    num_multimask_outputs: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "model_type": self.model_type,
            "embedding_shapes": [list(shape) for shape in self.embedding_shapes],
            "low_res_mask_size": self.low_res_mask_size,
            "num_multimask_outputs": self.num_multimask_outputs,
        }


def inspect_sam_coreml_model(
    model: nn.Module,
    *,
    profile: SAMCoreMLProfile,
) -> SAMCoreMLModelSignature:
    """Reject a graph whose family-level modules/config do not match profile."""

    graph = _unwrap_model(model)
    for name in ("prompt_encoder", "mask_decoder"):
        if not isinstance(getattr(graph, name, None), nn.Module):
            raise ValueError(f"LibreSAM Core ML graph is missing '.{name}'.")
    if profile.family == "mobilesam":
        if not isinstance(getattr(graph, "image_encoder", None), nn.Module):
            raise ValueError("MobileSAM Core ML graph is missing '.image_encoder'.")
        image_size = getattr(graph, "image_size", None)
        if image_size is not None and int(image_size) != profile.image_size:
            raise ValueError(
                "MobileSAM image_size does not match the fixed Core ML profile."
            )
        prompt_hidden = getattr(graph.prompt_encoder, "embed_dim", None)
        prompt_hw = getattr(graph.prompt_encoder, "image_embedding_size", None)
        if prompt_hidden is not None and int(prompt_hidden) != 256:
            raise ValueError("MobileSAM prompt embedding width must be 256.")
        if prompt_hw is not None and tuple(prompt_hw) != (64, 64):
            raise ValueError("MobileSAM prompt embedding grid must be 64x64.")
        model_type = None
    else:
        if not callable(getattr(graph, "get_image_embeddings", None)):
            raise ValueError(
                f"{profile.family} Core ML graph lacks get_image_embeddings()."
            )
        config = getattr(graph, "config", None)
        model_type = getattr(config, "model_type", None)
        if model_type not in profile.spec.model_types:
            raise ValueError(
                f"{profile.family} Core ML graph has model_type={model_type!r}; "
                f"expected one of {profile.spec.model_types!r}."
            )
        prompt_config = getattr(config, "prompt_encoder_config", None)
        prompt_image_size = getattr(prompt_config, "image_size", None)
        prompt_hidden = getattr(prompt_config, "hidden_size", None)
        if (
            prompt_image_size is not None
            and int(prompt_image_size) != profile.image_size
        ):
            raise ValueError(
                f"{profile.family} prompt image size does not match the profile."
            )
        if prompt_hidden is not None and int(prompt_hidden) != 256:
            raise ValueError(
                f"{profile.family} prompt embedding width must be 256."
            )
        if profile.family == "sam":
            vision_config = getattr(config, "vision_config", None)
            vision_image_size = getattr(vision_config, "image_size", None)
            output_channels = getattr(vision_config, "output_channels", None)
            if (
                vision_image_size is not None
                and int(vision_image_size) != profile.image_size
            ):
                raise ValueError("SAM-1 vision image size does not match the profile.")
            if output_channels is not None and int(output_channels) != 256:
                raise ValueError("SAM-1 image embedding width must be 256.")

    decoder = graph.mask_decoder
    num_masks = getattr(decoder, "num_multimask_outputs", None)
    if num_masks is None:
        config = getattr(graph, "config", None)
        mask_config = getattr(config, "mask_decoder_config", None)
        num_masks = getattr(mask_config, "num_multimask_outputs", None)
    if num_masks is not None and int(num_masks) != SAM_COREML_NUM_MULTIMASK_OUTPUTS:
        raise ValueError(
            "LibreSAM Core ML requires exactly three native multimask outputs; "
            f"graph declares {num_masks!r}."
        )

    if profile.family in ("edgetam", "sam2", "sam3"):
        feature_sizes = getattr(graph, "backbone_feature_sizes", None)
        expected_hw = [list(shape[-2:]) for shape in profile.embedding_shapes]
        if (
            feature_sizes is not None
            and [list(v) for v in feature_sizes] != expected_hw
        ):
            raise ValueError(
                f"{profile.family} backbone feature sizes do not match profile: "
                f"expected {expected_hw}, got {feature_sizes!r}."
            )

    return SAMCoreMLModelSignature(
        family=profile.family,
        model_type=model_type,
        embedding_shapes=profile.embedding_shapes,
        low_res_mask_size=profile.low_res_mask_size,
        num_multimask_outputs=SAM_COREML_NUM_MULTIMASK_OUTPUTS,
    )


def _sam2_fixed_get_pos_embed(backbone: nn.Module, _hw: Any) -> torch.Tensor:
    return getattr(backbone, _SAM2_FIXED_POSITION_BUFFER)


def freeze_sam2_coreml_position_embedding(
    model: nn.Module,
    *,
    profile: SAMCoreMLProfile,
) -> nn.Module:
    """Freeze SAM2's exact native Hiera positional tensor for fixed 1024 input.

    The tensor is calculated by the loaded model itself before replacement, so
    each size retains its own learned ``pos_embed``/``pos_embed_window`` values.
    The operation intentionally mutates the export-only graph instance.
    """

    if profile.family != "sam2":
        raise ValueError("SAM2 positional freezing requires a sam2 profile.")
    graph = _unwrap_model(model)
    vision_encoder = getattr(graph, "vision_encoder", None)
    backbone = getattr(vision_encoder, "backbone", None)
    if not isinstance(backbone, nn.Module):
        raise ValueError("SAM2 graph is missing vision_encoder.backbone.")
    get_pos_embed = getattr(backbone, "_get_pos_embed", None)
    if not callable(get_pos_embed):
        raise ValueError("SAM2 Hiera backbone is missing _get_pos_embed().")

    expected_hw = tuple(profile.embedding_shapes[0][-2:])
    if hasattr(backbone, _SAM2_FIXED_POSITION_BUFFER):
        fixed = getattr(backbone, _SAM2_FIXED_POSITION_BUFFER)
        if tuple(fixed.shape[1:3]) != expected_hw:
            raise ValueError(
                "Existing SAM2 fixed positional tensor has the wrong spatial shape."
            )
        return graph

    with torch.no_grad():
        fixed = get_pos_embed(expected_hw).detach().clone()
    if tuple(fixed.shape[1:3]) != expected_hw:
        raise ValueError(
            "SAM2 native positional tensor shape does not match the fixed "
            f"encoder frame: expected {expected_hw}, got {tuple(fixed.shape[1:3])}."
        )
    setattr(backbone, _SAM2_ORIGINAL_GET_POS_EMBED, get_pos_embed)
    backbone.register_buffer(
        _SAM2_FIXED_POSITION_BUFFER,
        fixed,
        persistent=False,
    )
    backbone._get_pos_embed = MethodType(_sam2_fixed_get_pos_embed, backbone)
    return graph


class SAMCoreMLEncoder(nn.Module):
    """Fixed model-ready image encoder used by the split bundle."""

    def __init__(self, model: nn.Module, profile: SAMCoreMLProfile) -> None:
        super().__init__()
        self.profile = profile
        self.model = _unwrap_model(model).eval()
        if profile.family == "sam2":
            freeze_sam2_coreml_position_embedding(self.model, profile=profile)
        self.input_names = (SAM_COREML_ENCODER_INPUT,)
        self.output_names = profile.embedding_names

    def forward(self, pixel_values: torch.Tensor):
        if self.profile.family == "mobilesam":
            return self.model.image_encoder(pixel_values)
        embeddings = self.model.get_image_embeddings(pixel_values)
        if len(self.profile.embedding_names) == 1:
            if isinstance(embeddings, (tuple, list)):
                return embeddings[0]
            return embeddings
        return tuple(embeddings)


class SAMCoreMLDecoder(nn.Module):
    """One prompt/mask-mode decoder with an exact positional input order."""

    def __init__(
        self,
        model: nn.Module,
        profile: SAMCoreMLProfile,
        *,
        prompt_mode: str,
        mask_mode: str,
    ) -> None:
        super().__init__()
        if prompt_mode not in SAM_COREML_PROMPT_MODES:
            raise ValueError(f"Invalid prompt_mode={prompt_mode!r}.")
        if mask_mode not in SAM_COREML_MASK_MODES:
            raise ValueError(f"Invalid mask_mode={mask_mode!r}.")
        self.model = _unwrap_model(model).eval()
        self.profile = profile
        self.prompt_mode = prompt_mode
        self.mask_mode = mask_mode
        self.multimask_output = mask_mode == "multimask"
        function_name = f"decode_{prompt_mode}_{mask_mode}"
        contract = sam_coreml_function_contracts(profile)[function_name]
        self.input_names = tuple(item["name"] for item in contract["inputs"])
        self.output_names = tuple(item["name"] for item in contract["outputs"])

        if profile.family == "mobilesam":
            with torch.no_grad():
                dense_pe = self.model.prompt_encoder.get_dense_pe().detach().clone()
            self.register_buffer(
                "_image_positional_embeddings",
                dense_pe,
                persistent=False,
            )

    def _split_inputs(self, inputs):
        embedding_count = len(self.profile.embedding_names)
        embeddings = tuple(inputs[:embedding_count])
        cursor = embedding_count
        points = labels = boxes = None
        if self.prompt_mode in ("points", "points_boxes"):
            points = inputs[cursor]
            labels = inputs[cursor + 1]
            cursor += 2
        if self.prompt_mode in ("boxes", "points_boxes"):
            boxes = inputs[cursor]
            cursor += 1
        if cursor != len(inputs):
            raise ValueError(
                f"{self.prompt_mode} decoder expected {cursor} inputs, "
                f"received {len(inputs)}."
            )
        return embeddings, points, labels, boxes

    def _mobile_prompt_embeddings(self, points, labels, boxes):
        encoder = self.model.prompt_encoder
        sparse_parts = []
        if points is not None:
            coords = points[0] + 0.5
            point_labels = labels[0]
            if boxes is None:
                coords = torch.cat(
                    (
                        coords,
                        torch.zeros(
                            (coords.shape[0], 1, 2),
                            dtype=coords.dtype,
                            device=coords.device,
                        ),
                    ),
                    dim=1,
                )
                point_labels = torch.cat(
                    (
                        point_labels,
                        -torch.ones(
                            (point_labels.shape[0], 1),
                            dtype=point_labels.dtype,
                            device=point_labels.device,
                        ),
                    ),
                    dim=1,
                )
            point_embeddings = encoder.pe_layer.forward_with_coords(
                coords,
                encoder.input_image_size,
            )
            negative = (point_labels == -1).unsqueeze(-1)
            zero = (point_labels == 0).unsqueeze(-1)
            positive = (point_labels == 1).unsqueeze(-1)
            point_embeddings = torch.where(
                negative,
                torch.zeros_like(point_embeddings),
                point_embeddings,
            )
            point_embeddings = (
                point_embeddings
                + negative.to(point_embeddings.dtype)
                * encoder.not_a_point_embed.weight
                + zero.to(point_embeddings.dtype) * encoder.point_embeddings[0].weight
                + positive.to(point_embeddings.dtype)
                * encoder.point_embeddings[1].weight
            )
            sparse_parts.append(point_embeddings)
        if boxes is not None:
            coords = (boxes[0] + 0.5).reshape(-1, 2, 2)
            box_embeddings = encoder.pe_layer.forward_with_coords(
                coords,
                encoder.input_image_size,
            )
            corner_weights = torch.cat(
                (
                    encoder.point_embeddings[2].weight,
                    encoder.point_embeddings[3].weight,
                ),
                dim=0,
            ).unsqueeze(0)
            sparse_parts.append(box_embeddings + corner_weights)

        sparse_embeddings = (
            sparse_parts[0]
            if len(sparse_parts) == 1
            else torch.cat(sparse_parts, dim=1)
        )
        batch = sparse_embeddings.shape[0]
        dense_embeddings = encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch,
            -1,
            encoder.image_embedding_size[0],
            encoder.image_embedding_size[1],
        )
        return sparse_embeddings, dense_embeddings

    def _mobile_forward(self, embeddings, points, labels, boxes):
        sparse, dense = self._mobile_prompt_embeddings(points, labels, boxes)
        masks, scores = self.model.mask_decoder(
            image_embeddings=embeddings[0],
            image_pe=self._image_positional_embeddings,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=self.multimask_output,
        )
        return masks.unsqueeze(0), scores.unsqueeze(0)

    def _hf_forward(self, embeddings, points, labels, boxes):
        image_embeddings: Any = (
            embeddings[0] if len(embeddings) == 1 else list(embeddings)
        )
        outputs = self.model(
            image_embeddings=image_embeddings,
            input_points=points,
            input_labels=labels,
            input_boxes=boxes,
            multimask_output=self.multimask_output,
        )
        return outputs.pred_masks, outputs.iou_scores

    def forward(self, *inputs):
        embeddings, points, labels, boxes = self._split_inputs(inputs)
        if self.profile.family == "mobilesam":
            return self._mobile_forward(embeddings, points, labels, boxes)
        return self._hf_forward(embeddings, points, labels, boxes)


def wrap_sam_coreml_components(
    model: nn.Module,
    *,
    profile: SAMCoreMLProfile,
) -> dict[str, nn.Module]:
    """Return the encoder and six exact decoder graphs for conversion."""

    inspect_sam_coreml_model(model, profile=profile)
    graph = _unwrap_model(model).eval()
    components: dict[str, nn.Module] = {
        SAM_COREML_ENCODER_FUNCTION: SAMCoreMLEncoder(graph, profile).eval()
    }
    for prompt_mode in SAM_COREML_PROMPT_MODES:
        for mask_mode in SAM_COREML_MASK_MODES:
            name = f"decode_{prompt_mode}_{mask_mode}"
            components[name] = SAMCoreMLDecoder(
                graph,
                profile,
                prompt_mode=prompt_mode,
                mask_mode=mask_mode,
            ).eval()
    validate_sam_coreml_component_graphs(components, profile=profile)
    return components


def validate_sam_coreml_component_graphs(
    components: Mapping[str, nn.Module],
    *,
    profile: SAMCoreMLProfile,
) -> None:
    """Check graph/function membership and exact ordered tensor names."""

    if list(components) != list(SAM_COREML_FUNCTION_NAMES):
        raise ValueError(
            "LibreSAM Core ML component graphs must contain exactly "
            f"{SAM_COREML_FUNCTION_NAMES!r} in manifest order."
        )
    contracts = sam_coreml_function_contracts(profile)
    for name, component in components.items():
        if not isinstance(component, nn.Module):
            raise ValueError(f"Core ML component {name!r} is not an nn.Module.")
        expected_inputs = tuple(item["name"] for item in contracts[name]["inputs"])
        expected_outputs = tuple(item["name"] for item in contracts[name]["outputs"])
        if tuple(getattr(component, "input_names", ())) != expected_inputs:
            raise ValueError(f"{name} graph input names do not match its contract.")
        if tuple(getattr(component, "output_names", ())) != expected_outputs:
            raise ValueError(f"{name} graph output names do not match its contract.")
        component_profile = getattr(component, "profile", None)
        if component_profile != profile:
            raise ValueError(f"{name} graph was built for a different profile.")


def _require_tensor(
    value: Any,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int | None, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor.")
    if value.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}, got {value.dtype}.")
    if value.ndim != len(shape):
        raise ValueError(f"{name} must have rank {len(shape)}, got {value.ndim}.")
    for axis, (actual, expected) in enumerate(zip(value.shape, shape)):
        if expected is not None and int(actual) != expected:
            raise ValueError(
                f"{name} axis {axis} must be {expected}, got {int(actual)}."
            )
    if dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinity.")
    return value


def _exact_named_values(
    values: Mapping[str, Any],
    expected_names: Sequence[str],
    *,
    kind: str,
) -> None:
    if not isinstance(values, Mapping):
        raise ValueError(f"Core ML {kind} values must be a name-to-tensor mapping.")
    if list(values) != list(expected_names):
        raise ValueError(
            f"Core ML {kind} names/order must be {list(expected_names)!r}, "
            f"got {list(values)!r}."
        )


def validate_sam_coreml_function_io(
    function_name: str,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    *,
    profile: SAMCoreMLProfile,
) -> None:
    """Validate actual graph/runtime tensors by declared name and semantics."""

    contracts = sam_coreml_function_contracts(profile)
    if function_name not in contracts:
        raise ValueError(f"Unknown LibreSAM Core ML function {function_name!r}.")
    contract = contracts[function_name]
    input_names = [item["name"] for item in contract["inputs"]]
    output_names = [item["name"] for item in contract["outputs"]]
    _exact_named_values(inputs, input_names, kind="input")
    _exact_named_values(outputs, output_names, kind="output")

    if function_name == SAM_COREML_ENCODER_FUNCTION:
        size = profile.image_size
        _require_tensor(
            inputs[SAM_COREML_ENCODER_INPUT],
            name=SAM_COREML_ENCODER_INPUT,
            dtype=torch.float32,
            shape=(1, 3, size, size),
        )
        for name, shape in zip(profile.embedding_names, profile.embedding_shapes):
            _require_tensor(
                outputs[name],
                name=name,
                dtype=torch.float32,
                shape=shape,
            )
        return

    prompt_mode, mask_mode = _decoder_function_parts(function_name)
    for name, shape in zip(profile.embedding_names, profile.embedding_shapes):
        _require_tensor(
            inputs[name],
            name=name,
            dtype=torch.float32,
            shape=shape,
        )

    point_count = None
    if prompt_mode in ("points", "points_boxes"):
        coords = _require_tensor(
            inputs[SAM_COREML_POINT_COORDS_INPUT],
            name=SAM_COREML_POINT_COORDS_INPUT,
            dtype=torch.float32,
            shape=(1, 1, None, 2),
        )
        labels = _require_tensor(
            inputs[SAM_COREML_POINT_LABELS_INPUT],
            name=SAM_COREML_POINT_LABELS_INPUT,
            dtype=torch.int32,
            shape=(1, 1, None),
        )
        point_count = int(coords.shape[2])
        if int(labels.shape[2]) != point_count:
            raise ValueError("point_coords and point_labels must share P.")
        if not 1 <= point_count <= profile.prompt_max_points:
            raise ValueError(
                f"Point count P must be in [1, {profile.prompt_max_points}], "
                f"got {point_count}; sentinel padding is forbidden."
            )
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("point_labels may contain only 0 or 1.")
        if bool((coords < 0).any()) or bool((coords > profile.image_size).any()):
            raise ValueError("point_coords fall outside the model-coordinate canvas.")

    if prompt_mode in ("boxes", "points_boxes"):
        boxes = _require_tensor(
            inputs[SAM_COREML_BOXES_INPUT],
            name=SAM_COREML_BOXES_INPUT,
            dtype=torch.float32,
            shape=(1, 1, 4),
        )
        if bool((boxes < 0).any()) or bool((boxes > profile.image_size).any()):
            raise ValueError("boxes fall outside the model-coordinate canvas.")
        if not bool(
            (boxes[..., 2] >= boxes[..., 0]).all()
            and (boxes[..., 3] >= boxes[..., 1]).all()
        ):
            raise ValueError("boxes must use ordered x1,y1,x2,y2 coordinates.")

    masks = 1 if mask_mode == "single" else SAM_COREML_NUM_MULTIMASK_OUTPUTS
    size = profile.low_res_mask_size
    _require_tensor(
        outputs[SAM_COREML_MASKS_OUTPUT],
        name=SAM_COREML_MASKS_OUTPUT,
        dtype=torch.float32,
        shape=(1, 1, masks, size, size),
    )
    iou = _require_tensor(
        outputs[SAM_COREML_IOU_OUTPUT],
        name=SAM_COREML_IOU_OUTPUT,
        dtype=torch.float32,
        shape=(1, 1, masks),
    )
    if profile.spec.iou_encoding == "sigmoid_probability":
        tolerance = 1e-6
        if bool((iou < -tolerance).any()) or bool((iou > 1.0 + tolerance).any()):
            raise ValueError("iou_scores must be in [0, 1] for this family.")


def validate_sam_coreml_graph_signature(
    function_name: str,
    *,
    input_specs: Sequence[Mapping[str, Any]],
    output_specs: Sequence[Mapping[str, Any]],
    profile: SAMCoreMLProfile,
) -> None:
    """Check converted/traced graph feature descriptions against the manifest.

    ``input_specs`` and ``output_specs`` are normalized dictionaries containing
    at least ``name``, ``dtype``, and ``shape``.  Range axes must retain their
    complete bounded descriptor rather than being collapsed to a default shape.
    """

    contract = sam_coreml_function_contracts(profile).get(function_name)
    if contract is None:
        raise ValueError(f"Unknown LibreSAM Core ML function {function_name!r}.")
    for kind, actual, expected in (
        ("input", list(input_specs), contract["inputs"]),
        ("output", list(output_specs), contract["outputs"]),
    ):
        normalized_expected = [
            {
                "name": item["name"],
                "dtype": item["dtype"],
                "shape": item["shape"],
            }
            for item in expected
        ]
        normalized_actual = [
            {
                "name": item.get("name"),
                "dtype": item.get("dtype"),
                "shape": item.get("shape"),
            }
            for item in actual
        ]
        if normalized_actual != normalized_expected:
            raise ValueError(
                f"{function_name} converted {kind} signature conflicts with "
                "the strict manifest."
            )


__all__ = [
    "SAM_COREML_ARTIFACT_SCOPE",
    "SAM_COREML_BOXES_INPUT",
    "SAM_COREML_COMPONENT_CONTRACT",
    "SAM_COREML_DECODER_FUNCTIONS",
    "SAM_COREML_DEFAULT_MAX_POINTS",
    "SAM_COREML_ENCODER_FUNCTION",
    "SAM_COREML_ENCODER_INPUT",
    "SAM_COREML_FUNCTION_NAMES",
    "SAM_COREML_HOST_OPERATIONS",
    "SAM_COREML_IMAGE_BATCH",
    "SAM_COREML_IOU_OUTPUT",
    "SAM_COREML_MASKS_OUTPUT",
    "SAM_COREML_MASK_MODES",
    "SAM_COREML_MAX_POINTS_LIMIT",
    "SAM_COREML_MINIMUM_DEPLOYMENT_TARGET",
    "SAM_COREML_NUM_MULTIMASK_OUTPUTS",
    "SAM_COREML_PIPELINE_SCHEMA_VERSION",
    "SAM_COREML_POINT_COORDS_INPUT",
    "SAM_COREML_POINT_LABELS_INPUT",
    "SAM_COREML_PROMPT_MODES",
    "SAM_COREML_QUERY_BATCH",
    "SAMCoreMLDecoder",
    "SAMCoreMLEncoder",
    "SAMCoreMLModelSignature",
    "SAMCoreMLProfile",
    "freeze_sam2_coreml_position_embedding",
    "inspect_sam_coreml_model",
    "sam_coreml_decoder_dynamic_shapes",
    "sam_coreml_encoder_input_contract",
    "sam_coreml_function_contracts",
    "sam_coreml_metadata",
    "validate_sam_coreml_component_graphs",
    "validate_sam_coreml_function_io",
    "validate_sam_coreml_graph_signature",
    "validate_sam_coreml_metadata",
    "validate_sam_coreml_profile",
    "wrap_sam_coreml_components",
]
