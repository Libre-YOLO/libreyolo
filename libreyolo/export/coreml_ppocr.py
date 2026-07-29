"""Strict Core ML component contract for LibrePPOCR.

LibrePPOCR is a host-orchestrated two-stage pipeline, not one image-to-text
graph.  Its Core ML artifact therefore contains two ML Program functions:

``detector``
    Consumes one already-resized and standardized BGR tensor and returns the
    DB shrink probability map.

``recognizer``
    Consumes a host-bucketed batch of BGR text crops normalized to ``[-1, 1]``
    and returns per-timestep CTC probabilities.

DB contour extraction, quad ordering/cropping, recognition bucketing, and CTC
decoding remain host operations.  Keeping those data-dependent operations out
of the graph preserves the native pipeline without pretending that conversion
alone produces an end-to-end OCR model.

This module deliberately has no ``coremltools`` dependency.  It supplies the
bounded flexible-shape ABI, graph adapters, and self-checking integrity
metadata that a specialized exporter/backend can consume.  The implementation
is derived only from LibreYOLO's Apache-2.0 PP-OCR port.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

PPOCR_COREML_COMPONENT_CONTRACT = "ppocr_det_rec_v1"
PPOCR_COREML_PIPELINE_SCHEMA_VERSION = 1
PPOCR_COREML_ARTIFACT_SCOPE = "host_orchestrated_pipeline_components"
PPOCR_COREML_MINIMUM_DEPLOYMENT_TARGETS = ("iOS18", "macOS15")

PPOCR_COREML_DETECTOR_FUNCTION = "detector"
PPOCR_COREML_RECOGNIZER_FUNCTION = "recognizer"
PPOCR_COREML_FUNCTION_NAMES = (
    PPOCR_COREML_DETECTOR_FUNCTION,
    PPOCR_COREML_RECOGNIZER_FUNCTION,
)
PPOCR_COREML_DEFAULT_FUNCTION = PPOCR_COREML_DETECTOR_FUNCTION

PPOCR_COREML_DETECTOR_INPUT = "detector_input"
PPOCR_COREML_DETECTOR_OUTPUT = "probability_map"
PPOCR_COREML_RECOGNIZER_INPUT = "recognizer_input"
PPOCR_COREML_RECOGNIZER_OUTPUT = "ctc_probabilities"

PPOCR_COREML_DETECTOR_BATCH = 1
PPOCR_COREML_DETECTOR_CHANNELS = 3
PPOCR_COREML_DETECTOR_MIN_SIDE = 32
PPOCR_COREML_DETECTOR_STRIDE = 32
PPOCR_COREML_DETECTOR_MAX_SIDE_LIMIT = 4000
PPOCR_COREML_DETECTOR_MEAN = (0.485, 0.456, 0.406)
PPOCR_COREML_DETECTOR_STD = (0.229, 0.224, 0.225)

PPOCR_COREML_RECOGNIZER_CHANNELS = 3
PPOCR_COREML_RECOGNIZER_HEIGHT = 48
PPOCR_COREML_RECOGNIZER_MIN_WIDTH = 320
PPOCR_COREML_RECOGNIZER_DEFAULT_BATCH_MAX = 6
PPOCR_COREML_RECOGNIZER_OVERFLOW_POLICY = "error"

PPOCR_COREML_HOST_OPERATIONS = (
    "detector_resize_and_normalize",
    "db_contours_and_quads",
    "quad_ordering_and_perspective_crops",
    "recognizer_resize_normalize_and_bucket",
    "ctc_decode",
)

_PPOCR_COREML_SIZES = frozenset({"t", "l"})
_PPOCR_PIPELINE_KEYS = frozenset(
    {
        "det_limit_side_len",
        "det_db_thresh",
        "det_db_box_thresh",
        "det_db_unclip_ratio",
        "rec_image_shape",
    }
)


def _strict_int(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result}.")
    return result


def _strict_number(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    if minimum is not None:
        invalid = result <= minimum if exclusive_minimum else result < minimum
        if invalid:
            comparator = "greater than" if exclusive_minimum else "at least"
            raise ValueError(f"{name} must be {comparator} {minimum}, got {result}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {result}.")
    return result


def ppocr_detector_tensor_upper_bound(det_limit_side_len: int) -> int:
    """Return the exact largest tensor side produced by ``det_resize``.

    Native preprocessing first caps the long side at ``det_limit_side_len``,
    applies the independent 4000-pixel safety cap, truncates the resize, and
    finally uses Python's round-to-even behavior at stride 32.  A limit of 48,
    for example, can therefore produce a 64-pixel tensor side.
    """

    limit = _strict_int(
        det_limit_side_len,
        name="det_limit_side_len",
        minimum=PPOCR_COREML_DETECTOR_MIN_SIDE,
    )
    effective_limit = min(limit, PPOCR_COREML_DETECTOR_MAX_SIDE_LIMIT)
    return max(
        int(round(effective_limit / PPOCR_COREML_DETECTOR_STRIDE))
        * PPOCR_COREML_DETECTOR_STRIDE,
        PPOCR_COREML_DETECTOR_MIN_SIDE,
    )


def ppocr_recognizer_timesteps(width: int) -> int:
    """Return the exact PP-OCRv5 CTC sequence length for an input width."""

    resolved = _strict_int(
        width,
        name="recognizer width",
        minimum=PPOCR_COREML_RECOGNIZER_MIN_WIDTH,
    )
    return (resolved + 3) // 8


def ppocr_recognizer_required_width(
    crop_height: int,
    crop_width: int,
) -> int:
    """Return the native padded bucket width required by one text crop."""

    height = _strict_int(crop_height, name="crop height", minimum=1)
    width = _strict_int(crop_width, name="crop width", minimum=1)
    minimum_ratio = PPOCR_COREML_RECOGNIZER_MIN_WIDTH / PPOCR_COREML_RECOGNIZER_HEIGHT
    return int(
        PPOCR_COREML_RECOGNIZER_HEIGHT * max(minimum_ratio, width / float(height))
    )


@dataclass(frozen=True)
class PPOCRCoreMLProfile:
    """Resolved, finite flexible-shape profile for both package functions."""

    size: str
    precision: str
    det_limit_side_len: int
    det_tensor_upper: int
    rec_batch_max: int
    rec_max_width: int

    def __post_init__(self) -> None:
        if self.size not in _PPOCR_COREML_SIZES:
            raise ValueError(f"Invalid LibrePPOCR Core ML size {self.size!r}.")
        if self.precision != "fp32":
            raise ValueError("LibrePPOCR Core ML profiles must use FP32.")
        limit = _strict_int(
            self.det_limit_side_len,
            name="det_limit_side_len",
            minimum=PPOCR_COREML_DETECTOR_MIN_SIDE,
        )
        expected_upper = ppocr_detector_tensor_upper_bound(limit)
        tensor_upper = _strict_int(
            self.det_tensor_upper,
            name="det_tensor_upper",
            minimum=PPOCR_COREML_DETECTOR_MIN_SIDE,
        )
        if tensor_upper != expected_upper:
            raise ValueError(
                "det_tensor_upper must equal the native detector resize bound "
                f"{expected_upper}, got {self.det_tensor_upper!r}."
            )
        batch_max = _strict_int(self.rec_batch_max, name="rec_batch_max", minimum=1)
        width_max = _strict_int(
            self.rec_max_width,
            name="rec_max_width",
            minimum=PPOCR_COREML_RECOGNIZER_MIN_WIDTH,
        )
        object.__setattr__(self, "det_limit_side_len", limit)
        object.__setattr__(self, "det_tensor_upper", tensor_upper)
        object.__setattr__(self, "rec_batch_max", batch_max)
        object.__setattr__(self, "rec_max_width", width_max)

    def as_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "precision": self.precision,
            "det_limit_side_len": self.det_limit_side_len,
            "det_min_side": PPOCR_COREML_DETECTOR_MIN_SIDE,
            "det_tensor_upper": self.det_tensor_upper,
            "det_stride": PPOCR_COREML_DETECTOR_STRIDE,
            "det_max_side_limit": PPOCR_COREML_DETECTOR_MAX_SIDE_LIMIT,
            "rec_batch_max": self.rec_batch_max,
            "rec_channels": PPOCR_COREML_RECOGNIZER_CHANNELS,
            "rec_height": PPOCR_COREML_RECOGNIZER_HEIGHT,
            "rec_min_width": PPOCR_COREML_RECOGNIZER_MIN_WIDTH,
            "rec_max_width": self.rec_max_width,
            "rec_width_overflow_policy": PPOCR_COREML_RECOGNIZER_OVERFLOW_POLICY,
        }


def validate_ppocr_coreml_profile(
    *,
    size: str | None,
    precision: str = "fp32",
    det_limit_side_len: int = 960,
    rec_batch_max: int = PPOCR_COREML_RECOGNIZER_DEFAULT_BATCH_MAX,
    rec_max_width: int,
) -> PPOCRCoreMLProfile:
    """Resolve a Core ML profile and reject any lossy or unbounded variant.

    ``rec_max_width`` is intentionally required.  Native recognition bucketing
    has no finite aspect-ratio ceiling, while ML Program ``RangeDim`` inputs
    must have a finite upper bound.  Requiring the export caller to choose and
    persist that bound prevents a hidden crop truncation policy.
    """

    if size not in _PPOCR_COREML_SIZES:
        raise NotImplementedError(
            f"LibrePPOCR Core ML export supports sizes 't' and 'l'; got size={size!r}."
        )
    if precision != "fp32":
        raise NotImplementedError(
            "LibrePPOCR Core ML export is FP32-only. CTC probability margins "
            f"have not been qualified for precision={precision!r}."
        )
    limit = _strict_int(
        det_limit_side_len,
        name="det_limit_side_len",
        minimum=PPOCR_COREML_DETECTOR_MIN_SIDE,
    )
    batch_max = _strict_int(
        rec_batch_max,
        name="rec_batch_max",
        minimum=1,
    )
    width_max = _strict_int(
        rec_max_width,
        name="rec_max_width",
        minimum=PPOCR_COREML_RECOGNIZER_MIN_WIDTH,
    )
    return PPOCRCoreMLProfile(
        size=size,
        precision=precision,
        det_limit_side_len=limit,
        det_tensor_upper=ppocr_detector_tensor_upper_bound(limit),
        rec_batch_max=batch_max,
        rec_max_width=width_max,
    )


def validate_ppocr_pipeline_config(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the checkpoint's complete OCR pipeline block."""

    if not isinstance(pipeline, Mapping):
        raise ValueError("LibrePPOCR pipeline metadata must be a mapping.")
    if not all(isinstance(key, str) for key in pipeline):
        raise ValueError("LibrePPOCR pipeline metadata keys must be strings.")
    keys = frozenset(pipeline)
    missing = sorted(_PPOCR_PIPELINE_KEYS - keys)
    extra = sorted(keys - _PPOCR_PIPELINE_KEYS)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unknown={extra}")
        raise ValueError(
            "LibrePPOCR pipeline metadata must contain exactly the v1 fields "
            f"({', '.join(details)})."
        )

    rec_shape = pipeline["rec_image_shape"]
    if not isinstance(rec_shape, (list, tuple)) or len(rec_shape) != 3:
        raise ValueError(
            f"pipeline['rec_image_shape'] must be [3, 48, 320], got {rec_shape!r}."
        )
    normalized_shape = [
        _strict_int(value, name=f"rec_image_shape[{index}]", minimum=1)
        for index, value in enumerate(rec_shape)
    ]
    expected_shape = [
        PPOCR_COREML_RECOGNIZER_CHANNELS,
        PPOCR_COREML_RECOGNIZER_HEIGHT,
        PPOCR_COREML_RECOGNIZER_MIN_WIDTH,
    ]
    if normalized_shape != expected_shape:
        raise ValueError(
            f"pipeline['rec_image_shape'] must be [3, 48, 320], got {rec_shape!r}."
        )

    return {
        "det_limit_side_len": _strict_int(
            pipeline["det_limit_side_len"],
            name="pipeline['det_limit_side_len']",
            minimum=PPOCR_COREML_DETECTOR_MIN_SIDE,
        ),
        "det_db_thresh": _strict_number(
            pipeline["det_db_thresh"],
            name="pipeline['det_db_thresh']",
            minimum=0.0,
            maximum=1.0,
        ),
        "det_db_box_thresh": _strict_number(
            pipeline["det_db_box_thresh"],
            name="pipeline['det_db_box_thresh']",
            minimum=0.0,
            maximum=1.0,
        ),
        "det_db_unclip_ratio": _strict_number(
            pipeline["det_db_unclip_ratio"],
            name="pipeline['det_db_unclip_ratio']",
            minimum=0.0,
            exclusive_minimum=True,
        ),
        "rec_image_shape": expected_shape,
    }


def validate_ppocr_charset(
    charset: Sequence[str],
    *,
    rec_num_classes: int | None = None,
) -> list[str]:
    """Validate the self-contained CTC alphabet without coercing entries."""

    if isinstance(charset, (str, bytes)) or not isinstance(charset, (list, tuple)):
        raise ValueError("LibrePPOCR charset metadata must be a list or tuple.")
    normalized = list(charset)
    if len(normalized) < 2:
        raise ValueError("LibrePPOCR charset must contain a blank and characters.")
    for index, entry in enumerate(normalized):
        if not isinstance(entry, str):
            raise ValueError(
                f"LibrePPOCR charset entry {index} must be a string, got "
                f"{type(entry).__name__}."
            )
    if normalized[0] != "blank":
        raise ValueError("LibrePPOCR charset index 0 must be the CTC 'blank'.")
    if normalized[-1] != " ":
        raise ValueError(
            "LibrePPOCR charset must end with the configured space character."
        )
    if rec_num_classes is not None:
        class_count = _strict_int(
            rec_num_classes,
            name="rec_num_classes",
            minimum=2,
        )
        if len(normalized) != class_count:
            raise ValueError(
                "LibrePPOCR recognizer output classes and charset length differ: "
                f"{class_count} != {len(normalized)}."
            )
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def ppocr_charset_sha256(charset: Sequence[str]) -> str:
    """Hash the exact ordered charset using canonical UTF-8 JSON."""

    return _canonical_json_sha256(validate_ppocr_charset(charset))


def _fixed_dimension(axis: str, value: int) -> dict[str, Any]:
    return {"axis": axis, "kind": "fixed", "value": value}


def _range_dimension(
    axis: str,
    *,
    lower_bound: int,
    upper_bound: int,
    default: int,
    multiple_of: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "axis": axis,
        "kind": "range",
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "default": default,
    }
    if multiple_of is not None:
        result["multiple_of"] = multiple_of
    return result


def ppocr_detector_coreml_input_contract(
    profile: PPOCRCoreMLProfile,
) -> dict[str, Any]:
    """Describe the detector's already-preprocessed TensorType boundary."""

    if not isinstance(profile, PPOCRCoreMLProfile):
        raise TypeError("profile must be a validated PPOCRCoreMLProfile.")
    return {
        "name": PPOCR_COREML_DETECTOR_INPUT,
        "kind": "tensor",
        "dtype": "float32",
        "layout": "NCHW",
        "color": "bgr",
        "range": "standardized",
        "mean": list(PPOCR_COREML_DETECTOR_MEAN),
        "std": list(PPOCR_COREML_DETECTOR_STD),
        "geometry": "ppocr_det_resize_max_round32",
        "resize_backend": "opencv",
        "resize_rounding": "python_round_half_even",
        "shape_mode": "range",
        "shape": [
            _fixed_dimension("N", PPOCR_COREML_DETECTOR_BATCH),
            _fixed_dimension("C", PPOCR_COREML_DETECTOR_CHANNELS),
            _range_dimension(
                "H",
                lower_bound=PPOCR_COREML_DETECTOR_MIN_SIDE,
                upper_bound=profile.det_tensor_upper,
                default=profile.det_tensor_upper,
                multiple_of=PPOCR_COREML_DETECTOR_STRIDE,
            ),
            _range_dimension(
                "W",
                lower_bound=PPOCR_COREML_DETECTOR_MIN_SIDE,
                upper_bound=profile.det_tensor_upper,
                default=profile.det_tensor_upper,
                multiple_of=PPOCR_COREML_DETECTOR_STRIDE,
            ),
        ],
    }


def ppocr_recognizer_coreml_input_contract(
    profile: PPOCRCoreMLProfile,
) -> dict[str, Any]:
    """Describe the recognizer's normalized, right-padded TensorType boundary."""

    if not isinstance(profile, PPOCRCoreMLProfile):
        raise TypeError("profile must be a validated PPOCRCoreMLProfile.")
    return {
        "name": PPOCR_COREML_RECOGNIZER_INPUT,
        "kind": "tensor",
        "dtype": "float32",
        "layout": "NCHW",
        "color": "bgr",
        "range": "minus_1_1",
        "geometry": "aspect_resize_height48_pad_right",
        "resize_backend": "opencv",
        "crop_width_rounding": "ceil",
        "bucket_width_rounding": "floor",
        "pad_value": 0.0,
        "overflow_policy": PPOCR_COREML_RECOGNIZER_OVERFLOW_POLICY,
        "shape_mode": "range",
        "shape": [
            _range_dimension(
                "N",
                lower_bound=1,
                upper_bound=profile.rec_batch_max,
                default=1,
            ),
            _fixed_dimension("C", PPOCR_COREML_RECOGNIZER_CHANNELS),
            _fixed_dimension("H", PPOCR_COREML_RECOGNIZER_HEIGHT),
            _range_dimension(
                "W",
                lower_bound=PPOCR_COREML_RECOGNIZER_MIN_WIDTH,
                upper_bound=profile.rec_max_width,
                default=PPOCR_COREML_RECOGNIZER_MIN_WIDTH,
            ),
        ],
    }


def ppocr_coreml_function_contracts(
    profile: PPOCRCoreMLProfile,
    *,
    rec_num_classes: int,
) -> dict[str, dict[str, Any]]:
    """Return the complete two-function package interface descriptor."""

    class_count = _strict_int(
        rec_num_classes,
        name="rec_num_classes",
        minimum=2,
    )
    return {
        PPOCR_COREML_DETECTOR_FUNCTION: {
            "function_name": PPOCR_COREML_DETECTOR_FUNCTION,
            "input": ppocr_detector_coreml_input_contract(profile),
            "outputs": [
                {
                    "name": PPOCR_COREML_DETECTOR_OUTPUT,
                    "role": "text_probability_map",
                    "dtype": "float32",
                    "encoding": "sigmoid_probabilities",
                    "rank": 4,
                    "shape_relation": {
                        "batch": "input.N",
                        "channels": 1,
                        "height": "input.H",
                        "width": "input.W",
                    },
                }
            ],
        },
        PPOCR_COREML_RECOGNIZER_FUNCTION: {
            "function_name": PPOCR_COREML_RECOGNIZER_FUNCTION,
            "input": ppocr_recognizer_coreml_input_contract(profile),
            "outputs": [
                {
                    "name": PPOCR_COREML_RECOGNIZER_OUTPUT,
                    "role": "ctc_probabilities",
                    "dtype": "float32",
                    "encoding": "softmax_probabilities",
                    "rank": 3,
                    "shape_relation": {
                        "batch": "input.N",
                        "timesteps": {
                            "input_axis": "W",
                            "add": 3,
                            "divisor": 8,
                            "rounding": "floor",
                        },
                        "classes": class_count,
                    },
                }
            ],
        },
    }


class PPOCRCoreMLDetector(nn.Module):
    """Isolate the detector as one multifunction ML Program entry point."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        if not isinstance(detector, nn.Module):
            raise TypeError("LibrePPOCR detector must be a torch.nn.Module.")
        self.detector = detector

    def forward(self, detector_input: torch.Tensor) -> torch.Tensor:
        return self.detector(detector_input)


class PPOCRCoreMLRecognizer(nn.Module):
    """Isolate the recognizer as one multifunction ML Program entry point."""

    def __init__(self, recognizer: nn.Module) -> None:
        super().__init__()
        if not isinstance(recognizer, nn.Module):
            raise TypeError("LibrePPOCR recognizer must be a torch.nn.Module.")
        self.recognizer = recognizer

    def forward(self, recognizer_input: torch.Tensor) -> torch.Tensor:
        return self.recognizer(recognizer_input)


@dataclass(frozen=True)
class PPOCRCoreMLModelSignature:
    """Tier and CTC width derived from the actual composite graph."""

    size: str
    rec_num_classes: int


def inspect_ppocr_coreml_model(composite: nn.Module) -> PPOCRCoreMLModelSignature:
    """Derive the tier and class count from graph parameters, not metadata."""

    if not isinstance(composite, nn.Module):
        raise TypeError("LibrePPOCR composite must be a torch.nn.Module.")
    detector = getattr(composite, "det", None)
    recognizer = getattr(composite, "rec", None)
    if not isinstance(detector, nn.Module) or not isinstance(recognizer, nn.Module):
        raise ValueError(
            "LibrePPOCR Core ML export requires a composite with '.det' and "
            "'.rec' torch modules."
        )

    state = composite.state_dict()

    def _tier(prefix: str) -> str:
        mobile = f"{prefix}.backbone.conv1.conv.weight" in state
        server = f"{prefix}.backbone.stem.stem1.conv.weight" in state
        if mobile == server:
            raise ValueError(
                f"Cannot derive LibrePPOCR {prefix} tier from its graph parameters."
            )
        return "t" if mobile else "l"

    detector_size = _tier("det")
    recognizer_size = _tier("rec")
    if detector_size != recognizer_size:
        raise ValueError(
            "LibrePPOCR detector and recognizer tiers differ: "
            f"{detector_size!r} != {recognizer_size!r}."
        )

    weight = state.get("rec.head.ctc_head.fc.weight")
    if not torch.is_tensor(weight) or weight.ndim != 2:
        raise ValueError(
            "Cannot derive LibrePPOCR CTC class count from "
            "'rec.head.ctc_head.fc.weight'."
        )
    return PPOCRCoreMLModelSignature(
        size=detector_size,
        rec_num_classes=int(weight.shape[0]),
    )


def validate_ppocr_coreml_model(
    composite: nn.Module,
    *,
    profile: PPOCRCoreMLProfile,
    rec_num_classes: int,
) -> PPOCRCoreMLModelSignature:
    """Cross-check caller metadata against the actual graph signature."""

    if not isinstance(profile, PPOCRCoreMLProfile):
        raise TypeError("profile must be a validated PPOCRCoreMLProfile.")
    declared_classes = _strict_int(
        rec_num_classes,
        name="rec_num_classes",
        minimum=2,
    )
    signature = inspect_ppocr_coreml_model(composite)
    if signature.size != profile.size:
        raise ValueError(
            "LibrePPOCR graph tier conflicts with the Core ML profile: "
            f"{signature.size!r} != {profile.size!r}."
        )
    if signature.rec_num_classes != declared_classes:
        raise ValueError(
            "LibrePPOCR graph CTC width conflicts with rec_num_classes: "
            f"{signature.rec_num_classes} != {declared_classes}."
        )
    return signature


def wrap_ppocr_coreml_components(
    composite: nn.Module,
    *,
    profile: PPOCRCoreMLProfile,
    rec_num_classes: int,
) -> dict[str, nn.Module]:
    """Return separately traceable detector and recognizer graph adapters."""

    validate_ppocr_coreml_model(
        composite,
        profile=profile,
        rec_num_classes=rec_num_classes,
    )
    detector = getattr(composite, "det", None)
    recognizer = getattr(composite, "rec", None)
    assert isinstance(detector, nn.Module)
    assert isinstance(recognizer, nn.Module)
    return {
        PPOCR_COREML_DETECTOR_FUNCTION: PPOCRCoreMLDetector(detector).eval(),
        PPOCR_COREML_RECOGNIZER_FUNCTION: PPOCRCoreMLRecognizer(recognizer).eval(),
    }


def validate_ppocr_detector_coreml_shape(
    height: int,
    width: int,
    *,
    profile: PPOCRCoreMLProfile,
) -> tuple[int, int]:
    """Reject shapes admitted by RangeDim but invalid for stride-32 fusion."""

    if not isinstance(profile, PPOCRCoreMLProfile):
        raise TypeError("profile must be a validated PPOCRCoreMLProfile.")
    resolved = (
        _strict_int(height, name="detector height"),
        _strict_int(width, name="detector width"),
    )
    for name, value in zip(("height", "width"), resolved):
        if (
            value < PPOCR_COREML_DETECTOR_MIN_SIDE
            or value > profile.det_tensor_upper
            or value % PPOCR_COREML_DETECTOR_STRIDE
        ):
            raise ValueError(
                f"LibrePPOCR detector {name}={value} is outside the bounded "
                "stride-32 profile. Core ML RangeDim cannot enforce the "
                "multiple-of-32 invariant, so the host must reject this input "
                "before prediction."
            )
    return resolved


def validate_ppocr_recognizer_coreml_shape(
    batch: int,
    width: int,
    *,
    profile: PPOCRCoreMLProfile,
) -> tuple[int, int]:
    """Reject recognition batches outside the finite package profile."""

    if not isinstance(profile, PPOCRCoreMLProfile):
        raise TypeError("profile must be a validated PPOCRCoreMLProfile.")
    resolved_batch = _strict_int(batch, name="recognizer batch")
    resolved_width = _strict_int(width, name="recognizer width")
    if not 1 <= resolved_batch <= profile.rec_batch_max:
        raise ValueError(
            f"LibrePPOCR recognizer batch={resolved_batch} is outside [1, "
            f"{profile.rec_batch_max}]."
        )
    if not (
        PPOCR_COREML_RECOGNIZER_MIN_WIDTH <= resolved_width <= profile.rec_max_width
    ):
        raise ValueError(
            f"LibrePPOCR recognizer width={resolved_width} is outside "
            f"[{PPOCR_COREML_RECOGNIZER_MIN_WIDTH}, {profile.rec_max_width}]. "
            "The declared overflow policy is 'error'; the host must not clamp "
            "or rescale the crop silently."
        )
    return resolved_batch, resolved_width


def validate_ppocr_recognizer_coreml_crop(
    crop_height: int,
    crop_width: int,
    *,
    profile: PPOCRCoreMLProfile,
) -> int:
    """Preflight a crop before native bucketing allocates its padded tensor."""

    required_width = ppocr_recognizer_required_width(crop_height, crop_width)
    validate_ppocr_recognizer_coreml_shape(1, required_width, profile=profile)
    return required_width


def validate_ppocr_detector_coreml_io(
    detector_input: torch.Tensor,
    probability_map: torch.Tensor,
    *,
    profile: PPOCRCoreMLProfile,
) -> None:
    """Validate one detector graph probe against the package ABI."""

    if not isinstance(profile, PPOCRCoreMLProfile):
        raise TypeError("profile must be a validated PPOCRCoreMLProfile.")
    if not torch.is_tensor(detector_input) or detector_input.dtype != torch.float32:
        raise ValueError("LibrePPOCR detector input must be a float32 tensor.")
    if detector_input.ndim != 4:
        raise ValueError("LibrePPOCR detector input must have rank 4 (NCHW).")
    batch, channels, height, width = tuple(int(v) for v in detector_input.shape)
    if batch != PPOCR_COREML_DETECTOR_BATCH or channels != 3:
        raise ValueError(
            "LibrePPOCR detector input shape must start with [1, 3], "
            f"got {tuple(detector_input.shape)}."
        )
    validate_ppocr_detector_coreml_shape(height, width, profile=profile)
    if not bool(torch.isfinite(detector_input).all()):
        raise ValueError("LibrePPOCR detector input contains non-finite values.")

    if not torch.is_tensor(probability_map) or probability_map.dtype != torch.float32:
        raise ValueError("LibrePPOCR detector output must be a float32 tensor.")
    expected_shape = (batch, 1, height, width)
    if tuple(probability_map.shape) != expected_shape:
        raise ValueError(
            "LibrePPOCR detector output shape must be "
            f"{expected_shape}, got {tuple(probability_map.shape)}."
        )
    if not bool(torch.isfinite(probability_map).all()):
        raise ValueError("LibrePPOCR detector output contains non-finite values.")
    if bool((probability_map < 0.0).any()) or bool((probability_map > 1.0).any()):
        raise ValueError("LibrePPOCR detector output must contain probabilities.")


def validate_ppocr_recognizer_coreml_io(
    recognizer_input: torch.Tensor,
    ctc_probabilities: torch.Tensor,
    *,
    profile: PPOCRCoreMLProfile,
    rec_num_classes: int,
) -> None:
    """Validate one recognizer graph probe against the package ABI."""

    if not isinstance(profile, PPOCRCoreMLProfile):
        raise TypeError("profile must be a validated PPOCRCoreMLProfile.")
    if not torch.is_tensor(recognizer_input) or recognizer_input.dtype != torch.float32:
        raise ValueError("LibrePPOCR recognizer input must be a float32 tensor.")
    if recognizer_input.ndim != 4:
        raise ValueError("LibrePPOCR recognizer input must have rank 4 (NCHW).")
    batch, channels, height, width = tuple(int(v) for v in recognizer_input.shape)
    if channels != 3 or height != PPOCR_COREML_RECOGNIZER_HEIGHT:
        raise ValueError(
            "LibrePPOCR recognizer input must have shape [N, 3, 48, W], "
            f"got {tuple(recognizer_input.shape)}."
        )
    validate_ppocr_recognizer_coreml_shape(batch, width, profile=profile)
    if not bool(torch.isfinite(recognizer_input).all()):
        raise ValueError("LibrePPOCR recognizer input contains non-finite values.")
    if bool((recognizer_input < -1.0).any()) or bool((recognizer_input > 1.0).any()):
        raise ValueError("LibrePPOCR recognizer input must be normalized to [-1, 1].")

    class_count = _strict_int(
        rec_num_classes,
        name="rec_num_classes",
        minimum=2,
    )
    if (
        not torch.is_tensor(ctc_probabilities)
        or ctc_probabilities.dtype != torch.float32
    ):
        raise ValueError("LibrePPOCR recognizer output must be a float32 tensor.")
    expected_shape = (batch, ppocr_recognizer_timesteps(width), class_count)
    if tuple(ctc_probabilities.shape) != expected_shape:
        raise ValueError(
            "LibrePPOCR recognizer output shape must be "
            f"{expected_shape}, got {tuple(ctc_probabilities.shape)}."
        )
    if not bool(torch.isfinite(ctc_probabilities).all()):
        raise ValueError("LibrePPOCR recognizer output contains non-finite values.")
    if bool((ctc_probabilities < 0.0).any()) or bool((ctc_probabilities > 1.0).any()):
        raise ValueError("LibrePPOCR recognizer output must contain probabilities.")
    sums = ctc_probabilities.sum(dim=-1)
    if not torch.allclose(
        sums,
        torch.ones_like(sums),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(
            "LibrePPOCR recognizer probabilities must sum to one per timestep."
        )


def ppocr_coreml_metadata(
    *,
    profile: PPOCRCoreMLProfile,
    charset: Sequence[str],
    pipeline: Mapping[str, Any],
    rec_num_classes: int,
) -> dict[str, Any]:
    """Build the complete contract-owned metadata for a multifunction package."""

    if not isinstance(profile, PPOCRCoreMLProfile):
        raise TypeError("profile must be a validated PPOCRCoreMLProfile.")
    class_count = _strict_int(
        rec_num_classes,
        name="rec_num_classes",
        minimum=2,
    )
    normalized_charset = validate_ppocr_charset(
        charset,
        rec_num_classes=class_count,
    )
    normalized_pipeline = validate_ppocr_pipeline_config(pipeline)
    if normalized_pipeline["det_limit_side_len"] != profile.det_limit_side_len:
        raise ValueError(
            "LibrePPOCR profile det_limit_side_len conflicts with pipeline "
            f"metadata: {profile.det_limit_side_len} != "
            f"{normalized_pipeline['det_limit_side_len']}."
        )
    functions = ppocr_coreml_function_contracts(
        profile,
        rec_num_classes=class_count,
    )
    return {
        "artifact_scope": PPOCR_COREML_ARTIFACT_SCOPE,
        "component_contract": PPOCR_COREML_COMPONENT_CONTRACT,
        "ppocr_coreml_schema_version": PPOCR_COREML_PIPELINE_SCHEMA_VERSION,
        "coreml_multifunction": True,
        "coreml_minimum_deployment_targets": list(
            PPOCR_COREML_MINIMUM_DEPLOYMENT_TARGETS
        ),
        "coreml_default_function": PPOCR_COREML_DEFAULT_FUNCTION,
        "coreml_function_names": list(PPOCR_COREML_FUNCTION_NAMES),
        "coreml_functions": functions,
        "coreml_functions_sha256": _canonical_json_sha256(functions),
        "ppocr_packaged_components": list(PPOCR_COREML_FUNCTION_NAMES),
        "ppocr_host_operations": list(PPOCR_COREML_HOST_OPERATIONS),
        "ppocr_coreml_profile": profile.as_dict(),
        "charset": normalized_charset,
        "charset_sha256": _canonical_json_sha256(normalized_charset),
        "rec_num_classes": class_count,
        "pipeline": normalized_pipeline,
        "precision": profile.precision,
        "dynamic": True,
    }


def _metadata_json_value(metadata: Mapping[str, Any], key: str) -> Any:
    if key not in metadata:
        raise ValueError(f"Strict LibrePPOCR Core ML metadata is missing {key!r}.")
    value = metadata[key]
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Strict LibrePPOCR Core ML metadata {key!r} must contain JSON."
            ) from exc
    return value


def _metadata_int_value(metadata: Mapping[str, Any], key: str) -> int:
    if key not in metadata:
        raise ValueError(f"Strict LibrePPOCR Core ML metadata is missing {key!r}.")
    value = metadata[key]
    if isinstance(value, str):
        if not value.isdecimal():
            raise ValueError(
                f"Strict LibrePPOCR Core ML metadata {key!r} must be an integer."
            )
        value = int(value)
    return _strict_int(value, name=f"metadata[{key!r}]")


def _metadata_bool_value(metadata: Mapping[str, Any], key: str) -> bool:
    if key not in metadata:
        raise ValueError(f"Strict LibrePPOCR Core ML metadata is missing {key!r}.")
    value = metadata[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in {"True", "true"}:
        return True
    if isinstance(value, str) and value in {"False", "false"}:
        return False
    raise ValueError(f"Strict LibrePPOCR Core ML metadata {key!r} must be a boolean.")


def _metadata_string_value(metadata: Mapping[str, Any], key: str) -> str:
    if key not in metadata:
        raise ValueError(f"Strict LibrePPOCR Core ML metadata is missing {key!r}.")
    value = metadata[key]
    if not isinstance(value, str):
        raise ValueError(
            f"Strict LibrePPOCR Core ML metadata {key!r} must be a string."
        )
    return value


def _assert_exact_json(expected: Any, actual: Any, *, path: str) -> None:
    """Compare a JSON-like value without Python's bool/int equivalence."""

    if type(actual) is not type(expected):
        raise ValueError(
            f"Strict LibrePPOCR Core ML metadata {path} has type "
            f"{type(actual).__name__}; expected {type(expected).__name__}."
        )
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(
                f"Strict LibrePPOCR Core ML metadata {path} has different fields."
            )
        for key, expected_value in expected.items():
            _assert_exact_json(
                expected_value,
                actual[key],
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(
                f"Strict LibrePPOCR Core ML metadata {path} has the wrong length."
            )
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            _assert_exact_json(
                expected_value,
                actual_value,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(expected, float):
        equal = math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
    else:
        equal = actual == expected
    if not equal:
        raise ValueError(
            f"Strict LibrePPOCR Core ML metadata {path}={actual!r}; "
            f"expected {expected!r}."
        )


def validate_ppocr_coreml_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate native or Core-ML-stringified metadata and return canonical data.

    Unknown top-level keys are permitted so the shared exporter can add the
    normal LibreYOLO provenance fields.  Every contract-owned field and every
    nested interface field is required and checked exactly.
    """

    if not isinstance(metadata, Mapping):
        raise ValueError("LibrePPOCR Core ML metadata must be a mapping.")

    profile_value = _metadata_json_value(metadata, "ppocr_coreml_profile")
    if not isinstance(profile_value, dict):
        raise ValueError(
            "Strict LibrePPOCR Core ML metadata 'ppocr_coreml_profile' "
            "must be an object."
        )
    try:
        profile = validate_ppocr_coreml_profile(
            size=profile_value["size"],
            precision=profile_value["precision"],
            det_limit_side_len=profile_value["det_limit_side_len"],
            rec_batch_max=profile_value["rec_batch_max"],
            rec_max_width=profile_value["rec_max_width"],
        )
    except KeyError as exc:
        raise ValueError(
            f"Strict LibrePPOCR Core ML profile is missing {exc.args[0]!r}."
        ) from exc

    charset_value = _metadata_json_value(metadata, "charset")
    pipeline_value = _metadata_json_value(metadata, "pipeline")
    class_count = _metadata_int_value(metadata, "rec_num_classes")
    expected = ppocr_coreml_metadata(
        profile=profile,
        charset=charset_value,
        pipeline=pipeline_value,
        rec_num_classes=class_count,
    )

    string_fields = (
        "artifact_scope",
        "component_contract",
        "coreml_default_function",
        "coreml_functions_sha256",
        "charset_sha256",
        "precision",
    )
    for key in string_fields:
        actual = _metadata_string_value(metadata, key)
        expected_value = expected[key]
        if key.endswith("_sha256"):
            equal = hmac.compare_digest(actual, expected_value)
        else:
            equal = actual == expected_value
        if not equal:
            raise ValueError(
                f"Strict LibrePPOCR Core ML metadata {key!r}={actual!r}; "
                f"expected {expected_value!r}."
            )

    schema_version = _metadata_int_value(
        metadata,
        "ppocr_coreml_schema_version",
    )
    if schema_version != PPOCR_COREML_PIPELINE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported LibrePPOCR Core ML metadata schema version {schema_version}."
        )
    for key in ("coreml_multifunction", "dynamic"):
        actual_bool = _metadata_bool_value(metadata, key)
        if actual_bool is not expected[key]:
            raise ValueError(
                f"Strict LibrePPOCR Core ML metadata {key!r} must be {expected[key]!r}."
            )

    json_fields = (
        "coreml_minimum_deployment_targets",
        "coreml_function_names",
        "coreml_functions",
        "ppocr_packaged_components",
        "ppocr_host_operations",
        "ppocr_coreml_profile",
        "charset",
        "pipeline",
    )
    for key in json_fields:
        actual_value = _metadata_json_value(metadata, key)
        _assert_exact_json(expected[key], actual_value, path=key)

    if "model_family" in metadata and str(metadata["model_family"]) != "ppocr":
        raise ValueError("LibrePPOCR Core ML metadata model_family must be 'ppocr'.")
    if "task" in metadata and str(metadata["task"]) != "ocr":
        raise ValueError("LibrePPOCR Core ML metadata task must be 'ocr'.")
    if "size" in metadata and str(metadata["size"]) != profile.size:
        raise ValueError("LibrePPOCR Core ML metadata size conflicts with its profile.")
    return expected


__all__ = [
    "PPOCR_COREML_ARTIFACT_SCOPE",
    "PPOCR_COREML_COMPONENT_CONTRACT",
    "PPOCR_COREML_DEFAULT_FUNCTION",
    "PPOCR_COREML_DETECTOR_FUNCTION",
    "PPOCR_COREML_DETECTOR_INPUT",
    "PPOCR_COREML_DETECTOR_MAX_SIDE_LIMIT",
    "PPOCR_COREML_DETECTOR_MEAN",
    "PPOCR_COREML_DETECTOR_MIN_SIDE",
    "PPOCR_COREML_DETECTOR_OUTPUT",
    "PPOCR_COREML_DETECTOR_STD",
    "PPOCR_COREML_DETECTOR_STRIDE",
    "PPOCR_COREML_FUNCTION_NAMES",
    "PPOCR_COREML_HOST_OPERATIONS",
    "PPOCR_COREML_MINIMUM_DEPLOYMENT_TARGETS",
    "PPOCR_COREML_PIPELINE_SCHEMA_VERSION",
    "PPOCR_COREML_RECOGNIZER_DEFAULT_BATCH_MAX",
    "PPOCR_COREML_RECOGNIZER_FUNCTION",
    "PPOCR_COREML_RECOGNIZER_HEIGHT",
    "PPOCR_COREML_RECOGNIZER_INPUT",
    "PPOCR_COREML_RECOGNIZER_MIN_WIDTH",
    "PPOCR_COREML_RECOGNIZER_OUTPUT",
    "PPOCR_COREML_RECOGNIZER_OVERFLOW_POLICY",
    "PPOCRCoreMLDetector",
    "PPOCRCoreMLModelSignature",
    "PPOCRCoreMLProfile",
    "PPOCRCoreMLRecognizer",
    "inspect_ppocr_coreml_model",
    "ppocr_charset_sha256",
    "ppocr_coreml_function_contracts",
    "ppocr_coreml_metadata",
    "ppocr_detector_coreml_input_contract",
    "ppocr_detector_tensor_upper_bound",
    "ppocr_recognizer_coreml_input_contract",
    "ppocr_recognizer_required_width",
    "ppocr_recognizer_timesteps",
    "validate_ppocr_charset",
    "validate_ppocr_coreml_metadata",
    "validate_ppocr_coreml_model",
    "validate_ppocr_coreml_profile",
    "validate_ppocr_detector_coreml_io",
    "validate_ppocr_detector_coreml_shape",
    "validate_ppocr_pipeline_config",
    "validate_ppocr_recognizer_coreml_crop",
    "validate_ppocr_recognizer_coreml_io",
    "validate_ppocr_recognizer_coreml_shape",
    "wrap_ppocr_coreml_components",
]
