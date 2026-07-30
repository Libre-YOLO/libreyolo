"""CoreML inference backend for LibreYOLO. macOS only.

Loads .mlpackage models produced by libreyolo.export.coreml and runs inference
via coremltools.models.MLModel. Mirrors OnnxBackend's public surface so the
rest of LibreYOLO (Results, drawing, etc.) sees the same interface.
"""

from __future__ import annotations

import ast
import gc
import json
import logging
import sys
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Optional

import numpy as np
import torch
from PIL import Image

from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from ..utils.general import COCO_CLASSES
from ..utils.image_loader import ImageLoader
from ..utils.predict_args import normalize_predict_kwargs
from ..utils.results import Boxes, Masks, Results
from ..utils.serialization import warn_on_metadata_schema_version
from .base import (
    BaseBackend,
    ImageSize,
    _imgsz_hw,
    _read_metadata_imgsz,
    _read_pose_metadata,
)

logger = logging.getLogger(__name__)

_COREML_PRODUCER_KEY = "libreyolo_producer"
_COREML_PRODUCER = "libreyolo"
_COREML_IO_SCHEMA_KEY = "coreml_io_schema_version"
_COREML_IO_SCHEMA_VERSION = "2"
_COREML_IO_SCHEMA_VERSIONS = frozenset({"1", "2"})
_COREML_IO_KEY = "coreml_io"
_COREML_ARRAY_DTYPES = {
    "float16": 65552,
    "float32": 65568,
    "float64": 65600,
    "double": 65600,
    "int32": 131104,
}
_COREML_RGB_COLORSPACE = 20
_SAM_COREML_FAMILIES = frozenset(
    {"edgetam", "mobilesam", "sam", "sam2", "sam3"}
)

# These are the only families shipped by the pre-contract CoreML exporter.
# A package without the producer/schema marker is never accepted merely
# because it happens to have a compatible input shape.
_LEGACY_COREML_FAMILIES = {"yolo9", "yolox", "rtdetr", "rfdetr"}
_LEGACY_REQUIRED_METADATA = {
    "schema_version",
    "libreyolo_version",
    "model_family",
    "task",
    "supported_tasks",
    "default_task",
    "names",
    "imgsz",
}

_INPUT_KINDS = {"image", "tensor"}
_INPUT_LAYOUTS = {"nchw", "nhwc"}
_INPUT_COLORS = {"rgb", "bgr"}
_INPUT_RANGES = {
    "uint8",
    "0_1",
    "minus_1_1",
    "0_255",
    "imagenet",
    "standardized",
}
_GEOMETRIES = {
    "stretch",
    "letterbox_top_left",
    "letterbox_center",
    "center_crop",
    "pad_bottom_right",
    "native",
    "owlv2_pad_square",
    "eomt_split",
    "eomt_pad_top_left",
}
_INTERPOLATIONS = {"nearest", "bilinear", "bicubic"}
_RESIZE_BACKENDS = {"pillow", "opencv", "torchvision"}
_SHAPE_MODES = {"fixed", "enumerated", "range"}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class _CoreMLOutput:
    name: str
    role: str
    rank: int | None = None
    dtype: str | None = None
    encoding: str | None = None
    shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _CoreMLValidationInput:
    color: str
    value_range: str
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class _CoreMLInput:
    name: str
    kind: str
    layout: str
    color: str
    value_range: str
    mean: tuple[float, float, float] | None
    std: tuple[float, float, float] | None
    geometry: str
    interpolation: str
    resize_backend: str
    resize_long_side: int | None
    resize_rounding: str
    pad_value: int
    crop_pct: float
    shape_mode: str
    validation: _CoreMLValidationInput


@dataclass(frozen=True)
class _CoreMLIO:
    input: _CoreMLInput
    outputs: tuple[_CoreMLOutput, ...]
    parser: str | None = None


@dataclass(frozen=True)
class _GeometryResult:
    image: Image.Image
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float

    @property
    def ratio(self) -> float:
        if abs(self.scale_x - self.scale_y) < 1e-12:
            return self.scale_x
        return 1.0


class _PPOCRCoreMLFunction:
    """Torch-callable facade over one named Core ML PPOCR function."""

    def __init__(
        self,
        runtime: Any,
        *,
        function_name: str,
        input_name: str,
        output_name: str,
        profile: Any,
        rec_num_classes: int,
    ) -> None:
        self.runtime = runtime
        self.function_name = function_name
        self.input_name = input_name
        self.output_name = output_name
        self.profile = profile
        self.rec_num_classes = int(rec_num_classes)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        from ..export.coreml_ppocr import (
            PPOCR_COREML_DETECTOR_FUNCTION,
            validate_ppocr_detector_coreml_io,
            validate_ppocr_detector_coreml_shape,
            validate_ppocr_recognizer_coreml_io,
            validate_ppocr_recognizer_coreml_shape,
        )

        if not torch.is_tensor(tensor):
            raise TypeError(
                f"LibrePPOCR {self.function_name} input must be a torch.Tensor."
            )
        if tensor.dtype != torch.float32 or tensor.ndim != 4:
            raise ValueError(
                f"LibrePPOCR {self.function_name} input must be an FP32 "
                f"rank-four tensor, got dtype={tensor.dtype}, "
                f"shape={tuple(tensor.shape)}."
            )
        if self.function_name == PPOCR_COREML_DETECTOR_FUNCTION:
            if tuple(tensor.shape[:2]) != (1, 3):
                raise ValueError(
                    "LibrePPOCR detector input must start with [1, 3]."
                )
            validate_ppocr_detector_coreml_shape(
                int(tensor.shape[-2]),
                int(tensor.shape[-1]),
                profile=self.profile,
            )
        else:
            if int(tensor.shape[1]) != 3 or int(tensor.shape[2]) != 48:
                raise ValueError(
                    "LibrePPOCR recognizer input must have shape "
                    "[N, 3, 48, W]."
                )
            validate_ppocr_recognizer_coreml_shape(
                int(tensor.shape[0]),
                int(tensor.shape[-1]),
                profile=self.profile,
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"LibrePPOCR {self.function_name} input contains non-finite "
                "values."
            )
        if (
            self.function_name != PPOCR_COREML_DETECTOR_FUNCTION
            and (
                bool((tensor < -1.0).any())
                or bool((tensor > 1.0).any())
            )
        ):
            raise ValueError(
                "LibrePPOCR recognizer input must be normalized to [-1, 1]."
            )
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        output = self.runtime.predict({self.input_name: array})
        if not isinstance(output, Mapping):
            raise RuntimeError(
                f"Core ML {self.function_name!r} returned a non-mapping output."
            )
        if set(output) != {self.output_name}:
            raise RuntimeError(
                f"Core ML {self.function_name!r} output names changed: "
                f"expected {[self.output_name]!r}, got {sorted(output)!r}."
            )
        value = np.asarray(output[self.output_name])
        if value.dtype != np.float32:
            raise RuntimeError(
                f"Core ML {self.function_name!r} returned dtype "
                f"{value.dtype.name!r}; the contract requires 'float32'."
            )
        result = torch.from_numpy(np.ascontiguousarray(value).copy())
        if self.function_name == PPOCR_COREML_DETECTOR_FUNCTION:
            validate_ppocr_detector_coreml_io(
                tensor.detach().cpu(),
                result,
                profile=self.profile,
            )
        else:
            validate_ppocr_recognizer_coreml_io(
                tensor.detach().cpu(),
                result,
                profile=self.profile,
                rec_num_classes=self.rec_num_classes,
            )
        return result


class _PPOCRCoreMLComposite:
    """Minimal ``.det``/``.rec`` surface expected by OCRInferenceRunner."""

    def __init__(self, detector: Any, recognizer: Any) -> None:
        self.det = detector
        self.rec = recognizer


class _PPOCRCoreMLRunnerProxy:
    """Native OCR runner facade backed by two named Core ML functions."""

    def __init__(
        self,
        *,
        detector: Any,
        recognizer: Any,
        profile: Any,
        charset: list[str],
        pipeline: dict[str, Any],
        names: dict[int, str],
    ) -> None:
        self.model = _PPOCRCoreMLComposite(detector, recognizer)
        self.device = torch.device("cpu")
        self.profile = profile
        self.charset = list(charset)
        self.pipeline_config = dict(pipeline)
        self.names = dict(names)

    def _validate_recognition_crops(
        self,
        crops: list[np.ndarray],
        rec_batch: int,
    ) -> None:
        """Fail before ``rec_batches`` can allocate an out-of-profile bucket."""
        from ..export.coreml_ppocr import (
            validate_ppocr_recognizer_coreml_crop,
            validate_ppocr_recognizer_coreml_shape,
        )

        validate_ppocr_recognizer_coreml_shape(
            max(1, int(rec_batch)),
            320,
            profile=self.profile,
        )
        for index, crop in enumerate(crops):
            if not isinstance(crop, np.ndarray) or crop.ndim != 3:
                raise ValueError(
                    f"LibrePPOCR crop {index} must be an HWC array."
                )
            try:
                validate_ppocr_recognizer_coreml_crop(
                    int(crop.shape[0]),
                    int(crop.shape[1]),
                    profile=self.profile,
                )
            except ValueError as exc:
                raise ValueError(
                    f"LibrePPOCR crop {index} exceeds the exported Core ML "
                    f"recognizer profile: {exc}"
                ) from exc


def _metadata_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", ""}:
        return False
    raise ValueError(f"CoreML metadata {key!r} must be true or false, got {value!r}.")


def _metadata_json(value: Any, *, key: str) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        raise ValueError(f"CoreML metadata {key!r} must contain JSON.")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"CoreML metadata {key!r} is not valid JSON.") from exc


def _strict_metadata_int(value: Any, *, key: str, minimum: int = 1) -> int:
    """Parse an integer metadata token without accepting bools/floats."""
    if isinstance(value, bool):
        raise ValueError(f"CoreML metadata {key!r} must be an integer.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        token = value.strip()
        if not token or not token.lstrip("-").isdigit():
            raise ValueError(f"CoreML metadata {key!r} must be an integer.")
        parsed = int(token)
    else:
        raise ValueError(f"CoreML metadata {key!r} must be an integer.")
    if parsed < minimum:
        comparator = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"CoreML metadata {key!r} must be {comparator}.")
    return parsed


def _triple(value: Any, *, key: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"CoreML IO field {key!r} must be a three-item list.")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CoreML IO field {key!r} must contain numeric values."
        ) from exc
    if not np.isfinite(result).all():
        raise ValueError(f"CoreML IO field {key!r} must contain finite values.")
    return result


def _resampling(name: str) -> Image.Resampling:
    return {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }[name]


def _resize_image(
    image: Image.Image,
    size: tuple[int, int],
    contract: _CoreMLInput,
) -> Image.Image:
    """Resize with the implementation declared by the artifact contract."""
    if contract.resize_backend == "pillow":
        return image.resize(size, _resampling(contract.interpolation))
    if contract.resize_backend == "torchvision":
        raise RuntimeError(
            "The torchvision antialiased float resize must use the dedicated "
            "TensorType preprocessing path."
        )

    import cv2

    interpolation = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }[contract.interpolation]
    resized = cv2.resize(
        np.asarray(image, dtype=np.uint8),
        size,
        interpolation=interpolation,
    )
    return Image.fromarray(resized)


def _resolve_letterbox_geometry(
    *,
    orig_h: int,
    orig_w: int,
    input_h: int,
    input_w: int,
    contract: _CoreMLInput,
) -> tuple[float, int, int, int, int]:
    ratio = min(input_h / orig_h, input_w / orig_w)
    if contract.resize_long_side is not None:
        ratio = min(
            ratio,
            contract.resize_long_side / orig_h,
            contract.resize_long_side / orig_w,
        )

    if contract.resize_rounding == "round":
        new_w = int(round(orig_w * ratio))
        new_h = int(round(orig_h * ratio))
    else:
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)

    if new_w <= 0 or new_h <= 0:
        if contract.resize_long_side is not None:
            raise ValueError(
                "CoreML capped letterbox geometry produced a zero-sized "
                f"dimension for source {orig_w}x{orig_h}, "
                f"resize_long_side={contract.resize_long_side}."
            )
        new_w = max(1, new_w)
        new_h = max(1, new_h)

    if contract.geometry == "letterbox_center":
        offset_x = (input_w - new_w) // 2
        offset_y = (input_h - new_h) // 2
    else:
        offset_x = offset_y = 0
    return ratio, new_w, new_h, offset_x, offset_y


def _apply_geometry(
    image: Image.Image,
    *,
    input_h: int,
    input_w: int,
    contract: _CoreMLInput,
) -> _GeometryResult:
    """Apply the host-side geometry declared by the CoreML artifact."""
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    geometry = contract.geometry

    if geometry == "stretch":
        resized = _resize_image(image, (input_w, input_h), contract)
        return _GeometryResult(
            resized,
            input_w / orig_w,
            input_h / orig_h,
            0.0,
            0.0,
        )

    if geometry in {"letterbox_top_left", "letterbox_center"}:
        ratio, new_w, new_h, offset_x, offset_y = _resolve_letterbox_geometry(
            orig_h=orig_h,
            orig_w=orig_w,
            input_h=input_h,
            input_w=input_w,
            contract=contract,
        )
        resized = _resize_image(image, (new_w, new_h), contract)
        canvas = Image.new(
            "RGB",
            (input_w, input_h),
            (contract.pad_value,) * 3,
        )
        canvas.paste(resized, (offset_x, offset_y))
        return _GeometryResult(
            canvas,
            ratio,
            ratio,
            float(offset_x),
            float(offset_y),
        )

    if geometry == "center_crop":
        if input_h != input_w:
            raise NotImplementedError(
                "CoreML center-crop preprocessing currently requires a square "
                "exported input."
            )
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        resize_short = max(1, int(input_h / contract.crop_pct))
        mode = {
            "nearest": InterpolationMode.NEAREST,
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
        }[contract.interpolation]
        # These are the exact two geometry operations used by
        # build_classify_transforms. In particular, the shorter-side target is
        # floor(imgsz / crop_pct), not round(...).
        resized = transforms.Resize(resize_short, interpolation=mode)(image)
        cropped = transforms.CenterCrop(input_h)(resized)
        new_w, new_h = resized.size
        left = max(int(round((new_w - input_w) / 2.0)), 0)
        top = max(int(round((new_h - input_h) / 2.0)), 0)
        return _GeometryResult(
            cropped,
            new_w / orig_w,
            new_h / orig_h,
            -float(left),
            -float(top),
        )

    if geometry == "pad_bottom_right":
        if orig_h > input_h or orig_w > input_w:
            raise ValueError(
                "CoreML fixed-canvas input is larger than the exported canvas: "
                f"got {orig_w}x{orig_h}, maximum is {input_w}x{input_h}."
            )
        arr = np.asarray(image, dtype=np.uint8)
        pad_h = input_h - orig_h
        pad_w = input_w - orig_w
        if pad_h or pad_w:
            mode = (
                "reflect"
                if orig_h > 1 and orig_w > 1 and pad_h < orig_h and pad_w < orig_w
                else "edge"
            )
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
        return _GeometryResult(Image.fromarray(arr), 1.0, 1.0, 0.0, 0.0)

    if geometry == "native":
        if contract.shape_mode == "fixed" and (orig_h, orig_w) != (
            input_h,
            input_w,
        ):
            raise ValueError(
                "CoreML fixed native geometry requires the source image to "
                f"match the exported canvas {input_w}x{input_h}; got "
                f"{orig_w}x{orig_h}. Re-export at this size, crop, or tile "
                "the image."
            )
        return _GeometryResult(image, 1.0, 1.0, 0.0, 0.0)

    if geometry in {"eomt_split", "eomt_pad_top_left"}:
        raise RuntimeError(
            f"CoreML geometry {geometry!r} requires EoMT's dedicated "
            "torchvision preprocessing path."
        )

    raise AssertionError(f"Unhandled CoreML geometry {geometry!r}.")


class _CoreMLValPreprocessor:
    """Detection-style validator adapter that emits canonical RGB 0..255."""

    def __init__(self, img_size: tuple[int, int], contract: _CoreMLInput):
        self.img_size = tuple(int(value) for value in img_size)
        self.contract = contract
        self.max_labels = 120

    @property
    def normalize(self) -> bool:
        return False

    @property
    def custom_normalization(self) -> bool:
        return True

    @property
    def wants_unresized_image(self) -> bool:
        return True

    @property
    def uses_letterbox(self) -> bool:
        return self.contract.geometry in {
            "letterbox_top_left",
            "letterbox_center",
        }

    def letterbox_scale(
        self, orig_h: int, orig_w: int, imgsz: int
    ) -> tuple[float, float, float]:
        del imgsz
        input_h, input_w = self.img_size
        ratio, _new_w, _new_h, offset_x, offset_y = (
            _resolve_letterbox_geometry(
                orig_h=orig_h,
                orig_w=orig_w,
                input_h=input_h,
                input_w=input_w,
                contract=self.contract,
            )
        )
        return ratio, float(offset_x), float(offset_y)

    def __call__(
        self,
        img: np.ndarray,
        targets: np.ndarray,
        input_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.contract.geometry in {"center_crop", "pad_bottom_right", "native"}:
            raise NotImplementedError(
                "Detection-style CoreML validation does not support "
                f"geometry={self.contract.geometry!r}."
            )
        target_h, target_w = (int(input_size[0]), int(input_size[1]))
        if self.contract.resize_backend == "torchvision":
            if self.contract.geometry != "stretch":
                raise NotImplementedError(
                    "The CoreML torchvision float resize currently supports "
                    "stretch geometry only."
                )
            import torch.nn.functional as F

            rgb = np.ascontiguousarray(img[:, :, ::-1], dtype=np.float32)
            tensor = (
                torch.from_numpy(rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .div(255.0)
            )
            resized = F.interpolate(
                tensor,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            chw = np.ascontiguousarray(resized[0].numpy() * 255.0)
            orig_h, orig_w = img.shape[:2]
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            if len(targets):
                source = np.asarray(targets, dtype=np.float32)
                n = min(len(source), self.max_labels)
                adjusted = source[:n].copy()
                adjusted[:, [0, 2]] *= scale_x
                adjusted[:, [1, 3]] *= scale_y
                padded_targets[:n] = adjusted
            return chw, padded_targets

        rgb = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
        transformed = _apply_geometry(
            rgb,
            input_h=target_h,
            input_w=target_w,
            contract=self.contract,
        )
        chw = np.asarray(transformed.image, dtype=np.float32).transpose(2, 0, 1)
        chw = np.ascontiguousarray(chw)

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets):
            source = np.asarray(targets, dtype=np.float32)
            n = min(len(source), self.max_labels)
            adjusted = source[:n].copy()
            adjusted[:, [0, 2]] = (
                adjusted[:, [0, 2]] * transformed.scale_x + transformed.offset_x
            )
            adjusted[:, [1, 3]] = (
                adjusted[:, [1, 3]] * transformed.scale_y + transformed.offset_y
            )
            padded_targets[:n] = adjusted
        return chw, padded_targets


def _to_compute_unit(compute_units: str):
    """Same mapping as the exporter — duplicated to avoid pulling export deps in."""
    import coremltools as ct

    key = compute_units.lower()
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    if key not in mapping:
        raise ValueError(
            f"Invalid compute_units {compute_units!r}. "
            f"Must be one of: {sorted(mapping)}"
        )
    return mapping[key]


_RFDETR_POSE_COREML_PASS_PROFILE = "rfdetr_pose_preserve_division_v1"
_RFDETR_POSE_COREML_DISABLED_PASSES = ("common::divide_to_multiply",)


def _spec_user_defined_metadata(spec: Any) -> dict[str, Any]:
    """Read package metadata from a protobuf spec without compiling a proxy."""
    description = getattr(spec, "description", None)
    metadata = getattr(description, "metadata", None)
    values = getattr(metadata, "userDefined", None)
    return dict(values or {})


def _preflight_multifunction_spec(
    spec: Any,
    meta: Mapping[str, Any],
    *,
    path: Path,
) -> str | None:
    """Validate a recognized multifunction package before native compilation."""
    description = getattr(spec, "description", None)
    functions = list(getattr(description, "functions", ()) or ())
    family = str(meta.get("model_family", "")).strip().lower()
    component = str(meta.get("component_contract", "")).strip()
    sam_marker = (
        family in {"edgetam", "mobilesam", "sam", "sam2", "sam3"}
        or component
        in {
            "sam_split_promptable_v1",
            "sam_split_promptable_v2",
            "sam_split_promptable_v3",
        }
    )
    ppocr_marker = (
        family == "ppocr" or component == "ppocr_det_rec_v1"
    )
    if sam_marker:
        from ..export.coreml import _validate_sam_multifunction_spec
        from ..export.coreml_sam import (
            validate_sam_coreml_metadata,
            validate_sam_coreml_profile,
        )

        canonical = validate_sam_coreml_metadata(meta)
        values = canonical["sam_coreml_profile"]
        profile = validate_sam_coreml_profile(
            family=values["family"],
            size=values["size"],
            precision=values["precision"],
            prompt_max_points=values["prompt_max_points"],
        )
        try:
            _validate_sam_multifunction_spec(spec, profile=profile)
        except RuntimeError as exc:
            raise ValueError(
                f"Invalid LibreSAM Core ML multifunction spec: {exc}"
            ) from exc
        return "sam"
    if ppocr_marker:
        from ..export.coreml import _validate_ppocr_multifunction_spec
        from ..export.coreml_ppocr import (
            validate_ppocr_coreml_metadata,
            validate_ppocr_coreml_profile,
        )

        canonical = validate_ppocr_coreml_metadata(meta)
        values = canonical["ppocr_coreml_profile"]
        profile = validate_ppocr_coreml_profile(
            size=values["size"],
            precision=values["precision"],
            det_limit_side_len=values["det_limit_side_len"],
            rec_batch_max=values["rec_batch_max"],
            rec_max_width=values["rec_max_width"],
        )
        try:
            _validate_ppocr_multifunction_spec(spec, profile=profile)
        except RuntimeError as exc:
            raise ValueError(
                f"Invalid LibrePPOCR Core ML multifunction spec: {exc}"
            ) from exc
        return "ppocr"
    if functions:
        generic_marker = (
            "coreml_multifunction" in meta
            and _metadata_bool(
                meta["coreml_multifunction"],
                key="coreml_multifunction",
            )
        )
        marker_text = (
            " despite coreml_multifunction=true"
            if generic_marker
            else ""
        )
        raise ValueError(
            f"CoreML artifact {path} is an unknown multifunction "
            f"package{marker_text}; no recognized strict component "
            "contract was declared."
        )
    return None


def _require_rfdetr_pose_cpu_profile(
    meta: Mapping[str, Any],
    *,
    compute_units: str,
) -> None:
    """Fail closed before Core ML compiles an unvalidated RF pose route."""
    family = str(meta.get("model_family", "")).strip().lower()
    try:
        task = normalize_task(meta.get("task"))
    except ValueError:
        task = str(meta.get("task", "")).strip().lower()
    if (family, task) != ("rfdetr", "pose"):
        return

    required_units = str(
        meta.get("coreml_required_compute_units", "")
    ).strip().lower()
    precision = str(meta.get("precision", "")).strip().lower()
    pass_profile = str(
        meta.get("coreml_conversion_pass_profile", "")
    ).strip()
    raw_disabled = meta.get("coreml_disabled_passes")
    if isinstance(raw_disabled, str):
        try:
            raw_disabled = json.loads(raw_disabled)
        except json.JSONDecodeError:
            raw_disabled = None
    disabled_passes = (
        tuple(raw_disabled)
        if isinstance(raw_disabled, (list, tuple))
        and all(isinstance(value, str) for value in raw_disabled)
        else ()
    )
    if (
        required_units != "cpu_only"
        or precision != "fp32"
        or pass_profile != _RFDETR_POSE_COREML_PASS_PROFILE
        or disabled_passes != _RFDETR_POSE_COREML_DISABLED_PASSES
    ):
        raise ValueError(
            "RF-DETR pose Core ML artifact lacks the validated CPU conversion "
            "profile. Re-export with FP32, compute_units='cpu_only', and "
            f"pass profile {_RFDETR_POSE_COREML_PASS_PROFILE!r}."
        )
    if str(compute_units).strip().lower() != "cpu_only":
        raise NotImplementedError(
            "RF-DETR pose Core ML inference currently requires "
            "compute_units='cpu_only'. GPU/ALL execution does not meet the "
            "fixed M4 conversion-fidelity gate; cpu_and_ne is also excluded "
            "because this FP32 ML Program is placed on CPU."
        )


def _normalize_metadata_supported_tasks(value) -> tuple[str, ...]:
    try:
        return normalize_supported_tasks(value)
    except ValueError:
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                raise
            return normalize_supported_tasks(parsed)
        raise


def _parse_validation_input(value: Any) -> _CoreMLValidationInput:
    if not isinstance(value, dict):
        raise ValueError("CoreML IO field 'validation' must be an object.")
    color = str(value.get("color", "")).strip().lower()
    value_range = str(value.get("range", "")).strip().lower()
    if color not in _INPUT_COLORS:
        raise ValueError(
            f"CoreML IO validation.color must be 'rgb' or 'bgr', got {color!r}."
        )
    if value_range not in _INPUT_RANGES - {"uint8"}:
        raise ValueError(
            "CoreML IO validation.range must be '0_1', 'minus_1_1', "
            f"'0_255', 'imagenet', or 'standardized', got {value_range!r}."
        )
    mean = std = None
    if value_range == "imagenet":
        mean = _IMAGENET_MEAN
        std = _IMAGENET_STD
        if "mean" in value or "std" in value:
            raise ValueError(
                "CoreML IO validation.range='imagenet' uses fixed ImageNet "
                "constants and must not also declare mean/std."
            )
    elif value_range == "standardized":
        mean = _triple(value.get("mean"), key="validation.mean")
        std = _triple(value.get("std"), key="validation.std")
        if any(item <= 0 for item in std):
            raise ValueError("CoreML IO validation.std values must be positive.")
    elif "mean" in value or "std" in value:
        raise ValueError(
            "CoreML IO validation.mean/std are only valid with "
            "validation.range='standardized'."
        )
    return _CoreMLValidationInput(color, value_range, mean, std)


def _parse_io_contract(meta: Mapping[str, Any]) -> _CoreMLIO:
    raw = _metadata_json(meta[_COREML_IO_KEY], key=_COREML_IO_KEY)
    if not isinstance(raw, dict):
        raise ValueError("CoreML IO metadata must be a JSON object.")

    input_raw = raw.get("input")
    if not isinstance(input_raw, dict):
        raise ValueError("CoreML IO field 'input' must be an object.")

    name = str(input_raw.get("name", "")).strip()
    kind = str(input_raw.get("kind", "")).strip().lower()
    layout = str(input_raw.get("layout", "")).strip().lower()
    color = str(input_raw.get("color", "")).strip().lower()
    value_range = str(input_raw.get("range", "")).strip().lower()
    geometry = str(input_raw.get("geometry", "")).strip().lower()
    interpolation = str(input_raw.get("interpolation", "")).strip().lower()
    resize_backend = str(input_raw.get("resize_backend", "")).strip().lower()
    resize_rounding_value = input_raw.get("resize_rounding")
    shape_mode = str(input_raw.get("shape_mode", "fixed")).strip().lower()

    if not name:
        raise ValueError("CoreML IO input.name must be a non-empty string.")
    if kind not in _INPUT_KINDS:
        raise ValueError(
            f"CoreML IO input.kind must be one of {sorted(_INPUT_KINDS)}, got {kind!r}."
        )
    if layout not in _INPUT_LAYOUTS:
        raise ValueError(
            f"CoreML IO input.layout must be 'nchw' or 'nhwc', got {layout!r}."
        )
    if color not in _INPUT_COLORS:
        raise ValueError(
            f"CoreML IO input.color must be 'rgb' or 'bgr', got {color!r}."
        )
    if value_range not in _INPUT_RANGES:
        raise ValueError(
            f"CoreML IO input.range must be one of {sorted(_INPUT_RANGES)}, "
            f"got {value_range!r}."
        )
    if geometry not in _GEOMETRIES:
        raise ValueError(
            f"CoreML IO input.geometry must be one of {sorted(_GEOMETRIES)}, "
            f"got {geometry!r}."
        )
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(
            "CoreML IO input.interpolation must be nearest, bilinear, or bicubic, "
            f"got {interpolation!r}."
        )
    if resize_backend not in _RESIZE_BACKENDS:
        raise ValueError(
            "CoreML IO input.resize_backend must be 'pillow', 'opencv', or "
            "'torchvision', "
            f"got {resize_backend!r}."
        )
    if geometry == "center_crop" and resize_backend != "pillow":
        raise ValueError(
            "CoreML IO center_crop geometry currently requires "
            "input.resize_backend='pillow'."
        )
    if resize_rounding_value in (None, ""):
        resize_rounding = (
            "round" if geometry == "letterbox_center" else "floor"
        )
    else:
        resize_rounding = str(resize_rounding_value).strip().lower()
    if resize_rounding not in {"floor", "round"}:
        raise ValueError(
            "CoreML IO input.resize_rounding must be 'floor' or 'round', "
            f"got {resize_rounding_value!r}."
        )
    resize_long_side_value = input_raw.get("resize_long_side")
    if resize_long_side_value in (None, ""):
        resize_long_side = None
    else:
        try:
            resize_long_side = int(resize_long_side_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CoreML IO input.resize_long_side must be a positive integer."
            ) from exc
        if resize_long_side <= 0:
            raise ValueError(
                "CoreML IO input.resize_long_side must be a positive integer."
            )
        if geometry not in {"letterbox_top_left", "letterbox_center"}:
            raise ValueError(
                "CoreML IO input.resize_long_side is valid only for "
                "letterbox geometry."
            )
    if shape_mode not in _SHAPE_MODES:
        raise ValueError(
            f"CoreML IO input.shape_mode must be one of {sorted(_SHAPE_MODES)}, "
            f"got {shape_mode!r}."
        )
    if kind == "image" and (
        layout != "nchw" or color != "rgb" or value_range != "uint8"
    ):
        raise ValueError(
            "CoreML ImageType artifacts must declare the host boundary as "
            "layout='nchw', color='rgb', range='uint8'."
        )

    mean = std = None
    if value_range == "imagenet":
        mean = _IMAGENET_MEAN
        std = _IMAGENET_STD
        if "mean" in input_raw or "std" in input_raw:
            raise ValueError(
                "CoreML IO input.range='imagenet' uses fixed ImageNet constants "
                "and must not also declare mean/std."
            )
    elif value_range == "standardized":
        mean = _triple(input_raw.get("mean"), key="input.mean")
        std = _triple(input_raw.get("std"), key="input.std")
        if any(item <= 0 for item in std):
            raise ValueError("CoreML IO input.std values must be positive.")
    elif "mean" in input_raw or "std" in input_raw:
        raise ValueError(
            "CoreML IO input.mean/std are only valid with input.range='standardized'."
        )

    try:
        pad_value = int(input_raw.get("pad_value", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("CoreML IO input.pad_value must be an integer.") from exc
    if not 0 <= pad_value <= 255:
        raise ValueError("CoreML IO input.pad_value must be in 0..255.")

    try:
        crop_pct = float(input_raw.get("crop_pct", 0.875))
    except (TypeError, ValueError) as exc:
        raise ValueError("CoreML IO input.crop_pct must be numeric.") from exc
    if not np.isfinite(crop_pct) or not 0 < crop_pct <= 1:
        raise ValueError("CoreML IO input.crop_pct must be in (0, 1].")

    validation = _parse_validation_input(raw.get("validation"))
    input_contract = _CoreMLInput(
        name=name,
        kind=kind,
        layout=layout,
        color=color,
        value_range=value_range,
        mean=mean,
        std=std,
        geometry=geometry,
        interpolation=interpolation,
        resize_backend=resize_backend,
        resize_long_side=resize_long_side,
        resize_rounding=resize_rounding,
        pad_value=pad_value,
        crop_pct=crop_pct,
        shape_mode=shape_mode,
        validation=validation,
    )

    outputs_raw = raw.get("outputs")
    if not isinstance(outputs_raw, list) or not outputs_raw:
        raise ValueError("CoreML IO outputs must be a non-empty list.")
    outputs = []
    names = set()
    for index, item in enumerate(outputs_raw):
        if not isinstance(item, dict):
            raise ValueError(f"CoreML IO outputs[{index}] must be an object.")
        output_name = str(item.get("name", "")).strip()
        role = str(item.get("role", "")).strip()
        if not output_name or not role:
            raise ValueError(
                f"CoreML IO outputs[{index}] requires non-empty name and role."
            )
        if output_name in names:
            raise ValueError(f"Duplicate CoreML output name {output_name!r}.")
        rank = item.get("rank")
        if rank is not None:
            try:
                rank = int(rank)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"CoreML IO outputs[{index}].rank must be an integer."
                ) from exc
            if rank <= 0:
                raise ValueError(f"CoreML IO outputs[{index}].rank must be positive.")
        dtype = item.get("dtype")
        if dtype is not None:
            dtype = str(dtype).strip().lower()
            if not dtype:
                raise ValueError(f"CoreML IO outputs[{index}].dtype must be non-empty.")
        encoding = item.get("encoding")
        if encoding is not None:
            encoding = str(encoding).strip()
            if not encoding:
                raise ValueError(
                    f"CoreML IO outputs[{index}].encoding must be non-empty."
                )
        shape_raw = item.get("shape")
        shape = None
        if shape_raw is not None:
            if not isinstance(shape_raw, (list, tuple)) or not shape_raw:
                raise ValueError(
                    f"CoreML IO outputs[{index}].shape must be a non-empty list."
                )
            try:
                shape = tuple(int(dimension) for dimension in shape_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"CoreML IO outputs[{index}].shape must contain integers."
                ) from exc
            if any(dimension <= 0 for dimension in shape):
                raise ValueError(
                    f"CoreML IO outputs[{index}].shape values must be positive."
                )
            if rank is not None and len(shape) != rank:
                raise ValueError(
                    f"CoreML IO outputs[{index}].shape rank disagrees with rank."
                )
        names.add(output_name)
        outputs.append(
            _CoreMLOutput(
                output_name,
                role,
                rank,
                dtype,
                encoding,
                shape,
            )
        )

    parser = raw.get("parser")
    if parser is not None:
        parser = str(parser).strip()
        if not parser:
            raise ValueError("CoreML IO parser must be a non-empty string.")
    return _CoreMLIO(input_contract, tuple(outputs), parser)


def _feature_names(features: Any) -> list[str]:
    if features is None:
        return []
    try:
        return [str(feature.name) for feature in features]
    except (AttributeError, TypeError):
        return []


def _feature_kind(feature: Any) -> str | None:
    feature_type = getattr(feature, "type", None)
    which_oneof = getattr(feature_type, "WhichOneof", None)
    if not callable(which_oneof):
        return None
    kind = which_oneof("Type")
    return {"imageType": "image", "multiArrayType": "tensor"}.get(kind, kind)


def _validate_strict_pose_contract(
    meta: Mapping[str, Any],
    *,
    family: str,
    nc: int,
    io_contract: _CoreMLIO,
) -> dict[str, Any]:
    """Validate pose metadata that changes runtime parser interpretation."""
    required = {"num_keypoints", "keypoint_dim", "pose_encoding"}
    missing = sorted(key for key in required if meta.get(key) in (None, ""))
    if missing:
        raise ValueError(
            "Strict pose CoreML artifacts require complete pose metadata; "
            f"missing {missing}."
        )
    num_keypoints = _strict_metadata_int(
        meta["num_keypoints"],
        key="num_keypoints",
    )
    keypoint_dim = _strict_metadata_int(meta["keypoint_dim"], key="keypoint_dim")
    pose: dict[str, Any] = {
        "num_keypoints": num_keypoints,
        "keypoint_dim": keypoint_dim,
    }
    raw_schema = meta.get("num_keypoints_per_class")
    schema = None
    if raw_schema not in (None, ""):
        if isinstance(raw_schema, str):
            try:
                raw_schema = json.loads(raw_schema)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Strict CoreML num_keypoints_per_class is not valid JSON."
                ) from exc
        if not isinstance(raw_schema, (list, tuple)) or not raw_schema:
            raise ValueError(
                "Strict CoreML num_keypoints_per_class must be a non-empty list."
            )
        try:
            schema = [
                _strict_metadata_int(
                    count,
                    key=f"num_keypoints_per_class[{index}]",
                    minimum=0,
                )
                for index, count in enumerate(raw_schema)
            ]
        except ValueError as exc:
            raise ValueError(
                "Strict CoreML num_keypoints_per_class must contain "
                "nonnegative integers."
            ) from exc
        if not any(count > 0 for count in schema):
            raise ValueError(
                "Strict CoreML num_keypoints_per_class must be nonnegative "
                "and contain at least one active class."
            )
        pose["num_keypoints_per_class"] = schema

    expected_encoding = (
        "rfdetr_grouppose_padded_v1"
        if family == "rfdetr" and schema
        else "rfdetr_flat_keypoints_v1"
        if family == "rfdetr"
        else "yolonas_split_xy_conf_v1"
        if family == "yolonas"
        else "ec_normalized_xy_v1"
        if family == "ec"
        else "keypoints_v1"
    )
    actual_encoding = str(meta["pose_encoding"]).strip()
    if actual_encoding != expected_encoding:
        raise ValueError(
            "Strict CoreML pose_encoding does not match the family/schema: "
            f"expected {expected_encoding!r}, got {actual_encoding!r}."
        )
    if family != "rfdetr" and schema is not None:
        raise ValueError(
            "Strict CoreML num_keypoints_per_class is currently valid only "
            "for RF-DETR GroupPose artifacts."
        )

    output_by_name = {output.name: output for output in io_contract.outputs}
    if family == "rfdetr":
        logits_shape = output_by_name["pred_logits"].shape
        keypoints_shape = output_by_name["pred_keypoints"].shape
        if schema:
            if keypoint_dim != 8:
                raise ValueError("RF-DETR GroupPose requires keypoint_dim=8.")
            if num_keypoints != max(schema):
                raise ValueError(
                    "RF-DETR GroupPose num_keypoints must equal max(schema)."
                )
            if nc != sum(count > 0 for count in schema):
                raise ValueError(
                    "RF-DETR GroupPose public nc must equal its active classes."
                )
            if logits_shape is not None and logits_shape[-1] != len(schema):
                raise ValueError(
                    "RF-DETR GroupPose logits width must equal schema length."
                )
            if keypoints_shape is not None and (
                len(keypoints_shape) != 4
                or keypoints_shape[-2:]
                != (len(schema) * max(schema), keypoint_dim)
            ):
                raise ValueError(
                    "RF-DETR GroupPose keypoint shape does not match its "
                    "padded class schema."
                )
        else:
            if keypoint_dim not in {2, 3}:
                raise ValueError(
                    "Classic RF-DETR pose keypoint_dim must be 2 or 3."
                )
            if logits_shape is not None and logits_shape[-1] != nc:
                raise ValueError(
                    "Classic RF-DETR pose logits width must equal public nc."
                )
            if keypoints_shape is not None:
                valid = (
                    len(keypoints_shape) == 3
                    and keypoints_shape[-1] == num_keypoints * keypoint_dim
                ) or (
                    len(keypoints_shape) == 4
                    and keypoints_shape[-2:] == (num_keypoints, keypoint_dim)
                )
                if not valid:
                    raise ValueError(
                        "Classic RF-DETR keypoint shape disagrees with pose metadata."
                    )
    elif family == "ec":
        if keypoint_dim != 2:
            raise ValueError("EC pose CoreML artifacts require keypoint_dim=2.")
        logits_shape = output_by_name["pred_logits"].shape
        keypoints_shape = output_by_name["pred_keypoints"].shape
        if logits_shape is not None and logits_shape[-1] != 2:
            raise ValueError("EC pose logits width must be two.")
        if keypoints_shape is not None:
            valid = (
                len(keypoints_shape) == 3
                and keypoints_shape[-1] == 2 * num_keypoints
            ) or (
                len(keypoints_shape) == 4
                and keypoints_shape[-2:] == (num_keypoints, 2)
            )
            if not valid:
                raise ValueError(
                    "EC keypoint shape disagrees with pose metadata."
                )
    elif family == "yolonas":
        if keypoint_dim != 3:
            raise ValueError("YOLO-NAS pose CoreML artifacts require keypoint_dim=3.")
        xy_shape = output_by_name["keypoints_xy"].shape
        confidence_shape = output_by_name["keypoints_conf"].shape
        if xy_shape is not None and xy_shape[-2:] != (num_keypoints, 2):
            raise ValueError("YOLO-NAS keypoints_xy shape disagrees with metadata.")
        if (
            xy_shape is not None
            and confidence_shape is not None
            and xy_shape[:-1] != confidence_shape
        ):
            raise ValueError("YOLO-NAS keypoint output axes do not match.")

    return pose


def _validate_picosam3_component_metadata(meta: Mapping[str, Any]) -> None:
    """Pin host-orchestration fields for the fixed PicoSAM3 ROI component."""
    from ..export.coreml_picosam3 import (
        PICOSAM3_COREML_COMPONENT_CONTRACT,
        PICOSAM3_COREML_INPUT_SIZE,
        PICOSAM3_COREML_ROI_PADDING,
    )

    expected_strings = {
        "artifact_scope": "roi_component",
        "component_contract": PICOSAM3_COREML_COMPONENT_CONTRACT,
        "prompt_type": "boxes",
    }
    for key, expected in expected_strings.items():
        actual = str(meta.get(key, "")).strip().lower()
        if actual != expected:
            raise ValueError(
                f"Strict PicoSAM3 CoreML metadata {key!r} must be "
                f"{expected!r}; got {meta.get(key)!r}."
            )

    roi_input_size = _strict_metadata_int(
        meta.get("roi_input_size"),
        key="roi_input_size",
    )
    roi_batch = _strict_metadata_int(meta.get("roi_batch"), key="roi_batch")
    if roi_input_size != PICOSAM3_COREML_INPUT_SIZE or roi_batch != 1:
        raise ValueError(
            "Strict PicoSAM3 CoreML metadata requires roi_input_size=96 and "
            f"roi_batch=1; got {roi_input_size} and {roi_batch}."
        )
    try:
        roi_padding = float(meta.get("roi_padding"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Strict PicoSAM3 CoreML metadata 'roi_padding' must be numeric."
        ) from exc
    if not np.isfinite(roi_padding) or not np.isclose(
        roi_padding,
        PICOSAM3_COREML_ROI_PADDING,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Strict PicoSAM3 CoreML metadata requires roi_padding=0.1; "
            f"got {meta.get('roi_padding')!r}."
        )


def _validate_rtmdet_ins_metadata(meta: Mapping[str, Any]) -> None:
    """Pin every host-side constant used to decode RTMDet-Ins raw outputs."""
    from ..export.coreml_rtmdet_ins import (
        RTMDET_INS_COREML_CONTRACT,
        RTMDET_INS_COREML_DYCONV_CHANNELS,
        RTMDET_INS_COREML_DYNAMIC_BIAS_NUMS,
        RTMDET_INS_COREML_DYNAMIC_WEIGHT_NUMS,
        RTMDET_INS_COREML_MASK_STRIDE,
        RTMDET_INS_COREML_MASK_THRESHOLD,
        RTMDET_INS_COREML_MAX_MASKS,
        RTMDET_INS_COREML_NMS_PRE,
        RTMDET_INS_COREML_NUM_GEN_PARAMS,
        RTMDET_INS_COREML_NUM_PROTOTYPES,
        RTMDET_INS_COREML_PRIOR_OFFSET,
        RTMDET_INS_COREML_STRIDES,
    )

    actual_contract = str(meta.get("rtmdet_ins_contract", "")).strip().lower()
    if actual_contract != RTMDET_INS_COREML_CONTRACT:
        raise ValueError(
            "Strict RTMDet-Ins CoreML metadata 'rtmdet_ins_contract' must be "
            f"{RTMDET_INS_COREML_CONTRACT!r}; got "
            f"{meta.get('rtmdet_ins_contract')!r}."
        )

    raw_strides = _metadata_json(
        meta.get("rtmdet_ins_strides"),
        key="rtmdet_ins_strides",
    )
    if not isinstance(raw_strides, list):
        raise ValueError(
            "Strict RTMDet-Ins CoreML metadata 'rtmdet_ins_strides' must be "
            "a JSON list."
        )
    strides = tuple(
        _strict_metadata_int(value, key=f"rtmdet_ins_strides[{index}]")
        for index, value in enumerate(raw_strides)
    )
    if strides != RTMDET_INS_COREML_STRIDES:
        raise ValueError(
            "Strict RTMDet-Ins CoreML strides must be "
            f"{RTMDET_INS_COREML_STRIDES}; got {strides}."
        )

    for key, expected in {
        "rtmdet_ins_dynamic_weight_nums": RTMDET_INS_COREML_DYNAMIC_WEIGHT_NUMS,
        "rtmdet_ins_dynamic_bias_nums": RTMDET_INS_COREML_DYNAMIC_BIAS_NUMS,
    }.items():
        raw_values = _metadata_json(meta.get(key), key=key)
        if not isinstance(raw_values, list):
            raise ValueError(
                f"Strict RTMDet-Ins CoreML metadata {key!r} must be a JSON list."
            )
        actual = tuple(
            _strict_metadata_int(value, key=f"{key}[{index}]")
            for index, value in enumerate(raw_values)
        )
        if actual != expected:
            raise ValueError(
                f"Strict RTMDet-Ins CoreML metadata {key!r} must be "
                f"{expected}; got {actual}."
            )

    expected_ints = {
        "rtmdet_ins_num_gen_params": RTMDET_INS_COREML_NUM_GEN_PARAMS,
        "rtmdet_ins_num_prototypes": RTMDET_INS_COREML_NUM_PROTOTYPES,
        "rtmdet_ins_mask_stride": RTMDET_INS_COREML_MASK_STRIDE,
        "rtmdet_ins_nms_pre": RTMDET_INS_COREML_NMS_PRE,
        "rtmdet_ins_max_masks": RTMDET_INS_COREML_MAX_MASKS,
        "rtmdet_ins_prior_offset": RTMDET_INS_COREML_PRIOR_OFFSET,
        "rtmdet_ins_dyconv_channels": RTMDET_INS_COREML_DYCONV_CHANNELS,
    }
    for key, expected in expected_ints.items():
        actual = _strict_metadata_int(
            meta.get(key),
            key=key,
            minimum=0 if expected == 0 else 1,
        )
        if actual != expected:
            raise ValueError(
                f"Strict RTMDet-Ins CoreML metadata {key!r} must be "
                f"{expected}; got {actual}."
            )
    try:
        mask_threshold = float(meta.get("rtmdet_ins_mask_threshold"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Strict RTMDet-Ins CoreML metadata 'rtmdet_ins_mask_threshold' "
            "must be numeric."
        ) from exc
    if not np.isfinite(mask_threshold) or not np.isclose(
        mask_threshold,
        RTMDET_INS_COREML_MASK_THRESHOLD,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Strict RTMDet-Ins CoreML mask threshold must be "
            f"{RTMDET_INS_COREML_MASK_THRESHOLD}; got {mask_threshold}."
        )


def _validate_eomt_metadata(
    meta: Mapping[str, Any],
    *,
    task: str,
    nc: int,
    imgsz: ImageSize,
    io_contract: _CoreMLIO,
) -> dict[str, Any]:
    """Validate EoMT's host-orchestrated query/mask component contract."""
    from ..export.coreml_eomt import (
        EOMT_COREML_ALIGN_CORNERS,
        EOMT_COREML_ANTIALIAS,
        EOMT_COREML_ARTIFACT_SCOPE,
        EOMT_COREML_ATTENTION_MASK,
        EOMT_COREML_CONTRACT,
        EOMT_COREML_MASK_STRIDE,
        EOMT_COREML_NUM_UPSCALE_BLOCKS,
        EOMT_COREML_PATCH_SIZE,
        EOMT_COREML_POSTPROCESS,
        EOMT_COREML_PREPROCESS,
        expected_eomt_coreml_shapes,
    )

    expected_strings = {
        "artifact_scope": EOMT_COREML_ARTIFACT_SCOPE[task],
        "eomt_contract": EOMT_COREML_CONTRACT,
        "eomt_preprocess": EOMT_COREML_PREPROCESS[task],
        "eomt_postprocess": EOMT_COREML_POSTPROCESS[task],
        "eomt_attention_mask": EOMT_COREML_ATTENTION_MASK,
    }
    for key, expected in expected_strings.items():
        actual = str(meta.get(key, "")).strip().lower()
        if actual != expected:
            raise ValueError(
                f"Strict EoMT CoreML metadata {key!r} must be "
                f"{expected!r}; got {meta.get(key)!r}."
            )

    image_size = _strict_metadata_int(
        meta.get("eomt_image_size"),
        key="eomt_image_size",
    )
    num_queries = _strict_metadata_int(
        meta.get("eomt_num_queries"),
        key="eomt_num_queries",
    )
    expected_hw = _imgsz_hw(imgsz)
    if expected_hw != (image_size, image_size):
        raise ValueError(
            "Strict EoMT CoreML eomt_image_size must match the fixed square "
            f"input canvas; metadata={image_size}, input={expected_hw}."
        )
    if "num_queries" in meta and _strict_metadata_int(
        meta["num_queries"],
        key="num_queries",
    ) != num_queries:
        raise ValueError(
            "Strict EoMT CoreML num_queries aliases disagree."
        )

    expected_ints = {
        "eomt_patch_size": EOMT_COREML_PATCH_SIZE,
        "eomt_mask_stride": EOMT_COREML_MASK_STRIDE,
        "eomt_num_upscale_blocks": EOMT_COREML_NUM_UPSCALE_BLOCKS,
    }
    for key, expected in expected_ints.items():
        actual = _strict_metadata_int(meta.get(key), key=key)
        if actual != expected:
            raise ValueError(
                f"Strict EoMT CoreML metadata {key!r} must be "
                f"{expected}; got {actual}."
            )
    if _metadata_bool(
        meta.get("eomt_mask_align_corners"),
        key="eomt_mask_align_corners",
    ) is not EOMT_COREML_ALIGN_CORNERS:
        raise ValueError(
            "Strict EoMT CoreML metadata requires "
            "eomt_mask_align_corners=false."
        )
    if _metadata_bool(
        meta.get("eomt_antialias"),
        key="eomt_antialias",
    ) is not EOMT_COREML_ANTIALIAS:
        raise ValueError(
            "Strict EoMT CoreML metadata requires eomt_antialias=true."
        )

    expected_shapes = expected_eomt_coreml_shapes(
        nc=nc,
        num_queries=num_queries,
        canvas_hw=expected_hw,
    )
    actual_shapes = {
        output.name: output.shape for output in io_contract.outputs
    }
    if actual_shapes != expected_shapes:
        raise ValueError(
            "Strict EoMT CoreML output shapes disagree with the compact "
            f"query ABI: expected {expected_shapes}, got {actual_shapes}."
        )

    thing_class_ids: list[int] | None = None
    if task == "panoptic":
        raw_ids = _metadata_json(
            meta.get("thing_class_ids"),
            key="thing_class_ids",
        )
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(
                "Strict EoMT panoptic CoreML metadata thing_class_ids must "
                "be a non-empty JSON list."
            )
        thing_class_ids = [
            _strict_metadata_int(
                value,
                key=f"thing_class_ids[{index}]",
                minimum=0,
            )
            for index, value in enumerate(raw_ids)
        ]
        if (
            thing_class_ids != sorted(set(thing_class_ids))
            or any(value >= nc for value in thing_class_ids)
        ):
            raise ValueError(
                "Strict EoMT panoptic thing_class_ids must be sorted, unique, "
                "and within the class range."
            )
        for key, expected in {
            "eomt_panoptic_score_threshold": 0.8,
            "eomt_panoptic_mask_threshold": 0.5,
            "eomt_panoptic_overlap_threshold": 0.8,
        }.items():
            try:
                actual = float(meta.get(key))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Strict EoMT CoreML metadata {key!r} must be numeric."
                ) from exc
            if not np.isfinite(actual) or not np.isclose(
                actual,
                expected,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Strict EoMT CoreML metadata {key!r} must be "
                    f"{expected}; got {meta.get(key)!r}."
                )

    return {
        "num_queries": num_queries,
        "image_size": image_size,
        "thing_class_ids": thing_class_ids,
    }


def _legacy_io_contract(
    *,
    family: str,
    task: str,
    spec: Any,
    output_names: list[str],
) -> _CoreMLIO:
    inputs = getattr(getattr(spec, "description", None), "input", None)
    input_names = _feature_names(inputs)
    if len(input_names) != 1:
        raise ValueError(
            "Legacy LibreYOLO CoreML artifacts must have exactly one input; "
            f"found {input_names or 'none'}."
        )

    geometry = "letterbox_top_left" if family in {"yolo9", "yolox"} else "stretch"
    validation_range = "0_255"
    outputs: tuple[_CoreMLOutput, ...]
    if set(output_names) == {"confidence", "coordinates"}:
        outputs = (
            _CoreMLOutput("confidence", "class_scores"),
            _CoreMLOutput("coordinates", "boxes_cxcywh"),
        )
    elif family in {"rtdetr", "rfdetr"}:
        if len(output_names) != 2:
            raise ValueError(
                f"Legacy {family} CoreML artifacts require two outputs, "
                f"found {output_names}."
            )
        outputs = (
            _CoreMLOutput(output_names[0], "class_logits"),
            _CoreMLOutput(output_names[1], "boxes_cxcywh"),
        )
    else:
        if len(output_names) != 1:
            raise ValueError(
                f"Legacy {family} CoreML artifacts require one raw output, "
                f"found {output_names}."
            )
        outputs = (_CoreMLOutput(output_names[0], "prediction"),)

    if task != "detect":
        raise ValueError(
            "Pre-contract LibreYOLO CoreML artifacts are supported only for "
            f"detection, got task={task!r}. Re-export with current LibreYOLO."
        )
    return _CoreMLIO(
        _CoreMLInput(
            name=input_names[0],
            kind="image",
            layout="nchw",
            color="rgb",
            value_range="uint8",
            mean=None,
            std=None,
            geometry=geometry,
            interpolation="bilinear",
            resize_backend="pillow",
            resize_long_side=None,
            resize_rounding=(
                "round" if geometry == "letterbox_center" else "floor"
            ),
            pad_value=114,
            crop_pct=0.875,
            shape_mode="fixed",
            validation=_CoreMLValidationInput("rgb", validation_range),
        ),
        outputs,
        parser=family,
    )


def _artifact_uses_strict_contract(meta: Mapping[str, Any], *, path: Path) -> bool:
    contract_keys = {
        _COREML_PRODUCER_KEY,
        _COREML_IO_SCHEMA_KEY,
        _COREML_IO_KEY,
    }
    present = contract_keys.intersection(meta)
    if present:
        missing = contract_keys - set(meta)
        if missing:
            raise ValueError(
                f"CoreML artifact {path} has an incomplete LibreYOLO contract; "
                f"missing metadata keys {sorted(missing)}."
            )
        if str(meta[_COREML_PRODUCER_KEY]).strip().lower() != _COREML_PRODUCER:
            raise ValueError(
                f"CoreML artifact {path} was not produced by LibreYOLO "
                f"({_COREML_PRODUCER_KEY}={meta[_COREML_PRODUCER_KEY]!r})."
            )
        version = str(meta[_COREML_IO_SCHEMA_KEY]).strip()
        if version not in _COREML_IO_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported CoreML IO schema version {version!r}; "
                "this LibreYOLO build supports "
                f"{sorted(_COREML_IO_SCHEMA_VERSIONS)!r}."
            )
        return True

    missing = _LEGACY_REQUIRED_METADATA - set(meta)
    family = str(meta.get("model_family", "")).strip().lower()
    if missing or family not in _LEGACY_COREML_FAMILIES:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if family not in _LEGACY_COREML_FAMILIES:
            details.append(
                f"model_family must be one of {sorted(_LEGACY_COREML_FAMILIES)}"
            )
        raise ValueError(
            f"CoreML artifact {path} is not a recognized LibreYOLO package "
            f"({'; '.join(details)}). Re-export it with current LibreYOLO."
        )
    logger.warning(
        "Loading pre-contract LibreYOLO CoreML artifact %s. Re-export it to "
        "embed the strict producer and IO schema.",
        path,
    )
    return False


def _validate_common_metadata(
    meta: Mapping[str, Any],
    *,
    strict: bool,
    path: Path,
) -> None:
    required = {
        "model_family",
        "task",
        "supported_tasks",
        "default_task",
        "names",
        "imgsz",
    }
    if strict:
        required |= {
            "artifact_format",
            "schema_version",
            "libreyolo_version",
            "size",
            "nc",
            "dynamic",
        }
    missing = required - set(meta)
    if missing:
        raise ValueError(
            f"CoreML artifact {path} is missing required metadata {sorted(missing)}."
        )
    if strict and str(meta.get("artifact_format", "")).strip().lower() != "coreml":
        raise ValueError(
            "Strict LibreYOLO CoreML artifacts must declare artifact_format='coreml'."
        )

    family = str(meta.get("model_family", "")).strip()
    size = str(meta.get("size", meta.get("model_size", ""))).strip()
    if not family or not size:
        raise ValueError(
            f"CoreML artifact {path} requires non-empty model_family and size."
        )
    if strict and "model_size" in meta:
        model_size = str(meta["model_size"]).strip()
        if model_size != size:
            raise ValueError(
                "Strict CoreML metadata aliases disagree: "
                f"size={size!r}, model_size={model_size!r}."
            )

    raw_names = _metadata_json(meta["names"], key="names")
    if not isinstance(raw_names, dict) or not raw_names:
        raise ValueError("CoreML metadata 'names' must be a non-empty JSON object.")
    try:
        names = {int(key): str(value) for key, value in raw_names.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("CoreML metadata 'names' keys must be integers.") from exc
    expected = list(range(len(names)))
    if sorted(names) != expected:
        raise ValueError(
            "CoreML metadata 'names' keys must be contiguous from zero; "
            f"found {sorted(names)}."
        )
    nc_raw = meta.get("nc", meta.get("nb_classes"))
    if nc_raw is not None:
        try:
            nc = int(nc_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("CoreML metadata 'nc' must be an integer.") from exc
        if nc != len(names):
            raise ValueError(
                f"CoreML metadata declares nc={nc} but has {len(names)} names."
            )
    if strict and "nc" in meta and "nb_classes" in meta:
        try:
            nc = int(meta["nc"])
            nb_classes = int(meta["nb_classes"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CoreML metadata nc/nb_classes must be integers."
            ) from exc
        if nc != nb_classes:
            raise ValueError(
                "Strict CoreML metadata aliases disagree: "
                f"nc={nc}, nb_classes={nb_classes}."
            )

    if strict and ("imgsz_h" in meta or "imgsz_w" in meta):
        if "imgsz_h" not in meta or "imgsz_w" not in meta:
            raise ValueError(
                "Strict CoreML metadata must declare both imgsz_h and imgsz_w."
            )
        try:
            imgsz = int(meta["imgsz"])
            imgsz_h = int(meta["imgsz_h"])
            imgsz_w = int(meta["imgsz_w"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CoreML metadata imgsz/imgsz_h/imgsz_w must be integers."
            ) from exc
        if imgsz != max(imgsz_h, imgsz_w):
            raise ValueError(
                "Strict CoreML metadata aliases disagree: "
                f"imgsz={imgsz}, imgsz_h={imgsz_h}, imgsz_w={imgsz_w}."
            )

    metadata_task = normalize_task(meta.get("task"))
    supported = _normalize_metadata_supported_tasks(meta.get("supported_tasks"))
    default_task = normalize_task(meta.get("default_task"))
    if metadata_task not in supported or default_task not in supported:
        raise ValueError(
            "CoreML metadata task/default_task must both be present in "
            f"supported_tasks; got task={metadata_task!r}, "
            f"default_task={default_task!r}, supported={supported!r}."
        )

    if strict and _metadata_bool(meta.get("dynamic"), key="dynamic"):
        io = _parse_io_contract(meta)
        if io.input.shape_mode == "fixed":
            raise ValueError(
                "CoreML metadata dynamic=true conflicts with "
                "coreml_io.input.shape_mode='fixed'."
            )


class CoreMLBackend(BaseBackend):
    """CoreML inference backend (macOS only).

    Args:
        model_path: Path to a .mlpackage directory.
        nb_classes: Number of classes (default: 80, overridden by metadata if present).
        device: Ignored — CoreML routes via compute_units instead.
        compute_units: 'validated' | 'all' | 'cpu_and_gpu' | 'cpu_and_ne' |
            'cpu_only'. Default 'cpu_only' is the broadly compatible path;
            'validated' opts into exact execution-profile matching.
    """

    def __init__(
        self,
        model_path: str,
        nb_classes: int = 80,
        device: str = "auto",
        compute_units: str = "cpu_only",
        task: str | None = None,
    ):
        if sys.platform != "darwin":
            raise RuntimeError(
                f"CoreML inference requires macOS. Current platform: {sys.platform}."
            )
        try:
            import coremltools as ct
        except ImportError as e:
            raise ImportError(
                "CoreML inference requires coremltools. "
                "Install with: pip install libreyolo[coreml]"
            ) from e

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"CoreML model not found: {model_path}")

        package_spec = ct.utils.load_spec(str(path))
        package_metadata = _spec_user_defined_metadata(package_spec)
        from ..export.coreml_identity import (
            COREML_DEPLOYMENT_ABI_SCHEMA,
            validate_coreml_deployment_abi,
        )
        from ..export.coreml_profiles import (
            COREML_EXECUTION_PROFILE_VERSION,
            resolve_coreml_runtime_compute_units,
        )

        declared_profile_version = str(
            package_metadata.get(
                "coreml_execution_profile_version",
                "",
            )
        ).strip()
        declared_abi_schema = str(
            package_metadata.get("coreml_profile_abi_schema", "")
        ).strip()
        if (
            declared_profile_version == COREML_EXECUTION_PROFILE_VERSION
            or declared_abi_schema == COREML_DEPLOYMENT_ABI_SCHEMA
        ):
            validate_coreml_deployment_abi(
                package_spec,
                package_metadata,
            )
        compute_units = resolve_coreml_runtime_compute_units(
            compute_units,
            package_metadata,
        )
        _require_rfdetr_pose_cpu_profile(
            package_metadata,
            compute_units=compute_units,
        )
        spec = package_spec
        meta = package_metadata
        multifunction_route = _preflight_multifunction_spec(
            spec,
            meta,
            path=path,
        )
        compute_unit = _to_compute_unit(compute_units)
        description = getattr(spec, "description", None)
        if multifunction_route == "sam":
            self.model = ct.models.MLModel(
                str(path),
                compute_units=compute_unit,
            )
            runtime_metadata = dict(
                self.model.user_defined_metadata or {}
            )
            if runtime_metadata != package_metadata:
                raise ValueError(
                    "Core ML runtime metadata differs from the package spec "
                    "metadata that selected the execution profile."
                )
            if (
                declared_profile_version
                == COREML_EXECUTION_PROFILE_VERSION
                or declared_abi_schema == COREML_DEPLOYMENT_ABI_SCHEMA
            ):
                validate_coreml_deployment_abi(
                    self.model.get_spec(),
                    runtime_metadata,
                )
            self._initialize_sam_multifunction(
                ct=ct,
                path=path,
                spec=spec,
                meta=meta,
                compute_units=compute_units,
                requested_task=task,
            )
            return
        if multifunction_route == "ppocr":
            self._initialize_ppocr_multifunction(
                ct=ct,
                path=path,
                spec=spec,
                meta=meta,
                compute_units=compute_units,
                requested_task=task,
            )
            return
        spec_output_names = _feature_names(
            getattr(description, "output", None)
        )
        if not spec_output_names:
            raise ValueError(f"CoreML artifact {path} declares no outputs.")

        strict_contract = _artifact_uses_strict_contract(meta, path=path)
        _validate_common_metadata(meta, strict=strict_contract, path=path)
        warn_on_metadata_schema_version(
            meta,
            artifact=f"CoreML metadata for {path}",
            logger=logger,
        )
        (
            model_family,
            model_size,
            metadata_task,
            supported_tasks,
            default_task,
            names,
            imgsz,
            has_embedded_nms,
            pose_metadata,
        ) = self._parse_metadata(
            meta,
            nb_classes,
            output_names=spec_output_names,
            parse_pose_metadata=not strict_contract,
        )
        spec_has_embedded_nms = set(spec_output_names) == {
            "confidence",
            "coordinates",
        }
        if strict_contract:
            if has_embedded_nms != spec_has_embedded_nms:
                raise ValueError(
                    "Strict CoreML NMS metadata does not match the package "
                    f"outputs: nms={has_embedded_nms}, "
                    f"outputs={spec_output_names}."
                )
        else:
            has_embedded_nms = has_embedded_nms or spec_has_embedded_nms
        if strict_contract and task is not None:
            requested_task = normalize_task(task)
            if requested_task != metadata_task:
                raise ValueError(
                    "A CoreML artifact has a task-specific graph and IO "
                    f"contract (task={metadata_task!r}); runtime task override "
                    f"{requested_task!r} is incompatible. Export a separate "
                    "artifact for that task."
                )
        resolved_task = resolve_task(
            explicit_task=task,
            checkpoint_task=metadata_task,
            default_task=default_task,
            supported_tasks=supported_tasks,
        )

        family_key = (model_family or "").lower()
        if family_key == "grounding_dino":
            raise NotImplementedError(
                "Grounding DINO Core ML artifacts are disabled because the "
                "exported graph failed Apple-silicon runtime validation."
            )
        if family_key == "rfdetr" and resolved_task == "segment":
            raise NotImplementedError(
                "RF-DETR segmentation Core ML artifacts are disabled because "
                "named outputs failed prepared-graph parity on Apple M4; "
                "proposal-order drift changes learned query-slot pairings."
            )
        eomt_metadata: dict[str, Any] | None = None
        if strict_contract:
            io_contract = _parse_io_contract(meta)
        else:
            io_contract = _legacy_io_contract(
                family=family_key,
                task=resolved_task,
                spec=spec,
                output_names=spec_output_names,
            )
        if strict_contract:
            self._validate_strict_profile(
                io_contract,
                family=family_key,
                task=resolved_task,
                size=model_size,
                imgsz=imgsz,
                has_embedded_nms=has_embedded_nms,
                io_schema_version=str(meta[_COREML_IO_SCHEMA_KEY]).strip(),
                nc=len(names),
            )
            if resolved_task == "pose":
                pose_metadata = _validate_strict_pose_contract(
                    meta,
                    family=family_key,
                    nc=len(names),
                    io_contract=io_contract,
                )
            if family_key == "picosam3":
                _validate_picosam3_component_metadata(meta)
            if family_key == "rtmdet" and resolved_task == "segment":
                _validate_rtmdet_ins_metadata(meta)
            if family_key == "eomt":
                eomt_metadata = _validate_eomt_metadata(
                    meta,
                    task=resolved_task,
                    nc=len(names),
                    imgsz=imgsz,
                    io_contract=io_contract,
                )
            if family_key == "depth_anything3":
                from ..export.coreml_depth_anything3 import (
                    validate_depth_anything3_coreml_metadata,
                )

                validate_depth_anything3_coreml_metadata(meta)
            if family_key == "owlv2":
                from ..export.coreml_owlv2 import (
                    validate_owlv2_coreml_metadata,
                )

                validate_owlv2_coreml_metadata(
                    meta,
                    size=str(model_size),
                    names=names,
                )
            if family_key == "omdet_turbo":
                from ..export.coreml_omdet_turbo import (
                    validate_omdet_turbo_coreml_metadata,
                )

                validate_omdet_turbo_coreml_metadata(
                    meta,
                    size=str(model_size),
                    names=names,
                )
            if "coreml_output_names" in meta:
                declared_names = _metadata_json(
                    meta["coreml_output_names"],
                    key="coreml_output_names",
                )
                expected_names = [output.name for output in io_contract.outputs]
                if declared_names != expected_names:
                    raise ValueError(
                        "Strict CoreML output-name aliases disagree: "
                        f"coreml_output_names={declared_names!r}, "
                        f"coreml_io.outputs={expected_names!r}."
                    )
        elif family_key == "picosam3":
            raise ValueError(
                "PicoSAM3 CoreML loading requires the strict roi_component "
                "contract emitted by LibreYOLO. Re-export this artifact."
            )
        if strict_contract and family_key == "yolonas":
            self._validate_yolonas_contract(
                io_contract,
                task=resolved_task,
                imgsz=imgsz,
            )
        gaze_metadata = self._parse_gaze_metadata(
            meta,
            task=resolved_task,
            strict=strict_contract,
        )
        classification_activation = self._parse_classification_activation(
            meta,
            task=resolved_task,
            strict=strict_contract,
        )
        frozen_classes = (
            _metadata_bool(meta.get("frozen_classes", False), key="frozen_classes")
            if "frozen_classes" in meta
            else False
        )
        if (
            strict_contract
            and (
                family_key in {"clip", "siglip2"}
                and resolved_task == "classify"
                or family_key in {
                    "omdet_turbo",
                    "owlv2",
                }
                and resolved_task == "detect"
            )
            and not frozen_classes
        ):
            raise ValueError(
                f"Strict {family_key} CoreML artifacts must declare "
                "frozen_classes=true because their text tower and class "
                "vocabulary are frozen into the exported graph."
            )
        if strict_contract and resolved_task == "classify":
            expected_classify_outputs = [("class_logits", "class_logits")]
            actual_classify_outputs = [
                (output.name, output.role) for output in io_contract.outputs
            ]
            if actual_classify_outputs != expected_classify_outputs:
                raise ValueError(
                    "Strict classification CoreML artifacts must declare one "
                    "class_logits output; "
                    f"got {actual_classify_outputs}."
                )
        if strict_contract and resolved_task == "gaze":
            expected_gaze_outputs = [
                ("yaw_logits", "yaw_logits"),
                ("pitch_logits", "pitch_logits"),
            ]
            actual_gaze_outputs = [
                (output.name, output.role) for output in io_contract.outputs
            ]
            if actual_gaze_outputs != expected_gaze_outputs:
                raise ValueError(
                    "Strict gaze CoreML artifacts must declare ordered "
                    f"yaw/pitch logits outputs {expected_gaze_outputs}; got "
                    f"{actual_gaze_outputs}."
                )

        self._strict_contract = strict_contract
        self._io_schema_version = (
            str(meta[_COREML_IO_SCHEMA_KEY]).strip() if strict_contract else None
        )
        self.io_contract = io_contract
        self.input_contract = io_contract.input
        # Contract order, not protobuf or prediction-dict order, is the runtime
        # ABI for all current artifacts.
        self.output_names = [output.name for output in io_contract.outputs]
        self.output_roles = [output.role for output in io_contract.outputs]
        if (
            resolved_task in {"detect", "segment", "pose", "obb"}
            and family_key != "picosam3"
        ):
            validation = io_contract.input.validation
            if validation.color != "rgb" or validation.value_range != "0_255":
                raise ValueError(
                    "Detection-style CoreML validation uses the canonical "
                    "RGB 0..255 adapter; coreml_io.validation must declare "
                    "color='rgb' and range='0_255'."
                )
        nms_conf = self._optional_float_metadata(meta, "nms_conf")
        nms_iou = self._optional_float_metadata(meta, "nms_iou")
        nms_max_det = self._optional_int_metadata(meta, "max_det")
        if strict_contract:
            nms_keys = {"nms_conf", "nms_iou", "max_det"}.intersection(meta)
            if has_embedded_nms:
                if family_key not in {"yolo9", "yolox"} or resolved_task != "detect":
                    raise ValueError(
                        "Strict CoreML embedded NMS is supported only for "
                        "YOLO9 and YOLOX detection artifacts."
                    )
                missing_thresholds = [
                    key
                    for key, value in (
                        ("nms_conf", nms_conf),
                        ("nms_iou", nms_iou),
                    )
                    if value is None
                ]
                if missing_thresholds:
                    raise ValueError(
                        "Strict embedded-NMS CoreML artifacts must declare "
                        f"{missing_thresholds}."
                    )
                for key, value in (("nms_conf", nms_conf), ("nms_iou", nms_iou)):
                    if not 0.0 <= float(value) <= 1.0:
                        raise ValueError(
                            f"CoreML metadata {key!r} must be in [0, 1]."
                        )
            elif nms_keys:
                raise ValueError(
                    "Raw-output CoreML artifacts must not declare baked NMS "
                    f"metadata {sorted(nms_keys)}."
                )
        self._has_embedded_nms = has_embedded_nms
        self.embedded_nms = has_embedded_nms
        self._nms_conf = nms_conf
        self._nms_iou = nms_iou
        self._nms_max_det = nms_max_det

        if strict_contract:
            if "crop_pct" in meta:
                try:
                    declared_crop_pct = float(meta["crop_pct"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "CoreML metadata 'crop_pct' must be numeric."
                    ) from exc
                if not np.isclose(
                    declared_crop_pct,
                    io_contract.input.crop_pct,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        "Strict CoreML crop_pct alias disagrees with coreml_io: "
                        f"metadata={declared_crop_pct}, "
                        f"contract={io_contract.input.crop_pct}."
                    )
            if "interpolation" in meta:
                declared_interpolation = str(meta["interpolation"]).strip().lower()
                if declared_interpolation != io_contract.input.interpolation:
                    raise ValueError(
                        "Strict CoreML interpolation alias disagrees with "
                        "coreml_io: "
                        f"metadata={declared_interpolation!r}, "
                        f"contract={io_contract.input.interpolation!r}."
                    )
            crop_pct = io_contract.input.crop_pct
            interpolation = io_contract.input.interpolation
        else:
            crop_pct = float(meta.get("crop_pct", io_contract.input.crop_pct))
            interpolation = str(
                meta.get("interpolation", io_contract.input.interpolation)
            )

        super().__init__(
            model_path=str(path),
            nb_classes=len(names) if names else nb_classes,
            device="coreml",
            imgsz=imgsz,
            model_family=model_family,
            names=names if names else self.build_names(nb_classes),
            model_size=model_size,
            task=resolved_task,
            supported_tasks=supported_tasks,
            default_task=default_task,
            crop_pct=crop_pct,
            interpolation=interpolation,
            classification_activation=classification_activation,
            **gaze_metadata,
            **pose_metadata,
        )
        self.frozen_classes = frozen_classes
        if family_key == "eomt":
            if eomt_metadata is None:
                raise ValueError(
                    "EoMT CoreML loading requires the strict compact-query "
                    "component contract emitted by LibreYOLO."
                )
            from ..models.eomt.model import LibreEoMT

            decoder = object.__new__(LibreEoMT)
            decoder.task = resolved_task
            decoder.input_size = int(eomt_metadata["image_size"])
            decoder.num_queries = int(eomt_metadata["num_queries"])
            decoder.nb_classes = len(names)
            decoder.names = self.names
            thing_ids = eomt_metadata["thing_class_ids"]
            decoder.thing_class_ids = (
                set(int(value) for value in thing_ids)
                if thing_ids is not None
                else None
            )
            self.num_queries = decoder.num_queries
            self.thing_class_ids = decoder.thing_class_ids
            self._eomt_decoder = decoder
        self._validate_spec_contract(spec, spec_output_names)
        self.model = ct.models.MLModel(
            str(path),
            compute_units=compute_unit,
        )
        runtime_metadata = dict(
            self.model.user_defined_metadata or {}
        )
        if runtime_metadata != package_metadata:
            raise ValueError(
                "Core ML runtime metadata differs from the package spec "
                "metadata that selected the execution profile."
            )
        if (
            declared_profile_version == COREML_EXECUTION_PROFILE_VERSION
            or declared_abi_schema == COREML_DEPLOYMENT_ABI_SCHEMA
        ):
            validate_coreml_deployment_abi(
                self.model.get_spec(),
                runtime_metadata,
            )

        # Dense validators choose their dataset geometry from these attributes.
        # Keeping them derived from the artifact contract prevents validation
        # from using a different canvas from predict().
        resize_mode = (
            "stretch" if self.input_contract.geometry == "stretch" else "letterbox"
        )
        if self.task == "depth":
            self.depth_resize_mode = resize_mode
            self.depth_resize_backend = self.input_contract.resize_backend
            self.depth_resize_interpolation = self.input_contract.interpolation
        if self.task == "semantic" and family_key == "eomt":
            self.semantic_resize_mode = "split"
            self.semantic_imgsz_divisor = 16
            self.semantic_resize_backend = "torchvision"
            self.semantic_resize_interpolation = "bilinear"
            self.semantic_resize_rounding = "floor"
        elif self.task == "semantic":
            self.semantic_resize_mode = resize_mode
            self.semantic_resize_backend = self.input_contract.resize_backend
            self.semantic_resize_interpolation = self.input_contract.interpolation
            self.semantic_resize_rounding = (
                "floor"
                if self.input_contract.geometry == "letterbox_top_left"
                else "round"
            )

    def _initialize_sam_multifunction(
        self,
        *,
        ct: Any,
        path: Path,
        spec: Any,
        meta: dict[str, Any],
        compute_units: str,
        requested_task: str | None,
    ) -> None:
        """Validate a LibreSAM package and prepare lazy exact-P dispatch."""
        from ..export.coreml import _validate_sam_multifunction_spec
        from ..export.coreml_sam import (
            SAM_COREML_ENCODER_FUNCTION,
            sam_coreml_runtime_function_names,
            validate_sam_coreml_metadata,
            validate_sam_coreml_profile,
        )
        from .coreml_sam import SAMCoreMLFunction

        required_common = {
            "schema_version",
            "libreyolo_version",
            "libreyolo_producer",
            "artifact_format",
            "model_family",
            "size",
            "model_size",
            "task",
            "supported_tasks",
            "default_task",
            "names",
            "nc",
            "nb_classes",
            "imgsz",
            "imgsz_h",
            "imgsz_w",
            "precision",
            "dynamic",
        }
        missing = sorted(
            key for key in required_common if meta.get(key) in (None, "")
        )
        if missing:
            raise ValueError(
                "Strict LibreSAM Core ML metadata is incomplete; "
                f"missing {missing}."
            )
        if str(meta["libreyolo_producer"]) != _COREML_PRODUCER:
            raise ValueError(
                "LibreSAM Core ML packages require "
                "libreyolo_producer='libreyolo'."
            )
        if str(meta["artifact_format"]).strip().lower() != "coreml":
            raise ValueError(
                "LibreSAM Core ML metadata artifact_format must be 'coreml'."
            )
        family = str(meta["model_family"]).strip().lower()
        if family not in {"edgetam", "mobilesam", "sam", "sam2", "sam3"}:
            raise ValueError(
                "LibreSAM multifunction metadata contains an unsupported "
                f"model_family={family!r}."
            )
        if str(meta["task"]).strip().lower() != "segment":
            raise ValueError(
                "LibreSAM multifunction metadata task must be 'segment'."
            )
        if (
            requested_task is not None
            and normalize_task(requested_task) != "segment"
        ):
            raise ValueError(
                "LibreSAM Core ML packages contain a segmentation-specific "
                f"function bundle; task={requested_task!r} is incompatible."
            )
        supported = _normalize_metadata_supported_tasks(meta["supported_tasks"])
        if (
            supported != ("segment",)
            or normalize_task(meta["default_task"]) != "segment"
        ):
            raise ValueError(
                "LibreSAM Core ML supported_tasks/default_task must be "
                "exactly ['segment']/'segment'."
            )
        size = str(meta["size"]).strip().lower()
        if str(meta["model_size"]).strip().lower() != size:
            raise ValueError("LibreSAM Core ML size/model_size aliases disagree.")
        if str(meta["precision"]).strip().lower() != "fp32":
            raise ValueError("LibreSAM Core ML packages must declare FP32.")
        if _metadata_bool(meta["dynamic"], key="dynamic"):
            raise ValueError(
                "LibreSAM Core ML packages must declare dynamic=false because "
                "every admitted point count is an exact fixed-shape function."
            )
        if "nms" in meta and _metadata_bool(meta["nms"], key="nms"):
            raise ValueError(
                "LibreSAM Core ML packages must not declare embedded NMS."
            )

        try:
            raw_names = _metadata_json(meta["names"], key="names")
            if not isinstance(raw_names, dict):
                raise TypeError
            names = {int(key): value for key, value in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "LibreSAM Core ML names must be a JSON object."
            ) from exc
        nc = _strict_metadata_int(meta["nc"], key="nc")
        nb_classes = _strict_metadata_int(
            meta["nb_classes"],
            key="nb_classes",
        )
        if (
            nc != 1
            or nb_classes != 1
            or names != {0: "object"}
        ):
            raise ValueError(
                "LibreSAM Core ML public class metadata must be exactly "
                "{0: 'object'}."
            )

        canonical = validate_sam_coreml_metadata(meta)
        profile_values = canonical["sam_coreml_profile"]
        profile = validate_sam_coreml_profile(
            family=profile_values["family"],
            size=profile_values["size"],
            precision=profile_values["precision"],
            prompt_max_points=profile_values["prompt_max_points"],
        )
        metadata_imgsz = _read_metadata_imgsz(
            meta,
            family,
            artifact="LibreSAM Core ML metadata",
        )
        if metadata_imgsz is None:
            raise ValueError(
                "LibreSAM Core ML metadata must declare its encoder frame."
            )
        if _imgsz_hw(metadata_imgsz) != (
            profile.image_size,
            profile.image_size,
        ):
            raise ValueError(
                "LibreSAM Core ML image-size metadata disagrees with the "
                f"profile: metadata={_imgsz_hw(metadata_imgsz)}, "
                f"profile={profile.image_size}."
            )
        specification_version = int(
            getattr(spec, "specificationVersion", 0) or 0
        )
        if specification_version < 9:
            raise ValueError(
                "LibreSAM multifunction packages require the iOS18/macOS15 "
                f"Core ML specification (version >= 9), got "
                f"{specification_version}."
            )
        try:
            _validate_sam_multifunction_spec(spec, profile=profile)
        except RuntimeError as exc:
            raise ValueError(
                f"Invalid LibreSAM Core ML multifunction spec: {exc}"
            ) from exc

        compute_unit = _to_compute_unit(compute_units)
        actual_name = getattr(
            self.model,
            "function_name",
            SAM_COREML_ENCODER_FUNCTION,
        )
        if actual_name != SAM_COREML_ENCODER_FUNCTION:
            raise ValueError(
                "Core ML loaded the wrong default LibreSAM function: "
                f"expected {SAM_COREML_ENCODER_FUNCTION!r}, "
                f"got {actual_name!r}."
            )
        self._sam_ct = ct
        self._sam_path = path
        self._sam_compute_unit = compute_unit
        self._sam_runtime_function_names = frozenset(
            sam_coreml_runtime_function_names(profile)
        )
        self._sam_runtimes = {SAM_COREML_ENCODER_FUNCTION: self.model}
        self._sam_functions = {
            SAM_COREML_ENCODER_FUNCTION: SAMCoreMLFunction(
                self.model,
                function_name=SAM_COREML_ENCODER_FUNCTION,
                profile=profile,
            )
        }
        self._sam_runtime_lock = RLock()
        self._sam_profile = profile
        self._sam_image = None
        self._sam_image_path = None
        self._sam_image_encoding = None
        self._sam_image_embeddings = None
        self._strict_contract = True
        self._io_schema_version = None
        self._has_embedded_nms = False
        self.embedded_nms = False
        self._nms_conf = None
        self._nms_iou = None
        self._nms_max_det = None
        self.output_names = list(profile.embedding_names) + [
            "low_res_masks",
            "iou_scores",
        ]
        self.output_roles = [
            *("image_embedding" for _ in profile.embedding_names),
            "mask_logits",
            "predicted_iou",
        ]
        release_gap = canonical.get("release_notice_gap")
        if release_gap:
            logger.warning("LibreSAM Core ML release-notice gap: %s", release_gap)
        if not canonical["artifact_redistributable"]:
            logger.warning(
                "This %s Core ML artifact is marked non-redistributable by "
                "its embedded license contract.",
                family,
            )
        warn_on_metadata_schema_version(
            meta,
            artifact=f"LibreSAM Core ML metadata for {path}",
            logger=logger,
        )
        super().__init__(
            model_path=str(path),
            nb_classes=1,
            device="coreml",
            imgsz=profile.image_size,
            model_family=family,
            names=names,
            model_size=size,
            task="segment",
            supported_tasks=("segment",),
            default_task="segment",
        )

    def _initialize_ppocr_multifunction(
        self,
        *,
        ct: Any,
        path: Path,
        spec: Any,
        meta: dict[str, Any],
        compute_units: str,
        requested_task: str | None,
    ) -> None:
        """Load and validate both functions of a strict LibrePPOCR package."""
        from ..export.coreml import _validate_ppocr_multifunction_spec
        from ..export.coreml_ppocr import (
            PPOCR_COREML_DETECTOR_FUNCTION,
            PPOCR_COREML_DETECTOR_INPUT,
            PPOCR_COREML_DETECTOR_OUTPUT,
            PPOCR_COREML_RECOGNIZER_FUNCTION,
            PPOCR_COREML_RECOGNIZER_INPUT,
            PPOCR_COREML_RECOGNIZER_OUTPUT,
            validate_ppocr_coreml_metadata,
            validate_ppocr_coreml_profile,
        )

        required_common = {
            "schema_version",
            "libreyolo_version",
            "libreyolo_producer",
            "artifact_format",
            "model_family",
            "size",
            "model_size",
            "task",
            "supported_tasks",
            "default_task",
            "names",
            "nc",
            "nb_classes",
            "imgsz",
            "imgsz_h",
            "imgsz_w",
            "precision",
            "dynamic",
        }
        missing = sorted(
            key for key in required_common if meta.get(key) in (None, "")
        )
        if missing:
            raise ValueError(
                "Strict LibrePPOCR Core ML metadata is incomplete; "
                f"missing {missing}."
            )
        if str(meta["libreyolo_producer"]) != _COREML_PRODUCER:
            raise ValueError(
                "LibrePPOCR Core ML packages require "
                "libreyolo_producer='libreyolo'."
            )
        if str(meta["artifact_format"]).strip().lower() != "coreml":
            raise ValueError(
                "LibrePPOCR Core ML metadata artifact_format must be 'coreml'."
            )
        if str(meta["model_family"]).strip().lower() != "ppocr":
            raise ValueError(
                "LibrePPOCR multifunction metadata model_family must be 'ppocr'."
            )
        if str(meta["task"]).strip().lower() != "ocr":
            raise ValueError(
                "LibrePPOCR multifunction metadata task must be 'ocr'."
            )
        if requested_task is not None and normalize_task(requested_task) != "ocr":
            raise ValueError(
                "LibrePPOCR Core ML packages contain an OCR-specific function "
                f"bundle; task={requested_task!r} is incompatible."
            )
        supported = _normalize_metadata_supported_tasks(meta["supported_tasks"])
        if supported != ("ocr",) or normalize_task(meta["default_task"]) != "ocr":
            raise ValueError(
                "LibrePPOCR Core ML supported_tasks/default_task must be "
                "exactly ['ocr']/'ocr'."
            )
        size = str(meta["size"]).strip().lower()
        if str(meta["model_size"]).strip().lower() != size:
            raise ValueError(
                "LibrePPOCR Core ML size/model_size aliases disagree."
            )
        if str(meta["precision"]).strip().lower() != "fp32":
            raise ValueError("LibrePPOCR Core ML packages must declare FP32.")
        if not _metadata_bool(meta["dynamic"], key="dynamic"):
            raise ValueError(
                "LibrePPOCR Core ML packages must declare dynamic=true for "
                "their bounded RangeDim axes."
            )
        if "nms" in meta and _metadata_bool(meta["nms"], key="nms"):
            raise ValueError(
                "LibrePPOCR Core ML packages must not declare embedded NMS."
            )

        try:
            raw_names = _metadata_json(meta["names"], key="names")
            if not isinstance(raw_names, dict):
                raise TypeError
            names = {int(key): value for key, value in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "LibrePPOCR Core ML names must be a JSON object."
            ) from exc
        nc = _strict_metadata_int(meta["nc"], key="nc")
        nb_classes = _strict_metadata_int(
            meta["nb_classes"],
            key="nb_classes",
        )
        if nc != 1 or nb_classes != 1 or names != {0: "text"}:
            raise ValueError(
                "LibrePPOCR Core ML public class metadata must be exactly "
                "{0: 'text'}; "
                f"got nc={nc}, nb_classes={nb_classes}, names={names!r}."
            )

        canonical = validate_ppocr_coreml_metadata(meta)
        profile_values = canonical["ppocr_coreml_profile"]
        profile = validate_ppocr_coreml_profile(
            size=profile_values["size"],
            precision=profile_values["precision"],
            det_limit_side_len=profile_values["det_limit_side_len"],
            rec_batch_max=profile_values["rec_batch_max"],
            rec_max_width=profile_values["rec_max_width"],
        )
        metadata_imgsz = _read_metadata_imgsz(
            meta,
            "ppocr",
            artifact="LibrePPOCR Core ML metadata",
        )
        if metadata_imgsz is None:
            raise ValueError(
                "LibrePPOCR Core ML metadata must declare its detector limit."
            )
        input_h, input_w = _imgsz_hw(metadata_imgsz)
        if (input_h, input_w) != (
            profile.det_tensor_upper,
            profile.det_tensor_upper,
        ):
            raise ValueError(
                "LibrePPOCR Core ML detector metadata disagrees with the "
                f"bounded profile: metadata={(input_h, input_w)}, "
                f"profile={profile.det_tensor_upper}."
            )
        specification_version = int(
            getattr(spec, "specificationVersion", 0) or 0
        )
        if specification_version < 9:
            raise ValueError(
                "LibrePPOCR multifunction packages require the iOS18/macOS15 "
                f"Core ML specification (version >= 9), got "
                f"{specification_version}."
            )
        try:
            _validate_ppocr_multifunction_spec(spec, profile=profile)
        except RuntimeError as exc:
            raise ValueError(
                f"Invalid LibrePPOCR Core ML multifunction spec: {exc}"
            ) from exc

        compute_unit = _to_compute_unit(compute_units)
        detector_runtime = ct.models.MLModel(
            str(path),
            compute_units=compute_unit,
            function_name=PPOCR_COREML_DETECTOR_FUNCTION,
        )
        recognizer_runtime = ct.models.MLModel(
            str(path),
            compute_units=compute_unit,
            function_name=PPOCR_COREML_RECOGNIZER_FUNCTION,
        )
        for expected_name, runtime in (
            (PPOCR_COREML_DETECTOR_FUNCTION, detector_runtime),
            (PPOCR_COREML_RECOGNIZER_FUNCTION, recognizer_runtime),
        ):
            runtime_metadata = dict(
                runtime.user_defined_metadata or {}
            )
            if runtime_metadata != meta:
                raise ValueError(
                    "LibrePPOCR Core ML runtime metadata differs from the "
                    "package spec validated before function compilation."
                )
            actual_name = getattr(runtime, "function_name", expected_name)
            if actual_name != expected_name:
                raise ValueError(
                    "Core ML loaded the wrong LibrePPOCR function: "
                    f"expected {expected_name!r}, got {actual_name!r}."
                )

        detector = _PPOCRCoreMLFunction(
            detector_runtime,
            function_name=PPOCR_COREML_DETECTOR_FUNCTION,
            input_name=PPOCR_COREML_DETECTOR_INPUT,
            output_name=PPOCR_COREML_DETECTOR_OUTPUT,
            profile=profile,
            rec_num_classes=canonical["rec_num_classes"],
        )
        recognizer = _PPOCRCoreMLFunction(
            recognizer_runtime,
            function_name=PPOCR_COREML_RECOGNIZER_FUNCTION,
            input_name=PPOCR_COREML_RECOGNIZER_INPUT,
            output_name=PPOCR_COREML_RECOGNIZER_OUTPUT,
            profile=profile,
            rec_num_classes=canonical["rec_num_classes"],
        )
        runner_proxy = _PPOCRCoreMLRunnerProxy(
            detector=detector,
            recognizer=recognizer,
            profile=profile,
            charset=canonical["charset"],
            pipeline=canonical["pipeline"],
            names=names,
        )
        from ..models.ppocr.inference import OCRInferenceRunner

        self.model = detector_runtime
        self._ppocr_detector_model = detector_runtime
        self._ppocr_recognizer_model = recognizer_runtime
        self._ppocr_profile = profile
        self._ppocr_runner_proxy = runner_proxy
        self._ppocr_runner = OCRInferenceRunner(runner_proxy)
        self.charset = list(canonical["charset"])
        self.pipeline_config = dict(canonical["pipeline"])
        self.rec_num_classes = int(canonical["rec_num_classes"])
        self._strict_contract = True
        self._io_schema_version = None
        self._has_embedded_nms = False
        self.embedded_nms = False
        self._nms_conf = None
        self._nms_iou = None
        self._nms_max_det = None
        self.output_names = [
            PPOCR_COREML_DETECTOR_OUTPUT,
            PPOCR_COREML_RECOGNIZER_OUTPUT,
        ]
        self.output_roles = [
            "text_probability_map",
            "ctc_probabilities",
        ]
        warn_on_metadata_schema_version(
            meta,
            artifact=f"LibrePPOCR Core ML metadata for {path}",
            logger=logger,
        )
        super().__init__(
            model_path=str(path),
            nb_classes=1,
            device="coreml",
            imgsz=profile.det_tensor_upper,
            model_family="ppocr",
            names=names,
            model_size=size,
            task="ocr",
            supported_tasks=("ocr",),
            default_task="ocr",
        )

    def set_image(self, source, color_format: str = "auto") -> "CoreMLBackend":
        """Cache and encode one source image for promptable Core ML families."""
        if self.model_family in _SAM_COREML_FAMILIES:
            if isinstance(source, (list, tuple)) or source is None:
                raise ValueError("LibreSAM set_image() requires one source image.")
            image = ImageLoader.load(source, color_format=color_format)
            encoding, embeddings = self._encode_sam_image(image)
            self._sam_image = image
            self._sam_image_path = (
                source if isinstance(source, (str, Path)) else None
            )
            self._sam_image_encoding = encoding
            self._sam_image_embeddings = embeddings
            return self
        if self.model_family != "picosam3":
            raise NotImplementedError(
                "set_image() is available only for promptable SAM Core ML "
                "artifacts."
            )
        if isinstance(source, (list, tuple)) or source is None:
            raise ValueError("PicoSAM3 set_image() requires one source image.")
        self._picosam3_image = ImageLoader.load(source, color_format=color_format)
        self._picosam3_image_path = (
            source if isinstance(source, (str, Path)) else None
        )
        return self

    def reset_image(self) -> "CoreMLBackend":
        """Clear an encode-once SAM/PicoSAM3 image session."""
        if self.model_family in _SAM_COREML_FAMILIES:
            self._sam_image = None
            self._sam_image_path = None
            self._sam_image_encoding = None
            self._sam_image_embeddings = None
            return self
        if self.model_family == "picosam3":
            self._picosam3_image = None
            self._picosam3_image_path = None
            return self
        raise NotImplementedError(
            "reset_image() is available only for promptable SAM Core ML "
            "artifacts."
        )

    def __call__(self, source=None, *args, **kwargs):
        """Route host-orchestrated Core ML component profiles."""
        if self.model_family == "owlv2" and "conf" not in kwargs:
            # Preserve LibreOWLv2's public default; generic exported
            # detectors otherwise default to 0.25.
            kwargs["conf"] = 0.1
        if self.model_family == "omdet_turbo":
            # Preserve LibreOMDetTurbo's public detector defaults.
            kwargs.setdefault("conf", 0.3)
            kwargs.setdefault("iou", 0.5)
        if self.model_family in _SAM_COREML_FAMILIES:
            if args:
                raise TypeError("LibreSAM prompts must be passed by keyword.")
            return self._predict_sam_components(source, **kwargs)
        if self.model_family == "ppocr":
            if args:
                raise TypeError(
                    "LibrePPOCR Core ML inference arguments must be passed "
                    "by keyword."
                )
            return self._predict_ppocr_pipeline(source, **kwargs)
        if self.model_family != "picosam3":
            return super().__call__(source, *args, **kwargs)
        if args:
            raise TypeError("PicoSAM3 prompts must be passed by keyword.")
        return self._predict_picosam3_component(source, **kwargs)

    def val(self, *args, **kwargs):
        """Validate only Core ML families with a fixed dataset metric contract."""
        if self.model_family in _SAM_COREML_FAMILIES:
            raise NotImplementedError(
                "LibreSAM is promptable and has no fixed class-set validation "
                "contract. Validate it with explicit prompts and masks instead."
            )
        if self.model_family in {
            "omdet_turbo",
            "owlv2",
        }:
            raise NotImplementedError(
                "Frozen-vocabulary open-vocabulary Core ML validation needs a dedicated "
                "open-vocabulary dataset contract. Use predict() with the "
                "classes frozen into the artifact."
            )
        return super().val(*args, **kwargs)

    def _predict_ppocr_pipeline(
        self,
        source=None,
        *,
        imgsz: ImageSize | None = None,
        rec_batch: int | None = None,
        batch: int = 1,
        device: str | None = None,
        **kwargs,
    ):
        """Run DB/CTC host orchestration over the two named package functions."""
        if batch != 1:
            raise ValueError(
                "LibrePPOCR processes source images sequentially. Use batch=1 "
                "and configure recognition crops with rec_batch=...."
            )
        if rec_batch is None:
            rec_batch_value = min(
                6,
                self._ppocr_profile.rec_batch_max,
            )
        elif isinstance(rec_batch, bool) or not isinstance(rec_batch, Integral):
            raise ValueError("LibrePPOCR rec_batch must be a positive integer.")
        else:
            rec_batch_value = int(rec_batch)
        if not (
            1 <= rec_batch_value <= self._ppocr_profile.rec_batch_max
        ):
            raise ValueError(
                "LibrePPOCR Core ML rec_batch must be within [1, "
                f"{self._ppocr_profile.rec_batch_max}], got {rec_batch!r}."
            )
        resolved_imgsz: int | None
        if imgsz is None:
            resolved_imgsz = None
        elif isinstance(imgsz, (tuple, list)):
            if len(imgsz) != 2 or int(imgsz[0]) != int(imgsz[1]):
                raise ValueError(
                    "LibrePPOCR uses one detector long-side limit; imgsz must "
                    "be an int or equal square pair."
                )
            resolved_imgsz = int(imgsz[0])
        else:
            resolved_imgsz = int(imgsz)
        if resolved_imgsz is not None and not (
            32 <= resolved_imgsz <= self._ppocr_profile.det_limit_side_len
        ):
            raise ValueError(
                "LibrePPOCR Core ML imgsz must be within [32, "
                f"{self._ppocr_profile.det_limit_side_len}], got "
                f"{resolved_imgsz}."
            )
        if device not in (None, "", "auto", "coreml", self.device):
            logger.warning(
                "LibrePPOCR Core ML functions are already loaded with their "
                "selected compute_units; predict(device=%s) is ignored.",
                device,
            )
        return self._ppocr_runner(
            source,
            imgsz=resolved_imgsz,
            rec_batch=rec_batch_value,
            device=None,
            **kwargs,
        )

    def _sam_function(self, function_name: str):
        """Keep exactly one native SAM function proxy resident at a time."""
        from ..export.coreml_sam import SAM_COREML_ENCODER_FUNCTION
        from .coreml_sam import SAMCoreMLFunction

        cached = self._sam_functions.get(function_name)
        if cached is not None:
            return cached
        if function_name not in self._sam_runtime_function_names:
            raise ValueError(
                f"Unknown LibreSAM Core ML runtime function {function_name!r}."
            )
        # Core ML Tools 9 on the validation M4 can abort in native proxy
        # construction when a second function from the same multifunction
        # package is loaded while the first proxy remains resident. A SAM
        # session therefore caches one active function, not an unbounded table
        # of up to 259 native proxies. Embeddings are ordinary tensors and
        # remain cached independently when the encoder proxy is released.
        self.model = None
        self._sam_functions.clear()
        self._sam_runtimes.clear()
        gc.collect()
        load_kwargs = {
            "compute_units": self._sam_compute_unit,
        }
        if function_name != SAM_COREML_ENCODER_FUNCTION:
            load_kwargs["function_name"] = function_name
        runtime = self._sam_ct.models.MLModel(
            str(self._sam_path),
            **load_kwargs,
        )
        actual_name = getattr(runtime, "function_name", function_name)
        if actual_name != function_name:
            raise ValueError(
                "Core ML loaded the wrong LibreSAM function: "
                f"expected {function_name!r}, got {actual_name!r}."
            )
        function = SAMCoreMLFunction(
            runtime,
            function_name=function_name,
            profile=self._sam_profile,
        )
        self.model = runtime
        self._sam_runtimes[function_name] = runtime
        self._sam_functions[function_name] = function
        return function

    def _run_sam_encoder(self, encoding):
        from ..export.coreml_sam import (
            SAM_COREML_ENCODER_FUNCTION,
            SAM_COREML_ENCODER_INPUT,
        )

        with self._sam_runtime_lock:
            outputs = self._sam_function(SAM_COREML_ENCODER_FUNCTION)(
                {SAM_COREML_ENCODER_INPUT: encoding.pixel_values}
            )
        return {
            name: outputs[name]
            for name in self._sam_profile.embedding_names
        }

    def _encode_sam_image(self, image: Image.Image):
        from .coreml_sam import prepare_sam_coreml_image

        encoding = prepare_sam_coreml_image(
            image,
            profile=self._sam_profile,
        )
        return encoding, self._run_sam_encoder(encoding)

    def _run_sam_decoder(
        self,
        function_name: str,
        embeddings: Mapping[str, torch.Tensor],
        *,
        point_coords: torch.Tensor | None = None,
        point_labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from ..export.coreml_sam import (
            SAM_COREML_BOXES_INPUT,
            SAM_COREML_IOU_OUTPUT,
            SAM_COREML_MASKS_OUTPUT,
            SAM_COREML_POINT_COORDS_INPUT,
            SAM_COREML_POINT_LABELS_INPUT,
            parse_sam_coreml_runtime_function,
            sam_coreml_runtime_function_name,
        )

        values: dict[str, torch.Tensor] = {
            name: embeddings[name] for name in self._sam_profile.embedding_names
        }
        if point_coords is not None:
            values[SAM_COREML_POINT_COORDS_INPUT] = point_coords
            if point_labels is None:
                raise RuntimeError("Point coordinates require point labels.")
            values[SAM_COREML_POINT_LABELS_INPUT] = point_labels
        if boxes is not None:
            values[SAM_COREML_BOXES_INPUT] = boxes
        runtime_name = function_name
        if point_coords is not None:
            if not torch.is_tensor(point_coords) or point_coords.ndim != 4:
                raise ValueError(
                    "LibreSAM point coordinates must be a rank-4 tensor."
                )
            runtime_name = sam_coreml_runtime_function_name(
                function_name,
                point_count=int(point_coords.shape[2]),
            )
            parse_sam_coreml_runtime_function(
                runtime_name,
                profile=self._sam_profile,
            )
        # Function selection, native proxy replacement, and prediction are one
        # critical section. Core ML Tools 9 can abort the process if another
        # thread replaces a multifunction proxy while prediction is in flight.
        with self._sam_runtime_lock:
            outputs = self._sam_function(runtime_name)(values)
        return (
            outputs[SAM_COREML_MASKS_OUTPUT],
            outputs[SAM_COREML_IOU_OUTPUT],
        )

    @staticmethod
    def _empty_sam_prompt(value: Any) -> bool:
        if value is None:
            return True
        try:
            return len(value) == 0
        except TypeError:
            return False

    def _sam_results(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        image: Image.Image,
        image_path: str | Path | None,
        *,
        conf_threshold: float,
        max_det: int,
    ) -> Results:
        from torchvision.ops import masks_to_boxes

        masks = masks.detach().cpu().bool()
        scores = scores.detach().cpu().float()
        if conf_threshold > 0:
            keep = scores >= conf_threshold
            masks, scores = masks[keep], scores[keep]
        if masks.shape[0]:
            nonempty = masks.flatten(1).any(dim=1)
            masks, scores = masks[nonempty], scores[nonempty]
        if masks.shape[0] > max_det:
            top = scores.argsort(descending=True)[:max_det]
            masks, scores = masks[top], scores[top]

        orig_shape = (image.height, image.width)
        if not masks.shape[0]:
            return Results(
                boxes=Boxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    torch.zeros(0, dtype=torch.float32),
                    torch.zeros(0, dtype=torch.float32),
                ),
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=self.names,
            )
        boxes = masks_to_boxes(masks.float())
        return Results(
            boxes=Boxes(
                boxes,
                scores,
                torch.zeros(masks.shape[0], dtype=torch.float32),
            ),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
            masks=Masks(masks, orig_shape),
        )

    def _sam_segment_everything(
        self,
        *,
        image: Image.Image,
        image_path: str | Path | None,
        encoding: Any,
        embeddings: Mapping[str, torch.Tensor],
        points_per_side: int,
        score_threshold: float,
        max_det: int,
    ) -> Results:
        from torchvision.ops import batched_nms, masks_to_boxes

        from ..models.sam.prompts import build_point_grid
        from .coreml_sam import (
            postprocess_sam_coreml_masks,
            transform_sam_coreml_points,
        )

        if isinstance(points_per_side, bool) or not isinstance(
            points_per_side, Integral
        ):
            raise ValueError("points_per_side must be a positive integer.")
        points_per_side = int(points_per_side)
        grid = build_point_grid(points_per_side)
        width, height = image.size
        all_masks: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        function_name = "decode_points_multimask"
        for x, y in grid:
            point_coords = transform_sam_coreml_points(
                [[float(x * width), float(y * height)]],
                encoding=encoding,
                profile=self._sam_profile,
            )
            point_labels = torch.ones((1, 1, 1), dtype=torch.int32)
            low_res, iou_scores = self._run_sam_decoder(
                function_name,
                embeddings,
                point_coords=point_coords,
                point_labels=point_labels,
            )
            masks = postprocess_sam_coreml_masks(
                low_res,
                encoding=encoding,
                profile=self._sam_profile,
            )
            scores = iou_scores[0, 0].float()
            best = int(scores.argmax())
            all_masks.append(masks[best])
            all_scores.append(scores[best])

        masks = torch.stack(all_masks)
        scores = torch.stack(all_scores)
        if score_threshold > 0:
            keep = scores >= score_threshold
            masks, scores = masks[keep], scores[keep]
        if masks.shape[0]:
            nonempty = masks.flatten(1).any(dim=1)
            masks, scores = masks[nonempty], scores[nonempty]
        if not masks.shape[0]:
            return self._sam_results(
                masks,
                scores,
                image,
                image_path,
                conf_threshold=0.0,
                max_det=max_det,
            )
        boxes = masks_to_boxes(masks.float())
        classes = torch.zeros(masks.shape[0], dtype=torch.int64)
        keep = batched_nms(boxes, scores, classes, 0.7)[:max_det]
        return self._sam_results(
            masks[keep],
            scores[keep],
            image,
            image_path,
            conf_threshold=0.0,
            max_det=max_det,
        )

    def _predict_sam_components(
        self,
        source=None,
        *,
        points=None,
        bboxes=None,
        labels=None,
        masks=None,
        text: str | None = None,
        conf: float | None = None,
        multimask: bool | None = None,
        max_det: int = 300,
        device: str | None = None,
        color_format: str = "auto",
        points_per_side: int | None = None,
        imgsz: ImageSize | None = None,
        classes=None,
        iou: float = 0.45,
        save: bool = False,
        output_path: str | None = None,
        batch: int = 1,
        stream: bool = False,
        vid_stride: int = 1,
        show: bool = False,
        **kwargs,
    ) -> Results:
        """Run exact host orchestration over a fixed-function SAM package."""
        del iou, vid_stride
        normalize_predict_kwargs(kwargs)
        if text is not None:
            raise NotImplementedError(
                "This Core ML package contains visual prompts only; text "
                "prompting (including SAM3 PCS) was not exported."
            )
        if masks is not None:
            raise NotImplementedError(
                "LibreSAM Core ML v1 supports point and box prompts, not "
                "mask prompts."
            )
        if batch != 1:
            raise ValueError("LibreSAM Core ML uses fixed batch=1.")
        if stream or show:
            raise NotImplementedError(
                "LibreSAM Core ML image prompting does not support stream/show."
            )
        if max_det < 1:
            raise ValueError("max_det must be at least 1.")
        if multimask is None:
            multimask = False
        elif not isinstance(multimask, bool):
            raise ValueError("multimask must be true or false.")
        if device not in (None, "", "auto", "coreml", self.device):
            logger.warning(
                "LibreSAM Core ML functions are already loaded on device=%s; "
                "predict(device=%s) is ignored.",
                self.device,
                device,
            )
        if imgsz is not None and _imgsz_hw(imgsz) != (
            self._sam_profile.image_size,
            self._sam_profile.image_size,
        ):
            raise ValueError(
                "LibreSAM Core ML uses the package's fixed encoder frame "
                f"{self._sam_profile.image_size}; got imgsz={imgsz!r}."
            )
        if classes is not None:
            try:
                selected_classes = [int(value) for value in classes]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "LibreSAM is class-agnostic; classes must be [0]."
                ) from exc
            if selected_classes != [0]:
                raise ValueError(
                    "LibreSAM is class-agnostic; classes must be [0]."
                )
        if conf is not None:
            try:
                conf_value = float(conf)
            except (TypeError, ValueError) as exc:
                raise ValueError("conf must be a finite number.") from exc
            if not np.isfinite(conf_value):
                raise ValueError("conf must be a finite number.")
        else:
            conf_value = 0.0

        if self._empty_sam_prompt(points):
            points = None
        if self._empty_sam_prompt(bboxes):
            bboxes = None
        from ..models.sam.prompts import (
            normalize_boxes,
            normalize_labels,
            normalize_points,
        )

        normalized_points = normalize_points(points)
        normalized_labels = normalize_labels(labels, normalized_points)
        normalized_boxes = normalize_boxes(bboxes)
        if (
            normalized_points is not None
            and normalized_boxes is not None
            and len(normalized_points) != len(normalized_boxes)
        ):
            raise ValueError(
                f"points has {len(normalized_points)} objects but bboxes has "
                f"{len(normalized_boxes)}; combined prompts pair per object."
            )
        if normalized_points is not None:
            overlong = [
                len(group)
                for group in normalized_points
                if len(group) > self._sam_profile.prompt_max_points
            ]
            if overlong:
                raise ValueError(
                    "A point group exceeds the exported prompt_max_points="
                    f"{self._sam_profile.prompt_max_points}; splitting one "
                    "object's click group would change its semantics."
                )

        if source is None:
            image = self._sam_image
            image_path = self._sam_image_path
            encoding = self._sam_image_encoding
            embeddings = self._sam_image_embeddings
            if image is None or encoding is None or embeddings is None:
                raise RuntimeError(
                    "No image set. Pass source=... or call set_image(...) first."
                )
        else:
            if isinstance(source, (list, tuple)) or (
                isinstance(source, (str, Path)) and Path(source).is_dir()
            ):
                raise NotImplementedError(
                    "LibreSAM Core ML accepts one source image per prompt call."
                )
            image = ImageLoader.load(source, color_format=color_format)
            image_path = source if isinstance(source, (str, Path)) else None
            from .coreml_sam import prepare_sam_coreml_image

            encoding = prepare_sam_coreml_image(
                image,
                profile=self._sam_profile,
            )
            embeddings = None

        if normalized_points is None and normalized_boxes is None:
            if labels is not None:
                raise ValueError("labels were given without points.")
            grid_size = 32 if points_per_side is None else points_per_side
            if embeddings is None:
                embeddings = self._run_sam_encoder(encoding)
            result = self._sam_segment_everything(
                image=image,
                image_path=image_path,
                encoding=encoding,
                embeddings=embeddings,
                points_per_side=grid_size,
                score_threshold=0.88 if conf is None else conf_value,
                max_det=max_det,
            )
            if save:
                self._save_annotated(
                    result,
                    image,
                    image_path if image_path is not None else "image",
                    output_path,
                )
            return result

        from .coreml_sam import (
            postprocess_sam_coreml_masks,
            transform_sam_coreml_box,
            transform_sam_coreml_points,
        )

        query_count = (
            len(normalized_points)
            if normalized_points is not None
            else len(normalized_boxes)
        )
        prepared_queries = []
        for index in range(query_count):
            point_coords = point_labels = box_tensor = None
            if normalized_points is not None:
                point_coords = transform_sam_coreml_points(
                    normalized_points[index],
                    encoding=encoding,
                    profile=self._sam_profile,
                )
                point_labels = torch.tensor(
                    normalized_labels[index],
                    dtype=torch.int32,
                ).reshape(1, 1, -1)
            if normalized_boxes is not None:
                box_tensor = transform_sam_coreml_box(
                    normalized_boxes[index],
                    encoding=encoding,
                    profile=self._sam_profile,
                )
            prepared_queries.append((point_coords, point_labels, box_tensor))

        # Every raw prompt has now been validated and transformed. Only now may
        # an explicit one-shot image cross the expensive encoder boundary.
        if embeddings is None:
            embeddings = self._run_sam_encoder(encoding)

        prompt_mode = (
            "points_boxes"
            if normalized_points is not None and normalized_boxes is not None
            else "points"
            if normalized_points is not None
            else "boxes"
        )
        mask_mode = "multimask" if multimask else "single"
        function_name = f"decode_{prompt_mode}_{mask_mode}"
        all_masks = []
        all_scores = []
        for point_coords, point_labels, box_tensor in prepared_queries:
            low_res, iou_scores = self._run_sam_decoder(
                function_name,
                embeddings,
                point_coords=point_coords,
                point_labels=point_labels,
                boxes=box_tensor,
            )
            all_masks.append(
                postprocess_sam_coreml_masks(
                    low_res,
                    encoding=encoding,
                    profile=self._sam_profile,
                )
            )
            all_scores.append(iou_scores[0, 0].float())
        result = self._sam_results(
            torch.cat(all_masks),
            torch.cat(all_scores),
            image,
            image_path,
            conf_threshold=conf_value,
            max_det=max_det,
        )
        if save:
            self._save_annotated(
                result,
                image,
                image_path if image_path is not None else "image",
                output_path,
            )
        return result

    def _predict_picosam3_component(
        self,
        source=None,
        *,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        text: str | None = None,
        conf: float | None = None,
        multimask: bool | None = None,
        max_det: int = 300,
        device: str | None = None,
        color_format: str = "auto",
        points_per_side: int | None = None,
        imgsz: ImageSize | None = None,
        classes=None,
        iou: float = 0.45,
        save: bool = False,
        output_path: str | None = None,
        batch: int = 1,
        stream: bool = False,
        vid_stride: int = 1,
        show: bool = False,
        **kwargs,
    ) -> Results:
        """Run one fixed-batch Core ML invocation per padded box ROI."""
        del iou, vid_stride
        normalize_predict_kwargs(kwargs)
        unsupported = [
            name
            for name, value in (
                ("points", points),
                ("labels", labels),
                ("masks", masks),
                ("text", text),
                ("points_per_side", points_per_side),
            )
            if value is not None
        ]
        if unsupported:
            raise ValueError(
                "PicoSAM3 supports only bboxes= ROI prompts; unsupported: "
                f"{', '.join(unsupported)}."
            )
        if multimask not in (None, False):
            raise ValueError("PicoSAM3 produces one mask per ROI.")
        if bboxes is None or (hasattr(bboxes, "__len__") and len(bboxes) == 0):
            raise ValueError("PicoSAM3 CoreML inference requires bboxes=.")
        if max_det < 1:
            raise ValueError("max_det must be >= 1.")
        if batch != 1:
            raise ValueError(
                "PicoSAM3 CoreML has a fixed batch-one ROI component; use batch=1."
            )
        if stream or show:
            raise NotImplementedError(
                "PicoSAM3 CoreML ROI inference does not support stream/show."
            )
        if classes is not None:
            try:
                selected_classes = [int(value) for value in classes]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "PicoSAM3 is class-agnostic; classes must be [0]."
                ) from exc
            if selected_classes != [0]:
                raise ValueError(
                    "PicoSAM3 is class-agnostic; classes must be [0]."
                )
        self._resolve_predict_imgsz(imgsz)
        if device not in (None, "", "auto", self.device):
            logger.warning(
                "PicoSAM3 CoreML is already loaded on device=%s; "
                "predict(device=%s) is ignored.",
                self.device,
                device,
            )
        conf_thres = 0.0 if conf is None else float(conf)
        if not 0.0 <= conf_thres <= 1.0:
            raise ValueError("conf must be between 0.0 and 1.0.")

        if source is None:
            image = getattr(self, "_picosam3_image", None)
            image_path = getattr(self, "_picosam3_image_path", None)
            if image is None:
                raise ValueError(
                    "Pass source= or call set_image() before PicoSAM3 predict()."
                )
            image = image.copy()
        else:
            if isinstance(source, (list, tuple)) or (
                isinstance(source, (str, Path)) and Path(source).is_dir()
            ):
                raise NotImplementedError(
                    "PicoSAM3 CoreML box prompts currently accept one source "
                    "image per predict() call."
                )
            image = ImageLoader.load(source, color_format=color_format)
            image_path = source if isinstance(source, (str, Path)) else None

        from ..models.picosam3.preprocess import (
            padded_square_roi,
            place_roi_logits,
        )
        from ..models.sam.prompts import normalize_boxes

        boxes = normalize_boxes(bboxes)
        width, height = image.size
        rois = [padded_square_roi(box, width, height) for box in boxes]
        logits = []
        for roi in rois:
            crop = image.crop(roi).resize(
                _imgsz_hw(self.imgsz)[::-1],
                Image.Resampling.BILINEAR,
            )
            chw = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)
            outputs = self._run_inference(
                np.ascontiguousarray(chw)[None],
            )
            logits.append(torch.from_numpy(np.asarray(outputs[0]).copy()))
        roi_logits = torch.cat(logits, dim=0)
        full_masks, scores = place_roi_logits(
            roi_logits,
            rois,
            height,
            width,
        )

        if conf_thres > 0:
            keep = scores >= conf_thres
            full_masks, scores = full_masks[keep], scores[keep]
        if full_masks.shape[0]:
            nonempty = full_masks.flatten(1).any(dim=1)
            full_masks, scores = full_masks[nonempty], scores[nonempty]
        if full_masks.shape[0] > max_det:
            top = scores.argsort(descending=True)[:max_det]
            full_masks, scores = full_masks[top], scores[top]

        orig_shape = (height, width)
        if full_masks.shape[0]:
            from torchvision.ops import masks_to_boxes

            result_boxes = masks_to_boxes(full_masks.float())
            result = Results(
                boxes=Boxes(
                    result_boxes,
                    scores,
                    torch.zeros(full_masks.shape[0], dtype=torch.float32),
                ),
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=self.names,
                masks=Masks(full_masks, orig_shape),
            )
        else:
            result = Results(
                boxes=Boxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    torch.zeros(0, dtype=torch.float32),
                    torch.zeros(0, dtype=torch.float32),
                ),
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=self.names,
            )

        if save:
            self._save_annotated(
                result,
                image,
                image_path if image_path is not None else "image",
                output_path,
            )
        return result

    @staticmethod
    def _optional_float_metadata(meta: Mapping[str, Any], key: str) -> float | None:
        if key not in meta:
            return None
        try:
            value = float(meta[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CoreML metadata {key!r} must be numeric.") from exc
        if not np.isfinite(value):
            raise ValueError(f"CoreML metadata {key!r} must be finite.")
        return value

    @staticmethod
    def _optional_int_metadata(meta: Mapping[str, Any], key: str) -> int | None:
        if key not in meta:
            return None
        try:
            value = int(meta[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CoreML metadata {key!r} must be an integer.") from exc
        if value <= 0:
            raise ValueError(f"CoreML metadata {key!r} must be positive.")
        return value

    @staticmethod
    def _validate_strict_profile(
        io_contract: _CoreMLIO,
        *,
        family: str,
        task: str,
        size: str | None,
        imgsz: ImageSize,
        has_embedded_nms: bool,
        io_schema_version: str,
        nc: int | None = None,
    ) -> None:
        """Pin untrusted package metadata to LibreYOLO's exact runtime ABI."""
        from ..export.coreml import (
            _expected_dense_candidate_count,
            _input_contract,
            _output_contract,
            _validate_export_profile,
            _validation_contract,
        )

        # This is also the load-side DEIMv2 licensing boundary: a package
        # cannot bypass the size gate merely because conversion happened
        # elsewhere.
        _validate_export_profile(family, task, size)
        if family == "birefnet":
            from ..export.coreml_birefnet import (
                validate_birefnet_coreml_profile,
            )

            validate_birefnet_coreml_profile(
                size=size,
                precision=None,
                canvas_hw=_imgsz_hw(imgsz),
            )
        if family == "depth_anything3":
            from ..export.coreml_depth_anything3 import (
                DEPTH_ANYTHING3_COREML_CANVAS,
            )

            if _imgsz_hw(imgsz) != (
                DEPTH_ANYTHING3_COREML_CANVAS,
                DEPTH_ANYTHING3_COREML_CANVAS,
            ):
                raise NotImplementedError(
                    "Depth Anything 3 CoreML requires its fixed 504x504 canvas; "
                    f"got {_imgsz_hw(imgsz)}."
                )
        if family == "swinir":
            from ..export.coreml_swinir import validate_swinir_coreml_profile

            validate_swinir_coreml_profile(size=size, canvas_hw=_imgsz_hw(imgsz))
        if family == "picosam3":
            from ..export.coreml_picosam3 import validate_picosam3_coreml_profile

            validate_picosam3_coreml_profile(
                size=size,
                canvas_hw=_imgsz_hw(imgsz),
            )
        if family == "owlv2":
            from ..export.coreml_owlv2 import validate_owlv2_coreml_profile

            validate_owlv2_coreml_profile(
                size=size,
                canvas_hw=_imgsz_hw(imgsz),
            )
        if family == "omdet_turbo":
            from ..export.coreml_omdet_turbo import (
                validate_omdet_turbo_coreml_profile,
            )

            validate_omdet_turbo_coreml_profile(
                size=size,
                canvas_hw=_imgsz_hw(imgsz),
            )
        if family == "rtmdet" and task == "segment":
            from ..export.coreml_rtmdet_ins import (
                validate_rtmdet_ins_coreml_profile,
            )

            validate_rtmdet_ins_coreml_profile(
                size=size,
                canvas_hw=_imgsz_hw(imgsz),
            )
        expected = _parse_io_contract(
            {
                _COREML_IO_KEY: {
                    "input": _input_contract(family, task, size),
                    "validation": _validation_contract(family, task),
                    "outputs": _output_contract(
                        family,
                        task,
                        nms=has_embedded_nms,
                    ),
                }
            }
        )

        if io_contract.input != expected.input:
            fields = tuple(_CoreMLInput.__dataclass_fields__)
            mismatches = {
                field: (
                    getattr(expected.input, field),
                    getattr(io_contract.input, field),
                )
                for field in fields
                if getattr(expected.input, field) != getattr(io_contract.input, field)
            }
            raise ValueError(
                f"Strict CoreML input/validation profile for {family}/{task} "
                f"was modified: {mismatches}."
            )

        expected_outputs = [
            (output.name, output.role, output.encoding)
            for output in expected.outputs
        ]
        actual_outputs = [
            (output.name, output.role, output.encoding)
            for output in io_contract.outputs
        ]
        if actual_outputs != expected_outputs:
            raise ValueError(
                f"Strict CoreML output profile for {family}/{task} must be "
                f"{expected_outputs}; got {actual_outputs}."
            )
        missing_tensor_abi = [
            output.name
            for output in io_contract.outputs
            if output.rank is None or output.dtype is None
        ]
        if missing_tensor_abi:
            raise ValueError(
                "Strict CoreML outputs must declare rank and dtype; missing for "
                f"{missing_tensor_abi}."
            )
        if io_schema_version == "2":
            missing_shapes = [
                output.name
                for output in io_contract.outputs
                if output.shape is None
            ]
            if missing_shapes:
                raise ValueError(
                    "Strict CoreML schema v2 outputs must declare exact fixed "
                    f"shapes; missing for {missing_shapes}."
                )
            if family == "rtmdet" and task == "segment":
                from ..export.coreml_rtmdet_ins import (
                    expected_rtmdet_ins_coreml_shapes,
                )

                expected_shapes = expected_rtmdet_ins_coreml_shapes(
                    nc=int(nc or 0),
                    canvas_hw=_imgsz_hw(imgsz),
                )
                actual_shapes = {
                    output.name: output.shape for output in io_contract.outputs
                }
                if actual_shapes != expected_shapes:
                    raise ValueError(
                        "Strict RTMDet-Ins CoreML output shapes disagree with "
                        f"the fixed raw-output ABI: expected {expected_shapes}, "
                        f"got {actual_shapes}."
                    )
            else:
                expected_candidates = _expected_dense_candidate_count(
                    family,
                    _imgsz_hw(imgsz),
                )
            if (
                not (family == "rtmdet" and task == "segment")
                and expected_candidates is not None
            ):
                output_by_name = {
                    output.name: output for output in io_contract.outputs
                }
                class_count = int(nc or 0)
                if has_embedded_nms:
                    confidence = output_by_name["confidence"].shape
                    coordinates = output_by_name["coordinates"].shape
                    expected_confidence = (expected_candidates, class_count)
                    expected_coordinates = (expected_candidates, 4)
                    if (
                        confidence != expected_confidence
                        or coordinates != expected_coordinates
                    ):
                        raise ValueError(
                            "Strict CoreML dense NMS output shapes disagree "
                            f"with {family}'s fixed stride grid: expected "
                            f"{expected_confidence}/{expected_coordinates}, "
                            f"got {confidence}/{coordinates}."
                        )
                else:
                    prediction = output_by_name["prediction"].shape
                    expected_shape = (
                        (1, expected_candidates, 5 + class_count)
                        if family == "yolox"
                        else (1, expected_candidates, 4 + class_count)
                        if family in {"picodet", "rtmdet"}
                        else (1, 4 + class_count, expected_candidates)
                    )
                    if prediction != expected_shape:
                        raise ValueError(
                            "Strict CoreML dense output shape disagrees with "
                            f"{family}'s fixed stride grid: expected "
                            f"{expected_shape}, got {prediction}."
                        )
            if family == "picosam3":
                mask_shape = {
                    output.name: output.shape for output in io_contract.outputs
                }["mask_logits"]
                input_h, input_w = _imgsz_hw(imgsz)
                expected_mask_shape = (1, 1, input_h, input_w)
                if mask_shape != expected_mask_shape:
                    raise ValueError(
                        "Strict PicoSAM3 CoreML mask output must match its "
                        f"fixed ROI canvas: expected {expected_mask_shape}, "
                        f"got {mask_shape}."
                    )
            if family == "birefnet":
                matte_shape = {
                    output.name: output.shape for output in io_contract.outputs
                }["matte"]
                input_h, input_w = _imgsz_hw(imgsz)
                expected_matte_shape = (1, 1, input_h, input_w)
                if matte_shape != expected_matte_shape:
                    raise ValueError(
                        "Strict BiRefNet CoreML matte output must match its "
                        f"fixed canvas: expected {expected_matte_shape}, "
                        f"got {matte_shape}."
                    )
            if family == "depth_anything3":
                from ..export.coreml_depth_anything3 import (
                    expected_depth_anything3_coreml_shapes,
                )

                expected_shapes = expected_depth_anything3_coreml_shapes(
                    batch=1,
                    canvas_hw=_imgsz_hw(imgsz),
                )
                actual_shapes = {
                    output.name: output.shape
                    for output in io_contract.outputs
                }
                if actual_shapes != expected_shapes:
                    raise ValueError(
                        "Strict Depth Anything 3 CoreML output shapes disagree "
                        f"with the raw component ABI: expected "
                        f"{expected_shapes}, got {actual_shapes}."
                    )
            if family == "owlv2":
                from ..export.coreml_owlv2 import (
                    expected_owlv2_coreml_shapes,
                )

                expected_shapes = expected_owlv2_coreml_shapes(
                    size=str(size),
                    nc=int(nc or 0),
                )
                actual_shapes = {
                    output.name: output.shape
                    for output in io_contract.outputs
                }
                if actual_shapes != expected_shapes:
                    raise ValueError(
                        "Strict OWLv2 CoreML output shapes disagree with the "
                        f"frozen detector ABI: expected {expected_shapes}, "
                        f"got {actual_shapes}."
                    )
            if family == "omdet_turbo":
                from ..export.coreml_omdet_turbo import (
                    expected_omdet_turbo_coreml_shapes,
                )

                expected_shapes = expected_omdet_turbo_coreml_shapes(
                    size=str(size),
                    nc=int(nc or 0),
                )
                actual_shapes = {
                    output.name: output.shape
                    for output in io_contract.outputs
                }
                if actual_shapes != expected_shapes:
                    raise ValueError(
                        "Strict OMDet-Turbo CoreML output shapes disagree "
                        f"with the frozen detector ABI: expected "
                        f"{expected_shapes}, got {actual_shapes}."
                    )
        if io_contract.input.shape_mode != "fixed":
            raise ValueError(
                "Current strict CoreML profiles require shape_mode='fixed'."
            )
        if io_contract.parser is not None:
            raise ValueError(
                "Strict CoreML v1 profiles do not accept a metadata-selected "
                "parser."
            )

    @staticmethod
    def _parse_gaze_metadata(
        meta: Mapping[str, Any],
        *,
        task: str,
        strict: bool,
    ) -> dict[str, int | float]:
        if task != "gaze":
            return {}

        required = {"num_bins", "bin_width_deg", "offset_deg", "gaze_input"}
        missing = sorted(key for key in required if meta.get(key) in (None, ""))
        if strict and missing:
            raise ValueError(
                "Strict gaze CoreML artifacts require complete gaze metadata; "
                f"missing {missing}."
            )
        if missing:
            return {}

        try:
            num_bins = int(meta["num_bins"])
            bin_width_deg = float(meta["bin_width_deg"])
            offset_deg = float(meta["offset_deg"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CoreML gaze metadata num_bins/bin_width_deg/offset_deg must "
                "be numeric."
            ) from exc
        if num_bins <= 0:
            raise ValueError("CoreML gaze metadata num_bins must be positive.")
        if not np.isfinite(bin_width_deg) or bin_width_deg <= 0:
            raise ValueError(
                "CoreML gaze metadata bin_width_deg must be finite and positive."
            )
        if not np.isfinite(offset_deg):
            raise ValueError("CoreML gaze metadata offset_deg must be finite.")
        if str(meta["gaze_input"]).strip().lower() != "face_crop":
            raise ValueError(
                "CoreML gaze artifacts require gaze_input='face_crop'; "
                f"got {meta['gaze_input']!r}."
            )
        return {
            "num_bins": num_bins,
            "bin_width_deg": bin_width_deg,
            "offset_deg": offset_deg,
        }

    @staticmethod
    def _parse_classification_activation(
        meta: Mapping[str, Any],
        *,
        task: str,
        strict: bool,
    ) -> str:
        if task != "classify":
            return "softmax"
        value = meta.get("classification_activation")
        if strict and value in (None, ""):
            raise ValueError(
                "Strict classification CoreML artifacts require "
                "classification_activation metadata."
            )
        activation = str(value or "softmax").strip().lower()
        if activation not in {"softmax", "sigmoid"}:
            raise ValueError(
                "CoreML classification_activation must be 'softmax' or "
                f"'sigmoid', got {value!r}."
            )
        return activation

    @staticmethod
    def _validate_yolonas_contract(
        io: _CoreMLIO,
        *,
        task: str,
        imgsz: ImageSize,
    ) -> None:
        from ..export.coreml_yolonas import (
            yolonas_coreml_input_contract,
            yolonas_coreml_output_contract,
        )

        input_h, input_w = _imgsz_hw(imgsz)
        if input_h != input_w:
            raise ValueError(
                "YOLO-NAS CoreML artifacts require a square fixed canvas; "
                f"got {(input_h, input_w)}."
            )

        expected_input = yolonas_coreml_input_contract(task)
        actual_input = {
            "name": io.input.name,
            "kind": io.input.kind,
            "layout": io.input.layout.upper(),
            "color": io.input.color,
            "range": io.input.value_range,
            "geometry": io.input.geometry,
            "interpolation": io.input.interpolation,
            "resize_backend": io.input.resize_backend,
            "resize_long_side": io.input.resize_long_side,
            "resize_rounding": io.input.resize_rounding,
            "pad_value": io.input.pad_value,
        }
        if actual_input != expected_input:
            raise ValueError(
                "YOLO-NAS CoreML input contract does not match the native "
                f"{task} profile: expected {expected_input}, got {actual_input}."
            )

        expected_outputs = [
            {
                "name": item["name"],
                "role": item["role"],
                "encoding": item.get("encoding"),
                "rank": item.get("rank"),
            }
            for item in yolonas_coreml_output_contract(task)
        ]
        actual_outputs = [
            {
                "name": output.name,
                "role": output.role,
                "encoding": output.encoding,
                "rank": output.rank,
            }
            for output in io.outputs
        ]
        if actual_outputs != expected_outputs:
            raise ValueError(
                "YOLO-NAS CoreML output contract does not match the native "
                f"{task} profile: expected {expected_outputs}, got {actual_outputs}."
            )

    def _validate_spec_contract(
        self,
        spec: Any,
        spec_output_names: list[str],
    ) -> None:
        description = getattr(spec, "description", None)
        inputs = getattr(description, "input", None)
        schema_v2 = self._io_schema_version == "2"
        input_names = _feature_names(inputs)
        if input_names != [self.input_contract.name]:
            raise ValueError(
                "CoreML package input does not match its IO contract: "
                f"spec={input_names}, contract={[self.input_contract.name]}."
            )

        if len(set(spec_output_names)) != len(spec_output_names):
            raise ValueError(
                f"CoreML package contains duplicate output names: {spec_output_names}."
            )
        if set(spec_output_names) != set(self.output_names):
            raise ValueError(
                "CoreML package outputs do not match its IO contract: "
                f"spec={spec_output_names}, contract={self.output_names}."
            )

        try:
            input_feature = list(inputs)[0]
        except (TypeError, IndexError):
            input_feature = None
        feature_kind = (
            _feature_kind(input_feature) if input_feature is not None else None
        )
        if schema_v2 and input_feature is not None and bool(
            getattr(getattr(input_feature, "type", None), "isOptional", False)
        ):
            raise ValueError("CoreML schema v2 input must not be optional.")
        if (
            schema_v2
            and feature_kind != self.input_contract.kind
            or not schema_v2
            and feature_kind is not None
            and feature_kind != self.input_contract.kind
        ):
            raise ValueError(
                "CoreML input type does not match its IO contract: "
                f"spec={feature_kind!r}, contract={self.input_contract.kind!r}."
            )

        input_h, input_w = _imgsz_hw(self.imgsz)
        feature_type = getattr(input_feature, "type", None)
        if feature_kind == "image":
            image_type = getattr(feature_type, "imageType", None)
            width = int(getattr(image_type, "width", 0) or 0)
            height = int(getattr(image_type, "height", 0) or 0)
            if schema_v2:
                size_flexibility = getattr(image_type, "WhichOneof", None)
                if not callable(size_flexibility):
                    raise ValueError(
                        "CoreML schema v2 ImageType must expose fixed-size metadata."
                    )
                if size_flexibility("SizeFlexibility") is not None:
                    raise ValueError(
                        "CoreML schema v2 ImageType must not declare flexible sizes."
                    )
                if int(getattr(image_type, "colorSpace", 0) or 0) != (
                    _COREML_RGB_COLORSPACE
                ):
                    raise ValueError(
                        "CoreML schema v2 ImageType must declare RGB color space."
                    )
                if width <= 0 or height <= 0:
                    raise ValueError(
                        "CoreML schema v2 ImageType must declare nonzero dimensions."
                    )
            if width and height and (height, width) != (input_h, input_w):
                raise ValueError(
                    "CoreML ImageType dimensions disagree with metadata: "
                    f"spec={(height, width)}, metadata={(input_h, input_w)}."
                )
        elif feature_kind == "tensor" and self.input_contract.shape_mode == "fixed":
            array_type = getattr(feature_type, "multiArrayType", None)
            shape = tuple(int(dim) for dim in getattr(array_type, "shape", ()))
            expected = (
                (1, 3, input_h, input_w)
                if self.input_contract.layout == "nchw"
                else (1, input_h, input_w, 3)
            )
            if schema_v2:
                shape_flexibility = getattr(array_type, "WhichOneof", None)
                if not callable(shape_flexibility):
                    raise ValueError(
                        "CoreML schema v2 TensorType must expose shape metadata."
                    )
                if shape_flexibility("ShapeFlexibility") is not None:
                    raise ValueError(
                        "CoreML schema v2 TensorType must not declare flexible shapes."
                    )
                if not shape:
                    raise ValueError(
                        "CoreML schema v2 TensorType must declare a fixed shape."
                    )
                if int(getattr(array_type, "dataType", 0) or 0) != (
                    _COREML_ARRAY_DTYPES["float32"]
                ):
                    raise ValueError(
                        "CoreML schema v2 TensorType input must be FLOAT32."
                    )
            if shape and shape != expected:
                raise ValueError(
                    "CoreML TensorType shape disagrees with metadata: "
                    f"spec={shape}, expected={expected}."
                )

        spec_outputs = {
            str(feature.name): feature
            for feature in (
                getattr(getattr(spec, "description", None), "output", ()) or ()
            )
        }
        for declared in self.io_contract.outputs:
            feature = spec_outputs[declared.name]
            feature_type = getattr(feature, "type", None)
            if schema_v2:
                if bool(getattr(feature_type, "isOptional", False)):
                    raise ValueError(
                        f"CoreML schema v2 output {declared.name!r} must not "
                        "be optional."
                    )
                if _feature_kind(feature) != "tensor":
                    raise ValueError(
                        f"CoreML schema v2 output {declared.name!r} must be "
                        "a multi-array tensor."
                    )
            array_type = getattr(feature_type, "multiArrayType", None)
            spec_shape = tuple(int(dim) for dim in getattr(array_type, "shape", ()))
            if schema_v2:
                shape_flexibility = getattr(array_type, "WhichOneof", None)
                if not callable(shape_flexibility):
                    raise ValueError(
                        f"CoreML schema v2 output {declared.name!r} must expose "
                        "shape metadata."
                    )
                if shape_flexibility("ShapeFlexibility") is not None:
                    raise ValueError(
                        f"CoreML schema v2 output {declared.name!r} must not "
                        "declare a flexible shape."
                    )
                if not spec_shape:
                    raise ValueError(
                        f"CoreML schema v2 output {declared.name!r} must declare "
                        "a fixed shape."
                    )
                expected_dtype = _COREML_ARRAY_DTYPES.get(str(declared.dtype))
                actual_dtype = int(getattr(array_type, "dataType", 0) or 0)
                if expected_dtype is None or actual_dtype != expected_dtype:
                    raise ValueError(
                        f"CoreML output {declared.name!r} dtype disagrees with "
                        f"metadata: spec={actual_dtype}, "
                        f"contract={declared.dtype!r}."
                    )
            if declared.shape is not None and spec_shape != declared.shape:
                raise ValueError(
                    f"CoreML output {declared.name!r} shape disagrees with "
                    f"metadata: spec={spec_shape}, contract={declared.shape}."
                )

    @staticmethod
    def _parse_metadata(
        meta: dict,
        default_nb_classes: int,
        *,
        output_names: list[str] | None = None,
        parse_pose_metadata: bool = True,
    ):
        model_family: Optional[str] = (
            str(meta.get("model_family")).strip().lower()
            if meta.get("model_family")
            else None
        )
        model_size: Optional[str] = meta.get("model_size") or meta.get("size") or None
        default_task = normalize_task(meta.get("default_task"), default="detect")
        metadata_task = normalize_task(meta.get("task"), default=default_task)
        supported_tasks = _normalize_metadata_supported_tasks(
            meta.get("supported_tasks", (metadata_task,))
        )
        names: Optional[dict] = None
        imgsz = 640
        has_embedded_nms = False
        # Strict artifacts are parsed only by _validate_strict_pose_contract
        # after their exact IO schema has been loaded.  Sending them through
        # the permissive legacy parser first would allow coercion or raise an
        # unrelated TypeError before the strict validator can diagnose them.
        pose_metadata = _read_pose_metadata(meta) if parse_pose_metadata else {}

        if "names" in meta:
            try:
                raw = _metadata_json(meta["names"], key="names")
                if not isinstance(raw, dict):
                    raise ValueError("names must be a JSON object")
                names = {int(k): v for k, v in raw.items()}
            except (ValueError, TypeError) as e:
                logger.warning("Failed to parse names metadata: %s", e)

        if names is None and (meta.get("nb_classes") or meta.get("nc")):
            try:
                nc = int(meta.get("nb_classes", meta.get("nc")))
                names = (
                    {i: n for i, n in enumerate(COCO_CLASSES)}
                    if nc == 80
                    else {i: f"class_{i}" for i in range(nc)}
                )
            except ValueError:
                pass

        metadata_imgsz = _read_metadata_imgsz(
            meta,
            model_family,
            artifact="CoreML metadata",
        )
        if metadata_imgsz is not None:
            imgsz = metadata_imgsz

        has_embedded_nms = _metadata_bool(meta.get("nms", False), key="nms")

        return (
            model_family,
            model_size,
            metadata_task,
            supported_tasks,
            default_task,
            names,
            imgsz,
            has_embedded_nms,
            pose_metadata,
        )

    def _parse_outputs(
        self,
        all_outputs: list,
        effective_imgsz: ImageSize,
        original_size: tuple,
        conf: float,
        ratio: float | None = None,
        iou: float = 0.45,
        max_det: int = 300,
        **kwargs,
    ):
        if ratio is None:
            ratio = 1.0
        if self.model_family == "omdet_turbo":
            from ..export.coreml_omdet_turbo import (
                postprocess_omdet_turbo_coreml_outputs,
            )

            by_name = dict(zip(self.output_names, all_outputs))
            decoded = postprocess_omdet_turbo_coreml_outputs(
                by_name["pred_logits"],
                by_name["pred_boxes"],
                original_size=original_size,
                conf=conf,
                iou=iou,
                max_det=max_det,
                classes=kwargs.get("classes"),
            )
            return (
                decoded["boxes"].numpy(),
                decoded["scores"].numpy(),
                decoded["classes"].numpy(),
                None,
            )
        if self.model_family == "owlv2":
            from ..export.coreml_owlv2 import (
                postprocess_owlv2_coreml_outputs,
            )

            by_name = dict(zip(self.output_names, all_outputs))
            decoded = postprocess_owlv2_coreml_outputs(
                by_name["pred_logits"],
                by_name["pred_boxes"],
                original_size=original_size,
                conf=conf,
                max_det=max_det,
            )
            return (
                decoded["boxes"].numpy(),
                decoded["scores"].numpy(),
                decoded["classes"].numpy(),
                None,
            )
        if self.model_family == "eomt":
            if self.task != "segment":
                raise ValueError(
                    "EoMT semantic and panoptic outputs use their dedicated "
                    "dense result parsers."
                )
            reconstructed = self._reconstruct_eomt_outputs(all_outputs)
            detections = self._eomt_decoder._postprocess_segment(
                reconstructed,
                conf,
                iou,
                original_size,
                max_det=max_det,
            )
            return (
                np.asarray(detections["boxes"], dtype=np.float32).reshape(-1, 4),
                np.asarray(detections["scores"], dtype=np.float32),
                np.asarray(detections["classes"], dtype=np.int64),
                detections["masks"],
            )
        if self._has_embedded_nms:
            if self.task != "detect":
                raise ValueError(
                    "CoreML embedded NMS is only valid for detection artifacts."
                )
            if self._nms_iou is not None and not np.isclose(
                iou, self._nms_iou, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    "This CoreML artifact has NMS IoU baked into the graph "
                    f"({self._nms_iou}); runtime iou={iou} cannot be applied."
                )
            if self._nms_conf is not None and conf < self._nms_conf:
                raise ValueError(
                    "This CoreML artifact discarded detections below its baked "
                    f"confidence {self._nms_conf}; runtime conf={conf} cannot "
                    "recover them."
                )
            if self._nms_max_det is not None and max_det > self._nms_max_det:
                raise ValueError(
                    "This CoreML artifact emits at most "
                    f"{self._nms_max_det} detections; runtime max_det={max_det} "
                    "cannot increase that limit."
                )
            return self._parse_embedded_nms(
                all_outputs, effective_imgsz, original_size, conf, ratio=ratio
            )
        if self.model_family in {"rtdetr", "rtdetrv2"}:
            # Strict CoreML artifacts define the semantic ABI explicitly:
            # class logits first, normalized cxcywh boxes second. Shape
            # guessing is ambiguous when nc == 4, because both tensors then
            # end in four values.
            role_to_output = dict(zip(self.output_roles, all_outputs))
            try:
                ordered = [
                    role_to_output["class_logits"],
                    role_to_output["boxes"],
                ]
            except KeyError as exc:  # pragma: no cover - strict profile guards this
                raise RuntimeError(
                    "CoreML RT-DETR outputs must declare class_logits and "
                    "boxes roles."
                ) from exc
            orig_w, orig_h = original_size
            boxes, scores, classes = self._parse_dfine(
                ordered,
                orig_w,
                orig_h,
                conf,
                max_det=max_det,
            )
            return boxes, scores, classes, None
        if self.model_family == "yolo1" and self.input_contract.geometry == "stretch":
            input_h, input_w = _imgsz_hw(effective_imgsz)
            parsed = super()._parse_outputs(
                all_outputs,
                effective_imgsz,
                (input_w, input_h),
                conf,
                ratio=1.0,
                iou=iou,
                max_det=max_det,
                **kwargs,
            )
            boxes, scores, classes, masks, obb, keypoints = self._unpack_parsed_outputs(
                parsed
            )
            orig_w, orig_h = original_size
            boxes = np.asarray(boxes, dtype=np.float32).copy()
            boxes[:, [0, 2]] *= orig_w / input_w
            boxes[:, [1, 3]] *= orig_h / input_h
            if masks is not None or obb is not None or keypoints is not None:
                raise NotImplementedError(
                    "YOLO1 CoreML stretch parsing only supports detection."
                )
            return boxes, scores, classes, None
        return super()._parse_outputs(
            all_outputs,
            effective_imgsz,
            original_size,
            conf,
            ratio=ratio,
            iou=iou,
            max_det=max_det,
            **kwargs,
        )

    def _reconstruct_eomt_outputs(self, all_outputs: list) -> dict[str, torch.Tensor]:
        """Rebuild full-canvas EoMT outputs from the compact Core ML ABI."""
        if self.model_family != "eomt":
            raise RuntimeError("EoMT output reconstruction used for another family.")
        by_name = {
            name: torch.from_numpy(
                np.ascontiguousarray(np.asarray(value, dtype=np.float32)).copy()
            )
            for name, value in zip(self.output_names, all_outputs)
        }
        try:
            class_logits = by_name["class_queries_logits"]
            mask_logits = by_name["masks_queries_logits"]
        except KeyError as exc:  # pragma: no cover - strict profile guards this
            raise RuntimeError(
                "EoMT CoreML outputs must contain class_queries_logits and "
                "masks_queries_logits."
            ) from exc
        from ..export.coreml_eomt import reconstruct_eomt_full_outputs

        return reconstruct_eomt_full_outputs(
            class_logits,
            mask_logits,
            nc=self.nb_classes,
            canvas_hw=_imgsz_hw(self.imgsz),
        )

    def _depth_anything3_inverse_output(self, all_outputs: list) -> np.ndarray:
        """Apply DA3's exact stochastic sky/inverse host contract once."""
        from ..export.coreml_depth_anything3 import (
            postprocess_depth_anything3_coreml,
        )

        by_name = dict(zip(self.output_names, all_outputs))
        try:
            relative_depth = by_name["relative_depth"]
            sky_score = by_name["sky_score"]
        except KeyError as exc:  # pragma: no cover - strict schema guards this
            raise RuntimeError(
                "Depth Anything 3 CoreML outputs must contain named "
                "relative_depth and sky_score tensors."
            ) from exc
        inverse = postprocess_depth_anything3_coreml(
            torch.from_numpy(
                np.ascontiguousarray(
                    np.asarray(relative_depth, dtype=np.float32)
                ).copy()
            ),
            torch.from_numpy(
                np.ascontiguousarray(
                    np.asarray(sky_score, dtype=np.float32)
                ).copy()
            ),
        )
        return inverse.detach().cpu().numpy()

    def _parse_depth_output(
        self,
        all_outputs,
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        if self.model_family != "depth_anything3":
            return super()._parse_depth_output(all_outputs, original_size)
        inverse = self._depth_anything3_inverse_output(all_outputs)
        return super()._parse_depth_output([inverse], original_size)

    def _parse_semantic_output(
        self,
        all_outputs,
        original_size: tuple[int, int],
        effective_imgsz: ImageSize,
        ratio: float,
    ) -> torch.Tensor:
        if self.model_family != "eomt":
            return super()._parse_semantic_output(
                all_outputs,
                original_size,
                effective_imgsz,
                ratio,
            )
        reconstructed = self._reconstruct_eomt_outputs(all_outputs)
        logits = self._eomt_decoder._postprocess_semantic_logits(
            reconstructed,
            original_size,
        )
        return logits.argmax(dim=1)[0].cpu()

    def _parse_panoptic_output(
        self,
        all_outputs,
        original_size: tuple[int, int],
        effective_imgsz: ImageSize,
        conf: float,
        iou: float,
        max_det: int,
        ratio: float,
    ) -> dict[str, Any]:
        if self.model_family != "eomt":
            return super()._parse_panoptic_output(
                all_outputs,
                original_size,
                effective_imgsz,
                conf,
                iou,
                max_det,
                ratio,
            )
        del effective_imgsz, iou, max_det, ratio
        reconstructed = self._reconstruct_eomt_outputs(all_outputs)
        return self._eomt_decoder._postprocess_panoptic(
            reconstructed,
            conf,
            original_size,
        )

    def _build_result(self, *args, iou: float, **kwargs):
        # Apple's NMS already ran inside the .mlpackage when embedded NMS is on;
        # neutralize BaseBackend's numpy NMS by using a threshold that never matches.
        if self._has_embedded_nms:
            iou = 1.0
        return super()._build_result(*args, iou=iou, **kwargs)

    def _preprocess(self, image, effective_imgsz: ImageSize, color_format):
        """Produce canonical RGB pixels matching the exported graph's input.

        The .mlpackage was traced with a wrapper that converts canonical
        RGB[0,1] to whatever the family expects internally (YOLOX BGR/0-255,
        RF-DETR ImageNet-normalized, etc.). Most profiles cross a uint8
        ImageType boundary; RF-DETR pose preserves antialiased fractional
        pixels through a float TensorType boundary.
        """
        if self.model_family == "omdet_turbo":
            from ..export.coreml_omdet_turbo import (
                preprocess_omdet_turbo_coreml_image,
            )

            img = ImageLoader.load(image, color_format=color_format)
            input_h, input_w = _imgsz_hw(effective_imgsz)
            if input_h != input_w:
                raise ValueError(
                    "OMDet-Turbo Core ML input canvas must be square."
                )
            tensor = preprocess_omdet_turbo_coreml_image(
                img,
                image_size=input_h,
            )
            return tensor, img.copy(), img.size, 1.0

        if self.model_family == "owlv2":
            from ..export.coreml_owlv2 import (
                preprocess_owlv2_coreml_image,
            )

            img = ImageLoader.load(image, color_format=color_format)
            input_h, input_w = _imgsz_hw(effective_imgsz)
            if input_h != input_w:
                raise ValueError("OWLv2 Core ML input canvas must be square.")
            tensor = preprocess_owlv2_coreml_image(
                img,
                image_size=input_h,
            )
            # The common TensorType encoder applies the declared 0..1 range.
            return tensor.mul(255.0), img.copy(), img.size, 1.0

        if self.model_family == "eomt":
            tensor, original_img, original_size, ratio = (
                self._eomt_decoder._preprocess(
                    image,
                    color_format,
                    input_size=int(_imgsz_hw(effective_imgsz)[0]),
                )
            )
            # CoreMLBackend's common TensorType encoder accepts canonical
            # pixel-domain values and applies the declared 0..1 conversion.
            return tensor.mul(255.0), original_img, original_size, ratio

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()
        input_h, input_w = _imgsz_hw(effective_imgsz)
        if (
            self.model_family == "rfdetr"
            and self.task == "pose"
            and self.input_contract.resize_backend == "torchvision"
        ):
            import torch.nn.functional as F

            rgb = np.ascontiguousarray(np.asarray(img, dtype=np.float32))
            tensor = (
                torch.from_numpy(rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .div(255.0)
            )
            resized = F.interpolate(
                tensor,
                size=(input_h, input_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            # The common runtime encoder accepts canonical pixel-domain values
            # and applies the TensorType range conversion below.
            return resized.mul(255.0), original_img, original_size, 1.0
        transformed = _apply_geometry(
            img,
            input_h=input_h,
            input_w=input_w,
            contract=self.input_contract,
        )
        chw = np.asarray(transformed.image, dtype=np.float32).transpose(2, 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
        return tensor, original_img, original_size, transformed.ratio

    def _get_val_preprocessor(self, img_size: ImageSize | None = None):
        """Use the same canonical canvas for detection prediction and validation."""
        if img_size is None:
            img_size = self._get_input_size()
        if (
            self.task in {"detect", "segment", "pose", "obb"}
            and self.model_family != "picosam3"
        ):
            return _CoreMLValPreprocessor(
                _imgsz_hw(img_size),
                self.input_contract,
            )
        if self.task == "point" and self.model_family == "fomo":
            from ..validation.preprocessors import FOMOValPreprocessor

            return FOMOValPreprocessor(img_size=_imgsz_hw(img_size))
        return super()._get_val_preprocessor(img_size)

    def _resolve_predict_imgsz(self, imgsz: ImageSize | None = None) -> ImageSize:
        """Reject shape overrides that the loaded artifact did not declare."""
        if self.model_family == "ppocr":
            if imgsz is None:
                return self.imgsz
            if isinstance(imgsz, (tuple, list)):
                if len(imgsz) != 2 or int(imgsz[0]) != int(imgsz[1]):
                    raise ValueError(
                        "LibrePPOCR uses one detector long-side limit; imgsz "
                        "must be an int or equal square pair."
                    )
                requested_limit = int(imgsz[0])
            else:
                requested_limit = int(imgsz)
            if not (
                32
                <= requested_limit
                <= self._ppocr_profile.det_limit_side_len
            ):
                raise ValueError(
                    "LibrePPOCR Core ML imgsz must be within [32, "
                    f"{self._ppocr_profile.det_limit_side_len}], got "
                    f"{requested_limit}."
                )
            return requested_limit
        if imgsz is None:
            return self.imgsz
        requested = _imgsz_hw(imgsz)
        exported = _imgsz_hw(self.imgsz)
        if self.input_contract.shape_mode == "fixed" and requested != exported:
            raise ValueError(
                "CoreML artifact has a fixed input canvas "
                f"{exported[1]}x{exported[0]}; runtime imgsz "
                f"{requested[1]}x{requested[0]} is incompatible. Re-export at "
                "the requested size."
            )
        if requested != exported:
            raise NotImplementedError(
                "Runtime CoreML flexible-shape selection is not implemented. "
                "Use the artifact's default imgsz."
            )
        return self.imgsz

    def _supports_rectangular_validation(self) -> bool:
        """Declare exact rectangular detection geometry for this artifact."""
        if (
            self.input_contract.shape_mode == "fixed"
            and self.task == "restore"
            and self.input_contract.geometry == "native"
        ):
            return True
        return (
            self.input_contract.shape_mode == "fixed"
            and self.task in {"detect", "segment", "obb"}
            and self.input_contract.geometry
            in {"stretch", "letterbox_top_left", "letterbox_center"}
        )

    def _parse_embedded_nms(
        self,
        all_outputs: list,
        effective_imgsz: ImageSize,
        original_size: tuple,
        conf: float,
        ratio: float = 1.0,
    ):
        output_by_name = {
            name: np.asarray(value)
            for name, value in zip(self.output_names, all_outputs)
        }
        confidence = output_by_name.get("confidence")
        coordinates = output_by_name.get("coordinates")
        if confidence is None or coordinates is None:
            raise RuntimeError(
                "CoreML embedded NMS output must include confidence and coordinates"
            )

        if confidence.ndim == 3:
            confidence = confidence[0]
        if coordinates.ndim == 3:
            coordinates = coordinates[0]

        max_scores = np.max(confidence, axis=1)
        class_ids = np.argmax(confidence, axis=1)
        mask = max_scores > conf
        boxes_raw = coordinates[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_raw) == 0:
            return np.empty((0, 4)), max_scores, class_ids, None

        orig_w, orig_h = original_size
        cx, cy, w, h = (
            boxes_raw[:, 0],
            boxes_raw[:, 1],
            boxes_raw[:, 2],
            boxes_raw[:, 3],
        )
        boxes = np.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            axis=1,
        )

        geometry = self.input_contract.geometry
        input_h, input_w = _imgsz_hw(effective_imgsz)
        if geometry in {"letterbox_top_left", "letterbox_center"}:
            ratio = min(input_h / orig_h, input_w / orig_w)
            offset_x = offset_y = 0
            if geometry == "letterbox_center":
                new_w = max(1, int(round(orig_w * ratio)))
                new_h = max(1, int(round(orig_h * ratio)))
                offset_x = (input_w - new_w) // 2
                offset_y = (input_h - new_h) // 2
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - offset_x) / ratio
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - offset_y) / ratio
        elif geometry == "stretch":
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        else:
            raise NotImplementedError(
                "CoreML embedded NMS coordinate inversion is not defined for "
                f"geometry={geometry!r}."
            )

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        return boxes, max_scores, class_ids, None

    def _canonicalize_validation_tensor(
        self,
        input_tensor: torch.Tensor,
    ) -> np.ndarray:
        """Invert the declared validator pixel domain to canonical RGB bytes."""
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.as_tensor(input_tensor)
        if input_tensor.ndim != 4 or input_tensor.shape[1] != 3:
            raise ValueError(
                "CoreML validation expects a [B, 3, H, W] tensor, got "
                f"{tuple(input_tensor.shape)}."
            )
        canonical = input_tensor.detach().cpu().float().numpy()
        if not np.isfinite(canonical).all():
            raise ValueError("CoreML validation input contains NaN or infinity.")

        validation = self.input_contract.validation
        if validation.value_range in {"imagenet", "standardized"}:
            mean = np.asarray(validation.mean, dtype=np.float32).reshape(1, 3, 1, 1)
            std = np.asarray(validation.std, dtype=np.float32).reshape(1, 3, 1, 1)
            canonical = (canonical * std + mean) * 255.0
        elif validation.value_range == "0_1":
            canonical = canonical * 255.0
        elif validation.value_range == "minus_1_1":
            canonical = (canonical + 1.0) * 127.5
        elif validation.value_range != "0_255":
            raise AssertionError(
                f"Unhandled validation range {validation.value_range!r}."
            )

        if validation.color == "bgr":
            canonical = canonical[:, ::-1, :, :]
        canonical = np.clip(canonical, 0.0, 255.0)
        return self._fit_validation_canvas(canonical)

    def _fit_validation_canvas(self, canonical: np.ndarray) -> np.ndarray:
        expected_h, expected_w = _imgsz_hw(self.imgsz)
        actual_h, actual_w = canonical.shape[-2:]
        if (actual_h, actual_w) == (expected_h, expected_w):
            return np.ascontiguousarray(canonical)

        if self.input_contract.geometry == "pad_bottom_right":
            if actual_h > expected_h or actual_w > expected_w:
                raise ValueError(
                    "CoreML validation input exceeds the fixed exported canvas: "
                    f"got {(actual_h, actual_w)}, maximum "
                    f"{(expected_h, expected_w)}."
                )
            pad_h = expected_h - actual_h
            pad_w = expected_w - actual_w
            mode = (
                "reflect"
                if actual_h > 1
                and actual_w > 1
                and pad_h < actual_h
                and pad_w < actual_w
                else "edge"
            )
            return np.ascontiguousarray(
                np.pad(
                    canonical,
                    ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
                    mode=mode,
                )
            )

        if (
            self.input_contract.geometry == "native"
            and self.input_contract.shape_mode != "fixed"
        ):
            return np.ascontiguousarray(canonical)

        raise ValueError(
            "CoreML validation tensor shape does not match the exported canvas: "
            f"got {(actual_h, actual_w)}, expected {(expected_h, expected_w)}. "
            "Validation must use the artifact's declared geometry."
        )

    def _forward(self, input_tensor: torch.Tensor):
        """Run validator batches through the declared canonical pixel boundary."""
        if self._has_embedded_nms:
            raise NotImplementedError(
                "CoreML validation requires a raw-output artifact; embedded NMS "
                "cannot preserve runtime thresholds and task associations."
            )
        canonical = self._canonicalize_validation_tensor(input_tensor)
        per_image = []
        for index in range(canonical.shape[0]):
            outputs = self._run_inference(canonical[index : index + 1])
            if self.model_family == "depth_anything3":
                outputs = [self._depth_anything3_inverse_output(outputs)]
            per_image.append(outputs)
        if not per_image:
            raise ValueError("CoreML validation received an empty batch.")

        combined = []
        output_count = len(per_image[0])
        if output_count <= 0 or any(
            len(outputs) != output_count for outputs in per_image
        ):
            raise RuntimeError(
                "CoreML validation produced an inconsistent output count."
            )
        for output_index in range(output_count):
            arrays = [np.asarray(outputs[output_index]) for outputs in per_image]
            if all(array.ndim > 0 and array.shape[0] == 1 for array in arrays):
                value = np.concatenate(arrays, axis=0)
            else:
                value = np.stack(arrays, axis=0)
            combined.append(torch.from_numpy(np.ascontiguousarray(value).copy()))
        return combined

    def _encode_runtime_input(self, blob: np.ndarray) -> Image.Image | np.ndarray:
        if blob.ndim != 4 or blob.shape[0] != 1 or blob.shape[1] != 3:
            raise ValueError(
                "CoreMLBackend expects canonical input shape (1, 3, H, W), "
                f"got {tuple(blob.shape)}."
            )
        if not np.isfinite(blob).all():
            raise ValueError("CoreML input contains NaN or infinity.")

        if self.input_contract.kind == "image":
            hwc = np.transpose(blob[0], (1, 2, 0))
            uint8 = np.rint(np.clip(hwc, 0.0, 255.0)).astype(np.uint8)
            return Image.fromarray(np.ascontiguousarray(uint8), mode="RGB")

        tensor = np.asarray(blob, dtype=np.float32)
        if self.input_contract.color == "bgr":
            tensor = tensor[:, ::-1, :, :]
        if self.input_contract.value_range == "0_1":
            tensor = tensor / 255.0
        elif self.input_contract.value_range == "minus_1_1":
            tensor = tensor / 127.5 - 1.0
        elif self.input_contract.value_range in {"0_255", "uint8"}:
            pass
        elif self.input_contract.value_range in {"imagenet", "standardized"}:
            mean = np.asarray(self.input_contract.mean, dtype=np.float32).reshape(
                1, 3, 1, 1
            )
            std = np.asarray(self.input_contract.std, dtype=np.float32).reshape(
                1, 3, 1, 1
            )
            tensor = (tensor / 255.0 - mean) / std
        else:  # pragma: no cover - schema validation makes this unreachable
            raise AssertionError(
                f"Unhandled CoreML input range {self.input_contract.value_range!r}."
            )
        if self.input_contract.layout == "nhwc":
            tensor = np.transpose(tensor, (0, 2, 3, 1))
        return np.ascontiguousarray(tensor, dtype=np.float32)

    def _run_inference(self, blob: np.ndarray) -> list:
        """Run CoreML with contract-ordered outputs and a named input."""
        if (
            self.model_family == "eomt"
            and isinstance(blob, np.ndarray)
            and blob.ndim == 4
            and blob.shape[0] > 1
        ):
            per_patch = [
                self._run_inference(blob[index : index + 1])
                for index in range(blob.shape[0])
            ]
            return [
                np.concatenate(
                    [np.asarray(outputs[output_index]) for outputs in per_patch],
                    axis=0,
                )
                for output_index in range(len(self.output_names))
            ]
        expected_h, expected_w = _imgsz_hw(self.imgsz)
        if self.input_contract.shape_mode == "fixed" and tuple(blob.shape[-2:]) != (
            expected_h,
            expected_w,
        ):
            raise ValueError(
                "CoreML fixed input shape mismatch: "
                f"got {tuple(blob.shape[-2:])}, expected {(expected_h, expected_w)}."
            )
        runtime_input = self._encode_runtime_input(blob)
        output = self.model.predict({self.input_contract.name: runtime_input})
        if not isinstance(output, Mapping):
            raise RuntimeError(
                "CoreML runtime returned a non-mapping output; named outputs "
                "are required by the LibreYOLO contract."
            )
        actual = set(output)
        expected = set(self.output_names)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise RuntimeError(
                "CoreML runtime output names do not match the artifact contract: "
                f"missing={missing}, unexpected={unexpected}."
            )
        values = []
        for declared in self.io_contract.outputs:
            value = np.asarray(output[declared.name])
            if not np.isfinite(value).all():
                raise RuntimeError(
                    f"CoreML output {declared.name!r} contains NaN or infinity."
                )
            if declared.rank is not None and value.ndim != declared.rank:
                raise RuntimeError(
                    f"CoreML output {declared.name!r} has rank {value.ndim}, "
                    f"but the artifact contract declares {declared.rank}."
                )
            if declared.dtype is not None and value.dtype.name != declared.dtype:
                raise RuntimeError(
                    f"CoreML output {declared.name!r} has dtype "
                    f"{value.dtype.name!r}, but the artifact contract declares "
                    f"{declared.dtype!r}."
                )
            if declared.shape is not None and tuple(value.shape) != declared.shape:
                raise RuntimeError(
                    f"CoreML output {declared.name!r} has shape "
                    f"{tuple(value.shape)}, but the artifact contract declares "
                    f"{declared.shape}."
                )
            values.append(value)
        if self.task == "gaze":
            if len(values) != 2 or any(
                value.ndim < 1 or value.shape[-1] != self.num_bins for value in values
            ):
                shapes = [tuple(value.shape) for value in values]
                raise RuntimeError(
                    "CoreML gaze outputs must contain yaw and pitch logits "
                    f"with width num_bins={self.num_bins}; got {shapes}."
                )
        return values
