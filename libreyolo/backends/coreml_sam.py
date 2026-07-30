"""Exact host operations for split LibreSAM Core ML packages.

The package boundary is intentionally model-ready: raw-image geometry,
prompt-coordinate transforms, query orchestration, and mask upscaling stay on
the host.  Keeping those operations here avoids sending SAM through the
generic detector preprocessing path, whose resize and padding semantics are
not equivalent.

Provenance
----------
The SAM-1, SAM2, and SAM3 pixel contracts are derived from Hugging Face
Transformers v5.3.0 (commit aad13b87ed59f2afcfaebc985f403301887a35fc,
Apache-2.0). EdgeTAM follows facebookresearch/EdgeTAM commit
7711e012a30a2402c4eaab637bdb00a521302c91 (Apache-2.0). MobileSAM composes
LibreYOLO's existing Apache-2.0-attributed preprocessing helpers. See
``THIRD_PARTY_NOTICES.txt`` and the model-family NOTICE files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as torch_functional

from ..export.coreml_sam import (
    SAM_COREML_BOXES_INPUT,
    SAM_COREML_POINT_COORDS_INPUT,
    SAM_COREML_POINT_LABELS_INPUT,
    SAMCoreMLProfile,
    sam_coreml_runtime_function_contract,
    validate_sam_coreml_function_io,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_MOBILE_MEAN = (123.675, 116.28, 103.53)
_MOBILE_STD = (58.395, 57.12, 57.375)
_SAM3_MEAN = (0.5, 0.5, 0.5)
_SAM3_STD = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class SAMCoreMLImageEncoding:
    """One model-ready image plus the geometry needed for prompts and masks."""

    pixel_values: torch.Tensor
    original_size: tuple[int, int]
    reshaped_input_size: tuple[int, int] | None


def _longest_side_shape(
    height: int,
    width: int,
    target: int,
) -> tuple[int, int]:
    scale = float(target) / max(height, width)
    return int(height * scale + 0.5), int(width * scale + 0.5)


def _preprocess_edgetam(
    image: Image.Image,
    *,
    size: int,
) -> torch.Tensor:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as vision_functional

    pixels = vision_functional.to_tensor(image)
    pixels = vision_functional.resize(
        pixels,
        [size, size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return vision_functional.normalize(
        pixels,
        mean=list(_IMAGENET_MEAN),
        std=list(_IMAGENET_STD),
    ).unsqueeze(0)


def _preprocess_mobile(
    image: Image.Image,
    *,
    size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    from ..models.mobilesam.preprocess import (
        encode_image_and_prompts,
        preprocess_tensor,
    )

    encoded = encode_image_and_prompts(image, target_length=size)
    reshaped = tuple(
        int(value) for value in encoded["reshaped_input_sizes"][0]
    )
    pixels = preprocess_tensor(
        encoded["pixel_values"],
        image_size=size,
        pixel_mean=torch.tensor(_MOBILE_MEAN, dtype=torch.float32).reshape(
            1,
            3,
            1,
            1,
        ),
        pixel_std=torch.tensor(_MOBILE_STD, dtype=torch.float32).reshape(
            1,
            3,
            1,
            1,
        ),
    )
    return pixels, reshaped


def _preprocess_sam1(
    image: Image.Image,
    *,
    size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = image.height, image.width
    reshaped = _longest_side_shape(height, width, size)
    resized = image.resize(
        (reshaped[1], reshaped[0]),
        resample=Image.Resampling.BILINEAR,
    )
    # Transformers v5.3 performs rescaling in float64 and explicitly casts to
    # float32 before float32 mean/std normalization.
    values = (
        np.asarray(resized, dtype=np.uint8).astype(np.float64) * (1.0 / 255.0)
    ).astype(np.float32)
    values = (
        values - np.asarray(_IMAGENET_MEAN, dtype=np.float32)
    ) / np.asarray(_IMAGENET_STD, dtype=np.float32)
    pixels = torch.from_numpy(
        np.ascontiguousarray(values.transpose(2, 0, 1))
    ).unsqueeze(0)
    pad_h = size - reshaped[0]
    pad_w = size - reshaped[1]
    pixels = torch_functional.pad(pixels, (0, pad_w, 0, pad_h))
    return pixels, reshaped


def _preprocess_fast_square(
    image: Image.Image,
    *,
    size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as vision_functional

    # Resize while still uint8. Moving the float conversion earlier changes
    # quantization and is observably different on non-square images.
    pixels = vision_functional.pil_to_tensor(image)
    pixels = vision_functional.resize(
        pixels,
        [size, size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    mean_raw = torch.tensor(mean, dtype=torch.float32) * 255.0
    std_raw = torch.tensor(std, dtype=torch.float32) * 255.0
    pixels = vision_functional.normalize(
        pixels.float(),
        mean=mean_raw.tolist(),
        std=std_raw.tolist(),
    )
    return pixels.unsqueeze(0)


def prepare_sam_coreml_image(
    image: Image.Image,
    *,
    profile: SAMCoreMLProfile,
) -> SAMCoreMLImageEncoding:
    """Apply the immutable model-ready pixel contract for one SAM family."""
    image = image.convert("RGB")
    original_size = (image.height, image.width)
    size = profile.image_size
    reshaped: tuple[int, int] | None = None
    if profile.family == "edgetam":
        pixels = _preprocess_edgetam(image, size=size)
    elif profile.family == "mobilesam":
        pixels, reshaped = _preprocess_mobile(image, size=size)
    elif profile.family == "sam":
        pixels, reshaped = _preprocess_sam1(image, size=size)
    elif profile.family == "sam2":
        pixels = _preprocess_fast_square(
            image,
            size=size,
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
        )
    elif profile.family == "sam3":
        pixels = _preprocess_fast_square(
            image,
            size=size,
            mean=_SAM3_MEAN,
            std=_SAM3_STD,
        )
    else:  # Profile construction already rejects this; keep the boundary safe.
        raise ValueError(f"Unsupported LibreSAM Core ML family {profile.family!r}.")
    pixels = pixels.to(dtype=torch.float32).contiguous()
    expected = (1, 3, size, size)
    if tuple(pixels.shape) != expected or not bool(torch.isfinite(pixels).all()):
        raise RuntimeError(
            "LibreSAM preprocessing violated its encoder ABI: "
            f"expected {expected}, got {tuple(pixels.shape)}."
        )
    return SAMCoreMLImageEncoding(
        pixel_values=pixels,
        original_size=original_size,
        reshaped_input_size=reshaped,
    )


def _coordinate_array(
    values: Any,
    *,
    width: int,
    height: int,
    boxes: bool,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        kind = "boxes" if boxes else "points"
        raise ValueError(f"{kind} must contain numeric coordinates.") from exc
    expected_last = 4 if boxes else 2
    if array.ndim < 1 or array.shape[-1] != expected_last:
        kind = "boxes" if boxes else "points"
        raise ValueError(
            f"{kind} must end in {expected_last} coordinates, got "
            f"shape={array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Prompt coordinates must be finite.")
    coordinate_pairs = array.reshape(-1, 2, 2) if boxes else array
    if (
        (coordinate_pairs[..., 0] < 0).any()
        or (coordinate_pairs[..., 0] > width).any()
        or (coordinate_pairs[..., 1] < 0).any()
        or (coordinate_pairs[..., 1] > height).any()
    ):
        raise ValueError(
            "Prompt coordinates must lie within the source image bounds."
        )
    if boxes and (
        (array[..., 2] < array[..., 0]).any()
        or (array[..., 3] < array[..., 1]).any()
    ):
        raise ValueError("boxes must use ordered x1,y1,x2,y2 coordinates.")
    return array


def transform_sam_coreml_points(
    points: Any,
    *,
    encoding: SAMCoreMLImageEncoding,
    profile: SAMCoreMLProfile,
) -> torch.Tensor:
    """Transform one object's source-pixel points to the decoder canvas."""
    height, width = encoding.original_size
    array = _coordinate_array(
        points,
        width=width,
        height=height,
        boxes=False,
    )
    size = profile.image_size
    if profile.family in {"mobilesam", "sam"}:
        from ..models.mobilesam.preprocess import ResizeLongestSide

        if profile.family == "sam":
            # SAM-1 point scaling enters the v5.3 processor as float64.
            working = np.asarray(points, dtype=np.float64)
        else:
            working = array
        transformed = ResizeLongestSide(size).apply_coords(
            working,
            (height, width),
        )
        result = torch.as_tensor(transformed, dtype=torch.float32)
    else:
        result = torch.as_tensor(array, dtype=torch.float32).clone()
        if profile.family == "edgetam":
            result[..., 0] /= width
            result[..., 1] /= height
            result *= size
        else:
            result[..., 0] *= float(size) / width
            result[..., 1] *= float(size) / height
    result = result.reshape(1, 1, -1, 2).contiguous()
    if (
        bool((result < 0).any())
        or bool((result > size).any())
        or not bool(torch.isfinite(result).all())
    ):
        raise ValueError("Transformed point coordinates violate the decoder canvas.")
    return result


def transform_sam_coreml_box(
    box: Any,
    *,
    encoding: SAMCoreMLImageEncoding,
    profile: SAMCoreMLProfile,
) -> torch.Tensor:
    """Transform one source-pixel xyxy box to the decoder canvas."""
    height, width = encoding.original_size
    array = _coordinate_array(
        box,
        width=width,
        height=height,
        boxes=True,
    ).reshape(-1, 4)
    if len(array) != 1:
        raise ValueError("One Core ML decoder query accepts exactly one box.")
    size = profile.image_size
    if profile.family in {"mobilesam", "sam"}:
        from ..models.mobilesam.preprocess import ResizeLongestSide

        transformed = ResizeLongestSide(size).apply_boxes(
            array.astype(np.float32, copy=False),
            (height, width),
        )
        result = torch.as_tensor(transformed, dtype=torch.float32)
    else:
        result = torch.as_tensor(array, dtype=torch.float32).clone()
        pairs = result.reshape(-1, 2, 2)
        if profile.family == "edgetam":
            pairs[..., 0] /= width
            pairs[..., 1] /= height
            pairs *= size
        else:
            pairs[..., 0] *= float(size) / width
            pairs[..., 1] *= float(size) / height
        result = pairs.reshape(-1, 4)
    result = result.reshape(1, 1, 4).contiguous()
    if (
        bool((result < 0).any())
        or bool((result > size).any())
        or not bool(torch.isfinite(result).all())
    ):
        raise ValueError("Transformed box coordinates violate the decoder canvas.")
    return result


def postprocess_sam_coreml_masks(
    low_res_masks: torch.Tensor,
    *,
    encoding: SAMCoreMLImageEncoding,
    profile: SAMCoreMLProfile,
) -> torch.Tensor:
    """Upscale one query's raw logits and apply the strict ``> 0`` threshold."""
    if low_res_masks.dtype != torch.float32 or low_res_masks.ndim != 5:
        raise ValueError(
            "LibreSAM low-resolution masks must be FP32 [1,1,M,H,W]."
        )
    logits = low_res_masks[0, 0]
    original_size = encoding.original_size
    if profile.family in {"mobilesam", "sam"}:
        reshaped = encoding.reshaped_input_size
        if reshaped is None:
            raise RuntimeError(
                "Longest-side SAM mask postprocessing needs the resized shape."
            )
        logits = torch_functional.interpolate(
            logits.unsqueeze(0),
            (profile.image_size, profile.image_size),
            mode="bilinear",
            align_corners=False,
        )
        logits = logits[..., : reshaped[0], : reshaped[1]]
        logits = torch_functional.interpolate(
            logits,
            original_size,
            mode="bilinear",
            align_corners=False,
        )[0]
    else:
        logits = torch_functional.interpolate(
            logits.unsqueeze(0),
            original_size,
            mode="bilinear",
            align_corners=False,
        )[0]
    return logits > 0.0


def _validate_runtime_inputs(
    function_name: str,
    inputs: Mapping[str, torch.Tensor],
    *,
    profile: SAMCoreMLProfile,
    contract: Mapping[str, Any] | None = None,
) -> None:
    if contract is None:
        contract = sam_coreml_runtime_function_contract(
            profile,
            function_name,
        )
    expected_names = [item["name"] for item in contract["inputs"]]
    if list(inputs) != expected_names:
        raise ValueError(
            f"LibreSAM Core ML inputs must be {expected_names!r}, got "
            f"{list(inputs)!r}."
        )
    dtype_by_name = {"float32": torch.float32, "int32": torch.int32}
    for value, feature in zip(inputs.values(), contract["inputs"]):
        if not torch.is_tensor(value):
            raise TypeError(f"{feature['name']} must be a torch.Tensor.")
        if value.dtype != dtype_by_name[feature["dtype"]]:
            raise ValueError(
                f"{feature['name']} must use {feature['dtype']}, got "
                f"{value.dtype}."
            )
        if value.ndim != len(feature["shape"]):
            raise ValueError(f"{feature['name']} has the wrong rank.")
        for axis, descriptor in enumerate(feature["shape"]):
            actual = int(value.shape[axis])
            if descriptor["kind"] == "fixed":
                valid = actual == int(descriptor["value"])
            else:
                valid = (
                    int(descriptor["lower_bound"])
                    <= actual
                    <= int(descriptor["upper_bound"])
                )
            if not valid:
                raise ValueError(
                    f"{feature['name']} axis {axis} violates its Core ML "
                    f"shape contract."
                )
        if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{feature['name']} contains NaN or infinity.")
    if SAM_COREML_POINT_LABELS_INPUT in inputs:
        labels = inputs[SAM_COREML_POINT_LABELS_INPUT]
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("point_labels may contain only 0 or 1.")
    if SAM_COREML_POINT_COORDS_INPUT in inputs:
        coords = inputs[SAM_COREML_POINT_COORDS_INPUT]
        if bool((coords < 0).any()) or bool((coords > profile.image_size).any()):
            raise ValueError("point_coords lie outside the model canvas.")
    if SAM_COREML_BOXES_INPUT in inputs:
        boxes = inputs[SAM_COREML_BOXES_INPUT]
        if (
            bool((boxes < 0).any())
            or bool((boxes > profile.image_size).any())
            or not bool(
                (boxes[..., 2] >= boxes[..., 0]).all()
                and (boxes[..., 3] >= boxes[..., 1]).all()
            )
        ):
            raise ValueError("boxes violate the model-canvas xyxy contract.")


class SAMCoreMLFunction:
    """Strict named facade over one function in a SAM Core ML package."""

    def __init__(
        self,
        runtime: Any,
        *,
        function_name: str,
        profile: SAMCoreMLProfile,
    ) -> None:
        self.runtime = runtime
        self.function_name = function_name
        self.profile = profile
        self.contract = sam_coreml_runtime_function_contract(
            profile,
            function_name,
        )
        self.input_names = tuple(
            item["name"] for item in self.contract["inputs"]
        )
        self.output_names = tuple(
            item["name"] for item in self.contract["outputs"]
        )

    def __call__(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _validate_runtime_inputs(
            self.function_name,
            inputs,
            profile=self.profile,
            contract=self.contract,
        )
        arrays = {}
        for name, tensor in inputs.items():
            dtype = np.int32 if tensor.dtype == torch.int32 else np.float32
            arrays[name] = np.ascontiguousarray(
                tensor.detach().cpu().numpy(),
                dtype=dtype,
            )
        raw_outputs = self.runtime.predict(arrays)
        if not isinstance(raw_outputs, Mapping):
            raise RuntimeError(
                f"Core ML function {self.function_name!r} returned a "
                "non-mapping output."
            )
        if set(raw_outputs) != set(self.output_names):
            raise RuntimeError(
                f"Core ML function {self.function_name!r} output names changed: "
                f"expected {list(self.output_names)!r}, got "
                f"{sorted(raw_outputs)!r}."
            )
        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            value = np.asarray(raw_outputs[name])
            if value.dtype != np.float32:
                raise RuntimeError(
                    f"Core ML function {self.function_name!r} output {name!r} "
                    f"must be float32, got {value.dtype.name!r}."
                )
            outputs[name] = torch.from_numpy(
                np.ascontiguousarray(value).copy()
            )
        validate_sam_coreml_function_io(
            self.function_name,
            inputs,
            outputs,
            profile=self.profile,
            _contract=self.contract,
        )
        return outputs


__all__ = [
    "SAMCoreMLFunction",
    "SAMCoreMLImageEncoding",
    "postprocess_sam_coreml_masks",
    "prepare_sam_coreml_image",
    "transform_sam_coreml_box",
    "transform_sam_coreml_points",
]
