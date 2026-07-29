"""Depth Anything 3's raw two-output Core ML component contract.

The native LibreYOLO network performs data-dependent sky replacement in
``forward``.  That operation contains Python control flow, boolean compaction,
a quantile, and (for large masks) random sampling.  None of those operations
belongs in the converted graph.  The Core ML component therefore ends at the
two deterministic DPT-head outputs:

``relative_depth``
    Positive relative depth after the head's exponential activation.

``sky_score``
    Non-negative sky score after the head's ReLU activation.

The host then applies :func:`postprocess_depth_anything3_coreml`, which matches
``LibreDepthAnything3Net._apply_mono_sky`` followed by LibreYOLO's reciprocal
depth transform.  In particular, the host contract deliberately preserves the
native random sampling-with-replacement step above 100,000 non-sky pixels.

This module is derived only from LibreYOLO's Apache-2.0-attributed
``libreyolo.models.depth_anything3`` implementation.  It does not import
``coremltools``; the shared exporter owns capture, conversion, metadata, and
package saving.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from ..models.depth_anything3.nn import IMAGENET_MEAN, IMAGENET_STD

DEPTH_ANYTHING3_COREML_CONTRACT = "depth_anything3_raw_depth_sky_v1"
DEPTH_ANYTHING3_COREML_HOST_POSTPROCESS = "mono_sky_quantile_inverse_v1"
DEPTH_ANYTHING3_COREML_POSITION_EMBEDDING = "fixed_eager_bicubic_v1"
DEPTH_ANYTHING3_COREML_CANVAS = 504
DEPTH_ANYTHING3_COREML_PATCH_SIZE = 14
DEPTH_ANYTHING3_COREML_SUPPORTED_SIZES = frozenset({"l"})

DEPTH_ANYTHING3_SKY_THRESHOLD = 0.3
DEPTH_ANYTHING3_MIN_REGION_PIXELS = 10
DEPTH_ANYTHING3_SKY_SAMPLE_LIMIT = 100_000
DEPTH_ANYTHING3_FAR_QUANTILE = 0.99
DEPTH_ANYTHING3_INVERSE_DEPTH_EPS = 1e-6


def _normalize_hw(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        height = width = value
    elif isinstance(value, Sequence) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
    else:
        raise ValueError(
            "Depth Anything 3 Core ML canvas must be an int or "
            f"(height, width), got {value!r}."
        )
    if height <= 0 or width <= 0:
        raise ValueError(
            "Depth Anything 3 Core ML canvas dimensions must be positive, "
            f"got {(height, width)}."
        )
    return height, width


def _unwrap_depth_anything3_net(model: nn.Module) -> nn.Module:
    """Find ``LibreDepthAnything3Net`` inside generic export wrappers."""

    current: Any = model
    visited: set[int] = set()
    for _ in range(12):
        if all(
            hasattr(current, attribute)
            for attribute in ("backbone", "head", "pixel_mean", "pixel_std")
        ):
            return current
        marker = id(current)
        if marker in visited:
            break
        visited.add(marker)
        nested = getattr(current, "model", None)
        if not isinstance(nested, nn.Module):
            break
        current = nested
    raise TypeError(
        "Depth Anything 3 Core ML export could not find "
        "LibreDepthAnything3Net inside the prepared "
        f"{type(model).__name__} graph."
    )


def _validate_normalization_buffers(net: nn.Module) -> None:
    mean = getattr(net, "pixel_mean", None)
    std = getattr(net, "pixel_std", None)
    if not torch.is_tensor(mean) or not torch.is_tensor(std):
        raise RuntimeError(
            "Depth Anything 3 Core ML export requires pixel_mean and "
            "pixel_std tensors."
        )
    expected_mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=mean.dtype,
        device=mean.device,
    ).view(1, 3, 1, 1)
    expected_std = torch.tensor(
        IMAGENET_STD,
        dtype=std.dtype,
        device=std.device,
    ).view(1, 3, 1, 1)
    if tuple(mean.shape) != (1, 3, 1, 1) or not torch.equal(mean, expected_mean):
        raise RuntimeError(
            "Depth Anything 3 Core ML export requires the canonical ImageNet "
            "pixel_mean buffer."
        )
    if tuple(std.shape) != (1, 3, 1, 1) or not torch.equal(std, expected_std):
        raise RuntimeError(
            "Depth Anything 3 Core ML export requires the canonical ImageNet "
            "pixel_std buffer."
        )


def validate_depth_anything3_coreml_profile(
    model: nn.Module,
    *,
    size: str | None,
    canvas_hw: int | Sequence[int],
) -> nn.Module:
    """Validate the fixed DA3MONO-LARGE component profile.

    The supported checkpoint has fixed learned DINOv2 position embeddings and
    a patch size of 14.  LibreYOLO's public tier uses 504 as the native
    upper-bound; this first Core ML contract intentionally fixes both axes to
    504 so trace shape and runtime geometry cannot drift.
    """

    normalized_size = str(size or "").strip().lower()
    if normalized_size not in DEPTH_ANYTHING3_COREML_SUPPORTED_SIZES:
        raise NotImplementedError(
            "Depth Anything 3 Core ML export supports only the permissively "
            "licensed DA3MONO-LARGE size='l'; "
            f"got size={size!r}."
        )
    height, width = _normalize_hw(canvas_hw)
    expected = (DEPTH_ANYTHING3_COREML_CANVAS, DEPTH_ANYTHING3_COREML_CANVAS)
    if (height, width) != expected:
        raise NotImplementedError(
            "Depth Anything 3 Core ML export currently requires the fixed "
            f"{expected[0]}x{expected[1]} canvas; got {height}x{width}."
        )

    net = _unwrap_depth_anything3_net(model)
    _validate_normalization_buffers(net)
    patch_size = int(getattr(net, "PATCH_SIZE", 0) or 0)
    if patch_size != DEPTH_ANYTHING3_COREML_PATCH_SIZE:
        raise RuntimeError(
            "Depth Anything 3 Core ML requires patch_size=14; "
            f"got {patch_size}."
        )

    head = net.head
    invariants = {
        "use_sky_head": True,
        "out_dim": 1,
        "activation": "exp",
        "sky_activation": "relu",
        "down_ratio": 1,
    }
    mismatches = {
        name: getattr(head, name, None)
        for name, expected_value in invariants.items()
        if getattr(head, name, None) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "Depth Anything 3 Core ML requires the canonical mono DPT head "
            f"configuration; mismatches={mismatches}."
        )
    return net


def freeze_depth_anything3_coreml_position_embedding(
    model: nn.Module,
    *,
    canvas_hw: int | Sequence[int] = DEPTH_ANYTHING3_COREML_CANVAS,
) -> nn.Module:
    """Eagerly bake DINOv2's position table for the fixed Core ML canvas.

    Core ML Tools 9 has no PyTorch frontend converter for
    ``upsample_bicubic2d``.  DA3MONO-LARGE's learned table is 37x37 (the
    pretraining 518 canvas), while LibreYOLO exports 36x36 patches at 504.
    The original encoder therefore performs one deterministic bicubic resize
    on every forward.  Baking that exact evaluated tensor as ``pos_embed``
    makes the encoder's own shape-equality fast path return it directly.

    This operation intentionally mutates the *prepared export graph*.  Callers
    must pass an export-only copy, never the user's live model.  It is
    idempotent for the same canvas and fail-closed for a conflicting rewrite.
    """

    height, width = _normalize_hw(canvas_hw)
    if height != width or (height, width) != (
        DEPTH_ANYTHING3_COREML_CANVAS,
        DEPTH_ANYTHING3_COREML_CANVAS,
    ):
        raise NotImplementedError(
            "Depth Anything 3 Core ML position baking requires the fixed "
            f"{DEPTH_ANYTHING3_COREML_CANVAS}x"
            f"{DEPTH_ANYTHING3_COREML_CANVAS} canvas; "
            f"got {height}x{width}."
        )
    net = _unwrap_depth_anything3_net(model)
    encoder = getattr(getattr(net, "backbone", None), "pretrained", None)
    position = getattr(encoder, "pos_embed", None)
    interpolate = getattr(encoder, "interpolate_pos_encoding", None)
    patch_size = int(getattr(encoder, "patch_size", 0) or 0)
    if (
        encoder is None
        or not isinstance(position, nn.Parameter)
        or not callable(interpolate)
        or patch_size != DEPTH_ANYTHING3_COREML_PATCH_SIZE
    ):
        raise RuntimeError(
            "Depth Anything 3 Core ML position baking requires the canonical "
            "DINOv2 encoder with a learnable pos_embed table and patch_size=14."
        )

    requested = (height, width)
    previous = getattr(encoder, "_libreyolo_coreml_position_hw", None)
    if previous is not None:
        if tuple(previous) != requested:
            raise RuntimeError(
                "Depth Anything 3 position embeddings were already baked for "
                f"{tuple(previous)}, not {requested}."
            )
        return net

    patch_height = height // patch_size
    patch_width = width // patch_size
    embedding_dim = int(position.shape[-1])
    probe = torch.empty(
        1,
        1 + patch_height * patch_width,
        embedding_dim,
        dtype=position.dtype,
        device=position.device,
    )
    with torch.no_grad():
        fixed = interpolate(probe, height, width).detach().clone()
    expected_shape = (1, 1 + patch_height * patch_width, embedding_dim)
    if tuple(fixed.shape) != expected_shape or not bool(torch.isfinite(fixed).all()):
        raise RuntimeError(
            "Depth Anything 3 fixed position embedding has an invalid result: "
            f"expected {expected_shape}, got {tuple(fixed.shape)}."
        )
    encoder.pos_embed = nn.Parameter(
        fixed,
        requires_grad=position.requires_grad,
    )
    encoder._libreyolo_coreml_position_hw = requested
    return net


class DepthAnything3CoreMLAdapter(nn.Module):
    """Expose deterministic DPT depth and sky tensors before host reduction."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = _unwrap_depth_anything3_net(model)

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image.shape[-2:]
        normalized = (image - self.model.pixel_mean) / self.model.pixel_std
        features, _ = self.model.backbone(
            normalized.unsqueeze(1),
            export_feat_layers=[],
        )
        output = self.model.head(
            features,
            height,
            width,
            patch_start_idx=0,
        )
        return output["depth"].float(), output["sky"].float()


def wrap_depth_anything3_coreml_contract(
    model: nn.Module,
    *,
    freeze_position_embedding: bool = True,
) -> nn.Module:
    """Return the deterministic raw-output graph used for Core ML capture.

    By default this applies the in-place export-only position-table rewrite
    documented by :func:`freeze_depth_anything3_coreml_position_embedding`.
    Unit fakes without the DINOv2 encoder may opt out explicitly.
    """

    if freeze_position_embedding:
        freeze_depth_anything3_coreml_position_embedding(model)
    return DepthAnything3CoreMLAdapter(model).eval()


def validate_depth_anything3_coreml_raw_outputs(
    relative_depth: torch.Tensor,
    sky_score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(relative_depth) or not torch.is_tensor(sky_score):
        raise TypeError(
            "Depth Anything 3 Core ML postprocessing requires tensor outputs."
        )
    if relative_depth.ndim != 4 or relative_depth.shape[1] != 1:
        raise ValueError(
            "Depth Anything 3 relative_depth must have shape [B, 1, H, W], "
            f"got {tuple(relative_depth.shape)}."
        )
    if tuple(sky_score.shape) != tuple(relative_depth.shape):
        raise ValueError(
            "Depth Anything 3 sky_score must have the same [B, 1, H, W] "
            "shape as relative_depth; "
            f"got depth={tuple(relative_depth.shape)}, "
            f"sky={tuple(sky_score.shape)}."
        )
    depth = relative_depth.float()
    sky = sky_score.float()
    if not bool(torch.isfinite(depth).all()):
        raise ValueError(
            "Depth Anything 3 relative_depth contains NaN or infinity."
        )
    if not bool(torch.isfinite(sky).all()):
        raise ValueError("Depth Anything 3 sky_score contains NaN or infinity.")
    if bool((depth < 0).any()):
        raise ValueError(
            "Depth Anything 3 relative_depth must be non-negative after its "
            "exponential activation."
        )
    if bool((sky < 0).any()):
        raise ValueError(
            "Depth Anything 3 sky_score must be non-negative after its ReLU "
            "activation."
        )
    return depth, sky


def postprocess_depth_anything3_coreml(
    relative_depth: torch.Tensor,
    sky_score: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply the exact native mono-sky rule and inverse-depth transform.

    The result has shape ``[B, 1, H, W]`` and follows LibreYOLO's relative
    inverse-depth convention (larger is closer).  Sampling above
    :data:`DEPTH_ANYTHING3_SKY_SAMPLE_LIMIT` is with replacement, matching
    ``torch.randint`` in the native implementation.  Supplying ``generator``
    is useful for deterministic tests; production's default ``None`` consumes
    the active PyTorch RNG exactly like native inference.
    """

    depth, sky = validate_depth_anything3_coreml_raw_outputs(
        relative_depth,
        sky_score,
    )
    sky_adjusted = depth
    for batch_index in range(depth.shape[0]):
        non_sky_mask = sky[batch_index] < DEPTH_ANYTHING3_SKY_THRESHOLD
        non_sky_count = int(non_sky_mask.sum().item())
        sky_count = int((~non_sky_mask).sum().item())
        if (
            non_sky_count <= DEPTH_ANYTHING3_MIN_REGION_PIXELS
            or sky_count <= DEPTH_ANYTHING3_MIN_REGION_PIXELS
        ):
            continue

        non_sky_depth = depth[batch_index][non_sky_mask]
        if non_sky_depth.numel() > DEPTH_ANYTHING3_SKY_SAMPLE_LIMIT:
            indices = torch.randint(
                0,
                non_sky_depth.numel(),
                (DEPTH_ANYTHING3_SKY_SAMPLE_LIMIT,),
                device=non_sky_depth.device,
                generator=generator,
            )
            non_sky_depth = non_sky_depth[indices]
        far_depth = torch.quantile(
            non_sky_depth,
            DEPTH_ANYTHING3_FAR_QUANTILE,
        )
        if sky_adjusted is depth:
            sky_adjusted = depth.clone()
        sky_adjusted[batch_index] = torch.where(
            non_sky_mask,
            depth[batch_index],
            far_depth,
        )

    return torch.reciprocal(
        sky_adjusted.clamp_min(DEPTH_ANYTHING3_INVERSE_DEPTH_EPS)
    )


def depth_anything3_coreml_input_contract() -> dict[str, object]:
    """Describe the fixed exported-runtime image geometry.

    As specified by ADR 0006, exported depth backends stretch the source to the
    fixed graph canvas and resize the depth map back to the original canvas.
    This intentionally differs from native DA3 keep-aspect preprocessing on
    non-square images and must remain documented as an approximation.
    """

    return {
        "name": "image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "stretch",
        "interpolation": "bilinear",
        "resize_backend": "opencv",
        "pad_value": 0,
    }


def depth_anything3_coreml_validation_contract() -> dict[str, str]:
    """Describe dense-validator tensors before canonical RGB-byte inversion."""

    return {"color": "rgb", "range": "0_1"}


def depth_anything3_coreml_output_contract() -> list[dict[str, object]]:
    """Return the strict deterministic two-tensor graph ABI."""

    return [
        {
            "name": "relative_depth",
            "role": "relative_depth",
            "encoding": "positive_depth_after_exp",
            "rank": 4,
        },
        {
            "name": "sky_score",
            "role": "sky_score",
            "encoding": "nonnegative_relu_score",
            "rank": 4,
        },
    ]


def expected_depth_anything3_coreml_shapes(
    *,
    batch: int,
    canvas_hw: int | Sequence[int],
) -> dict[str, tuple[int, ...]]:
    """Return exact fixed-shape graph outputs for schema enrichment."""

    height, width = _normalize_hw(canvas_hw)
    shape = (int(batch), 1, height, width)
    return {"relative_depth": shape, "sky_score": shape}


def depth_anything3_coreml_metadata() -> dict[str, object]:
    """Return host-orchestration metadata required by a strict loader."""

    return {
        "depth_anything3_contract": DEPTH_ANYTHING3_COREML_CONTRACT,
        "depth_anything3_host_postprocess": (
            DEPTH_ANYTHING3_COREML_HOST_POSTPROCESS
        ),
        "depth_anything3_sky_threshold": DEPTH_ANYTHING3_SKY_THRESHOLD,
        "depth_anything3_min_region_pixels": (
            DEPTH_ANYTHING3_MIN_REGION_PIXELS
        ),
        "depth_anything3_sky_sample_limit": (
            DEPTH_ANYTHING3_SKY_SAMPLE_LIMIT
        ),
        "depth_anything3_far_quantile": DEPTH_ANYTHING3_FAR_QUANTILE,
        "depth_anything3_inverse_depth_eps": (
            DEPTH_ANYTHING3_INVERSE_DEPTH_EPS
        ),
        "depth_anything3_sky_sampling": "random_with_replacement",
        "depth_anything3_non_square_geometry": (
            "fixed_stretch_approximation"
        ),
        "depth_anything3_position_embedding": (
            DEPTH_ANYTHING3_COREML_POSITION_EMBEDDING
        ),
    }


def validate_depth_anything3_coreml_metadata(
    metadata: Mapping[str, Any],
) -> None:
    """Validate every field that changes DA3 host-side interpretation."""

    expected_strings = {
        "depth_anything3_contract": DEPTH_ANYTHING3_COREML_CONTRACT,
        "depth_anything3_host_postprocess": (
            DEPTH_ANYTHING3_COREML_HOST_POSTPROCESS
        ),
        "depth_anything3_sky_sampling": "random_with_replacement",
        "depth_anything3_non_square_geometry": (
            "fixed_stretch_approximation"
        ),
        "depth_anything3_position_embedding": (
            DEPTH_ANYTHING3_COREML_POSITION_EMBEDDING
        ),
    }
    for key, expected in expected_strings.items():
        actual = str(metadata.get(key, "")).strip()
        if actual != expected:
            raise ValueError(
                f"Depth Anything 3 Core ML metadata {key!r} must equal "
                f"{expected!r}, got {actual!r}."
            )

    expected_integers = {
        "depth_anything3_min_region_pixels": (
            DEPTH_ANYTHING3_MIN_REGION_PIXELS
        ),
        "depth_anything3_sky_sample_limit": (
            DEPTH_ANYTHING3_SKY_SAMPLE_LIMIT
        ),
    }
    for key, expected in expected_integers.items():
        value = metadata.get(key)
        if isinstance(value, bool):
            raise ValueError(
                f"Depth Anything 3 Core ML metadata {key!r} must be an integer."
            )
        try:
            actual = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Depth Anything 3 Core ML metadata {key!r} must be an integer."
            ) from exc
        if str(value).strip() not in {str(expected), f"+{expected}"}:
            raise ValueError(
                f"Depth Anything 3 Core ML metadata {key!r} must equal "
                f"{expected}, got {value!r}."
            )
        if actual != expected:  # pragma: no cover - string gate is stronger
            raise AssertionError("unreachable integer metadata mismatch")

    expected_floats = {
        "depth_anything3_sky_threshold": DEPTH_ANYTHING3_SKY_THRESHOLD,
        "depth_anything3_far_quantile": DEPTH_ANYTHING3_FAR_QUANTILE,
        "depth_anything3_inverse_depth_eps": (
            DEPTH_ANYTHING3_INVERSE_DEPTH_EPS
        ),
    }
    for key, expected in expected_floats.items():
        try:
            actual = float(metadata.get(key))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Depth Anything 3 Core ML metadata {key!r} must be numeric."
            ) from exc
        if not math.isfinite(actual) or actual != expected:
            raise ValueError(
                f"Depth Anything 3 Core ML metadata {key!r} must equal "
                f"{expected}, got {metadata.get(key)!r}."
            )


__all__ = [
    "DEPTH_ANYTHING3_COREML_CANVAS",
    "DEPTH_ANYTHING3_COREML_CONTRACT",
    "DEPTH_ANYTHING3_COREML_HOST_POSTPROCESS",
    "DEPTH_ANYTHING3_COREML_PATCH_SIZE",
    "DEPTH_ANYTHING3_COREML_POSITION_EMBEDDING",
    "DEPTH_ANYTHING3_COREML_SUPPORTED_SIZES",
    "DEPTH_ANYTHING3_FAR_QUANTILE",
    "DEPTH_ANYTHING3_INVERSE_DEPTH_EPS",
    "DEPTH_ANYTHING3_MIN_REGION_PIXELS",
    "DEPTH_ANYTHING3_SKY_SAMPLE_LIMIT",
    "DEPTH_ANYTHING3_SKY_THRESHOLD",
    "DepthAnything3CoreMLAdapter",
    "depth_anything3_coreml_input_contract",
    "depth_anything3_coreml_metadata",
    "depth_anything3_coreml_output_contract",
    "depth_anything3_coreml_validation_contract",
    "expected_depth_anything3_coreml_shapes",
    "freeze_depth_anything3_coreml_position_embedding",
    "postprocess_depth_anything3_coreml",
    "validate_depth_anything3_coreml_metadata",
    "validate_depth_anything3_coreml_profile",
    "validate_depth_anything3_coreml_raw_outputs",
    "wrap_depth_anything3_coreml_contract",
]
