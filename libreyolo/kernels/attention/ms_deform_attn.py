"""Multi-scale deformable attention op slot (``ms_deform_attn``).

Slot signature (the classic Deformable-DETR layout):

- ``value``: ``(bs, Len_in, n_heads, c)``
- ``spatial_shapes``: ``(n_levels, 2)`` int64 tensor of ``(H, W)`` per level
- ``sampling_locations``: ``(bs, Len_q, n_heads, n_levels, n_points, 2)``
- ``attention_weights``: ``(bs, Len_q, n_heads, n_levels, n_points)``
- returns ``(bs, Len_q, n_heads * c)``, or None when the input is not
  eligible so the caller falls back to its portable path.

Like the GEMM slots, no reference implementation is registered: every model
family keeps its own upstream-parity ``grid_sample`` port as the default and
only consults this slot through :func:`maybe_ms_deform_attn`.

The in-tree provider loads the compiled CUDA kernel published at
``kernels-community/deformable-detr`` on the Hugging Face Hub (Apache-2.0)
via the optional ``kernels`` package. Nothing is vendored: the artifact is
fetched at runtime. Installing the ``libreyolo[hub-kernels]`` extra is the
opt-in; once the ``kernels`` package is present the provider is on by
default and ``LIBREYOLO_HUB_KERNELS=0`` disables it. The autograd bridge
below follows the ``MSDeformAttnFunction`` interface of Deformable-DETR
(https://github.com/fundamentalvision/Deformable-DETR, Apache-2.0,
Copyright (c) 2020 SenseTime).
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Optional

import torch

from .. import register, resolve

logger = logging.getLogger(__name__)

_HUB_REPO = "kernels-community/deformable-detr"
_MAX_IM2COL_STEP = 64

_hub_kernel = None
_hub_failed = False


def _hub_enabled() -> bool:
    """Hub kernels are on by default; installing the extra is the opt-in.

    The runtime fetch only ever happens when the optional ``kernels``
    package is installed (see :func:`_eligible`), so users who never
    installed ``libreyolo[hub-kernels]`` are unaffected.
    ``LIBREYOLO_HUB_KERNELS=0`` is the opt-out.
    """
    return os.environ.get("LIBREYOLO_HUB_KERNELS", "").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _eligible() -> bool:
    """Cheap predicate: the expensive Hub fetch is deferred to the first call."""
    return (
        _hub_enabled()
        and not _hub_failed
        and importlib.util.find_spec("kernels") is not None
        and torch.cuda.is_available()
    )


def _load_hub_kernel():
    """Fetch and cache the Hub kernel; a failure disables the provider."""
    global _hub_kernel, _hub_failed
    if _hub_kernel is not None or _hub_failed:
        return _hub_kernel
    try:
        from kernels import get_kernel

        _hub_kernel = get_kernel(_HUB_REPO)
    except Exception as exc:
        _hub_failed = True
        logger.warning("Hub kernel %s unavailable: %s", _HUB_REPO, exc)
    return _hub_kernel


class _MSDeformAttnFunction(torch.autograd.Function):
    """Autograd bridge over the compiled forward/backward pair."""

    @staticmethod
    def forward(
        ctx,
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output = _hub_kernel.ms_deform_attn_forward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
        ctx.save_for_backward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = (
            _hub_kernel.ms_deform_attn_backward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                grad_output.contiguous(),
                ctx.im2col_step,
            )
        )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def level_start_index(spatial_shapes: torch.Tensor) -> torch.Tensor:
    """Per-level start offsets into the flattened value, from (H, W) pairs."""
    areas = spatial_shapes[:, 0] * spatial_shapes[:, 1]
    return torch.cat([areas.new_zeros(1), areas.cumsum(0)[:-1]])


def _supported_inputs(
    value: torch.Tensor,
    spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> bool:
    if not isinstance(spatial_shapes, torch.Tensor):
        return False
    if not (
        value.is_cuda
        and sampling_locations.is_cuda
        and attention_weights.is_cuda
        and spatial_shapes.is_cuda
    ):
        return False
    # The compiled kernel dispatches on fp32; half inputs (e.g. autocast)
    # take the portable path.
    if not (
        value.dtype == torch.float32
        and sampling_locations.dtype == torch.float32
        and attention_weights.dtype == torch.float32
    ):
        return False
    if value.dim() != 4 or sampling_locations.dim() != 6 or attention_weights.dim() != 5:
        return False
    batch = value.shape[0]
    step = batch if batch < _MAX_IM2COL_STEP else _MAX_IM2COL_STEP
    return batch % step == 0


def hub_ms_deform_attn(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Run MSDA on the compiled Hub kernel, or return None to fall back."""
    global _hub_failed
    if not _supported_inputs(
        value, spatial_shapes, sampling_locations, attention_weights
    ):
        return None
    if _load_hub_kernel() is None:
        return None
    batch = value.shape[0]
    step = batch if batch < _MAX_IM2COL_STEP else _MAX_IM2COL_STEP
    shapes = spatial_shapes.to(dtype=torch.int64)
    try:
        return _MSDeformAttnFunction.apply(
            value.contiguous(),
            shapes,
            level_start_index(shapes),
            sampling_locations.contiguous(),
            attention_weights.contiguous(),
            step,
        )
    except Exception as exc:
        # A kernel that loads but rejects this torch/GPU combination must
        # never break inference: disable the provider and fall back.
        _hub_failed = True
        logger.warning("Hub kernel %s failed, falling back: %s", _HUB_REPO, exc)
        return None


def maybe_ms_deform_attn(
    value: torch.Tensor,
    spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Resolve the ``ms_deform_attn`` slot and run it, or return None.

    Callers keep their portable grid_sample port as the fallback. Tracing
    and export always take the portable path: exported graphs must not
    capture a runtime-fetched kernel.
    """
    if (
        torch.jit.is_tracing()
        or torch.compiler.is_compiling()
        or torch.onnx.is_in_onnx_export()
    ):
        return None
    impl = resolve("ms_deform_attn")
    if impl is None:
        return None
    return impl(value, spatial_shapes, sampling_locations, attention_weights)


register("ms_deform_attn", hub_ms_deform_attn, name="hub", predicate=_eligible)


__all__ = ["hub_ms_deform_attn", "level_start_index", "maybe_ms_deform_attn"]
