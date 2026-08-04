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
fetched at runtime, pinned to the audited revision in ``_HUB_REVISION`` so
a moved branch can never swap the binary that runs in-process. Installing
the ``libreyolo[hub-kernels]`` extra is the opt-in; once the ``kernels``
package is present the provider is on by default and
``LIBREYOLO_HUB_KERNELS=0`` disables it. The autograd bridge
below follows the ``MSDeformAttnFunction`` interface of Deformable-DETR
(https://github.com/fundamentalvision/Deformable-DETR, Apache-2.0,
Copyright (c) 2020 SenseTime).
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
from typing import Optional

import torch

from .. import register, resolve

logger = logging.getLogger(__name__)

_HUB_REPO = "kernels-community/deformable-detr"
# Commit pin: native code fetched from the Hub executes in-process (forward
# and backward), so selection by repo name alone would let a moved branch
# change the binary under users. Bump deliberately, with a GPU parity run
# (tests/unit/kernels/test_ms_deform_attn.py::test_hub_matches_portable_on_cuda).
_HUB_REVISION = "4d2393e5d7879f7cf68db04cc7c9c7342272bc05"
_MAX_IM2COL_STEP = 64

_hub_kernel = None
_hub_failed = False
# Why the load failed, for tests and for choosing the log level. "unsupported"
# is a normal outcome (no compiled build for this OS/torch/CUDA combination);
# "unresolvable" means the pin above cannot be fetched at all, which is a
# packaging bug on our side and is wrong on every platform.
_hub_failure_kind: Optional[str] = None


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


def _classify_load_failure(exc: BaseException) -> str:
    """Split "no build for this machine" from "the pin cannot be fetched".

    The first is the normal outcome on any platform the kernel authors did
    not build for, and the portable path is the right answer. The second
    means :data:`_HUB_REVISION` is unusable, which is wrong *everywhere* and
    silently costs every user the accelerated path -- so it must not look
    like routine fallback in the logs.
    """
    if isinstance(exc, ImportError):
        return "unsupported"
    name = type(exc).__name__
    if name in ("RevisionNotFoundError", "RepositoryNotFoundError", "EntryNotFoundError"):
        return "unresolvable"
    text = str(exc).lower()
    if "revision" in text or "rev id" in text:
        return "unresolvable"
    return "unsupported"


def _load_hub_kernel():
    """Fetch and cache the Hub kernel; a failure disables the provider."""
    global _hub_kernel, _hub_failed, _hub_failure_kind
    if _hub_kernel is not None or _hub_failed:
        return _hub_kernel
    try:
        from kernels import get_kernel

        _hub_kernel = get_kernel(_HUB_REPO, revision=_HUB_REVISION)
    except Exception as exc:
        _hub_failed = True
        _hub_failure_kind = _classify_load_failure(exc)
        if _hub_failure_kind == "unresolvable":
            logger.error(
                "Hub kernel %s cannot be resolved at the pinned revision %s, so "
                "the accelerated path is unavailable on every platform. This is "
                "a LibreYOLO packaging problem, not a problem with your machine; "
                "the usual cause is a `kernels` client outside the range pinned "
                "by the libreyolo[hub-kernels] extra, since newer clients reject "
                "commit-SHA revisions. Falling back to the portable "
                "implementation. Underlying error: %s",
                _HUB_REPO,
                _HUB_REVISION,
                exc,
            )
        else:
            logger.info(
                "No compiled %s build for this platform; using the portable "
                "implementation: %s",
                _HUB_REPO,
                exc,
            )
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
    if batch == 0:
        return False
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
    if _hub_kernel is None and torch.cuda.is_current_stream_capturing():
        # Never fetch/load the kernel module inside CUDA graph capture; the
        # load performs I/O and module initialization that would poison the
        # capture. Graph warmup iterations load it, after which captures
        # record the compiled op like any other.
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


def _not_exporting() -> bool:
    """Fallback for ``torch.compiler.is_exporting`` on torch versions without it."""
    return False


def ms_deform_attn_available() -> bool:
    """Whether the slot could run here, checked before adapting layouts.

    Families whose native layout differs from the slot's ask this first so
    the adaptation work (and any ``tolist()`` that would bake constants into
    a trace) never happens on the portable path. Tracing and export always
    report unavailable: exported graphs must not capture a runtime-fetched
    kernel. ``is_exporting`` covers non-strict ``torch.export``, which traces
    with FakeTensors without setting ``is_compiling``.
    """
    if (
        torch.jit.is_tracing()
        or torch.compiler.is_compiling()
        or getattr(torch.compiler, "is_exporting", _not_exporting)()
        or torch.onnx.is_in_onnx_export()
    ):
        return False
    return resolve("ms_deform_attn") is not None


@functools.lru_cache(maxsize=32)
def _cached_shapes(pairs: tuple, device: torch.device) -> torch.Tensor:
    return torch.tensor(pairs, dtype=torch.int64, device=device)


def spatial_shapes_tensor(value_spatial_shapes, device) -> torch.Tensor:
    """Normalize per-level ``(H, W)`` pairs to the int64 tensor the slot wants.

    Several families carry their spatial shapes as Python pairs; the compiled
    kernel reads them from device memory. The small results are cached so the
    host-to-device copy happens once per shape set rather than per call. Only
    valid under :func:`ms_deform_attn_available`, which rules out tracing.
    """
    if isinstance(value_spatial_shapes, torch.Tensor):
        if (
            value_spatial_shapes.dtype == torch.int64
            and value_spatial_shapes.device == device
        ):
            return value_spatial_shapes
        value_spatial_shapes = value_spatial_shapes.tolist()
    pairs = tuple((int(height), int(width)) for height, width in value_spatial_shapes)
    return _cached_shapes(pairs, device)


def maybe_ms_deform_attn(
    value: torch.Tensor,
    spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Resolve the ``ms_deform_attn`` slot and run it, or return None.

    Callers keep their portable grid_sample port as the fallback. Tracing
    and export always take the portable path: exported graphs must not
    capture a runtime-fetched kernel. ``is_exporting`` covers non-strict
    ``torch.export``, which traces with FakeTensors without setting
    ``is_compiling``.
    """
    if not ms_deform_attn_available():
        return None
    impl = resolve("ms_deform_attn")
    if impl is None:
        return None
    return impl(value, spatial_shapes, sampling_locations, attention_weights)


def maybe_ms_deform_attn_v2(
    value: torch.Tensor,
    spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list,
) -> Optional[torch.Tensor]:
    """Slot adapter for the RT-DETRv2-style flat-points layout.

    v2 cores carry ``sum(num_points_list)`` sampling points flattened across
    levels and support a different point count per level, while the slot's
    layout is ``(n_levels, n_points)``. Only a uniform count reshapes onto
    it, so a ragged ``num_points_list`` returns None and the caller keeps its
    portable path. ``value`` must already be in the slot's
    ``(bs, Len_in, n_heads, c)`` layout.
    """
    if not ms_deform_attn_available():
        return None
    levels = len(num_points_list)
    if levels == 0 or len(spatial_shapes) != levels:
        return None
    points = int(num_points_list[0])
    if any(int(count) != points for count in num_points_list):
        return None
    return maybe_ms_deform_attn(
        value,
        spatial_shapes,
        sampling_locations.unflatten(3, (levels, points)),
        attention_weights.unflatten(3, (levels, points)),
    )


register("ms_deform_attn", hub_ms_deform_attn, name="hub", predicate=_eligible)


__all__ = [
    "hub_ms_deform_attn",
    "level_start_index",
    "maybe_ms_deform_attn",
    "maybe_ms_deform_attn_v2",
    "ms_deform_attn_available",
    "spatial_shapes_tensor",
]
