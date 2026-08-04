"""Fused Triton grouped symmetric integer fake quantization."""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from libreyolo import kernels
from libreyolo.quant import fake_quant as _reference


_GROUPED_CONFIGS = (
    triton.Config({}, num_warps=1, num_stages=1),
    triton.Config({}, num_warps=2, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=8, num_stages=1),
)


@triton.autotune(configs=list(_GROUPED_CONFIGS), key=["n_rows", "n_cols", "qmax"])
@triton.jit
def _fake_quant_int_grouped_dynamic_kernel(
    weight_ptr,
    output_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    n_groups: tl.constexpr,
    qmin: tl.constexpr,
    qmax: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    row = group_id // n_groups
    group = group_id % n_groups
    columns = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    valid = columns < n_cols
    offsets = row * n_cols + columns
    values = tl.load(weight_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(amax / qmax, 1.0e-12)
    scaled = libdevice.div_rn(values, scale)
    codes = libdevice.float2int_rn(scaled)
    codes = tl.maximum(tl.minimum(codes, qmax), qmin)
    output = codes.to(tl.float32) * scale
    tl.store(output_ptr + offsets, output, mask=valid)


@triton.autotune(configs=list(_GROUPED_CONFIGS), key=["n_rows", "n_cols", "qmax"])
@triton.jit
def _fake_quant_int_grouped_frozen_kernel(
    weight_ptr,
    scale_ptr,
    output_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    n_groups: tl.constexpr,
    qmin: tl.constexpr,
    qmax: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    row = group_id // n_groups
    group = group_id % n_groups
    columns = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    valid = columns < n_cols
    offsets = row * n_cols + columns
    values = tl.load(weight_ptr + offsets, mask=valid).to(tl.float32)
    scale = tl.maximum(tl.load(scale_ptr + group_id), 1.0e-12)
    scaled = libdevice.div_rn(values, scale)
    codes = libdevice.float2int_rn(scaled)
    codes = tl.maximum(tl.minimum(codes, qmax), qmin)
    output = codes.to(tl.float32) * scale
    tl.store(output_ptr + offsets, output, mask=valid)


def _parameters(bits: int, group_size: int) -> tuple[int, int]:
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    return qmin, qmax


def _launch(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    n_rows, n_cols = weight.shape
    n_groups = triton.cdiv(n_cols, group_size)
    qmin, qmax = _parameters(bits, group_size)
    output = torch.empty_like(weight)
    kernel = (
        _fake_quant_int_grouped_dynamic_kernel
        if scale is None
        else _fake_quant_int_grouped_frozen_kernel
    )
    arguments = (weight, output) if scale is None else (weight, scale, output)
    kernel[(n_rows * n_groups,)](
        *arguments,
        n_rows=n_rows,
        n_cols=n_cols,
        n_groups=n_groups,
        qmin=qmin,
        qmax=qmax,
        GROUP_SIZE=group_size,
    )
    return output


class _FakeQuantIntGrouped(torch.autograd.Function):
    @staticmethod
    def forward(
        _ctx: Any,
        weight: torch.Tensor,
        bits: int,
        group_size: int,
        scale: torch.Tensor | None,
    ) -> torch.Tensor:
        return _launch(weight, bits, group_size, scale)

    @staticmethod
    def backward(_ctx: Any, grad_output: torch.Tensor):
        return grad_output, None, None, None


def _eligible() -> bool:
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


def _supported_inputs(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    scale: torch.Tensor | None,
) -> bool:
    if not (
        weight.is_cuda
        and weight.dtype == torch.float32
        and weight.is_contiguous()
        and weight.dim() == 2
        and weight.numel() > 0
        and (bits, group_size) in ((4, 128), (2, 64))
    ):
        return False
    n_groups = triton.cdiv(weight.shape[1], group_size)
    return scale is None or (
        scale.is_cuda
        and scale.device == weight.device
        and scale.dtype == torch.float32
        and scale.is_contiguous()
        and tuple(scale.shape) == (weight.shape[0], n_groups)
    )


def fake_quant_int_grouped(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply fused W4/W2 grouped fake quantization or safely use the oracle."""
    if not _supported_inputs(weight, bits, group_size, scale):
        return _reference.fake_quant_int_grouped(weight, bits, group_size, scale)
    return _FakeQuantIntGrouped.apply(weight, bits, group_size, scale)


def _cache_rows(kernel) -> list[dict[str, Any]]:
    return [
        {
            "rows": key[0],
            "columns": key[1],
            "qmax": key[2],
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
        for key, config in kernel.cache.items()
    ]


def autotune_cache() -> dict[str, list[dict[str, Any]]]:
    """Return dynamic and frozen configurations selected by the gate."""
    return {
        "dynamic": _cache_rows(_fake_quant_int_grouped_dynamic_kernel),
        "frozen": _cache_rows(_fake_quant_int_grouped_frozen_kernel),
    }


kernels.register(
    "fake_quant_int_grouped",
    fake_quant_int_grouped,
    name="triton",
    predicate=_eligible,
)


__all__ = ["autotune_cache", "fake_quant_int_grouped"]
