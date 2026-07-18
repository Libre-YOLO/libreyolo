"""Fused Triton NVFP4 weight fake quantization.

This is a clean-room implementation of LibreYOLO's reference arithmetic in
``quant.fake_quant``.  It reads each 16-value block once, derives its E4M3
scale, rounds onto the E2M1 grid, and writes the dequantized values once.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from libreyolo.quant import fake_quant as _reference
from libreyolo.quant import kernels


_NVFP4_CONFIGS = (
    triton.Config({"BLOCKS_PER_PROGRAM": 1}, num_warps=1, num_stages=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 4}, num_warps=1, num_stages=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 8}, num_warps=2, num_stages=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCKS_PER_PROGRAM": 64}, num_warps=8, num_stages=2),
)


@triton.autotune(configs=list(_NVFP4_CONFIGS), key=["n_rows", "n_cols"])
@triton.jit
def _fake_quant_nvfp4_weight_kernel(
    weight_ptr,
    tensor_amax_ptr,
    output_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    block_width: tl.constexpr = 16
    blocks_per_row: tl.constexpr = (n_cols + block_width - 1) // block_width
    total_blocks: tl.constexpr = n_rows * blocks_per_row

    block_ids = tl.program_id(0) * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    lanes = tl.arange(0, block_width)
    rows = block_ids[:, None] // blocks_per_row
    columns = (block_ids[:, None] % blocks_per_row) * block_width + lanes[None, :]
    valid = (block_ids[:, None] < total_blocks) & (columns < n_cols)
    offsets = rows * n_cols + columns
    values = tl.load(weight_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    tensor_amax = tl.load(tensor_amax_ptr).to(tl.float32)
    tensor_scale = tl.maximum(tensor_amax / (448.0 * 6.0), 1.0e-12)
    block_amax = tl.max(tl.abs(values), axis=1, keep_dims=True)
    block_scale = block_amax / (6.0 * tensor_scale)
    block_scale = tl.maximum(tl.minimum(block_scale, 448.0), 1.0e-12)
    block_scale = block_scale.to(tl.float8e4nv).to(tl.float32)
    effective_scale = tl.maximum(block_scale * tensor_scale, 1.0e-12)

    scaled = libdevice.div_rn(values, effective_scale)
    magnitude = tl.minimum(tl.abs(scaled), 6.0)
    level = tl.where(magnitude > 0.25, 0.5, 0.0)
    level = tl.where(magnitude > 0.75, 1.0, level)
    level = tl.where(magnitude > 1.25, 1.5, level)
    level = tl.where(magnitude > 1.75, 2.0, level)
    level = tl.where(magnitude > 2.5, 3.0, level)
    level = tl.where(magnitude > 3.5, 4.0, level)
    level = tl.where(magnitude > 5.0, 6.0, level)
    sign = tl.where(values > 0.0, 1.0, tl.where(values < 0.0, -1.0, 0.0))
    quantized = (level * sign) * effective_scale
    tl.store(output_ptr + offsets, quantized, mask=valid)


def _launch(weight: torch.Tensor, tensor_amax: torch.Tensor) -> torch.Tensor:
    rows, columns = weight.shape
    blocks_per_row = triton.cdiv(columns, 16)
    output = torch.empty_like(weight)

    def grid(meta):
        return (triton.cdiv(rows * blocks_per_row, meta["BLOCKS_PER_PROGRAM"]),)

    _fake_quant_nvfp4_weight_kernel[grid](
        weight,
        tensor_amax,
        output,
        n_rows=rows,
        n_cols=columns,
    )
    return output


class _FakeQuantNVFP4Weight(torch.autograd.Function):
    @staticmethod
    def forward(
        _ctx: Any, weight: torch.Tensor, tensor_amax: torch.Tensor
    ) -> torch.Tensor:
        return _launch(weight, tensor_amax)

    @staticmethod
    def backward(_ctx: Any, grad_output: torch.Tensor):
        return grad_output, None


def _eligible() -> bool:
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


def _supported_inputs(weight: torch.Tensor, tensor_amax: torch.Tensor) -> bool:
    return (
        weight.is_cuda
        and tensor_amax.is_cuda
        and weight.dtype == torch.float32
        and tensor_amax.dtype == torch.float32
        and weight.dim() == 2
        and weight.is_contiguous()
        and tensor_amax.numel() == 1
    )


def fake_quant_nvfp4_weight(
    weight: torch.Tensor, tensor_amax: torch.Tensor
) -> torch.Tensor:
    """Apply fused NVFP4 fake quantization or safely use the oracle."""
    if not _supported_inputs(weight, tensor_amax):
        return _reference.fake_quant_nvfp4_weight(weight, tensor_amax)
    return _FakeQuantNVFP4Weight.apply(weight, tensor_amax)


def autotune_cache() -> list[dict[str, Any]]:
    """Return selected configurations for the rolling benchmark report."""
    return [
        {
            "rows": key[0],
            "columns": key[1],
            "blocks_per_program": config.kwargs["BLOCKS_PER_PROGRAM"],
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
        for key, config in _fake_quant_nvfp4_weight_kernel.cache.items()
    ]


kernels.register(
    "fake_quant_nvfp4_weight",
    fake_quant_nvfp4_weight,
    name="triton",
    predicate=_eligible,
)


__all__ = ["autotune_cache", "fake_quant_nvfp4_weight"]
