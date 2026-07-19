"""Fused Triton MXFP4 weight and activation fake quantization."""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from libreyolo.quant import fake_quant as _reference
from libreyolo.quant import kernels


_MXFP4_CONFIGS = (
    triton.Config({"BLOCKS_PER_PROGRAM": 1}, num_warps=1, num_stages=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 4}, num_warps=1, num_stages=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 8}, num_warps=2, num_stages=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCKS_PER_PROGRAM": 64}, num_warps=8, num_stages=2),
)


@triton.autotune(configs=list(_MXFP4_CONFIGS), key=["n_rows", "n_cols"])
@triton.jit
def _fake_quant_mxfp4_kernel(
    input_ptr,
    output_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    block_width: tl.constexpr = 32
    blocks_per_row: tl.constexpr = (n_cols + block_width - 1) // block_width
    total_blocks: tl.constexpr = n_rows * blocks_per_row
    block_ids = tl.program_id(0) * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    lanes = tl.arange(0, block_width)
    rows = block_ids[:, None] // blocks_per_row
    columns = (block_ids[:, None] % blocks_per_row) * block_width + lanes[None, :]
    valid = (block_ids[:, None] < total_blocks) & (columns < n_cols)
    offsets = rows * n_cols + columns
    values = tl.load(input_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    block_amax = tl.max(tl.abs(values), axis=1, keep_dims=True)
    clamped_amax = tl.maximum(block_amax, 1.0e-30)
    amax_bits = clamped_amax.to(tl.int32, bitcast=True)
    amax_exponent = ((amax_bits >> 23) & 0xFF) - 127
    scale_exponent = tl.maximum(tl.minimum(amax_exponent - 2, 127), -127)
    normal_scale_bits = (scale_exponent + 127) << 23
    scale_bits = tl.where(scale_exponent == -127, 0x00400000, normal_scale_bits)
    effective_scale = scale_bits.to(tl.float32, bitcast=True)

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
    output = (level * sign) * effective_scale
    tl.store(output_ptr + offsets, output, mask=valid)


def _launch(inputs: torch.Tensor) -> torch.Tensor:
    matrix = inputs.reshape(-1, inputs.shape[-1])
    n_rows, n_cols = matrix.shape
    blocks_per_row = triton.cdiv(n_cols, 32)
    output = torch.empty_like(matrix)

    def grid(meta):
        return (triton.cdiv(n_rows * blocks_per_row, meta["BLOCKS_PER_PROGRAM"]),)

    _fake_quant_mxfp4_kernel[grid](
        matrix,
        output,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    return output.reshape_as(inputs)


class _FakeQuantMXFP4(torch.autograd.Function):
    @staticmethod
    def forward(_ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        return _launch(inputs)

    @staticmethod
    def backward(_ctx: Any, grad_output: torch.Tensor):
        return grad_output


def _eligible() -> bool:
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


def _supported_input(inputs: torch.Tensor) -> bool:
    return (
        inputs.is_cuda
        and inputs.dtype == torch.float32
        and inputs.is_contiguous()
        and inputs.dim() >= 1
        and inputs.numel() > 0
        and inputs.shape[-1] > 0
    )


def fake_quant_mxfp4_weight(weight: torch.Tensor) -> torch.Tensor:
    """Apply fused MXFP4 weight fake quantization or safely use the oracle."""
    if not _supported_input(weight):
        return _reference.fake_quant_mxfp4_weight(weight)
    return _FakeQuantMXFP4.apply(weight)


def fake_quant_mxfp4_dynamic(inputs: torch.Tensor) -> torch.Tensor:
    """Apply fused dynamic MXFP4 fake quantization or safely use the oracle."""
    if not _supported_input(inputs):
        return _reference.fake_quant_mxfp4_dynamic(inputs)
    return _FakeQuantMXFP4.apply(inputs)


def autotune_cache() -> list[dict[str, Any]]:
    """Return configurations selected across weight and activation shapes."""
    return [
        {
            "rows": key[0],
            "columns": key[1],
            "blocks_per_program": config.kwargs["BLOCKS_PER_PROGRAM"],
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
        for key, config in _fake_quant_mxfp4_kernel.cache.items()
    ]


kernels.register(
    "fake_quant_mxfp4_weight",
    fake_quant_mxfp4_weight,
    name="triton",
    predicate=_eligible,
)
kernels.register(
    "fake_quant_mxfp4_dynamic",
    fake_quant_mxfp4_dynamic,
    name="triton",
    predicate=_eligible,
)


__all__ = [
    "autotune_cache",
    "fake_quant_mxfp4_dynamic",
    "fake_quant_mxfp4_weight",
]
