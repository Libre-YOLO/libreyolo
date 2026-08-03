"""Fused Triton prologue/epilogue around the native FP8 GEMM.

These are finalized-only execution kernels: they wrap the ``fp8_gemm``
tensor-core path from ``scaled_mm_fp8.py`` with a fused activation cast and
a fused per-channel epilogue. They have no backward; the simulated tier
covers training.
"""

from __future__ import annotations

import importlib.util

import torch
import triton
import triton.language as tl

from libreyolo import kernels


@triton.jit
def _static_fp8_cast_kernel(
    input_ptr,
    inverse_scale_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused static-scale activation quantization for native FP8 GEMMs."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements
    # Scaling follows the prepared FP8 oracle in fp32. Fusing it into this
    # memory pass avoids materializing the fp32 activation tensor.
    values = tl.load(input_ptr + offsets, mask=valid).to(tl.float32)
    inverse_scale = tl.load(inverse_scale_ptr).to(tl.float32)
    values *= inverse_scale
    values = tl.maximum(tl.minimum(values, 448.0), -448.0)
    tl.store(output_ptr + offsets, values.to(tl.float8e4nv), mask=valid)


def static_fp8_cast(inputs: torch.Tensor, inverse_scale: torch.Tensor) -> torch.Tensor:
    """Multiply, saturate, and cast a contiguous CUDA tensor to E4M3."""
    if not (
        inputs.is_cuda
        and inputs.is_contiguous()
        and inverse_scale.is_cuda
        and inverse_scale.numel() == 1
    ):
        raise ValueError("static FP8 cast expects contiguous CUDA inputs and scale")
    output = torch.empty_like(inputs, dtype=torch.float8_e4m3fn)
    _static_fp8_cast_kernel[(triton.cdiv(inputs.numel(), 256),)](
        inputs,
        inverse_scale,
        output,
        n_elements=inputs.numel(),
        BLOCK_SIZE=256,
    )
    return output


@triton.jit
def _perchannel_epilogue_kernel(
    input_ptr,
    scale_ptr,
    bias_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    n_columns: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements
    columns = offsets % n_columns
    values = tl.load(input_ptr + offsets, mask=valid).to(tl.float32)
    scale = tl.load(scale_ptr + columns, mask=valid).to(tl.float32)
    values *= scale
    if HAS_BIAS:
        values += tl.load(bias_ptr + columns, mask=valid).to(tl.float32)
    tl.store(output_ptr + offsets, values, mask=valid)


def perchannel_fp8_epilogue(
    inputs: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Fuse row-scale restoration, bias, and FP16 storage in one pass."""
    if not (
        inputs.is_cuda
        and inputs.is_contiguous()
        and scale.is_cuda
        and scale.numel() == inputs.shape[-1]
        and (bias is None or (bias.is_cuda and bias.numel() == inputs.shape[-1]))
    ):
        raise ValueError("FP8 epilogue expects contiguous CUDA tensors")
    output = torch.empty_like(inputs, dtype=torch.float16)
    _perchannel_epilogue_kernel[(triton.cdiv(inputs.numel(), 256),)](
        inputs,
        scale,
        bias if bias is not None else scale,
        output,
        n_elements=inputs.numel(),
        n_columns=inputs.shape[-1],
        HAS_BIAS=bias is not None,
        BLOCK_SIZE=256,
    )
    return output


def _eligible() -> bool:
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


kernels.register(
    "fp8_cast_static",
    static_fp8_cast,
    name="triton",
    predicate=_eligible,
)
kernels.register(
    "fp8_perchannel_epilogue",
    perchannel_fp8_epilogue,
    name="triton",
    predicate=_eligible,
)


__all__ = [
    "perchannel_fp8_epilogue",
    "static_fp8_cast",
]
