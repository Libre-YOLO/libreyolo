"""Fused Triton FP8 E4M3 fake quantization."""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from libreyolo.quant import fake_quant as _reference
from libreyolo.quant import kernels


_FP8_CONFIGS = (
    triton.Config({"BLOCK_SIZE": 256}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
)


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


@triton.autotune(
    configs=list(_FP8_CONFIGS),
    key=["n_elements", "channel_size"],
)
@triton.jit
def _fake_quant_fp8_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    channel_size: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=valid).to(tl.float32)
    if PER_CHANNEL:
        scale_offsets = offsets // channel_size
        scale = tl.load(scale_ptr + scale_offsets, mask=valid)
    else:
        scale = tl.load(scale_ptr)
    scale = tl.maximum(scale, 1.0e-12)
    scaled = libdevice.div_rn(values, scale)
    scaled = tl.maximum(tl.minimum(scaled, 448.0), -448.0)
    quantized = scaled.to(tl.float8e4nv).to(tl.float32) * scale
    tl.store(output_ptr + offsets, quantized, mask=valid)


def _launch(inputs: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    per_channel = scale.numel() != 1
    channel_size = inputs.numel() // inputs.shape[0] if per_channel else 0
    output = torch.empty_like(inputs)

    def grid(meta):
        return (triton.cdiv(inputs.numel(), meta["BLOCK_SIZE"]),)

    _fake_quant_fp8_kernel[grid](
        inputs,
        scale,
        output,
        n_elements=inputs.numel(),
        channel_size=channel_size,
        PER_CHANNEL=per_channel,
    )
    return output


class _FakeQuantFP8(torch.autograd.Function):
    @staticmethod
    def forward(
        _ctx: Any,
        inputs: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return _launch(inputs, scale)

    @staticmethod
    def backward(_ctx: Any, grad_output: torch.Tensor):
        return grad_output, None


def _eligible() -> bool:
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


def _supported_inputs(inputs: torch.Tensor, scale: torch.Tensor) -> bool:
    if not (
        inputs.is_cuda
        and scale.is_cuda
        and inputs.device == scale.device
        and inputs.dtype == torch.float32
        and scale.dtype == torch.float32
        and inputs.is_contiguous()
        and scale.is_contiguous()
        and inputs.numel() > 0
        and scale.numel() > 0
    ):
        return False
    if scale.numel() == 1:
        return True
    return (
        inputs.dim() >= 1
        and scale.dim() == inputs.dim()
        and scale.shape[0] == inputs.shape[0]
        and all(dimension == 1 for dimension in scale.shape[1:])
    )


def fake_quant_fp8(inputs: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply fused FP8 E4M3 fake quantization or safely use the oracle."""
    if not _supported_inputs(inputs, scale):
        return _reference.fake_quant_fp8(inputs, scale)
    return _FakeQuantFP8.apply(inputs, scale)


def autotune_cache() -> list[dict[str, Any]]:
    """Return configurations selected for every canonical scale layout."""
    return [
        {
            "elements": key[0],
            "channel_size": key[1],
            "block_size": config.kwargs["BLOCK_SIZE"],
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
        for key, config in _fake_quant_fp8_kernel.cache.items()
    ]


kernels.register(
    "fake_quant_fp8",
    fake_quant_fp8,
    name="triton",
    predicate=_eligible,
)
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
    "autotune_cache",
    "fake_quant_fp8",
    "perchannel_fp8_epilogue",
    "static_fp8_cast",
]
