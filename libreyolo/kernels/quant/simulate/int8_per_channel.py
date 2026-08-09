"""Fused Triton per-channel symmetric INT8 fake quantization."""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from libreyolo import kernels
from libreyolo.quant import fake_quant as _reference


_DYNAMIC_CONFIGS = (
    triton.Config({}, num_warps=1, num_stages=1),
    triton.Config({}, num_warps=2, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=8, num_stages=1),
)

_FROZEN_CONFIGS = (
    triton.Config({"BLOCK_SIZE": 256}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
)


@triton.autotune(configs=list(_DYNAMIC_CONFIGS), key=["n_channels", "channel_size"])
@triton.jit
def _fake_quant_int8_per_channel_dynamic_kernel(
    weight_ptr,
    output_ptr,
    n_channels: tl.constexpr,
    channel_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = channel * channel_size + tl.arange(0, BLOCK_SIZE)
    valid = tl.arange(0, BLOCK_SIZE) < channel_size
    values = tl.load(weight_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(amax / 127.0, 1.0e-12)
    inverse_scale = libdevice.div_rn(1.0, scale)
    codes = libdevice.float2int_rn(values * inverse_scale)
    codes = tl.maximum(tl.minimum(codes, 127), -128)
    output = codes.to(tl.float32) * scale
    tl.store(output_ptr + offsets, output, mask=valid)


@triton.autotune(configs=list(_FROZEN_CONFIGS), key=["n_elements", "channel_size"])
@triton.jit
def _fake_quant_int8_per_channel_frozen_kernel(
    weight_ptr,
    scale_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    channel_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements
    values = tl.load(weight_ptr + offsets, mask=valid).to(tl.float32)
    channels = offsets // channel_size
    scale = tl.maximum(tl.load(scale_ptr + channels, mask=valid), 1.0e-12)
    inverse_scale = libdevice.div_rn(1.0, scale)
    codes = libdevice.float2int_rn(values * inverse_scale)
    codes = tl.maximum(tl.minimum(codes, 127), -128)
    output = codes.to(tl.float32) * scale
    tl.store(output_ptr + offsets, output, mask=valid)


@triton.autotune(configs=list(_FROZEN_CONFIGS), key=["n_elements", "channel_size"])
@triton.jit
def _fake_quant_int8_per_channel_backward_kernel(
    grad_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    channel_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements
    values = tl.load(weight_ptr + offsets, mask=valid).to(tl.float32)
    gradients = tl.load(grad_ptr + offsets, mask=valid).to(tl.float32)
    channels = offsets // channel_size
    scale = tl.maximum(tl.load(scale_ptr + channels, mask=valid), 1.0e-12)
    inverse_scale = libdevice.div_rn(1.0, scale)
    codes = libdevice.float2int_rn(values * inverse_scale)
    inside = (codes >= -128) & (codes <= 127)
    tl.store(output_ptr + offsets, tl.where(inside, gradients, 0.0), mask=valid)


def _flat_grid(n_elements: int):
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    return grid


def _launch_dynamic(weight: torch.Tensor) -> torch.Tensor:
    n_channels = weight.shape[0]
    channel_size = weight.numel() // n_channels
    block_size = triton.next_power_of_2(channel_size)
    output = torch.empty_like(weight)
    _fake_quant_int8_per_channel_dynamic_kernel[(n_channels,)](
        weight,
        output,
        n_channels=n_channels,
        channel_size=channel_size,
        BLOCK_SIZE=block_size,
    )
    return output


def _launch_frozen(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    channel_size = weight.numel() // weight.shape[0]
    output = torch.empty_like(weight)
    _fake_quant_int8_per_channel_frozen_kernel[_flat_grid(weight.numel())](
        weight,
        scale,
        output,
        n_elements=weight.numel(),
        channel_size=channel_size,
    )
    return output


def _launch_backward(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    gradients = grad_output.contiguous()
    channel_size = weight.numel() // weight.shape[0]
    grad_weight = torch.empty_like(weight)
    _fake_quant_int8_per_channel_backward_kernel[_flat_grid(weight.numel())](
        gradients,
        weight,
        scale,
        grad_weight,
        n_elements=weight.numel(),
        channel_size=channel_size,
    )
    return grad_weight


class _FakeQuantInt8PerChannel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.dynamic_scale = scale is None
        if scale is None:
            return _launch_dynamic(weight)
        ctx.save_for_backward(weight, scale)
        return _launch_frozen(weight, scale)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        if ctx.dynamic_scale:
            return grad_output, None
        weight, scale = ctx.saved_tensors
        return _launch_backward(grad_output, weight, scale), None


def _eligible() -> bool:
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


def _supported_inputs(
    weight: torch.Tensor,
    ch_axis: int,
    scale: torch.Tensor | None,
) -> bool:
    if not (
        weight.is_cuda
        and weight.dtype == torch.float32
        and weight.is_contiguous()
        and weight.dim() >= 1
        and weight.numel() > 0
        and weight.shape[0] > 0
        and weight.numel() // weight.shape[0] <= 65536
        and ch_axis == 0
    ):
        return False
    return scale is None or (
        scale.is_cuda
        and scale.device == weight.device
        and scale.dtype == torch.float32
        and scale.is_contiguous()
        and scale.numel() == weight.shape[0]
    )


def fake_quant_int8_per_channel(
    weight: torch.Tensor,
    ch_axis: int = 0,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply fused per-channel INT8 fake quantization or use the oracle."""
    if not _supported_inputs(weight, ch_axis, scale):
        return _reference.fake_quant_int8_per_channel(weight, ch_axis, scale)
    return _FakeQuantInt8PerChannel.apply(weight, scale)


def _cache_rows(kernel, dynamic: bool = False) -> list[dict[str, Any]]:
    rows = []
    for key, config in kernel.cache.items():
        row = {
            "elements_or_channels": key[0],
            "channel_size": key[1],
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
        if not dynamic:
            row["block_size"] = config.kwargs["BLOCK_SIZE"]
        rows.append(row)
    return rows


def autotune_cache() -> dict[str, list[dict[str, Any]]]:
    """Return dynamic, frozen, and backward configurations used by gates."""
    return {
        "dynamic": _cache_rows(
            _fake_quant_int8_per_channel_dynamic_kernel,
            dynamic=True,
        ),
        "frozen": _cache_rows(_fake_quant_int8_per_channel_frozen_kernel),
        "backward": _cache_rows(_fake_quant_int8_per_channel_backward_kernel),
    }


kernels.register(
    "fake_quant_int8_per_channel",
    fake_quant_int8_per_channel,
    name="triton",
    predicate=_eligible,
)


__all__ = ["autotune_cache", "fake_quant_int8_per_channel"]
