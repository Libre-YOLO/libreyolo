"""Fused Triton NVFP4 weight unpacking."""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
import triton
import triton.language as tl

from libreyolo.quant import kernels
from libreyolo.quant import packing as _reference


_UNPACK_CONFIGS = (
    triton.Config({"BLOCK_SIZE": 256}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
)


@triton.jit
def _decode_e2m1(code):
    index = code & 0x07
    level = tl.where(index == 0, 0.0, 0.5)
    level = tl.where(index == 2, 1.0, level)
    level = tl.where(index == 3, 1.5, level)
    level = tl.where(index == 4, 2.0, level)
    level = tl.where(index == 5, 3.0, level)
    level = tl.where(index == 6, 4.0, level)
    level = tl.where(index == 7, 6.0, level)
    sign = tl.where((code & 0x08) != 0, -1.0, 1.0)
    return level * sign


@triton.autotune(
    configs=list(_UNPACK_CONFIGS),
    key=["out_features", "payload_cols"],
)
@triton.jit
def _unpack_nvfp4_kernel(
    payload_ptr,
    block_scale_ptr,
    tensor_amax_ptr,
    output_ptr,
    out_features: tl.constexpr,
    payload_cols: tl.constexpr,
    in_features: tl.constexpr,
    n_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total_bytes: tl.constexpr = out_features * payload_cols
    byte_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_bytes = byte_offsets < total_bytes
    rows = byte_offsets // payload_cols
    byte_columns = byte_offsets % payload_cols
    output_columns = byte_columns * 2
    packed = tl.load(payload_ptr + byte_offsets, mask=valid_bytes).to(tl.int32)
    block_indices = output_columns // 16
    block_scale = tl.load(
        block_scale_ptr + rows * n_blocks + block_indices,
        mask=valid_bytes,
    ).to(tl.float32)
    tensor_scale = tl.maximum(tl.load(tensor_amax_ptr).to(tl.float32) / 2688.0, 1.0e-12)
    effective_scale = tl.maximum(block_scale * tensor_scale, 1.0e-12)
    output_offsets = rows * in_features + output_columns
    low = _decode_e2m1(packed & 0x0F) * effective_scale
    high = _decode_e2m1((packed >> 4) & 0x0F) * effective_scale
    tl.store(
        output_ptr + output_offsets,
        low,
        mask=valid_bytes & (output_columns < in_features),
    )
    tl.store(
        output_ptr + output_offsets + 1,
        high,
        mask=valid_bytes & (output_columns + 1 < in_features),
    )


def _launch(
    payload: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_amax: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    out_features, payload_cols = payload.shape
    n_blocks = triton.cdiv(in_features, 16)
    output = torch.empty(
        (out_features, in_features),
        device=payload.device,
        dtype=torch.float32,
    )

    def grid(meta):
        return (triton.cdiv(out_features * payload_cols, meta["BLOCK_SIZE"]),)

    _unpack_nvfp4_kernel[grid](
        payload,
        block_scale,
        tensor_amax,
        output,
        out_features=out_features,
        payload_cols=payload_cols,
        in_features=in_features,
        n_blocks=n_blocks,
    )
    return output


class _UnpackNVFP4(torch.autograd.Function):
    @staticmethod
    def forward(
        _ctx: Any,
        payload: torch.Tensor,
        block_scale: torch.Tensor,
        tensor_amax: torch.Tensor,
        in_features: int,
    ) -> torch.Tensor:
        return _launch(payload, block_scale, tensor_amax, in_features)

    @staticmethod
    def backward(_ctx: Any, _grad_output: torch.Tensor):
        return None, None, None, None


def _eligible() -> bool:
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


def _supported_inputs(
    payload: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_amax: torch.Tensor,
    in_features: int,
) -> bool:
    scale_dtypes = (torch.float32,)
    if hasattr(torch, "float8_e4m3fn"):
        scale_dtypes += (torch.float8_e4m3fn,)
    if not (
        payload.is_cuda
        and block_scale.is_cuda
        and tensor_amax.is_cuda
        and payload.device == block_scale.device == tensor_amax.device
        and payload.dtype == torch.uint8
        and block_scale.dtype in scale_dtypes
        and tensor_amax.dtype == torch.float32
        and payload.is_contiguous()
        and block_scale.is_contiguous()
        and tensor_amax.is_contiguous()
        and payload.dim() == 2
        and block_scale.dim() == 2
        and payload.shape[0] == block_scale.shape[0]
        and payload.numel() > 0
        and tensor_amax.numel() == 1
        and not block_scale.requires_grad
        and not tensor_amax.requires_grad
        and in_features > 0
    ):
        return False
    n_blocks = triton.cdiv(in_features, 16)
    return payload.shape[1] == n_blocks * 8 and block_scale.shape[1] == n_blocks


def unpack_nvfp4(
    payload: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_amax: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    """Decode and scale NVFP4 weights, or safely use the eager oracle."""
    if not _supported_inputs(payload, block_scale, tensor_amax, in_features):
        return _reference.unpack_nvfp4_weight(
            payload,
            block_scale,
            tensor_amax,
            in_features,
        )
    return _UnpackNVFP4.apply(payload, block_scale, tensor_amax, in_features)


def autotune_cache() -> list[dict[str, Any]]:
    """Return configurations selected for every canonical packed shape."""
    return [
        {
            "out_features": key[0],
            "payload_columns": key[1],
            "block_size": config.kwargs["BLOCK_SIZE"],
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
        for key, config in _unpack_nvfp4_kernel.cache.items()
    ]


kernels.register(
    "unpack_nvfp4",
    unpack_nvfp4,
    name="triton",
    predicate=_eligible,
)


__all__ = ["autotune_cache", "unpack_nvfp4"]
