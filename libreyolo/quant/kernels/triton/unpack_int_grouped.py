"""Fused Triton grouped integer weight unpacking."""

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


@triton.autotune(
    configs=list(_UNPACK_CONFIGS),
    key=["out_features", "payload_cols", "BITS"],
)
@triton.jit
def _unpack_int_grouped_kernel(
    payload_ptr,
    scale_ptr,
    output_ptr,
    out_features: tl.constexpr,
    payload_cols: tl.constexpr,
    in_features: tl.constexpr,
    n_groups: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    per_byte: tl.constexpr = 8 // BITS
    total_bytes: tl.constexpr = out_features * payload_cols
    byte_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_bytes = byte_offsets < total_bytes
    rows = byte_offsets // payload_cols
    byte_columns = byte_offsets % payload_cols
    output_columns = byte_columns * per_byte
    packed = tl.load(payload_ptr + byte_offsets, mask=valid_bytes).to(tl.int32)
    groups = output_columns // GROUP_SIZE
    scale = tl.maximum(
        tl.load(scale_ptr + rows * n_groups + groups, mask=valid_bytes),
        1.0e-12,
    )
    output_offsets = rows * in_features + output_columns

    if BITS == 4:
        low = (packed & 0x0F) - 8
        high = ((packed >> 4) & 0x0F) - 8
        tl.store(
            output_ptr + output_offsets,
            low.to(tl.float32) * scale,
            mask=valid_bytes & (output_columns < in_features),
        )
        tl.store(
            output_ptr + output_offsets + 1,
            high.to(tl.float32) * scale,
            mask=valid_bytes & (output_columns + 1 < in_features),
        )
    else:
        code0 = (packed & 0x03) - 2
        code1 = ((packed >> 2) & 0x03) - 2
        code2 = ((packed >> 4) & 0x03) - 2
        code3 = ((packed >> 6) & 0x03) - 2
        tl.store(
            output_ptr + output_offsets,
            code0.to(tl.float32) * scale,
            mask=valid_bytes & (output_columns < in_features),
        )
        tl.store(
            output_ptr + output_offsets + 1,
            code1.to(tl.float32) * scale,
            mask=valid_bytes & (output_columns + 1 < in_features),
        )
        tl.store(
            output_ptr + output_offsets + 2,
            code2.to(tl.float32) * scale,
            mask=valid_bytes & (output_columns + 2 < in_features),
        )
        tl.store(
            output_ptr + output_offsets + 3,
            code3.to(tl.float32) * scale,
            mask=valid_bytes & (output_columns + 3 < in_features),
        )


def _launch(
    payload: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    group_size: int,
    in_features: int,
) -> torch.Tensor:
    out_features, payload_cols = payload.shape
    n_groups = triton.cdiv(in_features, group_size)
    output = torch.empty(
        (out_features, in_features),
        device=payload.device,
        dtype=torch.float32,
    )

    def grid(meta):
        return (triton.cdiv(out_features * payload_cols, meta["BLOCK_SIZE"]),)

    _unpack_int_grouped_kernel[grid](
        payload,
        scale,
        output,
        out_features=out_features,
        payload_cols=payload_cols,
        in_features=in_features,
        n_groups=n_groups,
        GROUP_SIZE=group_size,
        BITS=bits,
    )
    return output


class _UnpackIntGrouped(torch.autograd.Function):
    @staticmethod
    def forward(
        _ctx: Any,
        payload: torch.Tensor,
        scale: torch.Tensor,
        bits: int,
        group_size: int,
        in_features: int,
    ) -> torch.Tensor:
        return _launch(payload, scale, bits, group_size, in_features)

    @staticmethod
    def backward(_ctx: Any, _grad_output: torch.Tensor):
        return None, None, None, None, None


def _eligible() -> bool:
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


def _supported_inputs(
    payload: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    group_size: int,
    in_features: int,
) -> bool:
    if not (
        payload.is_cuda
        and scale.is_cuda
        and payload.device == scale.device
        and payload.dtype == torch.uint8
        and scale.dtype == torch.float32
        and payload.is_contiguous()
        and scale.is_contiguous()
        and payload.dim() == 2
        and scale.dim() == 2
        and payload.shape[0] == scale.shape[0]
        and payload.numel() > 0
        and not scale.requires_grad
        and (bits, group_size) in ((4, 128), (2, 64))
        and in_features > 0
    ):
        return False
    per_byte = 8 // bits
    n_groups = triton.cdiv(in_features, group_size)
    padded_features = n_groups * group_size
    return payload.shape[1] * per_byte == padded_features and scale.shape[1] == n_groups


def unpack_int_grouped(
    payload: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    group_size: int,
    in_features: int,
) -> torch.Tensor:
    """Decode and scale grouped W4/W2 weights, or safely use the oracle."""
    if not _supported_inputs(payload, scale, bits, group_size, in_features):
        return _reference.unpack_int_grouped_weight(
            payload,
            scale,
            bits,
            group_size,
            in_features,
        )
    return _UnpackIntGrouped.apply(payload, scale, bits, group_size, in_features)


def autotune_cache() -> list[dict[str, Any]]:
    """Return configurations selected for every packed format and shape."""
    return [
        {
            "out_features": key[0],
            "payload_columns": key[1],
            "bits": key[2],
            "block_size": config.kwargs["BLOCK_SIZE"],
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
        for key, config in _unpack_int_grouped_kernel.cache.items()
    ]


kernels.register(
    "unpack_int_grouped",
    unpack_int_grouped,
    name="triton",
    predicate=_eligible,
)


__all__ = ["autotune_cache", "unpack_int_grouped"]
