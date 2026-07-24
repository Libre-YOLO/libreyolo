"""Fused Triton NVFP4 dynamic activation fake quantization."""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
import triton
import triton.language as tl

from libreyolo.quant import fake_quant as _reference
from libreyolo.quant import kernels

from .nvfp4_weight import _launch as _quantize_with_amax
from .nvfp4_weight import autotune_cache as _quantize_autotune_cache


_AMAX_CONFIGS = (
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=2),
)


@triton.autotune(configs=list(_AMAX_CONFIGS), key=["n_elements"])
@triton.jit
def _global_amax_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    local_amax = tl.max(tl.abs(values), axis=0)
    tl.atomic_max(output_ptr, local_amax, sem="relaxed")


def _global_amax(inputs: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(1, device=inputs.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(inputs.numel(), meta["BLOCK_SIZE"]),)

    _global_amax_kernel[grid](inputs, output, n_elements=inputs.numel())
    return output


def _launch(inputs: torch.Tensor) -> torch.Tensor:
    tensor_amax = _global_amax(inputs)
    matrix = inputs.reshape(-1, inputs.shape[-1])
    return _quantize_with_amax(matrix, tensor_amax).reshape_as(inputs)


class _FakeQuantNVFP4Dynamic(torch.autograd.Function):
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
        and inputs.dim() >= 1
        and inputs.is_contiguous()
        and inputs.numel() > 0
        and inputs.shape[-1] > 0
    )


def fake_quant_nvfp4_dynamic(inputs: torch.Tensor) -> torch.Tensor:
    """Apply dynamic NVFP4 fake quantization or safely use the oracle."""
    if not _supported_input(inputs):
        return _reference.fake_quant_nvfp4_dynamic(inputs)
    return _FakeQuantNVFP4Dynamic.apply(inputs)


def autotune_cache() -> dict[str, list[dict[str, Any]]]:
    """Return reduction and quantization configurations used by the gate."""
    reduction = [
        {
            "elements": key[0],
            "block_size": config.kwargs["BLOCK_SIZE"],
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
        for key, config in _global_amax_kernel.cache.items()
    ]
    return {"amax": reduction, "quantize": _quantize_autotune_cache()}


kernels.register(
    "fake_quant_nvfp4_dynamic",
    fake_quant_nvfp4_dynamic,
    name="triton",
    predicate=_eligible,
)


__all__ = ["autotune_cache", "fake_quant_nvfp4_dynamic"]
