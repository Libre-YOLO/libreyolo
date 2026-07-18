"""Triton quantization kernels (JIT, no build step)."""

from .nvfp4_weight import fake_quant_nvfp4_weight


__all__ = ["fake_quant_nvfp4_weight"]
