"""Attention kernels shared across model families."""

from .sdpa import fused_attention_modules, set_fused_attention

__all__ = ["fused_attention_modules", "set_fused_attention"]
