"""Fused scaled-dot-product attention: the shared policy, not a new kernel.

``torch.nn.functional.scaled_dot_product_attention`` dispatches to the flash,
memory-efficient or cuDNN attention kernels instead of materialising the
``(heads, q, k)`` score matrix. Model families written as
``q @ k.T -> softmax -> @ v`` can hand their attention to it, but two rules
decide *when*:

- **Export never does.** LibreYOLO defaults to ONNX opset 13, which has no
  symbolic for fused SDPA, so every swapped call site keeps the primitive-op
  equation under ``torch.onnx.is_in_onnx_export()``.
- **Byte-exact parity bars keep manual math by default.** Several ports are
  pinned to ``max_abs_diff == 0`` against a reference that itself runs manual
  attention (the Swin and OWLv2 parity harnesses explicitly switch the
  reference's fused path *off* to get that). Fused kernels accumulate in a
  different order, so those families carry ``fused_attn = False`` and only
  switch when a caller opts in with :func:`set_fused_attention`. Families
  whose bar is a tolerance use SDPA by default and carry no flag.

Which family is which is recorded in the module docstring of each rewired
attention class, next to the parity bar it has to meet.
"""

from __future__ import annotations

from torch import nn

__all__ = ["fused_attention_modules", "set_fused_attention"]


def fused_attention_modules(module: nn.Module):
    """Yield every submodule carrying an opt-in ``fused_attn`` flag."""
    for candidate in module.modules():
        if isinstance(getattr(candidate, "fused_attn", None), bool):
            yield candidate


def set_fused_attention(module: nn.Module, enabled: bool = True) -> int:
    """Switch fused SDPA on or off across a model; returns how many flags moved.

    Trades byte-exact agreement with the family's upstream reference for the
    fused attention kernels. Returning zero means the model has no opt-in
    attention: either it already uses SDPA unconditionally, or it has no
    scaled-dot-product attention at all.
    """
    count = 0
    for attention in fused_attention_modules(module):
        attention.fused_attn = enabled
        count += 1
    return count
