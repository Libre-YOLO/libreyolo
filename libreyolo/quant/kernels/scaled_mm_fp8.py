"""Native FP8 linear via torch._scaled_mm (cuBLASLt fp8 tensor cores).

Runs the finalized fp8 ``QuantLinear`` GEMM directly on the fp8 tensor cores
(Ada sm_89 / Hopper sm_90 / Blackwell sm_100+/sm_120) instead of unpacking to
higher precision:

- weights: the checkpoint's packed E4M3 codes are consumed as-is by the GEMM;
  the per-channel dequant scale rides the epilogue (consumer parts only wire
  per-tensor scales into cuBLASLt).
- activations: one half-precision multiply by the cached inverse of the
  calibrated static scale, an explicit clamp to the E4M3 range (the torch fp8
  cast is NOT saturating), and the cast to E4M3 - the same grid snap the
  simulation performs, executed in three half-precision passes.
- output: bf16 from the GEMM (unbounded exponent range, half the bandwidth of
  fp32), then a single fused ``addcmul`` applies the per-channel weight scale
  and bias before casting back to the input dtype.

Accumulation is fp32 inside the tensor cores, like the simulation's fp32
island; residual drift vs the simulated tier is half-precision rounding in
the prologue/epilogue plus summation order. This op has no reference
implementation (GEMM slots never do): callers must check
``resolve("fp8_gemm")`` and fall back to the simulated path when it returns
None (CPU, pre-Ada GPUs, ``LIBREYOLO_QUANT_KERNELS=off``, or misfit shapes).
"""

from __future__ import annotations

from typing import Optional

import torch

from . import register

_E4M3_MAX = 448.0


def _supported() -> bool:
    if not torch.cuda.is_available() or not hasattr(torch, "_scaled_mm"):
        return False
    major, minor = torch.cuda.get_device_capability()
    # fp8 tensor cores: Ada (8.9), Hopper (9.x), Blackwell (10.x / 12.x).
    return (major, minor) >= (8, 9)


def make_aux(act_scale, w_scale, bias, device):
    """Precompute the per-module tensors the hot path consumes.

    Returns ``(scale_a fp32 scalar, ones fp32 scalar, inv_scale fp16 scalar,
    w_row bf16 [1, N], bias_row bf16 [1, N] | None)``.
    """
    scale_a = act_scale.reshape(()).to(device=device, dtype=torch.float32)
    one = torch.ones((), device=device, dtype=torch.float32)
    inv = (1.0 / scale_a).to(torch.float16)
    w_row = w_scale.float().reshape(1, -1).to(device=device, dtype=torch.bfloat16)
    b_row = (
        bias.float().reshape(1, -1).to(device=device, dtype=torch.bfloat16)
        if bias is not None
        else None
    )
    return (scale_a, one, inv, w_row, b_row)


def fp8_linear_scaled_mm(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    aux,
) -> Optional[torch.Tensor]:
    """out = (fp8(x/act_scale) @ packed.T) * act_scale * w_scale + bias."""
    K = x.shape[-1]
    N = weight_packed.shape[0]
    if K % 16 or N % 16:
        return None
    scale_a, one, inv, w_row, b_row = aux
    x2 = x.reshape(-1, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if x2.dtype != torch.float16:
        x2 = x2.to(torch.float16)

    x8 = (x2 * inv).clamp_(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    try:
        out = torch._scaled_mm(
            x8,
            weight_packed.t(),  # [K, N] column-major view, no copy
            scale_a=scale_a,
            scale_b=one,
            out_dtype=torch.bfloat16,
        )
    except RuntimeError:
        return None  # layout/shape rejected by cuBLASLt -> simulated fallback

    out = torch.addcmul(b_row, out, w_row) if b_row is not None else out * w_row
    return out.to(x.dtype).reshape(*x.shape[:-1], N)


register("fp8_gemm", fp8_linear_scaled_mm, name="scaled_mm", predicate=_supported)
