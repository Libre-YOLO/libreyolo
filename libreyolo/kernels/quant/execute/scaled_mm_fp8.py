"""Native FP8 linear via torch._scaled_mm (cuBLASLt fp8 tensor cores).

Runs the finalized fp8 ``QuantLinear`` GEMM directly on the fp8 tensor cores
(Ada sm_89 / Hopper sm_90 / Blackwell sm_100+/sm_120) instead of unpacking to
higher precision:

- weights: the checkpoint's packed E4M3 codes are consumed as-is by the GEMM.
  Per-channel checkpoints apply row scales in an epilogue; explicitly marked
  tensorwise modules fuse the weight scale and bias into cuBLASLt.
- activations: fp32 scaling by the cached inverse calibrated scale, an
  explicit clamp to the E4M3 range (the torch fp8 cast is NOT saturating), and
  the cast to E4M3. Triton fuses these into one pass; the portable expression
  materializes the fp32 intermediate.
- output: per-channel modules fold a safe base scale into an fp16 GEMM and,
  when Triton is available, restore the bounded row-scale ratios and bias in
  one fused pass. The portable path retains the bf16 GEMM plus stock PyTorch
  epilogue. Tensorwise modules fuse the complete fp16 epilogue in cuBLASLt.

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

from libreyolo.kernels import register

_E4M3_MAX = 448.0


def _supported() -> bool:
    if not torch.cuda.is_available() or not hasattr(torch, "_scaled_mm"):
        return False
    major, minor = torch.cuda.get_device_capability()
    # fp8 tensor cores: Ada (8.9), Hopper (9.x), Blackwell (10.x / 12.x).
    return (major, minor) >= (8, 9)


def make_aux(act_scale, w_scale, bias, device, *, tensorwise: bool = False):
    """Precompute the per-module tensors the hot path consumes.

    The first element is always ``scale_a`` so callers can cheaply validate
    the cache device. Remaining entries depend on the scaling mode.
    """
    scale_a = act_scale.reshape(()).to(device=device, dtype=torch.float32)
    inv = (1.0 / scale_a).to(torch.float16)
    if tensorwise:
        scale_b = w_scale[0].reshape(()).to(device=device, dtype=torch.float32)
        bias_16 = (
            bias.to(device=device, dtype=torch.float16) if bias is not None else None
        )
        return (scale_a, inv, scale_b, bias_16)
    # Consumer cuBLASLt cannot consume rowwise scale_b. Fold the largest row
    # scale into the GEMM so its FP16 output stays in the real weight's range,
    # then restore each row with a bounded ratio in one fused epilogue.
    w_scale_device = w_scale.float().to(device)
    base_scale = w_scale_device.amax().clamp_min(1e-12).reshape(())
    w_ratio = (w_scale_device / base_scale).reshape(1, -1).to(torch.float16)
    bias_16 = (
        bias.float().reshape(1, -1).to(device=device, dtype=torch.float16)
        if bias is not None
        else None
    )
    one = torch.ones((), device=device, dtype=torch.float32)
    w_row_bf16 = w_scale_device.reshape(1, -1).to(torch.bfloat16)
    bias_bf16 = (
        bias.float().reshape(1, -1).to(device=device, dtype=torch.bfloat16)
        if bias is not None
        else None
    )
    return (
        scale_a,
        inv,
        base_scale,
        w_ratio,
        bias_16,
        one,
        w_row_bf16,
        bias_bf16,
    )


def _cast_static_fp8(x: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    # The optional Triton implementation fuses multiply, saturation, and cast
    # into one memory pass. Stock torch remains the portable fallback.
    from libreyolo.kernels import resolve

    impl = resolve("fp8_cast_static")
    if impl is not None:
        return impl(x, inv)
    return (
        (x.float() * inv.float())
        .clamp_(-_E4M3_MAX, _E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )


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
    (
        scale_a,
        inv,
        base_scale,
        w_ratio,
        bias,
        one,
        w_row_bf16,
        bias_bf16,
    ) = aux
    x2 = x.reshape(-1, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if x2.dtype != torch.float16:
        x2 = x2.to(torch.float16)

    x8 = _cast_static_fp8(x2, inv)
    from libreyolo.kernels import resolve

    epilogue = resolve("fp8_perchannel_epilogue")
    try:
        out = torch._scaled_mm(
            x8,
            weight_packed.t(),  # [K, N] column-major view, no copy
            scale_a=scale_a,
            scale_b=base_scale if epilogue is not None else one,
            out_dtype=torch.float16 if epilogue is not None else torch.bfloat16,
        )
    except RuntimeError:
        return None  # layout/shape rejected by cuBLASLt -> simulated fallback

    if epilogue is not None:
        out = epilogue(out, w_ratio, bias)
    else:
        out = (
            torch.addcmul(bias_bf16, out, w_row_bf16)
            if bias_bf16 is not None
            else out * w_row_bf16
        )
    return out.to(x.dtype).reshape(*x.shape[:-1], N)


def fp8_linear_tensorwise_scaled_mm(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    aux,
) -> Optional[torch.Tensor]:
    """FP8 GEMM with tensorwise weight scale and fused FP16 epilogue."""
    K = x.shape[-1]
    N = weight_packed.shape[0]
    if K % 16 or N % 16:
        return None
    scale_a, inv, scale_b, bias = aux
    x2 = x.reshape(-1, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if x2.dtype != torch.float16:
        x2 = x2.to(torch.float16)
    x8 = _cast_static_fp8(x2, inv)
    try:
        out = torch._scaled_mm(
            x8,
            weight_packed.t(),
            scale_a=scale_a,
            scale_b=scale_b,
            bias=bias,
            out_dtype=torch.float16,
        )
    except RuntimeError:
        return None
    return out.to(x.dtype).reshape(*x.shape[:-1], N)


register("fp8_gemm", fp8_linear_scaled_mm, name="scaled_mm", predicate=_supported)
register(
    "fp8_gemm_tensorwise",
    fp8_linear_tensorwise_scaled_mm,
    name="scaled_mm",
    predicate=_supported,
)
