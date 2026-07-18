"""Quantization arithmetic primitives.

All functions here simulate low-precision arithmetic in float32 with
straight-through-estimator (STE) gradients, so the same code path serves
post-training quantization, quantization-aware training, and quantization-
aware distillation. Simulation is numerics-true: a validation score computed
through these ops is a real claim about the quantized arithmetic.

Formats:
- INT8: per-channel symmetric weights, per-tensor affine activations.
- NVFP4: E2M1 elements in 16-element blocks, one FP8 E4M3 scale per block,
  one FP32 scale per tensor. Implemented from the publicly documented format
  definition.
"""

from contextlib import nullcontext

import torch

# E2M1 representable magnitudes (sign handled separately): subnormal 0.5,
# then normals 1, 1.5, 2, 3, 4, 6.
_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Decision midpoints between consecutive levels for nearest-value rounding.
_E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

E2M1_MAX = 6.0
E4M3_MAX = 448.0
NVFP4_BLOCK = 16

_EPS = 1e-12


def autocast_off(device_type: str):
    """Context manager disabling autocast so quantized ops run in fp32."""
    if device_type in ("cuda", "cpu"):
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def _ste(x_fake: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward x_fake, backward identity to x."""
    return x + (x_fake - x).detach()


def quantize_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round a positive tensor to the FP8 E4M3 grid (returned as fp32)."""
    x = x.clamp(min=_EPS, max=E4M3_MAX)
    if hasattr(torch, "float8_e4m3fn"):
        return x.to(torch.float8_e4m3fn).to(torch.float32)
    return x


def fake_quant_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value. Input must already be scaled so the
    representable range is [-6, 6]. STE gradient."""
    midpoints = _E2M1_MIDPOINTS.to(device=x.device, dtype=x.dtype)
    levels = _E2M1_LEVELS.to(device=x.device, dtype=x.dtype)
    mag = x.abs().clamp(max=E2M1_MAX)
    idx = torch.bucketize(mag, midpoints)
    quant = levels[idx] * torch.sign(x)
    return _ste(quant, x)


def fake_quant_int8_per_channel(weight: torch.Tensor, ch_axis: int = 0) -> torch.Tensor:
    """Symmetric per-channel INT8 fake-quantization of a weight tensor."""
    dims = [d for d in range(weight.dim()) if d != ch_axis]
    amax = weight.abs().amax(dim=dims, keepdim=True).detach()
    scale = (amax / 127.0).clamp(min=_EPS)
    quant = torch.clamp(torch.round(weight / scale), -127, 127) * scale
    return _ste(quant, weight)


def fake_quant_int8_affine(
    x: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> torch.Tensor:
    """Per-tensor affine INT8 fake-quantization with calibrated [lo, hi]."""
    lo = torch.minimum(lo, torch.zeros_like(lo))
    hi = torch.maximum(hi, torch.zeros_like(hi))
    scale = ((hi - lo) / 255.0).clamp(min=_EPS)
    zero_point = torch.round(-128.0 - lo / scale)
    quant = torch.clamp(torch.round(x / scale) + zero_point, -128, 127)
    quant = (quant - zero_point) * scale
    return _ste(quant, x)


def _blockwise_nvfp4(x: torch.Tensor, tensor_scale: torch.Tensor) -> torch.Tensor:
    """NVFP4 fake-quantization along the last dim in blocks of 16.

    tensor_scale is the FP32 per-tensor scale; each 16-element block gets an
    E4M3 scale chosen so the block maximum maps to the E2M1 maximum (6).
    """
    orig_shape = x.shape
    last = orig_shape[-1]
    pad = (-last) % NVFP4_BLOCK
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // NVFP4_BLOCK, NVFP4_BLOCK)

    block_amax = blocks.abs().amax(dim=-1, keepdim=True).detach()
    tensor_scale = tensor_scale.clamp(min=_EPS)
    block_scale = quantize_e4m3(block_amax / (E2M1_MAX * tensor_scale))
    eff_scale = (block_scale * tensor_scale).clamp(min=_EPS)

    quant = fake_quant_e2m1(blocks / eff_scale) * eff_scale
    quant = quant.reshape(*x.shape)
    if pad:
        quant = quant[..., :last]
    return quant


def fake_quant_nvfp4_weight(
    weight: torch.Tensor,
    tensor_amax: torch.Tensor,
) -> torch.Tensor:
    """NVFP4 fake-quantization of a [out, in] weight with a fixed per-tensor
    amax captured at quantize time (so QAT trains against stable scaling)."""
    tensor_scale = tensor_amax / (E4M3_MAX * E2M1_MAX)
    return _blockwise_nvfp4(weight, tensor_scale)


def fake_quant_nvfp4_dynamic(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 fake-quantization of activations with dynamic per-tensor scale,
    blocked along the last (feature) dimension."""
    tensor_amax = x.abs().amax().detach()
    tensor_scale = tensor_amax / (E4M3_MAX * E2M1_MAX)
    return _blockwise_nvfp4(x, tensor_scale)
