"""Packed storage for finalized quantized checkpoints.

The invariant that keeps finalized checkpoints explainable: unpacking a
packed weight reproduces bit for bit the values the fake-quant simulation
computes from the fp32 masters. Finalized inference is therefore numerically
identical to prepared inference; only the storage changes.

Layouts (documented for external consumers in docs/checkpoint_schema.md):

- int8: ``weight_packed`` int8 tensor in the original weight shape, plus the
  per-channel fp32 scale in ``_q_w_scale``. Dequant: ``packed * scale``.
- nvfp4: ``weight_packed`` uint8 tensor of shape [out, ceil(in/16)*8], two
  4-bit codes per byte (low nibble first; code = sign<<3 | level index into
  the E2M1 magnitude table), ``weight_block_scale`` float8_e4m3fn tensor of
  shape [out, ceil(in/16)], and the fp32 per-tensor amax in ``_q_w_amax``.
  Effective block scale: ``block_scale * amax / (448 * 6)``.
"""

import torch

from .fake_quant import (
    _E2M1_LEVELS,
    _E2M1_MIDPOINTS,
    E2M1_MAX,
    E4M3_MAX,
    NVFP4_BLOCK,
    quantize_e4m3,
)

_EPS = 1e-12


def _scale_view(scale: torch.Tensor, ndim: int) -> torch.Tensor:
    return scale.reshape(-1, *([1] * (ndim - 1)))


def pack_int8_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize an fp32 weight onto the int8 grid used by the simulation.

    Routes through torch.fake_quantize so the integer codes come from the
    exact same kernel arithmetic the simulation uses (CUDA's fast division
    can differ from a plain divide by one code at rounding boundaries);
    unpacking therefore reproduces the simulation bit for bit on every
    device.
    """
    weight = weight.float()
    scale = scale.float().clamp(min=_EPS)
    zero_point = torch.zeros(scale.shape, dtype=torch.int32, device=weight.device)
    fq = torch.fake_quantize_per_channel_affine(
        weight, scale, zero_point, 0, -128, 127
    )
    s = _scale_view(scale, weight.dim())
    return torch.round(fq / s).to(torch.int8)


def unpack_int8_weight(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize; bit-identical to fake_quant_int8_per_channel(masters)."""
    s = _scale_view(scale.float(), packed.dim()).clamp(min=_EPS)
    return packed.float() * s


def _nvfp4_eff_scales(weight2d: torch.Tensor, tensor_amax: torch.Tensor):
    """Blocked view plus the (e4m3 block scales, effective scales) pair."""
    out_features, in_features = weight2d.shape
    pad = (-in_features) % NVFP4_BLOCK
    if pad:
        weight2d = torch.nn.functional.pad(weight2d, (0, pad))
    blocks = weight2d.reshape(out_features, -1, NVFP4_BLOCK)
    tensor_scale = (tensor_amax.float() / (E4M3_MAX * E2M1_MAX)).clamp(min=_EPS)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    block_scale = quantize_e4m3(block_amax / (E2M1_MAX * tensor_scale))
    eff = (block_scale * tensor_scale).clamp(min=_EPS)
    return blocks, block_scale.squeeze(-1), eff


def pack_nvfp4_weight(weight: torch.Tensor, tensor_amax: torch.Tensor):
    """Pack an fp32 [out, in] weight into (payload uint8, block scales f8)."""
    blocks, block_scale, eff = _nvfp4_eff_scales(weight.float(), tensor_amax)
    scaled = blocks / eff
    midpoints = _E2M1_MIDPOINTS.to(device=weight.device, dtype=torch.float32)
    idx = torch.bucketize(scaled.abs().clamp(max=E2M1_MAX), midpoints).to(torch.uint8)
    sign = (scaled < 0).to(torch.uint8)
    codes = (idx | (sign << 3)).reshape(weight.shape[0], -1)  # [out, padded_in]
    payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    if hasattr(torch, "float8_e4m3fn"):
        block_scale = block_scale.to(torch.float8_e4m3fn)
    return payload.contiguous(), block_scale.contiguous()


def unpack_nvfp4_weight(
    payload: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_amax: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    """Dequantize; bit-identical to fake_quant_nvfp4_weight(masters)."""
    out_features = payload.shape[0]
    codes = torch.empty(
        (out_features, payload.shape[1] * 2), dtype=torch.uint8, device=payload.device
    )
    codes[:, 0::2] = payload & 0x0F
    codes[:, 1::2] = payload >> 4
    idx = (codes & 0x07).long()
    sign = torch.where((codes & 0x08).bool(), -1.0, 1.0)
    levels = _E2M1_LEVELS.to(device=payload.device, dtype=torch.float32)
    values = levels[idx] * sign

    tensor_scale = (tensor_amax.float() / (E4M3_MAX * E2M1_MAX)).clamp(min=_EPS)
    eff = (block_scale.float() * tensor_scale).clamp(min=_EPS)
    blocks = values.reshape(out_features, -1, NVFP4_BLOCK) * eff.unsqueeze(-1)
    return blocks.reshape(out_features, -1)[:, :in_features].contiguous()
