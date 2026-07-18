"""Quantized module variants.

Each class subclasses its float counterpart so the module keeps the same
state_dict key layout (``weight`` / ``bias`` stay where checkpoints expect
them) and remains a drop-in inside existing architectures, trainers, and
optimizers. Quantization state lives in extra ``_q_*`` buffers so it
round-trips through checkpoints.

Weights are fake-quantized from the fp32 master copy on every forward, which
keeps the modules correct for QAT (gradients flow to the masters through the
straight-through estimator). Computation runs in an fp32 island even under
autocast so the simulated arithmetic is exactly the declared scheme.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fake_quant import (
    autocast_off,
    fake_quant_int8_affine,
    fake_quant_int8_per_channel,
    fake_quant_nvfp4_dynamic,
    fake_quant_nvfp4_weight,
)


class _ActObserverMixin:
    """Shared INT8 activation observer / fake-quant state."""

    def _init_act_state(self):
        self.register_buffer("_q_act_lo", torch.zeros(1))
        self.register_buffer("_q_act_hi", torch.zeros(1))
        self.register_buffer("_q_calibrated", torch.zeros(1, dtype=torch.uint8))
        self._q_observing = False

    @property
    def q_calibrated(self) -> bool:
        return bool(self._q_calibrated.item())

    def _observe(self, x: torch.Tensor):
        with torch.no_grad():
            lo = x.amin().float().reshape(1)
            hi = x.amax().float().reshape(1)
            if self.q_calibrated:
                torch.minimum(self._q_act_lo, lo, out=self._q_act_lo)
                torch.maximum(self._q_act_hi, hi, out=self._q_act_hi)
            else:
                self._q_act_lo.copy_(lo)
                self._q_act_hi.copy_(hi)
                self._q_calibrated.fill_(1)

    def _maybe_quant_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._q_observing:
            self._observe(x)
            return x
        if self.q_calibrated:
            return fake_quant_int8_affine(x, self._q_act_lo, self._q_act_hi)
        return x


class QuantConv2d(nn.Conv2d, _ActObserverMixin):
    """INT8 W8A8 simulated convolution (per-channel weights, affine input)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_act_state()

    @classmethod
    def from_float(cls, conv: nn.Conv2d) -> "QuantConv2d":
        mod = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        mod.weight = conv.weight
        mod.bias = conv.bias
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        with autocast_off(x.device.type):
            x = x.float()
            x = self._maybe_quant_input(x)
            weight = fake_quant_int8_per_channel(self.weight.float())
            bias = self.bias.float() if self.bias is not None else None
            out = self._conv_forward(x, weight, bias)
        return out.to(in_dtype) if in_dtype != out.dtype else out


class QuantLinear(nn.Linear, _ActObserverMixin):
    """INT8 W8A8 simulated linear (per-channel weights, affine input)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_act_state()

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "QuantLinear":
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        mod.weight = linear.weight
        mod.bias = linear.bias
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        with autocast_off(x.device.type):
            x = x.float()
            x = self._maybe_quant_input(x)
            weight = fake_quant_int8_per_channel(self.weight.float())
            bias = self.bias.float() if self.bias is not None else None
            out = F.linear(x, weight, bias)
        return out.to(in_dtype) if in_dtype != out.dtype else out


class NVFP4Linear(nn.Linear):
    """NVFP4 W4A4 simulated linear.

    Weights: E2M1 in 16-element blocks with E4M3 block scales and a fixed
    fp32 per-tensor scale captured at quantize time. Activations: dynamically
    scaled per forward (two-level scaling), so no calibration is required.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_q_w_amax", torch.zeros(1))
        self._q_observing = False  # accepted for API symmetry; unused

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "NVFP4Linear":
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        mod.weight = linear.weight
        mod.bias = linear.bias
        with torch.no_grad():
            mod._q_w_amax.copy_(linear.weight.detach().abs().amax().float().reshape(1))
        return mod

    @property
    def q_calibrated(self) -> bool:
        return bool((self._q_w_amax > 0).item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        with autocast_off(x.device.type):
            x = fake_quant_nvfp4_dynamic(x.float())
            weight = fake_quant_nvfp4_weight(self.weight.float(), self._q_w_amax)
            bias = self.bias.float() if self.bias is not None else None
            out = F.linear(x, weight, bias)
        return out.to(in_dtype) if in_dtype != out.dtype else out


QUANT_MODULE_TYPES = (QuantConv2d, QuantLinear, NVFP4Linear)
