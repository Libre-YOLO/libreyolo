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

from . import kernels as K
from .fake_quant import (
    E4M3_MAX,
    autocast_off,
    fp8_weight_qparams,
    int8_weight_qparams,
    int_group_qparams,
)
from .packing import (
    pack_fp8_weight,
    pack_int8_weight,
    pack_int_grouped_weight,
    pack_mxfp4_weight,
    pack_nvfp4_weight,
    unpack_fp8_weight,
    unpack_int8_weight,
    unpack_int_grouped_weight,
    unpack_mxfp4_weight,
    unpack_nvfp4_weight,
)


class _ActObserverMixin:
    """Shared INT8 activation observer / fake-quant state."""

    def _init_act_state(self, out_channels: int):
        # Buffers must live on the weight's device from the start: modules are
        # built with device=weight.device by from_float, and a CPU-born buffer
        # would survive until the post-quantize .to() call, crashing
        # multi-batch calibration on GPU.
        device = self.weight.device
        self.register_buffer("_q_act_lo", torch.zeros(1, device=device))
        self.register_buffer("_q_act_hi", torch.zeros(1, device=device))
        self.register_buffer(
            "_q_calibrated", torch.zeros(1, dtype=torch.uint8, device=device)
        )
        self.register_buffer("_q_w_scale", torch.zeros(out_channels, device=device))
        self._q_observing = False
        self._q_export_mode = False

    def freeze_weight_qparams(self):
        """Bake the current per-channel weight scales into a buffer so ONNX
        tracing emits them as constants (export mode)."""
        with torch.no_grad():
            if getattr(self, "_q_wformat", "int8") == "fp8":
                self._q_w_scale.copy_(fp8_weight_qparams(self.weight.float()))
            else:
                self._q_w_scale.copy_(int8_weight_qparams(self.weight.float()))

    def _weight_scale(self):
        return self._q_w_scale if self._q_export_mode else None

    @property
    def is_finalized(self) -> bool:
        return "weight_packed" in self._buffers

    def make_finalized(self):
        """Crystallize: replace the fp32 master weight with packed low bits.

        Unpacking reproduces the simulation's weight values bit for bit, so
        finalized inference matches prepared inference exactly.
        """
        if self.is_finalized:
            return
        self.freeze_weight_qparams()
        with torch.no_grad():
            if getattr(self, "_q_wformat", "int8") == "fp8":
                packed = pack_fp8_weight(self.weight.detach(), self._q_w_scale)
            else:
                packed = pack_int8_weight(self.weight.detach(), self._q_w_scale)
        del self._parameters["weight"]
        self.register_buffer("weight_packed", packed)
        self._q_export_mode = False

    def make_prepared(self):
        """Reconstruct fp32 masters from packed weights (QAT-from-PTQ)."""
        if not self.is_finalized:
            return
        with torch.no_grad():
            if getattr(self, "_q_wformat", "int8") == "fp8":
                weight = unpack_fp8_weight(self.weight_packed, self._q_w_scale)
            else:
                weight = unpack_int8_weight(self.weight_packed, self._q_w_scale)
        del self._buffers["weight_packed"]
        self.weight = nn.Parameter(weight)

    def _effective_weight(self) -> torch.Tensor:
        fp8 = getattr(self, "_q_wformat", "int8") == "fp8"
        if self.is_finalized:
            if fp8:
                return unpack_fp8_weight(self.weight_packed, self._q_w_scale)
            return unpack_int8_weight(self.weight_packed, self._q_w_scale)
        if fp8:
            return K.fake_quant_fp8(
                self.weight.float(),
                fp8_weight_qparams(self.weight.float()).reshape(
                    -1, *([1] * (self.weight.dim() - 1))
                ),
            )
        return K.fake_quant_int8_per_channel(
            self.weight.float(), scale=self._weight_scale()
        )

    @property
    def q_calibrated(self) -> bool:
        return bool(self._q_calibrated.item())

    _Q_PERCENTILE = 0.999
    _Q_OBS_SAMPLE_CAP = 1_000_000

    def _observe(self, x: torch.Tensor):
        with torch.no_grad():
            if getattr(self, "_q_obs_method", "minmax") == "percentile":
                # Mean of per-batch (0.1%, 99.9%) quantiles: robust to the
                # activation outliers that make pure min/max ranges crush
                # int8 resolution (worst on the smallest models). Strided
                # subsampling keeps torch.quantile within its size limits.
                flat = x.detach().reshape(-1).float()
                if flat.numel() > self._Q_OBS_SAMPLE_CAP:
                    step = (flat.numel() + self._Q_OBS_SAMPLE_CAP - 1) // self._Q_OBS_SAMPLE_CAP
                    flat = flat[::step]
                qs = torch.quantile(
                    flat,
                    torch.tensor(
                        [1.0 - self._Q_PERCENTILE, self._Q_PERCENTILE],
                        device=flat.device,
                    ),
                )
                self._q_obs_lo_sum = getattr(self, "_q_obs_lo_sum", 0.0) + float(qs[0])
                self._q_obs_hi_sum = getattr(self, "_q_obs_hi_sum", 0.0) + float(qs[1])
                self._q_obs_n = getattr(self, "_q_obs_n", 0) + 1
                return
            lo = x.amin().float().reshape(1).to(self._q_act_lo.device)
            hi = x.amax().float().reshape(1).to(self._q_act_hi.device)
            if self.q_calibrated:
                torch.minimum(self._q_act_lo, lo, out=self._q_act_lo)
                torch.maximum(self._q_act_hi, hi, out=self._q_act_hi)
            else:
                self._q_act_lo.copy_(lo)
                self._q_act_hi.copy_(hi)
                self._q_calibrated.fill_(1)

    def finalize_observation(self):
        """Write percentile statistics into the calibrated range buffers."""
        n = getattr(self, "_q_obs_n", 0)
        if n:
            with torch.no_grad():
                self._q_act_lo.fill_(self._q_obs_lo_sum / n)
                self._q_act_hi.fill_(self._q_obs_hi_sum / n)
                self._q_calibrated.fill_(1)
        for attr in ("_q_obs_lo_sum", "_q_obs_hi_sum", "_q_obs_n"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _maybe_quant_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._q_observing:
            self._observe(x)
            return x
        if not self.q_calibrated:
            return x
        if getattr(self, "_q_aformat", "int8") == "fp8":
            amax = torch.maximum(self._q_act_lo.abs(), self._q_act_hi.abs())
            return K.fake_quant_fp8(x, (amax / E4M3_MAX).reshape(1))
        return K.fake_quant_int8_affine(
            x,
            self._q_act_lo,
            self._q_act_hi,
            scalars=getattr(self, "_q_export_mode", False),
        )


class QuantConv2d(nn.Conv2d, _ActObserverMixin):
    """INT8 W8A8 simulated convolution (per-channel weights, affine input)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_act_state(self.out_channels)

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
            weight = self._effective_weight()
            bias = self.bias.float() if self.bias is not None else None
            out = self._conv_forward(x, weight, bias)
        return out.to(in_dtype) if in_dtype != out.dtype else out


class QuantLinear(nn.Linear, _ActObserverMixin):
    """INT8 W8A8 simulated linear (per-channel weights, affine input)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_act_state(self.out_features)

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
            weight = self._effective_weight()
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
        self.register_buffer("_q_w_amax", torch.zeros(1, device=self.weight.device))
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

    @property
    def is_finalized(self) -> bool:
        return "weight_packed" in self._buffers

    def make_finalized(self):
        """Crystallize: replace fp32 masters with packed E2M1 + E4M3 scales."""
        if self.is_finalized:
            return
        with torch.no_grad():
            payload, block_scale = pack_nvfp4_weight(
                self.weight.detach(), self._q_w_amax
            )
        del self._parameters["weight"]
        self.register_buffer("weight_packed", payload)
        self.register_buffer("weight_block_scale", block_scale)

    def make_prepared(self):
        """Reconstruct fp32 masters from packed weights (QAT-from-PTQ)."""
        if not self.is_finalized:
            return
        with torch.no_grad():
            weight = unpack_nvfp4_weight(
                self.weight_packed,
                self.weight_block_scale,
                self._q_w_amax,
                self.in_features,
            )
        del self._buffers["weight_packed"]
        del self._buffers["weight_block_scale"]
        self.weight = nn.Parameter(weight)

    def _effective_weight(self) -> torch.Tensor:
        if self.is_finalized:
            # Registry slot: accelerated unpack when available, packing
            # reference otherwise (identical values either way).
            return K.unpack_nvfp4(
                self.weight_packed,
                self.weight_block_scale,
                self._q_w_amax,
                self.in_features,
            )
        return K.fake_quant_nvfp4_weight(self.weight.float(), self._q_w_amax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        with autocast_off(x.device.type):
            x = K.fake_quant_nvfp4_dynamic(x.float())
            weight = self._effective_weight()
            bias = self.bias.float() if self.bias is not None else None
            out = F.linear(x, weight, bias)
        return out.to(in_dtype) if in_dtype != out.dtype else out


class GroupQuantLinear(nn.Linear, _ActObserverMixin):
    """Grouped symmetric integer weight quantization (W4/W2 style linears).

    Weights: per-group symmetric int codes (``_q_bits`` in {2, 4}, group size
    along in_features). Activations: optional calibrated INT8 (W4A8/W2A8) or
    untouched float (W4A16 style).
    """

    def __init__(self, *args, bits: int = 4, group_size: int = 128, act_int8: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._q_bits = int(bits)
        self._q_group = int(group_size)
        self._q_act_enabled = bool(act_int8)
        self._init_act_state(self.out_features)
        ngroups = (self.in_features + self._q_group - 1) // self._q_group
        self.register_buffer(
            "_q_w_gscale",
            torch.zeros(self.out_features, ngroups, device=self.weight.device),
        )

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        act_int8: bool = False,
    ) -> "GroupQuantLinear":
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            bits=bits,
            group_size=group_size,
            act_int8=act_int8,
        )
        mod.weight = linear.weight
        mod.bias = linear.bias
        return mod

    def freeze_weight_qparams(self):
        with torch.no_grad():
            self._q_w_gscale.copy_(
                int_group_qparams(self.weight.float(), self._q_bits, self._q_group)
            )

    def make_finalized(self):
        if self.is_finalized:
            return
        self.freeze_weight_qparams()
        with torch.no_grad():
            payload, _ = pack_int_grouped_weight(
                self.weight.detach().float(),
                self._q_bits,
                self._q_group,
                scale=self._q_w_gscale,
            )
        del self._parameters["weight"]
        self.register_buffer("weight_packed", payload)

    def make_prepared(self):
        if not self.is_finalized:
            return
        with torch.no_grad():
            weight = unpack_int_grouped_weight(
                self.weight_packed,
                self._q_w_gscale,
                self._q_bits,
                self._q_group,
                self.in_features,
            )
        del self._buffers["weight_packed"]
        self.weight = nn.Parameter(weight)

    def _effective_weight(self) -> torch.Tensor:
        if self.is_finalized:
            # Registry slot: accelerated unpack when available, packing
            # reference otherwise (identical values either way).
            return K.unpack_int_grouped(
                self.weight_packed,
                self._q_w_gscale,
                self._q_bits,
                self._q_group,
                self.in_features,
            )
        return K.fake_quant_int_grouped(
            self.weight.float(), self._q_bits, self._q_group
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        with autocast_off(x.device.type):
            x = x.float()
            if self._q_act_enabled or self._q_observing:
                x = self._maybe_quant_input(x)
            weight = self._effective_weight()
            bias = self.bias.float() if self.bias is not None else None
            out = F.linear(x, weight, bias)
        return out.to(in_dtype) if in_dtype != out.dtype else out


class MXFP4Linear(nn.Linear):
    """OCP MXFP4 W4A4 simulated linear: E2M1 elements, 32-element blocks,
    power-of-two (E8M0) block scales, dynamic activation scaling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._q_observing = False  # API symmetry; unused

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "MXFP4Linear":
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

    @property
    def q_calibrated(self) -> bool:
        return True  # dynamic activation scaling: nothing to calibrate

    @property
    def is_finalized(self) -> bool:
        return "weight_packed" in self._buffers

    def make_finalized(self):
        if self.is_finalized:
            return
        with torch.no_grad():
            payload, exponents = pack_mxfp4_weight(self.weight.detach().float())
        del self._parameters["weight"]
        self.register_buffer("weight_packed", payload)
        self.register_buffer("weight_block_exp", exponents)

    def make_prepared(self):
        if not self.is_finalized:
            return
        with torch.no_grad():
            weight = unpack_mxfp4_weight(
                self.weight_packed, self.weight_block_exp, self.in_features
            )
        del self._buffers["weight_packed"]
        del self._buffers["weight_block_exp"]
        self.weight = nn.Parameter(weight)

    def _effective_weight(self) -> torch.Tensor:
        if self.is_finalized:
            return unpack_mxfp4_weight(
                self.weight_packed, self.weight_block_exp, self.in_features
            )
        return K.fake_quant_mxfp4_weight(self.weight.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        with autocast_off(x.device.type):
            x = K.fake_quant_mxfp4_dynamic(x.float())
            weight = self._effective_weight()
            bias = self.bias.float() if self.bias is not None else None
            out = F.linear(x, weight, bias)
        return out.to(in_dtype) if in_dtype != out.dtype else out


QUANT_MODULE_TYPES = (
    QuantConv2d,
    QuantLinear,
    NVFP4Linear,
    MXFP4Linear,
    GroupQuantLinear,
)
