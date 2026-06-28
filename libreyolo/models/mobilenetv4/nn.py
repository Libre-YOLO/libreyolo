"""MobileNetV4 (conv variants) — native LibreYOLO implementation.

The architecture and the per-variant block definitions are vendored from timm's
MobileNetV4 (``timm/models/mobilenetv3.py`` + ``_efficientnet_blocks.py`` +
``_efficientnet_builder.py``, Apache-2.0) and trimmed to the *conv-only* block
set used by ``mobilenetv4_conv_{small,medium,large}``:

    ConvBnAct ('cn'), EdgeResidual ('er'), UniversalInvertedResidual ('uir')

No attention (``mqa``) blocks, no squeeze-excite, ReLU throughout. Module and
attribute names mirror timm exactly so a timm pretrained ``state_dict`` loads
into this module with ``strict=True`` and produces bit-identical outputs (this
is enforced by the parity test). See the family NOTICE for attribution.

Upstream paper: "MobileNetV4 - Universal Models for the Mobile Ecosystem"
(https://arxiv.org/abs/2404.10518).
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Architecture definitions — copied verbatim from timm ``_gen_mobilenet_v4``.
# Each variant: stem channel count + a list of stages; each stage is a list of
# block-definition strings in timm's notation (see ``_decode_block_str``).
# --------------------------------------------------------------------------- #
ARCH_DEFS: Dict[str, Dict] = {
    "s": {
        "stem": 32,
        "stages": [
            ["cn_r1_k3_s2_e1_c32", "cn_r1_k1_s1_e1_c32"],
            ["cn_r1_k3_s2_e1_c96", "cn_r1_k1_s1_e1_c64"],
            ["uir_r1_a5_k5_s2_e3_c96", "uir_r4_a0_k3_s1_e2_c96", "uir_r1_a3_k0_s1_e4_c96"],
            [
                "uir_r1_a3_k3_s2_e6_c128",
                "uir_r1_a5_k5_s1_e4_c128",
                "uir_r1_a0_k5_s1_e4_c128",
                "uir_r1_a0_k5_s1_e3_c128",
                "uir_r2_a0_k3_s1_e4_c128",
            ],
            ["cn_r1_k1_s1_c960"],
        ],
    },
    "m": {
        "stem": 32,
        "stages": [
            ["er_r1_k3_s2_e4_c48"],
            ["uir_r1_a3_k5_s2_e4_c80", "uir_r1_a3_k3_s1_e2_c80"],
            [
                "uir_r1_a3_k5_s2_e6_c160",
                "uir_r2_a3_k3_s1_e4_c160",
                "uir_r1_a3_k5_s1_e4_c160",
                "uir_r1_a3_k3_s1_e4_c160",
                "uir_r1_a3_k0_s1_e4_c160",
                "uir_r1_a0_k0_s1_e2_c160",
                "uir_r1_a3_k0_s1_e4_c160",
            ],
            [
                "uir_r1_a5_k5_s2_e6_c256",
                "uir_r1_a5_k5_s1_e4_c256",
                "uir_r2_a3_k5_s1_e4_c256",
                "uir_r1_a0_k0_s1_e4_c256",
                "uir_r1_a3_k0_s1_e4_c256",
                "uir_r1_a3_k5_s1_e2_c256",
                "uir_r1_a5_k5_s1_e4_c256",
                "uir_r2_a0_k0_s1_e4_c256",
                "uir_r1_a5_k0_s1_e2_c256",
            ],
            ["cn_r1_k1_s1_c960"],
        ],
    },
    "l": {
        "stem": 24,
        "stages": [
            ["er_r1_k3_s2_e4_c48"],
            ["uir_r1_a3_k5_s2_e4_c96", "uir_r1_a3_k3_s1_e4_c96"],
            [
                "uir_r1_a3_k5_s2_e4_c192",
                "uir_r3_a3_k3_s1_e4_c192",
                "uir_r1_a3_k5_s1_e4_c192",
                "uir_r5_a5_k3_s1_e4_c192",
                "uir_r1_a3_k0_s1_e4_c192",
            ],
            [
                "uir_r4_a5_k5_s2_e4_c512",
                "uir_r1_a5_k0_s1_e4_c512",
                "uir_r1_a5_k3_s1_e4_c512",
                "uir_r2_a5_k0_s1_e4_c512",
                "uir_r1_a5_k3_s1_e4_c512",
                "uir_r1_a5_k5_s1_e4_c512",
                "uir_r3_a5_k0_s1_e4_c512",
            ],
            ["cn_r1_k1_s1_c960"],
        ],
    },
}

HEAD_HIDDEN_SIZE = 1280  # conv_head output channels (timm num_features)


# --------------------------------------------------------------------------- #
# timm helpers (vendored verbatim for exactness).
# --------------------------------------------------------------------------- #
def make_divisible(v, divisor: int = 8, min_value=None, round_limit: float = 0.9) -> int:
    """timm ``make_divisible`` — round channels to a multiple of ``divisor``."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def round_channels(channels, multiplier: float = 1.0, divisor: int = 8) -> int:
    """timm ``round_channels`` with multiplier 1.0 (conv variants)."""
    if not multiplier:
        return channels
    return make_divisible(channels * multiplier, divisor)


def _pad(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    """timm ``get_padding`` for the default (symmetric) pad mode."""
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


# --------------------------------------------------------------------------- #
# Block modules. Submodule names mirror timm so state dicts map 1:1.
# --------------------------------------------------------------------------- #
class ConvNormAct(nn.Module):
    """Conv -> BatchNorm -> (optional ReLU). Submodules: ``conv``, ``bn``."""

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, groups=1, apply_act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride,
            padding=_pad(kernel_size, stride), groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chs)
        self.apply_act = apply_act

    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.relu(x, inplace=True) if self.apply_act else x


class ConvBnAct(nn.Module):
    """timm ``ConvBnAct`` ('cn'). Submodules: ``conv``, ``bn1``."""

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, skip=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride,
            padding=_pad(kernel_size, stride), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.has_skip = skip and stride == 1 and in_chs == out_chs

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn1(self.conv(x)), inplace=True)
        if self.has_skip:
            x = x + shortcut
        return x


class EdgeResidual(nn.Module):
    """timm ``EdgeResidual`` / FusedIB ('er'). Submodules: ``conv_exp``, ``bn1``, ``conv_pwl``, ``bn2``."""

    def __init__(self, in_chs, out_chs, exp_kernel_size, stride, exp_ratio, noskip=False):
        super().__init__()
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.conv_exp = nn.Conv2d(
            in_chs, mid_chs, exp_kernel_size, stride,
            padding=_pad(exp_kernel_size, stride), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(mid_chs)
        self.conv_pwl = nn.Conv2d(mid_chs, out_chs, 1, 1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chs)
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn1(self.conv_exp(x)), inplace=True)
        x = self.bn2(self.conv_pwl(x))
        if self.has_skip:
            x = x + shortcut
        return x


class UniversalInvertedResidual(nn.Module):
    """timm ``UniversalInvertedResidual`` / UIB ('uir').

    Submodules: ``dw_start``, ``pw_exp``, ``dw_mid``, ``se``, ``pw_proj``,
    ``dw_end``, ``layer_scale`` (the last three are Identity for conv variants).
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size_start, dw_kernel_size_mid,
                 stride, exp_ratio, noskip=False):
        super().__init__()
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

        if dw_kernel_size_start:
            dw_start_stride = stride if not dw_kernel_size_mid else 1
            self.dw_start = ConvNormAct(
                in_chs, in_chs, dw_kernel_size_start, stride=dw_start_stride,
                groups=in_chs, apply_act=False,
            )
        else:
            self.dw_start = nn.Identity()

        mid_chs = make_divisible(in_chs * exp_ratio)
        self.pw_exp = ConvNormAct(in_chs, mid_chs, 1, stride=1, groups=1, apply_act=True)

        if dw_kernel_size_mid:
            self.dw_mid = ConvNormAct(
                mid_chs, mid_chs, dw_kernel_size_mid, stride=stride,
                groups=mid_chs, apply_act=True,
            )
        else:
            self.dw_mid = nn.Identity()

        self.se = nn.Identity()
        self.pw_proj = ConvNormAct(mid_chs, out_chs, 1, stride=1, groups=1, apply_act=False)
        self.dw_end = nn.Identity()
        self.layer_scale = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = x + shortcut
        return x


# --------------------------------------------------------------------------- #
# Arch-string decoding (trimmed timm ``_decode_block_str`` for cn/er/uir).
# --------------------------------------------------------------------------- #
def _parse_block_str(block_str: str) -> Dict:
    parts = block_str.split("_")
    block_type = parts[0]
    options: Dict[str, str] = {}
    for op in parts[1:]:
        if op in ("noskip", "skip"):
            options[op] = "1"
            continue
        m = re.match(r"([a-z]+)(.+)", op)
        if m:
            options[m.group(1)] = m.group(2)

    repeats = int(options["r"])
    ba = {
        "block_type": block_type,
        "out_chs": int(options["c"]),
        "stride": int(options["s"]),
    }
    if block_type == "cn":
        ba.update(kernel_size=int(options["k"]), skip="skip" in options)
    elif block_type == "er":
        ba.update(
            exp_kernel_size=int(options["k"]),
            exp_ratio=float(options["e"]),
            noskip="noskip" in options,
        )
    elif block_type == "uir":
        ba.update(
            dw_kernel_size_start=int(options.get("a", 0)),
            dw_kernel_size_mid=int(options["k"]),
            exp_ratio=float(options["e"]),
            noskip="noskip" in options,
        )
    else:  # pragma: no cover - conv variants never hit this
        raise ValueError(f"Unsupported MobileNetV4-conv block type: {block_type!r}")
    return ba, repeats


def _expand_stage(stage: List[str]) -> List[Dict]:
    """Flatten a stage's block strings into per-block arg dicts (repeats applied)."""
    expanded: List[Dict] = []
    for block_str in stage:
        ba, repeats = _parse_block_str(block_str)
        expanded.extend(deepcopy(ba) for _ in range(repeats))
    return expanded


def _make_block(ba: Dict, in_chs: int, out_chs: int, stride: int) -> nn.Module:
    bt = ba["block_type"]
    if bt == "cn":
        return ConvBnAct(in_chs, out_chs, ba["kernel_size"], stride, skip=ba["skip"])
    if bt == "er":
        return EdgeResidual(in_chs, out_chs, ba["exp_kernel_size"], stride,
                            ba["exp_ratio"], noskip=ba["noskip"])
    return UniversalInvertedResidual(
        in_chs, out_chs, ba["dw_kernel_size_start"], ba["dw_kernel_size_mid"],
        stride, ba["exp_ratio"], noskip=ba["noskip"],
    )


# --------------------------------------------------------------------------- #
# The model.
# --------------------------------------------------------------------------- #
class MobileNetV4(nn.Module):
    """MobileNetV4 conv backbone + classification head (timm-compatible)."""

    def __init__(self, size: str = "s", num_classes: int = 1000):
        super().__init__()
        if size not in ARCH_DEFS:
            raise ValueError(f"Unknown MobileNetV4 size {size!r}; choose from {list(ARCH_DEFS)}.")
        spec = ARCH_DEFS[size]
        self.size = size
        self.num_classes = num_classes

        stem_chs = round_channels(spec["stem"])
        self.conv_stem = nn.Conv2d(3, stem_chs, 3, 2, padding=_pad(3, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(stem_chs)

        stages: List[nn.Module] = []
        in_chs = stem_chs
        for stage in spec["stages"]:
            blocks: List[nn.Module] = []
            for block_idx, ba in enumerate(_expand_stage(stage)):
                stride = ba["stride"] if block_idx == 0 else 1  # only first block of a stage strides
                out_chs = round_channels(ba["out_chs"])
                blocks.append(_make_block(ba, in_chs, out_chs, stride))
                in_chs = out_chs
            stages.append(nn.Sequential(*blocks))
        self.blocks = nn.Sequential(*stages)

        self.num_features = in_chs  # block output channels (960)
        self.head_hidden_size = HEAD_HIDDEN_SIZE
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(in_chs, HEAD_HIDDEN_SIZE, 1, 1, padding=0, bias=False)
        self.norm_head = nn.BatchNorm2d(HEAD_HIDDEN_SIZE)
        self.act2 = nn.Identity()
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(HEAD_HIDDEN_SIZE, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.blocks(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = F.relu(self.norm_head(x), inplace=True)
        x = self.act2(x)
        x = self.flatten(x)
        if pre_logits:
            return x
        return self.classifier(x)

    def reset_classifier(self, num_classes: int) -> None:
        """Swap the final Linear for fine-tuning to a new class count."""
        self.num_classes = num_classes
        device = self.classifier.weight.device
        self.classifier = nn.Linear(self.head_hidden_size, num_classes).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))
