"""Backbones for the LibrePPOCR family: PP-LCNetV3 and PP-HGNetV2-B4.

Ported to PyTorch from PaddleOCR (Apache-2.0):
- ``ppocr/modeling/backbones/rec_lcnetv3.py`` (PPLCNetV3, det + rec variants)
- ``ppocr/modeling/backbones/rec_pphgnetv2.py`` (PPHGNetV2_B4, det + rec variants)

Module attribute names intentionally mirror the Paddle layer names so the
converted state dicts map one-to-one (BatchNorm ``_mean``/``_variance`` and
Linear transposes are handled by ``weights/convert_ppocr_weights.py``).

Inference-only: the multi-branch reparameterizable blocks are kept in their
training form (the published training checkpoints store the unfused branches)
and always run the full multi-branch forward in eval mode.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PP-LCNetV3
# =============================================================================

# k, in_c, out_c, stride, use_se per block. The det variant downsamples with
# integer strides; the rec variant uses (h, w) strides tuned for text lines.
NET_CONFIG_DET = {
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [
        [3, 128, 256, 2, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    "blocks6": [
        [5, 256, 512, 2, True],
        [5, 512, 512, 1, True],
        [5, 512, 512, 1, False],
        [5, 512, 512, 1, False],
    ],
}

NET_CONFIG_REC = {
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 1, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, (2, 1), False], [3, 128, 128, 1, False]],
    "blocks5": [
        [3, 128, 256, (1, 2), False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    "blocks6": [
        [5, 256, 512, (2, 1), True],
        [5, 512, 512, 1, True],
        [5, 512, 512, (2, 1), False],
        [5, 512, 512, 1, False],
    ],
}


def make_divisible(v: float, divisor: int = 16, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1,), float(scale_value)))
        self.bias = nn.Parameter(torch.full((1,), float(bias_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class Act(nn.Module):
    def __init__(self, act: str = "hswish"):
        super().__init__()
        if act == "hswish":
            self.act = nn.Hardswish()
        else:
            assert act == "relu"
            self.act = nn.ReLU()
        self.lab = LearnableAffineBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lab(self.act(x))


class LearnableRepLayer(nn.Module):
    """Multi-branch reparameterizable conv kept in unfused training form."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        num_conv_branches=1,
    ):
        super().__init__()
        self.stride = stride
        self.identity = (
            nn.BatchNorm2d(in_channels)
            if out_channels == in_channels and stride == 1
            else None
        )
        self.conv_kxk = nn.ModuleList(
            [
                ConvBNLayer(in_channels, out_channels, kernel_size, stride, groups=groups)
                for _ in range(num_conv_branches)
            ]
        )
        self.conv_1x1 = (
            ConvBNLayer(in_channels, out_channels, 1, stride, groups=groups)
            if kernel_size > 1
            else None
        )
        self.lab = LearnableAffineBlock()
        self.act = Act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0
        if self.identity is not None:
            out = out + self.identity(x)
        if self.conv_1x1 is not None:
            out = out + self.conv_1x1(x)
        for conv in self.conv_kxk:
            out = out + conv(x)
        out = self.lab(out)
        # Upstream skips the activation only for a literal integer stride of 2;
        # tuple strides such as (2, 1) keep it. Replicated exactly.
        if self.stride != 2:
            out = self.act(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return identity * x


class LCNetV3Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dw_size,
        use_se=False,
        conv_kxk_num=4,
    ):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = LearnableRepLayer(
            in_channels,
            in_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=in_channels,
            num_conv_branches=conv_kxk_num,
        )
        if use_se:
            self.se = SELayer(in_channels)
        self.pw_conv = LearnableRepLayer(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            num_conv_branches=conv_kxk_num,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        return self.pw_conv(x)


class PPLCNetV3(nn.Module):
    """PP-LCNetV3 backbone; ``det=True`` returns 4 pyramid features."""

    def __init__(self, scale: float = 1.0, conv_kxk_num: int = 4, det: bool = False):
        super().__init__()
        self.scale = scale
        self.det = det
        net_config = NET_CONFIG_DET if det else NET_CONFIG_REC

        self.conv1 = ConvBNLayer(3, make_divisible(16 * scale), 3, stride=2)

        def _stage(key: str) -> nn.Sequential:
            return nn.Sequential(
                *[
                    LCNetV3Block(
                        in_channels=make_divisible(in_c * scale),
                        out_channels=make_divisible(out_c * scale),
                        dw_size=k,
                        stride=s,
                        use_se=se,
                        conv_kxk_num=conv_kxk_num,
                    )
                    for (k, in_c, out_c, s, se) in net_config[key]
                ]
            )

        self.blocks2 = _stage("blocks2")
        self.blocks3 = _stage("blocks3")
        self.blocks4 = _stage("blocks4")
        self.blocks5 = _stage("blocks5")
        self.blocks6 = _stage("blocks6")
        self.out_channels = make_divisible(512 * scale)

        if det:
            mv_c = [16, 24, 56, 480]
            stage_out = [
                make_divisible(net_config["blocks3"][-1][2] * scale),
                make_divisible(net_config["blocks4"][-1][2] * scale),
                make_divisible(net_config["blocks5"][-1][2] * scale),
                make_divisible(net_config["blocks6"][-1][2] * scale),
            ]
            self.layer_list = nn.ModuleList(
                [
                    nn.Conv2d(stage_out[i], int(mv_c[i] * scale), 1)
                    for i in range(4)
                ]
            )
            self.out_channels = [int(c * scale) for c in mv_c]

    def forward(self, x: torch.Tensor):
        out_list = []
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        out_list.append(x)
        x = self.blocks4(x)
        out_list.append(x)
        x = self.blocks5(x)
        out_list.append(x)
        x = self.blocks6(x)
        out_list.append(x)

        if self.det:
            return [layer(feat) for layer, feat in zip(self.layer_list, out_list)]

        if self.training:
            return F.adaptive_avg_pool2d(x, (1, 40))
        return F.avg_pool2d(x, (3, 2))


# =============================================================================
# PP-HGNetV2-B4
# =============================================================================


class ConvBNAct(nn.Module):
    """Conv + BN + optional ReLU. ``padding="SAME"`` emulates Paddle's
    TF-style asymmetric same-padding (used by the k=2 stem convs)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        use_act=True,
    ):
        super().__init__()
        self.use_act = use_act
        self.same_pad = padding == "SAME"
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0 if self.same_pad else (kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if use_act:
            self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.same_pad:
            # Stride-1 k=2 SAME padding: one pixel on the bottom/right.
            x = F.pad(x, (0, 1, 0, 1))
        x = self.bn(self.conv(x))
        if self.use_act:
            x = self.act(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=1, use_act=False)
        self.conv2 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_act=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class StemBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, text_rec=False):
        super().__init__()
        self.stem1 = ConvBNAct(in_channels, mid_channels, 3, stride=2)
        self.stem2a = ConvBNAct(mid_channels, mid_channels // 2, 2, stride=1, padding="SAME")
        self.stem2b = ConvBNAct(mid_channels // 2, mid_channels, 2, stride=1, padding="SAME")
        self.stem3 = ConvBNAct(mid_channels * 2, mid_channels, 3, stride=1 if text_rec else 2)
        self.stem4 = ConvBNAct(mid_channels, out_channels, 1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x2 = self.stem2a(x)
        x2 = self.stem2b(x2)
        # Paddle MaxPool2D(k=2, s=1, padding="SAME"): pad bottom/right with -inf.
        x1 = F.max_pool2d(F.pad(x, (0, 1, 0, 1), value=float("-inf")), 2, stride=1)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        return self.stem4(x)


class HGV2Block(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size=3,
        layer_num=6,
        identity=False,
        light_block=True,
    ):
        super().__init__()
        self.identity = identity
        block_cls = LightConvBNAct if light_block else ConvBNAct
        self.layers = nn.ModuleList(
            [
                block_cls(
                    in_channels=in_channels if i == 0 else mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                )
                for i in range(layer_num)
            ]
        )
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(total_channels, out_channels // 2, 1, stride=1)
        self.aggregation_excitation_conv = ConvBNAct(out_channels // 2, out_channels, 1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.identity:
            x = x + identity
        return x


class HGV2Stage(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        block_num,
        layer_num=6,
        is_downsample=True,
        light_block=True,
        kernel_size=3,
        stride=2,
    ):
        super().__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.downsample = ConvBNAct(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels,
                use_act=False,
            )
        self.blocks = nn.Sequential(
            *[
                HGV2Block(
                    in_channels=in_channels if i == 0 else out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    layer_num=layer_num,
                    identity=i != 0,
                    light_block=light_block,
                )
                for i in range(block_num)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_downsample:
            x = self.downsample(x)
        return self.blocks(x)


# in_channels, mid_channels, out_channels, blocks, downsample, light, k, layers, stride
_B4_STAGES_DET = {
    "stage1": [48, 48, 128, 1, False, False, 3, 6, 2],
    "stage2": [128, 96, 512, 1, True, False, 3, 6, 2],
    "stage3": [512, 192, 1024, 3, True, True, 5, 6, 2],
    "stage4": [1024, 384, 2048, 1, True, True, 5, 6, 2],
}
_B4_STAGES_REC = {
    "stage1": [48, 48, 128, 1, True, False, 3, 6, (2, 1)],
    "stage2": [128, 96, 512, 1, True, False, 3, 6, (1, 2)],
    "stage3": [512, 192, 1024, 3, True, True, 5, 6, (2, 1)],
    "stage4": [1024, 384, 2048, 1, True, True, 5, 6, (2, 1)],
}


class PPHGNetV2B4(nn.Module):
    """PP-HGNetV2-B4; ``det=True`` returns 4 pyramid features, ``text_rec=True``
    pools text lines to height 1 like the LCNetV3 rec variant.

    The classification-only tail (``last_conv``/``fc``) of the upstream class is
    never used by the OCR forward paths and is omitted; the converter drops its
    weights.
    """

    def __init__(self, det: bool = False, text_rec: bool = False):
        super().__init__()
        assert det != text_rec, "PPHGNetV2B4 is built as either det or text_rec"
        self.det = det
        self.text_rec = text_rec
        stage_config = _B4_STAGES_DET if det else _B4_STAGES_REC

        self.stem = StemBlock(3, 32, 48, text_rec=text_rec)
        self.stages = nn.ModuleList()
        self.out_channels: List[int] = []
        for key in ("stage1", "stage2", "stage3", "stage4"):
            (
                in_c,
                mid_c,
                out_c,
                block_num,
                is_downsample,
                light_block,
                kernel_size,
                layer_num,
                stride,
            ) = stage_config[key]
            self.stages.append(
                HGV2Stage(
                    in_c,
                    mid_c,
                    out_c,
                    block_num,
                    layer_num,
                    is_downsample,
                    light_block,
                    kernel_size,
                    stride,
                )
            )
            self.out_channels.append(out_c)
        if text_rec:
            self.out_channels = _B4_STAGES_REC["stage4"][2]

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            if self.det:
                out.append(x)
        if self.det:
            return out
        if self.training:
            return F.adaptive_avg_pool2d(x, (1, 40))
        return F.avg_pool2d(x, (3, 2))


__all__ = ["PPLCNetV3", "PPHGNetV2B4", "make_divisible"]
