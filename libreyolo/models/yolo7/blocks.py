"""YOLOv7 building blocks — native port of the MIT MultimediaTechLab/YOLO modules.

Provenance: ported from MultimediaTechLab/YOLO (MIT, (c) 2024 Kin-Yiu Wong &
Hao-Tang Tsui) — the authors' own MIT re-release of YOLOv7. **Not** the GPL-3.0
``WongKinYiu/yolov7``. Attribute names mirror the upstream ``yolo/model/module.py``
so upstream ``v7.pt`` weights load with no key remapping (skill: metadata-wrap).

Only the blocks the v7 config actually uses are ported: Conv, Pool, SPPCSPConv,
RepConv, UpSample, and the IDetection head (implicit-knowledge, YOLOR-style).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def auto_pad(kernel_size: int, dilation: int = 1) -> int:
    """Same-padding for odd kernels: ``((k-1)*d)//2`` (matches upstream auto_pad)."""
    return ((kernel_size - 1) * dilation) // 2


def make_activation(activation) -> nn.Module:
    if not activation or str(activation).lower() in ("false", "none"):
        return nn.Identity()
    return nn.SiLU()


class Conv(nn.Module):
    """Conv2d -> BatchNorm2d(eps=1e-3, momentum=3e-2) -> SiLU.

    Matches upstream ``Conv``: bias-free conv, BN eps/momentum overrides, SiLU
    default activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        groups: int = 1,
        activation="SiLU",
        padding: int | None = None,
    ):
        super().__init__()
        if padding is None:
            padding = auto_pad(kernel_size)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2)
        self.act = make_activation(activation)
        self.out_channels = out_channels

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Pool(nn.Module):
    """MaxPool block (upstream ``Pool``); kernel 2 stride 2 by default."""

    def __init__(self, kernel_size: int = 2, stride: int | None = None, padding: int | None = None):
        super().__init__()
        if padding is None:
            padding = auto_pad(kernel_size)
        if stride is None:
            stride = kernel_size
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)


class UpSample(nn.Module):
    """nn.Upsample wrapper (upstream ``UpSample``)."""

    def __init__(self, scale_factor: float = 2.0, mode: str = "nearest"):
        super().__init__()
        self.UpSample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        return self.UpSample(x)


class SPPCSPConv(nn.Module):
    """Spatial-pyramid-pooling CSP block (upstream ``SPPCSPConv``)."""

    def __init__(self, in_channels: int, out_channels: int, expand: float = 0.5,
                 kernel_sizes=(5, 9, 13)):
        super().__init__()
        neck = int(2 * out_channels * expand)
        self.pre_conv = nn.Sequential(
            Conv(in_channels, neck, 1),
            Conv(neck, neck, 3),
            Conv(neck, neck, 1),
        )
        self.short_conv = Conv(in_channels, neck, 1)
        self.pools = nn.ModuleList(
            [Pool(kernel_size=k, stride=1) for k in kernel_sizes]
        )
        self.post_conv = nn.Sequential(Conv(4 * neck, neck, 1), Conv(neck, neck, 3))
        self.merge_conv = Conv(2 * neck, out_channels, 1)
        self.out_channels = out_channels

    def forward(self, x):
        feats = [self.pre_conv(x)]
        for pool in self.pools:
            feats.append(pool(feats[-1]))
        y1 = self.post_conv(torch.cat(feats, dim=1))
        y2 = self.short_conv(x)
        return self.merge_conv(torch.cat((y1, y2), dim=1))


class RepConv(nn.Module):
    """Re-parameterizable conv in training form (upstream ``RepConv``).

    ``act(conv1(x) + conv2(x))`` with a 3x3 and a 1x1 branch (both Conv without
    activation). v7.pt ships this unfused form, so we keep it as-is.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 *, activation="SiLU"):
        super().__init__()
        self.act = make_activation(activation)
        self.conv1 = Conv(in_channels, out_channels, kernel_size, activation=False)
        self.conv2 = Conv(in_channels, out_channels, 1, activation=False)
        self.out_channels = out_channels

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))


class ImplicitA(nn.Module):
    """YOLOR implicit knowledge (add). Learned per-channel bias added to input."""

    def __init__(self, channel: int):
        super().__init__()
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    """YOLOR implicit knowledge (multiply). Learned per-channel scale."""

    def __init__(self, channel: int):
        super().__init__()
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))

    def forward(self, x):
        return self.implicit * x


class IDetection(nn.Module):
    """YOLOv7 detection head with implicit knowledge (upstream ``IDetection``).

    ``implicit_a`` (add) -> 1x1 head conv -> ``implicit_m`` (multiply). Emits a
    raw ``anchor_num*(5+nc)`` channel map per scale.
    """

    def __init__(self, in_channels: int, num_classes: int, anchor_num: int = 3):
        super().__init__()
        out_channels = (num_classes + 5) * anchor_num
        self.head_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.implicit_a = ImplicitA(in_channels)
        self.implicit_m = ImplicitM(out_channels)

    def forward(self, x):
        return self.implicit_m(self.head_conv(self.implicit_a(x)))


class MultiheadDetection(nn.Module):
    """Triple detection head (upstream ``MultiheadDetection`` with version=v7)."""

    def __init__(self, in_channels, num_classes: int, anchor_num: int = 3):
        super().__init__()
        self.heads = nn.ModuleList(
            [IDetection(c, num_classes, anchor_num=anchor_num) for c in in_channels]
        )

    def forward(self, x_list):
        return [head(x) for x, head in zip(x_list, self.heads)]
