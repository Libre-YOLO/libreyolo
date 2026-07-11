"""Text-detection necks and DB heads for the LibrePPOCR family.

Ported to PyTorch from PaddleOCR (Apache-2.0):
- ``ppocr/modeling/necks/db_fpn.py`` (RSEFPN, LKPAN)
- ``ppocr/modeling/necks/intracl.py`` (IntraCLBlock)
- ``ppocr/modeling/heads/det_db_head.py`` (DBHead, PFHeadLocal)
- ``ppocr/modeling/backbones/det_mobilenet_v3.py`` (SEModule)

The tiers ship exactly the PP-OCRv5 det configs:
- mobile (``t``): PPLCNetV3(scale=0.75, det) + RSEFPN(96, shortcut) + DBHead
- server (``l``): PPHGNetV2-B4(det) + LKPAN(256, intracl) + PFHeadLocal

Inference-only: heads return the sigmoid shrink-probability map; the
threshold branch exists so training checkpoints load completely.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import PPHGNetV2B4, PPLCNetV3


class SEModule(nn.Module):
    """Squeeze-excitation with Paddle's slope-0.2 hard-sigmoid gate."""

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.avg_pool(x)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        # Paddle F.hardsigmoid(x, slope=0.2, offset=0.5); torch's builtin uses
        # slope 1/6, so this must stay explicit for parity.
        out = torch.clamp(0.2 * out + 0.5, 0.0, 1.0)
        return x * out


class RSELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super().__init__()
        self.in_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.se_block = SEModule(out_channels)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        if self.shortcut:
            return x + self.se_block(x)
        return self.se_block(x)


class RSEFPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int, shortcut: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.ModuleList(
            [RSELayer(c, out_channels, kernel_size=1, shortcut=shortcut) for c in in_channels]
        )
        self.inp_conv = nn.ModuleList(
            [
                RSELayer(out_channels, out_channels // 4, kernel_size=3, shortcut=shortcut)
                for _ in in_channels
            ]
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        c2, c3, c4, c5 = x
        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest")
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")
        return torch.cat([p5, p4, p3, p2], dim=1)


class IntraCLBlock(nn.Module):
    def __init__(self, in_channels: int = 96, reduce_factor: int = 4):
        super().__init__()
        reduced = in_channels // reduce_factor
        self.conv1x1_reduce_channel = nn.Conv2d(in_channels, reduced, 1)
        self.conv1x1_return_channel = nn.Conv2d(reduced, in_channels, 1)

        self.v_layer_7x1 = nn.Conv2d(reduced, reduced, (7, 1), padding=(3, 0))
        self.v_layer_5x1 = nn.Conv2d(reduced, reduced, (5, 1), padding=(2, 0))
        self.v_layer_3x1 = nn.Conv2d(reduced, reduced, (3, 1), padding=(1, 0))

        self.q_layer_1x7 = nn.Conv2d(reduced, reduced, (1, 7), padding=(0, 3))
        self.q_layer_1x5 = nn.Conv2d(reduced, reduced, (1, 5), padding=(0, 2))
        self.q_layer_1x3 = nn.Conv2d(reduced, reduced, (1, 3), padding=(0, 1))

        self.c_layer_7x7 = nn.Conv2d(reduced, reduced, (7, 7), padding=(3, 3))
        self.c_layer_5x5 = nn.Conv2d(reduced, reduced, (5, 5), padding=(2, 2))
        self.c_layer_3x3 = nn.Conv2d(reduced, reduced, (3, 3), padding=(1, 1))

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_new = self.conv1x1_reduce_channel(x)
        x_7 = self.c_layer_7x7(x_new) + self.v_layer_7x1(x_new) + self.q_layer_1x7(x_new)
        x_5 = self.c_layer_5x5(x_7) + self.v_layer_5x1(x_7) + self.q_layer_1x5(x_7)
        x_3 = self.c_layer_3x3(x_5) + self.v_layer_3x1(x_5) + self.q_layer_1x3(x_5)
        x_relation = self.relu(self.bn(self.conv1x1_return_channel(x_3)))
        return x + x_relation


class LKPAN(nn.Module):
    """Large-kernel PAN neck ("large" mode: plain 9x9 convs), with the
    IntraCL relation blocks the PP-OCRv5 server det config enables."""

    def __init__(self, in_channels: List[int], out_channels: int, intracl: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        self.pan_head_conv = nn.ModuleList()
        self.pan_lat_conv = nn.ModuleList()

        for i, c in enumerate(in_channels):
            self.ins_conv.append(nn.Conv2d(c, out_channels, 1, bias=False))
            self.inp_conv.append(
                nn.Conv2d(out_channels, out_channels // 4, 9, padding=4, bias=False)
            )
            if i > 0:
                self.pan_head_conv.append(
                    nn.Conv2d(
                        out_channels // 4,
                        out_channels // 4,
                        3,
                        padding=1,
                        stride=2,
                        bias=False,
                    )
                )
            self.pan_lat_conv.append(
                nn.Conv2d(out_channels // 4, out_channels // 4, 9, padding=4, bias=False)
            )

        self.intracl = intracl
        if intracl:
            self.incl1 = IntraCLBlock(out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(out_channels // 4, reduce_factor=2)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        c2, c3, c4, c5 = x
        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest")
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")

        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)

        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)

        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)

        if self.intracl:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")
        return torch.cat([p5, p4, p3, p2], dim=1)


class DBBranchHead(nn.Module):
    """The conv/deconv tower one DB branch (binarize or thresh) is made of.

    Upstream class name is ``Head``; renamed here to avoid a bare ``Head``
    in the family namespace. Attribute names match upstream for conversion.
    """

    def __init__(self, in_channels: int, kernel_list=(3, 2, 2)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels // 4,
            kernel_list[0],
            padding=kernel_list[0] // 2,
            bias=False,
        )
        self.conv_bn1 = nn.BatchNorm2d(in_channels // 4)
        self.conv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, kernel_list[1], stride=2
        )
        self.conv_bn2 = nn.BatchNorm2d(in_channels // 4)
        self.conv3 = nn.ConvTranspose2d(in_channels // 4, 1, kernel_list[2], stride=2)

    def forward(self, x: torch.Tensor, return_f: bool = False):
        x = F.relu(self.conv_bn1(self.conv1(x)))
        x = F.relu(self.conv_bn2(self.conv2(x)))
        f = x
        x = torch.sigmoid(self.conv3(x))
        if return_f:
            return x, f
        return x


class DBHead(nn.Module):
    """Differentiable Binarization head (https://arxiv.org/abs/1911.08947).

    Inference returns the shrink probability map ``(B, 1, H, W)``.
    """

    def __init__(self, in_channels: int, k: int = 50):
        super().__init__()
        self.k = k
        self.binarize = DBBranchHead(in_channels)
        self.thresh = DBBranchHead(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.binarize(x)


class LocalModule(nn.Module):
    def __init__(self, in_c: int, mid_c: int):
        super().__init__()
        # Upstream builds this from det_mobilenet_v3.ConvBNLayer (conv + BN +
        # relu); flattened here with matching child names last_3.{conv,bn}.
        self.last_3 = nn.Sequential()
        self.last_3.add_module("conv", nn.Conv2d(in_c + 1, mid_c, 3, stride=1, padding=1, bias=False))
        self.last_3.add_module("bn", nn.BatchNorm2d(mid_c))
        self.last_1 = nn.Conv2d(mid_c, 1, 1)

    def forward(self, x: torch.Tensor, init_map: torch.Tensor) -> torch.Tensor:
        out = torch.cat([init_map, x], dim=1)
        out = F.relu(self.last_3.bn(self.last_3.conv(out)))
        return self.last_1(out)


class PFHeadLocal(DBHead):
    """DB head with the local fusion branch (PP-OCRv5 server det, mode "large")."""

    def __init__(self, in_channels: int, k: int = 50, mode: str = "large"):
        super().__init__(in_channels, k)
        self.mode = mode
        mid = in_channels // 4 if mode == "large" else in_channels // 8
        self.cbn_layer = LocalModule(in_channels // 4, mid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shrink_maps, f = self.binarize(x, return_f=True)
        f = F.interpolate(f, scale_factor=2, mode="nearest")
        cbn_maps = torch.sigmoid(self.cbn_layer(f, shrink_maps))
        return 0.5 * (shrink_maps + cbn_maps)


class PPOCRDetModel(nn.Module):
    """Backbone + neck + head text detector for one tier."""

    def __init__(self, size: str):
        super().__init__()
        if size == "t":
            self.backbone = PPLCNetV3(scale=0.75, det=True)
            self.neck = RSEFPN(self.backbone.out_channels, 96, shortcut=True)
            self.head = DBHead(self.neck.out_channels)
        elif size == "l":
            self.backbone = PPHGNetV2B4(det=True)
            self.neck = LKPAN(self.backbone.out_channels, 256, intracl=True)
            self.head = PFHeadLocal(self.neck.out_channels, mode="large")
        else:
            raise ValueError(f"Unknown LibrePPOCR size: {size!r} (expected 't' or 'l')")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the shrink probability map ``(B, 1, H, W)`` in ``[0, 1]``."""
        return self.head(self.neck(self.backbone(x)))


__all__ = ["PPOCRDetModel", "RSEFPN", "LKPAN", "DBHead", "PFHeadLocal", "IntraCLBlock"]
