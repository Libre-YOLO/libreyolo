"""Native Real-ESRGAN modules for super-resolution.

This is a compact PyTorch port of the two Real-ESRGAN generator architectures:

* :class:`RRDBNet` - Residual-in-Residual Dense Block network (the ``x4plus`` /
  ``x2plus`` quality tier).
* :class:`SRVGGNetCompact` - a plain VGG-style network with a residual nearest
  upsample (the ``realesr-general-x4v3`` fast/video tier).

Module and parameter names intentionally mirror the upstream architectures so
official state dicts convert with a plain metadata-wrap (no key remapping).

Architecture lineage is BasicSR (https://github.com/XPixelGroup/BasicSR,
Apache-2.0); the released weights ship with Real-ESRGAN
(https://github.com/xinntao/Real-ESRGAN, BSD-3-Clause). See the family NOTICE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Pixel unshuffle: inverse of ``nn.PixelShuffle``.

    Reduces the spatial size by ``scale`` and enlarges the channel size by
    ``scale**2``. Used by RRDBNet for the x2 and x1 variants before the main
    trunk so the trunk always runs at the same downsampled resolution.
    """

    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    if hh % scale != 0 or hw % scale != 0:
        raise ValueError(
            f"pixel_unshuffle requires H and W to be divisible by scale={scale}, "
            f"got ({hh}, {hw}). Reflect-pad the input to a multiple of {scale} first."
        )
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used inside :class:`RRDB`."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, upstream scales the residual by 0.2 for better training.
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block used in the RRDBNet trunk."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


def _make_layer(block, num_blocks: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(*[block(**kwargs) for _ in range(num_blocks)])


class RRDBNet(nn.Module):
    """Residual-in-Residual Dense Block network (ESRGAN / Real-ESRGAN).

    The x2 and x1 variants first pixel-unshuffle the input to reduce spatial
    size and enlarge channels, so the trunk always upsamples by 4x; the net
    scale is ``4 / unshuffle_factor``.

    Args:
        num_in_ch: input channels (3 for RGB).
        num_out_ch: output channels (3 for RGB).
        scale: net upscale factor (1, 2, or 4).
        num_feat: trunk feature width (64).
        num_block: number of RRDB blocks (23 for x4plus/x2plus).
        num_grow_ch: dense growth channels (32).
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = _make_layer(
            RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample: two nearest-neighbour 2x steps
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style super-resolution network (realesr-general-x4v3).

    Upsampling happens only in the last layer via ``PixelShuffle``; a nearest
    upsample of the input is added so the network learns the residual.

    Args:
        num_in_ch: input channels (3 for RGB).
        num_out_ch: output channels (3 for RGB).
        num_feat: feature width (64).
        num_conv: number of body conv layers (32 for x4v3).
        upscale: upscale factor (4).
        act_type: activation, one of ``relu`` / ``prelu`` / ``leakyrelu``.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._activation(act_type, num_feat))
        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._activation(act_type, num_feat))
        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _activation(act_type: str, num_feat: int) -> nn.Module:
        if act_type == "relu":
            return nn.ReLU(inplace=True)
        if act_type == "prelu":
            return nn.PReLU(num_parameters=num_feat)
        if act_type == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        raise ValueError(f"Unsupported act_type: {act_type!r}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # add the nearest upsampled image so the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        out = out + base
        return out


__all__ = [
    "RRDB",
    "RRDBNet",
    "ResidualDenseBlock",
    "SRVGGNetCompact",
    "pixel_unshuffle",
]
