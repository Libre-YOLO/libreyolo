"""DeFE: the Density-Focal Extractor head.

Ported from Dome-DETR (https://github.com/RicePasteM/Dome-DETR),
commit 2dde3bc1946a3e9fad9abd0612b59fc39bd6b861, Apache License 2.0.
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

A lightweight depthwise-separable stack over the stride-4 projected feature
map. It emits two things:

- ``density``: a per-pixel foreground/density map in ``[0, 1]``, used by MWAS
  to pick which encoder windows are worth attending over and by PAQI to set a
  per-query IoU threshold.
- ``reg_value``: a scalar per image (an object-count proxy). Inference does not
  consume it; it exists because the upstream criterion supervises it, and the
  checkpoints carry its weights.

Differences from upstream: the module-level duplicate imports and the
training-only ``GaussHeatmapGenerator`` (which builds the density ground truth
from targets) are not carried here. Forward numerics are unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightAttention(nn.Module):
    """Squeeze-and-excitation style channel gate."""

    def __init__(self, channel: int, reduction: int = 8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        att = self.gap(x).view(b, c)
        att = self.fc(att).view(b, c, 1, 1)
        return x * att.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=in_ch,
        )
        self.pointwise = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)


class OptimizedDeFE(nn.Module):
    """Dilated depthwise-separable trunk with one channel-attention block."""

    # (out_channels, dilation) per layer; attention is inserted after index 2.
    CFG = ((256, 1), (256, 2), (256, 3), (256, 1), (256, 1))

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 256
        for idx, (out_ch, dilation) in enumerate(self.CFG):
            layers += [DepthwiseSeparableConv(in_ch, out_ch, dilation), nn.BatchNorm2d(out_ch)]
            in_ch = out_ch
            if idx == 2:
                layers.append(LightweightAttention(out_ch))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LiteDeFE(nn.Module):
    """The ``defe_type: light`` variant, the only one upstream ships weights for."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.AvgPool2d(kernel_size=2),
        )
        self.defe = OptimizedDeFE()
        self.density_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid(),
        )
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor):
        x = self.conv1(features)
        x = self.defe(x)

        density = F.interpolate(
            self.density_head(x), scale_factor=2, mode="bilinear", align_corners=False
        )

        # Normalised over the whole batch tensor, as upstream does, so the
        # 0.05 threshold in the encoder's window filter is scale free.
        density_max = density.max()
        if density_max > 0:
            density = density / density_max

        reg_value = self.regression_head(x)
        return density, reg_value
