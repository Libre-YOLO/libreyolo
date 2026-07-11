"""PicoSAM3 network ported from pbonazzi/picosam3 (Apache-2.0).

The module names intentionally match upstream commit
1b03949e43472953bb0021685c7fc3f5fdf48fde so the published
``PicoSAM3_SAM3_student_best.pt`` checkpoint loads strictly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _depthwise_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        ),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ECABlock(nn.Module):
    """Channel attention block used by the PicoSAM3 decoder."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.conv1x1(self.avg_pool(x)))


class PicoSAM3Network(nn.Module):
    """1.3M-parameter encoder-decoder producing one ROI mask logit map."""

    image_size: int = 96

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.encoder_stage1 = _depthwise_conv(in_channels, 48)
        self.down1 = nn.Conv2d(48, 48, 3, stride=2, padding=1, bias=False)
        self.encoder_stage2 = _depthwise_conv(48, 96)
        self.down2 = nn.Conv2d(96, 96, 3, stride=2, padding=1, bias=False)
        self.encoder_stage3 = _depthwise_conv(96, 160)
        self.down3 = nn.Conv2d(160, 160, 3, stride=2, padding=1, bias=False)
        self.encoder_stage4 = _depthwise_conv(160, 256)
        self.down4 = nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False)

        self.bottleneck = nn.Sequential(
            _depthwise_conv(256, 320),
            nn.Conv2d(
                320,
                320,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=320,
                bias=False,
            ),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
            nn.Conv2d(320, 320, kernel_size=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            _depthwise_conv(320, 192),
        )
        self.skip_conn4 = nn.Conv2d(256, 192, kernel_size=1)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            _depthwise_conv(192, 128),
        )
        self.skip_conn3 = nn.Conv2d(160, 128, kernel_size=1)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            _depthwise_conv(128, 80),
        )
        self.skip_conn2 = nn.Conv2d(96, 80, kernel_size=1)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            _depthwise_conv(80, 40),
        )
        self.skip_conn1 = nn.Conv2d(48, 40, kernel_size=1)
        self.eca = ECABlock(40)
        self.refine = nn.Sequential(
            nn.Conv2d(40, 40, 3, padding=1, groups=40, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.encoder_stage1(x)
        f2 = self.encoder_stage2(self.down1(f1))
        f3 = self.encoder_stage3(self.down2(f2))
        f4 = self.encoder_stage4(self.down3(f3))
        bottleneck = self.bottleneck(self.down4(f4))

        u1 = self.up1(bottleneck) + self.skip_conn4(f4)
        u2 = self.up2(u1) + self.skip_conn3(f3)
        u3 = self.up3(u2) + self.skip_conn2(f2)
        u4 = self.up4(u3) + self.skip_conn1(f1)
        return self.refine(self.eca(u4))


__all__ = ["ECABlock", "PicoSAM3Network"]
