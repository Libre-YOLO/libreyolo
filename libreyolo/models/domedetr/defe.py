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

``GaussHeatmapGenerator`` builds the density ground truth that supervises
``density``; it runs only when targets are supplied. Forward numerics are
unchanged from upstream.
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


class GaussHeatmapGenerator:
    """Rasterise target boxes into the density map DeFE is trained against.

    Each box contributes a Gaussian whose sigma scales with its own width and
    height, so a cluster of tiny objects produces a bright, tight blob and a
    single large object a broad dim one. The result is peak-normalised, which
    matches how :class:`LiteDeFE` normalises its prediction.

    Boxes are ``cxcywh`` normalised to ``[0, 1]``.
    """

    def __init__(self, img_size=(800, 800), sigma_ratio: float = 1.2):
        self.img_size = img_size
        self.sigma_ratio = sigma_ratio

    def __call__(self, bboxes: torch.Tensor) -> torch.Tensor:
        height, width = self.img_size
        heatmap = torch.zeros((height, width), dtype=torch.float32)

        for box in bboxes:
            x_center, y_center, box_w, box_h = (float(v) for v in box)
            cx_px = int(x_center * width)
            cy_px = int(y_center * height)
            w_px = max(int(box_w * width), 1)
            h_px = max(int(box_h * height), 1)

            kernel = self._gaussian_kernel(
                max(w_px * self.sigma_ratio, 1.0), max(h_px * self.sigma_ratio, 1.0)
            )
            if kernel.numel() == 0:
                continue

            k_h, k_w = kernel.shape
            radius_x, radius_y = k_w // 2, k_h // 2

            x_start = max(cx_px - radius_x, 0)
            y_start = max(cy_px - radius_y, 0)
            x_end = min(cx_px + radius_x + 1, width)
            y_end = min(cy_px + radius_y + 1, height)
            if x_end <= x_start or y_end <= y_start:
                continue

            # Crop the kernel by however much it overhangs the image.
            k_start_x = max(radius_x - (cx_px - x_start), 0)
            k_start_y = max(radius_y - (cy_px - y_start), 0)
            k_end_x = k_w - max((cx_px + radius_x + 1) - x_end, 0)
            k_end_y = k_h - max((cy_px + radius_y + 1) - y_end, 0)

            cropped = kernel[k_start_y:k_end_y, k_start_x:k_end_x]
            if cropped.numel() == 0:
                continue
            cropped = cropped[: y_end - y_start, : x_end - x_start]

            heatmap[y_start:y_end, x_start:x_end] += cropped

        peak = heatmap.max()
        if peak > 0:
            heatmap = heatmap / peak
        return heatmap.unsqueeze(0)

    @staticmethod
    def _gaussian_kernel(sigma_x: float, sigma_y: float) -> torch.Tensor:
        sigma_x = max(sigma_x, 0.1)
        sigma_y = max(sigma_y, 0.1)
        kernel_w = int(6 * sigma_x) + 1
        kernel_h = int(6 * sigma_y) + 1
        kernel_w += 1 - kernel_w % 2
        kernel_h += 1 - kernel_h % 2

        x = torch.arange(kernel_w, dtype=torch.float32) - kernel_w // 2
        y = torch.arange(kernel_h, dtype=torch.float32) - kernel_h // 2
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        kernel = torch.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))
        total = kernel.sum()
        if total > 0:
            kernel = kernel / total
        return kernel
