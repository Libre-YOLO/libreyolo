"""Native DeepLabv3 inference graph with torchvision-compatible state keys.

The DeepLabv3 head and orchestration are derived from ``pytorch/vision``
v0.26.0 at commit ``336d36e8db990a905498c73933e35231876e28bc``
(BSD-3-Clause). ResNet and MobileNetV3 remain torchvision building blocks,
matching LibreYOLO's existing Faster R-CNN backbone precedent. The graph omits
the training-only auxiliary FCN head; the converter removes those tensors and
strict-loads every retained parameter.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, resnet101, resnet50
from torchvision.models._utils import IntermediateLayerGetter


SIZE_CONFIGS = {
    "r50": {"backbone": "resnet50", "output_stride": 8},
    "r101": {"backbone": "resnet101", "output_stride": 8},
    "mv3": {"backbone": "mobilenet_v3_large", "output_stride": 16},
}


class ASPPConv(nn.Sequential):
    """One atrous 3x3 branch in the spatial-pyramid head."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    """Image-level pooling branch introduced by DeepLabv3."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[-2:]
        for layer in self:
            x = layer(x)
        return F.interpolate(
            x,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )


class ASPP(nn.Module):
    """Parallel 1x1, atrous, and image-pooling branches plus projection."""

    def __init__(
        self,
        in_channels: int,
        atrous_rates: Sequence[int],
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        branches: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        ]
        branches.extend(
            ASPPConv(in_channels, out_channels, rate) for rate in atrous_rates
        )
        branches.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(
                len(branches) * out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.convs], dim=1))


class DeepLabHead(nn.Sequential):
    """ASPP followed by the dense 21-way classifier."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        atrous_rates: Sequence[int] = (12, 24, 36),
    ) -> None:
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )


def _build_resnet_backbone(size: str) -> IntermediateLayerGetter:
    builder = resnet50 if size == "r50" else resnet101
    backbone = builder(
        weights=None,
        replace_stride_with_dilation=[False, True, True],
    )
    return IntermediateLayerGetter(backbone, return_layers={"layer4": "out"})


def _build_mobilenet_backbone() -> tuple[IntermediateLayerGetter, int]:
    features = mobilenet_v3_large(weights=None, dilated=True).features
    stage_indices = [0]
    stage_indices.extend(
        index for index, block in enumerate(features) if getattr(block, "_is_cn", False)
    )
    stage_indices.append(len(features) - 1)
    out_position = stage_indices[-1]
    out_channels = int(features[out_position].out_channels)
    backbone = IntermediateLayerGetter(
        features,
        return_layers={str(out_position): "out"},
    )
    return backbone, out_channels


class LibreDeepLabv3Net(nn.Module):
    """Inference-only DeepLabv3 graph returning full-canvas dense logits."""

    def __init__(self, size: str = "r50", num_classes: int = 21) -> None:
        super().__init__()
        if size not in SIZE_CONFIGS:
            raise ValueError(
                f"Unknown DeepLabv3 size {size!r}; choose from {tuple(SIZE_CONFIGS)}."
            )
        self.size = size
        self.num_classes = int(num_classes)
        if size == "mv3":
            self.backbone, head_channels = _build_mobilenet_backbone()
        else:
            self.backbone = _build_resnet_backbone(size)
            head_channels = 2048
        self.classifier = DeepLabHead(head_channels, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        logits = self.classifier(features["out"])
        return F.interpolate(
            logits,
            size=input_shape,
            mode="bilinear",
            align_corners=False,
        )


__all__ = [
    "ASPP",
    "ASPPConv",
    "ASPPPooling",
    "DeepLabHead",
    "LibreDeepLabv3Net",
    "SIZE_CONFIGS",
]
