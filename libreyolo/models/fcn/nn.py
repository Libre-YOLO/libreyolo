"""Native FCN semantic-segmentation inference architecture.

This file is derived from torchvision v0.26.0 at commit
``336d36e8db990a905498c73933e35231876e28bc`` under the BSD-3-Clause
license. See ``docs/provenance/fcn.md`` and the repository notice files.
Copyright (c) Soumith Chintala 2016 and the torchvision contributors.

The 2015 FCN work established end-to-end pixels-to-pixels semantic prediction.
These shipped models are torchvision's later dilated ResNet-50 and ResNet-101
adaptation with a compact FCN head, not the original paper's VGG-based FCN-8s
skip-fusion graph. Training losses are intentionally excluded; the complete
inference graph, including the auxiliary head used by the published
checkpoints, is retained.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import resnet50, resnet101
from torchvision.models._utils import IntermediateLayerGetter

__all__ = ["FCNHead", "LibreFCNModel"]


_PIXEL_MEAN = (0.485, 0.456, 0.406)
_PIXEL_STD = (0.229, 0.224, 0.225)


class FCNHead(nn.Sequential):
    """Project a ResNet feature map to per-class semantic logits."""

    def __init__(self, in_channels: int, channels: int) -> None:
        intermediate_channels = in_channels // 4
        layers = [
            nn.Conv2d(
                in_channels,
                intermediate_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(intermediate_channels, channels, kernel_size=1),
        ]
        super().__init__(*layers)


class LibreFCNModel(nn.Module):
    """Torchvision-compatible FCN graph with optional input normalization."""

    _BACKBONES: dict[str, tuple[Callable[..., nn.Module], int, int]] = {
        "r50": (resnet50, 2048, 1024),
        "r101": (resnet101, 2048, 1024),
    }

    def __init__(
        self,
        size: str = "r50",
        num_classes: int = 21,
        *,
        aux_loss: bool = True,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        try:
            builder, out_channels, aux_channels = self._BACKBONES[size]
        except KeyError as exc:
            raise ValueError(f"Unknown FCN size {size!r}") from exc

        backbone = builder(
            weights=None,
            replace_stride_with_dilation=[False, True, True],
        )
        return_layers = {"layer4": "out"}
        if aux_loss:
            return_layers["layer3"] = "aux"

        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.classifier = FCNHead(out_channels, int(num_classes))
        self.aux_classifier = (
            FCNHead(aux_channels, int(num_classes)) if aux_loss else None
        )
        self.normalize_input = bool(normalize_input)
        self.register_buffer(
            "pixel_mean",
            torch.tensor(_PIXEL_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(_PIXEL_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: Tensor) -> OrderedDict[str, Tensor]:
        input_shape = x.shape[-2:]
        if self.normalize_input:
            x = (x - self.pixel_mean) / self.pixel_std

        features = self.backbone(x)
        result = OrderedDict()

        logits = self.classifier(features["out"])
        result["out"] = F.interpolate(
            logits,
            size=input_shape,
            mode="bilinear",
            align_corners=False,
        )

        if self.aux_classifier is not None:
            aux_logits = self.aux_classifier(features["aux"])
            result["aux"] = F.interpolate(
                aux_logits,
                size=input_shape,
                mode="bilinear",
                align_corners=False,
            )

        return result
