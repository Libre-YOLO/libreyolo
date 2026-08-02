"""Native FCOS ResNet-50/FPN inference graph.

This file is derived from torchvision v0.26.0 at commit
``336d36e8db990a905498c73933e35231876e28bc`` under the BSD-3-Clause
license. See ``docs/provenance/fcos.md`` and the repository notice files.
Copyright (c) Soumith Chintala 2016 and the torchvision contributors.

FCOS established the fully convolutional anchor-free detector pattern: each
feature-map location predicts a class, four box-edge distances, and a
centerness score. Training-only assignment and loss code is intentionally out
of scope; this module keeps the official inference state-dict layout and raw
head numerics.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import (
    ExtraFPNBlock,
    FeaturePyramidNetwork,
    LastLevelP6P7,
)
from torchvision.ops.misc import FrozenBatchNorm2d

__all__ = [
    "FCOSAnchorGenerator",
    "FCOSClassificationHead",
    "FCOSHead",
    "FCOSRegressionHead",
    "LibreFCOSModel",
]


class BackboneWithFPN(nn.Module):
    """Expose ResNet stages 2-4 through the official five-level FCOS FPN."""

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: dict[str, str],
        in_channels_list: list[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        return self.fpn(self.body(images))


def _resnet50_fpn_backbone() -> BackboneWithFPN:
    """Build the checkpoint-compatible ResNet-50 + P3-P7 feature pyramid."""
    backbone = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)

    # The official checkpoint freezes stem/layer1 during training. Preserve
    # that parameter contract even though this port is inference-only.
    layers_to_train = ("layer2", "layer3", "layer4")
    for name, parameter in backbone.named_parameters():
        if not name.startswith(layers_to_train):
            parameter.requires_grad_(False)

    returned_layers = [2, 3, 4]
    return_layers = {
        f"layer{layer}": str(index) for index, layer in enumerate(returned_layers)
    }
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2 * 2 ** (layer - 1) for layer in returned_layers
    ]
    return BackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        256,
        extra_blocks=LastLevelP6P7(256, 256),
    )


class FCOSClassificationHead(nn.Module):
    """Four-convolution classification tower with the official prior bias."""

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        num_convs: int = 4,
        prior_probability: float = 0.01,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_anchors = int(num_anchors)
        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32)

        conv: list[nn.Module] = []
        for _ in range(num_convs):
            conv.extend(
                (
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    norm_layer(in_channels),
                    nn.ReLU(),
                )
            )
        self.conv = nn.Sequential(*conv)
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(
            in_channels,
            num_anchors * num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.constant_(
            self.cls_logits.bias,
            -math.log((1 - prior_probability) / prior_probability),
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        outputs: list[Tensor] = []
        for feature in features:
            logits = self.cls_logits(self.conv(feature))
            batch, _, height, width = logits.shape
            logits = logits.view(batch, -1, self.num_classes, height, width)
            logits = logits.permute(0, 3, 4, 1, 2)
            outputs.append(logits.reshape(batch, -1, self.num_classes))
        return torch.cat(outputs, dim=1)


class FCOSRegressionHead(nn.Module):
    """Four-convolution box tower with ltrb and centerness predictions."""

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_convs: int = 4,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32)

        conv: list[nn.Module] = []
        for _ in range(num_convs):
            conv.extend(
                (
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    norm_layer(in_channels),
                    nn.ReLU(),
                )
            )
        self.conv = nn.Sequential(*conv)
        self.bbox_reg = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, num_anchors, kernel_size=3, stride=1, padding=1
        )
        for layer in (self.bbox_reg, self.bbox_ctrness):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, features: list[Tensor]) -> tuple[Tensor, Tensor]:
        all_regression: list[Tensor] = []
        all_ctrness: list[Tensor] = []
        for feature in features:
            tower = self.conv(feature)
            regression = nn.functional.relu(self.bbox_reg(tower))
            ctrness = self.bbox_ctrness(tower)

            batch, _, height, width = regression.shape
            regression = regression.view(batch, -1, 4, height, width)
            regression = regression.permute(0, 3, 4, 1, 2)
            all_regression.append(regression.reshape(batch, -1, 4))

            ctrness = ctrness.view(batch, -1, 1, height, width)
            ctrness = ctrness.permute(0, 3, 4, 1, 2)
            all_ctrness.append(ctrness.reshape(batch, -1, 1))

        return torch.cat(all_regression, dim=1), torch.cat(all_ctrness, dim=1)


class FCOSHead(nn.Module):
    """Checkpoint-compatible classification and regression head."""

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        num_convs: int = 4,
    ) -> None:
        super().__init__()
        self.classification_head = FCOSClassificationHead(
            in_channels, num_anchors, num_classes, num_convs
        )
        self.regression_head = FCOSRegressionHead(in_channels, num_anchors, num_convs)

    def forward(self, features: list[Tensor]) -> dict[str, Tensor]:
        cls_logits = self.classification_head(features)
        bbox_regression, bbox_ctrness = self.regression_head(features)
        return {
            "cls_logits": cls_logits,
            "bbox_regression": bbox_regression,
            "bbox_ctrness": bbox_ctrness,
        }


class FCOSAnchorGenerator(nn.Module):
    """Generate the one square anchor per FCOS feature-map location."""

    def __init__(self) -> None:
        super().__init__()
        self.sizes = ((8,), (16,), (32,), (64,), (128,))
        self.cell_anchors = [self._generate_anchor(size) for size in self.sizes]

    @staticmethod
    def _generate_anchor(scales: tuple[int, ...]) -> Tensor:
        scales_tensor = torch.as_tensor(scales, dtype=torch.float32)
        widths = scales_tensor
        heights = scales_tensor
        anchors = torch.stack((-widths, -heights, widths, heights), dim=1) / 2
        return anchors.round()

    def forward(
        self, images: Tensor, feature_maps: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        grid_sizes = [feature.shape[-2:] for feature in feature_maps]
        image_height, image_width = images.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        anchors_per_level: list[Tensor] = []
        level_sizes: list[int] = []

        for grid_size, base_anchor in zip(grid_sizes, self.cell_anchors):
            grid_height, grid_width = grid_size
            stride_height = torch.empty((), dtype=torch.int64, device=device).fill_(
                image_height // grid_height
            )
            stride_width = torch.empty((), dtype=torch.int64, device=device).fill_(
                image_width // grid_width
            )
            shifts_x = (
                torch.arange(0, grid_width, dtype=torch.int32, device=device)
                * stride_width
            )
            shifts_y = (
                torch.arange(0, grid_height, dtype=torch.int32, device=device)
                * stride_height
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack(
                (
                    shift_x.reshape(-1),
                    shift_y.reshape(-1),
                    shift_x.reshape(-1),
                    shift_y.reshape(-1),
                ),
                dim=1,
            )
            base = base_anchor.to(dtype=dtype, device=device)
            anchors_per_level.append(
                (shifts.view(-1, 1, 4) + base.view(1, -1, 4)).reshape(-1, 4)
            )
            level_sizes.append(int(grid_height * grid_width))

        anchors = torch.cat(anchors_per_level, dim=0)
        anchors = anchors.unsqueeze(0).expand(images.shape[0], -1, -1)
        sizes = torch.tensor(level_sizes, dtype=torch.int64, device=device)
        sizes = sizes.unsqueeze(0).expand(images.shape[0], -1)
        return anchors, sizes


class LibreFCOSModel(nn.Module):
    """Checkpoint-compatible FCOS graph returning unfiltered raw predictions."""

    def __init__(self, num_classes: int = 91) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone = _resnet50_fpn_backbone()
        self.head = FCOSHead(self.backbone.out_channels, 1, self.num_classes)
        self.anchor_generator = FCOSAnchorGenerator()

    def forward_head(self, images: Tensor) -> tuple[dict[str, Tensor], list[Tensor]]:
        """Run the official backbone and head on already-preprocessed images."""
        features = list(self.backbone(images).values())
        return self.head(features), features

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        if images.ndim != 4:
            raise ValueError("FCOS expects a BCHW image tensor")
        head_outputs, features = self.forward_head(images)
        anchors, level_sizes = self.anchor_generator(images, features)
        return {
            **head_outputs,
            "anchors": anchors,
            "level_sizes": level_sizes,
        }
