"""Native RetinaNet inference architecture.

This file is derived from torchvision v0.26.0 at commit
``336d36e8db990a905498c73933e35231876e28bc`` under the BSD-3-Clause
license. See ``docs/provenance/retinanet.md`` and the repository notice files.
Copyright (c) Soumith Chintala 2016 and the torchvision contributors.

The graph retains torchvision's state-dict names for exact checkpoint loading.
Training-only matching and focal-loss code are intentionally excluded.
"""

from __future__ import annotations

import math
from collections import OrderedDict
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

from ...utils.coco import COCO91_TO_COCO80

__all__ = [
    "LibreRetinaNetModel",
    "RetinaNetClassificationHead",
    "RetinaNetHead",
    "RetinaNetRegressionHead",
]


_ANCHOR_SIZES = tuple(
    (base, int(base * 2 ** (1.0 / 3)), int(base * 2 ** (2.0 / 3)))
    for base in (32, 64, 128, 256, 512)
)
_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(_ANCHOR_SIZES)
_COCO91_SOURCE_INDICES = tuple(
    source for source, _target in sorted(COCO91_TO_COCO80.items(), key=lambda item: item[1])
)


def _v1_to_v2_weights(state_dict, prefix: str) -> None:
    """Translate torchvision's pre-v2 sequential-conv key layout in-place."""
    for index in range(4):
        for parameter_type in ("weight", "bias"):
            old_key = f"{prefix}conv.{2 * index}.{parameter_type}"
            new_key = f"{prefix}conv.{index}.0.{parameter_type}"
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)


class _Conv2dNormActivation(nn.Sequential):
    """The exact Conv2d-[Norm]-ReLU block used by torchvision's heads."""

    def __init__(
        self,
        channels: int,
        norm_layer: Optional[Callable[..., nn.Module]],
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=norm_layer is None,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(channels))
        layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class RetinaNetClassificationHead(nn.Module):
    """Four shared convolutions and per-anchor sigmoid class logits."""

    _version = 2

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        prior_probability: float = 0.01,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            *[
                _Conv2dNormActivation(in_channels, norm_layer)
                for _ in range(4)
            ]
        )
        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
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
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        version = local_metadata.get("version")
        if version is None or version < 2:
            _v1_to_v2_weights(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        all_logits = []
        for feature in features:
            logits = self.cls_logits(self.conv(feature))
            batch, _, height, width = logits.shape
            logits = logits.view(
                batch, -1, self.num_classes, height, width
            )
            logits = logits.permute(0, 3, 4, 1, 2)
            all_logits.append(logits.reshape(batch, -1, self.num_classes))
        return torch.cat(all_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    """Four shared convolutions and four box deltas per anchor."""

    _version = 2

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            *[
                _Conv2dNormActivation(in_channels, norm_layer)
                for _ in range(4)
            ]
        )
        self.bbox_reg = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.normal_(self.bbox_reg.weight, std=0.01)
        nn.init.zeros_(self.bbox_reg.bias)
        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        version = local_metadata.get("version")
        if version is None or version < 2:
            _v1_to_v2_weights(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        all_regression = []
        for feature in features:
            regression = self.bbox_reg(self.conv(feature))
            batch, _, height, width = regression.shape
            regression = regression.view(batch, -1, 4, height, width)
            regression = regression.permute(0, 3, 4, 1, 2)
            all_regression.append(regression.reshape(batch, -1, 4))
        return torch.cat(all_regression, dim=1)


class RetinaNetHead(nn.Module):
    """Paired classification and box-regression subnetworks."""

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.classification_head = RetinaNetClassificationHead(
            in_channels,
            num_anchors,
            num_classes,
            norm_layer=norm_layer,
        )
        self.regression_head = RetinaNetRegressionHead(
            in_channels,
            num_anchors,
            norm_layer=norm_layer,
        )

    def forward(self, features: list[Tensor]) -> dict[str, Tensor]:
        return {
            "cls_logits": self.classification_head(features),
            "bbox_regression": self.regression_head(features),
        }


class BackboneWithFPN(nn.Module):
    """Expose ResNet C3-C5 through a five-level P3-P7 FPN."""

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: dict[str, str],
        in_channels_list: list[int],
        out_channels: int,
        extra_blocks: ExtraFPNBlock,
    ) -> None:
        super().__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list,
            out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        return self.fpn(self.body(images))


def _resnet_fpn_backbone(*, v2: bool) -> BackboneWithFPN:
    norm_layer = nn.BatchNorm2d if v2 else FrozenBatchNorm2d
    backbone = resnet50(weights=None, norm_layer=norm_layer)
    return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
    extra_input_channels = 2048 if v2 else 256
    return BackboneWithFPN(
        backbone,
        return_layers,
        [512, 1024, 2048],
        256,
        LastLevelP6P7(extra_input_channels, 256),
    )


class AnchorGenerator(nn.Module):
    """Generate torchvision-compatible RetinaNet anchors for P3-P7."""

    __annotations__ = {"cell_anchors": list[Tensor]}

    def __init__(
        self,
        sizes: tuple[tuple[int, ...], ...] = _ANCHOR_SIZES,
        aspect_ratios: tuple[tuple[float, ...], ...] = _ASPECT_RATIOS,
    ) -> None:
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [
            self.generate_anchors(size, ratio)
            for size, ratio in zip(sizes, aspect_ratios)
        ]

    @staticmethod
    def generate_anchors(
        scales,
        aspect_ratios,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        scales_tensor = torch.as_tensor(scales, dtype=dtype, device=device)
        ratios_tensor = torch.as_tensor(
            aspect_ratios, dtype=dtype, device=device
        )
        height_ratios = torch.sqrt(ratios_tensor)
        width_ratios = 1 / height_ratios
        widths = (width_ratios[:, None] * scales_tensor[None, :]).reshape(-1)
        heights = (height_ratios[:, None] * scales_tensor[None, :]).reshape(-1)
        return (
            torch.stack((-widths, -heights, widths, heights), dim=1) / 2
        ).round()

    def num_anchors_per_location(self) -> list[int]:
        return [
            len(sizes) * len(ratios)
            for sizes, ratios in zip(self.sizes, self.aspect_ratios)
        ]

    def _set_cell_anchors(self, dtype: torch.dtype, device: torch.device) -> None:
        self.cell_anchors = [
            anchor.to(dtype=dtype, device=device) for anchor in self.cell_anchors
        ]

    def forward(
        self,
        image_size,
        feature_maps: list[Tensor],
    ) -> tuple[Tensor, list[int]]:
        grid_sizes = [feature.shape[-2:] for feature in feature_maps]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        self._set_cell_anchors(dtype, device)
        anchors_per_level: list[Tensor] = []
        level_counts: list[int] = []
        for grid_size, base_anchors in zip(grid_sizes, self.cell_anchors):
            grid_height, grid_width = grid_size
            stride_height = torch.empty(
                (), dtype=torch.int64, device=device
            ).fill_(image_size[0] // grid_height)
            stride_width = torch.empty(
                (), dtype=torch.int64, device=device
            ).fill_(image_size[1] // grid_width)
            shifts_x = (
                torch.arange(
                    0, grid_width, dtype=torch.int32, device=device
                )
                * stride_width
            )
            shifts_y = (
                torch.arange(
                    0, grid_height, dtype=torch.int32, device=device
                )
                * stride_height
            )
            shift_y, shift_x = torch.meshgrid(
                shifts_y, shifts_x, indexing="ij"
            )
            shifts = torch.stack(
                (
                    shift_x.reshape(-1),
                    shift_y.reshape(-1),
                    shift_x.reshape(-1),
                    shift_y.reshape(-1),
                ),
                dim=1,
            )
            level = (
                shifts.reshape(-1, 1, 4) + base_anchors.reshape(1, -1, 4)
            ).reshape(-1, 4)
            anchors_per_level.append(level)
            level_counts.append(level.shape[0])
        return torch.cat(anchors_per_level, dim=0), level_counts


class BoxCoder:
    """Decode RetinaNet's anchor-relative box deltas."""

    def __init__(
        self,
        weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        bbox_xform_clip: float = math.log(1000.0 / 16),
    ) -> None:
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        boxes = boxes.to(rel_codes.dtype)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        center_x = boxes[:, 0] + 0.5 * widths
        center_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = torch.clamp(rel_codes[:, 2::4] / ww, max=self.bbox_xform_clip)
        dh = torch.clamp(rel_codes[:, 3::4] / wh, max=self.bbox_xform_clip)

        predicted_center_x = dx * widths[:, None] + center_x[:, None]
        predicted_center_y = dy * heights[:, None] + center_y[:, None]
        predicted_width = torch.exp(dw) * widths[:, None]
        predicted_height = torch.exp(dh) * heights[:, None]
        half_width = torch.tensor(
            0.5, dtype=predicted_center_x.dtype, device=predicted_width.device
        ) * predicted_width
        half_height = torch.tensor(
            0.5, dtype=predicted_center_y.dtype, device=predicted_height.device
        ) * predicted_height
        return torch.stack(
            (
                predicted_center_x - half_width,
                predicted_center_y - half_height,
                predicted_center_x + half_width,
                predicted_center_y + half_height,
            ),
            dim=2,
        ).flatten(1)


class LibreRetinaNetModel(nn.Module):
    """Inference-only RetinaNet graph with decoded export-friendly output."""

    def __init__(self, size: str, num_classes: int) -> None:
        super().__init__()
        if size not in {"r50", "r50v2"}:
            raise ValueError(f"Unsupported RetinaNet size: {size}")
        self.size = size
        self.num_classes = num_classes
        self.backbone = _resnet_fpn_backbone(v2=size == "r50v2")
        self.anchor_generator = AnchorGenerator()
        norm_layer = partial(nn.GroupNorm, 32) if size == "r50v2" else None
        self.head = RetinaNetHead(
            self.backbone.out_channels,
            self.anchor_generator.num_anchors_per_location()[0],
            num_classes,
            norm_layer=norm_layer,
        )
        self.box_coder = BoxCoder()
        self.register_buffer(
            "_coco91_source_indices",
            torch.tensor(_COCO91_SOURCE_INDICES, dtype=torch.int64),
            persistent=False,
        )

        if size == "r50":
            # The released v1 checkpoint is evaluated with FrozenBN eps=0.
            for module in self.modules():
                if isinstance(module, FrozenBatchNorm2d):
                    module.eps = 0.0

    def forward_head(
        self, images: Tensor
    ) -> tuple[dict[str, Tensor], list[Tensor]]:
        """Return raw heads and ordered P3-P7 features for parity tests."""
        features_dict = self.backbone(images)
        if isinstance(features_dict, Tensor):
            features_dict = OrderedDict([("0", features_dict)])
        features = list(features_dict.values())
        return self.head(features), features

    def forward(self, images: Tensor) -> Tensor:
        """Return ``(B, anchors, 4 + classes)`` decoded boxes and scores."""
        head_outputs, features = self.forward_head(images)
        anchors, _ = self.anchor_generator(images.shape[-2:], features)
        decoded = torch.stack(
            [
                self.box_coder.decode_single(regression, anchors)
                for regression in head_outputs["bbox_regression"]
            ]
        )
        scores = torch.sigmoid(head_outputs["cls_logits"])
        if self.num_classes == 91:
            scores = scores.index_select(2, self._coco91_source_indices)
        return torch.cat((decoded, scores), dim=2)
