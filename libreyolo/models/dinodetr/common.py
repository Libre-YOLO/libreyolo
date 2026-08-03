# SPDX-License-Identifier: Apache-2.0
# Ported from https://github.com/IDEA-Research/DINO at
# d84a491d41898b3befd8294d1cf2614661fc0953.
# Copyright 2022 IDEA.
"""Backbones, tensor containers, and positional encoding for DINO-DETR."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter

from ..deformable_detr.common import (
    FrozenBatchNorm2d,
    NestedTensor,
    nested_tensor_from_tensor_list,
)


def inverse_sigmoid(x: Tensor, eps: float = 1e-3) -> Tensor:
    """DINO's bounded logit transform (its epsilon differs from Deformable DETR)."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class PositionEmbeddingSineHW(nn.Module):
    """DINO's axis-specific sine/cosine positional encoding."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature_h: int = 10000,
        temperature_w: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature_h = temperature_h
        self.temperature_w = temperature_w
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize must be true when scale is supplied")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, tensor_list: NestedTensor) -> Tensor:
        images = tensor_list.tensors
        mask = tensor_list.mask
        if mask is None:
            raise ValueError("DINO positional encoding requires a padding mask")
        valid = ~mask
        y_embed = valid.cumsum(1, dtype=torch.float32)
        x_embed = valid.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_x = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=images.device
        )
        dim_y = dim_x.clone()
        dim_x = self.temperature_w ** (
            2 * (dim_x.div(2, rounding_mode="floor")) / self.num_pos_feats
        )
        dim_y = self.temperature_h ** (
            2 * (dim_y.div(2, rounding_mode="floor")) / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dim_x
        pos_y = y_embed[:, :, :, None] / dim_y
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class ResNetBackbone(nn.Module):
    """ResNet-50 with DINO's selected intermediate stages."""

    def __init__(self, return_interm_indices: tuple[int, ...]):
        super().__init__()
        if return_interm_indices not in ((1, 2, 3), (0, 1, 2, 3)):
            raise ValueError(
                f"Unsupported DINO stage selection: {return_interm_indices}"
            )
        backbone = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
        for name, parameter in backbone.named_parameters():
            if not any(stage in name for stage in ("layer2", "layer3", "layer4")):
                parameter.requires_grad_(False)
        return_layers = {
            f"layer{5 - len(return_interm_indices) + index}": str(layer_index)
            for index, layer_index in enumerate(return_interm_indices)
        }
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        all_channels = [256, 512, 1024, 2048]
        self.num_channels = all_channels[4 - len(return_interm_indices) :]

    def forward(self, tensor_list: NestedTensor) -> OrderedDict[str, NestedTensor]:
        features = self.body(tensor_list.tensors)
        output: OrderedDict[str, NestedTensor] = OrderedDict()
        for name, feature in features.items():
            mask = tensor_list.mask
            if mask is None:
                raise ValueError("DINO backbone requires a padding mask")
            resized = F.interpolate(mask[None].float(), size=feature.shape[-2:])
            output[name] = NestedTensor(feature, resized.to(torch.bool)[0])
        return output


class Joiner(nn.Sequential):
    """Combine a visual backbone with DINO positional encoding."""

    def __init__(self, backbone: nn.Module, position_embedding: nn.Module):
        super().__init__(backbone, position_embedding)
        self.num_channels = list(backbone.num_channels)

    def forward(
        self, tensor_list: NestedTensor
    ) -> tuple[list[NestedTensor], list[Tensor]]:
        features = self[0](tensor_list)
        output = list(features.values())
        positions = [self[1](feature).to(feature.tensors.dtype) for feature in output]
        return output, positions


def build_backbone(size: str) -> Joiner:
    """Build one of the three released DINO-DETR visual backbones."""
    position = PositionEmbeddingSineHW(
        128, temperature_h=20, temperature_w=20, normalize=True
    )
    if size == "r50":
        backbone: nn.Module = ResNetBackbone((1, 2, 3))
    elif size == "r50s5":
        backbone = ResNetBackbone((0, 1, 2, 3))
    elif size == "swinl":
        from .swin import SwinTransformer

        backbone = SwinTransformer()
    else:
        raise ValueError(f"Unknown DINO-DETR size {size!r}")
    return Joiner(backbone, position)


__all__ = [
    "FrozenBatchNorm2d",
    "Joiner",
    "NestedTensor",
    "PositionEmbeddingSineHW",
    "ResNetBackbone",
    "build_backbone",
    "inverse_sigmoid",
    "nested_tensor_from_tensor_list",
]
