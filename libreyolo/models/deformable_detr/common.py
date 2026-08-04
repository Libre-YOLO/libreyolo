# SPDX-License-Identifier: Apache-2.0
# Ported from https://github.com/fundamentalvision/Deformable-DETR
# commit 11169a60c33333af00a4849f1808023eba96a931.
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Modified from DETR (https://github.com/facebookresearch/detr).
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Tensor containers, positional encoding, and ResNet backbone."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter


class NestedTensor:
    """Image batch paired with a mask that marks padded pixels."""

    def __init__(self, tensors: Tensor, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking: bool = False) -> "NestedTensor":
        tensors = self.tensors.to(device, non_blocking=non_blocking)
        mask = (
            self.mask.to(device, non_blocking=non_blocking)
            if self.mask is not None
            else None
        )
        return NestedTensor(tensors, mask)

    def decompose(self) -> tuple[Tensor, Optional[Tensor]]:
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)


def _max_by_axis(shapes: list[list[int]]) -> list[int]:
    maximums = shapes[0]
    for shape in shapes[1:]:
        for index, item in enumerate(shape):
            maximums[index] = max(maximums[index], item)
    return maximums


def nested_tensor_from_tensor_list(
    tensor_list: Tensor | list[Tensor] | tuple[Tensor, ...],
) -> NestedTensor:
    """Pad a list of CHW tensors, or wrap an already batched BCHW tensor."""
    if isinstance(tensor_list, Tensor):
        if tensor_list.ndim == 3:
            tensor_list = tensor_list.unsqueeze(0)
        if tensor_list.ndim != 4:
            raise ValueError("Expected a CHW or BCHW image tensor")
        mask = torch.zeros(
            tensor_list.shape[0],
            tensor_list.shape[-2],
            tensor_list.shape[-1],
            dtype=torch.bool,
            device=tensor_list.device,
        )
        return NestedTensor(tensor_list, mask)

    images = list(tensor_list)
    if not images or images[0].ndim != 3:
        raise ValueError("Expected a non-empty sequence of CHW image tensors")
    max_size = _max_by_axis([list(image.shape) for image in images])
    batch_shape = [len(images), *max_size]
    tensor = torch.zeros(batch_shape, dtype=images[0].dtype, device=images[0].device)
    mask = torch.ones(
        (batch_shape[0], batch_shape[2], batch_shape[3]),
        dtype=torch.bool,
        device=images[0].device,
    )
    for image, padded, image_mask in zip(images, tensor, mask):
        padded[: image.shape[0], : image.shape[1], : image.shape[2]].copy_(image)
        image_mask[: image.shape[1], : image.shape[2]] = False
    return NestedTensor(tensor, mask)


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Numerically stable inverse of sigmoid on values in [0, 1]."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class PositionEmbeddingSine(nn.Module):
    """Two-dimensional sine/cosine positional encoding."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, tensor_list: NestedTensor) -> Tensor:
        x = tensor_list.tensors
        mask = tensor_list.mask
        if mask is None:
            raise ValueError("PositionEmbeddingSine requires a padding mask")
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class FrozenBatchNorm2d(nn.Module):
    """Batch normalization with fixed statistics and affine parameters."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + self.eps).rsqrt()
        return x * scale + (bias - running_mean * scale)


class BackboneBase(nn.Module):
    """Expose selected ResNet stages as nested tensors."""

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        return_interm_layers: bool,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or not any(
                stage in name for stage in ("layer2", "layer3", "layer4")
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor) -> OrderedDict[str, NestedTensor]:
        xs = self.body(tensor_list.tensors)
        output: OrderedDict[str, NestedTensor] = OrderedDict()
        for name, x in xs.items():
            mask = tensor_list.mask
            if mask is None:
                raise ValueError("Backbone requires a padding mask")
            resized_mask = F.interpolate(mask[None].float(), size=x.shape[-2:])
            output[name] = NestedTensor(x, resized_mask.to(torch.bool)[0])
        return output


class Backbone(BackboneBase):
    """ResNet-50 backbone with frozen batch normalization."""

    def __init__(self, return_interm_layers: bool, dilation: bool):
        backbone = resnet50(
            weights=None,
            replace_stride_with_dilation=[False, False, dilation],
            norm_layer=FrozenBatchNorm2d,
        )
        super().__init__(
            backbone,
            train_backbone=False,
            return_interm_layers=return_interm_layers,
        )
        if dilation:
            self.strides[-1] //= 2


class Joiner(nn.Sequential):
    """Compose the visual backbone and positional encoder."""

    def __init__(self, backbone: Backbone, position_embedding: PositionEmbeddingSine):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(
        self, tensor_list: NestedTensor
    ) -> tuple[list[NestedTensor], list[Tensor]]:
        xs = self[0](tensor_list)
        output = [x for _, x in sorted(xs.items())]
        positions = [self[1](x).to(x.tensors.dtype) for x in output]
        return output, positions


def build_backbone(num_feature_levels: int, dilation: bool) -> Joiner:
    """Build the released ResNet-50 + sine-position backbone."""
    position_embedding = PositionEmbeddingSine(128, normalize=True)
    backbone = Backbone(
        return_interm_layers=num_feature_levels > 1,
        dilation=dilation,
    )
    return Joiner(backbone, position_embedding)


__all__ = [
    "Backbone",
    "FrozenBatchNorm2d",
    "Joiner",
    "NestedTensor",
    "PositionEmbeddingSine",
    "build_backbone",
    "inverse_sigmoid",
    "nested_tensor_from_tensor_list",
]
