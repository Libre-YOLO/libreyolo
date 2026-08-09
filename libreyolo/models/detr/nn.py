# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0. This file is derived from
# facebookresearch/detr at commit 29901c51d7fe8712168b8d0d64351170bc0f83e0.
# LibreYOLO modifications combine the detection, backbone, positional-encoding,
# and transformer modules; disable runtime backbone downloads; accept a fixed
# batched tensor; and expose an export wrapper. See NOTICE in this directory.
"""Native PyTorch port of the original DETR detection architecture."""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter

SIZE_CONFIGS = {
    "r50": ("resnet50", False),
    "r50dc5": ("resnet50", True),
    "r101": ("resnet101", False),
    "r101dc5": ("resnet101", True),
}


class NestedTensor:
    """Image tensor plus a mask whose true entries denote padded pixels."""

    def __init__(self, tensors: Tensor, mask: Optional[Tensor]) -> None:
        self.tensors = tensors
        self.mask = mask

    def decompose(self) -> tuple[Tensor, Optional[Tensor]]:
        return self.tensors, self.mask


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed statistics and affine parameters."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

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
        tracked_key = prefix + "num_batches_tracked"
        if tracked_key in state_dict:
            del state_dict[tracked_key]
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
        scale = weight * (running_var + 1e-5).rsqrt()
        shifted_bias = bias - running_mean * scale
        return x * scale + shifted_bias


class BackboneBase(nn.Module):
    """Expose the final ResNet stage using DETR's original key layout."""

    def __init__(self, backbone: nn.Module, num_channels: int) -> None:
        super().__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers={"layer4": "0"})
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor) -> Dict[str, NestedTensor]:
        features = self.body(tensor_list.tensors)
        output: Dict[str, NestedTensor] = {}
        for name, feature in features.items():
            mask = tensor_list.mask
            if mask is None:
                raise ValueError("DETR requires an image padding mask")
            resized_mask = F.interpolate(
                mask[None].float(), size=feature.shape[-2:]
            ).to(torch.bool)[0]
            output[name] = NestedTensor(feature, resized_mask)
        return output


class Backbone(BackboneBase):
    """ResNet-50/101 backbone with DETR's FrozenBatchNorm2d."""

    def __init__(self, name: str, dilation: bool) -> None:
        # Official DETR initialized an ImageNet backbone here and immediately
        # overwrote it with the detector checkpoint. LibreYOLO builds without a
        # download; the serialized module graph and keys are identical.
        backbone = getattr(torchvision.models, name)(
            weights=None,
            replace_stride_with_dilation=[False, False, dilation],
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 2048
        super().__init__(backbone, num_channels)


class PositionEmbeddingSine(nn.Module):
    """Two-dimensional sine/cosine positional encoding."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize must be true when scale is supplied")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, tensor_list: NestedTensor) -> Tensor:
        x = tensor_list.tensors
        mask = tensor_list.mask
        if mask is None:
            raise ValueError("DETR requires an image padding mask")
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

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


class Joiner(nn.Sequential):
    """Pair backbone feature maps with their positional encodings."""

    def __init__(self, backbone: Backbone, position_embedding: nn.Module) -> None:
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(
        self, tensor_list: NestedTensor
    ) -> tuple[List[NestedTensor], List[Tensor]]:
        features = self[0](tensor_list)
        output: List[NestedTensor] = []
        positions: List[Tensor] = []
        for feature in features.values():
            output.append(feature)
            positions.append(self[1](feature).to(feature.tensors.dtype))
        return output, positions


class Transformer(nn.Module):
    """DETR transformer with explicit positional inputs."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        return_intermediate_dec: bool = False,
    ) -> None:
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self, src: Tensor, mask: Tensor, query_embed: Tensor, pos_embed: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch_size, channels, height, width = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        mask = mask.flatten(1)

        target = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hidden = self.decoder(
            target,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return (
            hidden.transpose(1, 2),
            memory.permute(1, 2, 0).view(batch_size, channels, height, width),
        )


class TransformerEncoder(nn.Module):
    def __init__(
        self, encoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module]
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module],
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        query = key = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            query,
            key,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        src2 = self.norm1(src)
        query = key = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            query,
            key,
            value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        return src + self.dropout2(src2)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        query = key = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            query,
            key,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return self.norm3(tgt)

    def forward_pre(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        tgt2 = self.norm1(tgt)
        query = key = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            query,
            key,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        return tgt + self.dropout3(tgt2)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class MLP(nn.Module):
    """Three-layer box regression feed-forward network."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip(
                [input_dim] + hidden, hidden + [output_dim], strict=True
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class LibreDETRModel(nn.Module):
    """Vanilla DETR with state-dict names matching the official checkpoints."""

    def __init__(self, size: str, nc: int, num_queries: int = 100) -> None:
        super().__init__()
        try:
            backbone_name, dilation = SIZE_CONFIGS[size]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported DETR size {size!r}; choose one of {tuple(SIZE_CONFIGS)}"
            ) from exc

        hidden_dim = 256
        backbone = Backbone(backbone_name, dilation=dilation)
        self.backbone = Joiner(
            backbone,
            PositionEmbeddingSine(hidden_dim // 2, normalize=True),
        )
        self.transformer = Transformer(
            d_model=hidden_dim,
            return_intermediate_dec=True,
        )
        self.num_queries = num_queries
        self.class_embed = nn.Linear(hidden_dim, nc + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, hidden_dim, kernel_size=1
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW input, got shape {tuple(x.shape)}")
        # LibreYOLO preprocesses every image to the same fixed square, so this
        # batch has no padding. This is exactly the mask official DETR creates
        # when its forward receives a same-shaped tensor batch.
        mask = torch.zeros(
            (x.shape[0], x.shape[-2], x.shape[-1]),
            dtype=torch.bool,
            device=x.device,
        )
        features, positions = self.backbone(NestedTensor(x, mask))
        source, feature_mask = features[-1].decompose()
        if feature_mask is None:
            raise ValueError("DETR backbone returned no padding mask")
        hidden = self.transformer(
            self.input_proj(source),
            feature_mask,
            self.query_embed.weight,
            positions[-1],
        )[0]
        output_classes = self.class_embed(hidden)
        output_boxes = self.bbox_embed(hidden).sigmoid()
        return {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_boxes[-1],
        }


class DETRExportWrapper(nn.Module):
    """Flatten the DETR output dictionary into two named graph outputs."""

    def __init__(self, model: LibreDETRModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        output = self.model(x)
        return output["pred_logits"], output["pred_boxes"]


def _get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation must be relu, gelu, or glu, not {activation!r}")
