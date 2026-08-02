# SPDX-License-Identifier: Apache-2.0
# Ported from https://github.com/fundamentalvision/Deformable-DETR
# commit 11169a60c33333af00a4849f1808023eba96a931.
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Modified from DETR (https://github.com/facebookresearch/detr).
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Multi-scale deformable encoder and decoder."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from .common import inverse_sigmoid
from .ms_deform_attn import MSDeformAttn


def _get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}")


class DeformableTransformer(nn.Module):
    """Six-layer deformable encoder/decoder used by released checkpoints."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        return_intermediate_dec: bool = False,
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        enc_n_points: int = 4,
        two_stage: bool = False,
        two_stage_num_proposals: int = 300,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec
        )
        self.level_embed = nn.Parameter(torch.empty(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)
        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        for module in self.modules():
            if isinstance(module, MSDeformAttn):
                module._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals: Tensor) -> Tensor:
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device
        )
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        proposals = proposals.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        return torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)

    def gen_encoder_output_proposals(
        self,
        memory: Tensor,
        memory_padding_mask: Tensor,
        spatial_shapes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, _, _ = memory.shape
        proposals = []
        current = 0
        for level, (height, width) in enumerate(spatial_shapes):
            mask = memory_padding_mask[:, current : current + height * width].view(
                batch, height, width, 1
            )
            valid_height = torch.sum(~mask[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0,
                    height - 1,
                    height,
                    dtype=torch.float32,
                    device=memory.device,
                ),
                torch.linspace(
                    0,
                    width - 1,
                    width,
                    dtype=torch.float32,
                    device=memory.device,
                ),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat(
                [valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1
            ).view(batch, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch, -1, -1, -1) + 0.5) / scale
            width_height = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposals.append(torch.cat((grid, width_height), -1).view(batch, -1, 4))
            current += height * width

        output_proposals = torch.cat(proposals, 1)
        valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
            -1, keepdim=True
        )
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~valid, float("inf"))

        output_memory = memory.masked_fill(memory_padding_mask.unsqueeze(-1), 0.0)
        output_memory = output_memory.masked_fill(~valid, 0.0)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        return torch.stack(
            [valid_width.float() / width, valid_height.float() / height], -1
        )

    def forward(
        self,
        srcs: list[Tensor],
        masks: list[Tensor],
        pos_embeds: list[Tensor],
        query_embed: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        if not self.two_stage and query_embed is None:
            raise ValueError("One-stage Deformable DETR requires query embeddings")

        src_flatten = []
        mask_flatten = []
        level_pos_flatten = []
        spatial_shapes = []
        for level, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            batch, channels, height, width = src.shape
            del batch, channels
            spatial_shapes.append((height, width))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            level_pos_flatten.append(pos_embed + self.level_embed[level].view(1, 1, -1))
        src_flatten_tensor = torch.cat(src_flatten, 1)
        mask_flatten_tensor = torch.cat(mask_flatten, 1)
        level_pos_tensor = torch.cat(level_pos_flatten, 1)
        spatial_shapes_tensor = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten_tensor.device
        )
        level_start_index = torch.cat(
            (
                spatial_shapes_tensor.new_zeros((1,)),
                spatial_shapes_tensor.prod(1).cumsum(0)[:-1],
            )
        )
        valid_ratios = torch.stack([self.get_valid_ratio(mask) for mask in masks], 1)

        memory = self.encoder(
            src_flatten_tensor,
            spatial_shapes_tensor,
            level_start_index,
            valid_ratios,
            level_pos_tensor,
            mask_flatten_tensor,
        )

        batch, _, channels = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten_tensor, spatial_shapes_tensor
            )
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](
                output_memory
            )
            enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory)
                + output_proposals
            )

            topk_indices = torch.topk(
                enc_outputs_class[..., 0],
                self.two_stage_num_proposals,
                dim=1,
            )[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact,
                1,
                topk_indices.unsqueeze(-1).repeat(1, 1, 4),
            ).detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
            )
            query_embed, target = torch.split(pos_trans_out, channels, dim=2)
        else:
            query_embed, target = torch.split(query_embed, channels, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(batch, -1, -1)
            target = target.unsqueeze(0).expand(batch, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points
            enc_outputs_class = None
            enc_outputs_coord_unact = None

        hidden_states, inter_references = self.decoder(
            target,
            reference_points,
            memory,
            spatial_shapes_tensor,
            level_start_index,
            valid_ratios,
            query_embed,
            mask_flatten_tensor,
        )
        return (
            hidden_states,
            init_reference_out,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        )


class DeformableTransformerEncoderLayer(nn.Module):
    """One deformable self-attention encoder block."""

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src: Tensor) -> Tensor:
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout3(src2))

    def forward(
        self,
        src: Tensor,
        pos: Tensor | None,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = self.norm1(src + self.dropout1(src2))
        return self.forward_ffn(src)


class DeformableTransformerEncoder(nn.Module):
    """Stack of deformable encoder blocks."""

    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(
        spatial_shapes: Tensor, valid_ratios: Tensor, device: torch.device
    ) -> Tensor:
        reference_points = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5,
                    height - 0.5,
                    height,
                    dtype=torch.float32,
                    device=device,
                ),
                torch.linspace(
                    0.5,
                    width - 0.5,
                    width,
                    dtype=torch.float32,
                    device=device,
                ),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            reference_points.append(torch.stack((ref_x, ref_y), -1))
        reference_points_tensor = torch.cat(reference_points, 1)
        return reference_points_tensor[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        src: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        pos: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for layer in self.layers:
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    """Self-attention, deformable cross-attention, and FFN decoder block."""

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, target: Tensor) -> Tensor:
        target2 = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        return self.norm3(target + self.dropout4(target2))

    def forward(
        self,
        target: Tensor,
        query_pos: Tensor,
        reference_points: Tensor,
        src: Tensor,
        src_spatial_shapes: Tensor,
        level_start_index: Tensor,
        src_padding_mask: Tensor | None = None,
    ) -> Tensor:
        query = key = self.with_pos_embed(target, query_pos)
        target2 = self.self_attn(
            query.transpose(0, 1),
            key.transpose(0, 1),
            target.transpose(0, 1),
        )[0].transpose(0, 1)
        target = self.norm2(target + self.dropout2(target2))
        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        target = self.norm1(target + self.dropout1(target2))
        return self.forward_ffn(target)


class DeformableTransformerDecoder(nn.Module):
    """Stack of decoder blocks with optional iterative box refinement."""

    def __init__(
        self, decoder_layer: nn.Module, num_layers: int, return_intermediate: bool
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self,
        target: Tensor,
        reference_points: Tensor,
        src: Tensor,
        src_spatial_shapes: Tensor,
        src_level_start_index: Tensor,
        src_valid_ratios: Tensor,
        query_pos: Tensor | None = None,
        src_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        output = target
        intermediate = []
        intermediate_reference_points = []
        for layer_index, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                if reference_points.shape[-1] != 2:
                    raise ValueError("Reference points must have width 2 or 4")
                reference_points_input = (
                    reference_points[:, :, None] * src_valid_ratios[:, None]
                )
            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
            )

            if self.bbox_embed is not None:
                delta = self.bbox_embed[layer_index](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = (
                        delta + inverse_sigmoid(reference_points)
                    ).sigmoid()
                else:
                    new_reference_points = delta
                    new_reference_points[..., :2] = delta[..., :2] + inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


__all__ = [
    "DeformableTransformer",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "DeformableTransformerEncoder",
    "DeformableTransformerEncoderLayer",
]
