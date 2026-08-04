# SPDX-License-Identifier: Apache-2.0
# Ported from https://github.com/IDEA-Research/DINO at
# d84a491d41898b3befd8294d1cf2614661fc0953.
# Copyright 2022 IDEA.
# Includes work derived from Conditional DETR, DETR, and Deformable DETR.
"""Inference-only deformable transformer used by released DINO checkpoints."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from ..deformable_detr.ms_deform_attn import MSDeformAttn
from .common import inverse_sigmoid


def _get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def _gen_sineembed_for_position(position: Tensor) -> Tensor:
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=position.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)

    embeddings = []
    # Upstream concatenates y, x, width, height in this order.
    for coordinate in (1, 0, 2, 3):
        values = position[:, :, coordinate] * scale
        values = values[:, :, None] / dim_t
        embeddings.append(
            torch.stack(
                (values[:, :, 0::2].sin(), values[:, :, 1::2].cos()), dim=3
            ).flatten(2)
        )
    return torch.cat(embeddings, dim=2)


def _gen_encoder_output_proposals(
    memory: Tensor,
    padding_mask: Tensor,
    spatial_shapes: Tensor,
) -> tuple[Tensor, Tensor]:
    batch = memory.shape[0]
    proposals = []
    current = 0
    for level, (height, width) in enumerate(spatial_shapes):
        mask = padding_mask[:, current : current + height * width].view(
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
        grid = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), -1)
        scale = torch.cat(
            (valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)), 1
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
        padding_mask.unsqueeze(-1), float("inf")
    )
    output_proposals = output_proposals.masked_fill(~valid, float("inf"))
    output_memory = memory.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    output_memory = output_memory.masked_fill(~valid, 0.0)
    return output_memory, output_proposals


class MLP(nn.Module):
    """Small feed-forward network with checkpoint-compatible layer names."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip([input_dim, *hidden], [*hidden, output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class DeformableTransformerEncoderLayer(nn.Module):
    """One multi-scale deformable encoder layer."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float,
        n_levels: int,
        n_heads: int,
        n_points: int,
    ):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, position: Tensor | None) -> Tensor:
        return tensor if position is None else tensor + position

    def forward(
        self,
        src: Tensor,
        position: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        attended = self.self_attn(
            self.with_pos_embed(src, position),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = self.norm1(src + self.dropout1(attended))
        feed_forward = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout3(feed_forward))


class TransformerEncoder(nn.Module):
    """Released DINO six-layer deformable encoder."""

    def __init__(self, layer: nn.Module, num_layers: int, d_model: int):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.query_scale = None
        self.num_queries = 900
        self.deformable_encoder = True
        self.num_layers = num_layers
        self.norm = None
        self.d_model = d_model
        self.enc_layer_dropout_prob = None
        self.two_stage_type = "standard"

    @staticmethod
    def get_reference_points(
        spatial_shapes: Tensor, valid_ratios: Tensor, device: torch.device
    ) -> Tensor:
        points = []
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
            points.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(points, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        src: Tensor,
        position: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, src.device
        )
        for layer in self.layers:
            output = layer(
                output,
                position,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    """Self-attention, deformable cross-attention, and FFN decoder layer."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float,
        n_levels: int,
        n_heads: int,
        n_points: int,
    ):
        super().__init__()
        self.module_seq = ["sa", "ca", "ffn"]
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.key_aware_type = None
        self.key_aware_proj = None
        self.decoder_sa_type = "sa"
        self.label_embedding = None

    @staticmethod
    def with_pos_embed(tensor: Tensor, position: Tensor | None) -> Tensor:
        return tensor if position is None else tensor + position

    def forward(
        self,
        target: Tensor,
        query_position: Tensor,
        reference_points: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor,
        level_start_index: Tensor,
        spatial_shapes: Tensor,
        self_attention_mask: Tensor | None,
    ) -> Tensor:
        query = key = self.with_pos_embed(target, query_position)
        attended = self.self_attn(query, key, target, attn_mask=self_attention_mask)[0]
        target = self.norm2(target + self.dropout2(attended))
        attended = self.cross_attn(
            self.with_pos_embed(target, query_position).transpose(0, 1),
            reference_points.transpose(0, 1).contiguous(),
            memory.transpose(0, 1),
            spatial_shapes,
            level_start_index,
            memory_padding_mask,
        ).transpose(0, 1)
        target = self.norm1(target + self.dropout1(attended))
        feed_forward = self.linear2(
            self.dropout3(self.activation(self.linear1(target)))
        )
        return self.norm3(target + self.dropout4(feed_forward))


class TransformerDecoder(nn.Module):
    """Released DINO decoder with iterative reference-box refinement."""

    def __init__(
        self, layer: nn.Module, num_layers: int, d_model: int, num_levels: int
    ):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = True
        self.query_dim = 4
        self.num_feature_levels = num_levels
        self.use_detached_boxes_dec_out = False
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None
        self.query_scale = None
        self.bbox_embed: nn.ModuleList | None = None
        self.class_embed: nn.ModuleList | None = None
        self.d_model = d_model
        self.modulate_hw_attn = True
        self.deformable_decoder = True
        self.ref_anchor_head = None
        self.decoder_query_perturber = None
        self.box_pred_damping = None
        self.dec_layer_number = None
        self.dec_layer_dropout_prob = None
        self.rm_detach = None

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor,
        position: Tensor,
        reference_points_unsigmoid: Tensor,
        level_start_index: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        target_mask: Tensor | None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        output = target
        intermediate = []
        reference_points = reference_points_unsigmoid.sigmoid()
        all_reference_points = [reference_points]
        if self.bbox_embed is None:
            raise RuntimeError("DINO decoder box heads are not attached")

        for layer_index, layer in enumerate(self.layers):
            reference_points_input = (
                reference_points[:, :, None]
                * torch.cat((valid_ratios, valid_ratios), -1)[None, :]
            )
            sine_embedding = _gen_sineembed_for_position(
                reference_points_input[:, :, 0, :]
            )
            query_position = self.ref_point_head(sine_embedding)
            output = layer(
                output,
                query_position,
                reference_points_input,
                memory,
                memory_padding_mask,
                level_start_index,
                spatial_shapes,
                target_mask,
            )
            new_reference_points = (
                self.bbox_embed[layer_index](output) + inverse_sigmoid(reference_points)
            ).sigmoid()
            reference_points = new_reference_points.detach()
            all_reference_points.append(new_reference_points)
            intermediate.append(self.norm(output))

        return (
            [value.transpose(0, 1) for value in intermediate],
            [value.transpose(0, 1) for value in all_reference_points],
        )


class DeformableTransformer(nn.Module):
    """Fixed transformer configuration shared by all released DINO variants."""

    def __init__(self, num_feature_levels: int):
        super().__init__()
        d_model = 256
        num_layers = 6
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, 2048, 0.0, num_feature_levels, 8, 4
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers, d_model)
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model, 2048, 0.0, num_feature_levels, 8, 4
        )
        self.decoder = TransformerDecoder(
            decoder_layer, num_layers, d_model, num_feature_levels
        )
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_layers
        self.num_unicoder_layers = 0
        self.num_decoder_layers = num_layers
        self.deformable_encoder = True
        self.deformable_decoder = True
        self.two_stage_keep_all_tokens = False
        self.num_queries = 900
        self.random_refpoints_xy = False
        self.use_detached_boxes_dec_out = False
        self.d_model = d_model
        self.nhead = 8
        self.dec_layers = num_layers
        self.num_patterns = 0
        self.level_embed = nn.Parameter(torch.empty(num_feature_levels, d_model))
        self.learnable_tgt_init = True
        self.embed_init_tgt = True
        self.tgt_embed = nn.Embedding(self.num_queries, d_model)
        self.two_stage_type = "standard"
        self.two_stage_pat_embed = 0
        self.two_stage_add_query_num = 0
        self.two_stage_learn_wh = False
        self.two_stage_wh_embedding = None
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.enc_out_class_embed: nn.Module | None = None
        self.enc_out_bbox_embed: nn.Module | None = None
        self.dec_layer_number = None
        self.rm_self_attn_layers = None
        self.rm_detach = None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        for module in self.modules():
            if isinstance(module, MSDeformAttn):
                module._reset_parameters()
        nn.init.normal_(self.level_embed)

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        return torch.stack(
            (valid_width.float() / width, valid_height.float() / height), -1
        )

    def forward(
        self,
        srcs: list[Tensor],
        masks: list[Tensor],
        reference_query: Tensor | None,
        position_embeddings: list[Tensor],
        target_query: Tensor | None,
        attention_mask: Tensor | None = None,
    ) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor, Tensor]:
        src_flatten = []
        mask_flatten = []
        level_position_flatten = []
        spatial_shapes = []
        for level, (src, mask, position) in enumerate(
            zip(srcs, masks, position_embeddings)
        ):
            _, _, height, width = src.shape
            spatial_shapes.append((height, width))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            position = position.flatten(2).transpose(1, 2)
            level_position_flatten.append(
                position + self.level_embed[level].view(1, 1, -1)
            )
        src_flatten_tensor = torch.cat(src_flatten, 1)
        mask_flatten_tensor = torch.cat(mask_flatten, 1)
        position_flatten_tensor = torch.cat(level_position_flatten, 1)
        shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten_tensor.device
        )
        level_start_index = torch.cat(
            (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(mask) for mask in masks], 1)
        memory = self.encoder(
            src_flatten_tensor,
            position_flatten_tensor,
            shapes,
            level_start_index,
            valid_ratios,
            mask_flatten_tensor,
        )
        output_memory, output_proposals = _gen_encoder_output_proposals(
            memory, mask_flatten_tensor, shapes
        )
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        if self.enc_out_class_embed is None or self.enc_out_bbox_embed is None:
            raise RuntimeError("DINO encoder output heads are not attached")
        encoder_classes = self.enc_out_class_embed(output_memory)
        encoder_boxes = self.enc_out_bbox_embed(output_memory) + output_proposals
        topk_indices = torch.topk(encoder_classes.max(-1)[0], self.num_queries, dim=1)[
            1
        ]
        gathered_boxes = torch.gather(
            encoder_boxes, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4)
        )
        selected_reference_query = gathered_boxes.detach()
        initial_box_proposal = torch.gather(
            output_proposals, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4)
        ).sigmoid()
        gathered_memory = torch.gather(
            output_memory,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, self.d_model),
        )
        batch = memory.shape[0]
        selected_target_query = (
            self.tgt_embed.weight[:, None, :].repeat(1, batch, 1).transpose(0, 1)
        )
        if reference_query is not None:
            reference_query = torch.cat(
                (reference_query, selected_reference_query), dim=1
            )
            if target_query is None:
                raise ValueError("DINO reference and target queries must be paired")
            target_query = torch.cat((target_query, selected_target_query), dim=1)
        else:
            reference_query = selected_reference_query
            target_query = selected_target_query

        hidden_states, references = self.decoder(
            target_query.transpose(0, 1),
            memory.transpose(0, 1),
            mask_flatten_tensor,
            position_flatten_tensor.transpose(0, 1),
            reference_query.transpose(0, 1),
            level_start_index,
            shapes,
            valid_ratios,
            attention_mask,
        )
        return (
            hidden_states,
            references,
            gathered_memory.unsqueeze(0),
            gathered_boxes.sigmoid().unsqueeze(0),
            initial_box_proposal,
        )


__all__ = ["DeformableTransformer", "MLP"]
