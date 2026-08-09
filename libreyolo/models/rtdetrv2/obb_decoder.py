"""Five-coordinate RT-DETRv2 decoder for oriented boxes.

Adapted from ``RicePasteM/RiO-DETR`` at commit
``22d5232a4e0df6ac4bc26ed1c8aac8b4060449c7`` (Apache-2.0), specifically
``engine/rtv4/rtdetrv2_obb_decoder.py``.  Shared horizontal RT-DETRv2
building blocks remain in :mod:`libreyolo.models.rtdetrv2.decoder`.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .decoder import (
    MLP,
    MSDeformableAttention,
    RTDETRTransformerv2,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from .utils import deformable_attention_core_func_v2

__all__ = ["RTDETRTransformerv2OBB"]


class OBBMSDeformableAttention(MSDeformableAttention):
    """RT-DETRv2 deformable attention with angle-aware sampling."""

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reference_points.shape[-1] != 5:
            raise ValueError(
                "OBB reference_points must contain (cx, cy, w, h, angle), "
                f"got last dimension {reference_points.shape[-1]}"
            )

        bs, len_q = query.shape[:2]
        len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(bs, len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, len_q, self.num_heads, sum(self.num_points_list), 2
        )
        attention_weights = self.attention_weights(query).reshape(
            bs, len_q, self.num_heads, sum(self.num_points_list)
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        references = reference_points.expand(-1, -1, self.num_levels, -1)
        expanded = []
        for level, num_points in enumerate(self.num_points_list):
            ref = references[:, :, level : level + 1, :]
            expanded.append(ref.unsqueeze(2).repeat(1, 1, 1, num_points, 1))
        expanded_references = torch.cat(expanded, dim=3)

        point_scale = (
            self.num_points_scale
            if torch.jit.is_tracing()
            else self.num_points_scale.to(device=query.device, dtype=query.dtype)
        ).view(1, 1, 1, -1, 1)
        angle = expanded_references[..., 4:] * math.pi
        angle = angle.expand(-1, -1, self.num_heads, -1, -1)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rotation = torch.cat([cos_a, -sin_a, sin_a, cos_a], dim=-1)
        rotation = rotation.reshape(*rotation.shape[:-1], 2, 2)
        wh = (expanded_references[..., 2:4] * self.offset_scale).expand(
            -1, -1, self.num_heads, -1, -1
        )
        rotated_extent = torch.einsum("...ij,...j->...i", rotation, wh)
        sampling_locations = (
            expanded_references[..., :2]
            + sampling_offsets * point_scale * rotated_extent
        )

        output = self.ms_deformable_attn_core(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list,
        )
        return self.output_proj(output)


class OBBTransformerDecoderLayer(TransformerDecoderLayer):
    """Horizontal decoder layer with the OBB cross-attention equation."""

    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        cross_attn_method="default",
    ):
        super().__init__(
            d_model=d_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=n_levels,
            n_points=n_points,
            cross_attn_method=cross_attn_method,
        )
        self.cross_attn = OBBMSDeformableAttention(
            d_model,
            n_head,
            n_levels,
            n_points,
            method=cross_attn_method,
        )
        # The released checkpoints use the upstream pure-PyTorch sampling
        # equation.  Its reduction differs by a few ULPs from LibreYOLO's
        # optional fused kernel, so retain it for exact raw-output parity.
        self.cross_attn.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2,
            method=cross_attn_method,
            allow_acceleration=False,
        )


class RTDETRTransformerv2OBB(RTDETRTransformerv2):
    """RT-DETRv2 decoder emitting normalized ``(cx, cy, w, h, angle/pi)``."""

    def __init__(
        self,
        *args,
        anchor_aspect_ratio: float = 1.0,
        **kwargs,
    ):
        if anchor_aspect_ratio != 1.0:
            raise ValueError(
                "The released RT-DETRv2 OBB baselines require anchor_aspect_ratio=1.0"
            )
        super().__init__(*args, **kwargs)
        self.anchor_aspect_ratio = anchor_aspect_ratio

        decoder_layer = OBBTransformerDecoderLayer(
            self.hidden_dim,
            self.nhead,
            kwargs.get("dim_feedforward", 1024),
            kwargs.get("dropout", 0.0),
            kwargs.get("activation", "relu"),
            self.num_levels,
            kwargs.get("num_points", 4),
            cross_attn_method=self.cross_attn_method,
        )
        eval_idx = kwargs.get("eval_idx", -1)
        self.decoder = TransformerDecoder(
            self.hidden_dim, decoder_layer, self.num_layers, eval_idx
        )

        self.query_pos_head = MLP(5, 2 * self.hidden_dim, self.hidden_dim, 2)
        self.enc_bbox_head = MLP(self.hidden_dim, self.hidden_dim, 5, 3)
        self.dec_bbox_head = nn.ModuleList(
            [
                MLP(self.hidden_dim, self.hidden_dim, 5, 3)
                for _ in range(self.num_layers)
            ]
        )
        self._reset_obb_parameters()

    def _reset_obb_parameters(self) -> None:
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for reg in self.dec_bbox_head:
            init.constant_(reg.layers[-1].weight, 0)
            init.constant_(reg.layers[-1].bias, 0)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _get_decoder_input(
        self,
        memory,
        spatial_shapes,
        denoising_logits=None,
        denoising_bbox_unact=None,
    ):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(
                spatial_shapes, device=memory.device
            )
        else:
            anchors, valid_mask = self._get_anchors_for_spatial_shapes(
                spatial_shapes, memory
            )

        valid_values = (
            valid_mask if torch.jit.is_tracing() else valid_mask.to(memory.dtype)
        )
        memory = valid_values * memory
        output_memory = self.enc_output(memory)
        enc_outputs_logits = self.enc_score_head(output_memory)

        anchors_5d = torch.cat([anchors, torch.zeros_like(anchors[..., :1])], dim=-1)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors_5d
        enc_outputs_coord_unact = enc_outputs_coord_unact.clamp(-20.0, 20.0)

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = self._select_topk(
            output_memory,
            enc_outputs_logits,
            enc_outputs_coord_unact,
            self.num_queries,
        )

        if self.training or self.emit_loss_outputs:
            enc_topk_bboxes_list.append(F.sigmoid(enc_topk_bbox_unact))
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.cat(
                [denoising_bbox_unact, enc_topk_bbox_unact], dim=1
            )
            content = torch.cat([denoising_logits, content], dim=1)

        return (
            content,
            enc_topk_bbox_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
        )

    def forward(self, feats, targets=None):
        if self.training and targets is not None:
            raise NotImplementedError(
                "RT-DETRv2 OBB training is not implemented in LibreYOLO"
            )
        return super().forward(feats, targets=None)
