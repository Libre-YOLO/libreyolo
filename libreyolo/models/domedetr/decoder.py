"""Dome-DETR decoder: D-FINE's FDR decoder plus PAQI query initialisation.

Ported from Dome-DETR (https://github.com/RicePasteM/Dome-DETR),
commit 2dde3bc1946a3e9fad9abd0612b59fc39bd6b861, Apache License 2.0.
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
Modified from D-FINE (https://github.com/Peterande/D-FINE).
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

The decoder stack itself (``TransformerDecoder``, its layer, ``Integral``,
``LQE``, ``MLP``) is unchanged from D-FINE and is imported from
``libreyolo/models/dfine/decoder.py``. What Dome-DETR replaces is the query
initialiser:

**PAQI (Progressive Adaptive Query Initialization).** D-FINE takes a fixed
top-300. PAQI takes the top ``max_num_select`` (1500 on AI-TOD-V2), keeps the
strongest ``min_num_select`` (300) unconditionally, then keeps only those of
the remainder whose centre falls in a window MWAS marked occupied, and finally
runs density-adaptive NMS over the union. The surviving query count therefore
varies per image, which is the whole point on drone imagery: a sparse frame
spends few queries, a crowded one spends many.

The variable count is also why this family cannot reuse D-FINE's
``_select_topk`` verbatim: that one drops the encoder logits in eval mode, and
PAQI needs them as the NMS scores.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..dfine.decoder import MLP, Integral, TransformerDecoder, TransformerDecoderLayer
from ..dfine.ms_deform import bias_init_with_prob
from .dynamic_nms import dynamic_nms


class DomeTransformer(nn.Module):
    """D-FINE transformer with density-adaptive query initialisation."""

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        feat_channels=(256, 256, 256, 256),
        feat_strides=(4, 8, 16, 32),
        num_levels=4,
        num_points=(4, 4, 4, 4),
        nhead=8,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        min_num_select=300,
        max_num_select=1500,
    ):
        super().__init__()
        feat_channels = list(feat_channels)
        feat_strides = list(feat_strides)
        if len(feat_channels) > num_levels:
            raise ValueError("feat_channels cannot exceed num_levels")
        if len(feat_strides) != len(feat_channels):
            raise ValueError("feat_strides and feat_channels must be the same length")
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        self.min_num_select = min_num_select
        self.max_num_select = max_num_select

        if query_select_method not in ("default", "one2many", "agnostic"):
            raise ValueError(f"bad query_select_method: {query_select_method!r}")
        if cross_attn_method not in ("default", "discrete"):
            raise ValueError(f"bad cross_attn_method: {cross_attn_method!r}")
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        self._build_input_proj_layer(feat_channels)

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
        )
        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
        )
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            activation=activation,
        )

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            # Unused at inference but present in every shipped checkpoint.
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            init.normal_(self.denoising_class_embed.weight[:-1])

        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=activation)

        self.enc_output = nn.Sequential(
            OrderedDict(
                [
                    ("proj", nn.Linear(hidden_dim, hidden_dim)),
                    ("norm", nn.LayerNorm(hidden_dim)),
                ]
            )
        )

        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=activation)

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=activation)
        self.dec_bbox_head = nn.ModuleList(
            [
                MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, act=activation)
                for _ in range(self.eval_idx + 1)
            ]
            + [
                MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, act=activation)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        self.integral = Integral(self.reg_max)

        self._reset_parameters(feat_channels)

    # -- construction helpers (shared shape with D-FINE) -------------------

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )

        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        in_channels, self.hidden_dim, 3, 2, padding=1, bias=False
                                    ),
                                ),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
                in_channels = self.hidden_dim

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.dec_score_head[self.eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity()
                for i in range(len(self.dec_bbox_head))
            ]
        )

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        return torch.concat(feat_flatten, 1), spatial_shapes

    def _generate_anchors(
        self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"
    ):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        # Upstream stores +inf for out-of-range anchors. LibreYOLO's DETR
        # families use a large finite sentinel instead: inf overflows in fp16
        # and TensorRT mishandles it. Both saturate the downstream sigmoid to
        # the same degenerate box, and the parity test confirms the swap does
        # not move any output (these anchors sit on zeroed memory tokens and
        # never win top-k).
        anchors = torch.where(valid_mask, anchors, torch.full_like(anchors, 1e4))
        return anchors, valid_mask

    def _select_topk(self, memory, outputs_logits, outputs_anchors_unact, topk: int):
        """Top-k selection that always returns logits (PAQI needs them as scores)."""
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        else:
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_anchors = outputs_anchors_unact.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1])
        )
        topk_logits = outputs_logits.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])
        )
        topk_memory = memory.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        )
        return topk_memory, topk_logits, topk_anchors

    # -- PAQI --------------------------------------------------------------

    def _get_decoder_input(self, memory, spatial_shapes, defe_window_mask, defe_feature):
        anchors, valid_mask = self._generate_anchors(
            spatial_shapes, dtype=memory.dtype, device=memory.device
        )
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        memory = valid_mask.to(memory.dtype) * memory

        output_memory = self.enc_output(memory)
        enc_outputs_logits = self.enc_score_head(output_memory)

        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            output_memory, enc_outputs_logits, anchors, self.max_num_select
        )

        B = enc_topk_anchors.size(0)
        min_num, max_num = self.min_num_select, self.max_num_select

        memory_first = enc_topk_memory[:, :min_num, :]
        logits_first = enc_topk_logits[:, :min_num]
        anchors_first = enc_topk_anchors[:, :min_num, :]

        memory_second = enc_topk_memory[:, min_num:max_num, :]
        logits_second = enc_topk_logits[:, min_num:max_num]
        anchors_second = enc_topk_anchors[:, min_num:max_num, :]

        if defe_window_mask is not None:
            # Upstream derives the row index from cx and the column from cy,
            # i.e. the two are transposed relative to their names. On the
            # square window grids every shipped config produces this has no
            # effect; it is kept as-is so the port stays bit-identical.
            n_x, n_y = defe_window_mask.shape[1], defe_window_mask.shape[2]
            cx, cy = F.sigmoid(anchors_second[..., 0]), F.sigmoid(anchors_second[..., 1])
            window_col = (cx * n_x).long().clamp(0, n_x - 1)
            window_row = (cy * n_y).long().clamp(0, n_y - 1)
            selected_mask = defe_window_mask[
                torch.arange(B, device=enc_topk_anchors.device).view(-1, 1),
                window_row,
                window_col,
            ]
        else:
            selected_mask = torch.ones_like(anchors_second[..., 0], dtype=torch.bool)

        combined_memory, combined_logits, combined_bbox_unact = [], [], []
        total_per_batch = []

        for b in range(B):
            mask_b = selected_mask[b]
            mem_combined = torch.cat([memory_first[b], memory_second[b][mask_b]], dim=0)
            log_combined = torch.cat([logits_first[b], logits_second[b][mask_b]], dim=0)
            anc_combined = torch.cat([anchors_first[b], anchors_second[b][mask_b]], dim=0)

            bbox_combined_unact = self.enc_bbox_head(mem_combined) + anc_combined
            bbox_combined = F.sigmoid(bbox_combined_unact)

            if log_combined.size(0) > 0:
                cx, cy = bbox_combined[:, 0], bbox_combined[:, 1]
                w, h = bbox_combined[:, 2], bbox_combined[:, 3]
                boxes = torch.stack(
                    [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1
                )

                # Same transposed row/col convention as above, kept for parity.
                cf_h, cf_w = defe_feature.shape[2:]
                density_row = (cx * (cf_w - 1)).long().clamp(0, cf_w - 1)
                density_col = (cy * (cf_h - 1)).long().clamp(0, cf_h - 1)
                density_values = defe_feature[b, :, density_row, density_col].squeeze(0).detach()

                # Dense regions keep more boxes: threshold rises with density.
                iou_thresholds = 0.4 + 0.5 * density_values
                scores, class_ids = log_combined.max(dim=1)
                keep_idx = dynamic_nms(boxes, scores, class_ids, iou_thresholds)

                # The unconditional top-`min_num` bypass NMS entirely.
                final_keep_idx = torch.arange(min_num, device=keep_idx.device)
                final_keep_idx = torch.cat([final_keep_idx, keep_idx[keep_idx >= min_num]])

                mem_combined = mem_combined[final_keep_idx]
                log_combined = log_combined[final_keep_idx]
                bbox_combined_unact = bbox_combined_unact[final_keep_idx]

            combined_memory.append(mem_combined)
            combined_logits.append(log_combined)
            combined_bbox_unact.append(bbox_combined_unact)
            total_per_batch.append(mem_combined.size(0))

        max_total = max(total_per_batch)
        padded_memory = torch.zeros(
            (B, max_total, memory_first.size(-1)), device=memory.device, dtype=memory.dtype
        )
        padded_bbox_unact = torch.zeros(
            (B, max_total, 4), device=memory.device, dtype=memory.dtype
        )
        # Pad the class logits with a large negative rather than zero: zeros
        # would sigmoid to 0.5 and surface as detections on every short image
        # in a mixed batch. Single-image inference never pads, so this does not
        # move the parity numbers.
        padded_logits = torch.full(
            (B, max_total, self.num_classes),
            torch.finfo(memory.dtype).min,
            device=memory.device,
            dtype=memory.dtype,
        )

        for b in range(B):
            n = total_per_batch[b]
            padded_memory[b, :n] = combined_memory[b]
            padded_logits[b, :n] = combined_logits[b]
            padded_bbox_unact[b, :n] = combined_bbox_unact[b]

        content = padded_memory.detach()
        return content, padded_bbox_unact.detach(), padded_logits, total_per_batch

    def forward(self, encoder_out, targets=None):
        if self.training:
            raise NotImplementedError(
                "LibreDOMEDETR ships inference-only; Dome-DETR training is not wired."
            )
        del targets

        feats = encoder_out["feats"]
        memory, spatial_shapes = self._get_encoder_input(feats)

        defe = encoder_out.get("defe")
        defe_window_mask = defe.get("defe_window_mask") if defe else None
        defe_feature = defe.get("density_map_pooled") if defe else None

        init_ref_contents, init_ref_points_unact, _, batch_queries_num = self._get_decoder_input(
            memory, spatial_shapes, defe_window_mask, defe_feature
        )

        out_bboxes, out_logits, *_ = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
        )

        return {
            "pred_logits": out_logits[-1],
            "pred_boxes": out_bboxes[-1],
            "batch_queries_num": batch_queries_num,
        }
