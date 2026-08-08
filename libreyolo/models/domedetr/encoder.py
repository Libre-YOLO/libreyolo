"""Dome-DETR hybrid encoder: D-FINE's AIFI + CSP FPN/PAN, plus DeFE and MWAS.

Ported from Dome-DETR (https://github.com/RicePasteM/Dome-DETR).
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
Modified from D-FINE (https://github.com/Peterande/D-FINE).
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

The CSP/RepNCSPELAN4 blocks and the AIFI transformer encoder are byte-identical
to the D-FINE ones already in tree, so they are imported from
``libreyolo/models/dfine/encoder.py`` rather than duplicated. What is new here:

- a fourth (stride-4) feature level, which is where tiny objects live;
- ``self.DeFE``, the density head, run on the stride-4 projection;
- ``self.mwas_processor``, which rewrites the stride-8 projection in place.

The attribute names ``DeFE`` and ``mwas_processor`` match upstream exactly
because they are state-dict key prefixes; renaming them would turn a
metadata-wrap conversion into a key remap for no gain.

Upstream also carries a deformable-encoder branch (``use_deformable=True``)
that depends on a prebuilt MSDeformAttn CUDA extension. Every shipped
Dome-DETR config sets ``use_deformable: False``, so that branch is not ported;
requesting it raises rather than silently running a different model.
"""

from __future__ import annotations

from collections import OrderedDict
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dfine.encoder import (
    ConvNormLayer_fuse,
    RepNCSPELAN4,
    SCDown,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .defe import LiteDeFE
from .mwas import WindowProcessor


# Thresholds the window filter walks down until some cell survives, matching
# upstream's ``init_thresh=0.05, step=0.01`` descent.
_DEFE_THRESHOLDS = (0.05, 0.04, 0.03, 0.02, 0.01, 0.0)


class DomeHybridEncoder(nn.Module):
    """Four-level hybrid encoder with density-guided window attention."""

    def __init__(
        self,
        in_channels=(64, 256, 512, 1024),
        feat_strides=(4, 8, 16, 32),
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=(3,),
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        use_hybrid=True,
        use_deformable=False,
        num_feature_levels=4,
        use_defe=True,
        defe_type="light",
        use_mwas=True,
        mwas_window_size=10,
    ):
        super().__init__()
        if use_deformable:
            raise NotImplementedError(
                "Dome-DETR's deformable encoder branch is not ported: no shipped "
                "config enables it and it requires a prebuilt MSDeformAttn extension."
            )

        self.num_feature_levels = num_feature_levels
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = list(use_encoder_idx)
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.pos_embeds: list[torch.Tensor] = []
        self.in_channels = list(in_channels)
        self.feat_strides = list(feat_strides)
        self.out_channels = [hidden_dim for _ in in_channels]
        self.out_strides = self.feat_strides
        self.use_hybrid = use_hybrid
        self.use_defe = use_defe
        self.defe_type = defe_type
        self.use_mwas = use_mwas
        self.mwas_window_size = mwas_window_size

        self.input_proj = nn.ModuleList()
        for in_channel in self.in_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                            ("norm", nn.BatchNorm2d(hidden_dim)),
                        ]
                    )
                )
            )

        if self.num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=enc_act,
            )
            self.encoder = nn.ModuleList(
                [
                    TransformerEncoder(encoder_layer, num_encoder_layers)
                    for _ in range(len(self.use_encoder_idx))
                ]
            )

        if self.use_hybrid:
            self.lateral_convs = nn.ModuleList()
            self.fpn_blocks = nn.ModuleList()
            for _ in range(len(self.in_channels) - 1, 0, -1):
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
                self.fpn_blocks.append(
                    RepNCSPELAN4(
                        hidden_dim * 2,
                        hidden_dim,
                        hidden_dim * 2,
                        round(expansion * hidden_dim // 2),
                        round(3 * depth_mult),
                    )
                )
            self.downsample_convs = nn.ModuleList()
            self.pan_blocks = nn.ModuleList()
            for _ in range(len(self.in_channels) - 1):
                self.downsample_convs.append(nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2)))
                self.pan_blocks.append(
                    RepNCSPELAN4(
                        hidden_dim * 2,
                        hidden_dim,
                        hidden_dim * 2,
                        round(expansion * hidden_dim // 2),
                        round(3 * depth_mult),
                    )
                )

        if self.use_defe:
            if self.use_mwas:
                # Name matches upstream: it is a state-dict key prefix.
                self.mwas_processor = WindowProcessor(
                    embed_dim=self.hidden_dim, dim_feedforward=dim_feedforward, num_layers=1
                )
            if self.defe_type != "light":
                raise ValueError(f"Invalid defe_type: {self.defe_type!r}")
            self.DeFE = LiteDeFE()  # noqa: N815 - upstream state-dict prefix

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                self.pos_embeds.append(
                    self.build_2d_sincos_position_embedding(
                        ceil(self.eval_spatial_size[1] / stride),
                        ceil(self.eval_spatial_size[0] / stride),
                        self.hidden_dim,
                        self.pe_temperature,
                    )
                )

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos embedding")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    @staticmethod
    def adaptive_defe_filter(defe_feature: torch.Tensor) -> torch.Tensor:
        """Binarise the density map, lowering the threshold until a cell survives.

        Upstream walks a per-sample Python loop from 0.05 down to 0.0 and, if
        even ``> 0.0`` is empty, lights up a random point. DeFE's density head
        ends in a sigmoid, so a strictly-positive map always clears one of the
        thresholds and that random branch is unreachable; this version drops
        the RNG (nondeterministic inference is worse than an empty mask) and
        keeps the descent, which is the part that has an effect.
        """
        mask = torch.zeros_like(defe_feature, dtype=torch.bool)
        for b in range(defe_feature.shape[0]):
            single = defe_feature[b : b + 1]
            for thresh in _DEFE_THRESHOLDS:
                candidate = single > thresh
                if bool(candidate.any()):
                    mask[b : b + 1] = candidate
                    break
        return mask

    def forward(self, feats, img_inputs=None, targets=None):
        """Project, densify, sparsify, then fuse.

        Note on non-square input: upstream unpacks the stride-8 shape as
        ``W, H = proj_feats[1].shape[2:]``, which swaps the two, then builds
        the MWAS position grid and the pooled density map from the swapped
        pair. On the 800x800 eval size every shipped checkpoint uses (and on
        any letterboxed square input) the swap is inert. This port uses the
        unswapped names, which is what the window slicing downstream actually
        needs; it agrees with upstream bit for bit on square input and is
        merely self-consistent rather than transposed on non-square input.
        """
        if len(feats) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} feature levels, got {len(feats)}")

        out: dict = {"img_inputs": img_inputs}
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.use_defe:
            defe_feature, reg_value = self.DeFE(proj_feats[0])
            # int() so the window grid is a graph constant rather than a traced
            # shape read: ONNX rejects adaptive pooling whose output_size is not
            # constant. Safe because this family exports at a fixed square size.
            h, w = int(proj_feats[1].shape[2]), int(proj_feats[1].shape[3])
            defe_feature_pooled = F.adaptive_max_pool2d(
                defe_feature, (h // self.mwas_window_size, w // self.mwas_window_size)
            )
            out["defe"] = {
                "reg_value": reg_value,
                "density_map": defe_feature,
                "defe_feature": defe_feature,
                "density_map_pooled": defe_feature_pooled,
            }
            if self.use_mwas:
                defe_feature_filtered = self.adaptive_defe_filter(
                    F.interpolate(
                        defe_feature_pooled, size=(h, w), mode="bilinear", align_corners=True
                    )
                ).float()
                glob_pos_embed = (
                    self.build_2d_sincos_position_embedding(h, w, embed_dim=self.hidden_dim)
                    .permute(0, 2, 1)
                    .view(-1, h, w)
                    .to(proj_feats[1].device)
                )
                enhanced_memory, defe_window_mask = self.mwas_processor(
                    proj_feats[1], defe_feature_filtered, self.mwas_window_size, glob_pos_embed
                )
                proj_feats[1] = enhanced_memory
                out["defe"]["defe_window_mask"] = defe_window_mask

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = self.pos_embeds[i].to(src_flatten.device)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = (
                    memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                )

        if self.use_hybrid:
            inner_outs = [proj_feats[-1]]
            for idx in range(len(self.in_channels) - 1, 0, -1):
                feat_high = inner_outs[0]
                feat_low = proj_feats[idx - 1]
                feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
                inner_outs[0] = feat_high
                upsample_feat = F.interpolate(
                    feat_high,
                    size=(feat_low.shape[2], feat_low.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                    torch.concat([upsample_feat, feat_low], dim=1)
                )
                inner_outs.insert(0, inner_out)

            outs = [inner_outs[0]]
            for idx in range(len(self.in_channels) - 1):
                feat_low = outs[-1]
                feat_high = inner_outs[idx + 1]
                downsample_feat = self.downsample_convs[idx](feat_low)
                outs.append(self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1)))
        else:
            outs = proj_feats

        out["feats"] = outs
        return out
