"""MWAS: Masked Window Attention Sparsification.

Ported from Dome-DETR (https://github.com/RicePasteM/Dome-DETR),
commit 2dde3bc1946a3e9fad9abd0612b59fc39bd6b861, Apache License 2.0.
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

The stride-8 feature map is tiled into ``window_size`` x ``window_size``
windows. Windows the DeFE mask marks empty are skipped; the rest go through an
axis-permuted encoder that alternates attention *within* each window and
*across* the selected windows, and the result is added back in place.

Two forward paths, same numerics:

- ``_forward_gather`` mirrors upstream exactly: ``torch.nonzero`` picks the
  occupied windows and only those are encoded. Data-dependent shapes, so it
  traces to a graph that is only valid for the image it was traced on.
- ``_forward_static`` keeps every window in the tensor and hides the empty ones
  from the cross-window attention with a ``key_padding_mask``, zeroing their
  contribution before the scatter-add. Shapes are then a function of the input
  resolution alone, which is what ONNX needs.

Because the empty windows are masked out of the attention *keys*, the occupied
windows attend over algebraically the same set in both paths: this is a
reformulation, not a reduced-accuracy fallback. It is not bit-identical
though. Softmax over a padded key set and the larger fused matmuls reassociate
the floating-point sums, which measures at ~1e-5 max abs difference on the
encoder output (see ``tests/unit/test_domedetr.py``, which pins the gap
rather than asserting zero). That is the same order as ONNX Runtime's own
divergence from PyTorch, so it does not widen the export error budget.

This mirrors the approach on upstream's own ``onnx-export`` branch.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dfine.ms_deform import get_activation


class AxisPermutedEncoderLayer(nn.Module):
    """Post-norm encoder layer taking q/k/v separately.

    Split from the usual self-attention layer because MWAS reuses one set of
    weights along two different axes (tokens, then windows), so the caller
    supplies the query/key tensors rather than deriving them here.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, q, k, v, src_mask=None, key_padding_mask=None) -> torch.Tensor:
        src = residual = v
        if self.normalize_before:
            src = self.norm1(src)

        src, _ = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=key_padding_mask
        )

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class AxisPermutedEncoder(nn.Module):
    """Alternates within-window and across-window attention, sharing weights."""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None, glob_pos_embeds=None) -> torch.Tensor:
        output = src
        B, C, _, _ = glob_pos_embeds.shape
        glob_pos_embeds = glob_pos_embeds.reshape(B, C, -1).permute(0, 2, 1)
        pos_embed = pos_embed.reshape(C, -1).permute(1, 0).unsqueeze(0)
        for layer in self.layers:
            q = k = self.with_pos_embed(output, glob_pos_embeds + pos_embed)
            output = layer(q, k, output, src_mask=src_mask)

            output = output.permute(1, 0, 2).contiguous()
            q = k = self.with_pos_embed(
                output, (glob_pos_embeds + pos_embed).permute(1, 0, 2).contiguous()
            )
            output = layer(q, k, output, src_mask=src_mask)
            output = output.permute(1, 0, 2).contiguous()

        if self.norm is not None:
            output = self.norm(output)

        return output


class WindowProcessor(nn.Module):
    """Runs encoder attention only on the windows DeFE marks as occupied."""

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        dim_feedforward=1024,
        num_layers=1,
        dropout=0.0,
        activation="relu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.rel_pos_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

        encoder_layer = AxisPermutedEncoderLayer(
            self.embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.window_encoder = AxisPermutedEncoder(encoder_layer, self.num_layers)

        # Flipped by the exporter (and by the unit test that pins the two
        # paths together); ONNX export sets it implicitly.
        self.force_static_path = False

    def _use_static_path(self) -> bool:
        return self.force_static_path or torch.onnx.is_in_onnx_export()

    def forward(self, backbone_memory, defe_feature_filtered, window_size, glob_pos_embed):
        if self._use_static_path():
            return self._forward_static(
                backbone_memory, defe_feature_filtered, window_size, glob_pos_embed
            )
        return self._forward_gather(
            backbone_memory, defe_feature_filtered, window_size, glob_pos_embed
        )

    # -- upstream's gather path ------------------------------------------

    def _forward_gather(self, backbone_memory, defe_feature_filtered, window_size, glob_pos_embed):
        B, _, H, W = backbone_memory.shape
        if H % window_size or W % window_size:
            raise ValueError(
                f"MWAS needs the stride-8 map ({H}x{W}) divisible by "
                f"window_size={window_size}"
            )

        rel_pos_embed = self._get_rel_embedding((window_size, window_size)).to(
            backbone_memory.device
        )
        reconstructed = backbone_memory.clone()
        windows, defe_mask = self._prepare_windows(
            backbone_memory, defe_feature_filtered, window_size
        )

        for b in range(B):
            valid_windows = torch.nonzero(defe_mask[b])
            if valid_windows.numel() == 0:
                # DeFE's density head ends in a sigmoid, so the encoder's
                # threshold search always yields at least one window on real
                # weights. Leaving the map untouched is the honest no-op.
                continue

            window_features, glob_pos_embeds = self._process_windows(
                windows[b], valid_windows, H, W, window_size, glob_pos_embed
            )
            encoded_features = self._encode_features(
                window_features, rel_pos_embed, glob_pos_embeds
            )
            self._reconstruct_features(
                reconstructed, b, encoded_features, valid_windows, window_size, window_size
            )

        return reconstructed, defe_mask

    # -- static-shape path used for export --------------------------------

    def _forward_static(self, backbone_memory, defe_feature_filtered, window_size, glob_pos_embed):
        B, C, H, W = backbone_memory.shape
        if H % window_size or W % window_size:
            raise ValueError(
                f"MWAS needs the stride-8 map ({H}x{W}) divisible by "
                f"window_size={window_size}"
            )

        num_win_h = H // window_size
        num_win_w = W // window_size
        num_windows = num_win_h * num_win_w
        token_len = window_size * window_size

        rel_pos_embed = self._get_rel_embedding((window_size, window_size)).to(
            backbone_memory.device
        )
        windows, defe_mask = self._prepare_windows(
            backbone_memory, defe_feature_filtered, window_size
        )

        glob_pos_windows = (
            glob_pos_embed.view(C, num_win_h, window_size, num_win_w, window_size)
            .permute(1, 3, 0, 2, 4)
            .unsqueeze(0)
            .expand(B, -1, -1, -1, -1, -1)
        )

        features = windows.reshape(B, num_windows, C, token_len).permute(0, 1, 3, 2)
        glob_pos_embeds = glob_pos_windows.reshape(B, num_windows, C, token_len).permute(
            0, 1, 3, 2
        )
        rel_pos = rel_pos_embed.reshape(C, token_len).permute(1, 0).view(1, 1, token_len, C)
        valid_mask = defe_mask.reshape(B, num_windows)

        for layer in self.window_encoder.layers:
            within_input = features.reshape(B * num_windows, token_len, C)
            within_pos = glob_pos_embeds.reshape(B * num_windows, token_len, C) + rel_pos.reshape(
                1, token_len, C
            )
            within_out = layer(
                within_input + within_pos, within_input + within_pos, within_input
            )
            features = within_out.reshape(B, num_windows, token_len, C)

            cross_input = features.permute(0, 2, 1, 3).reshape(B * token_len, num_windows, C)
            cross_pos = (
                (glob_pos_embeds + rel_pos).permute(0, 2, 1, 3).reshape(B * token_len, num_windows, C)
            )
            key_padding_mask = (~valid_mask).unsqueeze(1).expand(B, token_len, num_windows)
            key_padding_mask = key_padding_mask.reshape(B * token_len, num_windows)
            cross_out = layer(
                cross_input + cross_pos,
                cross_input + cross_pos,
                cross_input,
                key_padding_mask=key_padding_mask,
            )
            features = cross_out.reshape(B, token_len, num_windows, C).permute(0, 2, 1, 3)

        if self.window_encoder.norm is not None:
            features = self.window_encoder.norm(features)

        encoded = features.permute(0, 1, 3, 2).reshape(B, num_windows, C, window_size, window_size)
        encoded = encoded * valid_mask.view(B, num_windows, 1, 1, 1).to(encoded.dtype)
        encoded = encoded.reshape(B, num_win_h, num_win_w, C, window_size, window_size)
        updates = encoded.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return backbone_memory + updates, defe_mask

    # -- shared helpers ---------------------------------------------------

    def _prepare_windows(self, features, mask, window_size):
        """Tile the feature map and max-pool the DeFE mask down to one bit per window."""
        B, C, h_feat, w_feat = features.shape
        h_mask, w_mask = mask.shape[-2:]

        num_win_h = h_feat // window_size
        num_win_w = w_feat // window_size

        kernel_h = h_mask // h_feat * window_size
        kernel_w = w_mask // w_feat * window_size

        windows = features.view(B, C, num_win_h, window_size, num_win_w, window_size).permute(
            0, 2, 4, 1, 3, 5
        )

        pooled_mask = F.max_pool2d(
            mask.float(),
            kernel_size=(kernel_h, kernel_w),
            stride=(kernel_h, kernel_w),
        )
        defe_mask = pooled_mask.squeeze(1) > 0  # (B, num_win_h, num_win_w)

        return windows, defe_mask

    def _process_windows(self, windows, valid_indices, H, W, window_size, glob_pos_embed):
        batch_features = []
        batch_glob_pos_embed = []
        for i, j in valid_indices:
            batch_features.append(windows[i, j].unsqueeze(0))
            batch_glob_pos_embed.append(
                self._get_abs_embedding(glob_pos_embed, i, j, window_size, window_size)
            )
        return torch.cat(batch_features, dim=0), torch.stack(batch_glob_pos_embed)

    def _encode_features(self, features, rel_pos_embed, glob_pos_embeds):
        B, C, h, w = features.shape
        features = features.view(B, C, -1).permute(0, 2, 1)
        features = self.window_encoder(
            features, pos_embed=rel_pos_embed, glob_pos_embeds=glob_pos_embeds
        )
        return features.permute(0, 2, 1).view(B, C, h, w)

    @staticmethod
    def _reconstruct_features(reconstructed, batch_idx, feats, indices, win_h, win_w):
        for idx, (i, j) in enumerate(indices):
            h_start = i * win_h
            w_start = j * win_w
            reconstructed[
                batch_idx, :, h_start : h_start + win_h, w_start : w_start + win_w
            ] += feats[idx]

    @staticmethod
    def _get_abs_embedding(global_emb, i, j, win_h, win_w):
        x0 = j * win_w
        y0 = i * win_h
        return global_emb[:, y0 : y0 + win_h, x0 : x0 + win_w]

    def _get_rel_embedding(self, window_size):
        h, w = window_size
        coords = self._get_relative_coords(h, w)
        return self.rel_pos_encoder(coords.to(self.rel_pos_encoder[0].weight.device)).permute(
            2, 0, 1
        )

    @staticmethod
    def _get_relative_coords(h, w):
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        return torch.stack([grid_x / (w - 1), grid_y / (h - 1)], dim=-1)
