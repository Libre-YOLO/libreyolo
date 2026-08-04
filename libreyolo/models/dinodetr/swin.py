# SPDX-License-Identifier: Apache-2.0
# Ported from https://github.com/IDEA-Research/DINO at
# d84a491d41898b3befd8294d1cf2614661fc0953.
# Copyright 2022 IDEA.
# The upstream module derives from microsoft/Swin-Transformer (MIT).
"""Swin-L/384 backbone used by the released five-scale DINO checkpoint."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from ..deformable_detr.common import NestedTensor
from ...kernels.attention.sdpa import manual_attention_required


def _to_2tuple(value: int) -> tuple[int, int]:
    return value, value


class DropPath(nn.Module):
    """Per-sample stochastic depth; inactive in this inference-only port."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep)
        return x * random_tensor.div_(keep)


class Mlp(nn.Module):
    """Swin feed-forward block."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def window_partition(x: Tensor, window_size: int) -> Tensor:
    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, channels)
    )


def window_reverse(
    windows: Tensor, window_size: int, height: int, width: int
) -> Tensor:
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class WindowAttention(nn.Module):
    """Window self-attention with learned relative position bias."""

    def __init__(self, dim: int, window_size: tuple[int, int], num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        table_size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads)
        )

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)
        # Opt-in fused SDPA. Default off: tests/unit/test_dinodetr_parity.py
        # pins max_abs_diff == 0.0 against the upstream checkpoint and fused
        # kernels accumulate in a different order. Flip with
        # libreyolo.kernels.attention.set_fused_attention(model).
        self.fused_attn = False

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                batch_windows,
                tokens,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        if self.fused_attn and not manual_attention_required():
            # SDPA takes one additive float mask, so the relative position bias
            # and the shifted-window mask are summed into it. The window mask is
            # (num_windows, tokens, tokens) and the batch is laid out as
            # (batch, num_windows) flattened, so repeat tiles it per window.
            attn_mask = relative_bias.unsqueeze(0)
            if mask is not None:
                attn_mask = attn_mask + mask.unsqueeze(1).repeat(
                    batch_windows // mask.shape[0], 1, 1, 1
                )
            fused = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
                scale=self.scale,
            )
            fused = fused.transpose(1, 2).reshape(batch_windows, tokens, channels)
            return self.proj_drop(self.proj(fused))
        attention = (query * self.scale) @ key.transpose(-2, -1)
        attention = attention + relative_bias.unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                tokens,
                tokens,
            )
            attention = attention + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, tokens, tokens)
        attention = self.attn_drop(self.softmax(attention))
        x = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    """One shifted- or unshifted-window Swin block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        drop_path: float,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = 4.0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, _to_2tuple(window_size), num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)
        self.H: int | None = None
        self.W: int | None = None

    def forward(self, x: Tensor, mask_matrix: Tensor) -> Tensor:
        batch, length, channels = x.shape
        if self.H is None or self.W is None or length != self.H * self.W:
            raise ValueError("Swin block received an invalid spatial shape")
        height, width = self.H, self.W
        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))
        padded_h, padded_w = x.shape[1:3]
        if self.shift_size > 0:
            shifted = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            attention_mask = mask_matrix
        else:
            shifted = x
            attention_mask = None
        windows = window_partition(shifted, self.window_size).view(
            -1, self.window_size * self.window_size, channels
        )
        windows = self.attn(windows, mask=attention_mask).view(
            -1, self.window_size, self.window_size, channels
        )
        shifted = window_reverse(windows, self.window_size, padded_h, padded_w)
        if self.shift_size > 0:
            x = torch.roll(
                shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted
        if pad_right > 0 or pad_bottom > 0:
            x = x[:, :height, :width, :].contiguous()
        x = x.view(batch, height * width, channels)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    """Merge each 2x2 patch group and double the channel width."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError("PatchMerging received an invalid spatial shape")
        x = x.view(batch, height, width, channels)
        if height % 2 or width % 2:
            x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        x = torch.cat(
            (x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]),
            dim=-1,
        )
        x = x.view(batch, -1, 4 * channels)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    """One hierarchical Swin stage."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        drop_path: list[float],
        downsample: bool,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = False
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim,
                    num_heads,
                    window_size,
                    0 if index % 2 == 0 else window_size // 2,
                    drop_path[index],
                )
                for index in range(depth)
            ]
        )
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(
        self, x: Tensor, height: int, width: int
    ) -> tuple[Tensor, int, int, Tensor, int, int]:
        padded_h = math.ceil(height / self.window_size) * self.window_size
        padded_w = math.ceil(width / self.window_size) * self.window_size
        # This is algebraically identical to upstream's nine in-place slice
        # assignments, but functional construction also lowers cleanly to ONNX.
        height_coords = torch.arange(padded_h, device=x.device)
        width_coords = torch.arange(padded_w, device=x.device)
        height_regions = (height_coords >= padded_h - self.window_size).to(
            torch.float32
        ) + (height_coords >= padded_h - self.shift_size).to(torch.float32)
        width_regions = (width_coords >= padded_w - self.window_size).to(
            torch.float32
        ) + (width_coords >= padded_w - self.shift_size).to(torch.float32)
        image_mask = (
            (height_regions[:, None] * 3 + width_regions[None, :])
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        mask_windows = window_partition(image_mask, self.window_size).view(
            -1, self.window_size * self.window_size
        )
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(
            attention_mask != 0, -100.0
        ).masked_fill(attention_mask == 0, 0.0)
        for block in self.blocks:
            block.H, block.W = height, width
            x = block(x, attention_mask)
        if self.downsample is None:
            return x, height, width, x, height, width
        downsampled = self.downsample(x, height, width)
        return (
            x,
            height,
            width,
            downsampled,
            (height + 1) // 2,
            (width + 1) // 2,
        )


class PatchEmbed(nn.Module):
    """Turn an RGB image into non-overlapping patch tokens."""

    def __init__(self, embed_dim: int = 192, patch_size: int = 4):
        super().__init__()
        self.patch_size = _to_2tuple(patch_size)
        self.in_chans = 3
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        if width % self.patch_size[1]:
            x = F.pad(x, (0, self.patch_size[1] - width % self.patch_size[1]))
        if height % self.patch_size[0]:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - height % self.patch_size[0]))
        x = self.proj(x)
        out_h, out_w = x.shape[2:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).view(-1, self.embed_dim, out_h, out_w)


class SwinTransformer(nn.Module):
    """Exact released Swin-L/384 hierarchy, returning all four stages."""

    def __init__(self):
        super().__init__()
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
        window_size = 12
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = False
        self.patch_norm = True
        self.out_indices = (0, 1, 2, 3)
        self.frozen_stages = -1
        self.dilation = False
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(0.0)
        path_rates = [value.item() for value in torch.linspace(0, 0.2, sum(depths))]
        self.num_features = [embed_dim * 2**index for index in range(4)]
        self.num_channels = list(self.num_features)
        self.layers = nn.ModuleList()
        offset = 0
        for index, depth in enumerate(depths):
            self.layers.append(
                BasicLayer(
                    self.num_features[index],
                    depth,
                    num_heads[index],
                    window_size,
                    path_rates[offset : offset + depth],
                    downsample=index < self.num_layers - 1,
                )
            )
            offset += depth
        for index in self.out_indices:
            self.add_module(f"norm{index}", nn.LayerNorm(self.num_features[index]))

    def forward(self, tensor_list: NestedTensor) -> dict[int, NestedTensor]:
        x = self.patch_embed(tensor_list.tensors)
        height, width = x.shape[2:]
        x = self.pos_drop(x.flatten(2).transpose(1, 2))
        outputs: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            stage, stage_h, stage_w, x, height, width = layer(x, height, width)
            if index in self.out_indices:
                stage = getattr(self, f"norm{index}")(stage)
                outputs.append(
                    stage.view(-1, stage_h, stage_w, self.num_features[index])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
        nested: dict[int, NestedTensor] = {}
        for index, output in enumerate(outputs):
            mask = tensor_list.mask
            if mask is None:
                raise ValueError("Swin DINO backbone requires a padding mask")
            resized = F.interpolate(mask[None].float(), size=output.shape[-2:])
            nested[index] = NestedTensor(output, resized.to(torch.bool)[0])
        return nested


__all__ = ["SwinTransformer"]
