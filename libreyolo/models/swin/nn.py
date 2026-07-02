"""Native Swin Transformer backbone (timm-parity).

Clean-room port of the Apache-2.0 timm ``SwinTransformer`` (features_only), used
as the vision backbone by OMDet-Turbo and Grounding DINO. Module/parameter names
mirror timm so the timm ``state_dict`` loads directly and the forward is
bit-identical. Eager attention only (portable, deterministic).

Operates in channels-last (NHWC) internally, matching timm. Returns the stage
feature maps selected by ``out_indices`` as NHWC tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, c)


def window_reverse(windows: torch.Tensor, ws: int, h: int, w: int) -> torch.Tensor:
    c = windows.shape[-1]
    x = windows.view(-1, h // ws, w // ws, ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, h, w, c)


def get_relative_position_index(win_h: int, win_w: int) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid(torch.arange(win_h), torch.arange(win_w), indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    rel = rel.permute(1, 2, 0).contiguous()
    rel[:, :, 0] += win_h - 1
    rel[:, :, 1] += win_w - 1
    rel[:, :, 0] *= 2 * win_w - 1
    return rel.sum(-1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)
        self.window_area = window_size * window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        self.register_buffer(
            "relative_position_index",
            get_relative_position_index(window_size, window_size),
            persistent=False,
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def _rel_pos_bias(self) -> torch.Tensor:
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_area, self.window_area, -1
        )
        return bias.permute(2, 0, 1).contiguous().unsqueeze(0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self._rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, -1)
        return self.proj(x)


class SwinMlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMlp(dim, int(dim * mlp_ratio))

    def _attn_mask(self, hp: int, wp: int, device, dtype) -> torch.Tensor | None:
        if not self.shift_size:
            return None
        ws, ss = self.window_size, self.shift_size
        img_mask = torch.zeros((1, hp, wp, 1), dtype=dtype, device=device)
        cnt = 0
        for h in ((0, -ws), (-ws, -ss), (-ss, None)):
            for w in ((0, -ws), (-ws, -ss), (-ss, None)):
                img_mask[:, h[0]:h[1], w[0]:w[1], :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws).view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def _forward_attn(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        ws, ss = self.window_size, self.shift_size
        shifted = torch.roll(x, shifts=(-ss, -ss), dims=(1, 2)) if ss else x
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        shifted = F.pad(shifted, (0, 0, 0, pad_w, 0, pad_h))
        _, hp, wp, _ = shifted.shape
        x_windows = window_partition(shifted, ws).view(-1, ws * ws, c)
        mask = self._attn_mask(hp, wp, x.device, x.dtype)
        attn_windows = self.attn(x_windows, mask=mask).view(-1, ws, ws, c)
        shifted = window_reverse(attn_windows, ws, hp, wp)[:, :h, :w, :].contiguous()
        return torch.roll(shifted, shifts=(ss, ss), dims=(1, 2)) if ss else shifted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        x = x + self._forward_attn(self.norm1(x))
        x = x.reshape(b, -1, c)
        x = x + self.mlp(self.norm2(x))
        return x.reshape(b, h, w, c)


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        _, h, w, _ = x.shape
        x = x.reshape(b, h // 2, 2, w // 2, 2, c).permute(0, 1, 3, 4, 2, 5).flatten(3)
        return self.reduction(self.norm(x))


class SwinStage(nn.Module):
    def __init__(self, dim, out_dim, depth, num_heads, window_size, downsample):
        super().__init__()
        self.downsample = PatchMerging(dim, out_dim) if downsample else nn.Identity()
        shift = window_size // 2
        self.blocks = nn.ModuleList(
            [
                SwinBlock(out_dim, num_heads, window_size, 0 if i % 2 == 0 else shift)
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).permute(0, 2, 3, 1)  # NCHW -> NHWC
        return self.norm(x)


@dataclass
class SwinDims:
    embed_dim: int = 96
    depths: tuple = (2, 2, 6, 2)
    num_heads: tuple = (3, 6, 12, 24)
    window_size: int = 7
    patch_size: int = 4
    out_indices: tuple = (1, 2, 3)


class SwinBackbone(nn.Module):
    """timm-parity Swin backbone returning NHWC feature maps at ``out_indices``."""

    def __init__(self, d: SwinDims = SwinDims()):
        super().__init__()
        self.out_indices = set(d.out_indices)
        self.patch_embed = PatchEmbed(3, d.embed_dim, d.patch_size)
        dims = [d.embed_dim * (2**i) for i in range(len(d.depths))]  # [96,192,384,768]
        self.layers = nn.ModuleList()
        in_dim = d.embed_dim
        for i, depth in enumerate(d.depths):
            self.layers.append(
                SwinStage(in_dim, dims[i], depth, d.num_heads[i], d.window_size, downsample=i > 0)
            )
            in_dim = dims[i]

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        x = self.patch_embed(pixel_values)
        feats = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                feats.append(x)
        return feats


__all__ = ["SwinBackbone", "SwinDims"]
