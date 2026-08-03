"""Native BiRefNet architecture (Bilateral Reference dichotomous segmentation).

Faithful port of the MIT-licensed BiRefNet (https://github.com/ZhengPeng7/BiRefNet,
Peng Zheng et al.), covering the inference (``eval``) forward path of the released
``BiRefNet`` (general, Swin-L) and ``BiRefNet_lite`` (Swin-T) weights. Module and
parameter names mirror upstream so the released ``state_dict`` loads with
``strict=True`` and the forward is bit-identical at fp32.

The backbone is the original Swin Transformer v1 lineage (per-output-stage norms,
patch-merging at the end of each stage, explicit H/W threading), which is a
different Swin than the timm-parity tower in ``libreyolo/models/swin/nn.py`` used
by the open-vocabulary detectors; the two have incompatible module layouts and
state-dict key schemas (see the family NOTICE for the reuse verdict).

Design constraints:
- Manual window attention (no ``scaled_dot_product_attention``) so results are
  deterministic and bit-exact against the upstream non-SDPA path.
- BatchNorm everywhere (the released weights were trained with batch size > 1).
- torch + torchvision only, no intra-package imports, so the parity harness can
  import this module standalone in a throwaway environment.
- Training-only branches (multi-scale supervision heads, gradient-label targets)
  are instantiated for strict loading but not exercised in the eval forward.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


# ======================================================================
# Swin Transformer v1 backbone (original lineage)
# ======================================================================


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """(num_windows*B, window_size, window_size, C) -> (B, H, W, C)."""
    C = int(windows.shape[-1])
    x = windows.view(-1, H // window_size, W // window_size, window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class WindowAttention(nn.Module):
    """Window multi-head self-attention with a relative-position-bias table.

    Defaults to the manual computation ``(q * scale) @ k^T + bias`` so the
    forward is bit-identical to the upstream reference run with SDPA disabled.
    ``fused_attn`` opts into the fused kernels and gives that up.
    """

    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        # Persistent to match the upstream checkpoint (the buffer is saved).
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        self.softmax = nn.Softmax(dim=-1)
        # Opt-in fused SDPA. Default off: weights/parity_birefnet.py pins
        # max_abs_diff == 0.0 against upstream BiRefNet and fused kernels
        # accumulate in a different order. Flip with
        # libreyolo.kernels.attention.set_fused_attention(model).
        self.fused_attn = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        relative_position_bias = (
            self.relative_position_bias_table[self.relative_position_index.view(-1)]
            .view(N, N, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(dtype=q.dtype, device=q.device)
        )
        if self.fused_attn and not torch.onnx.is_in_onnx_export():
            # SDPA takes one additive float mask, so the relative position bias
            # and the shifted-window mask are summed into it. The window mask is
            # (nW, N, N) and the batch is laid out as (batch, nW) flattened, so
            # repeat tiles it per window.
            attn_mask = relative_position_bias
            if mask is not None:
                attn_mask = attn_mask + mask.to(dtype=q.dtype).unsqueeze(1).repeat(
                    B_ // mask.shape[0], 1, 1, 1
                )
            fused = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
                scale=self.scale,
            )
            return self.proj_drop(self.proj(fused.transpose(1, 2).reshape(B_, N, C)))
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + relative_position_bias
        if mask is not None:
            mask = mask.to(dtype=q.dtype)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0,
                 qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, (window_size, window_size), num_heads, qkv_bias=qkv_bias)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.H = None
        self.W = None

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask).view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0, qkv_bias=True,
                 norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None
        # The shifted-window attention mask is fixed for a given (H, W) grid;
        # cache it so the dual-scale encoder does not rebuild the same tensor on
        # every forward call (the input-resolution contract is fixed anyway).
        self._attn_mask_cache: dict = {}

    def _get_attn_mask(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(H), int(W), str(device), dtype)
        cached = self._attn_mask_cache.get(key)
        if cached is not None:
            return cached
        Hp = int((H + self.window_size - 1) // self.window_size) * self.window_size
        Wp = int((W + self.window_size - 1) // self.window_size) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)).to(dtype)
        self._attn_mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int):
        attn_mask = self._get_attn_mask(H, W, x.device, x.dtype)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class SwinTransformer(nn.Module):
    """Swin v1 backbone returning the 4 stage feature maps as NCHW tensors."""

    def __init__(self, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm,
                 patch_norm=True, out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            patch_size=4, in_channels=3, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        self.pos_drop = nn.Dropout(0.0)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            ))

        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        for i_layer in out_indices:
            self.add_module(f"norm{i_layer}", norm_layer(self.num_features[i_layer]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        outs = []
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            x_out, H, W, x, Wh, Ww = self.layers[i](x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)


# ======================================================================
# Bilateral-reference decoder
# ======================================================================


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.modulator_conv = nn.Conv2d(in_channels, 1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.regular_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))
        return deform_conv2d(
            input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias,
            padding=(self.padding, self.padding), mask=modulator, stride=self.stride,
        )


class _ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding):
        super().__init__()
        self.atrous_conv = DeformableConv2d(in_channels, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.atrous_conv(x)))


class ASPPDeformable(nn.Module):
    def __init__(self, in_channels, out_channels=None, parallel_block_sizes=(1, 3, 7)):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256
        self.aspp1 = _ASPPModuleDeformable(in_channels, self.in_channelster, 1, padding=0)
        self.aspp_deforms = nn.ModuleList([
            _ASPPModuleDeformable(in_channels, self.in_channelster, conv_size, padding=int(conv_size // 2))
            for conv_size in parallel_block_sizes
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, self.in_channelster, 1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_channelster),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(self.in_channelster * (2 + len(self.aspp_deforms)), out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.aspp1(x)
        x_aspp_deforms = [aspp_deform(x) for aspp_deform in self.aspp_deforms]
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, *x_aspp_deforms, x5), dim=1)
        return self.dropout(self.relu(self.bn1(self.conv1(x))))


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super().__init__()
        inter_channels = 64
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = ASPPDeformable(in_channels=inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu_in(self.bn_in(self.conv_in(x)))
        x = self.dec_att(x)
        return self.bn_out(self.conv_out(x))


class BasicLatBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleConvs(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.conv1(x))


def _image2patches_split(x: torch.Tensor, patch_ref: torch.Tensor) -> torch.Tensor:
    """Native equivalent of BiRefNet ``image2patches`` with the split transform.

    ``b c (hg h) (wg w) -> b (c hg wg) h w`` where the grid comes from the ratio
    of ``x`` to ``patch_ref``.
    """
    B, C, H, W = x.shape
    gh = H // patch_ref.shape[-2]
    gw = W // patch_ref.shape[-1]
    h = H // gh
    w = W // gw
    x = x.view(B, C, gh, h, gw, w).permute(0, 1, 2, 4, 3, 5).contiguous()
    return x.view(B, C * gh * gw, h, w)


class Decoder(nn.Module):
    """Bilateral-reference decoder (inference path)."""

    def __init__(self, channels: List[int]):
        super().__init__()
        N_dec_ipt = 64
        ic = 64
        self.split = True

        ipt_blk_in_channels = [2 ** i * 3 for i in (10, 8, 6, 4, 0)]
        ipt_blk_out_channels = [channels[i] // 8 for i in range(4)]
        self.ipt_blk5 = SimpleConvs(ipt_blk_in_channels[0], ipt_blk_out_channels[0], inter_channels=ic)
        self.ipt_blk4 = SimpleConvs(ipt_blk_in_channels[1], ipt_blk_out_channels[0], inter_channels=ic)
        self.ipt_blk3 = SimpleConvs(ipt_blk_in_channels[2], ipt_blk_out_channels[1], inter_channels=ic)
        self.ipt_blk2 = SimpleConvs(ipt_blk_in_channels[3], ipt_blk_out_channels[2], inter_channels=ic)
        self.ipt_blk1 = SimpleConvs(ipt_blk_in_channels[4], ipt_blk_out_channels[3], inter_channels=ic)

        bb_neck_out_channels = list(channels)
        dec_blk_out_channels = list(bb_neck_out_channels[1:]) + [bb_neck_out_channels[-1] // 2]
        dec_blk_in_channels = [bb_neck_out_channels[i] + ipt_blk_out_channels[max(0, i - 1)] for i in range(4)]

        self.decoder_block4 = BasicDecBlk(dec_blk_in_channels[0], dec_blk_out_channels[0])
        self.decoder_block3 = BasicDecBlk(dec_blk_in_channels[1], dec_blk_out_channels[1])
        self.decoder_block2 = BasicDecBlk(dec_blk_in_channels[2], dec_blk_out_channels[2])
        self.decoder_block1 = BasicDecBlk(dec_blk_in_channels[3], dec_blk_out_channels[3])
        self.conv_out1 = nn.Sequential(nn.Conv2d(dec_blk_out_channels[3] + ipt_blk_out_channels[3], 1, 1, 1, 0))

        self.lateral_block4 = BasicLatBlk(bb_neck_out_channels[1], dec_blk_out_channels[0])
        self.lateral_block3 = BasicLatBlk(bb_neck_out_channels[2], dec_blk_out_channels[1])
        self.lateral_block2 = BasicLatBlk(bb_neck_out_channels[3], dec_blk_out_channels[2])

        # Multi-scale supervision heads (training-only, kept for strict loading).
        self.conv_ms_spvn_4 = nn.Conv2d(dec_blk_out_channels[0], 1, 1, 1, 0)
        self.conv_ms_spvn_3 = nn.Conv2d(dec_blk_out_channels[1], 1, 1, 1, 0)
        self.conv_ms_spvn_2 = nn.Conv2d(dec_blk_out_channels[2], 1, 1, 1, 0)

        # Gradient-reference (out_ref): the attention branch IS used at inference.
        _N = 16
        self.gdt_convs_4 = nn.Sequential(nn.Conv2d(dec_blk_out_channels[0], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
        self.gdt_convs_3 = nn.Sequential(nn.Conv2d(dec_blk_out_channels[1], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
        self.gdt_convs_2 = nn.Sequential(nn.Conv2d(dec_blk_out_channels[2], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
        self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        x, x1, x2, x3, x4 = features
        outs: List[torch.Tensor] = []

        patches = _image2patches_split(x, x4) if self.split else x
        x4 = torch.cat((x4, self.ipt_blk5(F.interpolate(patches, size=x4.shape[2:], mode="bilinear", align_corners=True))), 1)
        p4 = self.decoder_block4(x4)
        p4_gdt = self.gdt_convs_4(p4)
        gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
        p4 = p4 * gdt_attn_4
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode="bilinear", align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)

        patches = _image2patches_split(x, _p3) if self.split else x
        _p3 = torch.cat((_p3, self.ipt_blk4(F.interpolate(patches, size=x3.shape[2:], mode="bilinear", align_corners=True))), 1)
        p3 = self.decoder_block3(_p3)
        p3_gdt = self.gdt_convs_3(p3)
        gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
        p3 = p3 * gdt_attn_3
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode="bilinear", align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)

        patches = _image2patches_split(x, _p2) if self.split else x
        _p2 = torch.cat((_p2, self.ipt_blk3(F.interpolate(patches, size=x2.shape[2:], mode="bilinear", align_corners=True))), 1)
        p2 = self.decoder_block2(_p2)
        p2_gdt = self.gdt_convs_2(p2)
        gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
        p2 = p2 * gdt_attn_2
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode="bilinear", align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)

        patches = _image2patches_split(x, _p1) if self.split else x
        _p1 = torch.cat((_p1, self.ipt_blk2(F.interpolate(patches, size=x1.shape[2:], mode="bilinear", align_corners=True))), 1)
        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode="bilinear", align_corners=True)

        patches = _image2patches_split(x, _p1) if self.split else x
        _p1 = torch.cat((_p1, self.ipt_blk1(F.interpolate(patches, size=x.shape[2:], mode="bilinear", align_corners=True))), 1)
        p1_out = self.conv_out1(_p1)

        outs.append(p1_out)
        return outs


# ======================================================================
# Full model
# ======================================================================


class BiRefNetDims:
    """Per-size Swin backbone dimensions (plain class; standalone-import safe)."""

    def __init__(self, embed_dim, depths, num_heads, window_size, bb_channels):
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.bb_channels = bb_channels  # lateral channels per stage, deep->shallow


BIREFNET_DIMS = {
    # t = BiRefNet_lite (Swin-T tier), l = BiRefNet general (Swin-L tier).
    "t": BiRefNetDims(96, (2, 2, 6, 2), (3, 6, 12, 24), 7, (768, 384, 192, 96)),
    "l": BiRefNetDims(192, (2, 2, 18, 2), (6, 12, 24, 48), 12, (1536, 768, 384, 192)),
}


class LibreBiRefNetModel(nn.Module):
    """BiRefNet: image -> single-channel logit map (sigmoid -> alpha matte).

    ``mul_scl_ipt='cat'`` (dual-scale encoder), ``cxt_num=3`` (multi-scale skip
    context), ``dec_att='ASPPDeformable'``, ``squeeze_block='BasicDecBlk_x1'`` -
    the released General / lite configuration.
    """

    def __init__(self, size: str = "l", dims: BiRefNetDims | None = None):
        # ``dims`` lets architecture-sharing families (feynobg: BiRefNet with a
        # deeper stage 3) reuse this module with their own dimension table.
        super().__init__()
        if dims is None:
            if size not in BIREFNET_DIMS:
                raise ValueError(f"Unknown BiRefNet size {size!r}; expected one of {sorted(BIREFNET_DIMS)}")
            dims = BIREFNET_DIMS[size]
        self.size = size
        d = dims

        self.bb = SwinTransformer(
            embed_dim=d.embed_dim, depths=d.depths, num_heads=d.num_heads,
            window_size=d.window_size, out_indices=(0, 1, 2, 3),
        )

        # mul_scl_ipt='cat' doubles the lateral channels; cxt_num=3 skip context.
        channels = [c * 2 for c in d.bb_channels]
        self._channels = channels
        cxt = channels[1:][::-1][-3:]
        self._cxt = cxt

        self.squeeze_module = nn.Sequential(BasicDecBlk(channels[0] + sum(cxt), channels[0]))
        self.decoder = Decoder(channels)

    def forward_enc(self, x: torch.Tensor):
        x1, x2, x3, x4 = self.bb(x)
        B, C, H, W = x.shape
        x_pyramid = F.interpolate(x, size=(H // 2, W // 2), mode="bilinear", align_corners=True)
        x1_, x2_, x3_, x4_ = self.bb(x_pyramid)
        x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode="bilinear", align_corners=True)], dim=1)
        x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode="bilinear", align_corners=True)], dim=1)
        x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode="bilinear", align_corners=True)], dim=1)
        x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode="bilinear", align_corners=True)], dim=1)

        # cxt: multi-scale skip context concatenated onto the deepest feature.
        x4 = torch.cat(
            (
                *[
                    F.interpolate(x1, size=x4.shape[2:], mode="bilinear", align_corners=True),
                    F.interpolate(x2, size=x4.shape[2:], mode="bilinear", align_corners=True),
                    F.interpolate(x3, size=x4.shape[2:], mode="bilinear", align_corners=True),
                ][-len(self._cxt):],
                x4,
            ),
            dim=1,
        )
        return x1, x2, x3, x4

    def forward_dec(self, x: torch.Tensor, feats) -> torch.Tensor:
        """Squeeze and decode the encoder features.

        Separate from ``forward_enc`` because this half cannot be captured: the
        deformable ASPP blocks call ``torchvision.ops.deform_conv2d``, whose
        CUDA kernel replays to a different result under graph capture. The
        encoder ahead of it captures cleanly, so the two are split to let the
        encoder be replayed while this runs eagerly, leaving results unchanged.
        """
        x1, x2, x3, x4 = feats
        x4 = self.squeeze_module(x4)
        scaled_preds = self.decoder([x, x1, x2, x3, x4])
        # Inference contract: the final (full-resolution) logit map.
        return scaled_preds[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_dec(x, self.forward_enc(x))


__all__ = ["LibreBiRefNetModel", "BiRefNetDims", "BIREFNET_DIMS"]
