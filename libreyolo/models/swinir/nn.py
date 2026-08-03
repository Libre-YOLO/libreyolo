"""Native SwinIR generator for image super-resolution.

Adapted from ``models/network_swinir.py`` in the official SwinIR repository,
commit 6545850fbf8df298df73d81f3e8cba638787c8bd (Apache-2.0). This modified
version removes training-only FLOP helpers and the timm runtime dependency,
adds type annotations, and keeps upstream module names for strict checkpoint
loading. See ``NOTICE`` in this directory.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint


def _to_2tuple(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence):
        return int(value[0]), int(value[1])
    return int(value), int(value)


class DropPath(nn.Module):
    """Per-sample stochastic depth, equivalent to timm's inference behavior."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Split a ``[B, H, W, C]`` tensor into non-overlapping windows."""

    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    )


def window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    """Reassemble non-overlapping windows into ``[B, H, W, C]``."""

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
    """Windowed multi-head attention with learned relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows, tokens, 3, self.num_heads, channels // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        bias = relative_bias.permute(2, 0, 1).unsqueeze(0)
        if torch.onnx.is_in_onnx_export():
            # LibreYOLO defaults to ONNX opset 13, which has no symbolic for
            # fused SDPA: keep the exact primitive-op equation while tracing.
            attention = (q * self.scale) @ k.transpose(-2, -1)
            attention = attention + bias
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
            x = attention @ v
        else:
            # SDPA takes one additive float mask, so the relative position bias
            # and the shifted-window mask are summed into it. The window mask
            # is (num_windows, tokens, tokens) and the batch is laid out as
            # (batch, num_windows) flattened, so repeat tiles it per window.
            attn_mask = bias
            if mask is not None:
                num_windows = mask.shape[0]
                attn_mask = bias + mask.unsqueeze(1).repeat(
                    batch_windows // num_windows, 1, 1, 1
                )
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
                scale=self.scale,
            )
        x = x.transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError("shift_size must be in [0, window_size).")

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=_to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        attn_mask = (
            self.calculate_mask(input_resolution) if self.shift_size > 0 else None
        )
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size: tuple[int, int]) -> torch.Tensor:
        height, width = x_size
        image_mask = torch.zeros((1, height, width, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = h_slices
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                image_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = window_partition(image_mask, self.window_size).view(
            -1, self.window_size * self.window_size
        )
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        height, width = x_size
        batch, _, channels = x.shape
        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        shifted = (
            torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if self.shift_size > 0
            else x
        )
        windows = window_partition(shifted, self.window_size).view(
            -1, self.window_size * self.window_size, channels
        )
        mask = (
            self.attn_mask
            if self.input_resolution == x_size
            else self.calculate_mask(x_size).to(device=x.device, dtype=x.dtype)
        )
        attended = self.attn(windows, mask=mask).view(
            -1, self.window_size, self.window_size, channels
        )
        shifted = window_reverse(attended, self.window_size, height, width)
        x = (
            torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            if self.shift_size > 0
            else shifted
        )
        x = x.view(batch, height * width, channels)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[index]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for index in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, x_size, use_reentrant=False)
            else:
                x = block(x, x_size)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: type[nn.Module] | None = None,
    ):
        super().__init__()
        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)
        self.patches_resolution = [
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        ]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x) if self.norm is not None else x


class PatchUnEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: type[nn.Module] | None = None,
    ):
        super().__init__()
        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)
        self.patches_resolution = [
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        ]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        batch = x.shape[0]
        return x.transpose(1, 2).view(batch, self.embed_dim, x_size[0], x_size[1])


class RSTB(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        resi_connection: str = "1conv",
    ):
        super().__init__()
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported residual connection: {resi_connection!r}.")
        self.patch_embed = PatchEmbed(img_size, patch_size, 0, dim, None)
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, 0, dim, None)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        residual = self.residual_group(x, x_size)
        residual = self.patch_unembed(residual, x_size)
        return self.patch_embed(self.conv(residual)) + x


class Upsample(nn.Sequential):
    def __init__(self, scale: int, num_feat: int):
        modules: list[nn.Module] = []
        if scale > 0 and (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                modules.extend(
                    (nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2))
                )
        elif scale == 3:
            modules.extend(
                (nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3))
            )
        else:
            raise ValueError(f"Unsupported scale: {scale}.")
        super().__init__(*modules)


class UpsampleOneStep(nn.Sequential):
    def __init__(
        self,
        scale: int,
        num_feat: int,
        num_out_ch: int,
        input_resolution: tuple[int, int] | None = None,
    ):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        super().__init__(
            nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        )


class SwinIR(nn.Module):
    """SwinIR super-resolution generator with upstream-compatible parameters."""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 64,
        patch_size: int | tuple[int, int] = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Sequence[int] = (6, 6, 6, 6),
        num_heads: Sequence[int] = (6, 6, 6, 6),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        upscale: int = 2,
        img_range: float = 1.0,
        upsampler: str = "",
        resi_connection: str = "1conv",
    ):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = (
            torch.tensor(rgb_mean).view(1, 3, 1, 1)
            if in_chans == 3
            else torch.zeros(1, 1, 1, 1)
        )
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            embed_dim,
            embed_dim,
            norm_layer if patch_norm else None,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, embed_dim, embed_dim)
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, embed_dim)
            )
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [item.item() for item in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            start = sum(depths[:index])
            end = sum(depths[: index + 1])
            self.layers.append(
                RSTB(
                    dim=embed_dim,
                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                    depth=depths[index],
                    num_heads=num_heads[index],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[start:end],
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    img_size=img_size,
                    patch_size=patch_size,
                    resi_connection=resi_connection,
                )
            )
        self.norm = norm_layer(self.num_features)
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported residual connection: {resi_connection!r}.")

        if upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch, tuple(patches_resolution)
            )
        elif upsampler == "nearest+conv":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if not pad_h and not pad_w:
            return x
        mode = (
            "reflect"
            if height > 1 and width > 1 and pad_h < height and pad_w < width
            else "replicate"
        )
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        return self.patch_unembed(self.norm(x), x_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2:]
        x = self.check_image_size(x)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        x = (x - mean) * self.img_range

        if self.upsampler in {"pixelshuffle", "pixelshuffledirect", "nearest+conv"}:
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            if self.upsampler == "pixelshuffle":
                x = self.conv_last(self.upsample(self.conv_before_upsample(x)))
            elif self.upsampler == "pixelshuffledirect":
                x = self.upsample(x)
            else:
                x = self.conv_before_upsample(x)
                x = self.lrelu(
                    self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest"))
                )
                if self.upscale == 4:
                    x = self.lrelu(
                        self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest"))
                    )
                x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            first = self.conv_first(x)
            residual = self.conv_after_body(self.forward_features(first)) + first
            x = x + self.conv_last(residual)
        x = x / self.img_range + mean
        return x[:, :, : height * self.upscale, : width * self.upscale]


__all__ = ["SwinIR"]
