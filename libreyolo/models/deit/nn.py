"""Native DeiT patch-16 image classifiers.

This implementation is derived from the Apache-2.0 timm Vision Transformer
implementation at ``huggingface/pytorch-image-models`` commit
``e98c05a5a15e81188ec62dd5380b8f5c3251075a``. The relevant upstream files are
``timm/models/vision_transformer.py`` and
``timm/layers/{attention,patch_embed,mlp}.py``. Module names and forward
operations intentionally match timm 1.0.28 so the official
``deit_{tiny,small,base}_patch16_224.fb_in1k`` state dictionaries load with
``strict=True`` and produce bit-identical eager logits. See ``NOTICE``.

Only plain 224-pixel DeiT variants ship. Distillation-token, DeiT III, and
384-pixel positional-embedding variants require separate public size contracts
and are out of scope for this first museum port.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_DEFS: Dict[str, Dict[str, int]] = {
    "t": {"embed_dim": 192, "depth": 12, "num_heads": 3},
    "s": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "b": {"embed_dim": 768, "depth": 12, "num_heads": 12},
}

IMAGE_SIZE = 224
PATCH_SIZE = 16
MLP_RATIO = 4
NORM_EPS = 1e-6


class PatchEmbed(nn.Module):
    """Fixed 224x224 image to 14x14 patch-token embedding."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.img_size = (IMAGE_SIZE, IMAGE_SIZE)
        self.patch_size = (PATCH_SIZE, PATCH_SIZE)
        self.grid_size = (IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
            bias=True,
        )
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        torch._assert(
            height == IMAGE_SIZE,
            f"Input height ({height}) does not match DeiT ({IMAGE_SIZE}).",
        )
        torch._assert(
            width == IMAGE_SIZE,
            f"Input width ({width}) does not match DeiT ({IMAGE_SIZE}).",
        )
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class Attention(nn.Module):
    """Multi-head self-attention matching timm's fused DeiT path."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dim = dim
        self.scale = self.head_dim**-0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)
        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        x = self.norm(x)
        x = self.proj(x)
        return self.proj_drop(x)


class Mlp(nn.Module):
    """Transformer MLP with timm-compatible parameter names."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden_dim = dim * MLP_RATIO
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)


class Block(nn.Module):
    """Pre-normalized DeiT transformer block."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=NORM_EPS)
        self.attn = Attention(dim, num_heads)
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=NORM_EPS)
        self.mlp = Mlp(dim)
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class DeiT(nn.Module):
    """Plain DeiT patch-16 classifier with timm-compatible state keys."""

    def __init__(self, size: str = "t", num_classes: int = 1000) -> None:
        super().__init__()
        if size not in ARCH_DEFS:
            raise ValueError(f"Unknown DeiT size {size!r}; choose from {list(ARCH_DEFS)}.")
        spec = ARCH_DEFS[size]
        embed_dim = int(spec["embed_dim"])
        depth = int(spec["depth"])
        num_heads = int(spec["num_heads"])

        self.size = size
        self.num_classes = num_classes
        self.in_chans = 3
        self.global_pool = "token"
        self.num_features = embed_dim
        self.head_hidden_size = embed_dim
        self.embed_dim = embed_dim
        self.num_prefix_tokens = 1

        self.patch_embed = PatchEmbed(embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.empty(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(0.0)
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=NORM_EPS)
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(0.0)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_classifier(self, num_classes: int) -> None:
        """Replace the ImageNet head while preserving its device and dtype."""
        self.num_classes = num_classes
        weight = self.head.weight
        self.head = nn.Linear(self.embed_dim, num_classes).to(
            device=weight.device, dtype=weight.dtype
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        return self.norm(x)

    def forward_head(
        self, x: torch.Tensor, pre_logits: bool = False
    ) -> torch.Tensor:
        x = x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))
