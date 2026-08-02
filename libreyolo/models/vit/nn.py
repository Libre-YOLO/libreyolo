"""Classic Vision Transformer (ViT) for LibreYOLO.

Derived from the Apache-2.0 timm Vision Transformer implementation at
``huggingface/pytorch-image-models`` v1.0.28. This file is modified for
LibreYOLO: it keeps only the fixed-224, patch-16, learned-class-token graph
used by the four shipped AugReg checkpoints and removes timm's dynamic-image,
alternate-pooling, register-token, and feature-extraction extensions. Module
names intentionally remain checkpoint-compatible. See ``NOTICE`` for the
exact upstream pin and attribution.

The architecture originates with "An Image Is Worth 16x16 Words" (ICLR 2021)
and uses a learned class token, learned absolute position embeddings, pre-norm
self-attention/MLP blocks, a final LayerNorm, and a linear classifier.

Copyright 2020 Ross Wightman
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 224
PATCH_SIZE = 16
NORM_EPS = 1e-6


@dataclass(frozen=True)
class ViTSpec:
    """One fixed patch-16 ViT capacity tier."""

    embed_dim: int
    depth: int
    num_heads: int


ARCH_DEFS: Dict[str, ViTSpec] = {
    "ti": ViTSpec(embed_dim=192, depth=12, num_heads=3),
    "s": ViTSpec(embed_dim=384, depth=12, num_heads=6),
    "b": ViTSpec(embed_dim=768, depth=12, num_heads=12),
    "l": ViTSpec(embed_dim=1024, depth=24, num_heads=16),
}


class LayerNorm(nn.LayerNorm):
    """LayerNorm matching timm's default non-fast path and epsilon."""

    def __init__(self, dim: int):
        super().__init__(dim, eps=NORM_EPS, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )


class PatchEmbed(nn.Module):
    """Patchify a fixed 224px image with a stride-16 convolution."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.img_size = (IMG_SIZE, IMG_SIZE)
        self.patch_size = (PATCH_SIZE, PATCH_SIZE)
        self.grid_size = (IMG_SIZE // PATCH_SIZE, IMG_SIZE // PATCH_SIZE)
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
        torch._assert(
            x.shape[-2] == IMG_SIZE,
            f"Input height must be {IMG_SIZE} for the fixed ViT graph.",
        )
        torch._assert(
            x.shape[-1] == IMG_SIZE,
            f"Input width must be {IMG_SIZE} for the fixed ViT graph.",
        )
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class Attention(nn.Module):
    """Multi-head self-attention with one fused QKV projection."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads:
            raise ValueError("ViT embedding width must be divisible by num_heads.")
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
        self.gate = None
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        x = self.norm(x)
        x = self.proj(x)
        return self.proj_drop(x)


class Mlp(nn.Module):
    """ViT MLP: Linear, GELU, dropout, Linear, dropout."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
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
    """Pre-normalized self-attention and MLP residual block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """Fixed patch-16 classic ViT with timm-compatible state-dict names."""

    def __init__(
        self,
        size: str = "ti",
        num_classes: int = 1000,
        init_weights: bool = True,
    ):
        super().__init__()
        if size not in ARCH_DEFS:
            raise ValueError(
                f"Unknown ViT size {size!r}; choose from {list(ARCH_DEFS)}."
            )
        spec = ARCH_DEFS[size]
        self.size = size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = spec.embed_dim
        self.num_prefix_tokens = 1

        self.patch_embed = PatchEmbed(spec.embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, spec.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.empty(1, self.patch_embed.num_patches + 1, spec.embed_dim)
        )
        self.pos_drop = nn.Dropout(0.0)
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.Sequential(
            *[Block(spec.embed_dim, spec.num_heads) for _ in range(spec.depth)]
        )
        self.norm = LayerNorm(spec.embed_dim)
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(0.0)
        self.head = nn.Linear(spec.embed_dim, num_classes)

        if init_weights:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.normal_(self.cls_token, std=1e-6)
            self.apply(self._init_linear)

    @staticmethod
    def _init_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_classifier(self, num_classes: int) -> None:
        """Replace the classifier while preserving the current device/dtype."""
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
