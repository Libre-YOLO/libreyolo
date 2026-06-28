"""Native CLIP towers (image ViT + text transformer) for LibreCLIP.

A clean-room re-implementation of the standard OpenAI/open_clip CLIP
architecture in plain ``torch`` — no ``open_clip`` dependency at runtime. The
module structure (and therefore the ``state_dict`` key names) is intentionally
identical to open_clip's ``CLIP`` so LAION/OpenCLIP weights load directly and
forward parity is exact by construction:

* ``visual``            — :class:`VisionTransformer` (class-token pooling).
* ``transformer``       — text :class:`Transformer` (causal mask).
* ``token_embedding`` / ``positional_embedding`` / ``ln_final`` /
  ``text_projection`` / ``logit_scale`` — top-level text + scale params.

All variants ship fp32; the custom open_clip ``LayerNorm`` (fp32-cast) is
numerically identical to :class:`torch.nn.LayerNorm` at fp32, so we use the
stock layer.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Per-variant configuration
# ---------------------------------------------------------------------------


@dataclass
class CLIPConfig:
    embed_dim: int
    image_size: int
    patch_size: int
    vision_width: int
    vision_layers: int
    context_length: int = 77
    vocab_size: int = 49408
    text_width: int = 512
    text_heads: int = 8
    text_layers: int = 12

    @property
    def vision_heads(self) -> int:
        # open_clip derives vision heads as width // 64.
        return self.vision_width // 64


# OpenCLIP ViT-B/32, B/16 (LAION-2B) and an optional L/14.
CLIP_CONFIGS: Dict[str, CLIPConfig] = {
    "b32": CLIPConfig(
        embed_dim=512, image_size=224, patch_size=32,
        vision_width=768, vision_layers=12,
        text_width=512, text_heads=8, text_layers=12,
    ),
    "b16": CLIPConfig(
        embed_dim=512, image_size=224, patch_size=16,
        vision_width=768, vision_layers=12,
        text_width=512, text_heads=8, text_layers=12,
    ),
    "l14": CLIPConfig(
        embed_dim=768, image_size=224, patch_size=14,
        vision_width=1024, vision_layers=24,
        text_width=768, text_heads=12, text_layers=12,
    ),
}


# ---------------------------------------------------------------------------
# Transformer blocks (keys mirror nn.MultiheadAttention + open_clip naming)
# ---------------------------------------------------------------------------


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", nn.GELU()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        attn_out = self.attn(
            self.ln_1(x), self.ln_1(x), self.ln_1(x),
            need_weights=False, attn_mask=mask,
        )[0]
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads, mlp_ratio) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for block in self.resblocks:
            x = block(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    """Image tower: patch-embed conv + class token + transformer, tok-pooled."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        num_patches = self.grid_size * self.grid_size
        scale = width**-0.5

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, mlp_ratio)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, grid**2, width]
        cls = self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1).to(x.dtype)
        x = torch.cat([cls, x], dim=1)  # [B, grid**2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x)
        pooled = x[:, 0]  # class-token pooling
        return pooled @ self.proj


class CLIPModel(nn.Module):
    """Full CLIP: image + text towers, sharing ``state_dict`` keys with open_clip.

    ``encode_image``/``encode_text`` return *un-normalized* features (callers
    L2-normalize); ``logit_scale`` is the learned softmax temperature.
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        self.context_length = config.context_length

        self.visual = VisionTransformer(
            image_size=config.image_size,
            patch_size=config.patch_size,
            width=config.vision_width,
            layers=config.vision_layers,
            heads=config.vision_heads,
            output_dim=config.embed_dim,
        )

        self.transformer = Transformer(config.text_width, config.text_layers, config.text_heads)
        self.token_embedding = nn.Embedding(config.vocab_size, config.text_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(config.context_length, config.text_width)
        )
        self.ln_final = nn.LayerNorm(config.text_width)
        self.text_projection = nn.Parameter(torch.empty(config.text_width, config.embed_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        self.register_buffer("attn_mask", self._build_causal_mask(), persistent=False)
        self._init_text_parameters()

    def _init_text_parameters(self) -> None:
        # positional_embedding / text_projection are allocated with torch.empty;
        # initialize them (as open_clip does) so a fresh, un-loaded model is sane
        # rather than holding uninitialized memory. Real checkpoints overwrite
        # these on load.
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.config.text_width**-0.5)

    def _build_causal_mask(self) -> torch.Tensor:
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # 0 on/below diagonal, -inf above
        return mask

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text)  # [B, n_ctx, width]
        x = x + self.positional_embedding
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        # EOT pooling: the end-of-text token has the highest id in each row.
        pooled = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
        return pooled @ self.text_projection


def build_clip_model(size: str) -> CLIPModel:
    if size not in CLIP_CONFIGS:
        raise ValueError(
            f"Unknown LibreCLIP size {size!r}; choose one of {sorted(CLIP_CONFIGS)}."
        )
    return CLIPModel(CLIP_CONFIGS[size])
