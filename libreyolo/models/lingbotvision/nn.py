"""LingBot-Vision ViT backbone + dense semantic head.

The backbone is a native port of the LingBot-Vision Vision Transformer
(Robbyant's Apache-2.0 release, https://github.com/robbyant/lingbot-vision):
a DINOv3-style ViT with axial RoPE over the patch grid, a [CLS] token plus 4
register ("storage") tokens, LayerScale, and a masked-K-bias fused QKV
projection. Module and parameter names mirror the reference implementation
exactly, so upstream backbone checkpoints load with a pure metadata wrap (no
key remapping) and the parity script can cross-load either direction.

Only the inference forward path is ported. The reference training machinery
(multi-crop token lists, masked-token modeling, stochastic depth, RoPE
coordinate augmentation) is deliberately absent: LibreYOLO trains this family
by attaching a dense head to the frozen or fine-tuned backbone, never by
re-running the upstream self-supervised pretraining.

``LingBotVisionSemanticSegmenter`` follows the linear-probing protocol of the
upstream technical report: a single 1x1 convolution over the frozen patch-token
grid. Inputs are ``[0, 1]`` RGB floats; ImageNet standardization is applied
inside ``forward`` (the RF-DETR semantic house convention). In training mode
with targets, the loss is computed internally and returned as
``{"total_loss": ..., "sem": ...}``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

IGNORE_INDEX = 255

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class LingBotVisionSizeConfig:
    embed_dim: int
    depth: int
    num_heads: int
    ffn_layer: str  # "mlp" | "swiglu"
    qkv_bias: bool


# All sizes: patch 16, LayerNorm eps 1e-5 ("layernormbf16" in the reference
# configs), LayerScale init 1e-5, 4 storage tokens, mask_k_bias, RoPE base 100
# with fp32 tables and "separate" coordinate normalization.
SIZE_CONFIGS: Dict[str, LingBotVisionSizeConfig] = {
    "s": LingBotVisionSizeConfig(384, 12, 6, "mlp", True),
    "b": LingBotVisionSizeConfig(768, 12, 12, "mlp", True),
    "l": LingBotVisionSizeConfig(1024, 24, 16, "mlp", True),
    "g": LingBotVisionSizeConfig(1536, 40, 24, "swiglu", False),
}

PATCH_SIZE = 16
N_STORAGE_TOKENS = 4
LAYERSCALE_INIT = 1e-5
ROPE_BASE = 100.0


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = LAYERSCALE_INIT) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_values))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B, C, H, W) -> (B, h, w, D)."""

    def __init__(self, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # B D h w
        return x.permute(0, 2, 3, 1)  # B h w D


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN with the reference hidden-width rounding (2/3 ratio, align 8)."""

    def __init__(self, in_features: int, hidden_features: int, bias: bool = True) -> None:
        super().__init__()
        d = int(hidden_features * 2 / 3)
        swiglu_hidden = d + (-d % 8)
        self.w1 = nn.Linear(in_features, swiglu_hidden, bias=bias)
        self.w2 = nn.Linear(in_features, swiglu_hidden, bias=bias)
        self.w3 = nn.Linear(swiglu_hidden, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def _rope_rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (_rope_rotate_half(x) * sin)


class RopePositionEmbedding(nn.Module):
    """Axial rotary position embedding over the 2D patch grid.

    No learnable parameters; ``periods`` is a persistent buffer so checkpoints
    carry it (and upstream checkpoints populate it on load). Coordinates are
    normalized per axis to [-1, 1] ("separate" mode) and the sin/cos tables are
    computed in fp32, matching the reference configs. The reference
    training-time coordinate augmentations (shift/jitter/rescale) are not
    ported; they are inactive in eval mode and this port never pretrains.
    """

    def __init__(self, embed_dim: int, num_heads: int, base: float = ROPE_BASE) -> None:
        super().__init__()
        if embed_dim % (4 * num_heads):
            raise ValueError(f"embed_dim={embed_dim} must be divisible by 4*num_heads={4 * num_heads}")
        d_head = embed_dim // num_heads
        self.d_head = d_head
        periods = base ** (2 * torch.arange(d_head // 4, dtype=torch.float32) / (d_head // 2))
        self.register_buffer("periods", periods, persistent=True)

    def forward(self, *, H: int, W: int) -> Tuple[Tensor, Tensor]:
        device = self.periods.device
        coords_h = torch.arange(0.5, H, device=device, dtype=torch.float32) / H
        coords_w = torch.arange(0.5, W, device=device, dtype=torch.float32) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods.float()[None, None, :]  # [HW, 2, D/4]
        angles = angles.flatten(1, 2).tile(2)  # [HW, D]
        return torch.sin(angles), torch.cos(angles)


class LinearKMaskedBias(nn.Linear):
    """Fused QKV linear whose bias is elementwise-masked by a persistent buffer.

    With ``mask_k_bias`` the mask zeroes the K third of the bias so keys carry
    no bias while Q and V biases pass through. The buffer ships in upstream
    checkpoints; a fresh module initializes it directly (ones with the K third
    zeroed) rather than the reference's NaN-until-init_weights sentinel.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        o = self.out_features
        if o % 3:
            raise ValueError("fused QKV out_features must be divisible by 3")
        if self.bias is not None:
            mask = torch.ones_like(self.bias)
            mask[o // 3 : 2 * o // 3] = 0
            self.register_buffer("bias_mask", mask)

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = LinearKMaskedBias(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = (t.transpose(1, 2) for t in torch.unbind(qkv, 2))  # [B, heads, N, d]
        if rope is not None:
            sin, cos = rope
            prefix = N - sin.shape[-2]
            in_dtype = q.dtype
            q, k = q.to(sin.dtype), k.to(sin.dtype)
            q = torch.cat((q[:, :, :prefix, :], _rope_apply(q[:, :, prefix:, :], sin, cos)), dim=-2)
            k = torch.cat((k[:, :, :prefix, :], _rope_apply(k[:, :, prefix:, :], sin, cos)), dim=-2)
            q, k = q.to(in_dtype), k.to(in_dtype)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_layer: str, qkv_bias: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        if ffn_layer == "swiglu":
            self.mlp: nn.Module = SwiGLUFFN(dim, int(dim * 4))
        else:
            self.mlp = Mlp(dim, int(dim * 4))
        self.ls2 = LayerScale(dim)

    def forward(self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


class LingBotVisionBackbone(nn.Module):
    """The LingBot-Vision ViT trunk, inference path only.

    ``forward`` returns ``(cls_token, patch_tokens)`` where ``patch_tokens`` is
    ``[B, h*w, D]`` after the final LayerNorm. Parameter names match the
    reference (``patch_embed.proj``, ``cls_token``, ``storage_tokens``,
    ``mask_token``, ``rope_embed.periods``, ``blocks.N.*``, ``norm``), so an
    upstream ``model.pt`` loads with ``strict=True``. ``mask_token`` exists only
    for checkpoint compatibility; the masked-modeling path is not ported.
    """

    def __init__(self, size: str = "s") -> None:
        super().__init__()
        if size not in SIZE_CONFIGS:
            raise ValueError(f"Unknown LingBot-Vision size {size!r}; expected one of {sorted(SIZE_CONFIGS)}")
        cfg = SIZE_CONFIGS[size]
        self.size = size
        self.embed_dim = cfg.embed_dim
        self.patch_size = PATCH_SIZE
        self.n_storage_tokens = N_STORAGE_TOKENS

        self.patch_embed = PatchEmbed(PATCH_SIZE, 3, cfg.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, N_STORAGE_TOKENS, cfg.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, cfg.embed_dim))
        self.rope_embed = RopePositionEmbedding(cfg.embed_dim, num_heads=cfg.num_heads)
        self.blocks = nn.ModuleList(
            SelfAttentionBlock(cfg.embed_dim, cfg.num_heads, cfg.ffn_layer, cfg.qkv_bias)
            for _ in range(cfg.depth)
        )
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=1e-5)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, _, H, W = x.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"Input {H}x{W} must be divisible by patch size {self.patch_size}")
        h, w = H // self.patch_size, W // self.patch_size

        tokens = self.patch_embed(x).flatten(1, 2)  # [B, h*w, D]
        prefix = torch.cat([self.cls_token, self.storage_tokens], dim=1).expand(B, -1, -1)
        tokens = torch.cat([prefix, tokens], dim=1)

        rope = self.rope_embed(H=h, W=w)
        for blk in self.blocks:
            tokens = blk(tokens, rope=rope)
        tokens = self.norm(tokens)

        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1 + self.n_storage_tokens :]
        return cls_token, patch_tokens


class LingBotVisionSemanticSegmenter(nn.Module):
    """LingBot-Vision backbone + 1x1 dense head (the report's linear probe).

    Input: ``[B, 3, H, W]`` RGB floats in ``[0, 1]``; ImageNet standardization
    is applied internally. Eval forward returns ``{"semantic_logits": [B, nc,
    H/16, W/16]}`` (patch-grid resolution; callers interpolate). Training
    forward with ``targets`` (``[B, H', W']`` class IDs, 255 = ignore)
    interpolates logits to the target size and returns
    ``{"total_loss": ..., "sem": ...}``.
    """

    IGNORE_INDEX = IGNORE_INDEX

    def __init__(self, size: str = "s", num_classes: int = 150) -> None:
        super().__init__()
        self.size = size
        self.num_classes = num_classes
        self.backbone = LingBotVisionBackbone(size=size)
        self.predict = nn.Conv2d(self.backbone.embed_dim, num_classes, kernel_size=1)
        self._init_head(self.predict)
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @staticmethod
    def _init_head(module: nn.Conv2d) -> None:
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, targets: Optional[Tensor] = None):
        x = (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)
        _, patch_tokens = self.backbone(x)
        B, N, D = patch_tokens.shape
        h = x.shape[-2] // self.backbone.patch_size
        w = x.shape[-1] // self.backbone.patch_size
        feat = patch_tokens.transpose(1, 2).reshape(B, D, h, w)
        logits = self.predict(feat)

        if self.training and targets is not None:
            logits_full = F.interpolate(
                logits.float(), size=targets.shape[-2:], mode="bilinear", align_corners=False
            )
            targets = targets.long()
            if bool((targets != self.IGNORE_INDEX).any()):
                loss = F.cross_entropy(logits_full, targets, ignore_index=self.IGNORE_INDEX)
            else:
                # cross_entropy returns NaN when every pixel is ignored; emit a
                # finite zero that still carries a grad_fn.
                loss = logits_full.sum() * 0.0
            return {"total_loss": loss, "sem": loss}

        return {"semantic_logits": logits}


__all__ = [
    "IGNORE_INDEX",
    "LingBotVisionBackbone",
    "LingBotVisionSemanticSegmenter",
    "LingBotVisionSizeConfig",
    "SIZE_CONFIGS",
]
