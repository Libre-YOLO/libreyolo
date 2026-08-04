"""Native SigLIP towers (image ViT + text transformer) for LibreSigLIP2.

A clean-room re-implementation of the SigLIP architecture in plain ``torch`` -
no ``transformers`` dependency at runtime. The module structure (and therefore
the ``state_dict`` key names) is intentionally identical to the reference
``transformers`` ``SiglipModel`` so the Apache-2.0 SigLIP 2 checkpoints load
directly (0 missing / 0 unexpected keys) and forward parity is exact by
construction.

Architecture note (recorded because it is easy to get wrong): the SigLIP 2
release ships two model families. The fixed-resolution checkpoints shipped here
(``siglip2-base-patch16-256`` -> ``b16`` and ``siglip2-so400m-patch14-384`` ->
``so400m``) carry ``model_type: "siglip"`` and reuse the original SigLIP
transformer, i.e. transformers' ``SiglipModel``:

* ``vision_model`` - Conv2d patch embed (with bias, no class token), learned
  position embedding, pre-norm encoder, ``post_layernorm``, and a multi-head
  attention **pooling** head (``head``) rather than CLIP's class-token + proj.
* ``text_model`` - token + learned position embeddings, a **bidirectional**
  (no causal mask) pre-norm encoder, ``final_layer_norm``, last-token pooling,
  and a final ``head`` linear projection.
* ``logit_scale`` / ``logit_bias`` - the learned sigmoid temperature and bias.

Only the NaFlex variants use the patchified ``Siglip2Model`` (out of v1 scope).

Encoder self-attention replicates the reference eager attention exactly
(softmax computed in fp32); the pooling head reuses ``nn.MultiheadAttention``,
the same class the reference uses, so both are bit-identical at fp32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...kernels.attention.sdpa import manual_attention_required


# ---------------------------------------------------------------------------
# Per-variant configuration
# ---------------------------------------------------------------------------


@dataclass
class SigLIP2Config:
    # Vision tower
    vision_width: int
    vision_layers: int
    vision_heads: int
    vision_intermediate: int
    image_size: int
    patch_size: int
    # Text tower
    text_width: int
    text_layers: int
    text_heads: int
    text_intermediate: int
    vocab_size: int
    max_position_embeddings: int
    projection_size: int
    # Shared
    num_channels: int = 3
    layer_norm_eps: float = 1e-6

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


# The two fixed-resolution SigLIP 2 checkpoints shipped in v1. Both reuse the
# SigLIP v1 architecture (``model_type: "siglip"``); the size codes bake the
# native resolution in, like CLIP's b32/b16/l14.
SIGLIP2_CONFIGS: Dict[str, SigLIP2Config] = {
    # google/siglip2-base-patch16-256
    "b16": SigLIP2Config(
        vision_width=768, vision_layers=12, vision_heads=12, vision_intermediate=3072,
        image_size=256, patch_size=16,
        text_width=768, text_layers=12, text_heads=12, text_intermediate=3072,
        vocab_size=256000, max_position_embeddings=64, projection_size=768,
    ),
    # google/siglip2-so400m-patch14-384
    "so400m": SigLIP2Config(
        vision_width=1152, vision_layers=27, vision_heads=16, vision_intermediate=4304,
        image_size=384, patch_size=14,
        text_width=1152, text_layers=27, text_heads=16, text_intermediate=4304,
        vocab_size=256000, max_position_embeddings=64, projection_size=1152,
    ),
}


# ---------------------------------------------------------------------------
# Shared transformer blocks (keys mirror transformers' Siglip* modules)
# ---------------------------------------------------------------------------


class SiglipAttention(nn.Module):
    """Multi-head attention with separate q/k/v/out projections (eager math).

    Replicates ``transformers.models.siglip.modeling_siglip.eager_attention_forward``
    exactly: scaled dot product, additive mask, softmax in fp32 cast back to the
    input dtype. At fp32 the cast is a no-op; forcing ``attn_implementation="eager"``
    on the reference makes the two paths bit-identical.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim * num_heads != dim:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        # Opt-in fused SDPA. Default off by design: the manual matmul above is
        # what makes the text tower and vision encoder bit-identical to the
        # reference (tests/unit/test_siglip2_parity.py, _EXACT_TOL = 0.0), and
        # the reference is forced to attn_implementation="eager" to get there.
        # Flip with libreyolo.kernels.attention.set_fused_attention(model).
        self.fused_attn = False

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_shape = x.shape[:-1]  # (B, T)
        hidden_shape = (*input_shape, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(x).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        if self.fused_attn and not manual_attention_required():
            # attn_mask is already additive float when present.
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
                scale=self.scale,
            )
        else:
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.out_proj(attn_output)


class SiglipMLP(nn.Module):
    """Two-layer MLP with the tanh-approximated GELU SigLIP uses (``gelu_pytorch_tanh``)."""

    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate)
        # gelu_pytorch_tanh == nn.functional.gelu(x, approximate="tanh").
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(intermediate, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(x)))


class SiglipEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, intermediate: int, eps: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=eps)
        self.self_attn = SiglipAttention(dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = SiglipMLP(dim, intermediate)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), attn_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class SiglipEncoder(nn.Module):
    def __init__(self, dim: int, layers: int, num_heads: int, intermediate: int, eps: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(dim, num_heads, intermediate, eps) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------


class SiglipVisionEmbeddings(nn.Module):
    """Conv2d patch embed (with bias, no class token) + learned position embed."""

    def __init__(self, config: SigLIP2Config):
        super().__init__()
        dim = config.vision_width
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",
        )
        num_patches = config.num_patches
        self.position_embedding = nn.Embedding(num_patches, dim)
        self.register_buffer(
            "position_ids", torch.arange(num_patches).expand((1, -1)), persistent=False
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)  # [B, dim, gh, gw]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)  # [B, num_patches, dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Attention pooling (MAP) head: a learned probe attends over the tokens.

    The reference stores its projections in a ``torch.nn.MultiheadAttention``
    (``attention.in_proj_weight`` / ``attention.out_proj.*``). We keep that
    submodule for weight storage and exact ``state_dict`` key compatibility, but
    run the attention with explicit eager math in :meth:`_map_attention`.

    Why not just call the module: ``nn.MultiheadAttention`` selects a fused
    native kernel when no forward hooks are attached, which drifts from the
    plain-math result by ~5e-7. The reference model carries output-capture hooks
    that force it onto the slow (eager) path, so its output equals the explicit
    eager computation exactly. Replicating that eager math here is what makes the
    two bit-identical (max_abs_diff == 0) rather than merely close.
    """

    def __init__(self, config: SigLIP2Config):
        super().__init__()
        dim = config.vision_width
        self.num_heads = config.vision_heads
        self.probe = nn.Parameter(torch.randn(1, 1, dim))
        self.attention = nn.MultiheadAttention(dim, config.vision_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(dim, config.vision_intermediate)

    def _map_attention(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Eager multi-head cross-attention using ``self.attention``'s weights."""
        embed_dim = self.attention.embed_dim
        num_heads = self.num_heads
        head_dim = embed_dim // num_heads
        in_w = self.attention.in_proj_weight
        in_b = self.attention.in_proj_bias
        q = F.linear(query, in_w[:embed_dim], in_b[:embed_dim])
        k = F.linear(kv, in_w[embed_dim : 2 * embed_dim], in_b[embed_dim : 2 * embed_dim])
        v = F.linear(kv, in_w[2 * embed_dim :], in_b[2 * embed_dim :])

        batch, tgt_len, _ = q.shape
        src_len = k.shape[1]
        q = q.view(batch, tgt_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, src_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch, src_len, num_heads, head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5)
        # Match SiglipAttention: compute the softmax in fp32 and cast back, so
        # fp16 callers stay numerically stable (a no-op at fp32).
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, tgt_len, embed_dim)
        return F.linear(out, self.attention.out_proj.weight, self.attention.out_proj.bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        # expand is a zero-copy view (matches the reference); repeat would
        # allocate batch_size x 1 x dim every forward.
        probe = self.probe.expand(batch_size, -1, -1)
        hidden_state = self._map_attention(probe, hidden_state)
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SigLIP2Config):
        super().__init__()
        dim = config.vision_width
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(
            dim, config.vision_layers, config.vision_heads, config.vision_intermediate,
            config.layer_norm_eps,
        )
        self.post_layernorm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.head = SiglipMultiheadAttentionPoolingHead(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixel_values)
        x = self.encoder(x)  # full (bidirectional) attention, no mask
        x = self.post_layernorm(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Text tower
# ---------------------------------------------------------------------------


class SiglipTextEmbeddings(nn.Module):
    def __init__(self, config: SigLIP2Config):
        super().__init__()
        dim = config.text_width
        self.token_embedding = nn.Embedding(config.vocab_size, dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, dim)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.token_embedding(input_ids)
        return inputs_embeds + self.position_embedding(position_ids)


class SiglipTextModel(nn.Module):
    def __init__(self, config: SigLIP2Config):
        super().__init__()
        dim = config.text_width
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder = SiglipEncoder(
            dim, config.text_layers, config.text_heads, config.text_intermediate,
            config.layer_norm_eps,
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.head = nn.Linear(dim, config.projection_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # SigLIP's text tower is bidirectional and, in the canonical usage, does
        # not mask padding (it attends over all max_position tokens). It pools
        # the *last* token's hidden state (unlike CLIP's argmax-EOT pooling).
        x = self.embeddings(input_ids)
        x = self.encoder(x, attn_mask=None)
        x = self.final_layer_norm(x)
        pooled = x[:, -1, :]
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class SigLIP2Model(nn.Module):
    """Full SigLIP: vision + text towers sharing ``state_dict`` keys with the
    reference ``SiglipModel``.

    ``encode_image`` / ``encode_text`` return *un-normalized* pooled features
    (callers L2-normalize); ``logit_scale`` is the learned sigmoid temperature
    and ``logit_bias`` the learned per-pair bias.
    """

    def __init__(self, config: SigLIP2Config):
        super().__init__()
        self.config = config
        self.context_length = config.max_position_embeddings
        self.vision_model = SiglipVisionModel(config)
        self.text_model = SiglipTextModel(config)
        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_model(input_ids)


def build_siglip2_model(size: str) -> SigLIP2Model:
    if size not in SIGLIP2_CONFIGS:
        raise ValueError(
            f"Unknown LibreSigLIP2 size {size!r}; choose one of {sorted(SIGLIP2_CONFIGS)}."
        )
    return SigLIP2Model(SIGLIP2_CONFIGS[size])
