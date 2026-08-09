"""Native BERT text encoder (parity with transformers ``BertModel``).

Clean-room port of the Apache-2.0 BERT encoder, used as Grounding DINO's text
backbone. Mirrors the transformers module/param tree so the ``state_dict`` loads
directly and the forward is bit-identical. Eager attention (deterministic).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...kernels.attention.sdpa import manual_attention_required


class BertEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_size
        self.word_embeddings = nn.Embedding(cfg.vocab_size, h, padding_idx=getattr(cfg, "pad_token_id", 0))
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, h)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, h)
        self.LayerNorm = nn.LayerNorm(h, eps=cfg.layer_norm_eps)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        n = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(n, device=input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        emb = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.LayerNorm(emb)


class BertSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def _shape(self, x):
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x, attn_mask):
        q, k, v = self._shape(self.query(x)), self._shape(self.key(x)), self._shape(self.value(x))
        if manual_attention_required():
            # Graph capture (ONNX, and the jit.trace-based TorchScript /
            # CoreML / NCNN exporters) must see the primitive-op equation;
            # eager inference keeps the fused kernels below.
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores = scores + attn_mask
            probs = scores.softmax(-1)
            ctx = probs @ v
        else:
            mask = attn_mask
            if mask is not None and mask.dtype == torch.bool:
                # Grounding DINO's block-diagonal mask arrives boolean and the
                # eager path above ADDS it (True -> +1, False -> +0), a weak
                # bias rather than a hard -inf mask. SDPA reads a bool mask as
                # "allowed", so cast to keep the additive semantics.
                mask = mask.to(q.dtype)
            ctx = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        b, n = x.shape[0], x.shape[1]
        return ctx.view(b, n, -1)


class BertSelfOutput(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, hidden, residual):
        return self.LayerNorm(self.dense(hidden) + residual)


class BertAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self = BertSelfAttention(cfg)
        self.output = BertSelfOutput(cfg)

    def forward(self, x, attn_mask):
        return self.output(self.self(x, attn_mask), x)


class BertLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = BertAttention(cfg)
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.output = nn.Module()
        self.output.dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.output.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, x, attn_mask):
        x = self.attention(x, attn_mask)
        h = F.gelu(self.intermediate.dense(x))
        return self.output.LayerNorm(self.output.dense(h) + x)


class BertModel(nn.Module):
    """Encoder-only BERT (no pooler) matching Grounding DINO's text backbone."""

    def __init__(self, cfg):
        super().__init__()
        self.embeddings = BertEmbeddings(cfg)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([BertLayer(cfg) for _ in range(cfg.num_hidden_layers)])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        x = self.embeddings(input_ids, token_type_ids, position_ids)
        ext = None
        if attention_mask is not None:
            am = attention_mask.to(x.dtype)
            if am.dim() == 2:  # [B, S] padding mask -> additive [B,1,1,S]
                ext = (1.0 - am[:, None, None, :]) * torch.finfo(x.dtype).min
            else:
                # Pre-built 4D bool mask (e.g. Grounding DINO's block-diagonal mask):
                # transformers' create_bidirectional_mask leaves it boolean and the eager
                # path ADDS it directly (True -> +1, False -> +0), a weak bias rather than
                # a hard -inf mask. Replicate exactly for parity.
                ext = am
        for layer in self.encoder.layer:
            x = layer(x, ext)
        return x


__all__ = ["BertModel"]
