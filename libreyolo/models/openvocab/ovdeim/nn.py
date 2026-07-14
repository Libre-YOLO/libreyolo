"""Top-level OV-DEIM model wiring for LibreYOLO.

The detector (backbone + encoder + decoder + text_adapter) mirrors upstream
OV-DEIM (https://github.com/wleilei/OV-DEIM, Apache-2.0) attribute names so
released checkpoints convert with a plain key strip. The online text tower is
the MobileCLIP-B(LT) text encoder, which is a standard CLIP text transformer;
it reuses LibreCLIP's open_clip-compatible modules and loads Apple's released
text-tower weights unchanged.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.nn import init

from ...clip.nn import Transformer as CLIPTextTransformer
from .decoder import OVDEIMDecoder, RMSNorm
from .encoder import HybridEncoder


# Per-size configuration transcribed from upstream config/dinov3_ori/{dinov3_s,m,l}.py.
OVDEIM_SIZE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "s": {
        "backbone": {
            "name": "vit_tiny",
            "embed_dim": 192,
            "interaction_indexes": [3, 7, 11],
            "num_heads": 3,
            "conv_inplane": 16,
        },
        "encoder": {
            "in_channels": [192, 192, 192],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 192,
            "use_encoder_idx": [2],
            "num_encoder_layers": 1,
            "nhead": 8,
            "dim_feedforward": 512,
            "dropout": 0.0,
            "enc_act": "gelu",
            "expansion": 0.34,
            "depth_mult": 0.67,
            "act": "silu",
        },
        "decoder": {
            "feat_channels": [192, 192, 192],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 192,
            "dim_feedforward": 512,
            "num_levels": 3,
            "num_layers": 4,
            "num_queries": 300,
            "learn_query_content": False,
            "activation": "silu",
            "eval_idx": -1,
            "num_points": [4, 6, 4],
            "cross_attn_method": "default",
            "query_select_method": "default",
            "num_enc_queries": 0,
        },
        "model": {"img_dim": 192, "text_dim": 512},
    },
    "m": {
        "backbone": {
            "name": "vit_tinyplus",
            "embed_dim": 256,
            "interaction_indexes": [3, 7, 11],
            "num_heads": 4,
            "conv_inplane": 16,
        },
        "encoder": {
            "in_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "use_encoder_idx": [2],
            "num_encoder_layers": 1,
            "nhead": 8,
            "dim_feedforward": 512,
            "dropout": 0.0,
            "enc_act": "gelu",
            "expansion": 0.67,
            "depth_mult": 1.0,
            "act": "silu",
        },
        "decoder": {
            "feat_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "dim_feedforward": 512,
            "num_levels": 3,
            "num_layers": 4,
            "num_queries": 300,
            "learn_query_content": False,
            "activation": "silu",
            "eval_idx": -1,
            "num_points": [4, 6, 4],
            "cross_attn_method": "default",
            "query_select_method": "default",
            "num_enc_queries": 0,
        },
        "model": {"img_dim": 256, "text_dim": 512},
    },
    "l": {
        "backbone": {
            "name": "dinov3_vits16",
            "embed_dim": 224,
            "hidden_dim": 224,
            "conv_inplane": 32,
            "interaction_indexes": [5, 8, 11],
            "num_heads": None,
        },
        "encoder": {
            "in_channels": [224, 224, 224],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 224,
            "use_encoder_idx": [2],
            "num_encoder_layers": 1,
            "nhead": 8,
            "dim_feedforward": 896,
            "dropout": 0.0,
            "enc_act": "gelu",
            "expansion": 1.0,
            "depth_mult": 1.0,
            "act": "silu",
        },
        "decoder": {
            "feat_channels": [224, 224, 224],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 224,
            "dim_feedforward": 1792,
            "num_levels": 3,
            "num_layers": 4,
            "num_queries": 300,
            "learn_query_content": False,
            "activation": "silu",
            "eval_idx": -1,
            "num_points": [4, 6, 4],
            "cross_attn_method": "default",
            "query_select_method": "default",
            "num_enc_queries": 0,
        },
        "model": {"img_dim": 224, "text_dim": 512},
    },
}


class GatedFFNBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * dim * 2)
        self.w3 = nn.Linear(dim * 2, dim)
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x1 = self.norm(x)
        x12 = self.w12(x1)
        x1, x2 = x12.chunk(2, dim=-1)
        gated = nn.functional.silu(x1) * x2
        return x + self.w3(gated)


class TextAdapter(nn.Module):
    """Learned adapter from CLIP text space into the decoder's visual space."""

    def __init__(self, text_dim: int, img_dim: int, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([GatedFFNBlock(text_dim) for _ in range(num_layers)])
        self.proj_out = nn.Linear(text_dim, img_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.layers:
            init.xavier_uniform_(layer.w12.weight)
            init.constant_(layer.w12.bias, 0)
            init.xavier_uniform_(layer.w3.weight)
            init.constant_(layer.w3.bias, 0)
        init.xavier_uniform_(self.proj_out.weight)
        init.constant_(self.proj_out.bias, 0)

    def forward(self, text_feats):
        x = text_feats
        for layer in self.layers:
            x = layer(x)
        return self.proj_out(x)


class MobileCLIPTextEncoder(nn.Module):
    """MobileCLIP-B(LT) text tower: a standard CLIP-B text transformer.

    ``state_dict`` keys mirror open_clip's ``CustomTextCLIP`` text tower with
    the ``text.`` prefix stripped, so Apple's released weights load unchanged.
    """

    CONTEXT_LENGTH = 77
    VOCAB_SIZE = 49408
    WIDTH = 512
    HEADS = 8
    LAYERS = 12
    EMBED_DIM = 512

    def __init__(self):
        super().__init__()
        self.transformer = CLIPTextTransformer(self.WIDTH, self.LAYERS, self.HEADS)
        self.token_embedding = nn.Embedding(self.VOCAB_SIZE, self.WIDTH)
        self.positional_embedding = nn.Parameter(torch.empty(self.CONTEXT_LENGTH, self.WIDTH))
        self.ln_final = nn.LayerNorm(self.WIDTH)
        self.text_projection = nn.Parameter(torch.empty(self.WIDTH, self.EMBED_DIM))
        self.register_buffer("attn_mask", self._build_causal_mask(), persistent=False)

        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.WIDTH**-0.5)

    def _build_causal_mask(self) -> torch.Tensor:
        mask = torch.empty(self.CONTEXT_LENGTH, self.CONTEXT_LENGTH)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        # EOT pooling: the end-of-text token has the highest id in each row.
        pooled = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
        return pooled @ self.text_projection


class LibreOVDEIMModel(nn.Module):
    """OV-DEIM detector + online MobileCLIP text tower.

    ``forward(images, text_feats)`` runs the detector against already-encoded
    (L2-normalized) text embeddings; ``encode_texts`` produces them from token
    id tensors. Submodule names (``backbone``/``encoder``/``decoder``/
    ``text_adapter``) match upstream checkpoints; the text tower lives under
    ``text_encoder``.
    """

    def __init__(self, size: str):
        super().__init__()
        if size not in OVDEIM_SIZE_CONFIGS:
            raise ValueError(f"Unknown OV-DEIM size {size!r}; choose one of {sorted(OVDEIM_SIZE_CONFIGS)}.")
        # Imported lazily: the deimv2 backbone package logs at import time.
        from ...deimv2.engine.backbone.dinov3_adapter import DINOv3STAs

        cfg = copy.deepcopy(OVDEIM_SIZE_CONFIGS[size])
        self.size = size
        self.backbone = DINOv3STAs(**cfg["backbone"])
        self.encoder = HybridEncoder(**cfg["encoder"])
        self.decoder = OVDEIMDecoder(num_denoising=0, **cfg["decoder"])
        self.text_adapter = TextAdapter(
            cfg["model"]["text_dim"], cfg["model"]["img_dim"], num_layers=1
        )
        self.text_encoder = MobileCLIPTextEncoder()

    def encode_texts(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode tokenized prompts into L2-normalized CLIP text embeddings."""
        feats = self.text_encoder(token_ids)
        return nn.functional.normalize(feats, dim=-1)

    def forward(self, x: torch.Tensor, text_feats: torch.Tensor):
        feats = self.backbone(x)
        adapted = self.text_adapter(text_feats)
        feats = self.encoder(feats)
        return self.decoder(feats, adapted)
