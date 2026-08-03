"""Native OWLv2 open-vocabulary detector.

Clean-room reimplementation of the OWLv2 (OWL-ViT v2) detection model, ported
from the Apache-2.0 ``transformers`` reference
(``transformers/models/owlv2/modeling_owlv2.py``) and the OWL-ViT paper. Module
and parameter names mirror the upstream tree so weight conversion is a
metadata-wrap and the forward is bit-identical (``max_abs_diff == 0``).

OWLv2 is a two-tower CLIP-style detector: a ViT image encoder and a CLIP text
encoder, plus lightweight per-patch detection heads (class-embedding scoring,
box MLP, objectness MLP). It shares its transformer blocks with LibreCLIP.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Owlv2Dims:
    """Architecture dimensions (read from the HF config at conversion time)."""

    # vision tower
    vision_hidden: int
    vision_layers: int
    vision_heads: int
    vision_intermediate: int
    patch_size: int
    image_size: int
    vision_act: str
    # text tower
    text_hidden: int
    text_layers: int
    text_heads: int
    text_intermediate: int
    vocab_size: int
    max_position_embeddings: int
    text_act: str
    # shared
    projection_dim: int
    layer_norm_eps: float = 1e-5

    @property
    def num_patches_side(self) -> int:
        return self.image_size // self.patch_size


# Base patch-16 ensemble (google/owlv2-base-patch16-ensemble).
OWLV2_DIMS = {
    "b16": Owlv2Dims(
        vision_hidden=768, vision_layers=12, vision_heads=12, vision_intermediate=3072,
        patch_size=16, image_size=960, vision_act="quick_gelu",
        text_hidden=512, text_layers=12, text_heads=8, text_intermediate=2048,
        vocab_size=49408, max_position_embeddings=16, text_act="quick_gelu",
        projection_dim=512, layer_norm_eps=1e-5,
    ),
    "l14": Owlv2Dims(
        vision_hidden=1024, vision_layers=24, vision_heads=16, vision_intermediate=4096,
        patch_size=14, image_size=1008, vision_act="quick_gelu",
        text_hidden=768, text_layers=12, text_heads=12, text_intermediate=3072,
        vocab_size=49408, max_position_embeddings=16, text_act="quick_gelu",
        projection_dim=768, layer_norm_eps=1e-5,
    ),
}


def _act(name: str):
    if name == "quick_gelu":
        return lambda x: x * torch.sigmoid(1.702 * x)
    if name in ("gelu", "gelu_new"):
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"Unsupported activation {name!r}")


class Owlv2Attention(nn.Module):
    """Multi-head self-attention, eager path matching transformers exactly."""

    def __init__(self, hidden: int, num_heads: int):
        super().__init__()
        self.embed_dim = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        # Opt-in fused SDPA. Default off: weights/parity_owlv2.py pins
        # objectness_logits to max_abs_diff == 0 and switches the transformers
        # reference's own SDPA off to get it (its comment measures ~1e-4 on
        # long vision sequences). Flip with
        # libreyolo.kernels.attention.set_fused_attention(model).
        self.fused_attn = False

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        shape = (b, n, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(1, 2)
        k = self.k_proj(x).view(*shape).transpose(1, 2)
        v = self.v_proj(x).view(*shape).transpose(1, 2)
        if self.fused_attn and not torch.onnx.is_in_onnx_export():
            # attn_mask is already the additive float causal+padding mask.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
                scale=self.scale,
            )
            return self.out_proj(out.transpose(1, 2).reshape(b, n, -1))
        attn = torch.matmul(q, k.transpose(2, 3)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, -1)
        return self.out_proj(out)


class Owlv2MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int, act: str):
        super().__init__()
        self.activation_fn = _act(act)
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(x)))


class Owlv2EncoderLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, intermediate: int, act: str, eps: float):
        super().__init__()
        self.self_attn = Owlv2Attention(hidden, heads)
        self.layer_norm1 = nn.LayerNorm(hidden, eps=eps)
        self.mlp = Owlv2MLP(hidden, intermediate, act)
        self.layer_norm2 = nn.LayerNorm(hidden, eps=eps)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), attn_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class Owlv2Encoder(nn.Module):
    def __init__(self, hidden, layers, heads, intermediate, act, eps):
        super().__init__()
        self.layers = nn.ModuleList(
            [Owlv2EncoderLayer(hidden, heads, intermediate, act, eps) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class Owlv2TextEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, max_pos: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden)
        self.position_embedding = nn.Embedding(max_pos, hidden)
        self.register_buffer(
            "position_ids", torch.arange(max_pos).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[-1]
        pos = self.position_ids[:, :seq_len]
        return self.token_embedding(input_ids) + self.position_embedding(pos)


class Owlv2TextTransformer(nn.Module):
    def __init__(self, d: Owlv2Dims):
        super().__init__()
        self.embeddings = Owlv2TextEmbeddings(d.vocab_size, d.text_hidden, d.max_position_embeddings)
        self.encoder = Owlv2Encoder(
            d.text_hidden, d.text_layers, d.text_heads, d.text_intermediate, d.text_act, d.layer_norm_eps
        )
        self.final_layer_norm = nn.LayerNorm(d.text_hidden, eps=d.layer_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        x = self.embeddings(input_ids)
        b, n, _ = x.shape
        min_val = torch.finfo(x.dtype).min
        # causal mask (upper triangular) combined with padding mask, additive [b,1,n,n]
        causal = torch.full((n, n), min_val, dtype=x.dtype, device=x.device).triu(1)
        mask = causal[None, None, :, :].expand(b, 1, n, n).clone()
        if attention_mask is not None:
            pad = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * min_val
            mask = mask + pad
        x = self.encoder(x, mask)
        x = self.final_layer_norm(x)
        # pooled = features at the EOS token (highest id in each sequence)
        pooled = x[torch.arange(x.shape[0], device=x.device), input_ids.to(torch.int).argmax(dim=-1)]
        return x, pooled


class Owlv2VisionEmbeddings(nn.Module):
    def __init__(self, d: Owlv2Dims):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.randn(d.vision_hidden))
        self.patch_embedding = nn.Conv2d(3, d.vision_hidden, kernel_size=d.patch_size, stride=d.patch_size, bias=False)
        self.num_positions = d.num_patches_side**2 + 1
        self.position_embedding = nn.Embedding(self.num_positions, d.vision_hidden)
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b = pixel_values.shape[0]
        patch = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        cls = self.class_embedding.expand(b, 1, -1)
        x = torch.cat([cls, patch], dim=1)
        return x + self.position_embedding(self.position_ids)


class Owlv2VisionTransformer(nn.Module):
    def __init__(self, d: Owlv2Dims):
        super().__init__()
        self.embeddings = Owlv2VisionEmbeddings(d)
        self.pre_layernorm = nn.LayerNorm(d.vision_hidden, eps=d.layer_norm_eps)
        self.encoder = Owlv2Encoder(
            d.vision_hidden, d.vision_layers, d.vision_heads, d.vision_intermediate, d.vision_act, d.layer_norm_eps
        )
        self.post_layernorm = nn.LayerNorm(d.vision_hidden, eps=d.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixel_values)
        x = self.pre_layernorm(x)
        x = self.encoder(x, None)
        return x  # last_hidden_state (post_layernorm applied by the detector head)


class Owlv2Model(nn.Module):
    """Two-tower container mirroring transformers ``Owlv2Model``."""

    def __init__(self, d: Owlv2Dims):
        super().__init__()
        self.text_model = Owlv2TextTransformer(d)
        self.vision_model = Owlv2VisionTransformer(d)
        self.visual_projection = nn.Linear(d.vision_hidden, d.projection_dim, bias=False)
        self.text_projection = nn.Linear(d.text_hidden, d.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))


class Owlv2BoxPredictionHead(nn.Module):
    def __init__(self, width: int, out_dim: int = 4):
        super().__init__()
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.dense0(x))
        x = self.gelu(self.dense1(x))
        return self.dense2(x)


class Owlv2ClassPredictionHead(nn.Module):
    def __init__(self, vision_hidden: int, text_hidden: int):
        super().__init__()
        self.query_dim = vision_hidden
        self.dense0 = nn.Linear(vision_hidden, text_hidden)
        self.logit_shift = nn.Linear(vision_hidden, 1)
        self.logit_scale = nn.Linear(vision_hidden, 1)
        self.elu = nn.ELU()

    def forward(self, image_embeds, query_embeds, query_mask):
        image_class_embeds = self.dense0(image_embeds)
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)
        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.elu(self.logit_scale(image_embeds)) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale
        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = torch.unsqueeze(query_mask, dim=-2)
            pred_logits = torch.where(query_mask == 0, torch.finfo(pred_logits.dtype).min, pred_logits)
            pred_logits = pred_logits.to(torch.float32)
        return pred_logits, image_class_embeds


class Owlv2DetectionModel(nn.Module):
    """Native equivalent of transformers ``Owlv2ForObjectDetection``.

    forward(input_ids, pixel_values, attention_mask) ->
        dict(logits, pred_boxes, objectness_logits)
    with pred_boxes in normalized cxcywh (sigmoid-activated), matching upstream.
    """

    def __init__(self, dims: Owlv2Dims):
        super().__init__()
        self.dims = dims
        self.owlv2 = Owlv2Model(dims)
        self.class_head = Owlv2ClassPredictionHead(dims.vision_hidden, dims.text_hidden)
        self.box_head = Owlv2BoxPredictionHead(dims.vision_hidden, out_dim=4)
        self.objectness_head = Owlv2BoxPredictionHead(dims.vision_hidden, out_dim=1)
        self.layer_norm = nn.LayerNorm(dims.vision_hidden, eps=dims.layer_norm_eps)
        side = dims.num_patches_side
        self.num_patches_height = side
        self.num_patches_width = side
        self.register_buffer("box_bias", self.compute_box_bias(side, side), persistent=False)

    @staticmethod
    def normalize_grid_corner_coordinates(h: int, w: int) -> torch.Tensor:
        x = torch.arange(1, w + 1, dtype=torch.float32)
        y = torch.arange(1, h + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        coords = torch.stack((xx, yy), dim=-1)
        coords[..., 0] /= w
        coords[..., 1] /= h
        return coords.view(-1, 2)

    def compute_box_bias(self, h: int, w: int) -> torch.Tensor:
        coords = torch.clip(self.normalize_grid_corner_coordinates(h, w), 0.0, 1.0)
        coord_bias = torch.log(coords + 1e-4) - torch.log1p(-coords + 1e-4)
        size = torch.full_like(coord_bias, 1.0)
        size[..., 0] /= w
        size[..., 1] /= h
        size_bias = torch.log(size + 1e-4) - torch.log1p(-size + 1e-4)
        return torch.cat([coord_bias, size_bias], dim=-1)

    def _image_embedder(self, pixel_values: torch.Tensor) -> torch.Tensor:
        last_hidden_state = self.owlv2.vision_model(pixel_values)
        image_embeds = self.owlv2.vision_model.post_layernorm(last_hidden_state)
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], image_embeds[:, :-1].shape)
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)
        b, _, d = image_embeds.shape
        return image_embeds.reshape(b, self.num_patches_height, self.num_patches_width, d)

    def _text_embedder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        _, pooled = self.owlv2.text_model(input_ids, attention_mask)
        text_embeds = self.owlv2.text_projection(pooled)
        # Owlv2Model returns L2-normalized text_embeds (no epsilon); replicate for parity.
        return text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True)

    def forward(self, input_ids, pixel_values, attention_mask=None):
        feature_map = self._image_embedder(pixel_values)
        b, h, w, d = feature_map.shape
        image_feats = feature_map.reshape(b, h * w, d)

        text_embeds = self._text_embedder(input_ids, attention_mask)
        max_text_queries = input_ids.shape[0] // b
        query_embeds = text_embeds.reshape(b, max_text_queries, text_embeds.shape[-1])

        ids = input_ids.reshape(b, max_text_queries, input_ids.shape[-1])
        query_mask = ids[..., 0] > 0

        pred_logits, class_embeds = self.class_head(image_feats, query_embeds, query_mask)
        objectness_logits = self.objectness_head(image_feats.detach())[..., 0]
        pred_boxes = self.box_head(image_feats) + self.box_bias.to(feature_map.device)
        pred_boxes = torch.sigmoid(pred_boxes)

        return {
            "logits": pred_logits,
            "pred_boxes": pred_boxes,
            "objectness_logits": objectness_logits,
            "class_embeds": class_embeds,
        }


__all__ = ["Owlv2Dims", "OWLV2_DIMS", "Owlv2DetectionModel"]
