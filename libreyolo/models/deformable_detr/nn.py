"""Initial Deformable DETR architecture scaffold.

The parity implementation replaces this scaffold in the architecture commit.
The temporary modules deliberately expose the native upstream key hierarchy so
the registry and factory contracts can be developed and tested independently.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


DEFORMABLE_DETR_CONFIGS = {
    "r50ss": {"num_feature_levels": 1, "with_box_refine": False, "two_stage": False},
    "r50ssdc5": {"num_feature_levels": 1, "with_box_refine": False, "two_stage": False},
    "r50": {"num_feature_levels": 4, "with_box_refine": False, "two_stage": False},
    "r50refine": {"num_feature_levels": 4, "with_box_refine": True, "two_stage": False},
    "r50twostage": {
        "num_feature_levels": 4,
        "with_box_refine": True,
        "two_stage": True,
    },
}


class _BackboneBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)


class _BackboneStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = _BackboneBody()


class _DeformableAttentionScaffold(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.sampling_offsets = nn.Linear(hidden_dim, hidden_dim)


class _EncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_attn = _DeformableAttentionScaffold(hidden_dim)


class _DecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.cross_attn = _DeformableAttentionScaffold(hidden_dim)


class _Encoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([_EncoderLayer(hidden_dim)])


class _Decoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderLayer(hidden_dim)])


class _Transformer(nn.Module):
    def __init__(self, hidden_dim: int, levels: int, two_stage: bool):
        super().__init__()
        self.encoder = _Encoder(hidden_dim)
        self.decoder = _Decoder(hidden_dim)
        self.level_embed = nn.Parameter(torch.empty(levels, hidden_dim))
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)


class _BBoxMLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, 4),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


class LibreDeformableDETRModel(nn.Module):
    """Small structural placeholder used by the family/factory skeleton."""

    def __init__(self, size: str, nc: int = 91):
        super().__init__()
        config = DEFORMABLE_DETR_CONFIGS[size]
        hidden_dim = 256
        self.num_queries = 300
        self.backbone = nn.Sequential(_BackboneStage())
        self.transformer = _Transformer(
            hidden_dim, config["num_feature_levels"], config["two_stage"]
        )
        self.input_proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(64, hidden_dim, kernel_size=1))]
        )
        if not config["two_stage"]:
            self.query_embed = nn.Embedding(self.num_queries, hidden_dim * 2)

        class_head = nn.Linear(hidden_dim, nc)
        box_head = _BBoxMLP(hidden_dim)
        prediction_layers = 7 if config["two_stage"] else 6
        if config["with_box_refine"]:
            self.class_embed = nn.ModuleList(
                [copy.deepcopy(class_head) for _ in range(prediction_layers)]
            )
            self.bbox_embed = nn.ModuleList(
                [copy.deepcopy(box_head) for _ in range(prediction_layers)]
            )
        else:
            self.class_embed = nn.ModuleList(
                [class_head for _ in range(prediction_layers)]
            )
            self.bbox_embed = nn.ModuleList(
                [box_head for _ in range(prediction_layers)]
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = x.shape[0]
        hidden = x.mean(dim=(1, 2, 3), keepdim=False).view(batch, 1, 1)
        hidden = hidden.expand(batch, self.num_queries, 256)
        return {
            "pred_logits": self.class_embed[-1](hidden),
            "pred_boxes": self.bbox_embed[-1](hidden).sigmoid(),
        }


__all__ = ["DEFORMABLE_DETR_CONFIGS", "LibreDeformableDETRModel"]
