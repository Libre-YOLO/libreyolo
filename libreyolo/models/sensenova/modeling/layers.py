# SPDX-License-Identifier: MIT
#
# Clean-source implementations of the small embedding/connector layers the
# Bagel-MoT architecture expects. The corresponding file in the upstream
# SenseNova-Vision repository (modeling/bagel/modeling_utils.py) is marked
# CC BY-NC 4.0 because it derives from facebookresearch/DiT, so it is NOT
# ported into LibreYOLO. This file re-derives the same standard components
# from their original permissive sources instead:
#
# - The 2D sine-cosine position table follows the Apache-2.0 implementation
#   in Hugging Face transformers (models/vit_mae/modeling_vit_mae.py,
#   `get_2d_sincos_pos_embed`), which carries Meta's copyright under
#   Apache-2.0 and predates DiT's non-commercial fork of the same routine.
# - The sinusoidal timestep embedding follows openai/guided-diffusion
#   (guided_diffusion/nn.py, `timestep_embedding`, MIT), the source DiT
#   itself credits.
# - Module structure and parameter names (mlp.0/mlp.2, fc1/fc2, pos_embed)
#   are fixed by the released SenseNova-Vision-7B-MoT checkpoint's state
#   dict and carry no expression beyond that interface.

import math

import torch
from torch import nn

from transformers.activations import ACT2FN


def sincos_position_embedding_1d(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    """Sine-cosine table for one axis: [sin(p*w), cos(p*w)] per position.

    Frequencies follow the transformer convention 1/10000^(2i/d). Returns a
    float32 tensor of shape (len(positions), embed_dim).
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = torch.arange(embed_dim // 2, dtype=torch.float64) / (embed_dim / 2.0)
    omega = 1.0 / torch.pow(torch.tensor(10000.0, dtype=torch.float64), omega)
    out = positions.reshape(-1).to(torch.float64)[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def sincos_position_embedding_2d(embed_dim: int, grid_size: int) -> torch.Tensor:
    """2D sine-cosine position table over a square grid, flattened row-major.

    Position p = row * grid_size + col maps to the concatenation of the
    1D table over the column index (first half) and the row index (second
    half), matching the axis order of the transformers ViTMAE reference so
    the table is interchangeable with released checkpoints.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    rows = torch.arange(grid_size, dtype=torch.float64)
    cols = torch.arange(grid_size, dtype=torch.float64)
    grid_rows = rows[:, None].expand(grid_size, grid_size)
    grid_cols = cols[None, :].expand(grid_size, grid_size)
    emb_cols = sincos_position_embedding_1d(embed_dim // 2, grid_cols)
    emb_rows = sincos_position_embedding_1d(embed_dim // 2, grid_rows)
    return torch.cat([emb_cols, emb_rows], dim=1)


class PositionEmbedding(nn.Module):
    """Frozen sine-cosine position table indexed by flattened position ids."""

    def __init__(self, max_num_patch_per_side: int, hidden_size: int):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side**2, hidden_size),
            requires_grad=False,
        )
        self._init_weights()

    def _init_weights(self):
        table = sincos_position_embedding_2d(self.hidden_size, self.max_num_patch_per_side)
        self.pos_embed.data.copy_(table)

    def forward(self, position_ids):
        return self.pos_embed[position_ids]


class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into hidden-size vectors."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        # Layout (Linear, SiLU, Linear) is fixed by the checkpoint keys
        # time_embedder.mlp.0.* / time_embedder.mlp.2.*.
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Sinusoidal timestep embedding: [cos(t*f), sin(t*f)] per frequency.

        Frequencies decay geometrically from 1 to 1/max_period, following the
        guided-diffusion reference; cos comes first, and an odd `dim` gets one
        zero pad column.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class MLPconnector(nn.Module):
    """Two-layer projector between the vision tower and the language model."""

    def __init__(self, in_dim: int, out_dim: int, hidden_act: str):
        super().__init__()
        # fc1/fc2 names are fixed by the checkpoint keys connector.fc1.* /
        # connector.fc2.*.
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
