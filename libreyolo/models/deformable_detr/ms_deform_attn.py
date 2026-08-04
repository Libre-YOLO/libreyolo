# SPDX-License-Identifier: Apache-2.0
# Ported from https://github.com/fundamentalvision/Deformable-DETR
# commit 11169a60c33333af00a4849f1808023eba96a931.
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# The upstream module was modified from Deformable-Convolution-V2-PyTorch.
"""Portable multi-scale deformable attention implemented with grid_sample."""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.init import constant_, xavier_uniform_

from ...kernels.attention.ms_deform_attn import maybe_ms_deform_attn


def _is_power_of_2(value: int) -> bool:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid input for _is_power_of_2: {value!r}")
    return value != 0 and value & (value - 1) == 0


def ms_deform_attn_core_pytorch(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Portable deformable attention core used on every device/backend.

    When the optional accelerated ``ms_deform_attn`` kernel slot resolves
    (see ``libreyolo/kernels/attention/ms_deform_attn.py``) it takes over;
    the grid_sample path below stays the default and the export path.
    """
    accelerated = maybe_ms_deform_attn(
        value, value_spatial_shapes, sampling_locations, attention_weights
    )
    if accelerated is not None:
        return accelerated
    batch, _, heads, channels = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    split_sizes = [height * width for height, width in value_spatial_shapes]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampled_values = []
    for level, (height, width) in enumerate(value_spatial_shapes):
        value_level = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * heads, channels, height, width)
        )
        sampling_grid = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled_values.append(
            F.grid_sample(
                value_level,
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch * heads, 1, queries, levels * points
    )
    output = (torch.stack(sampled_values, dim=-2).flatten(-2) * attention_weights).sum(
        -1
    )
    output = output.view(batch, heads * channels, queries)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention without compiled custom operators."""

    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, got {d_model} and {n_heads}"
            )
        if not _is_power_of_2(d_model // n_heads):
            warnings.warn(
                "A power-of-two dimension per attention head is recommended.",
                stacklevel=2,
            )

        self.im2col_step = 64  # retained for checkpoint/API compatibility
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2
        )
        grid_init = grid_init.repeat(1, self.n_levels, self.n_points, 1)
        for point in range(self.n_points):
            grid_init[:, :, point, :] *= point + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: Tensor,
        input_level_start_index: Tensor,
        input_padding_mask: Tensor | None = None,
    ) -> Tensor:
        del input_level_start_index  # core derives splits directly from shapes
        batch, query_length, _ = query.shape
        input_batch, input_length, _ = input_flatten.shape
        if batch != input_batch:
            raise ValueError("Query and value batches differ")

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(
            batch,
            input_length,
            self.n_heads,
            self.d_model // self.n_heads,
        )
        sampling_offsets = self.sampling_offsets(query).view(
            batch,
            query_length,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        )
        attention_weights = self.attention_weights(query).view(
            batch,
            query_length,
            self.n_heads,
            self.n_levels * self.n_points,
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch,
            query_length,
            self.n_heads,
            self.n_levels,
            self.n_points,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = reference_points[:, :, None, :, None, :] + (
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + (
                sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dimension of reference_points must be 2 or 4, got "
                f"{reference_points.shape[-1]}"
            )

        output = ms_deform_attn_core_pytorch(
            value,
            input_spatial_shapes,
            sampling_locations,
            attention_weights,
        )
        return self.output_proj(output)


__all__ = ["MSDeformAttn", "ms_deform_attn_core_pytorch"]
