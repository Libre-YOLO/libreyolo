"""Multi-scale deformable attention helpers.

Ported from D-FINE (https://github.com/Peterande/D-FINE).
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.

Uses ``F.grid_sample`` rather than a CUDA deformable-attention kernel, which
keeps ONNX export portable (opset >= 16).
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias according to a given probability value."""
    return float(-math.log((1 - prior_prob) / prior_prob))


def get_activation(act, inpace: bool = True):
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act

    act = act.lower()
    if act in ("silu", "swish"):
        m = nn.SiLU()
    elif act == "relu":
        m = nn.ReLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    elif act == "gelu":
        m = nn.GELU()
    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError(f"Unknown activation: {act}")

    if hasattr(m, "inplace"):
        m.inplace = inpace
    return m


def _grid_sample_bilinear_manual(input, grid, padding_mode="zeros", align_corners=False):
    """ONNX/TRT-safe bilinear grid_sample: gather-based, emits NO GridSample op.
    Numerically matches F.grid_sample(mode='bilinear'); sidesteps the TensorRT
    GridSample kernel that miscomputes boundary samples for deformable attention."""
    N, C, H, W = input.shape
    Hg, Wg = grid.shape[1], grid.shape[2]
    if align_corners:
        ix = (grid[..., 0] + 1) * (W - 1) / 2
        iy = (grid[..., 1] + 1) * (H - 1) / 2
    else:
        ix = (grid[..., 0] + 1) * W / 2 - 0.5
        iy = (grid[..., 1] + 1) * H / 2 - 0.5
    ix0 = ix.floor().long(); iy0 = iy.floor().long()
    ix1 = ix0 + 1; iy1 = iy0 + 1
    wx1 = (ix - ix0.float()).to(input.dtype).unsqueeze(1)
    wy1 = (iy - iy0.float()).to(input.dtype).unsqueeze(1)
    one = wx1.new_tensor(1.0); wx0 = one - wx1; wy0 = one - wy1
    if padding_mode == "border":
        ix0 = ix0.clamp(0, W - 1); iy0 = iy0.clamp(0, H - 1)
        ix1 = ix1.clamp(0, W - 1); iy1 = iy1.clamp(0, H - 1)
    else:
        in_x0 = (ix0 >= 0) & (ix0 < W); in_x1 = (ix1 >= 0) & (ix1 < W)
        in_y0 = (iy0 >= 0) & (iy0 < H); in_y1 = (iy1 >= 0) & (iy1 < H)
        ix0 = ix0.clamp(0, W - 1); iy0 = iy0.clamp(0, H - 1)
        ix1 = ix1.clamp(0, W - 1); iy1 = iy1.clamp(0, H - 1)
    flat = input.flatten(2)
    def _g(iy_, ix_):
        idx = (iy_ * W + ix_).flatten(1).unsqueeze(1).expand(N, C, -1)
        return flat.gather(2, idx).view(N, C, Hg, Wg)
    v00 = _g(iy0, ix0); v10 = _g(iy0, ix1); v01 = _g(iy1, ix0); v11 = _g(iy1, ix1)
    if padding_mode == "zeros":
        v00 = v00 * (in_x0 & in_y0).unsqueeze(1); v10 = v10 * (in_x1 & in_y0).unsqueeze(1)
        v01 = v01 * (in_x0 & in_y1).unsqueeze(1); v11 = v11 * (in_x1 & in_y1).unsqueeze(1)
    return wx0 * wy0 * v00 + wx1 * wy0 * v10 + wx0 * wy1 * v01 + wx1 * wy1 * v11


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method: str = "default",
):
    """Multi-scale deformable attention aggregator (grid_sample variant).

    Args:
        value: list of ``(bs, n_head, c, H_l * W_l)`` tensors, one per level.
        value_spatial_shapes: list of ``(H_l, W_l)`` per level.
        sampling_locations: ``(bs, Len_q, n_head, sum(num_points), 2)``.
        attention_weights: ``(bs, Len_q, n_head, sum(num_points))``.
        num_points_list: list of sampling points per level.
        method: ``"default"`` (bilinear) or ``"discrete"`` (nearest integer).

    Returns: ``(bs, Len_q, n_head * c)``.
    """
    bs, n_head, c, _ = value[0].shape
    _, Len_q, _, _, _ = sampling_locations.shape

    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    else:
        raise ValueError(f"Unknown method: {method}")

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l = sampling_locations_list[level]

        if method == "default":
            if torch.onnx.is_in_onnx_export():
                sampling_value_l = _grid_sample_bilinear_manual(
                    value_l, sampling_grid_l, padding_mode="zeros", align_corners=False
                )
            else:
                sampling_value_l = F.grid_sample(
                    value_l,
                    sampling_grid_l,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
        else:  # discrete
            sampling_coord = (
                sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5
            ).to(torch.int64)
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(
                bs * n_head, Len_q * num_points_list[level], 2
            )
            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value_l.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l = value_l[
                s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]
            ]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                bs * n_head, c, Len_q, num_points_list[level]
            )

        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        bs * n_head, 1, Len_q, sum(num_points_list)
    )
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)
