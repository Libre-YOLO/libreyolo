"""Focused RF-DETR graph-capture regressions for Core ML conversion."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from libreyolo.models.rfdetr.tensors import _bilinear_grid_sample
from libreyolo.models.rfdetr.transformer import (
    MSDeformAttn,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
)

pytestmark = pytest.mark.unit


class _ProposalGrid(nn.Module):
    def forward(self, memory: torch.Tensor):
        return gen_encoder_output_proposals(
            memory,
            spatial_shapes=((2, 3), (1, 2)),
            unsigmoid=False,
        )


class _SineEmbedding(nn.Module):
    def forward(self, positions: torch.Tensor):
        return gen_sineembed_for_position(positions, dim=8)


class _DeformAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MSDeformAttn(
            d_model=16,
            n_levels=2,
            n_heads=4,
            n_points=3,
        ).eval()
        self.attention.export()
        self.register_buffer(
            "spatial_shapes",
            torch.tensor(((2, 3), (1, 2)), dtype=torch.long),
        )
        self.register_buffer(
            "level_starts",
            torch.tensor((0, 6), dtype=torch.long),
        )

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
    ) -> torch.Tensor:
        return self.attention(
            query,
            reference_points,
            input_flatten,
            self.spatial_shapes,
            self.level_starts,
            input_spatial_shapes_hw=((2, 3), (1, 2)),
        )


def _reference_sine_embedding(positions: torch.Tensor) -> torch.Tensor:
    scale = 2 * torch.pi
    dim_t = positions.new_ones((8,)).cumsum(0) - 1
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / 8)
    values = []
    for coordinate in range(4):
        embedded = positions[:, :, coordinate, None] * scale / dim_t
        values.append(
            torch.stack(
                (
                    embedded[:, :, 0::2].sin(),
                    embedded[:, :, 1::2].cos(),
                ),
                dim=3,
            ).flatten(2)
        )
    return torch.cat((values[1], values[0], values[2], values[3]), dim=2)


def _reference_deform_attention(
    module: MSDeformAttn,
    query: torch.Tensor,
    reference_points: torch.Tensor,
    input_flatten: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the original rank-six deformable-attention layout."""
    batch_size, len_query, _ = query.shape
    _, len_input, _ = input_flatten.shape
    shapes = ((2, 3), (1, 2))

    value = module.value_proj(input_flatten)
    sampling_offsets = module.sampling_offsets(query).view(
        batch_size,
        len_query,
        module.n_heads,
        module.n_levels,
        module.n_points,
        2,
    )
    attention_weights = module.attention_weights(query).view(
        batch_size,
        len_query,
        module.n_heads,
        module.n_levels * module.n_points,
    )
    spatial_shapes = torch.tensor(
        shapes,
        dtype=torch.long,
        device=query.device,
    )
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            (spatial_shapes[:, 1], spatial_shapes[:, 0]),
            dim=-1,
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
    else:
        sampling_locations = (
            reference_points[:, :, None, :, None, :2]
            + sampling_offsets
            / module.n_points
            * reference_points[:, :, None, :, None, 2:]
            * 0.5
        )
    attention_weights = torch.softmax(attention_weights, dim=-1)

    head_dim = module.d_model // module.n_heads
    value = (
        value.transpose(1, 2)
        .contiguous()
        .view(batch_size, module.n_heads, head_dim, len_input)
    )
    value_list = value.split(
        [height * width for height, width in shapes],
        dim=3,
    )
    sampling_grids = 2 * sampling_locations - 1
    sampled = []
    for level_index, (height, width) in enumerate(shapes):
        level_value = value_list[level_index].view(
            batch_size * module.n_heads,
            head_dim,
            height,
            width,
        )
        level_grid = sampling_grids[:, :, :, level_index].transpose(1, 2).flatten(0, 1)
        sampled.append(
            _bilinear_grid_sample(
                level_value,
                level_grid,
                padding_mode="zeros",
                align_corners=False,
            )
        )
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * module.n_heads,
        1,
        len_query,
        module.n_levels * module.n_points,
    )
    sampled_values = torch.stack(sampled, dim=-2).flatten(-2)
    output = (
        (sampled_values * attention_weights)
        .sum(-1)
        .view(
            batch_size,
            module.n_heads * head_dim,
            len_query,
        )
    )
    return module.output_proj(output.transpose(1, 2).contiguous())


def _reference_points(
    batch_size: int,
    len_query: int,
    coordinate_count: int,
) -> torch.Tensor:
    points = torch.linspace(
        0.1,
        0.9,
        batch_size * len_query * 2 * coordinate_count,
    ).reshape(batch_size, len_query, 2, coordinate_count)
    if coordinate_count == 4:
        points[..., 2:] = points[..., 2:] * 0.4 + 0.1
    return points


def test_proposal_grid_is_exact_without_tensor_new_ones(monkeypatch):
    memory = torch.linspace(-1.0, 1.0, 2 * 8 * 4).reshape(2, 8, 4)

    def reject_new_ones(*args, **kwargs):
        raise AssertionError("RF-DETR proposal generation must not use new_ones")

    monkeypatch.setattr(torch.Tensor, "new_ones", reject_new_ones)
    output_memory, proposals = _ProposalGrid()(memory)

    level0 = torch.tensor(
        [
            [1 / 6, 1 / 4, 0.05, 0.05],
            [1 / 2, 1 / 4, 0.05, 0.05],
            [5 / 6, 1 / 4, 0.05, 0.05],
            [1 / 6, 3 / 4, 0.05, 0.05],
            [1 / 2, 3 / 4, 0.05, 0.05],
            [5 / 6, 3 / 4, 0.05, 0.05],
        ],
        dtype=memory.dtype,
    )
    level1 = torch.tensor(
        [
            [1 / 4, 1 / 2, 0.10, 0.10],
            [3 / 4, 1 / 2, 0.10, 0.10],
        ],
        dtype=memory.dtype,
    )
    expected = torch.cat((level0, level1)).unsqueeze(0).expand(2, -1, -1)

    torch.testing.assert_close(output_memory, memory, rtol=0.0, atol=0.0)
    torch.testing.assert_close(proposals, expected, rtol=0.0, atol=0.0)


def test_proposal_grid_trace_has_exact_parity_and_no_new_ones():
    module = _ProposalGrid().eval()
    first = torch.linspace(-2.0, 2.0, 2 * 8 * 4).reshape(2, 8, 4)
    second = torch.linspace(3.0, -3.0, 3 * 8 * 4).reshape(3, 8, 4)

    traced = torch.jit.trace(
        module,
        first,
        check_trace=True,
        check_inputs=[(second,)],
    )

    expected = module(second)
    actual = traced(second)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert "aten::new_ones" not in str(traced.inlined_graph)


def test_sine_embedding_trace_matches_original_new_ones_math():
    module = _SineEmbedding().eval()
    first = torch.linspace(0.05, 0.95, 2 * 3 * 4).reshape(2, 3, 4)
    second = torch.linspace(0.9, 0.1, 5 * 2 * 4).reshape(5, 2, 4)

    torch.testing.assert_close(
        module(first),
        _reference_sine_embedding(first),
        rtol=0.0,
        atol=0.0,
    )
    traced = torch.jit.trace(
        module,
        first,
        check_trace=True,
        check_inputs=[(second,)],
    )
    torch.testing.assert_close(
        traced(second),
        _reference_sine_embedding(second),
        rtol=0.0,
        atol=0.0,
    )
    assert "aten::new_ones" not in str(traced.inlined_graph)


@pytest.mark.parametrize("coordinate_count", (2, 4))
def test_deform_attention_trace_matches_original_rank_six_math(coordinate_count):
    torch.manual_seed(29)
    module = _DeformAttention().eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.2, 0.2)

    query = torch.linspace(-0.7, 0.8, 2 * 5 * 16).reshape(2, 5, 16)
    input_flatten = torch.linspace(-1.1, 1.2, 2 * 8 * 16).reshape(2, 8, 16)
    reference_points = _reference_points(2, 5, coordinate_count)
    expected = _reference_deform_attention(
        module.attention,
        query,
        reference_points,
        input_flatten,
    )

    actual = module(query, reference_points, input_flatten)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    traced = torch.jit.trace(
        module,
        (query, reference_points, input_flatten),
        check_trace=True,
    )
    torch.testing.assert_close(
        traced(query, reference_points, input_flatten),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    tensor_ranks = [
        output.type().dim()
        for node in traced.inlined_graph.nodes()
        for output in node.outputs()
        if isinstance(output.type(), torch._C.TensorType)
        and output.type().dim() is not None
    ]
    assert tensor_ranks
    assert max(tensor_ranks) <= 5
