from __future__ import annotations

import pytest
import torch
from torch import nn

from libreyolo.export.coreml import (
    _CoreMLRtdetrDeformableAttention,
    _prepare_coreml_deformable_attention,
)
from libreyolo.models.rtdetr.nn import MSDeformableAttention

pytestmark = pytest.mark.unit


class _FixedShapesAttention(nn.Module):
    def __init__(self, attention: nn.Module):
        super().__init__()
        self.attention = attention

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.attention(
            query,
            reference_points,
            value,
            [[2, 2], [1, 2]],
            value_mask,
        )


def _jit_tensor_ranks(graph) -> list[int]:
    ranks = []
    for node in graph.nodes():
        for output in node.outputs():
            try:
                sizes = output.type().sizes()
            except RuntimeError:
                continue
            if sizes is not None:
                ranks.append(len(sizes))
    return ranks


def _attention_inputs(*, reference_width: int):
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(1, 5, 8, generator=generator)
    value = torch.randn(1, 6, 8, generator=generator)
    if reference_width == 2:
        reference_points = torch.rand(1, 5, 2, 2, generator=generator)
    else:
        reference_points = torch.rand(1, 5, 1, 4, generator=generator)
        reference_points[..., 2:] = reference_points[..., 2:] * 0.5 + 0.1
    value_mask = torch.tensor([[True, True, False, True, True, True]])
    return query, reference_points, value, [[2, 2], [1, 2]], value_mask


@pytest.mark.parametrize("reference_width", [2, 4])
def test_coreml_attention_matches_eager_without_rank_six(reference_width):
    torch.manual_seed(3)
    eager = MSDeformableAttention(
        embed_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
    ).eval()
    prepared = _CoreMLRtdetrDeformableAttention(eager).eval()
    inputs = _attention_inputs(reference_width=reference_width)

    with torch.inference_mode():
        expected = eager(*inputs)
        actual = prepared(*inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    traced = torch.jit.trace(
        _FixedShapesAttention(prepared),
        (inputs[0], inputs[1], inputs[2], inputs[4]),
        check_trace=True,
    )
    tensor_ranks = _jit_tensor_ranks(traced.inlined_graph)
    assert tensor_ranks
    assert max(tensor_ranks) <= 5


def test_coreml_attention_replacement_is_scoped_and_restored():
    attention = MSDeformableAttention(
        embed_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
    )
    model = nn.Sequential(attention)

    with (
        pytest.raises(RuntimeError, match="trace stopped"),
        _prepare_coreml_deformable_attention(model, "rtdetr", "detect"),
    ):
        assert isinstance(model[0], _CoreMLRtdetrDeformableAttention)
        raise RuntimeError("trace stopped")

    assert model[0] is attention


def test_other_families_are_not_rewritten():
    attention = MSDeformableAttention(
        embed_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
    )
    model = nn.Sequential(attention)

    with _prepare_coreml_deformable_attention(model, "rtdetrv2", "detect"):
        assert model[0] is attention
