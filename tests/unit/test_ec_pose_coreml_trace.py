from __future__ import annotations

import pytest
import torch
from torch import nn

from libreyolo.export.coreml import (
    _CoreMLECPoseDeformableAttention,
    _prepare_coreml_deformable_attention,
)
from libreyolo.models.ec.decoder import MSDeformAttnPose

pytestmark = pytest.mark.unit


class _FixedShapesPoseAttention(nn.Module):
    def __init__(self, attention: nn.Module):
        super().__init__()
        self.attention = attention

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        level_zero: torch.Tensor,
        level_one: torch.Tensor,
    ) -> torch.Tensor:
        return self.attention(
            query,
            reference_points,
            (level_zero, level_one),
            [(2, 2), (1, 2)],
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


def _attention_inputs(*, reference_width: int, reference_levels: int):
    generator = torch.Generator().manual_seed(29)
    query = torch.randn(1, 6, 8, generator=generator)
    if reference_width == 2:
        reference_points = torch.rand(
            1,
            3,
            reference_levels,
            2,
            2,
            generator=generator,
        )
    else:
        reference_points = torch.rand(
            1,
            3,
            reference_levels,
            2,
            4,
            generator=generator,
        )
        reference_points[..., 2:] = reference_points[..., 2:] * 0.5 + 0.1
    level_zero = torch.randn(2, 4, 4, generator=generator)
    level_one = torch.randn(2, 4, 2, generator=generator)
    return query, reference_points, level_zero, level_one


@pytest.mark.parametrize(
    ("reference_width", "reference_levels"),
    [(2, 1), (2, 2), (4, 1), (4, 2)],
)
def test_coreml_ec_pose_attention_matches_eager_without_rank_six(
    reference_width,
    reference_levels,
):
    torch.manual_seed(7)
    eager = MSDeformAttnPose(
        d_model=8,
        n_levels=2,
        n_heads=2,
        n_points=2,
    ).eval()
    for parameter in eager.parameters():
        torch.nn.init.uniform_(parameter, -0.2, 0.2)
    prepared = _CoreMLECPoseDeformableAttention(eager).eval()
    inputs = _attention_inputs(
        reference_width=reference_width,
        reference_levels=reference_levels,
    )

    with torch.inference_mode():
        expected = _FixedShapesPoseAttention(eager)(*inputs)
        actual = _FixedShapesPoseAttention(prepared)(*inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    traced = torch.jit.trace(
        _FixedShapesPoseAttention(prepared),
        inputs,
        check_trace=True,
    )
    tensor_ranks = _jit_tensor_ranks(traced.inlined_graph)
    assert tensor_ranks
    assert max(tensor_ranks) <= 5


def test_coreml_ec_pose_attention_replacement_is_scoped_and_restored():
    attention = MSDeformAttnPose(
        d_model=8,
        n_levels=2,
        n_heads=2,
        n_points=2,
    )
    model = nn.Sequential(attention)

    with (
        pytest.raises(RuntimeError, match="trace stopped"),
        _prepare_coreml_deformable_attention(model, "ec", "pose"),
    ):
        assert isinstance(model[0], _CoreMLECPoseDeformableAttention)
        raise RuntimeError("trace stopped")

    assert model[0] is attention


@pytest.mark.parametrize(
    ("family", "task"),
    [
        ("ec", "detect"),
        ("ec", "segment"),
        ("rfdetr", "pose"),
    ],
)
def test_non_ec_pose_models_are_not_rewritten(family, task):
    attention = MSDeformAttnPose(
        d_model=8,
        n_levels=2,
        n_heads=2,
        n_points=2,
    )
    model = nn.Sequential(attention)

    with _prepare_coreml_deformable_attention(model, family, task):
        assert model[0] is attention
