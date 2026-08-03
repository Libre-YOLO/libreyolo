"""CPU tests for the ``ms_deform_attn`` op slot and its call-site adapters."""

from __future__ import annotations

import pytest
import torch

from libreyolo import kernels
from libreyolo.kernels.attention.ms_deform_attn import (
    hub_ms_deform_attn,
    level_start_index,
    maybe_ms_deform_attn,
)
from libreyolo.models.deformable_detr.ms_deform_attn import (
    ms_deform_attn_core_pytorch as classic_core,
)
from libreyolo.models.rfdetr.transformer import (
    ms_deform_attn_core_pytorch as rfdetr_core,
)

pytestmark = pytest.mark.unit

BATCH, LEN_Q, HEADS, CHANNELS = 2, 3, 2, 4
SHAPES = [(4, 6), (2, 3)]
LEVELS, POINTS = len(SHAPES), 2
LEN_IN = sum(h * w for h, w in SHAPES)


@pytest.fixture(autouse=True)
def _clean_registry_env(monkeypatch):
    monkeypatch.delenv("LIBREYOLO_KERNELS", raising=False)
    monkeypatch.delenv("LIBREYOLO_QUANT_KERNELS", raising=False)
    monkeypatch.delenv("LIBREYOLO_HUB_KERNELS", raising=False)
    kernels.clear_cache()
    yield
    kernels.unregister("ms_deform_attn", "mock")
    kernels.clear_cache()


def _classic_inputs():
    generator = torch.Generator().manual_seed(0)
    value = torch.randn(BATCH, LEN_IN, HEADS, CHANNELS, generator=generator)
    spatial_shapes = torch.tensor(SHAPES, dtype=torch.int64)
    sampling_locations = torch.rand(
        BATCH, LEN_Q, HEADS, LEVELS, POINTS, 2, generator=generator
    )
    attention_weights = torch.rand(
        BATCH, LEN_Q, HEADS, LEVELS, POINTS, generator=generator
    )
    attention_weights = attention_weights / attention_weights.sum(
        dim=(-2, -1), keepdim=True
    )
    return value, spatial_shapes, sampling_locations, attention_weights


def test_slot_resolves_to_none_by_default():
    assert kernels.resolve("ms_deform_attn") is None
    value, shapes, locations, weights = _classic_inputs()
    assert maybe_ms_deform_attn(value, shapes, locations, weights) is None


def test_hub_impl_rejects_cpu_inputs():
    value, shapes, locations, weights = _classic_inputs()
    assert hub_ms_deform_attn(value, shapes, locations, weights) is None


def test_level_start_index():
    shapes = torch.tensor(SHAPES, dtype=torch.int64)
    expected = torch.tensor([0, SHAPES[0][0] * SHAPES[0][1]], dtype=torch.int64)
    assert torch.equal(level_start_index(shapes), expected)


def _register_mock(monkeypatch, recorded):
    def mock_impl(value, spatial_shapes, sampling_locations, attention_weights):
        recorded.append(
            (
                tuple(value.shape),
                tuple(spatial_shapes.shape),
                tuple(sampling_locations.shape),
                tuple(attention_weights.shape),
            )
        )
        heads_times_c = value.shape[2] * value.shape[3]
        return torch.full(
            (value.shape[0], sampling_locations.shape[1], heads_times_c), 7.0
        )

    kernels.register("ms_deform_attn", mock_impl, name="mock")
    monkeypatch.setenv("LIBREYOLO_KERNELS", "mock")
    kernels.clear_cache()


def test_classic_call_site_routes_through_slot(monkeypatch):
    recorded = []
    _register_mock(monkeypatch, recorded)
    value, shapes, locations, weights = _classic_inputs()
    out = classic_core(value, shapes, locations, weights)
    assert torch.equal(
        out, torch.full((BATCH, LEN_Q, HEADS * CHANNELS), 7.0)
    )
    assert recorded == [
        (
            (BATCH, LEN_IN, HEADS, CHANNELS),
            (LEVELS, 2),
            (BATCH, LEN_Q, HEADS, LEVELS, POINTS, 2),
            (BATCH, LEN_Q, HEADS, LEVELS, POINTS),
        )
    ]


def test_rfdetr_call_site_adapts_layout(monkeypatch):
    recorded = []
    _register_mock(monkeypatch, recorded)
    value, shapes, locations, weights = _classic_inputs()
    rfdetr_value = value.permute(0, 2, 3, 1).contiguous()
    rfdetr_weights = weights.flatten(-2)
    out = rfdetr_core(rfdetr_value, shapes, locations, rfdetr_weights)
    assert torch.equal(
        out, torch.full((BATCH, LEN_Q, HEADS * CHANNELS), 7.0)
    )
    # The adapter must hand the slot the classic layout.
    assert recorded == [
        (
            (BATCH, LEN_IN, HEADS, CHANNELS),
            (LEVELS, 2),
            (BATCH, LEN_Q, HEADS, LEVELS, POINTS, 2),
            (BATCH, LEN_Q, HEADS, LEVELS, POINTS),
        )
    ]


def test_rfdetr_adapter_is_numerically_equivalent():
    """The rfdetr layout adaptation must express the same attention problem."""
    value, shapes, locations, weights = _classic_inputs()
    classic_out = classic_core(value, shapes, locations, weights)
    rfdetr_out = rfdetr_core(
        value.permute(0, 2, 3, 1).contiguous(),
        shapes,
        locations,
        weights.flatten(-2),
    )
    torch.testing.assert_close(rfdetr_out, classic_out, rtol=1e-5, atol=1e-5)


def test_export_hw_path_skips_slot(monkeypatch):
    recorded = []
    _register_mock(monkeypatch, recorded)
    value, shapes, locations, weights = _classic_inputs()
    rfdetr_core(
        value.permute(0, 2, 3, 1).contiguous(),
        shapes,
        locations,
        weights.flatten(-2),
        value_spatial_shapes_hw=SHAPES,
    )
    assert recorded == []
