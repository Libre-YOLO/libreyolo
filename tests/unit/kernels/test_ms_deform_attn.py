"""CPU tests for the ``ms_deform_attn`` op slot and its call-site adapters."""

from __future__ import annotations

import importlib.util

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
    MSDeformAttn as RFDETRMSDeformAttn,
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
    # Hub kernels are on by default when the `kernels` package is installed;
    # pin them off so these tests behave the same on any machine.
    monkeypatch.setenv("LIBREYOLO_HUB_KERNELS", "0")
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


def test_slot_resolves_to_none_when_disabled():
    assert kernels.resolve("ms_deform_attn") is None
    value, shapes, locations, weights = _classic_inputs()
    assert maybe_ms_deform_attn(value, shapes, locations, weights) is None


def test_hub_default_on_and_env_opt_out(monkeypatch):
    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.delenv("LIBREYOLO_HUB_KERNELS", raising=False)
    assert module._hub_enabled()
    for value in ("0", "false", "off", "no"):
        monkeypatch.setenv("LIBREYOLO_HUB_KERNELS", value)
        assert not module._hub_enabled()


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


def test_rfdetr_layout_is_numerically_equivalent():
    """The rfdetr core's layout must express the same attention problem."""
    value, shapes, locations, weights = _classic_inputs()
    classic_out = classic_core(value, shapes, locations, weights)
    rfdetr_out = rfdetr_core(
        value.permute(0, 2, 3, 1).contiguous(),
        shapes,
        locations,
        weights.flatten(-2),
    )
    torch.testing.assert_close(rfdetr_out, classic_out, rtol=1e-5, atol=1e-5)


D_MODEL = HEADS * CHANNELS


def _rfdetr_attention_module():
    torch.manual_seed(0)
    return RFDETRMSDeformAttn(
        d_model=D_MODEL, n_levels=LEVELS, n_heads=HEADS, n_points=POINTS
    )


def _rfdetr_attention_inputs():
    generator = torch.Generator().manual_seed(1)
    query = torch.randn(BATCH, LEN_Q, D_MODEL, generator=generator)
    reference_points = torch.rand(BATCH, LEN_Q, LEVELS, 2, generator=generator)
    input_flatten = torch.randn(BATCH, LEN_IN, D_MODEL, generator=generator)
    spatial_shapes = torch.tensor(SHAPES, dtype=torch.int64)
    return query, reference_points, input_flatten, spatial_shapes


def test_rfdetr_attention_forward_routes_through_slot(monkeypatch):
    """The real RF-DETR attention module must consult the slot in eager mode.

    This is the regression test for the slot being wired at a call site the
    model actually reaches: RF-DETR always threads ``input_spatial_shapes_hw``
    through its decoder, so gating the slot on that argument being None
    (an earlier revision of this PR) made the kernel unreachable.
    """
    recorded = []
    _register_mock(monkeypatch, recorded)
    module = _rfdetr_attention_module()
    query, reference_points, input_flatten, spatial_shapes = (
        _rfdetr_attention_inputs()
    )
    out = module(
        query,
        reference_points,
        input_flatten,
        spatial_shapes,
        level_start_index(spatial_shapes),
        input_spatial_shapes_hw=SHAPES,
    )
    # The slot receives the classic Deformable-DETR layout...
    assert recorded == [
        (
            (BATCH, LEN_IN, HEADS, CHANNELS),
            (LEVELS, 2),
            (BATCH, LEN_Q, HEADS, LEVELS, POINTS, 2),
            (BATCH, LEN_Q, HEADS, LEVELS, POINTS),
        )
    ]
    # ...and its output feeds output_proj: mock returns all-7s, so the
    # module output equals output_proj of an all-7s tensor.
    expected = module.output_proj(
        torch.full((BATCH, LEN_Q, D_MODEL), 7.0)
    )
    torch.testing.assert_close(out, expected)


def test_rfdetr_attention_export_mode_skips_slot(monkeypatch):
    recorded = []
    _register_mock(monkeypatch, recorded)
    module = _rfdetr_attention_module()
    module.export()
    query, reference_points, input_flatten, spatial_shapes = (
        _rfdetr_attention_inputs()
    )
    module(
        query,
        reference_points,
        input_flatten,
        spatial_shapes,
        level_start_index(spatial_shapes),
        input_spatial_shapes_hw=SHAPES,
    )
    assert recorded == []


def test_rfdetr_attention_slot_matches_portable(monkeypatch):
    """A slot impl wrapping the portable core must reproduce module output."""

    def portable_impl(value, spatial_shapes, sampling_locations, attention_weights):
        # Wrap the rfdetr portable core (slot-free) rather than classic_core,
        # which consults the slot itself and would recurse into this impl.
        return rfdetr_core(
            value.permute(0, 2, 3, 1).contiguous(),
            spatial_shapes,
            sampling_locations,
            attention_weights.flatten(-2),
        )

    module = _rfdetr_attention_module()
    query, reference_points, input_flatten, spatial_shapes = (
        _rfdetr_attention_inputs()
    )
    args = (
        query,
        reference_points,
        input_flatten,
        spatial_shapes,
        level_start_index(spatial_shapes),
    )
    baseline = module(*args, input_spatial_shapes_hw=SHAPES)

    kernels.register("ms_deform_attn", portable_impl, name="mock")
    monkeypatch.setenv("LIBREYOLO_KERNELS", "mock")
    kernels.clear_cache()
    routed = module(*args, input_spatial_shapes_hw=SHAPES)
    torch.testing.assert_close(routed, baseline, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("kernels") is None,
    reason="needs CUDA and the `kernels` package (libreyolo[hub-kernels])",
)
def test_hub_matches_portable_on_cuda():
    """Forward/backward parity of the pinned Hub kernel vs the portable core.

    This is the GPU smoke for the provider: run it on any CUDA box with the
    ``hub-kernels`` extra installed before bumping ``_HUB_REVISION``.
    """
    value, shapes, locations, weights = _classic_inputs()
    value = value.cuda().requires_grad_(True)
    shapes = shapes.cuda()
    locations = locations.cuda().requires_grad_(True)
    weights = weights.cuda().requires_grad_(True)

    hub_out = hub_ms_deform_attn(value, shapes, locations, weights)
    if hub_out is None:
        pytest.skip("hub kernel unavailable on this box (load failed)")
    hub_out.sum().backward()
    hub_grads = (value.grad.clone(), locations.grad.clone(), weights.grad.clone())

    value.grad = locations.grad = weights.grad = None
    # classic_core consults the slot itself; the autouse fixture pins
    # LIBREYOLO_HUB_KERNELS=0 for this file, so it runs the portable path.
    ref_out = classic_core(value, shapes, locations, weights)
    ref_out.sum().backward()
    ref_grads = (value.grad, locations.grad, weights.grad)

    torch.testing.assert_close(hub_out, ref_out, rtol=1e-4, atol=1e-5)
    for hub_grad, ref_grad in zip(hub_grads, ref_grads):
        torch.testing.assert_close(hub_grad, ref_grad, rtol=1e-3, atol=1e-4)


# =============================================================================
# Pinned-snapshot fallback loader (the ``kernels``-resolver compatibility path)
# =============================================================================


def test_pinned_variant_name_maps_platform(monkeypatch):
    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.setattr(torch, "__version__", "2.11.0+cu128")
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(module._platform, "system", lambda: "Linux")
    monkeypatch.setattr(module._platform, "machine", lambda: "x86_64")
    assert module._pinned_variant_name() == "torch211-cxx11-cu128-x86_64-linux"

    monkeypatch.setattr(module._platform, "system", lambda: "Windows")
    monkeypatch.setattr(module._platform, "machine", lambda: "AMD64")
    assert module._pinned_variant_name() == "torch211-cu128-x86_64-windows"

    monkeypatch.setattr(module._platform, "machine", lambda: "arm64")
    monkeypatch.setattr(module._platform, "system", lambda: "Linux")
    assert module._pinned_variant_name() == "torch211-cxx11-cu128-aarch64-linux"

    # CPU-only torch has no CUDA build to match.
    monkeypatch.setattr(torch.version, "cuda", None)
    assert module._pinned_variant_name() is None


def test_load_hub_kernel_falls_back_when_resolver_rejects_pin(monkeypatch):
    """A ``kernels`` release that cannot resolve the SHA pin must not kill the
    provider: the direct snapshot loader is tried before giving up."""
    import sys
    import types

    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.setattr(module, "_hub_kernel", None)
    monkeypatch.setattr(module, "_hub_failed", False)

    fake_kernels = types.ModuleType("kernels")

    def rejects_sha(*args, **kwargs):
        raise ValueError("Invalid rev id")

    fake_kernels.get_kernel = rejects_sha
    monkeypatch.setitem(sys.modules, "kernels", fake_kernels)

    sentinel = object()
    monkeypatch.setattr(module, "_load_pinned_snapshot", lambda: sentinel)
    assert module._load_hub_kernel() is sentinel
    assert module._hub_failed is False


def test_load_hub_kernel_disables_when_both_paths_fail(monkeypatch):
    import sys
    import types

    from libreyolo.kernels.attention import ms_deform_attn as module

    monkeypatch.setattr(module, "_hub_kernel", None)
    monkeypatch.setattr(module, "_hub_failed", False)

    fake_kernels = types.ModuleType("kernels")

    def rejects_sha(*args, **kwargs):
        raise ValueError("Invalid rev id")

    fake_kernels.get_kernel = rejects_sha
    monkeypatch.setitem(sys.modules, "kernels", fake_kernels)

    def no_snapshot():
        raise OSError("offline")

    monkeypatch.setattr(module, "_load_pinned_snapshot", no_snapshot)
    assert module._load_hub_kernel() is None
    assert module._hub_failed is True
