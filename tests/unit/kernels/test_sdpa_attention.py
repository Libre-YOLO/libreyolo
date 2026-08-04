"""Per-family SDPA swaps: same equation, same state_dict, untouched export path.

Every family below had ``q @ k.T -> softmax -> @ v`` replaced by
``F.scaled_dot_product_attention``. Which of the two paths runs by default is
decided by the family's parity bar (see ``libreyolo/kernels/attention/sdpa.py``):

* **default-on** - the bar is a tolerance, so SDPA runs unless ONNX export is
  tracing;
* **opt-in** - the bar is ``max_abs_diff == 0`` against a reference that itself
  runs manual attention, so manual math stays the default and
  ``set_fused_attention`` is the switch.

The checks are the same either way: the two paths agree numerically (including
under every mask shape the family passes), no parameter appears or disappears,
and ONNX export always takes the manual path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from libreyolo.kernels.attention import fused_attention_modules, set_fused_attention
from libreyolo.kernels.attention.sdpa import manual_attention_required

pytestmark = pytest.mark.unit

DIM, HEADS, BATCH, TOKENS = 32, 4, 2, 16
WINDOW = (4, 4)
NUM_WINDOWS = 4
ATOL = 1e-5


def _seeded(*shape, device="cpu"):
    generator = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=generator, device=device)


def _window_mask():
    """A shifted-window mask in the (num_windows, tokens, tokens) layout."""
    mask = torch.zeros(NUM_WINDOWS, TOKENS, TOKENS)
    mask[1, :, TOKENS // 2 :] = -100.0
    mask[3, : TOKENS // 2, :] = -100.0
    return mask


def _additive_mask(rows=TOKENS, cols=TOKENS):
    mask = torch.zeros(BATCH, 1, rows, cols)
    mask[0, :, :, cols // 2 :] = torch.finfo(torch.float32).min
    return mask


# --- one (build, call) pair per rewired attention module -------------------


def _segformer(sr_ratio):
    from libreyolo.models.segformer.nn import EfficientSelfAttention

    module = EfficientSelfAttention(DIM, HEADS, sr_ratio).eval()
    return module, lambda m: m(_seeded(BATCH, TOKENS, DIM), 4, 4)


def _depth_anything():
    from libreyolo.models.depth_anything._vendor.dinov2_layers.attention import Attention

    return Attention(DIM, HEADS).eval(), lambda m: m(_seeded(BATCH, TOKENS, DIM))


def _bert(mask):
    from libreyolo.models.bert.nn import BertSelfAttention

    cfg = SimpleNamespace(num_attention_heads=HEADS, hidden_size=DIM)
    return BertSelfAttention(cfg).eval(), lambda m: m(_seeded(BATCH, TOKENS, DIM), mask)


def _gdino(mask):
    from libreyolo.models.grounding_dino.nn import GDMultiheadAttention

    def call(module):
        x = _seeded(BATCH, TOKENS, DIM)
        return module(x, x, x, attention_mask=mask)

    return GDMultiheadAttention(DIM, HEADS).eval(), call


def _swinir(mask):
    from libreyolo.models.swinir.nn import WindowAttention

    module = WindowAttention(DIM, WINDOW, HEADS).eval()
    torch.nn.init.normal_(module.relative_position_bias_table, std=0.5)
    windows = BATCH * NUM_WINDOWS
    return module, lambda m: m(_seeded(windows, TOKENS, DIM), mask)


def _ppocr():
    from libreyolo.models.ppocr.rec import Attention

    return Attention(DIM, HEADS).eval(), lambda m: m(_seeded(BATCH, TOKENS, DIM))


def _swin(mask, tf_order):
    from libreyolo.models.swin.nn import WindowAttention

    module = WindowAttention(DIM, HEADS, WINDOW[0], tf_order=tf_order).eval()
    torch.nn.init.normal_(module.relative_position_bias_table, std=0.5)
    windows = BATCH * NUM_WINDOWS
    return module, lambda m: m(_seeded(windows, TOKENS, DIM), mask)


def _dinodetr_swin(mask):
    from libreyolo.models.dinodetr.swin import WindowAttention

    module = WindowAttention(DIM, WINDOW, HEADS).eval()
    torch.nn.init.normal_(module.relative_position_bias_table, std=0.5)
    windows = BATCH * NUM_WINDOWS
    return module, lambda m: m(_seeded(windows, TOKENS, DIM), mask)


def _birefnet(mask):
    from libreyolo.models.birefnet.nn import WindowAttention

    module = WindowAttention(DIM, WINDOW, HEADS).eval()
    torch.nn.init.normal_(module.relative_position_bias_table, std=0.5)
    windows = BATCH * NUM_WINDOWS
    return module, lambda m: m(_seeded(windows, TOKENS, DIM), mask)


def _owlv2(mask):
    from libreyolo.models.owlv2.nn import Owlv2Attention

    return Owlv2Attention(DIM, HEADS).eval(), lambda m: m(_seeded(BATCH, TOKENS, DIM), mask)


def _lwdetr_vit(use_cae):
    from libreyolo.models.lwdetr.nn import Attention

    module = Attention(DIM, HEADS, use_cae=use_cae).eval()
    return module, lambda m: m(_seeded(BATCH, TOKENS, DIM))


def _lwdetr_mha():
    from libreyolo.models.lwdetr.nn import MultiheadAttention

    def call(module):
        x = _seeded(BATCH, TOKENS, DIM)
        return module(x, x, x)

    return MultiheadAttention(DIM, HEADS).eval(), call


def _siglip2(mask):
    from libreyolo.models.siglip2.nn import SiglipAttention

    return SiglipAttention(DIM, HEADS).eval(), lambda m: m(_seeded(BATCH, TOKENS, DIM), mask)


def _zipdepth():
    from libreyolo.models.zipdepth.nn import EfficientGlobalAttention

    module = EfficientGlobalAttention(DIM, num_tokens=4, num_heads=HEADS).eval()
    return module, lambda m: m(_seeded(BATCH, DIM, 4, 4))


def _mobilesam_tinyvit():
    from libreyolo.models.mobilesam.nn import Attention

    # This class overrides train() with an @torch.no_grad() wrapper that
    # returns None, so .eval() cannot be chained.
    module = Attention(DIM, key_dim=8, num_heads=HEADS, resolution=WINDOW)
    torch.nn.init.normal_(module.attention_biases, std=0.5)
    module.eval()  # caches `ab` from the perturbed biases
    return module, lambda m: m(_seeded(BATCH, TOKENS, DIM))


def _mobilesam_transformer():
    from libreyolo.models.mobilesam.transformer import Attention

    def call(module):
        x = _seeded(BATCH, TOKENS, DIM)
        return module(x, x, x)

    return Attention(DIM, HEADS).eval(), call


DEFAULT_ON = {
    "segformer": lambda: _segformer(1),
    "segformer_reduced": lambda: _segformer(2),
    "depth_anything": _depth_anything,
    "bert": lambda: _bert(None),
    "bert_float_mask": lambda: _bert(_additive_mask()),
    "bert_bool_mask": lambda: _bert(torch.eye(TOKENS, dtype=torch.bool)[None, None]),
    "grounding_dino": lambda: _gdino(None),
    "grounding_dino_mask": lambda: _gdino(_additive_mask()),
    "swinir": lambda: _swinir(None),
    "swinir_masked": lambda: _swinir(_window_mask()),
    "ppocr": _ppocr,
}

OPT_IN = {
    "swin": lambda: _swin(None, False),
    "swin_masked": lambda: _swin(_window_mask(), False),
    "swin_tf_order": lambda: _swin(_window_mask(), True),
    "dinodetr_swin": lambda: _dinodetr_swin(None),
    "dinodetr_swin_masked": lambda: _dinodetr_swin(_window_mask()),
    "birefnet": lambda: _birefnet(None),
    "birefnet_masked": lambda: _birefnet(_window_mask()),
    "owlv2": lambda: _owlv2(None),
    "owlv2_masked": lambda: _owlv2(_additive_mask()),
    "lwdetr_vit": lambda: _lwdetr_vit(False),
    "lwdetr_vit_cae": lambda: _lwdetr_vit(True),
    "lwdetr_mha": _lwdetr_mha,
    "siglip2": lambda: _siglip2(None),
    "siglip2_masked": lambda: _siglip2(_additive_mask()),
    "zipdepth": _zipdepth,
    "mobilesam_tinyvit": _mobilesam_tinyvit,
    "mobilesam_transformer": _mobilesam_transformer,
}


@pytest.mark.parametrize("case", sorted(OPT_IN))
def test_opt_in_default_is_manual_and_fused_agrees(case):
    """Manual math is what runs by default; opting in changes only rounding."""
    module, call = OPT_IN[case]()
    assert getattr(module, "fused_attn") is False
    with torch.no_grad():
        manual = call(module)
        assert set_fused_attention(module) >= 1
        fused = call(module)
    torch.testing.assert_close(fused, manual, rtol=1e-5, atol=ATOL)


@pytest.mark.parametrize("case", sorted(OPT_IN))
def test_opt_in_export_ignores_the_flag(case, monkeypatch):
    """Even opted in, ONNX export must trace the primitive-op equation."""
    module, call = OPT_IN[case]()
    with torch.no_grad():
        manual = call(module)
        set_fused_attention(module)
        monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
        exported = call(module)
    torch.testing.assert_close(exported, manual, rtol=0, atol=0)


@pytest.mark.parametrize("case", sorted(OPT_IN))
def test_opt_in_round_trips_back_to_manual(case):
    module, _ = OPT_IN[case]()
    assert set_fused_attention(module, True) >= 1
    assert all(m.fused_attn for m in fused_attention_modules(module))
    assert set_fused_attention(module, False) >= 1
    assert not any(m.fused_attn for m in fused_attention_modules(module))


@pytest.mark.parametrize("case", sorted(DEFAULT_ON))
def test_default_on_matches_its_export_path(case, monkeypatch):
    """SDPA runs by default and agrees with the primitive-op export path."""
    module, call = DEFAULT_ON[case]()
    with torch.no_grad():
        fused = call(module)
        monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
        exported = call(module)
    torch.testing.assert_close(fused, exported, rtol=1e-5, atol=ATOL)


@pytest.mark.parametrize("case", sorted({**DEFAULT_ON, **OPT_IN}))
def test_state_dict_keys_are_untouched(case):
    """A forward-math swap must not add, drop or rename a single parameter."""
    module, _ = {**DEFAULT_ON, **OPT_IN}[case]()
    before = sorted(module.state_dict())
    set_fused_attention(module, True)
    assert sorted(module.state_dict()) == before
    # `fused_attn` is a plain bool attribute, never a buffer or parameter.
    assert not any(key.endswith("fused_attn") for key in before)


def test_bool_mask_keeps_the_additive_bert_semantics():
    """transformers' eager path ADDS the bool mask; SDPA would read it as a gate.

    Without the cast in BertSelfAttention a False entry would become -inf
    instead of +0, which is a different model, not a rounding difference.
    """
    from libreyolo.models.bert.nn import BertSelfAttention

    cfg = SimpleNamespace(num_attention_heads=HEADS, hidden_size=DIM)
    module = BertSelfAttention(cfg).eval()
    x = _seeded(BATCH, TOKENS, DIM)
    mask = torch.zeros(BATCH, 1, TOKENS, TOKENS, dtype=torch.bool)
    mask[:, :, :, : TOKENS // 2] = True

    with torch.no_grad():
        fused = module(x, mask)
        as_gate = module(x, torch.where(mask, 0.0, float("-inf")))
        as_bias = module(x, mask.to(x.dtype))
    torch.testing.assert_close(fused, as_bias, rtol=1e-5, atol=ATOL)
    assert (fused - as_gate).abs().max().item() > 1e-3


def test_set_fused_attention_reports_zero_on_a_model_without_the_flag():
    assert set_fused_attention(torch.nn.Linear(4, 4)) == 0


# --- capture gating -------------------------------------------------------


@pytest.mark.parametrize("case", sorted({**DEFAULT_ON, **OPT_IN}))
def test_jit_trace_captures_the_primitive_op_equation(case):
    """TorchScript / CoreML / NCNN capture with jit.trace, which sets no ONNX flag.

    Their artifacts were validated on primitive-op graphs, so a traced graph
    must not contain aten::scaled_dot_product_attention even when the family
    runs SDPA eagerly.
    """
    module, call = {**DEFAULT_ON, **OPT_IN}[case]()
    set_fused_attention(module, True)

    class Traceable(torch.nn.Module):
        """Wrapper so masks and None arguments stay inside the traced region.

        jit.trace only accepts tensors as example inputs; several of these
        modules take an optional mask. The inputs are built inside forward and
        become trace constants, which is fine - the assertion is about which
        ops the graph contains, not about re-running it.
        """

        def __init__(self):
            super().__init__()
            self.inner = module

        def forward(self, unused):
            return call(self.inner)

    with torch.no_grad():
        traced = torch.jit.trace(Traceable(), torch.zeros(1), check_trace=False)
    assert "scaled_dot_product_attention" not in str(traced.inlined_graph), case


def test_manual_attention_required_covers_onnx_and_trace(monkeypatch):
    assert not manual_attention_required()
    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
    assert manual_attention_required()
    monkeypatch.undo()
    monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
    assert manual_attention_required()


def test_manual_attention_required_leaves_torch_compile_alone(monkeypatch):
    """torch.compile lowers SDPA better than the manual equation."""
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert not manual_attention_required()


# --- the timm-compat flags that used to be vestigial ----------------------


@pytest.mark.parametrize("family", ["vit", "deit"])
def test_timm_compat_fused_attn_flag_is_honored(family):
    """set_fused_attention must not report switching a flag nothing reads."""
    if family == "vit":
        from libreyolo.models.vit.nn import Attention
    else:
        from libreyolo.models.deit.nn import Attention

    module = Attention(DIM, HEADS).eval()
    assert module.fused_attn is True, "these two default to SDPA, unlike the campaign families"
    x = _seeded(BATCH, TOKENS, DIM)
    with torch.no_grad():
        fused = module(x)
        assert set_fused_attention(module, False) == 1
        manual = module(x)
    # Same equation, so the outputs still agree - but the flag must actually
    # have selected a different code path.
    torch.testing.assert_close(fused, manual, rtol=1e-5, atol=ATOL)
    assert not torch.equal(fused, manual), (
        f"{family}: set_fused_attention reported a switch that changed nothing"
    )


def test_set_fused_attention_accepts_a_task_wrapper():
    """Users hold the task object, whose nn.Module lives on `.model`."""
    module, _ = OPT_IN["swin"]()
    wrapper = SimpleNamespace(model=module)
    assert set_fused_attention(wrapper) >= 1
    assert all(m.fused_attn for m in fused_attention_modules(wrapper))
    with pytest.raises(TypeError, match="expected an nn.Module"):
        set_fused_attention(SimpleNamespace())
