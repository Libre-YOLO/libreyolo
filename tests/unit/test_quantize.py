"""Unit tests for the PyTorch-native quantization API."""

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit

from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.quant import (
    NVFP4Linear,
    QuantConv2d,
    QuantizationError,
)
from libreyolo.quant.fake_quant import (
    E2M1_MAX,
    fake_quant_e2m1,
    fake_quant_int8_affine,
    fake_quant_int8_per_channel,
    fake_quant_nvfp4_weight,
    int8_weight_qparams,
)
from libreyolo.quant.packing import (
    pack_int8_weight,
    pack_nvfp4_weight,
    unpack_int8_weight,
    unpack_nvfp4_weight,
)


# ---------------------------------------------------------------------------
# Arithmetic primitives
# ---------------------------------------------------------------------------


def test_e2m1_rounds_to_codebook():
    vals = torch.tensor([0.0, 0.24, 0.26, 0.5, 1.2, 1.6, 2.4, 3.4, 4.9, 5.1, 7.5, -2.6])
    expected = torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 6.0, -3.0])
    assert torch.equal(fake_quant_e2m1(vals), expected)


def test_e2m1_clamps_to_max():
    out = fake_quant_e2m1(torch.tensor([100.0, -100.0]))
    assert out.abs().max().item() == E2M1_MAX


def test_int8_per_channel_error_is_small():
    torch.manual_seed(0)
    w = torch.randn(32, 16, 3, 3)
    wq = fake_quant_int8_per_channel(w)
    rel = (w - wq).norm() / w.norm()
    assert rel < 0.01


def test_int8_affine_respects_range():
    x = torch.linspace(-1.0, 3.0, 1000)
    lo = torch.tensor([-1.0])
    hi = torch.tensor([3.0])
    xq = fake_quant_int8_affine(x, lo, hi)
    assert (x - xq).abs().max() < (4.0 / 255.0)


def test_nvfp4_weight_error_reasonable():
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 0.02
    wq = fake_quant_nvfp4_weight(w, w.abs().amax().reshape(1))
    rel = (w - wq).norm() / w.norm()
    assert rel < 0.15


def test_ste_gradients_flow():
    w = torch.randn(8, 16, requires_grad=True)
    out = fake_quant_int8_per_channel(w)
    out.sum().backward()
    assert w.grad is not None
    assert torch.isfinite(w.grad).all()


# ---------------------------------------------------------------------------
# Quantized modules
# ---------------------------------------------------------------------------


def test_nvfp4_linear_matches_float_approximately():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    qlin = NVFP4Linear.from_float(lin)
    x = torch.randn(4, 64)
    rel = (lin(x) - qlin(x)).norm() / lin(x).norm()
    assert rel < 0.25


def test_quant_conv_preserves_state_dict_keys():
    conv = nn.Conv2d(8, 16, 3, padding=1)
    qconv = QuantConv2d.from_float(conv)
    keys = set(qconv.state_dict().keys())
    assert {"weight", "bias"} <= keys
    assert qconv.weight is conv.weight


def test_percentile_observation_rejects_outliers():
    conv = nn.Conv2d(3, 8, 3, padding=1)

    def calibrate(method):
        q = QuantConv2d.from_float(conv)
        q._q_obs_method = method
        q._q_observing = True
        torch.manual_seed(0)
        with torch.no_grad():
            for _ in range(4):
                x = torch.randn(2, 3, 16, 16)
                x[0, 0, 0, 0] = 100.0  # single extreme outlier per batch
                q(x)
        q._q_observing = False
        q.finalize_observation()
        return float(q._q_act_hi)

    hi_minmax = calibrate("minmax")
    hi_pct = calibrate("percentile")
    assert hi_minmax >= 100.0
    assert hi_pct < 10.0, f"percentile should clip the outlier, got {hi_pct}"


def test_invalid_calibration_algorithm_rejected(yolo9t):
    with pytest.raises(QuantizationError, match="calibration algorithm"):
        yolo9t.quantize(recipe="int8", calib=None, algorithm="entropy")


def test_multi_batch_observation_widens_ranges():
    # Regression: the min/max merge branch only fires from the second
    # calibration batch onwards (coco8's single batch never exercised it).
    conv = nn.Conv2d(3, 8, 3, padding=1)
    qconv = QuantConv2d.from_float(conv)
    qconv._q_observing = True
    with torch.no_grad():
        qconv(torch.full((1, 3, 8, 8), 0.5))
        qconv(torch.full((1, 3, 8, 8), -2.0))
        qconv(torch.full((1, 3, 8, 8), 3.0))
    qconv._q_observing = False
    assert qconv.q_calibrated
    assert qconv._q_act_lo.item() <= -2.0
    assert qconv._q_act_hi.item() >= 3.0


def test_quant_buffers_live_on_weight_device():
    # Regression: CPU-born buffers crashed multi-batch GPU calibration.
    conv = nn.Conv2d(3, 8, 3, padding=1)
    if torch.cuda.is_available():
        conv = conv.cuda()
    qconv = QuantConv2d.from_float(conv)
    assert qconv._q_act_lo.device == qconv.weight.device
    assert qconv._q_w_scale.device == qconv.weight.device
    lin = nn.Linear(8, 4)
    if torch.cuda.is_available():
        lin = lin.cuda()
    qlin = NVFP4Linear.from_float(lin)
    assert qlin._q_w_amax.device == qlin.weight.device


# ---------------------------------------------------------------------------
# Packing (finalized checkpoints)
# ---------------------------------------------------------------------------


def test_int8_pack_unpack_matches_simulation():
    torch.manual_seed(0)
    w = torch.randn(16, 8, 3, 3)
    scale = int8_weight_qparams(w)
    packed = pack_int8_weight(w, scale)
    assert packed.dtype == torch.int8
    assert torch.equal(unpack_int8_weight(packed, scale), fake_quant_int8_per_channel(w))


def test_nvfp4_pack_unpack_matches_simulation():
    torch.manual_seed(0)
    for in_features in (48, 20):  # multiple and non-multiple of the 16-block
        w = torch.randn(8, in_features) * 0.05
        amax = w.abs().amax().reshape(1)
        payload, block_scale = pack_nvfp4_weight(w, amax)
        assert payload.dtype == torch.uint8
        assert payload.shape == (8, ((in_features + 15) // 16) * 8)
        out = unpack_nvfp4_weight(payload, block_scale, amax, in_features)
        assert torch.equal(out, fake_quant_nvfp4_weight(w, amax))


def test_nvfp4_linear_finalize_roundtrip():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    q = NVFP4Linear.from_float(lin)
    x = torch.randn(4, 64)
    with torch.no_grad():
        ref = q(x)
        q.make_finalized()
        assert q.is_finalized and "weight" not in dict(q.named_parameters())
        assert torch.equal(q(x), ref)
        q.make_prepared()
        assert not q.is_finalized
        assert torch.equal(q(x), ref)


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------


def test_kernel_registry_reference_forced_by_env(monkeypatch):
    # Order-independent: accelerated kernels may already be registered by
    # other test modules; forcing the env must always yield the reference.
    from libreyolo.quant import fake_quant, kernels

    monkeypatch.setenv("LIBREYOLO_QUANT_KERNELS", "reference")
    kernels.clear_cache()
    try:
        for op in kernels.REFERENCE_OPS:
            assert kernels.resolve(op) is getattr(fake_quant, op)
        assert kernels.active()["fake_quant_int8_per_channel"] == "reference"
    finally:
        monkeypatch.delenv("LIBREYOLO_QUANT_KERNELS", raising=False)
        kernels.clear_cache()


def test_kernel_registry_custom_impl_and_env_override(monkeypatch):
    from libreyolo.quant import fake_quant, kernels

    calls = {}

    def custom(weight, ch_axis=0, scale=None):
        calls["hit"] = True
        return fake_quant.fake_quant_int8_per_channel(weight, ch_axis, scale)

    kernels.register(
        "fake_quant_int8_per_channel", custom, name="testkern",
        predicate=lambda: True,
    )
    try:
        kernels.clear_cache()
        assert kernels.resolve("fake_quant_int8_per_channel") is custom
        out = kernels.fake_quant_int8_per_channel(torch.randn(4, 8))
        assert calls.get("hit") and out.shape == (4, 8)
        assert kernels.active()["fake_quant_int8_per_channel"] == "testkern"

        # A quantized module transparently uses the registered kernel.
        calls.clear()
        conv = QuantConv2d.from_float(nn.Conv2d(3, 8, 3, padding=1))
        with torch.no_grad():
            conv(torch.randn(1, 3, 8, 8))
        assert calls.get("hit")

        # Env override forces the reference implementation.
        monkeypatch.setenv("LIBREYOLO_QUANT_KERNELS", "off")
        kernels.clear_cache()
        assert (
            kernels.resolve("fake_quant_int8_per_channel")
            is fake_quant.fake_quant_int8_per_channel
        )
    finally:
        monkeypatch.delenv("LIBREYOLO_QUANT_KERNELS", raising=False)
        kernels.unregister("fake_quant_int8_per_channel", "testkern")
        kernels.clear_cache()


def test_kernel_registry_predicate_gates_selection():
    # Order-independent: a failing predicate must never win the slot,
    # whichever other implementations happen to be registered.
    from libreyolo.quant import kernels

    never_impl = lambda *a, **k: None  # noqa: E731
    kernels.register(
        "fake_quant_int8_per_channel", never_impl, name="never",
        predicate=lambda: False,
    )
    try:
        kernels.clear_cache()
        assert kernels.resolve("fake_quant_int8_per_channel") is not never_impl
    finally:
        kernels.unregister("fake_quant_int8_per_channel", "never")
        kernels.clear_cache()


def test_intree_triton_kernels_activate():
    # On machines with triton + CUDA the in-tree kernels must self-activate
    # lazily on first resolution (no explicit import required).
    import importlib.util

    if importlib.util.find_spec("triton") is None or not torch.cuda.is_available():
        pytest.skip("triton + CUDA required")
    from libreyolo.quant import kernels

    kernels.clear_cache()
    assert kernels.active()["fake_quant_nvfp4_weight"] == "triton"
    assert kernels.active()["unpack_nvfp4"] == "triton"


# ---------------------------------------------------------------------------
# New precision recipes: arithmetic + module roundtrips
# ---------------------------------------------------------------------------


def test_fp8_fake_quant_and_packing():
    from libreyolo.quant.fake_quant import fake_quant_fp8, fp8_weight_qparams
    from libreyolo.quant.packing import pack_fp8_weight, unpack_fp8_weight

    torch.manual_seed(0)
    w = torch.randn(16, 8, 3, 3)
    scale = fp8_weight_qparams(w)
    view = scale.reshape(-1, 1, 1, 1)
    fq = fake_quant_fp8(w, view)
    rel = (w - fq).norm() / w.norm()
    assert rel < 0.05
    packed = pack_fp8_weight(w, scale)
    assert packed.dtype == torch.float8_e4m3fn
    assert torch.equal(unpack_fp8_weight(packed, scale), fq)


def test_grouped_int_fake_quant_and_packing():
    from libreyolo.quant.fake_quant import fake_quant_int_grouped
    from libreyolo.quant.packing import (
        pack_int_grouped_weight,
        unpack_int_grouped_weight,
    )

    torch.manual_seed(0)
    for bits, group, in_features in ((4, 128, 300), (2, 64, 100)):
        w = torch.randn(8, in_features) * 0.05
        fq = fake_quant_int_grouped(w, bits, group)
        payload, scale = pack_int_grouped_weight(w, bits, group)
        assert payload.dtype == torch.uint8
        out = unpack_int_grouped_weight(payload, scale, bits, group, in_features)
        assert torch.equal(out, fq)
        levels = torch.unique(torch.round(out / scale.unsqueeze(-1).clamp(min=1e-12)
                                          .expand(-1, -1, group)
                                          .reshape(8, -1)[:, :in_features]))
        assert levels.numel() <= 2 ** bits


def test_mxfp4_fake_quant_and_packing():
    from libreyolo.quant.fake_quant import fake_quant_mxfp4_weight
    from libreyolo.quant.packing import pack_mxfp4_weight, unpack_mxfp4_weight

    torch.manual_seed(0)
    for in_features in (64, 40):
        w = torch.randn(8, in_features) * 0.05
        fq = fake_quant_mxfp4_weight(w)
        rel = (w - fq).norm() / w.norm()
        assert rel < 0.2
        payload, exponents = pack_mxfp4_weight(w)
        assert exponents.dtype == torch.int8
        out = unpack_mxfp4_weight(payload, exponents, in_features)
        assert torch.equal(out, fq)


def test_group_and_mxfp4_linear_finalize_roundtrips():
    from libreyolo.quant import GroupQuantLinear, MXFP4Linear

    torch.manual_seed(0)
    x = torch.randn(4, 200)
    for mod in (
        GroupQuantLinear.from_float(nn.Linear(200, 32), bits=4, group_size=128),
        GroupQuantLinear.from_float(
            nn.Linear(200, 32), bits=2, group_size=64, act_int8=True
        ),
        MXFP4Linear.from_float(nn.Linear(200, 32)),
    ):
        with torch.no_grad():
            ref = mod(x)
            mod.make_finalized()
            assert mod.is_finalized
            assert torch.equal(mod(x), ref)
            mod.make_prepared()
            assert not mod.is_finalized
            assert torch.equal(mod(x), ref)


def test_new_recipe_wiring_on_models(yolo9t):
    from libreyolo.quant import QuantizationError as QErr

    # Linear-only recipes rejected on the conv-heavy family
    for recipe in ("w4a16", "w4a8", "mxfp4", "int2"):
        with pytest.raises(QErr, match="GEMM-only"):
            LibreYOLO9(None, size="t", device="cpu").quantize(recipe=recipe)

    # fp8 quantizes convs on yolo9 and round-trips through save/load
    yolo9t.quantize(recipe="fp8", calib=None, verbose=False)
    info = yolo9t.quant_info()
    assert info["recipe"] == "fp8"
    assert info["module_counts"].get("conv_fp8", 0) > 0
    yolo9t.model.eval()
    with torch.no_grad():
        out = yolo9t.model(torch.randn(1, 3, 640, 640))
    assert torch.isfinite(_leaf(out)).all()


def test_bf16_recipe_roundtrip(tmp_path):
    m = LibreYOLO9(None, size="t", device="cpu")
    m.quantize(recipe="bf16", verbose=False)
    m.model.eval()
    with torch.no_grad():
        out = m.model(torch.randn(1, 3, 640, 640))
    assert _leaf(out).dtype == torch.float32
    path = tmp_path / "LibreYOLO9t-bf16.pt"
    m.save(str(path))
    m2 = LibreYOLO9(str(path), size="t", device="cpu")
    assert m2.quant_info()["recipe"] == "bf16"
    assert next(m2.model.parameters()).dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Model-level API (fresh yolo9-t, no downloads)
# ---------------------------------------------------------------------------


@pytest.fixture()
def yolo9t():
    return LibreYOLO9(None, size="t", device="cpu")


def test_quantize_int8_swaps_and_keeps_keys(yolo9t):
    keys_before = set(yolo9t.model.state_dict().keys())
    yolo9t.quantize(recipe="int8", calib=None, verbose=False)
    info = yolo9t.quant_info()
    assert info["recipe"] == "int8"
    assert info["module_counts"]["conv_int8"] > 0
    assert keys_before <= set(yolo9t.model.state_dict().keys())
    # Head and stem stay float per the family default policy.
    for name, module in yolo9t.model.named_modules():
        if name.startswith("head.") or name.startswith("backbone.conv0."):
            assert not isinstance(module, QuantConv2d), name


def test_quantize_twice_raises(yolo9t):
    yolo9t.quantize(recipe="int8", calib=None, verbose=False)
    with pytest.raises(QuantizationError):
        yolo9t.quantize(recipe="int8", calib=None, verbose=False)


def test_yolo9_nvfp4_rejected(yolo9t):
    with pytest.raises(QuantizationError, match="GEMM-only"):
        yolo9t.quantize(recipe="nvfp4")


def test_unknown_recipe_rejected(yolo9t):
    with pytest.raises(QuantizationError, match="Unknown quantization recipe"):
        yolo9t.quantize(recipe="int3")


def test_quantized_forward_and_qat_gradients(yolo9t):
    yolo9t.quantize(recipe="int8", calib=None, verbose=False)
    yolo9t.model.train()
    out = yolo9t.model(torch.randn(1, 3, 640, 640))
    tensors = out.values() if isinstance(out, dict) else out
    loss = sum(
        t.float().abs().mean()
        for t in tensors
        if torch.is_tensor(t) and t.is_floating_point()
    )
    loss.backward()
    qmod = next(
        m for m in yolo9t.model.modules() if isinstance(m, QuantConv2d)
    )
    assert qmod.weight.grad is not None
    assert torch.isfinite(qmod.weight.grad).all()


def test_save_load_roundtrip(tmp_path, yolo9t):
    yolo9t.quantize(recipe="int8", calib=None, verbose=False)
    path = tmp_path / "LibreYOLO9t-int8.pt"
    yolo9t.save(str(path))

    reloaded = LibreYOLO9(str(path), size="t", device="cpu")
    info = reloaded.quant_info()
    assert info["recipe"] == "int8"
    n_quant = sum(1 for m in reloaded.model.modules() if isinstance(m, QuantConv2d))
    assert n_quant == info["module_count"]

    # Weights and quant buffers survive the roundtrip bit-exactly.
    src = yolo9t.model.state_dict()
    dst = reloaded.model.state_dict()
    assert set(src.keys()) == set(dst.keys())
    for key in src:
        assert torch.equal(src[key].cpu(), dst[key].cpu()), key


def test_export_rejected_for_fp16_and_wrong_formats(yolo9t):
    yolo9t.quantize(recipe="int8", calib=None, verbose=False)
    with pytest.raises(QuantizationError, match="format='onnx'"):
        yolo9t.export(format="torchscript")


def test_export_rejected_for_fp16_recipe():
    m = LibreYOLO9(None, size="t", device="cpu")
    m.quantize(recipe="fp16", verbose=False)
    with pytest.raises(QuantizationError, match="half=True"):
        m.export(format="onnx")


def test_quantized_onnx_export_emits_qdq(tmp_path, yolo9t):
    onnx = pytest.importorskip("onnx")
    yolo9t.quantize(recipe="int8", calib=None, verbose=False)
    ckpt = tmp_path / "LibreYOLO9t-int8.pt"
    yolo9t.save(str(ckpt))
    reloaded = LibreYOLO9(str(ckpt), size="t", device="cpu")
    path = reloaded.export(format="onnx", simplify=False)
    graph = onnx.load(path).graph
    n_q = sum(1 for n in graph.node if n.op_type == "QuantizeLinear")
    n_dq = sum(1 for n in graph.node if n.op_type == "DequantizeLinear")
    assert n_q > 100, f"expected weight QDQ pairs, got {n_q}"
    assert n_dq >= n_q
    # Export mode must be reset afterwards.
    q_modules = [m for m in reloaded.model.modules() if isinstance(m, QuantConv2d)]
    assert q_modules and all(not m._q_export_mode for m in q_modules)


def test_dequantize_restores_exact_float_forward(yolo9t):
    yolo9t.model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        ref = yolo9t.model(x)
    yolo9t.quantize(recipe="int8", calib=None, verbose=False)
    yolo9t.dequantize()
    assert yolo9t.quant_info() is None
    assert not any(
        isinstance(m, QuantConv2d) for m in yolo9t.model.modules()
    )
    yolo9t.model.eval()
    with torch.no_grad():
        out = yolo9t.model(x)
    ref_leaf = next(iter(ref.values())) if isinstance(ref, dict) else ref
    out_leaf = next(iter(out.values())) if isinstance(out, dict) else out
    while isinstance(ref_leaf, (tuple, list)):
        ref_leaf, out_leaf = ref_leaf[0], out_leaf[0]
    assert torch.equal(ref_leaf, out_leaf)


def test_fp16_dequantize_restores_float_dtype(yolo9t):
    yolo9t.quantize(recipe="fp16", verbose=False)
    yolo9t.dequantize()
    assert yolo9t.quant_info() is None
    assert next(yolo9t.model.parameters()).dtype == torch.float32
    yolo9t.model.eval()
    with torch.no_grad():
        out = yolo9t.model(torch.randn(1, 3, 640, 640))
    leaf = next(iter(out.values())) if isinstance(out, dict) else out
    while isinstance(leaf, (tuple, list)):
        leaf = leaf[0]
    assert leaf.dtype == torch.float32


def _leaf(out):
    leaf = next(iter(out.values())) if isinstance(out, dict) else out
    while isinstance(leaf, (tuple, list)):
        leaf = leaf[0]
    return leaf


def test_finalized_pt_export_roundtrip(tmp_path, yolo9t):
    from libreyolo.quant import reprepare_model

    yolo9t.quantize(recipe="int8", calib=None, verbose=False)
    yolo9t.model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        ref = _leaf(yolo9t.model(x))

    prepared = tmp_path / "prep.pt"
    yolo9t.save(str(prepared))
    final = yolo9t.export(format="pt", out=str(tmp_path / "final.pt"), remainder="fp32")

    import os
    assert os.path.getsize(final) < os.path.getsize(prepared)

    ckpt = torch.load(final, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    assert ckpt["quant"]["state"] == "finalized"
    assert "backbone.conv1.conv.weight_packed" in sd
    assert "backbone.conv1.conv.weight" not in sd
    assert sd["backbone.conv1.conv.weight_packed"].dtype == torch.int8

    m2 = LibreYOLO9(final, size="t", device="cpu")
    assert m2.quant_info()["state"] == "finalized"
    m2.model.eval()
    with torch.no_grad():
        out2 = _leaf(m2.model(x))
    assert torch.equal(ref, out2)

    reprepare_model(m2)
    assert m2.quant_info()["state"] == "prepared"
    m2.model.eval()
    with torch.no_grad():
        out3 = _leaf(m2.model(x))
    assert torch.equal(ref, out3)


def test_finalized_w4a16_fp16_remainder_preserves_quant_dtypes(tmp_path):
    from libreyolo.quant import (
        GroupQuantLinear,
        apply_quant_structure,
        export_finalized_pt,
        quantize_model,
        reprepare_model,
    )

    class TinyWrapper:
        FAMILY = "rfdetr"

        def __init__(self):
            self.model = nn.Sequential(nn.Linear(130, 7), nn.LayerNorm(7))
            self.device = torch.device("cpu")
            self.size = "n"
            self.task = "detect"
            self.nb_classes = 1
            self.names = {0: "class_0"}

        def _get_model_name(self):
            return "rfdetr"

        def _get_input_size(self):
            return 640

    source = TinyWrapper()
    quantize_model(
        source,
        recipe="w4a16",
        calib=None,
        keep_high_precision=(),
        verbose=False,
    )
    path = export_finalized_pt(
        source,
        out=tmp_path / "tiny-w4a16.pt",
        remainder="fp16",
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    assert state["0._q_w_gscale"].dtype == torch.float32
    assert state["0.weight_packed"].dtype == torch.uint8
    assert state["0.bias"].dtype == torch.float16
    assert state["1.weight"].dtype == torch.float16

    loaded = TinyWrapper()
    apply_quant_structure(loaded, checkpoint["quant"])
    loaded.model.load_state_dict(state)
    quant_linear = loaded.model[0]
    assert isinstance(quant_linear, GroupQuantLinear)
    assert quant_linear.is_finalized
    assert quant_linear._q_w_gscale.dtype == torch.float32
    assert quant_linear.weight_packed.dtype == torch.uint8
    assert quant_linear.bias.dtype == torch.float16
    assert loaded.model[1].weight.dtype == torch.float16

    reprepare_model(loaded)
    assert not quant_linear.is_finalized
    assert quant_linear.weight.dtype == torch.float32
    assert quant_linear.bias.dtype == torch.float32
    assert loaded.model[1].weight.dtype == torch.float32


@pytest.mark.parametrize(
    "recipe,scale_buffer,scale_dtype",
    [
        ("nvfp4", "weight_block_scale", torch.float8_e4m3fn),
        ("mxfp4", "weight_block_exp", torch.int8),
    ],
)
def test_finalized_block_recipes_fp16_remainder_preserves_scale_dtypes(
    tmp_path, recipe, scale_buffer, scale_dtype
):
    from libreyolo.quant import (
        apply_quant_structure,
        export_finalized_pt,
        quantize_model,
    )

    class TinyWrapper:
        FAMILY = "rfdetr"

        def __init__(self):
            self.model = nn.Sequential(nn.Linear(130, 7), nn.LayerNorm(7))
            self.device = torch.device("cpu")
            self.size = "n"
            self.task = "detect"
            self.nb_classes = 1
            self.names = {0: "class_0"}

        def _get_model_name(self):
            return "rfdetr"

        def _get_input_size(self):
            return 640

    source = TinyWrapper()
    quantize_model(
        source,
        recipe=recipe,
        calib=None,
        keep_high_precision=(),
        verbose=False,
    )
    path = export_finalized_pt(
        source,
        out=tmp_path / f"tiny-{recipe}.pt",
        remainder="fp16",
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    assert state[f"0.{scale_buffer}"].dtype == scale_dtype
    assert state["0.weight_packed"].dtype == torch.uint8
    assert state["1.weight"].dtype == torch.float16

    loaded = TinyWrapper()
    apply_quant_structure(loaded, checkpoint["quant"])
    loaded.model.load_state_dict(state)
    assert loaded.model[0].is_finalized
    assert getattr(loaded.model[0], scale_buffer).dtype == scale_dtype
    assert loaded.model[0].weight_packed.dtype == torch.uint8


def test_fp16_roundtrip_keeps_float32_io(tmp_path, yolo9t):
    yolo9t.quantize(recipe="fp16", verbose=False)
    yolo9t.model.eval()
    with torch.no_grad():
        out = yolo9t.model(torch.randn(1, 3, 640, 640))
    leaf = next(iter(out.values())) if isinstance(out, dict) else out
    while isinstance(leaf, (tuple, list)):
        leaf = leaf[0]
    assert leaf.dtype == torch.float32

    path = tmp_path / "LibreYOLO9t-fp16.pt"
    yolo9t.save(str(path))
    reloaded = LibreYOLO9(str(path), size="t", device="cpu")
    assert reloaded.quant_info()["recipe"] == "fp16"
    assert next(reloaded.model.parameters()).dtype == torch.float16


# ---------------------------------------------------------------------------
# birefnet (matte) family
# ---------------------------------------------------------------------------


@pytest.fixture()
def birefnet_t():
    from libreyolo.models.birefnet.model import LibreBiRefNet

    return LibreBiRefNet(None, size="t", device="cpu")


def test_birefnet_nvfp4_swaps_linears_and_respects_keeps(birefnet_t):
    birefnet_t.quantize(recipe="nvfp4", verbose=False)
    quant_names = [
        n for n, m in birefnet_t.model.named_modules() if isinstance(m, NVFP4Linear)
    ]
    assert quant_names, "expected Swin Linear layers to be swapped to NVFP4Linear"
    for pattern in ("patch_embed", "conv_out1", "gdt_convs_attn"):
        assert not any(pattern in n for n in quant_names), pattern
    info = birefnet_t.quant_info()
    assert info["recipe"] == "nvfp4"
    assert info["execution"] == "simulated"


def test_birefnet_fp8_swaps_convs_and_linears(birefnet_t):
    birefnet_t.quantize(recipe="fp8", calib=None, verbose=False)
    kinds = {
        type(m).__name__
        for m in birefnet_t.model.modules()
        if isinstance(m, (QuantConv2d,)) or type(m).__name__ == "QuantLinear"
    }
    assert kinds, "expected fp8 quant modules on both Conv2d and Linear"


def test_birefnet_int2_rejected(birefnet_t):
    with pytest.raises(QuantizationError, match="inference-only"):
        birefnet_t.quantize(recipe="int2", verbose=False)


def test_feynobg_supported_and_int2_rejected():
    # feynobg shares the birefnet architecture; the gate check is family-level
    # and cheap to exercise without instantiating the 263M model.
    from libreyolo.quant.api import SUPPORTED_FAMILIES, _check_support

    assert "feynobg" in SUPPORTED_FAMILIES
    _check_support("feynobg", "fp8")
    _check_support("feynobg", "nvfp4")
    with pytest.raises(QuantizationError, match="inference-only"):
        _check_support("feynobg", "int2")


def test_birefnet_nvfp4_save_load_roundtrip(tmp_path, birefnet_t):
    from libreyolo import LibreYOLO

    birefnet_t.quantize(recipe="nvfp4", verbose=False)
    path = tmp_path / "LibreBiRefNett-matte-nvfp4.pt"
    birefnet_t.save(str(path))

    # The real user path: metadata dispatch picks family+size, then the quant
    # manifest rebuilds the quantized structure before the state dict loads.
    reloaded = LibreYOLO(str(path), device="cpu")
    info = reloaded.quant_info()
    assert info["recipe"] == "nvfp4"
    assert reloaded.size == "t"
    assert reloaded.task == "matte"

    src = birefnet_t.model.state_dict()
    dst = reloaded.model.state_dict()
    assert set(src.keys()) == set(dst.keys())
    for key in src:
        assert torch.equal(src[key].cpu(), dst[key].cpu()), key
