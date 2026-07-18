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
