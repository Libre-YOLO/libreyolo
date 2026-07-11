"""Unit tests for the ZipDepth family (no external weights needed)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from libreyolo.models.depth_anything.model import LibreDepthAnythingV2
from libreyolo.models.zipdepth.model import LibreZipDepth
from libreyolo.models.zipdepth.nn import LibreZipDepthNet
from libreyolo.models.zipdepth.utils import (
    compute_input_hw,
    make_divisible,
    preprocess_numpy,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Checkpoint detection
# ---------------------------------------------------------------------------


def _state_dict(size: str) -> dict:
    return LibreZipDepthNet(size=size).state_dict()


def test_can_load_own_state_dicts():
    assert LibreZipDepth.can_load(_state_dict("b")) is True
    assert LibreZipDepth.can_load(_state_dict("bnpu")) is True


def test_can_load_rejects_depth_anything_layout():
    fake_da2 = {
        "pretrained.cls_token": torch.zeros(1, 1, 384),
        "depth_head.projects.0.weight": torch.zeros(48, 384, 1, 1),
    }
    assert LibreZipDepth.can_load(fake_da2) is False


def test_depth_anything_rejects_zipdepth_state_dict():
    assert LibreDepthAnythingV2.can_load(_state_dict("b")) is False


def test_can_load_raises_on_fused_state_dict():
    net = LibreZipDepthNet(size="b")
    net.fuse_for_inference()
    with pytest.raises(ValueError, match="fused"):
        LibreZipDepth.can_load(net.state_dict())


def test_detect_size_discriminates_decoder_flavor():
    assert LibreZipDepth.detect_size(_state_dict("b")) == "b"
    assert LibreZipDepth.detect_size(_state_dict("bnpu")) == "bnpu"


def test_detect_size_from_filename():
    assert LibreZipDepth.detect_size_from_filename("LibreZipDepthb-depth.pt") == "b"
    assert (
        LibreZipDepth.detect_size_from_filename("LibreZipDepthbnpu-depth.pt") == "bnpu"
    )
    assert (
        LibreZipDepth.detect_size_from_filename("LibreDepthAnythingV2s-depth.pt")
        is None
    )


# ---------------------------------------------------------------------------
# Preprocessing geometry
# ---------------------------------------------------------------------------


def test_make_divisible_matches_upstream_rounding():
    # int(round(x / 32) * 32) with a floor of 32, including banker's rounding.
    assert make_divisible(384) == 384
    assert make_divisible(400) == 384  # 400/32 = 12.5, round-half-even -> 12
    assert make_divisible(432) == 448  # 432/32 = 13.5, round-half-even -> 14
    assert make_divisible(1) == 32


def test_compute_input_hw_keeps_short_side_near_input_size():
    h, w = compute_input_hw(480, 640, 384)
    assert h % 32 == 0 and w % 32 == 0
    assert h == 384 and w == 512


def test_preprocess_numpy_shape_and_range():
    img = np.random.randint(0, 256, (333, 500, 3), dtype=np.uint8)
    chw, ratio = preprocess_numpy(img, 384)
    assert ratio == 1.0
    assert chw.dtype == np.float32
    assert chw.shape[0] == 3
    assert chw.shape[1] % 32 == 0 and chw.shape[2] % 32 == 0
    assert 0.0 <= chw.min() and chw.max() <= 1.0


def test_preprocess_rejects_bad_imgsz():
    model = LibreZipDepth(None, size="b", device="cpu")
    with pytest.raises(ValueError, match="divisible"):
        model._preprocess(np.zeros((64, 64, 3), dtype=np.uint8), input_size=100)


# ---------------------------------------------------------------------------
# Forward + postprocess contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", ["b", "bnpu"])
def test_forward_returns_full_resolution_map(size):
    net = LibreZipDepthNet(size=size).eval()
    with torch.no_grad():
        out = net(torch.rand(1, 3, 96, 128))
    assert out.shape == (1, 1, 96, 128)
    assert float(out.min()) >= 0.0


def test_postprocess_resizes_to_original_canvas():
    model = LibreZipDepth(None, size="b", device="cpu")
    raw = torch.rand(1, 1, 96, 128)
    parsed = model._postprocess(
        raw, conf_thres=0.25, iou_thres=0.45, original_size=(200, 150)
    )
    assert set(parsed) == {"depth"}
    assert tuple(parsed["depth"].shape) == (150, 200)


def test_fuse_is_idempotent_and_close():
    net = LibreZipDepthNet(size="b").eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        ref = net(x)
    net.fuse_for_inference()
    net.fuse_for_inference()  # idempotent
    with torch.no_grad():
        fused = net(x)
    assert torch.allclose(ref, fused, atol=1e-5)


def test_export_rejects_batch_gt_1():
    # The exported batch axis is static (dynamic forced off) and backends
    # schedule one image per run, so batch != 1 artifacts are unusable.
    model = LibreZipDepth(None, size="b", device="cpu")
    with pytest.raises(ValueError, match="batch-1"):
        model.export(format="onnx", batch=2)


@pytest.mark.parametrize("format", ["onnx", "torchscript", "ncnn"])
def test_exported_depth_parity(tmp_path, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "ncnn" and (
        importlib.util.find_spec("pnnx") is None
        or importlib.util.find_spec("ncnn") is None
    ):
        pytest.skip("PNNX and NCNN are required")
    from libreyolo import LibreYOLO

    torch.manual_seed(0)
    model = LibreZipDepth(None, size="b", device="cpu")
    model.model.eval()
    image = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    native = model.predict(image, imgsz=64).depth_map.data.numpy()
    artifact = model.export(
        format=format,
        output_path=str(tmp_path / f"zipdepth.{format}"),
        imgsz=64,
        dynamic=False,
        simplify=False,
    )
    actual = LibreYOLO(artifact, device="cpu").predict(image).depth_map.data.numpy()

    mse = float(np.mean((native - actual) ** 2))
    peak = max(float(np.max(np.abs(native))), 1e-6)
    psnr = float("inf") if mse == 0 else 20.0 * np.log10(peak / np.sqrt(mse))
    assert psnr > 40.0


def test_metadata_contract():
    model = LibreZipDepth(None, size="b", device="cpu")
    assert model.task == "depth"
    assert model.nb_classes == 1
    assert model.names == {0: "depth"}
    assert model.depth_imgsz_divisor == 32
