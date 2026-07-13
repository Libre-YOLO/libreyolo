"""Unit tests for the LibreRealESRGAN super-resolution family."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo.data.restore_dataset import RestoreDataset, restore_collate_fn
from libreyolo.models.nafnet import LibreNAFNet
from libreyolo.models.realesrgan import LibreRealESRGAN
from libreyolo.models.realesrgan.nn import RRDBNet, SRVGGNetCompact
from libreyolo.postprocess.realesrgan import postprocess as sr_postprocess
from libreyolo.tasks import normalize_task

pytestmark = pytest.mark.unit


def _random_image(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


# --------------------------------------------------------------------------- #
# task aliases + filename/metadata detection
# --------------------------------------------------------------------------- #
def test_super_resolution_task_aliases():
    for alias in ("sr", "super-resolution", "super_resolution", "upscale"):
        assert normalize_task(alias) == "restore"


def test_filename_and_size_detection():
    assert (
        LibreRealESRGAN.detect_size_from_filename("LibreRealESRGANx4-restore.pt")
        == "x4"
    )
    assert (
        LibreRealESRGAN.detect_size_from_filename("LibreRealESRGANx2-restore.pt")
        == "x2"
    )
    assert (
        LibreRealESRGAN.detect_size_from_filename("LibreRealESRGANx4t-restore.pt")
        == "x4t"
    )
    assert (
        LibreRealESRGAN.detect_task_from_filename("LibreRealESRGANx4-restore.pt")
        == "restore"
    )


@pytest.mark.parametrize("size,scale", [("x4", 4), ("x2", 2), ("x4t", 4)])
def test_restore_scale_property(size, scale):
    m = LibreRealESRGAN(size=size, device="cpu")
    assert m.restore_scale == scale
    assert m.task == "restore"
    assert m.names == {0: "image"}


# --------------------------------------------------------------------------- #
# can_load discriminators (bidirectional vs NAFNet, and RRDBNet vs SRVGG)
# --------------------------------------------------------------------------- #
def test_can_load_detects_own_state_dicts_and_sizes():
    for size, expected in (("x4", "x4"), ("x2", "x2"), ("x4t", "x4t")):
        sd = LibreRealESRGAN(size=size, device="cpu").model.state_dict()
        assert LibreRealESRGAN.can_load(sd) is True
        assert LibreRealESRGAN.detect_size(sd) == expected


def test_can_load_bidirectional_rejection_with_nafnet():
    sr_sd = LibreRealESRGAN(size="x4", device="cpu").model.state_dict()
    naf_sd = LibreNAFNet(size="s", device="cpu").model.state_dict()
    assert LibreRealESRGAN.can_load(naf_sd) is False
    assert LibreNAFNet.can_load(sr_sd) is False


# --------------------------------------------------------------------------- #
# forward + postprocess shapes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size,scale", [("x4", 4), ("x2", 2), ("x4t", 4)])
def test_forward_upscales_by_scale(size, scale):
    m = LibreRealESRGAN(size=size, device="cpu")
    x = torch.zeros(1, 3, 16, 24)
    with torch.no_grad():
        y = m.model(x)
    assert tuple(y.shape) == (1, 3, 16 * scale, 24 * scale)


def test_postprocess_crops_to_scale_times_original():
    out = torch.zeros(1, 3, 32, 32)
    out[:, 1] = 0.5
    restored = sr_postprocess(out, original_size=(5, 3), scale=4)  # (W, H) = (5, 3)
    assert restored.shape == (12, 20, 3)  # 3*4, 5*4
    assert restored.dtype == np.uint8
    assert np.all(restored[..., 1] == 128)


# --------------------------------------------------------------------------- #
# predict path: restored shape == scale x input, restore_scale threaded
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size,scale", [("x4", 4), ("x2", 2), ("x4t", 4)])
def test_predict_restored_shape_and_scale(size, scale):
    m = LibreRealESRGAN(size=size, device="cpu")
    img = _random_image(20, 28, seed=1)
    r = m.predict(img)
    assert r.restore_scale == scale
    assert r.restored.array.shape == (20 * scale, 28 * scale, 3)
    assert r.restored.array.dtype == np.uint8
    assert r.summary() == [
        {"name": "restored", "shape": [20 * scale, 28 * scale, 3], "scale": scale}
    ]


def test_predict_tiling_produces_correct_shape():
    m = LibreRealESRGAN(size="x4t", device="cpu")
    img = _random_image(40, 40, seed=2)
    untiled = m.predict(img).restored.array
    tiled = m.predict(img, tile=16, tile_pad=8).restored.array
    assert tiled.shape == untiled.shape == (160, 160, 3)


def test_train_raises_not_implemented():
    m = LibreRealESRGAN(size="x4", device="cpu")
    with pytest.raises(NotImplementedError):
        m.train(data="whatever.yaml")


# --------------------------------------------------------------------------- #
# factory round-trip with v1.0 metadata (incl. scale)
# --------------------------------------------------------------------------- #
def test_factory_loads_metadata_wrapped_checkpoint_with_scale(tmp_path):
    from libreyolo.models import LibreYOLO
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    src = LibreRealESRGAN(model_path=None, size="x4", device="cpu")
    checkpoint = wrap_libreyolo_checkpoint(
        src.model.state_dict(),
        model_family="realesrgan",
        size="x4",
        task="restore",
        nc=1,
        names={0: "image"},
        imgsz=64,
        scale=4,
    )
    path = tmp_path / "LibreRealESRGANx4-restore.pt"
    torch.save(checkpoint, path)

    loaded = LibreYOLO(str(path), device="cpu")
    assert isinstance(loaded, LibreRealESRGAN)
    assert loaded.size == "x4"
    assert loaded.task == "restore"
    assert loaded.restore_scale == 4


# --------------------------------------------------------------------------- #
# scale-aware validation: passes on HR pairs, errors on shape mismatch
# --------------------------------------------------------------------------- #
def _write_pairs(root, split, lr_hw, hr_hw, n=3):
    in_dir = root / "inputs" / split
    tgt_dir = root / "targets" / split
    in_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _random_image(*lr_hw, seed=i).save(in_dir / f"{i}.png")
        _random_image(*hr_hw, seed=100 + i).save(tgt_dir / f"{i}.png")
    cfg = {
        "path": str(root),
        "train": str(root / "inputs" / split),
        "val": str(root / "inputs" / split),
        "input_dir": "inputs",
        "target_dir": "targets",
        "nc": 1,
        "names": {0: "image"},
    }
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg))
    return yaml_path


def test_restore_dataset_scale_pairs_and_mismatch(tmp_path):
    # Correct x4 pairs: LR 8x10, HR 32x40.
    root = tmp_path / "ok"
    _write_pairs(root, "val", (8, 10), (32, 40))
    ds = RestoreDataset(
        {
            "val": str(root / "inputs" / "val"),
            "input_dir": "inputs",
            "target_dir": "targets",
        },
        split="val",
        imgsz=64,
        augment=False,
        scale=4,
    )
    inp, target, info, _ = ds[0]
    assert inp.shape[-2:] == (8, 10)
    assert target.shape[-2:] == (32, 40)
    assert info["target_shape"] == (32, 40)

    # Mismatched pairs (target not scale x input) must error clearly.
    bad = tmp_path / "bad"
    _write_pairs(bad, "val", (16, 16), (16, 16))
    ds_bad = RestoreDataset(
        {
            "val": str(bad / "inputs" / "val"),
            "input_dir": "inputs",
            "target_dir": "targets",
        },
        split="val",
        imgsz=64,
        augment=False,
        scale=4,
    )
    with pytest.raises(ValueError, match="scale"):
        _ = ds_bad[0]


def test_restore_collate_pads_inputs_and_targets_separately():
    inp = [torch.zeros(3, 8, 10), torch.zeros(3, 6, 12)]
    tgt = [torch.zeros(3, 32, 40), torch.zeros(3, 24, 48)]
    batch = [
        (inp[0], tgt[0], {"orig_shape": (8, 10), "target_shape": (32, 40)}, 0),
        (inp[1], tgt[1], {"orig_shape": (6, 12), "target_shape": (24, 48)}, 1),
    ]
    imgs, targets, _, _ = restore_collate_fn(batch)
    assert imgs.shape[-2:] == (8, 12)  # input max
    assert targets.shape[-2:] == (32, 48)  # target max (independent)


def test_x2_validation_pads_odd_input_and_crops_to_target_canvas():
    from libreyolo.validation.restore_validator import RestoreValidator

    model = LibreRealESRGAN(size="x2", device="cpu")
    validator = object.__new__(RestoreValidator)
    validator.model = model
    images = torch.rand(1, 3, 3, 5)
    targets = torch.rand(1, 3, 6, 10)
    batch = (
        images,
        targets,
        [{"orig_shape": (3, 5), "target_shape": (6, 10)}],
        [0],
    )

    padded, _, _, _ = validator._preprocess_batch(batch)
    with torch.no_grad():
        output = model._forward(padded)
    restored = validator._postprocess_predictions(output, batch)

    assert padded.shape[-2:] == (4, 6)
    assert output.shape[-2:] == (8, 12)
    assert restored.shape[-2:] == (6, 10)


def test_val_runs_on_correct_hr_pairs(tmp_path):
    root = tmp_path / "srval"
    yaml_path = _write_pairs(root, "val", (8, 8), (32, 32))
    m = LibreRealESRGAN(size="x4", device="cpu")
    metrics = m.val(data=str(yaml_path), split="val", batch=1, workers=0, device="cpu")
    assert "metrics/PSNR" in metrics
    assert np.isfinite(metrics["metrics/PSNR"])


# --------------------------------------------------------------------------- #
# ONNX dynamic export round-trip
# --------------------------------------------------------------------------- #
def test_onnx_dynamic_export_roundtrip(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from libreyolo.backends.onnx import OnnxBackend

    m = LibreRealESRGAN(size="x4t", device="cpu")
    onnx_path = tmp_path / "sr.onnx"
    m.export(format="onnx", imgsz=64, dynamic=True, output_path=str(onnx_path))
    backend = OnnxBackend(str(onnx_path))
    img = _random_image(24, 32, seed=5)  # different from traced 64x64
    native = m.predict(img).restored.array
    onnx_out = backend.predict(img).restored.array
    assert onnx_out.shape == native.shape == (96, 128, 3)
    assert int(np.abs(native.astype(int) - onnx_out.astype(int)).max()) <= 1


@pytest.mark.parametrize("format", ["torchscript", "ncnn", "tflite"])
def test_fixed_export_roundtrip(tmp_path, format):
    if format == "ncnn":
        pytest.importorskip("pnnx")
        pytest.importorskip("ncnn")
    if format == "tflite":
        pytest.importorskip("onnx2tf")
        pytest.importorskip("ai_edge_litert")
    from libreyolo import LibreYOLO

    model = LibreRealESRGAN(size="x4t", device="cpu")
    image = _random_image(16, 16, seed=6)
    native = model.predict(image).restored.array
    artifact = tmp_path / f"realesrgan.{format}"
    model.export(
        format=format,
        imgsz=16,
        dynamic=False,
        output_path=str(artifact),
    )
    exported = LibreYOLO(str(artifact), device="cpu").predict(image).restored.array

    assert exported.shape == native.shape == (64, 64, 3)
    assert int(np.abs(native.astype(int) - exported.astype(int)).max()) <= 1


# --------------------------------------------------------------------------- #
# gated tensor parity against upstream (max_abs_diff == 0)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.environ.get("REALESRGAN_OFFICIAL_CKPT_DIR"),
    reason="Set REALESRGAN_OFFICIAL_CKPT_DIR to run tensor parity vs upstream",
)
def test_tensor_parity_against_upstream():
    from basicsr.archs.rrdbnet_arch import RRDBNet as UpRRDBNet  # type: ignore
    from basicsr.archs.srvgg_arch import SRVGGNetCompact as UpSRVGG  # type: ignore

    d = os.environ["REALESRGAN_OFFICIAL_CKPT_DIR"]

    def sd(path):
        ck = torch.load(os.path.join(d, path), map_location="cpu", weights_only=True)
        return ck["params_ema"] if "params_ema" in ck else ck["params"]

    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    cases = [
        (
            UpRRDBNet(3, 3, scale=4, num_feat=64, num_block=23, num_grow_ch=32),
            RRDBNet(3, 3, scale=4, num_feat=64, num_block=23, num_grow_ch=32),
            "RealESRGAN_x4plus.pth",
        ),
        (
            UpRRDBNet(3, 3, scale=2, num_feat=64, num_block=23, num_grow_ch=32),
            RRDBNet(3, 3, scale=2, num_feat=64, num_block=23, num_grow_ch=32),
            "RealESRGAN_x2plus.pth",
        ),
        (
            UpSRVGG(3, 3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"),
            SRVGGNetCompact(
                3, 3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"
            ),
            "realesr-general-x4v3.pth",
        ),
    ]
    for up, mine, fname in cases:
        state = sd(fname)
        up.load_state_dict(state, strict=True)
        mine.load_state_dict(state, strict=True)
        up.eval()
        mine.eval()
        with torch.no_grad():
            diff = (up(x) - mine(x)).abs().max().item()
        assert diff == 0.0, f"{fname}: max_abs_diff={diff}"
