from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo.backends.base import BaseBackend
from libreyolo.data.restore_dataset import RestoreDataset, restore_collate_fn
from libreyolo.models.nafnet import LibreNAFNet
from libreyolo.models.nafnet.utils import preprocess_image
from libreyolo.postprocess.nafnet import postprocess as nafnet_postprocess
from libreyolo.tasks import normalize_task, task_to_suffix
from libreyolo.utils.results import Results, RestoredImage
from libreyolo.validation.restore_validator import psnr_rgb, ssim_rgb


pytestmark = pytest.mark.unit


def test_restored_image_array_save_and_results_summary(tmp_path):
    arr = np.zeros((4, 5, 3), dtype=np.uint8)
    arr[..., 0] = 255
    restored = RestoredImage(torch.from_numpy(arr))
    result = Results(
        boxes=None,
        orig_shape=(4, 5),
        names={0: "image"},
        restored=restored,
    )

    assert result.restored.array.dtype == np.uint8
    assert result.restored.array.shape == (4, 5, 3)
    assert len(result) == 1
    assert result.summary() == [{"name": "restored", "shape": [4, 5, 3]}]
    assert np.array_equal(result[0].restored.array, arr)

    out = tmp_path / "restored.png"
    result.restored.save(out)
    with Image.open(out) as im:
        assert im.mode == "RGB"
        assert im.size == (5, 4)


def test_nafnet_postprocess_crops_to_original_hwc_uint8():
    output = torch.zeros(1, 3, 8, 8)
    output[:, 1] = 0.5
    restored = nafnet_postprocess(output, original_size=(5, 3))

    assert restored.shape == (3, 5, 3)
    assert restored.dtype == np.uint8
    assert np.all(restored[..., 1] == 128)


def test_nafnet_preprocess_keeps_native_resolution_and_pads_to_stride():
    img = Image.fromarray(np.zeros((17, 19, 3), dtype=np.uint8), mode="RGB")
    tensor, _, original_size, ratio = preprocess_image(img, pad_multiple=16)

    assert original_size == (19, 17)
    assert ratio == 1.0
    assert tuple(tensor.shape) == (1, 3, 32, 32)


def test_librenafnet_predict_returns_restored_original_shape():
    model = LibreNAFNet(model_path=None, size="s", device="cpu")
    model.model.eval()
    img = Image.fromarray(np.zeros((5, 7, 3), dtype=np.uint8), mode="RGB")

    result = model(img)

    assert result.boxes is None
    assert result.restored is not None
    assert result.restored.array.shape == (5, 7, 3)
    assert result.restored.array.dtype == np.uint8


def test_exported_backend_restore_preprocess_uses_fixed_canvas_without_resize():
    img = Image.fromarray(np.full((5, 7, 3), 13, dtype=np.uint8), mode="RGB")
    tensor, _, original_size, ratio = BaseBackend._preprocess_restore(
        img,
        input_size=(16, 16),
        color_format="rgb",
    )

    assert original_size == (7, 5)
    assert ratio == 1.0
    assert tuple(tensor.shape) == (1, 3, 16, 16)
    assert torch.allclose(tensor[:, :, :5, :7], torch.full((1, 3, 5, 7), 13 / 255))


def test_exported_backend_restore_preprocess_rejects_images_larger_than_canvas():
    img = Image.fromarray(np.zeros((17, 16, 3), dtype=np.uint8), mode="RGB")
    with pytest.raises(ValueError, match="fixed-resolution"):
        BaseBackend._preprocess_restore(img, input_size=(16, 16), color_format="rgb")


def test_nafnet_fixed_onnx_export_roundtrip(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from libreyolo.backends.onnx import OnnxBackend

    export_path = tmp_path / "nafnet_restore.onnx"
    model = LibreNAFNet(model_path=None, size="s", device="cpu")
    model.model.eval()

    path = model.export(
        format="onnx",
        output_path=str(export_path),
        imgsz=16,
        dynamic=True,
        simplify=False,
        opset=13,
    )
    backend = OnnxBackend(path, device="cpu")
    result = backend(Image.fromarray(np.zeros((5, 7, 3), dtype=np.uint8), mode="RGB"))

    assert result.boxes is None
    assert result.restored.array.shape == (5, 7, 3)
    assert result.restored.array.dtype == np.uint8


def test_restore_task_aliases_and_filename_suffix():
    assert normalize_task("deblur") == "restore"
    assert normalize_task("denoise") == "restore"
    assert task_to_suffix("restore") == "restore"
    assert LibreNAFNet.detect_size_from_filename("LibreNAFNets-restore.pt") == "s"
    assert LibreNAFNet.detect_task_from_filename("LibreNAFNetl-restore.pt") == "restore"
    assert LibreNAFNet.detect_task_from_filename("LibreNAFNetl.pt") is None


def test_libreyolo_factory_loads_metadata_wrapped_nafnet_checkpoint(tmp_path):
    from libreyolo.models import LibreYOLO
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    source = LibreNAFNet(model_path=None, size="s", device="cpu")
    checkpoint = wrap_libreyolo_checkpoint(
        source.model.state_dict(),
        model_family="nafnet",
        size="s",
        task="restore",
        nc=1,
        names={0: "image"},
        imgsz=256,
    )
    path = tmp_path / "LibreNAFNets-restore.pt"
    torch.save(checkpoint, path)

    loaded = LibreYOLO(str(path), device="cpu")

    assert isinstance(loaded, LibreNAFNet)
    assert loaded.size == "s"
    assert loaded.task == "restore"
    assert loaded.nb_classes == 1
    assert loaded.names == {0: "image"}


def test_librenafnet_train_runs_tiny_restore_smoke(tmp_path):
    root = tmp_path / "restore"
    for split in ("train", "val"):
        input_dir = root / "inputs" / split
        target_dir = root / "targets" / split
        input_dir.mkdir(parents=True)
        target_dir.mkdir(parents=True)
        base = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        Image.fromarray(base, mode="RGB").save(input_dir / "sample.png")
        Image.fromarray(255 - base, mode="RGB").save(target_dir / "sample.png")

    cfg = {
        "train": str(root / "inputs" / "train"),
        "val": str(root / "inputs" / "val"),
        "input_dir": "inputs",
        "target_dir": "targets",
        "nc": 1,
        "names": {0: "image"},
    }
    yaml_path = root / "restore.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    model = LibreNAFNet(model_path=None, size="s", device="cpu")
    results = model.train(
        data=str(yaml_path),
        epochs=1,
        batch=1,
        imgsz=4,
        workers=0,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="restore_smoke",
        exist_ok=True,
        amp=False,
        ema=False,
        eval_interval=0,
        save_period=0,
        patience=1,
        degradation="deblur",
        dataset="tiny",
    )

    assert math.isfinite(results["final_loss"])
    last_checkpoint = Path(results["last_checkpoint"])
    assert last_checkpoint.exists()
    checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["model_family"] == "nafnet"
    assert checkpoint["task"] == "restore"
    assert checkpoint["degradation"] == "deblur"
    assert checkpoint["dataset"] == "tiny"


def test_restore_metrics_are_perfect_for_identical_rgb():
    target = torch.rand(3, 16, 16)
    # A perfect reconstruction (mse == 0) is capped at a large finite PSNR
    # rather than +inf, so a single identity image can't poison the aggregate
    # PSNR / fitness used for best-checkpoint selection.
    assert psnr_rgb(target, target) == pytest.approx(100.0)
    assert ssim_rgb(target, target) == pytest.approx(1.0)


def test_restore_psnr_protocol_is_rgb_unit_range_no_border_crop():
    target = torch.zeros(3, 8, 8)
    pred = torch.full_like(target, 0.5)

    assert psnr_rgb(pred, target) == pytest.approx(6.0205999, rel=1e-6)


def test_restore_dataset_pairs_targets_and_applies_coupled_augmentation(tmp_path):
    root = tmp_path / "restore"
    input_dir = root / "inputs" / "train"
    target_dir = root / "targets" / "train"
    input_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)

    base = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    target = 255 - base
    Image.fromarray(base, mode="RGB").save(input_dir / "sample.png")
    Image.fromarray(target, mode="RGB").save(target_dir / "sample.png")

    cfg = {
        "train": str(input_dir),
        "input_dir": "inputs",
        "target_dir": "targets",
        "nc": 1,
        "names": {0: "image"},
    }
    random.seed(0)
    dataset = RestoreDataset(cfg, split="train", imgsz=4, augment=True)
    inp, tgt, info, idx = dataset[0]

    assert idx == 0
    assert info["orig_shape"] == (8, 8)
    assert inp.shape == tgt.shape == (3, 4, 4)
    assert torch.allclose(inp + tgt, torch.ones_like(inp), atol=1 / 255)

    batch = restore_collate_fn([dataset[0], dataset[0]])
    assert batch[0].shape[0] == 2
    assert batch[1].shape == batch[0].shape


def _make_restore_pair(root: Path, imgsz: int):
    """A square input/target pair where target = 255 - input, all pixels distinct."""
    input_dir = root / "inputs" / "train"
    target_dir = root / "targets" / "train"
    input_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    base = (np.arange(imgsz * imgsz * 3) % 256).astype(np.uint8).reshape(
        imgsz, imgsz, 3
    )
    target = (255 - base).astype(np.uint8)
    Image.fromarray(base, mode="RGB").save(input_dir / "sample.png")
    Image.fromarray(target, mode="RGB").save(target_dir / "sample.png")
    cfg = {
        "train": str(input_dir),
        "input_dir": "inputs",
        "target_dir": "targets",
        "nc": 1,
        "names": {0: "image"},
    }
    return cfg, base, target


def _restore_expected(base, target, seed, imgsz):
    """Replay the _augment_pair RNG sequence to predict the coupled transform."""
    random.seed(seed)
    top = random.randint(0, max(0, imgsz - imgsz))
    left = random.randint(0, max(0, imgsz - imgsz))
    inp = base[top : top + imgsz, left : left + imgsz]
    tgt = target[top : top + imgsz, left : left + imgsz]
    if random.random() < 0.5:
        inp, tgt = inp[:, ::-1], tgt[:, ::-1]
    did_vflip = random.random() < 0.5
    if did_vflip:
        inp, tgt = inp[::-1, :], tgt[::-1, :]
    k = random.randint(0, 3)
    if k:
        inp, tgt = np.rot90(inp, k), np.rot90(tgt, k)
    return np.ascontiguousarray(inp), np.ascontiguousarray(tgt), did_vflip, k


def test_restore_paired_ops_keep_alignment_and_are_seed_deterministic(tmp_path):
    imgsz = 4
    cfg, base, target = _make_restore_pair(tmp_path / "restore", imgsz)
    dataset = RestoreDataset(cfg, split="train", imgsz=imgsz, augment=True)

    # Determinism: same seed -> byte-identical tensors.
    random.seed(123)
    inp_a, tgt_a, _, _ = dataset[0]
    random.seed(123)
    inp_b, tgt_b, _, _ = dataset[0]
    assert torch.equal(inp_a, inp_b)
    assert torch.equal(tgt_a, tgt_b)

    # Alignment: target = 255 - input must survive every coupled op, so the
    # summed pair stays all-ones regardless of the sampled geometry.
    for seed in range(20):
        random.seed(seed)
        inp, tgt, _, _ = dataset[0]
        assert inp.shape == tgt.shape == (3, imgsz, imgsz)
        assert torch.allclose(inp + tgt, torch.ones_like(inp), atol=1 / 255)


def test_restore_vflip_and_rot90_actually_occur_and_match_prediction(tmp_path):
    imgsz = 4
    cfg, base, target = _make_restore_pair(tmp_path / "restore", imgsz)
    dataset = RestoreDataset(cfg, split="train", imgsz=imgsz, augment=True)

    # Pick a seed that triggers both a vertical flip and a non-zero rot90.
    seed = next(
        s
        for s in range(1000)
        if (lambda r: r[2] and r[3] != 0)(_restore_expected(base, target, s, imgsz))
    )
    exp_inp, exp_tgt, did_vflip, k = _restore_expected(base, target, seed, imgsz)
    assert did_vflip and k != 0

    random.seed(seed)
    inp, tgt, _, _ = dataset[0]
    assert torch.equal(inp, torch.from_numpy(exp_inp).permute(2, 0, 1).float().div(255.0))
    assert torch.equal(tgt, torch.from_numpy(exp_tgt).permute(2, 0, 1).float().div(255.0))


def test_restore_augment_false_is_native_and_untransformed(tmp_path):
    imgsz = 8
    cfg, base, target = _make_restore_pair(tmp_path / "restore", imgsz)
    dataset = RestoreDataset(cfg, split="train", imgsz=4, augment=False)

    inp, tgt, info, _ = dataset[0]
    # augment=False keeps native resolution and applies no geometry.
    assert inp.shape == (3, imgsz, imgsz)
    expected = torch.from_numpy(base).permute(2, 0, 1).float().div(255.0)
    assert torch.equal(inp, expected)
    assert info["orig_shape"] == (imgsz, imgsz)
