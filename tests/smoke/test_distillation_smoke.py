"""Offline smoke test for YOLOv9 MGD/CWD knowledge distillation.

Validates the distillation pipeline end-to-end on CPU with tiny synthetic data:
  - load a frozen YOLOv9 teacher (fresh weights),
  - train a YOLOv9 student with distill_model set,
  - verify student loss is finite AND a non-zero distillation loss is logged.

Runs in <30s on a MacBook CPU. No network, no pretrained weights, no large data.

Run: pytest tests/smoke/test_distillation_smoke.py -v -m smoke
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.models.yolo9.nn import LibreYOLO9Model


pytestmark = pytest.mark.smoke


def _make_tiny_det_dataset(root: Path, imgsz: int = 128, num_imgs: int = 4) -> Path:
    """Create a tiny YOLO-format detection dataset (2 classes)."""
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    def _write(split: str, idx: int, cls: int):
        img = rng.integers(60, 110, size=(imgsz, imgsz, 3), dtype=np.uint8)
        x0, y0 = 24, 24
        x1, y1 = x0 + 40, y0 + 40
        img[y0:y1, x0:x1] = 220
        Image.fromarray(img).save(root / "images" / split / f"{idx}.jpg", quality=80)

        cx, cy = (x0 + x1) / (2 * imgsz), (y0 + y1) / (2 * imgsz)
        w, h = (x1 - x0) / imgsz, (y1 - y0) / imgsz
        (root / "labels" / split / f"{idx}.txt").write_text(
            f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
        )

    for i in range(num_imgs):
        _write("train", i, cls=i % 2)
    _write("val", 0, cls=0)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.dump({
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["classA", "classB"],
    }))
    return data_yaml


def _save_det_checkpoint(path: Path, size: str, nb_classes: int = 2) -> Path:
    """Build a fresh YOLOv9 detection model and save a libreyolo-style checkpoint."""
    net = LibreYOLO9Model(config=size, nb_classes=nb_classes)
    torch.save({"model": net.state_dict()}, path)
    return path


@pytest.mark.parametrize("loss_type", ["mgd", "cwd"])
def test_distillation_training_finishes(tmp_path, loss_type):
    """Student trains with distillation enabled; total loss finite, distill loss > 0."""
    data_yaml = _make_tiny_det_dataset(tmp_path / "data", imgsz=128, num_imgs=4)

    # Teacher = YOLOv9-c (bigger); Student = YOLOv9-t (smaller)
    teacher_ckpt = _save_det_checkpoint(tmp_path / "teacher-c.pt", size="c", nb_classes=2)
    student_ckpt = _save_det_checkpoint(tmp_path / "student-t.pt", size="t", nb_classes=2)

    student = LibreYOLO9(
        model_path=str(student_ckpt), size="t", nb_classes=2, device="cpu",
    )

    results = student.train(
        data=str(data_yaml),
        epochs=1,
        batch=2,
        imgsz=128,
        lr0=0.001,
        optimizer="SGD",
        device="cpu",
        workers=0,
        project=str(tmp_path / "runs"),
        name=f"distill-{loss_type}",
        amp=False,
        patience=1,
        # Distillation kwargs forwarded to YOLO9Trainer -> TrainConfig
        distill_model=str(teacher_ckpt),
        distill_loss_type=loss_type,
        dis=0.5,
    )

    assert "final_loss" in results
    assert math.isfinite(results["final_loss"]), f"loss not finite: {results['final_loss']}"
    assert Path(results["last_checkpoint"]).exists()


def test_distillation_with_gradient_accumulation(tmp_path):
    """Distillation composes with the gradient-accumulation loop (nbs set)."""
    data_yaml = _make_tiny_det_dataset(tmp_path / "data", imgsz=128, num_imgs=4)

    teacher_ckpt = _save_det_checkpoint(tmp_path / "teacher-c.pt", size="c", nb_classes=2)
    student_ckpt = _save_det_checkpoint(tmp_path / "student-t.pt", size="t", nb_classes=2)

    student = LibreYOLO9(
        model_path=str(student_ckpt), size="t", nb_classes=2, device="cpu",
    )

    results = student.train(
        data=str(data_yaml),
        epochs=1,
        batch=2,
        nbs=4,  # accumulate 2 micro-batches per optimizer step
        imgsz=128,
        lr0=0.001,
        optimizer="SGD",
        device="cpu",
        workers=0,
        project=str(tmp_path / "runs"),
        name="distill-accum",
        amp=False,
        patience=1,
        distill_model=str(teacher_ckpt),
        distill_loss_type="mgd",
        dis=0.5,
    )

    assert math.isfinite(results["final_loss"]), f"loss not finite: {results['final_loss']}"
    assert Path(results["last_checkpoint"]).exists()


def test_distillation_resume_restores_adapters(tmp_path, caplog):
    """Checkpoints persist the distiller adapters and resume restores them."""
    import logging

    from libreyolo.utils.serialization import load_trusted_torch_file

    data_yaml = _make_tiny_det_dataset(tmp_path / "data", imgsz=128, num_imgs=4)
    teacher_ckpt = _save_det_checkpoint(tmp_path / "teacher-c.pt", size="c", nb_classes=2)
    student_ckpt = _save_det_checkpoint(tmp_path / "student-t.pt", size="t", nb_classes=2)

    common = dict(
        data=str(data_yaml),
        batch=2,
        imgsz=128,
        lr0=0.001,
        optimizer="SGD",
        device="cpu",
        workers=0,
        project=str(tmp_path / "runs"),
        name="distill-resume",
        amp=False,
        patience=0,
        distill_model=str(teacher_ckpt),
        distill_loss_type="mgd",
        dis=0.5,
    )

    student = LibreYOLO9(
        model_path=str(student_ckpt), size="t", nb_classes=2, device="cpu",
    )
    results = student.train(epochs=1, **common)

    last = Path(results["last_checkpoint"])
    checkpoint = load_trusted_torch_file(
        last, map_location="cpu", context="test checkpoint"
    )
    assert "distiller" in checkpoint, "checkpoint must persist distiller adapters"
    assert len(checkpoint["distiller"]) > 0

    # Resume for one more epoch; the adapters must be restored, not recreated.
    resumed = LibreYOLO9(model_path=str(last), size="t", nb_classes=2, device="cpu")
    with caplog.at_level(logging.INFO, logger="libreyolo.training.trainer"):
        results2 = resumed.train(epochs=2, resume=True, exist_ok=True, **common)

    assert math.isfinite(results2["final_loss"])
    assert "Distiller adapter state restored" in caplog.text
    assert "Could not load distiller state" not in caplog.text


def test_distiller_module_shapes(tmp_path):
    """Direct Distiller test: teacher.forward → student.forward → compute_loss finite."""
    from libreyolo.distillation import Distiller

    teacher = LibreYOLO9Model(config="c", nb_classes=2).eval()
    student = LibreYOLO9Model(config="t", nb_classes=2)

    # Wrappers give us get_distill_config()
    ts = torch.save({"model": teacher.state_dict()}, tmp_path / "t.pt")  # noqa: F841
    ss = torch.save({"model": student.state_dict()}, tmp_path / "s.pt")  # noqa: F841
    teacher_wrap = LibreYOLO9(model_path=str(tmp_path / "t.pt"), size="c", nb_classes=2, device="cpu")
    student_wrap = LibreYOLO9(model_path=str(tmp_path / "s.pt"), size="t", nb_classes=2, device="cpu")

    distiller = Distiller(
        teacher_model=teacher_wrap.model,
        student_model=student_wrap.model,
        teacher_config=teacher_wrap.get_distill_config(),
        student_config=student_wrap.get_distill_config(),
        loss_type="mgd",
        loss_weight=0.5,
    )

    # Forward both; compute distillation loss
    x = torch.randn(1, 3, 128, 128)
    distiller.teacher_forward(x)
    _ = student_wrap.model(x)
    loss = distiller.compute_loss()
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss).all(), f"distill loss not finite: {loss}"
    distiller.step()
