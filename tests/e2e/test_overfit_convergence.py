"""
Overfit convergence gate — the cheapest catch-all that a family's training is
*functional at all*.

Unlike RF1 (``test_rf1_training.py``), which fine-tunes on the real ``marbles``
dataset, runs full held-out validation, and **skips experimental families**,
this test asks a much cheaper and more fundamental question that is fair for ANY
trainer — experimental or not:

    "Given a handful of fixed images, can the model memorize them?"

A correct training loop — gradients flow, the loss is wired, the target
assigner actually produces matches — drives the training loss down hard and
reaches a non-trivial mAP on those same images. A fundamentally broken loop
(dead gradients, a mis-wired loss, an assigner that never matches, a frozen
head) cannot: the loss stays flat and mAP stays ~0.

What this gate DOES prove: training is not *broken* for a family.
What it does NOT prove: that a family trains *well* / converges to a good model
on real data — that is RF1/RF5's job. Think of this as the floor of the
confidence ladder: it is the one rung that can run on *every* family, including
the experimental ones RF1 refuses to touch.

Hermetic: generates a tiny synthetic dataset on the fly (no download, no HF, no
GPU required). Runs on CPU.

Usage:
    pytest tests/e2e/test_overfit_convergence.py -v -m e2e
    pytest tests/e2e/test_overfit_convergence.py -k yolo9 -v
    pytest tests/e2e/test_overfit_convergence.py::test_overfit_negative_control -v
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from libreyolo import LibreYOLO
from .conftest import (
    FAMILY_MARKERS,
    cuda_cleanup,
    require_test_weights,
)

pytestmark = [pytest.mark.e2e, pytest.mark.overfit]

# One smallest case per *detection* family. RF-DETR is intentionally excluded:
# it uses a different train() signature (``batch_size`` / ``output_dir``) and is
# already convergence-covered by RF1 + its own e2e training tests. Non-detection
# families (l2cs gaze, fomo) are out of scope for a box-mAP overfit check.
OVERFIT_MODELS = [
    ("yolox", "n", "LibreYOLOXn.pt"),
    ("yolo9", "t", "LibreYOLO9t.pt"),
    ("yolo9_e2e", "t", "LibreYOLO9E2Et.pt"),
    ("yolonas", "s", "downloads/yolonas/yolo_nas_s_coco.pth"),
    ("dfine", "n", "LibreDFINEn.pt"),
    ("deim", "n", "weights/LibreDEIMn.pt"),
    ("deimv2", "atto", "LibreDEIMv2atto.pt"),
    ("ec", "s", "LibreECs.pt"),
    ("rtdetr", "r18", "LibreRTDETRr18.pt"),
    ("rtdetrv2", "r18", "weights/LibreRTDETRv2r18.pt"),
    ("rtdetrv4", "s", "weights/LibreRTDETRv4s.pt"),
    ("picodet", "s", "LibrePICODETs.pt"),
    ("rtmdet", "t", "LibreRTMDett.pt"),
]

# DETR-style families converge materially slower even on a memorization task, so
# they get a longer budget and (where known from RF1) a friendlier LR.
_DETR_FAMILIES = {"dfine", "deim", "deimv2", "ec", "rtdetr", "rtdetrv2", "rtdetrv4"}

# Minimum fraction the (windowed-average) training loss must drop. A correct
# loop overfitting a handful of images crushes this; a broken loop can't move
# it. Kept conservative so DETR aux-loss noise doesn't cause false failures.
MIN_LOSS_DROP = 0.25
# A broken / no-learning run must stay under this — the separation from
# MIN_LOSS_DROP is what gives the gate teeth (see the negative-control test).
FLAT_LOSS_DROP = 0.10
# Even a low mAP floor cleanly separates "learned to localize" from a broken
# loop that outputs nothing (mAP == 0). This is deliberately not a quality bar.
MIN_TRAIN_MAP = 0.05


def _make_synthetic_dataset(root: Path, n: int = 4) -> str:
    """Write a tiny hermetic YOLO detection dataset: bright boxes on gray noise.

    Each image has exactly one, clearly-separable object at a fixed location, so
    a functional training loop can memorize the set in a few dozen steps.
    """
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    h = w = 640
    bw, bh = 180, 140
    for i in range(n):
        img = rng.integers(50, 110, (h, w, 3), dtype=np.uint8)  # mid-gray noise
        cx = int(w * (0.30 + 0.12 * i))
        cy = int(h * (0.35 + 0.10 * i))
        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = cx + bw // 2, cy + bh // 2
        cv2.rectangle(img, (x1, y1), (x2, y2), (235, 235, 235), thickness=-1)
        cv2.imwrite(str(img_dir / f"img{i}.jpg"), img)
        (lbl_dir / f"img{i}.txt").write_text(
            f"0 {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}\n"
        )

    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/train",  # overfit → evaluate on the memorized images
        "names": {0: "object"},
        "nc": 1,
    }
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.dump(data, default_flow_style=False))
    return str(data_yaml)


def _relative_loss_drop(losses: list[float]) -> float:
    """Windowed relative drop: mean(first fifth) vs mean(last fifth).

    Windowing smooths the per-epoch noise (especially DETR's ~38 weighted aux
    terms) so a single noisy epoch can't fake — or mask — real convergence.
    """
    losses = [float(x) for x in losses if x is not None]
    if len(losses) < 2:
        return 0.0
    k = max(1, len(losses) // 5)
    early = float(np.mean(losses[:k]))
    late = float(np.mean(losses[-k:]))
    if early <= 0:
        return 0.0
    return (early - late) / early


def _epochs_for(family: str) -> int:
    return 45 if family in _DETR_FAMILIES else 25


def _train_overrides(family: str, size: str) -> dict:
    """Family LR/knob overrides that make a *memorization* run reliable.

    Mirrors the RF1 recipe where one exists; otherwise leans on config defaults.
    Multi-scale / aug-stop are neutralized so the loss trend is clean.
    """
    # Strip stochastic augmentation: this gate tests the core loop
    # (gradients / loss / assigner), not augmentation. A clean, deterministic
    # memorization signal is far less flaky than one fighting mosaic variance.
    common = {"mosaic_prob": 0.0, "mixup_prob": 0.0}
    if family in _DETR_FAMILIES:
        common.update(multi_scale=False)
    if family == "dfine":
        return {**common, "lr0": 1e-4, "aug_stop_epoch_ratio": 0.0}
    if family == "deim":
        return {**common, "lr0": 1e-4, "aug_stop_epoch_ratio": 0.0}
    if family == "deimv2":
        lr = {"atto": 2e-3, "femto": 1.6e-3, "pico": 1.6e-3}.get(size, 5e-4)
        return {**common, "lr0": lr, "aug_stop_epoch_ratio": 0.0}
    if family == "rtdetr":
        return {**common, "lr0": 2e-4, "mosaic_prob": 0.0, "hsv_prob": 0.0}
    if family == "ec":
        return {**common, "allow_experimental": True, "aug_stop_epoch_ratio": 0.0}
    return common


def _overfit_ids():
    return [f"{f}-{s}" for f, s, _ in OVERFIT_MODELS]


def _overfit_params():
    params = []
    for family, size, weights in OVERFIT_MODELS:
        marker = FAMILY_MARKERS.get(family)
        marks = [marker] if marker is not None else []
        params.append(pytest.param(family, size, weights, marks=marks))
    return params


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("overfit_ds")
    yaml_path = _make_synthetic_dataset(root)
    return yaml_path


@pytest.mark.parametrize("family,size,weights", _overfit_params(), ids=_overfit_ids())
def test_overfit_convergence(family, size, weights, synthetic_dataset, tmp_path):
    """A functional trainer memorizes a few images: loss collapses, mAP > 0."""
    weights = require_test_weights(weights, expected_family=family)
    model = LibreYOLO(weights, size=size)
    try:
        results = model.train(
            data=synthetic_dataset,
            epochs=_epochs_for(family),
            batch=4,
            workers=0,
            seed=0,
            save_period=999,
            project=str(tmp_path),
            name=f"overfit_{family}_{size}",
            exist_ok=True,
            **_train_overrides(family, size),
        )

        losses = results["epoch_losses"]
        drop = _relative_loss_drop(losses)
        val = model.val(
            data=synthetic_dataset,
            split="val",
            batch=4,
            conf=0.001,
            iou=0.6,
            workers=0,
        )
        train_map = val["metrics/mAP50-95"]
        print(
            f"\n  [{family}-{size}] loss {losses[0]:.4f} -> {losses[-1]:.4f} "
            f"(drop {drop:.1%}), overfit mAP50-95={train_map:.4f}"
        )

        assert drop >= MIN_LOSS_DROP, (
            f"{family}-{size}: training loss barely moved ({drop:.1%} < "
            f"{MIN_LOSS_DROP:.0%}) over {len(losses)} epochs on a memorizable "
            f"set — the training loop is not learning."
        )
        assert train_map >= MIN_TRAIN_MAP, (
            f"{family}-{size}: overfit mAP50-95={train_map:.4f} < {MIN_TRAIN_MAP} "
            f"— the model failed to localize images it was trained on."
        )
        shutil.rmtree(tmp_path, ignore_errors=True)
    finally:
        del model
        cuda_cleanup()


@pytest.mark.yolo9
def test_overfit_negative_control(synthetic_dataset, tmp_path):
    """Sanity that the gate has teeth: with lr0=0 the model cannot learn, so the
    loss must stay ~flat and the drop metric must fall well below the pass
    threshold. If this ever "passes" the overfit check, the metric is bogus."""
    model = LibreYOLO("LibreYOLO9t.pt", size="t")
    try:
        results = model.train(
            data=synthetic_dataset,
            epochs=25,
            batch=4,
            workers=0,
            seed=0,
            lr0=0.0,
            warmup_lr_start=0.0,
            mosaic_prob=0.0,
            mixup_prob=0.0,
            save_period=999,
            project=str(tmp_path),
            name="overfit_negctl",
            exist_ok=True,
        )
        drop = _relative_loss_drop(results["epoch_losses"])
        print(f"\n  [negative-control lr0=0] loss drop {drop:.1%}")
        assert drop < FLAT_LOSS_DROP, (
            f"Negative control learned with lr0=0 (drop {drop:.1%}) — the "
            f"overfit metric cannot distinguish a broken run from a real one."
        )
        shutil.rmtree(tmp_path, ignore_errors=True)
    finally:
        del model
        cuda_cleanup()
