"""Per-family training profiler support, through the real train() API.

The unit suite proves the profiler hooks exist in every G0/G1 training loop
(``test_training_profiler.py`` drives the loop methods directly). This file
proves the whole pipeline per family: ``profile=True`` must travel from the
public ``train()`` kwargs into the trainer config, build the profiler in
``setup()``, drive the hooked loop, and emit the analysis artifacts.

The assertion is on the artifacts, not on run completion: the failure mode
this feature ships against is the *silent no-op* (a run that trains happily
and writes nothing), so a test that only checks train() returned would prove
nothing. Every trainable family from ``libreyolo.models.registry`` (groups
g0, g1 and g2 — all tasks: detect, point, classify, semantic, restore) is
enrolled; adding a family to those groups without profiler support fails the
registry-coverage test at the bottom. g3 is inference-only and g4 is frozen,
so training profiling does not apply there; the s tier is excluded from group
rollouts by definition.

No downloads: the dataset is generated and every model is built with random
init. Families that pull a pretrained backbone at construction are covered by
the ``external_data`` marker, mirroring ``test_cuda_graph_training_families``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

DEVICE = "0" if torch.cuda.is_available() else "cpu"

# (family id, class name, size, ctor model_path, ctor kwargs, dataset fixture,
#  extra train kwargs)
#
# imgsz 320 everywhere except rfdetr, which validates imgsz against its
# DINOv2 windowing and defaults to its native input size when omitted.
# multi_scale=False for the families that resize every batch by default:
# profiling wants a steady per-step shape so the window measures one
# configuration instead of a mix. Sizes/tasks mirror
# ``test_cuda_graph_training_families`` — the previous cross-family rollout.
CASES = [
    # g0
    ("yolo9", "LibreYOLO9", "t", None, {}, "detect_dataset", {"imgsz": 320}),
    ("rfdetr", "LibreRFDETR", "n", {}, {}, "detect_dataset", {}),
    # g1
    ("yolo9_e2e", "LibreYOLO9E2E", "t", None, {}, "detect_dataset", {"imgsz": 320}),
    ("yolo9_p2", "LibreYOLO9P2", "t", None, {}, "detect_dataset", {"imgsz": 320}),
    (
        "ec",
        "LibreEC",
        "s",
        None,
        {},
        "detect_dataset",
        {"imgsz": 320, "multi_scale": False},
    ),
    ("rtdetr", "LibreRTDETR", "r18", None, {}, "detect_dataset", {"imgsz": 320}),
    ("rtdetrv2", "LibreRTDETRv2", "r18", None, {}, "detect_dataset", {"imgsz": 320}),
    (
        "rtdetrv4",
        "LibreRTDETRv4",
        "s",
        None,
        {},
        "detect_dataset",
        {"imgsz": 320, "multi_scale": False},
    ),
    (
        "dfine",
        "LibreDFINE",
        "n",
        None,
        {},
        "detect_dataset",
        {"imgsz": 320, "multi_scale": False},
    ),
    (
        "deim",
        "LibreDEIM",
        "n",
        None,
        {},
        "detect_dataset",
        {"imgsz": 320, "multi_scale": False},
    ),
    (
        "deimv2",
        "LibreDEIMv2",
        "atto",
        None,
        {},
        "detect_dataset",
        {"imgsz": 320, "multi_scale": False},
    ),
    ("yolonas", "LibreYOLONAS", "s", None, {}, "detect_dataset", {"imgsz": 320}),
    # g2 — supporting trainables (all through the BaseTrainer loop)
    ("yolox", "LibreYOLOX", "t", None, {}, "detect_dataset", {"imgsz": 320}),
    (
        "yolo7",
        "LibreYOLO7",
        "b",
        None,
        {},
        "detect_dataset",
        {"imgsz": 320},
    ),
    (
        "rtmdet",
        "LibreRTMDet",
        "t",
        None,
        {},
        "detect_dataset",
        {"imgsz": 320},
    ),
    (
        "picodet",
        "LibrePICODET",
        "s",
        None,
        {},
        "detect_dataset",
        {"imgsz": 320},
    ),
    (
        "fomo",
        "LibreFOMO",
        "s",
        None,
        {"task": "point"},
        "detect_dataset",
        {"imgsz": 96},
    ),
    (
        "segformer",
        "LibreSegformer",
        "b0",
        None,
        {"task": "semantic"},
        "semantic_dataset",
        {"imgsz": 256},
    ),
    (
        "lingbotvision",
        "LibreLingBotVision",
        "s",
        None,
        {"task": "semantic"},
        "semantic_dataset",
        {"imgsz": 224},
    ),
    (
        "dinov2",
        "LibreDINOv2",
        "s",
        None,
        {"task": "semantic"},
        "semantic_dataset",
        {"imgsz": 224},
    ),
    (
        "nafnet",
        "LibreNAFNet",
        "s",
        None,
        {"task": "restore"},
        "restore_dataset",
        {"imgsz": 256},
    ),
    (
        "resnet",
        "LibreResNet",
        "18",
        None,
        {"task": "classify"},
        "classify_dataset",
        {"imgsz": 224},
    ),
    (
        "convnext",
        "LibreConvNeXt",
        "t",
        None,
        {"task": "classify"},
        "classify_dataset",
        {"imgsz": 224},
    ),
    (
        "mobilenetv4",
        "LibreMobileNetV4",
        "s",
        None,
        {"task": "classify"},
        "classify_dataset",
        {"imgsz": 224},
    ),
    (
        "efficientnetv2",
        "LibreEfficientNetV2",
        "b0",
        None,
        {"task": "classify"},
        "classify_dataset",
        {"imgsz": 224},
    ),
]


@pytest.fixture(scope="module")
def detect_dataset(tmp_path_factory) -> Path:
    """A generated 24-image YOLO-format detection dataset (offline)."""
    root = tmp_path_factory.mktemp("profile_train_data")
    rng = np.random.default_rng(0)
    for split, count in (("train", 16), ("valid", 8)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(count):
            pixels = rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / split / "images" / f"{i:03d}.jpg")
            rows = []
            for _ in range(rng.integers(2, 6)):
                cx, cy = rng.uniform(0.2, 0.8, 2)
                w, h = rng.uniform(0.05, 0.15, 2)
                rows.append(f"{rng.integers(0, 2)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (root / split / "labels" / f"{i:03d}.txt").write_text("\n".join(rows))
    (root / "data.yaml").write_text(
        yaml.dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "valid/images",
                "nc": 2,
                "names": ["a", "b"],
            }
        )
    )
    return root / "data.yaml"


@pytest.fixture(scope="module")
def classify_dataset(tmp_path_factory) -> Path:
    """A generated 3-class ImageFolder dataset."""
    root = tmp_path_factory.mktemp("profile_train_cls")
    rng = np.random.default_rng(1)
    for split, count in (("train", 18), ("val", 6)):
        for i in range(count):
            folder = root / split / f"class{i % 3}"
            folder.mkdir(parents=True, exist_ok=True)
            pixels = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{i:03d}.jpg")
    return root


@pytest.fixture(scope="module")
def semantic_dataset(tmp_path_factory) -> Path:
    """A generated 3-class dense-mask dataset."""
    root = tmp_path_factory.mktemp("profile_train_sem")
    rng = np.random.default_rng(2)
    for split, count in (("train", 16), ("val", 4)):
        (root / "images" / split).mkdir(parents=True)
        (root / "masks" / split).mkdir(parents=True)
        for i in range(count):
            pixels = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / "images" / split / f"{i:03d}.jpg")
            mask = np.zeros((256, 256), np.uint8)
            for label in (1, 2):
                y, x = rng.integers(0, 150, 2)
                mask[y : y + 80, x : x + 80] = label
            Image.fromarray(mask).save(root / "masks" / split / f"{i:03d}.png")
    (root / "data.yaml").write_text(
        yaml.dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "masks_dir": "masks",
                "nc": 3,
                "names": ["background", "a", "b"],
            }
        )
    )
    return root / "data.yaml"


@pytest.fixture(scope="module")
def restore_dataset(tmp_path_factory) -> Path:
    """A generated paired noisy/clean restoration dataset."""
    root = tmp_path_factory.mktemp("profile_train_restore")
    rng = np.random.default_rng(3)
    for split, count in (("train", 12), ("val", 4)):
        (root / split / "inputs").mkdir(parents=True)
        (root / split / "targets").mkdir(parents=True)
        for i in range(count):
            clean = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            noisy = np.clip(
                clean.astype(np.float32) + rng.normal(0, 12, clean.shape), 0, 255
            ).astype(np.uint8)
            Image.fromarray(noisy).save(root / split / "inputs" / f"{i:03d}.png")
            Image.fromarray(clean).save(root / split / "targets" / f"{i:03d}.png")
    (root / "data.yaml").write_text(
        yaml.dump(
            {
                "path": str(root),
                "train": "train/inputs",
                "val": "val/inputs",
                "input_dir": "inputs",
                "target_dir": "targets",
                "nc": 1,
                "names": ["restore"],
            }
        )
    )
    return root / "data.yaml"


@pytest.mark.external_data  # several families fetch a pretrained backbone
@pytest.mark.parametrize(
    "family,class_name,size,ctor,ctor_kwargs,dataset_fixture,extra",
    CASES,
    ids=[case[0] for case in CASES],
)
def test_profile_emits_report(
    family, class_name, size, ctor, ctor_kwargs, dataset_fixture, extra,
    request, tmp_path,
):
    """profile=True + profile_then_stop=True must emit the analysis artifacts
    and truncate the run — for every trainable family, through the public API."""
    import libreyolo
    from libreyolo.models.registry import group_of

    dataset = request.getfixturevalue(dataset_fixture)
    torch.manual_seed(0)
    model = getattr(libreyolo, class_name)(ctor, size, **ctor_kwargs)
    result = model.train(
        data=str(dataset),
        epochs=3,
        batch=4,
        workers=0,
        device=DEVICE,
        project=str(tmp_path),
        name="prof",
        exist_ok=True,
        profile=True,
        profile_then_stop=True,
        profile_warmup=1,
        profile_steps=2,
        profile_open=False,
        eval_interval=100,
        **({"pretrained": False} if group_of(family) in {"g0", "g1"} else {}),
        **extra,
    )

    save_dir = tmp_path / "prof"
    profile_json = save_dir / "profile.json"
    assert profile_json.exists(), (
        f"{family}: profile.json was not emitted — profile=True was a no-op"
    )
    assert (save_dir / "profile_trace.json").exists(), f"{family}: no trace"
    assert (save_dir / "profile_summary.json").exists(), f"{family}: no summary"

    prof = json.loads(profile_json.read_text(encoding="utf-8"))
    assert prof.get("schema") == "libreyolo.profile.analysis/v1", f"{family}"
    assert prof.get("img_per_s") and prof["img_per_s"] > 0, (
        f"{family}: no throughput measured: {prof.get('img_per_s')}"
    )
    # The phase brackets must have landed: forward/backward attribution is
    # what distinguishes a real profile from a bare wall-clock number.
    phase_names = {p["phase"] for p in prof.get("phases", [])}
    assert {"forward", "backward"} <= phase_names, (
        f"{family}: phases missing from trace: {sorted(phase_names)}"
    )

    # profile_then_stop: the run must truncate once the window fills, not
    # train for the requested 3 epochs. Families with a default effective
    # batch (rfdetr: nbs=16 -> 4-step accumulation) get their window rounded
    # up to accumulation boundaries, which can push the close past epoch 1 on
    # a tiny dataset — so the bound is "fewer than requested", not "one".
    assert len(result["epoch_losses"]) < 3, (
        f"{family}: expected a truncated run, got all "
        f"{len(result['epoch_losses'])} epochs"
    )


def test_all_trainable_families_are_enrolled():
    """Adding a family to g0/g1/g2 without enrolling it here must fail loudly.

    Profiler support is a rollout-group feature: the trainable groups promise
    it (see ``libreyolo/models/registry.py``). This keeps CASES in lockstep
    with the registry so the promise is enforced by CI rather than by review.
    """
    from libreyolo.models.registry import families_in

    enrolled = {case[0] for case in CASES}
    expected = (
        set(families_in("g0")) | set(families_in("g1")) | set(families_in("g2"))
    )
    assert enrolled == expected, (
        f"profiler e2e coverage drifted from the registry: "
        f"missing={sorted(expected - enrolled)} extra={sorted(enrolled - expected)}"
    )
