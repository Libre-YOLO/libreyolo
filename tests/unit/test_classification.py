"""Image-classification task tests.

Covers the shared classification stack (ImageFolder dataset, collate,
ClassifyValidator, Results.probs) and the LibreDINOv2 model wiring (classify is
a DINOv2 linear probe, not the RF-DETR detector, so it lives in the dinov2
family). All tests run on CPU with a tiny synthetic ImageFolder so they need no
network or GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


def _make_imagefolder(root, n_classes=3, n_per=5, size=64):
    """Create a tiny train/val ImageFolder where each class has a distinct hue.

    Distinct per-class colors make the set trivially separable so a couple of
    training steps demonstrably reduce the loss.
    """
    classes = [f"c{i}" for i in range(n_classes)]
    for split in ("train", "val"):
        for ci, name in enumerate(classes):
            cls_dir = root / split / name
            cls_dir.mkdir(parents=True, exist_ok=True)
            base = np.zeros((size, size, 3), dtype=np.uint8)
            base[:, :, ci % 3] = 200  # dominant channel per class
            for j in range(n_per):
                noisy = np.clip(
                    base + np.random.randint(0, 40, base.shape, dtype=np.int16),
                    0,
                    255,
                ).astype(np.uint8)
                Image.fromarray(noisy).save(cls_dir / f"{name}_{j}.png")
    return classes


def _make_named_imagefolder(root, classes, n_per=2, size=32):
    for split in ("train", "val"):
        for ci, name in enumerate(classes):
            cls_dir = root / split / name
            cls_dir.mkdir(parents=True, exist_ok=True)
            base = np.zeros((size, size, 3), dtype=np.uint8)
            base[:, :, ci % 3] = 200
            for j in range(n_per):
                Image.fromarray(base).save(cls_dir / f"{name}_{j}.png")


def test_classify_dataset_and_collate(tmp_path):
    from libreyolo.data import ClassifyDataset, classify_collate_fn, get_class_names

    classes = _make_imagefolder(tmp_path, n_classes=3, n_per=4)
    assert get_class_names(tmp_path, "train") == sorted(classes)

    ds = ClassifyDataset(tmp_path, split="train", imgsz=32, augment=False)
    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(label, int)

    batch = [ds[i] for i in range(4)]
    imgs, labels, infos, ids = classify_collate_fn(batch)
    assert imgs.shape == (4, 3, 32, 32)
    assert labels.shape == (4,) and labels.dtype == torch.long
    assert len(infos) == 4 and len(ids) == 4


def test_classify_dataset_rejects_unknown_split_classes(tmp_path):
    from libreyolo.data import ClassifyDataset

    _make_named_imagefolder(tmp_path, ["cat"])
    extra_dir = tmp_path / "val" / "dog"
    extra_dir.mkdir()
    Image.new("RGB", (16, 16)).save(extra_dir / "dog.png")

    with pytest.raises(ValueError, match="unknown classes"):
        ClassifyDataset(
            tmp_path,
            split="val",
            imgsz=32,
            augment=False,
            class_to_idx={"cat": 0},
        )


def test_classify_validator_uses_model_name_order(tmp_path):
    from libreyolo.validation import ClassifyValidator, ValidationConfig

    _make_named_imagefolder(tmp_path, ["cat", "dog"])

    class _Model:
        names = {0: "dog", 1: "cat"}
        nb_classes = 2

    validator = ClassifyValidator(
        model=_Model(),
        config=ValidationConfig(
            data=str(tmp_path),
            batch_size=4,
            imgsz=32,
            device="cpu",
            num_workers=0,
            split="val",
            verbose=False,
        ),
    )

    dataloader = validator._setup_dataloader()
    labels_by_path = {
        Path(path).parent.name: int(label)
        for path, label in dataloader.dataset._impl.samples
    }

    assert labels_by_path["dog"] == 0
    assert labels_by_path["cat"] == 1


@pytest.mark.external_data
@pytest.mark.network
@pytest.mark.slow
def test_dinov2_classify_forward():
    """LibreDINOv2 classify build + forward (DINOv2 backbone; random-init if offline)."""
    from libreyolo.models.dinov2.model import LibreDINOv2

    m = LibreDINOv2(
        model_path=None, size="n", task="classify", nb_classes=4, device="cpu"
    )
    assert m.task == "classify"
    assert m.input_size == 224
    # The classify wrapper exposes the linear-probe classifier and builds no
    # DETR decoder (model is None signals the non-detection head path).
    assert m.model.model is None
    assert m.model.classifier is not None

    x = torch.randn(1, 3, 224, 224)
    m.model.train()
    out = m.model(x, targets=torch.tensor([2]))
    assert "total_loss" in out

    m.model.eval()
    with torch.no_grad():
        assert m.model(x).shape == (1, 4)


def test_safe_zip_extraction_rejects_path_traversal(tmp_path):
    import zipfile

    from libreyolo.data.classify_dataset import _safe_extract_zip

    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", "payload")
    with zipfile.ZipFile(bad_zip) as zf:
        with pytest.raises(ValueError):
            _safe_extract_zip(zf, tmp_path / "dest")


def test_dinov2_classify_can_load_and_detect_nb_classes():
    pytest.importorskip("transformers")
    from libreyolo.models.dinov2.model import LibreDINOv2
    from libreyolo.models.rfdetr.model import LibreRFDETR

    weights = {
        "backbone.encoder.encoder.embeddings.position_embeddings": torch.empty(
            1, 1370, 384
        ),
        "linear.weight": torch.empty(4, 256),
    }

    # Classify checkpoints (backbone + linear head) belong to LibreDINOv2 now;
    # RF-DETR must no longer claim them.
    assert LibreDINOv2.can_load(weights)
    assert not LibreRFDETR.can_load(weights)
    assert LibreDINOv2.detect_nb_classes(weights) == 4


def test_dinov2_classify_load_infers_nc_from_linear_weight(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from libreyolo.models.dinov2.model import LibreDINOv2

    class _LoadResult:
        missing_keys = []
        unexpected_keys = []

    class _FakeClassifier(torch.nn.Module):
        def __init__(self, nb_classes):
            super().__init__()
            self.linear = torch.nn.Linear(2, nb_classes)
            self.nb_classes = nb_classes

    class _FakeWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = None  # signals the non-detection (linear-head) path
            self.classifier = _FakeClassifier(80)
            self.nb_classes = 80

        def load_state_dict(self, loaded, strict=False):
            state = loaded.get("model", loaded)
            expected = self.classifier.linear.out_features
            actual = state["linear.weight"].shape[0]
            if expected != actual:
                raise RuntimeError(f"expected {expected} classifier rows, got {actual}")
            return _LoadResult()

    monkeypatch.setattr(LibreDINOv2, "_init_model", lambda self: _FakeWrapper())
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model": {
                "backbone.stem.weight": torch.ones(1),
                "linear.weight": torch.ones(4, 2),
                "linear.bias": torch.ones(4),
            },
            "model_family": "dinov2",
            "size": "n",
            "task": "classify",
        },
        path,
    )

    model = LibreDINOv2(str(path), size="n", task="classify", device="cpu")

    assert model.nb_classes == 4
    assert model.model.classifier.linear.out_features == 4


def test_classify_family_train_end_to_end(tmp_path):
    """A classify family must actually fine-tune through the shared trainer.

    Regression guard for two integration bugs that unit-level dataset/forward
    tests never exercised:

    * the trainer built ``ClassifyDataset`` with ``crop_pct=``/``interpolation=``
      as direct kwargs while the dataset takes them via ``transform_kwargs`` —
      every ``model.train(...)`` raised ``TypeError`` on setup; and
    * the classify configs inherited the detection ``eval_interval=10``, so the
      documented short ``epochs=5`` fine-tune never validated and never wrote a
      ``best.pt``.

    Uses the smallest family (MobileNetV4-s) on a random-init model + tiny
    synthetic ImageFolder, so it stays CPU-only and needs no network.
    """
    from libreyolo import LibreMobileNetV4

    _make_imagefolder(tmp_path / "data", n_classes=2, n_per=6, size=64)

    model = LibreMobileNetV4(size="s", device="cpu")
    assert model.nb_classes == 1000

    metrics = model.train(
        data=str(tmp_path / "data"),
        epochs=1,
        batch=4,
        imgsz=32,
        workers=0,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="cls_smoke",
        exist_ok=True,
    )

    # Head was rebuilt 1000 -> 2 for the dataset's class count.
    assert model.nb_classes == 2

    # eval_interval=1 => the single epoch validated and produced a best.pt.
    best = tmp_path / "runs" / "cls_smoke" / "weights" / "best.pt"
    assert best.exists(), "classification training must save best.pt"
    epoch_metrics = metrics.get("epoch_metrics", [])
    assert epoch_metrics and epoch_metrics[-1].get("validated") is True
    val = epoch_metrics[-1].get("val_metrics") or {}
    scalars = val.get("metrics", val)
    assert "metrics/accuracy_top1" in scalars

    # The saved checkpoint reloads as a 2-class classifier and predicts.
    reloaded = LibreMobileNetV4(str(best), device="cpu")
    assert reloaded.nb_classes == 2
    result = reloaded(str(tmp_path / "data" / "val" / "c0" / "c0_0.png"))
    assert result.probs.data.shape[0] == 2


