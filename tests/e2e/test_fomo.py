"""End-to-end smoke tests for LibreFOMO."""

from pathlib import Path

import pytest
import torch

from libreyolo import LibreYOLO


pytestmark = [pytest.mark.e2e, pytest.mark.fomo]


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_load_cloud_checkpoint(size: str) -> None:
    """Download and load the cloud weights, check metadata, and run a forward pass."""
    model_name = f"LibreFOMO{size}.pt"

    model = LibreYOLO(model_name, device="cpu")

    assert model.size == size
    assert model.task == "point"
    assert model.nb_classes == 1

    imgsz = model.model.imgsz
    dummy_input = torch.zeros(1, 3, imgsz, imgsz)

    with torch.no_grad():
        out = model._forward(dummy_input)

    expected_hw = imgsz // 8
    assert out.shape == (1, 2, expected_hw, expected_hw), f"Unexpected output shape: {out.shape}"


def _write_tiny_point_dataset(root: Path, imgsz: int = 96) -> Path:
    from PIL import Image
    import yaml

    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (imgsz, imgsz), color=(128, 64, 32)).save(img_dir / "sample.jpg")
        (lbl_dir / "sample.txt").write_text("0 0.5 0.5 0.05 0.05\n", encoding="utf-8")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": {0: "object"},
            }
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_fomo_training_e2e(tmp_path: Path) -> None:
    """Verify that a LibreFOMO model trains successfully on a synthetic dataset."""
    from libreyolo import LibreFOMO

    data_yaml = _write_tiny_point_dataset(tmp_path, imgsz=96)

    model = LibreFOMO(model_path=None, size="s", nb_classes=1, device="cpu")

    results = model.train(
        allow_experimental=True,
        data=str(data_yaml),
        epochs=1,
        batch=2,
        imgsz=96,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="fomo_e2e_test",
        exist_ok=True,
        workers=0,
        scheduler="cosine",
        mosaic_prob=1.0,
        flip_prob=0.5,
    )

    assert "final_loss" in results
    assert len(results["epoch_losses"]) == 1


def _write_multiclass_point_dataset(root: Path, imgsz: int = 96) -> Path:
    from PIL import Image
    import yaml

    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (imgsz, imgsz), color=(128, 64, 32)).save(img_dir / "sample.jpg")
        (lbl_dir / "sample.txt").write_text("0 0.25 0.25 0.05 0.05\n1 0.75 0.75 0.05 0.05\n", encoding="utf-8")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "nc": 2,
                "names": {0: "person", 1: "dog"},
            }
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_fomo_multiclass_training_e2e(tmp_path: Path) -> None:
    """Verify that a multiclass LibreFOMO model trains successfully on a synthetic dataset."""
    from libreyolo import LibreFOMO

    data_yaml = _write_multiclass_point_dataset(tmp_path, imgsz=96)

    model = LibreFOMO(model_path=None, size="s", nb_classes=2, device="cpu")

    results = model.train(
        allow_experimental=True,
        data=str(data_yaml),
        epochs=1,
        batch=2,
        imgsz=96,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="fomo_multiclass_e2e_test",
        exist_ok=True,
        workers=0,
        scheduler="cosine",
        mosaic_prob=1.0,
        flip_prob=0.5,
    )

    assert "final_loss" in results
    assert len(results["epoch_losses"]) == 1


