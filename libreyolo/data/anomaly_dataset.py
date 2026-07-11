"""MVTec-style category dataset resolution for anomaly detection."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_anomaly_root(data: str | Path) -> Path:
    data_path = Path(data)
    if data_path.is_dir():
        return data_path.resolve()
    if data_path.is_file() and data_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        with data_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        root = Path(config.get("path", data_path.parent))
        if not root.is_absolute():
            root = (data_path.parent / root).resolve()
        category = config.get("category")
        return (root / str(category)).resolve() if category else root.resolve()
    raise ValueError(f"Anomaly data must be a category directory or YAML, got {data!r}.")


def resolve_good_training_images(data: str | Path) -> list[Path]:
    root = resolve_anomaly_root(data)
    images = _image_files(root / "train" / "good")
    if not images:
        raise ValueError(f"No good training images found under {root / 'train' / 'good'}.")
    return images


def resolve_anomaly_test_samples(
    data: str | Path,
) -> list[tuple[Path, int, Path | None]]:
    root = resolve_anomaly_root(data)
    test_root = root / "test"
    if not test_root.is_dir():
        raise ValueError(f"Anomaly test directory not found: {test_root}.")
    samples: list[tuple[Path, int, Path | None]] = []
    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        label = 0 if defect_dir.name.lower() == "good" else 1
        for image_path in _image_files(defect_dir):
            mask_path = None
            if label:
                gt_dir = root / "ground_truth" / defect_dir.name
                candidates = (
                    gt_dir / f"{image_path.stem}_mask.png",
                    gt_dir / f"{image_path.stem}.png",
                )
                mask_path = next((path for path in candidates if path.is_file()), None)
            samples.append((image_path, label, mask_path))
    if not samples:
        raise ValueError(f"No anomaly test images found under {test_root}.")
    return samples


__all__ = [
    "IMAGE_EXTENSIONS",
    "resolve_anomaly_root",
    "resolve_anomaly_test_samples",
    "resolve_good_training_images",
]
