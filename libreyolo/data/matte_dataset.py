"""Paired image/matte dataset resolution for the matte task.

The matte task validates (and, in a future release, fine-tunes) on paired data:
an RGB image and a single-channel ground-truth alpha matte with the same stem.
Two layouts are accepted, both documented in ``docs/dataset_schema.md``:

Directory layout::

    <root>/
      images/   subject.jpg ...
      mattes/   subject.png ...   (grayscale, 0=background 255=foreground)

YAML layout (``data=matte.yaml``)::

    path: /abs/or/rel/root
    val_images: images        # dir under path (or absolute)
    val_mattes: mattes         # dir under path (or absolute)
    train_images: train/images # optional, for the future fine-tune
    train_mattes: train/mattes

``mattes/`` (or ``gt/`` / ``masks/`` / ``alpha/``) is auto-detected in the
directory layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_MATTE_DIRNAMES = ("mattes", "matte", "gt", "masks", "mask", "alpha")


def _index_by_stem(directory: Path) -> dict:
    out = {}
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            out.setdefault(p.stem, p)
    return out


def _pair(image_dir: Path, matte_dir: Path) -> List[Tuple[Path, Path]]:
    images = _index_by_stem(image_dir)
    mattes = _index_by_stem(matte_dir)
    pairs = [(images[stem], mattes[stem]) for stem in images if stem in mattes]
    if not pairs:
        raise ValueError(
            f"No image/matte pairs found matching by stem between {image_dir} and {matte_dir}."
        )
    return sorted(pairs)


def resolve_matte_pairs(data, split: str = "val") -> List[Tuple[Path, Path]]:
    """Resolve ``data`` (a directory or a dataset YAML) to ``[(image, matte), ...]``."""
    data_path = Path(str(data))

    if data_path.is_dir():
        # Directory layout: images/ + auto-detected matte dir.
        image_dir = data_path / "images" if (data_path / "images").is_dir() else data_path
        matte_dir = None
        for name in _MATTE_DIRNAMES:
            if (data_path / name).is_dir():
                matte_dir = data_path / name
                break
        if matte_dir is None:
            raise ValueError(
                f"Matte directory not found under {data_path}; expected one of {_MATTE_DIRNAMES}."
            )
        return _pair(image_dir, matte_dir)

    if data_path.suffix.lower() in (".yaml", ".yml") and data_path.is_file():
        import yaml

        with open(data_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        root = Path(cfg.get("path", data_path.parent))
        if not root.is_absolute():
            root = (data_path.parent / root).resolve()

        def _resolve(key: str) -> Path:
            val = cfg.get(key)
            if val is None:
                raise ValueError(f"Matte YAML {data_path} is missing '{key}'.")
            p = Path(val)
            return p if p.is_absolute() else (root / p)

        img_key = f"{split}_images"
        matte_key = f"{split}_mattes"
        return _pair(_resolve(img_key), _resolve(matte_key))

    raise ValueError(
        f"Matte data must be a directory or a dataset YAML; got {data!r}."
    )


__all__ = ["resolve_matte_pairs"]
