"""Rebuild the LibreYOLO classification datasets from clean upstream sources.

We never mirror a third party's *packaged* artifact (their zip / weights) — it
may carry a license they attached to the repackaging. Instead we rebuild from
the canonical upstream and publish our own artifact with clear provenance.

Source: fast.ai Imagenette (Apache-2.0)
  https://github.com/fastai/imagenette
  https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz

Produces (ImageFolder ``train/val/<wordnet_id>`` layout, preserved verbatim):
  - imagenette160.zip : full set (9,469 train / 3,925 val, 10 classes)
  - smoke10.zip       : 2 images/class/split CI smoke set (replaces the former
                        ImageNet-derived ``imagenet10``, which cannot be
                        cleanly redistributed under ImageNet's terms)

Usage:
  python scripts/build_imagenette.py --out ./build             # build zips only
  HF_TOKEN=... python scripts/build_imagenette.py --out ./build --upload

Reruns are safe: a valid cached download is reused, extraction is redone from
scratch, and the zips are rebuilt — so ``--upload`` can be retried without a
fresh ``build``. The ``--upload`` step writes to the ``LibreYOLO`` HF org and
requires a token with write scope (env ``HF_TOKEN``); the token is never
written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

SOURCE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
# Pin the upstream artifact so a silent re-cut is caught.
SOURCE_SHA256 = "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"

# Full-set image counts for the pinned archive. An extraction that stops short
# (disk full, interrupted) would silently pack fewer images, so we assert these
# after building the full zip.
EXPECTED_COUNTS = {"train": 9469, "val": 3925}

# WordNet id -> human-readable label (documentation only; folders keep the ids
# so the class set matches canonical Imagenette).
WNID = {
    "n01440764": "tench", "n02102040": "English springer",
    "n02979186": "cassette player", "n03000684": "chain saw",
    "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump",
    "n03445777": "golf ball", "n03888257": "parachute",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract ``tar`` into ``dest`` rejecting members that escape it.

    ``filter="data"`` blocks path-traversal / unsafe members and is the default
    from Python 3.14 (available on 3.12+ and recent 3.8–3.11 patch releases).
    Fall back to a plain extract only where the kwarg is unavailable.
    """
    try:
        tar.extractall(dest, filter="data")
    except TypeError:
        tar.extractall(dest)


def download(out: Path) -> Path:
    """Fetch the pinned upstream tarball into ``out``, verifying its sha256.

    A cached file is reused only if it still matches the pin, and a fresh
    download lands on a ``.part`` file that is verified before being promoted —
    so an interrupted or corrupted download self-heals on the next run instead
    of wedging every future run with a stale mismatching file.
    """
    tgz = out / "imagenette2-160.tgz"
    if tgz.exists() and _sha256(tgz) == SOURCE_SHA256:
        print(f"sha256 OK (cached): {SOURCE_SHA256}")
        return tgz

    part = out / "imagenette2-160.tgz.part"
    print(f"downloading {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, part)  # noqa: S310
    digest = _sha256(part)
    if digest != SOURCE_SHA256:
        part.unlink(missing_ok=True)
        raise SystemExit(f"sha256 mismatch: expected {SOURCE_SHA256}, got {digest}")
    part.replace(tgz)
    print(f"sha256 OK: {digest}")
    return tgz


def build(out: Path) -> tuple[Path, Path]:
    tgz = download(out)

    # Extract into a fresh tree each run so a previous (partial or stale)
    # extraction can never leak files into the published archives.
    extract = out / "extract"
    if extract.exists():
        shutil.rmtree(extract)
    extract.mkdir(parents=True)
    with tarfile.open(tgz) as t:
        _extractall(t, extract)

    train_dirs = [p for p in extract.rglob("train") if p.is_dir()]
    if not train_dirs:
        raise SystemExit(
            f"no 'train/' directory found under {extract} after extraction - "
            "the upstream archive layout may have changed."
        )
    root = train_dirs[0].parent

    def files(split: str, per_class: int | None = None):
        for cls in sorted(d for d in (root / split).iterdir() if d.is_dir()):
            members = [f for f in sorted(cls.glob("*")) if f.is_file()]
            for f in (members[:per_class] if per_class else members):
                yield cls.name, f

    full = out / "imagenette160.zip"
    counts = {"train": 0, "val": 0}
    with zipfile.ZipFile(full, "w", zipfile.ZIP_STORED) as z:
        for split in ("train", "val"):
            for cls, f in files(split):
                z.write(f, arcname=f"{split}/{cls}/{f.name}")
                counts[split] += 1
    for split, expected in EXPECTED_COUNTS.items():
        if counts[split] != expected:
            raise SystemExit(
                f"{split}: packed {counts[split]} images, expected {expected} - "
                "extraction looks incomplete; delete the build dir and retry."
            )
    print(f"built {full.name} ({counts['train']} train / {counts['val']} val)")

    smoke = out / "smoke10.zip"
    with zipfile.ZipFile(smoke, "w", zipfile.ZIP_STORED) as z:
        for split in ("train", "val"):
            for cls, f in files(split, per_class=2):
                z.write(f, arcname=f"{split}/{cls}/{f.name}")
    print(f"built {smoke.name}")
    return full, smoke


def _card(pretty: str, body: str) -> str:
    return (
        "---\n"
        "license: apache-2.0\n"
        "task_categories:\n- image-classification\n"
        "tags:\n- imagenette\n- classification\n- libreyolo\n"
        f"pretty_name: {pretty}\n"
        "---\n\n" + body
    )


def upload(full: Path, smoke: Path) -> None:
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "--upload requires the 'huggingface_hub' package "
            "(pip install huggingface_hub)."
        ) from exc

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("--upload requires HF_TOKEN in the environment")
    api = HfApi(token=token)
    classes = "\n".join(f"- `{k}` — {v}" for k, v in WNID.items())
    jobs = [
        ("LibreYOLO/imagenette160", full, _card(
            "Imagenette 160 (LibreYOLO)",
            "# Imagenette 160 (LibreYOLO)\n\n"
            "LibreYOLO-hosted copy of Imagenette, rebuilt from the canonical "
            f"upstream (fast.ai, Apache-2.0), source `sha256` `{SOURCE_SHA256}`. "
            "Repackaged `.tgz`->`.zip`, original `train/val/<wnid>` layout "
            "preserved; no third-party assets.\n\n"
            f"10 classes:\n{classes}\n")),
        ("LibreYOLO/smoke10", smoke, _card(
            "smoke10 (LibreYOLO CI smoke set)",
            "# smoke10\n\nTiny 2-image-per-class subset of Imagenette "
            "(Apache-2.0) for CI smoke tests. Replaces the former "
            "ImageNet-derived `imagenet10`.\n")),
    ]
    for repo_id, zip_path, card in jobs:
        api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="dataset")
        api.upload_file(path_or_fileobj=str(zip_path), path_in_repo=zip_path.name,
                        repo_id=repo_id, repo_type="dataset")
        print(f"uploaded https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("build"))
    ap.add_argument("--upload", action="store_true", help="publish to LibreYOLO HF org")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    full, smoke = build(args.out)
    if args.upload:
        upload(full, smoke)


if __name__ == "__main__":
    main()
