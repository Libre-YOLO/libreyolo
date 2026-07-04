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

The ``--upload`` step writes to the ``LibreYOLO`` HF org and requires a token
with write scope (env ``HF_TOKEN``). The token is never written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

SOURCE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
# Pin the upstream artifact so a silent re-cut is caught.
SOURCE_SHA256 = "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"

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


def download(out: Path) -> Path:
    tgz = out / "imagenette2-160.tgz"
    if not tgz.exists():
        print(f"downloading {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, tgz)  # noqa: S310
    digest = _sha256(tgz)
    if digest != SOURCE_SHA256:
        raise SystemExit(f"sha256 mismatch: expected {SOURCE_SHA256}, got {digest}")
    print(f"sha256 OK: {digest}")
    return tgz


def build(out: Path) -> tuple[Path, Path]:
    tgz = download(out)
    extract = out / "extract"
    extract.mkdir(exist_ok=True)
    with tarfile.open(tgz) as t:
        t.extractall(extract)
    root = next(p for p in extract.rglob("train") if p.is_dir()).parent

    def files(split: str):
        for cls in sorted(d for d in (root / split).iterdir() if d.is_dir()):
            for f in sorted(cls.glob("*")):
                if f.is_file():
                    yield split, cls.name, f

    full = out / "imagenette160.zip"
    with zipfile.ZipFile(full, "w", zipfile.ZIP_STORED) as z:
        for split, cls, f in list(files("train")) + list(files("val")):
            z.write(f, arcname=f"{split}/{cls}/{f.name}")
    print(f"built {full.name}")

    smoke = out / "smoke10.zip"
    with zipfile.ZipFile(smoke, "w", zipfile.ZIP_STORED) as z:
        for split in ("train", "val"):
            for cls in sorted(WNID):
                for f in sorted((root / split / cls).glob("*"))[:2]:
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
    from huggingface_hub import HfApi

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
