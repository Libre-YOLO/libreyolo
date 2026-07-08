"""Build a paired super-resolution validation set the license-compliant way.

We never host DIV2K: the classic SR training set is widely described as
research-only, so LibreYOLO does not redistribute it. Instead this script builds
low/high-resolution pairs by bicubic-downscaling images YOU supply, and it gates
on a permissive license so a hosted set can be rebuilt from CC-BY sources (an
OpenImages slice, verified per image) rather than from DIV2K.

Two modes:

1. Synthetic (default, no external data): generate deterministic structured
   images and their bicubic downscales. Good for regression / plumbing gates
   only - synthetic bicubic pairs do NOT measure real-degradation SR quality.

       python scripts/build_sr_valset.py --out DATASETS_DIR/sr8 --scale 4 --n 8

2. From your own images (e.g. a CC-BY OpenImages slice): downscale each HR image
   bicubic to make the LR input. The LICENSE GATE runs first: every source image
   must be listed in a `licenses.csv` (image_id,license,attribution) whose
   license is a CC-BY* / public-domain / permissive value, or the build aborts.

       python scripts/build_sr_valset.py --out DATASETS_DIR/sr-val --scale 4 \
           --images ./cc_by_images --licenses ./cc_by_images/licenses.csv

Output layout (paired restore-dataset convention, matches sr8.yaml):
    <out>/inputs/val/<name>.png    LR input
    <out>/targets/val/<name>.png    HR target (scale x the input)
    <out>/ATTRIBUTION.txt          per-image license + attribution (from --licenses)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

# License values accepted by the gate. CC-BY* requires attribution (recorded in
# ATTRIBUTION.txt); DIV2K / research-only / non-commercial values are rejected.
_ALLOWED_LICENSE_TOKENS = (
    "cc-by",
    "cc by",
    "cc0",
    "public domain",
    "publicdomain",
    "pdm",
    "u.s. government",
)
_REJECTED_LICENSE_TOKENS = (
    "nc",
    "non-commercial",
    "noncommercial",
    "nd",
    "no-deriv",
    "research only",
    "research-only",
    "div2k",
)


def _license_ok(value: str) -> bool:
    v = value.strip().lower()
    if any(bad in v for bad in _REJECTED_LICENSE_TOKENS):
        return False
    return any(good in v for good in _ALLOWED_LICENSE_TOKENS)


def _read_license_gate(licenses_csv: Path) -> dict[str, tuple[str, str]]:
    """Return {image_id: (license, attribution)}; abort on any non-permissive row."""
    approved: dict[str, tuple[str, str]] = {}
    rejected: list[tuple[str, str]] = []
    with open(licenses_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_id = (row.get("image_id") or row.get("id") or "").strip()
            lic = (row.get("license") or row.get("License") or "").strip()
            attribution = (row.get("attribution") or row.get("author") or "").strip()
            if not image_id:
                continue
            if _license_ok(lic):
                approved[image_id] = (lic, attribution)
            else:
                rejected.append((image_id, lic))
    if rejected:
        preview = ", ".join(f"{i} ({l!r})" for i, l in rejected[:5])
        raise SystemExit(
            f"License gate failed: {len(rejected)} image(s) are not permissive "
            f"(CC-BY* / public-domain), e.g. {preview}. Exclude them or fix the "
            "license manifest. DIV2K and non-commercial sources are not hosted."
        )
    if not approved:
        raise SystemExit("License gate produced no approved images.")
    return approved


def _synthetic_hr(index: int, size: int) -> Image.Image:
    rng = np.random.default_rng(1000 + index)
    low = rng.random((size // 8, size // 8, 3)).astype(np.float32)
    base = np.array(
        Image.fromarray((low * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC),
        dtype=np.float32,
    )
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base[..., 0] = 0.6 * base[..., 0] + 0.4 * (128 + 90 * np.sin(xx / 11.0))
    base[..., 1] = 0.6 * base[..., 1] + 0.4 * (128 + 90 * np.cos(yy / 13.0))
    return Image.fromarray(base.clip(0, 255).astype(np.uint8), "RGB")


def _make_pair(hr: Image.Image, scale: int) -> tuple[Image.Image, Image.Image]:
    w, h = hr.size
    w -= w % scale
    h -= h % scale
    hr = hr.crop((0, 0, w, h))
    lr = hr.resize((w // scale, h // scale), Image.BICUBIC)
    return lr, hr


def build(out: Path, scale: int, n: int, images: Path | None, licenses: Path | None,
          hr_size: int) -> None:
    in_dir = out / "inputs" / "val"
    tgt_dir = out / "targets" / "val"
    in_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    attributions: list[str] = []
    if images is not None:
        if licenses is None:
            raise SystemExit("--images requires --licenses (per-image license gate).")
        approved = _read_license_gate(Path(licenses))
        srcs = sorted(
            p for p in Path(images).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and p.stem in approved
        )
        if not srcs:
            raise SystemExit("No source images matched approved license entries.")
        for p in srcs[: n or len(srcs)]:
            hr = Image.open(p).convert("RGB")
            lr, hr = _make_pair(hr, scale)
            lr.save(in_dir / f"{p.stem}.png")
            hr.save(tgt_dir / f"{p.stem}.png")
            lic, attr = approved[p.stem]
            attributions.append(f"{p.stem}: {lic}" + (f" - {attr}" if attr else ""))
    else:
        for i in range(n):
            lr, hr = _make_pair(_synthetic_hr(i, hr_size), scale)
            lr.save(in_dir / f"{i}.png")
            hr.save(tgt_dir / f"{i}.png")
        attributions.append("Synthetic structured images (no external data).")

    (out / "ATTRIBUTION.txt").write_text(
        "Super-resolution pairs built by scripts/build_sr_valset.py\n"
        f"scale={scale}\n\n" + "\n".join(attributions) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(list(in_dir.glob('*.png')))} LR/HR pairs (scale {scale}) to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Output dataset root")
    ap.add_argument("--scale", type=int, default=4, choices=[2, 4])
    ap.add_argument("--n", type=int, default=8, help="Number of pairs (0 = all with --images)")
    ap.add_argument("--images", default=None, help="Directory of CC-BY source images (HR)")
    ap.add_argument("--licenses", default=None, help="CSV: image_id,license,attribution")
    ap.add_argument("--hr-size", type=int, default=256, help="Synthetic HR image size")
    args = ap.parse_args()
    build(
        Path(args.out), args.scale, args.n,
        Path(args.images) if args.images else None,
        Path(args.licenses) if args.licenses else None,
        args.hr_size,
    )
