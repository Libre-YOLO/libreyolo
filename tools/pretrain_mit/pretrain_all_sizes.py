#!/usr/bin/env python3
"""ImageNet-1K classification pretraining sweep for every MiT encoder size (b0-b5).

Produces one encoder-only checkpoint per size under ``--output-dir``:
    mit_b0_imagenet1k_encoder.pt
    mit_b1_imagenet1k_encoder.pt
    ...

Each is consumable directly by ``LibreSegformer(size="b0", pretrained_encoder=
"runs/pretrain_mit/mit_b0_imagenet1k_encoder.pt")`` before an ADE20K/
Cityscapes fine-tune (see ``scripts/train_segformer_all_sizes.py``'s
``--pretrained-encoders-dir`` flag, which points at this same directory).

This sweep, and everything else in ``tools/pretrain_mit/``, is throwaway
tooling: it is never imported by ``libreyolo/`` and can be deleted once the
encoder checkpoints it produces are on disk. See ``README.md`` in this
directory.

Usage
-----
Cheap end-to-end pipeline check (imagenette160, few epochs, smallest two sizes):
    python tools/pretrain_mit/pretrain_all_sizes.py --data imagenette160 --sizes b0,b1 --epochs 5

Full ImageNet-1K sweep, all six sizes (multi-day job; point --data at a local
ImageFolder root with train/val sub-directories):
    python tools/pretrain_mit/pretrain_all_sizes.py --data /data/imagenet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_classify import ALL_SIZES, train_one_size  # noqa: E402

# b3-b5 have much deeper stage-3 encoders (18/27/40 blocks); halve the batch
# for those unless the user set --batch explicitly.
_LARGE_SIZES = ("b3", "b4", "b5")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="ImageFolder root with train/val, or a known name (e.g. imagenette160).")
    p.add_argument(
        "--sizes",
        default=",".join(ALL_SIZES),
        help=f"Comma-separated subset of {{{','.join(ALL_SIZES)}}}, smallest to largest (default: all six).",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=None, help="Default 256, auto-halved to 128 for b3/b4/b5.")
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.8, help="MixUp alpha (DeiT/PVT recipe: 0.8).")
    p.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha (DeiT/PVT recipe: 1.0).")
    p.add_argument("--random-erasing", type=float, default=0.25, dest="random_erasing",
                   help="RandomErasing probability (DeiT/PVT recipe: 0.25).")
    p.add_argument("--min-crop-scale", type=float, default=0.08, dest="min_crop_scale",
                   help="RandomResizedCrop min area fraction (DeiT/PVT recipe: 0.08).")
    p.add_argument("--auto-augment", default="randaugment", choices=["randaugment", "autoaugment", "augmix", "none"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true", default=False)
    p.add_argument("--no-ema", action="store_true", default=False)
    p.add_argument("--output-dir", default="runs/pretrain_mit")
    return p.parse_args()


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


def main() -> None:
    args = parse_args()
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    unknown = [s for s in sizes if s not in ALL_SIZES]
    if unknown:
        print(f"Unknown MiT size(s): {unknown}. Choose from {ALL_SIZES}.")
        sys.exit(1)

    summary = []
    failed = []

    for i, size in enumerate(sizes, start=1):
        batch = args.batch if args.batch is not None else (128 if size in _LARGE_SIZES else 256)
        size_args = argparse.Namespace(**{**vars(args), "size": size, "batch": batch, "resume": None})

        print(f"\n[{i}/{len(sizes)}] Pretraining MiT-{size} on ImageNet-1K (batch={batch}) ...")
        start = time.monotonic()
        try:
            result = train_one_size(size, size_args)
        except Exception as exc:  # noqa: BLE001 — a single size's failure must not abort the sweep
            elapsed = time.monotonic() - start
            print(f"  FAILED after {_format_duration(elapsed)}: {exc}")
            failed.append(size)
            summary.append({"size": size, "status": "failed", "error": str(exc), "elapsed": elapsed})
            continue

        elapsed = time.monotonic() - start
        print(f"  done in {_format_duration(elapsed)} — best top1: {result['best_top1']:.4f} — saved to {result['encoder_checkpoint']}")
        summary.append({"size": size, "status": "ok", "elapsed": elapsed, **result})

    print("\n" + "=" * 72)
    print(f"{'Size':<6}{'Status':<10}{'Best top1':<12}{'Wall-clock':<14}Encoder checkpoint")
    print("-" * 72)
    for row in summary:
        if row["status"] == "ok":
            print(
                f"{row['size']:<6}{'ok':<10}{row['best_top1']:<12.4f}"
                f"{_format_duration(row['elapsed']):<14}{row['encoder_checkpoint']}"
            )
        else:
            print(f"{row['size']:<6}{'failed':<10}{'-':<12}{_format_duration(row['elapsed']):<14}{row['error']}")
    print("=" * 72)

    if failed:
        print(f"\n{len(failed)}/{len(sizes)} size(s) failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
