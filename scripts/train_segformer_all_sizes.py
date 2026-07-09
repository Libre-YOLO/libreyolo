#!/usr/bin/env python3
"""Train every LibreSegformer size (b0 through b5), smallest to largest.

LibreSegformer ships no pretrained weights for any size: the upstream
``nvidia/segformer-b0..b5-*`` checkpoints are under a non-permissive,
non-commercial NVIDIA license and are never mirrored or converted (see
``libreyolo/models/segformer/NOTICE``). Every size trains fully from scratch.
"Smallest to largest" is wall-clock ordering only, not a curriculum or
warm-start chain — each size gets a fresh random init.

``--data`` accepts either a built-in dataset name (e.g. ``ade20k``, resolved
against libreyolo/config/datasets/ade20k.yaml — no ``.yaml`` needed, though
``ade20k.yaml`` also works) or a path to your own semantic dataset YAML (see
docs/dataset_schema.md). Built-in ``ade20k``/``cityscapes`` are manual-download
datasets: the YAML only describes the label space; you must place the actual
images/masks under ``$LIBREYOLO_DATASETS_DIR`` (default ``~/datasets``) first
— see the comment at the top of the respective YAML file for the exact layout.

Usage
-----
All six sizes, defaults, single GPU:
    python scripts/train_segformer_all_sizes.py --data ade20k

Only the two smallest sizes, quick check:
    python scripts/train_segformer_all_sizes.py --data ade20k --sizes b0,b1 --epochs 2

Custom batch/imgsz/lr, single GPU:
    python scripts/train_segformer_all_sizes.py --data ade20k --batch 16 --imgsz 512 --lr0 6e-4

Two GPUs (DDP is spawned automatically — no torchrun needed):
    python scripts/train_segformer_all_sizes.py --data ade20k --devices 0,1

Your own dataset YAML instead of a built-in name:
    python scripts/train_segformer_all_sizes.py --data /path/to/data.yaml

Note: b3-b5 have much deeper encoders (18/27/40 transformer blocks in stage 3)
than b0-b2, so activation memory rises sharply. --batch is auto-halved for
b3/b4/b5 unless you pass --batch explicitly.

Note: --allow-download-scripts is required only if your dataset YAML embeds a
Python download block. Only pass it for datasets you trust.

Note: --pretrained-encoders-dir points at encoder-only checkpoints produced by
tools/pretrain_mit/ (ImageNet-1K classification pretraining, external to this
library -- see that directory's README). For each size, if
<dir>/mit_<size>_imagenet1k_encoder.pt exists it is loaded before training;
sizes without a matching file fall back to random init with a warning.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running directly from the repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ALL_SIZES = ("b0", "b1", "b2", "b3", "b4", "b5")
# b3-b5 have much deeper encoders (18/27/40 blocks in stage 3); halve the
# batch for those unless the user set --batch explicitly.
_LARGE_SIZES = ("b3", "b4", "b5")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", required=True, help="Path to a semantic dataset YAML (see docs/dataset_schema.md).")
    p.add_argument(
        "--sizes",
        default=",".join(ALL_SIZES),
        help=f"Comma-separated subset of {{{','.join(ALL_SIZES)}}}, smallest to largest (default: all six).",
    )
    p.add_argument("--epochs", type=int, default=160, help="Epochs per size (default: 160, shared across sizes).")
    p.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Global batch size for every size (default: 8, auto-halved to 4 for b3/b4/b5).",
    )
    p.add_argument("--imgsz", type=int, default=512, help="Input size, must be divisible by 32 (default: 512).")
    p.add_argument("--lr0", type=float, default=6e-4, help="Learning rate, applied to every size (default: 6e-4).")
    p.add_argument("--devices", default="0", help="GPU index or comma-separated indices (default: 0).")
    p.add_argument("--workers", type=int, default=4, help="DataLoader worker processes (default: 4).")
    p.add_argument("--output-root", default="runs/train", help="Each size writes to <output-root>/segformer_<size>/.")
    p.add_argument(
        "--allow-download-scripts",
        action="store_true",
        default=False,
        dest="allow_download_scripts",
        help="Allow the data YAML's embedded Python download block to run. Only use for datasets you trust.",
    )
    p.add_argument(
        "--pretrained-encoders-dir",
        default=None,
        help="Directory of tools/pretrain_mit/ encoder checkpoints (mit_<size>_imagenet1k_encoder.pt). Optional.",
    )
    return p.parse_args()


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"


def train_one_size(size: str, args: argparse.Namespace) -> dict:
    from libreyolo.models.segformer.model import LibreSegformer

    batch = args.batch
    if batch is None:
        batch = 4 if size in _LARGE_SIZES else 8

    pretrained_encoder = None
    if args.pretrained_encoders_dir:
        candidate = Path(args.pretrained_encoders_dir) / f"mit_{size}_imagenet1k_encoder.pt"
        if candidate.exists():
            pretrained_encoder = str(candidate)
            print(f"  Loading pretrained encoder: {candidate}")
        else:
            print(f"  No pretrained encoder found at {candidate}; training {size} from scratch.")

    output_dir = f"{args.output_root}/segformer_{size}"
    model = LibreSegformer(
        model_path=None, size=size, task="semantic", device="auto", pretrained_encoder=pretrained_encoder
    )

    result = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        device=args.devices,
        workers=args.workers,
        project=args.output_root,
        name=f"segformer_{size}",
        exist_ok=True,
        allow_download_scripts=args.allow_download_scripts,
    )
    result["_output_dir"] = output_dir
    result["_batch"] = batch
    return result


def main() -> None:
    args = parse_args()
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    unknown = [s for s in sizes if s not in ALL_SIZES]
    if unknown:
        print(f"Unknown SegFormer size(s): {unknown}. Choose from {ALL_SIZES}.")
        sys.exit(1)

    summary = []
    failed = []

    for i, size in enumerate(sizes, start=1):
        print(f"\n[{i}/{len(sizes)}] Training SegFormer-{size} ...")
        start = time.monotonic()
        try:
            result = train_one_size(size, args)
        except Exception as exc:  # noqa: BLE001 — a single size's failure must not abort the sweep
            elapsed = time.monotonic() - start
            print(f"  FAILED after {_format_duration(elapsed)}: {exc}")
            failed.append(size)
            summary.append(
                {"size": size, "status": "failed", "error": str(exc), "elapsed": elapsed}
            )
            continue

        elapsed = time.monotonic() - start
        val_metrics = (result.get("epoch_metrics") or [{}])[-1].get("val_metrics", {}) or {}
        best_miou = val_metrics.get("metrics/mIoU", 0.0)
        checkpoint = result.get("best_checkpoint") or result.get("last_checkpoint")
        print(
            f"  done in {_format_duration(elapsed)} — best mIoU: {best_miou:.4f} "
            f"— saved to {checkpoint}"
        )
        summary.append(
            {
                "size": size,
                "status": "ok",
                "best_miou": best_miou,
                "checkpoint": checkpoint,
                "elapsed": elapsed,
            }
        )

    print("\n" + "=" * 72)
    print(f"{'Size':<6}{'Status':<10}{'Best mIoU':<12}{'Wall-clock':<12}Checkpoint")
    print("-" * 72)
    for row in summary:
        if row["status"] == "ok":
            print(
                f"{row['size']:<6}{'ok':<10}{row['best_miou']:<12.4f}"
                f"{_format_duration(row['elapsed']):<12}{row['checkpoint']}"
            )
        else:
            print(f"{row['size']:<6}{'failed':<10}{'-':<12}{_format_duration(row['elapsed']):<12}{row['error']}")
    print("=" * 72)

    if failed:
        print(f"\n{len(failed)}/{len(sizes)} size(s) failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
