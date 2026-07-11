#!/usr/bin/env python3
"""ImageNet-1K classification pretraining for one MiT encoder size.

This exists to give the LibreSegformer encoder a better-than-random starting
point before ADE20K/Cityscapes fine-tuning (see the repo's ``future_task.md``
history). It is deliberately NOT built on ``libreyolo.training.trainer.
BaseTrainer``: wiring into that shared trainer would require adding
``task="classify"`` support to ``LibreSegformer`` itself (a new
``SUPPORTED_TASKS`` entry, a classify-mode ``_preprocess``/``_postprocess``,
a registered trainer class) -- a permanent library change for a pipeline that
runs once per size and is discarded once it has produced encoder weights.

This script only reuses pure data-layer utilities that don't register
anything with the model system (``ClassifyDataset``, ``build_classify_
transforms``, ``build_classify_collate``, ``resolve_classify_data``) plus the
standalone LR scheduler / EMA helper classes. The only permanent trace this
leaves in ``libreyolo/`` is one small, generic constructor kwarg on
``LibreSegformer`` (``pretrained_encoder=``) that partial-loads the checkpoint
this script produces.

Checkpoint format written on every new best-val-accuracy epoch:
    {"encoder": <SegformerEncoder.state_dict()>, "size": "b0", "source": "imagenet1k-classify"}

Usage
-----
Validate the whole pipeline cheaply first (small dataset, few epochs):
    python tools/pretrain_mit/train_classify.py --size b0 --data imagenette160 --epochs 5

Full ImageNet-1K pretraining (point --data at a local ImageFolder root with
train/ and val/ sub-directories -- ImageNet-1K is not auto-downloadable):
    python tools/pretrain_mit/train_classify.py --size b0 --data /data/imagenet

Note on cost: DeiT-style recipes for small ViT-ish encoders typically use
~300 epochs on the full 1.28M-image ImageNet-1K train split. The default here
is 300 epochs to match that recipe (multi-day-per-size cost); lower --epochs
for a cheaper partial-pretraining starting point that still beats random init.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from classify_head import MiTClassifier  # noqa: E402

from libreyolo.data.classify_dataset import (  # noqa: E402
    ClassifyDataset,
    build_classify_collate,
    classify_collate_fn,
    get_class_names,
    resolve_classify_data,
)
from libreyolo.training.ema import ModelEMA  # noqa: E402
from libreyolo.training.scheduler import CosineAnnealingScheduler  # noqa: E402
from libreyolo.utils.serialization import load_trusted_torch_file  # noqa: E402

ALL_SIZES = ("b0", "b1", "b2", "b3", "b4", "b5")

# DeiT/PVT calibrate the base LR for a global batch of 1024; --lr0 is scaled
# linearly from this reference to the actual batch (see train_one_size).
REFERENCE_BATCH = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", required=True, choices=ALL_SIZES)
    p.add_argument(
        "--data",
        required=True,
        help="ImageFolder root with train/val sub-dirs, or a known name (e.g. imagenette160) for a cheap pipeline check.",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=256)
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
    p.add_argument("--resume", default=None, help="Path to a previous full training-state checkpoint (last.pt) to resume from.")
    return p.parse_args()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def soft_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy that accepts hard int labels (val split) or the soft,
    class-probability labels MixUp/CutMix produce for the train split."""
    if labels.dim() == 1:
        return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    log_probs = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0:
        num_classes = logits.shape[-1]
        labels = labels * (1 - label_smoothing) + label_smoothing / num_classes
    return -(labels * log_probs).sum(dim=-1).mean()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    for imgs, labels, _, _ in tqdm(loader, desc="  eval", leave=False, unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        maxk = min(5, logits.shape[-1])
        _, pred = logits.topk(maxk, dim=1)
        correct = pred.eq(labels.view(-1, 1))
        top1_correct += correct[:, :1].any(dim=1).sum().item()
        top5_correct += correct[:, :maxk].any(dim=1).sum().item()
        total += labels.shape[0]
    return top1_correct / max(total, 1), top5_correct / max(total, 1)


def train_one_size(size: str, args: argparse.Namespace) -> dict:
    """Run the full classification pretraining loop for one encoder size.

    Returns a summary dict; also writes, on every new best-val-top1 epoch, the
    encoder-only checkpoint that ``LibreSegformer(pretrained_encoder=...)``
    consumes.
    """
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir) / f"mit_{size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_ckpt_path = Path(args.output_dir) / f"mit_{size}_imagenet1k_encoder.pt"
    metrics_path = output_dir / "metrics.jsonl"

    dataset_root = resolve_classify_data(args.data)
    class_names = get_class_names(dataset_root)
    num_classes = len(class_names)

    auto_augment = None if args.auto_augment == "none" else args.auto_augment
    train_ds = ClassifyDataset(
        dataset_root,
        split="train",
        imgsz=args.imgsz,
        augment=True,
        transform_kwargs={
            "auto_augment": auto_augment,
            "erasing": args.random_erasing,
            "scale": (args.min_crop_scale, 1.0),
        },
    )
    val_ds = ClassifyDataset(
        dataset_root,
        split="val",
        imgsz=args.imgsz,
        augment=False,
        class_to_idx=train_ds.class_to_idx,
    )

    collate_fn = build_classify_collate(num_classes, mixup=args.mixup, cutmix=args.cutmix)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=classify_collate_fn,
        pin_memory=(device == "cuda"),
    )

    model = MiTClassifier(size=size, num_classes=num_classes).to(device)
    # DeiT/PVT linear LR scaling: --lr0 is calibrated for a global batch of
    # REFERENCE_BATCH (1024). Scale it to the actual batch so every size --
    # including the ones the launcher auto-halves to 128 -- trains at the
    # paper-consistent rate instead of a fixed (too-high) LR.
    effective_lr = args.lr0 * args.batch / REFERENCE_BATCH
    print(f"[{size}] effective LR: {effective_lr:.2e} "
          f"(--lr0 {args.lr0:.2e} x batch {args.batch} / {REFERENCE_BATCH})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingScheduler(
        lr=effective_lr,
        iters_per_epoch=len(train_loader),
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
    )
    amp_enabled = (not args.no_amp) and device == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)
    ema = None if args.no_ema else ModelEMA(model, decay=0.9998)

    start_epoch = 1
    best_top1 = 0.0
    if args.resume:
        state = load_trusted_torch_file(args.resume, map_location="cpu", context="MiT classify resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if ema is not None and state.get("ema") is not None:
            ema.ema.load_state_dict(state["ema"])
        start_epoch = state["epoch"] + 1
        best_top1 = state.get("best_top1", 0.0)

    global_iter = (start_epoch - 1) * len(train_loader)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.monotonic()
        running_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"[{size}] epoch {epoch}/{args.epochs}",
            unit="batch",
            leave=False,
        )
        for step, (imgs, labels, _, _) in enumerate(pbar, start=1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lr = scheduler.update_lr(global_iter)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp_enabled):
                logits = model(imgs)
                loss = soft_cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            running_loss += loss.item()
            global_iter += 1
            pbar.set_postfix(loss=f"{running_loss / step:.4f}", lr=f"{lr:.2e}")

        eval_model = ema.ema if ema is not None else model
        val_top1, val_top5 = evaluate(eval_model, val_loader, device)
        train_loss = running_loss / max(len(train_loader), 1)
        elapsed = time.monotonic() - epoch_start

        is_best = val_top1 > best_top1
        best_top1 = max(best_top1, val_top1)

        with open(metrics_path, "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "val/top1": val_top1,
                        "val/top5": val_top5,
                        "best_top1": best_top1,
                        "lr": lr,
                        "time": elapsed,
                    }
                )
                + "\n"
            )
        print(
            f"[{size}] epoch {epoch}/{args.epochs} loss={train_loss:.4f} "
            f"val_top1={val_top1:.4f} val_top5={val_top5:.4f} ({elapsed:.1f}s)"
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.ema.state_dict() if ema is not None else None,
                "epoch": epoch,
                "best_top1": best_top1,
            },
            output_dir / "last.pt",
        )
        if is_best:
            encoder_source = eval_model.encoder if hasattr(eval_model, "encoder") else model.encoder
            torch.save(
                {"encoder": encoder_source.state_dict(), "size": size, "source": "imagenet1k-classify"},
                encoder_ckpt_path,
            )

    return {
        "size": size,
        "best_top1": best_top1,
        "encoder_checkpoint": str(encoder_ckpt_path),
        "num_classes": num_classes,
    }


def main() -> None:
    args = parse_args()
    result = train_one_size(args.size, args)
    print(f"\nDone. Best val top1={result['best_top1']:.4f}. Encoder saved to {result['encoder_checkpoint']}")


if __name__ == "__main__":
    main()
