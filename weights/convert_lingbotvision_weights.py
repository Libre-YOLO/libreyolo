"""Convert upstream LingBot-Vision backbone weights to a LibreYOLO checkpoint.

Upstream (https://github.com/robbyant/lingbot-vision, Apache-2.0) publishes
backbone-only ``model.pt`` files — there is no upstream task head. This script
wraps the backbone into a LibreLingBotVision semantic checkpoint with a
freshly initialized 1x1 dense head, ready to fine-tune (or to receive a
LibreYOLO-trained head via ``--head`` from a training run's best.pt).

Usage:
    # Backbone + fresh head (training-init checkpoint):
    python weights/convert_lingbotvision_weights.py model.pt \
        weights/LibreLingBotVisions-sem.pt --size s --nc 150

    # Backbone + head taken from a LibreYOLO training run:
    python weights/convert_lingbotvision_weights.py model.pt \
        weights/LibreLingBotVisions-sem.pt --size s --nc 150 \
        --head runs/train/lingbotvision_exp/weights/best.pt \
        --names-from runs/train/lingbotvision_exp/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


def convert(
    input_path: str,
    output_path: str,
    size: str,
    nc: int,
    head_path: str | None = None,
    names_from: str | None = None,
    imgsz: int = 512,
) -> None:
    add_repo_root_to_path()
    from libreyolo.models.lingbotvision.nn import LingBotVisionSemanticSegmenter

    raw = load_checkpoint(input_path)
    backbone_sd = extract_state_dict(raw, prefer_ema=False)
    # Upstream checkpoints are either a raw state dict or {"backbone": sd};
    # keys may carry a "backbone." prefix from full-training dumps.
    if all(k.startswith("backbone.") for k in backbone_sd):
        backbone_sd = {k[len("backbone.") :]: v for k, v in backbone_sd.items()}
    print(f"Extracted {len(backbone_sd)} backbone tensors from {input_path}")

    model = LingBotVisionSemanticSegmenter(size=size, num_classes=nc)
    result = model.backbone.load_state_dict(backbone_sd, strict=True)
    print(f"Backbone loaded: missing={list(result.missing_keys)} unexpected={list(result.unexpected_keys)}")

    names = None
    if head_path is not None:
        head_ckpt = load_checkpoint(head_path)
        head_sd = extract_state_dict(head_ckpt, prefer_ema=False)
        head_keys = {k: v for k, v in head_sd.items() if k.startswith("predict.")}
        if not head_keys:
            raise SystemExit(f"--head {head_path} carries no predict.* keys")
        result = model.load_state_dict(head_keys, strict=False)
        if result.unexpected_keys:
            raise SystemExit(f"Unexpected head keys: {result.unexpected_keys}")
        print(f"Head loaded from {head_path} ({len(head_keys)} tensors)")
    if names_from is not None:
        names_ckpt = load_checkpoint(names_from)
        if isinstance(names_ckpt, dict) and isinstance(names_ckpt.get("names"), dict):
            names = names_ckpt["names"]
            print(f"Names taken from {names_from} ({len(names)} classes)")

    wrapped = wrap_libreyolo_checkpoint(
        model.state_dict(),
        model_family="lingbotvision",
        size=size,
        nc=nc,
        names=names,
        task="semantic",
        imgsz=imgsz,
    )

    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.rename(out)
    print(f"Wrote {out} (size={size}, nc={nc})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Upstream lingbot-vision backbone checkpoint (model.pt)")
    p.add_argument("output", help="Output LibreLingBotVision<size>-sem.pt")
    p.add_argument("--size", required=True, choices=["s", "b", "l", "g"])
    p.add_argument("--nc", type=int, default=150)
    p.add_argument("--head", default=None, help="Optional checkpoint providing trained predict.* keys")
    p.add_argument("--names-from", default=None, help="Optional checkpoint providing the names dict")
    p.add_argument("--imgsz", type=int, default=512)
    args = p.parse_args()
    convert(args.input, args.output, args.size, args.nc, args.head, args.names_from, args.imgsz)
