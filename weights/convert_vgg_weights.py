"""Convert official torchvision VGG weights to LibreYOLO checkpoints.

Source: https://github.com/pytorch/vision at
``10f68dbd78b9aa5cab9328f3b2e99cfb0b608122`` (BSD-3-Clause).
The source state dict is wrapped with LibreYOLO metadata; learned tensors and
their names are unchanged.

Usage::

    python weights/convert_vgg_weights.py
    python weights/convert_vgg_weights.py --size 16bn
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    imagenet1k_names,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

SIZES = ("16", "19", "16bn", "19bn")
IMGSZ = 224


def _official_state_dict(size: str):
    from torchvision.models import (
        VGG16_BN_Weights,
        VGG16_Weights,
        VGG19_BN_Weights,
        VGG19_Weights,
    )

    weights_by_size = {
        "16": VGG16_Weights.IMAGENET1K_V1,
        "19": VGG19_Weights.IMAGENET1K_V1,
        "16bn": VGG16_BN_Weights.IMAGENET1K_V1,
        "19bn": VGG19_BN_Weights.IMAGENET1K_V1,
    }
    return weights_by_size[size].get_state_dict(progress=True, check_hash=True)


def convert(size: str, output_dir: str | Path = "weights") -> Path:
    if size not in SIZES:
        raise ValueError(f"Unknown VGG size {size!r}; choose from {SIZES}.")

    state_dict = _official_state_dict(size)

    add_repo_root_to_path()
    from libreyolo.models.vgg.nn import VGG

    native = VGG(size=size, num_classes=1000, init_weights=False)
    incompatible = native.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "VGG state-dict mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    print("missing: []")
    print("unexpected: []")

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="vgg",
        size=size,
        nc=1000,
        names=imagenet1k_names(),
        task="classify",
        imgsz=IMGSZ,
        supported_tasks=("classify",),
        default_task="classify",
    )

    output = Path(output_dir) / f"LibreVGG{size}-cls.pt"
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
    temporary.replace(output)
    print(f"Wrote {output} (nc=1000, task=classify, imgsz={IMGSZ})")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, default=None)
    parser.add_argument("--output-dir", default="weights")
    args = parser.parse_args()
    for variant in (args.size,) if args.size else SIZES:
        convert(variant, args.output_dir)
