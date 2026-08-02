"""Convert Apache-2.0 timm DeiT weights to LibreYOLO checkpoints.

The source models are the plain ImageNet-1k DeiT patch-16 releases mirrored by
timm under explicit Apache-2.0 terms. Learned tensors are not modified: the
native LibreYOLO graph mirrors the timm 1.0.28 state-dict surface, so conversion
is a strict-load check followed by checkpoint metadata wrapping.

Sources:
    https://github.com/facebookresearch/deit (Apache-2.0)
    https://github.com/huggingface/pytorch-image-models (Apache-2.0)

Usage::

    python weights/convert_deit_weights.py
    python weights/convert_deit_weights.py --size t
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

TAGS = {
    "t": "deit_tiny_patch16_224.fb_in1k",
    "s": "deit_small_patch16_224.fb_in1k",
    "b": "deit_base_patch16_224.fb_in1k",
}
IMGSZ = {"t": 224, "s": 224, "b": 224}
EXPECTED_CROP_PCT = 0.9


def convert(size: str) -> Path:
    """Download one official timm model and wrap its unchanged tensors."""
    import timm

    add_repo_root_to_path()
    from libreyolo.models.deit.nn import DeiT

    tag = TAGS[size]
    source = timm.create_model(tag, pretrained=True).eval()
    cfg = source.pretrained_cfg
    crop_pct = float(cfg.get("crop_pct", 0.0))
    interpolation = str(cfg.get("interpolation", ""))
    if crop_pct != EXPECTED_CROP_PCT or interpolation != "bicubic":
        raise RuntimeError(
            f"Upstream preprocessing changed for {tag}: "
            f"crop_pct={crop_pct}, interpolation={interpolation!r}."
        )

    state_dict = source.state_dict()
    native = DeiT(size=size, num_classes=1000)
    result = native.load_state_dict(state_dict, strict=True)
    print("missing:", result.missing_keys)
    print("unexpected:", result.unexpected_keys)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Strict DeiT state-dict load failed for {tag}.")

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="deit",
        size=size,
        nc=1000,
        names=imagenet1k_names(),
        task="classify",
        imgsz=IMGSZ[size],
        supported_tasks=("classify",),
        default_task="classify",
    )

    output = Path("weights") / f"LibreDeiT{size}-cls.pt"
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(wrapped, temporary)
    temporary.replace(output)
    print(
        f"Wrote {output} (timm {tag}, nc=1000, task=classify, "
        f"imgsz={IMGSZ[size]})"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        choices=list(TAGS),
        default=None,
        help="Variant to convert (default: all).",
    )
    args = parser.parse_args()
    for requested_size in [args.size] if args.size else list(TAGS):
        convert(requested_size)
