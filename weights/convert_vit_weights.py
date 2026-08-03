"""Convert Apache-2.0 timm AugReg ViT weights to LibreYOLO checkpoints.

The native ``libreyolo.models.vit.nn.VisionTransformer`` mirrors the classic
ViT module names in timm v1.0.28, pinned to commit
``8ef73809f622e0031bd7f4940265734aef8b9978``. Conversion therefore preserves
every learned tensor unchanged and adds only LibreYOLO checkpoint metadata.

All four source model cards explicitly declare ``license: apache-2.0``. The
checkpoints are redistributable and are rehosted with their upstream license
and attribution notice. See ``libreyolo/models/vit/NOTICE``.

Usage::

    python weights/convert_vit_weights.py
    python weights/convert_vit_weights.py --size ti
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
    "ti": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "s": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "b": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "l": "vit_large_patch16_224.augreg_in21k_ft_in1k",
}


def convert(size: str) -> Path:
    """Download, strictly validate, and metadata-wrap one ViT tier."""
    import timm

    add_repo_root_to_path()
    from libreyolo.models.vit.nn import VisionTransformer

    tag = TAGS[size]
    reference = timm.create_model(tag, pretrained=True)
    state_dict = reference.state_dict()

    native = VisionTransformer(size=size, num_classes=1000, init_weights=False)
    result = native.load_state_dict(state_dict, strict=True)
    assert not result.missing_keys and not result.unexpected_keys, (
        result.missing_keys[:5],
        result.unexpected_keys[:5],
    )

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="vit",
        size=size,
        nc=1000,
        names=imagenet1k_names(),
        task="classify",
        imgsz=224,
    )

    output = Path("weights") / f"LibreViT{size}-cls.pt"
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
    temporary.replace(output)
    print(f"Wrote {output} (timm {tag}, nc=1000, task=classify, imgsz=224)")
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
    for selected_size in [args.size] if args.size else list(TAGS):
        convert(selected_size)
