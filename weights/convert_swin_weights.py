"""Convert released Swin V1 classifiers to LibreYOLO checkpoints.

The four source graphs are the Microsoft Swin V1 release mirrored by timm:
tiny, small, and base use the ImageNet-1k checkpoints; large uses the released
ImageNet-22k-pretrained checkpoint fine-tuned on ImageNet-1k. The source
storage repository and weights are MIT. timm's loader and model definitions
are Apache-2.0. Learned tensors are unchanged; conversion adds only LibreYOLO
metadata and canonical ImageNet-1k names.

Usage::

    python weights/convert_swin_weights.py
    python weights/convert_swin_weights.py --size t
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
    "t": "swin_tiny_patch4_window7_224.ms_in1k",
    "s": "swin_small_patch4_window7_224.ms_in1k",
    "b": "swin_base_patch4_window7_224.ms_in1k",
    "l": "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
}
IMGSZ = {size: 224 for size in TAGS}


def convert(size: str) -> Path:
    """Download one released checkpoint, verify strict loading, and wrap it."""
    import timm

    add_repo_root_to_path()
    from libreyolo.models.swin.classifier import SwinClassifier
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    tag = TAGS[size]
    reference = timm.create_model(tag, pretrained=True)
    state_dict = reference.state_dict()

    native = SwinClassifier(size=size, num_classes=1000)
    result = native.load_state_dict(state_dict, strict=True)
    assert not result.missing_keys and not result.unexpected_keys, (
        result.missing_keys[:5],
        result.unexpected_keys[:5],
    )

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="swin",
        size=size,
        nc=1000,
        names=imagenet1k_names(),
        task="classify",
        imgsz=IMGSZ[size],
        supported_tasks=("classify",),
        default_task="classify",
    )
    validate_checkpoint_metadata(checkpoint, strict=True)

    output = Path("weights") / f"LibreSwin{size}-cls.pt"
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
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
    arguments = parser.parse_args()
    for selected_size in [arguments.size] if arguments.size else list(TAGS):
        convert(selected_size)
