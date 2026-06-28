"""Convert timm MobileNetV4-conv pretrained weights to LibreYOLO classify checkpoints.

The timm framework code *and* these specific checkpoints are Apache-2.0
(ImageNet-1k only, trained by Ross Wightman — no distillation, no extra data),
so they are redistributable and commercial-use OK. See the family NOTICE.

The native ``libreyolo.models.mobilenetv4.nn.MobileNetV4`` mirrors timm's module
names, so the timm ``state_dict`` loads with ``strict=True`` and is bit-identical
on inference (proven by the parity test). This script just metadata-wraps it.

Names are stored as indexed placeholders so ``model.val()`` reproduces the
upstream benchmark on a standard ImageNet ImageFolder (torchvision sorts the
wnid class folders into exactly timm's class-index order).

Usage::

    python weights/convert_mobilenetv4_weights.py            # all sizes
    python weights/convert_mobilenetv4_weights.py --size s
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    build_class_names,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

TAGS = {
    "s": "mobilenetv4_conv_small.e2400_r224_in1k",
    "m": "mobilenetv4_conv_medium.e500_r224_in1k",
    "l": "mobilenetv4_conv_large.e500_r256_in1k",
}
IMGSZ = {"s": 224, "m": 224, "l": 256}


def convert(size: str) -> Path:
    import timm

    add_repo_root_to_path()
    from libreyolo.models.mobilenetv4.nn import MobileNetV4

    tag = TAGS[size]
    timm_model = timm.create_model(tag, pretrained=True)
    state_dict = timm_model.state_dict()

    # Sanity: the timm state dict must load into the native model with no
    # missing/unexpected keys (the precondition for bit-exact parity).
    ours = MobileNetV4(size=size, num_classes=1000)
    result = ours.load_state_dict(state_dict, strict=True)
    assert not result.missing_keys and not result.unexpected_keys, (
        result.missing_keys[:5],
        result.unexpected_keys[:5],
    )

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="mobilenetv4",
        size=size,
        nc=1000,
        names=build_class_names(1000),
        task="classify",
        imgsz=IMGSZ[size],
    )

    out = Path("weights") / f"LibreMobileNetV4{size}-cls.pt"
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.rename(out)  # atomic
    print(f"Wrote {out}  (timm {tag}, nc=1000, task=classify, imgsz={IMGSZ[size]})")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["s", "m", "l"], default=None,
                        help="Variant to convert (default: all).")
    args = parser.parse_args()
    for s in [args.size] if args.size else ["s", "m", "l"]:
        convert(s)
