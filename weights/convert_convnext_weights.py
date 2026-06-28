"""Convert timm ConvNeXt V1 pretrained weights to LibreYOLO classify checkpoints.

The timm framework code is Apache-2.0 and the original ConvNeXt code is MIT;
the ``convnext_{tiny,small,base}.fb_in1k`` checkpoints are ImageNet-1k only
(no extra data, no distillation) and Apache-2.0 in timm, so they are
redistributable and commercial-use OK. See the family NOTICE.

ConvNeXt-V2's small checkpoints are CC-BY-NC and are deliberately NOT handled
here.

The native ``libreyolo.models.convnext.nn.ConvNeXt`` mirrors timm's module
names, so the timm ``state_dict`` loads with ``strict=True`` and is bit-identical
on inference (proven by the parity test). This script just metadata-wraps it.

Names are stored as indexed placeholders so ``model.val()`` reproduces the
upstream benchmark on a standard ImageNet ImageFolder (torchvision sorts the
wnid class folders into exactly timm's class-index order).

Usage::

    python weights/convert_convnext_weights.py            # all sizes
    python weights/convert_convnext_weights.py --size t
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
    "t": "convnext_tiny.fb_in1k",
    "s": "convnext_small.fb_in1k",
    "b": "convnext_base.fb_in1k",
}
IMGSZ = {"t": 224, "s": 224, "b": 224}


def convert(size: str) -> Path:
    import timm

    add_repo_root_to_path()
    from libreyolo.models.convnext.nn import ConvNeXt

    tag = TAGS[size]
    timm_model = timm.create_model(tag, pretrained=True)
    state_dict = timm_model.state_dict()

    # Sanity: the timm state dict must load into the native model with no
    # missing/unexpected keys (the precondition for bit-exact parity).
    ours = ConvNeXt(size=size, num_classes=1000)
    result = ours.load_state_dict(state_dict, strict=True)
    assert not result.missing_keys and not result.unexpected_keys, (
        result.missing_keys[:5],
        result.unexpected_keys[:5],
    )

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="convnext",
        size=size,
        nc=1000,
        names=build_class_names(1000),
        task="classify",
        imgsz=IMGSZ[size],
    )

    out = Path("weights") / f"LibreConvNeXt{size}-cls.pt"
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(out)  # atomic overwrite (Windows-safe on rerun)
    print(f"Wrote {out}  (timm {tag}, nc=1000, task=classify, imgsz={IMGSZ[size]})")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["t", "s", "b"], default=None,
                        help="Variant to convert (default: all).")
    args = parser.parse_args()
    for s in [args.size] if args.size else ["t", "s", "b"]:
        convert(s)
