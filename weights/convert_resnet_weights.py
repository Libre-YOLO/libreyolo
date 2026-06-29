"""Convert timm ResNet (a1_in1k) pretrained weights to LibreYOLO classify checkpoints.

timm's `resnet*.a1_in1k` checkpoints are Apache-2.0 (ImageNet-1k, the "ResNet
Strikes Back" A1 recipe), so they are redistributable and commercial-use OK.
The native `libreyolo.models.resnet.nn.ResNet` mirrors timm module names, so the
timm state_dict loads strict and inference is bit-identical (parity test).

Class names are the canonical ImageNet-1k labels, index-aligned to the standard
wnid-sorted class ordering (torchvision sorts wnid folders into timm's class-index
order), so `model.val()` reproduces the upstream benchmark and predictions expose
readable labels.

Usage::

    python weights/convert_resnet_weights.py            # all sizes
    python weights/convert_resnet_weights.py --size 50
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
    "18": "resnet18.a1_in1k",
    "34": "resnet34.a1_in1k",
    "50": "resnet50.a1_in1k",
    "101": "resnet101.a1_in1k",
}
IMGSZ = 224


def convert(size: str) -> Path:
    import timm

    add_repo_root_to_path()
    from libreyolo.models.resnet.nn import ResNet

    tag = TAGS[size]
    timm_model = timm.create_model(tag, pretrained=True)
    state_dict = timm_model.state_dict()

    # Sanity: strict-load the timm state dict into the native model.
    ours = ResNet(size=size, num_classes=1000)
    result = ours.load_state_dict(state_dict, strict=True)
    assert not result.missing_keys and not result.unexpected_keys, (
        result.missing_keys[:5],
        result.unexpected_keys[:5],
    )

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="resnet",
        size=size,
        nc=1000,
        names=imagenet1k_names(),
        task="classify",
        imgsz=IMGSZ,
    )

    out = Path("weights") / f"LibreResNet{size}-cls.pt"
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(out)  # atomic overwrite (Windows-safe on rerun)
    print(f"Wrote {out}  (timm {tag}, nc=1000, task=classify, imgsz={IMGSZ})")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["18", "34", "50", "101"], default=None,
                        help="Variant to convert (default: all).")
    args = parser.parse_args()
    for s in [args.size] if args.size else ["18", "34", "50", "101"]:
        convert(s)
