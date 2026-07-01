"""Convert timm EfficientNetV2-base pretrained weights to LibreYOLO classify checkpoints.

The timm framework code *and* these specific checkpoints are Apache-2.0
(ImageNet-1k only, ported by Ross Wightman — no ImageNet-21k, no JFT, no extra
data), so they are redistributable and commercial-use OK. See the family NOTICE.

The native ``libreyolo.models.efficientnetv2.nn.EfficientNetV2`` mirrors timm's
module names, so the timm ``state_dict`` loads with ``strict=True`` and is
bit-identical on inference (proven by the parity test). This script just
metadata-wraps it.

Class names are the canonical ImageNet-1k labels, index-aligned to the standard
wnid-sorted class ordering (torchvision sorts the wnid class folders into exactly
timm's class-index order), so ``model.val()`` reproduces the upstream benchmark
and predictions expose readable labels.

Usage::

    python weights/convert_efficientnetv2_weights.py            # all sizes
    python weights/convert_efficientnetv2_weights.py --size b0
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

# ImageNet-1k checkpoints only (Apache-2.0). Do NOT add .in21k_ft_in1k / .ns_jft.
TAGS = {
    "b0": "tf_efficientnetv2_b0.in1k",
    "b1": "tf_efficientnetv2_b1.in1k",
    "b2": "tf_efficientnetv2_b2.in1k",
    "b3": "tf_efficientnetv2_b3.in1k",
}
# Per-variant *test* (eval) resolution from timm's pretrained cfg.
IMGSZ = {"b0": 224, "b1": 240, "b2": 260, "b3": 300}


def convert(size: str) -> Path:
    import timm

    add_repo_root_to_path()
    from libreyolo.models.efficientnetv2.nn import EfficientNetV2

    tag = TAGS[size]
    timm_model = timm.create_model(tag, pretrained=True)
    state_dict = timm_model.state_dict()

    # Sanity: the timm state dict must load into the native model with no
    # missing/unexpected keys (the precondition for bit-exact parity).
    ours = EfficientNetV2(size=size, num_classes=1000)
    result = ours.load_state_dict(state_dict, strict=True)
    assert not result.missing_keys and not result.unexpected_keys, (
        result.missing_keys[:5],
        result.unexpected_keys[:5],
    )

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="efficientnetv2",
        size=size,
        nc=1000,
        names=imagenet1k_names(),
        task="classify",
        imgsz=IMGSZ[size],
    )

    out = Path("weights") / f"LibreEfficientNetV2{size}-cls.pt"
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(out)  # atomic, overwrites an existing file (cross-platform)
    print(f"Wrote {out}  (timm {tag}, nc=1000, task=classify, imgsz={IMGSZ[size]})")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["b0", "b1", "b2", "b3"], default=None,
                        help="Variant to convert (default: all).")
    args = parser.parse_args()
    for s in [args.size] if args.size else ["b0", "b1", "b2", "b3"]:
        convert(s)
