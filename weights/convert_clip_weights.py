"""Convert OpenCLIP LAION-2B weights into LibreYOLO (LibreCLIP) checkpoints.

LibreCLIP re-implements the CLIP towers natively (``libreyolo/models/clip/nn.py``)
with the *same* module structure as open_clip, so the upstream ``state_dict``
loads with 0 missing / 0 unexpected keys and forward parity is exact. This
script downloads the MIT-redistributable OpenCLIP LAION-2B weights via
``open_clip`` (the convert-only ``libreyolo[clip-convert]`` extra) and wraps them
in the LibreYOLO v1.0 checkpoint schema with ``model_family="clip"``,
``task="classify"`` and the ImageNet-1k default label set.

Data provenance: the LAION-2B training set has a documented CSAM-content history
(Stanford, 2023) → use Re-LAION-derived weights and see the family NOTICE.

Usage::

    python weights/convert_clip_weights.py                 # b32 + b16 -> weights/
    python weights/convert_clip_weights.py --sizes b32 --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _conversion_utils import add_repo_root_to_path

add_repo_root_to_path()

from libreyolo.models.clip.labels import imagenet1k_classnames  # noqa: E402
from libreyolo.models.clip.nn import build_clip_model  # noqa: E402
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint  # noqa: E402

# size code -> (open_clip arch, MIT-redistributable LAION-2B pretrained tag)
SIZE_TO_OPENCLIP = {
    "b32": ("ViT-B-32", "laion2b_s34b_b79k"),
    "b16": ("ViT-B-16", "laion2b_s34b_b88k"),
    "l14": ("ViT-L-14", "laion2b_s32b_b82k"),
}


def convert_one(size: str, out_dir: Path, verify: bool = False) -> Path:
    import open_clip

    arch, pretrained = SIZE_TO_OPENCLIP[size]
    print(f"[{size}] loading open_clip {arch} ({pretrained}) ...")
    oc_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    oc_model.eval()

    native = build_clip_model(size)
    result = native.load_state_dict(oc_model.state_dict(), strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"[{size}] key mismatch loading open_clip weights:\n"
            f"  missing: {missing}\n  unexpected: {unexpected}"
        )

    names = {i: n for i, n in enumerate(imagenet1k_classnames())}
    checkpoint = wrap_libreyolo_checkpoint(
        native.state_dict(),
        model_family="clip",
        size=size,
        task="classify",
        nc=len(names),
        names=names,
        source=f"openclip:{arch}:{pretrained}",
        license="MIT (weights); LAION-2B data — see NOTICE",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"LibreCLIP{size}-cls.pt"
    torch.save(checkpoint, out_path)
    print(f"[{size}] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    if verify:
        from libreyolo import LibreCLIP

        model = LibreCLIP(str(out_path), size=size, device="cpu")
        model.set_classes(["a cat", "a dog", "a truck"])
        dummy = torch.zeros(1, 3, model.input_size, model.input_size)
        with torch.no_grad():
            logits = model._forward(dummy)
        assert logits.shape == (1, 3), logits.shape
        print(f"[{size}] verify OK — forward logits {tuple(logits.shape)}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", default=["b32", "b16"], choices=list(SIZE_TO_OPENCLIP)
    )
    parser.add_argument("--out", default="weights", help="output directory")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    for size in args.sizes:
        convert_one(size, out_dir, verify=args.verify)


if __name__ == "__main__":
    main()
