"""Convert Apache-2.0 google/siglip2-* weights into LibreYOLO (LibreSigLIP2) checkpoints.

LibreSigLIP2 re-implements the SigLIP towers natively
(``libreyolo/models/siglip2/nn.py``) with the *same* module structure as the
reference ``SiglipModel``, so the upstream ``state_dict`` loads with 0 missing /
0 unexpected keys and forward parity is exact (see ``weights/parity_siglip2.py``).
This script downloads the Apache-2.0 fixed-resolution SigLIP 2 checkpoints via
``transformers`` (the convert-only ``libreyolo[siglip2-convert]`` extra) and wraps
them in the LibreYOLO v1.0 checkpoint schema with ``model_family="siglip2"``,
``task="classify"`` and the ImageNet-1k default label set.

The shipped sizes reuse the SigLIP v1 architecture (``model_type: "siglip"``):
``b16`` = google/siglip2-base-patch16-256, ``so400m`` = google/siglip2-so400m-patch14-384.

Usage::

    python weights/convert_siglip2_weights.py                 # b16 + so400m -> weights/
    python weights/convert_siglip2_weights.py --sizes b16 --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _conversion_utils import add_repo_root_to_path

add_repo_root_to_path()

from libreyolo.models.siglip2.labels import imagenet1k_classnames  # noqa: E402
from libreyolo.models.siglip2.nn import SIGLIP2_CONFIGS, build_siglip2_model  # noqa: E402
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint  # noqa: E402

# size code -> Apache-2.0 upstream HF repo (fixed-resolution SigLIP 2)
SIZE_TO_HF = {
    "b16": "google/siglip2-base-patch16-256",
    "so400m": "google/siglip2-so400m-patch14-384",
}


def convert_one(size: str, out_dir: Path, verify: bool = False) -> Path:
    from transformers import AutoModel

    repo = SIZE_TO_HF[size]
    print(f"[{size}] loading {repo} ...")
    ref = AutoModel.from_pretrained(repo)
    ref.eval()

    native = build_siglip2_model(size)
    result = native.load_state_dict(ref.state_dict(), strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"[{size}] key mismatch loading SigLIP 2 weights:\n"
            f"  missing: {missing}\n  unexpected: {unexpected}"
        )

    names = {i: n for i, n in enumerate(imagenet1k_classnames())}
    checkpoint = wrap_libreyolo_checkpoint(
        native.state_dict(),
        model_family="siglip2",
        size=size,
        task="classify",
        nc=len(names),
        names=names,
        imgsz=SIGLIP2_CONFIGS[size].image_size,
        source=f"huggingface:{repo}",
        license="Apache-2.0",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"LibreSigLIP2{size}-cls.pt"
    torch.save(checkpoint, out_path)
    print(f"[{size}] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    if verify:
        from libreyolo import LibreSigLIP2

        model = LibreSigLIP2(str(out_path), size=size, device="cpu")
        model.set_classes(["a cat", "a dog", "a truck"])
        dummy = torch.zeros(1, 3, model.input_size, model.input_size)
        with torch.no_grad():
            logits = model._forward(dummy)
        assert logits.shape == (1, 3), logits.shape
        print(f"[{size}] verify OK - forward logits {tuple(logits.shape)}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", default=["b16", "so400m"], choices=list(SIZE_TO_HF)
    )
    parser.add_argument("--out", default="weights", help="output directory")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    for size in args.sizes:
        convert_one(size, out_dir, verify=args.verify)


if __name__ == "__main__":
    main()
