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
import hashlib
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
SOURCES = {
    "t": (
        "timm/deit_tiny_patch16_224.fb_in1k",
        "80e968688553f219e4a86f940ed945a23709c16f",
        "21d4764d94f6c3ffdb6da3581115a0a1ee2d505537d96883b540e54766407c9e",
    ),
    "s": (
        "timm/deit_small_patch16_224.fb_in1k",
        "91327a9c99f98fe6b524cd4d397b7226b80e1365",
        "1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b",
    ),
    "b": (
        "timm/deit_base_patch16_224.fb_in1k",
        "b78cc5532a69df6bcad9c3a8d76653fd20b31ac6",
        "cd2da27b74ed7f68b599f16c77af3e1e80f01c75f9ad96029d22ce747a247e8e",
    ),
}
IMGSZ = {"t": 224, "s": 224, "b": 224}
EXPECTED_CROP_PCT = 0.9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(size: str) -> Path:
    """Download one pinned timm model and wrap its unchanged tensors."""
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    add_repo_root_to_path()
    from libreyolo.models.deit.nn import DeiT

    tag = TAGS[size]
    repo_id, revision, expected_sha256 = SOURCES[size]
    source_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            revision=revision,
            token=False,
        )
    )
    actual_sha256 = _sha256(source_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {repo_id}@{revision}: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )

    source = timm.create_model(tag, pretrained=False).eval()
    cfg = source.pretrained_cfg
    if cfg.get("license") != "apache-2.0":
        raise RuntimeError(
            f"Upstream license changed for {tag}: {cfg.get('license')!r}."
        )
    crop_pct = float(cfg.get("crop_pct", 0.0))
    interpolation = str(cfg.get("interpolation", ""))
    if crop_pct != EXPECTED_CROP_PCT or interpolation != "bicubic":
        raise RuntimeError(
            f"Upstream preprocessing changed for {tag}: "
            f"crop_pct={crop_pct}, interpolation={interpolation!r}."
        )

    source_result = source.load_state_dict(
        load_file(source_path, device="cpu"), strict=True
    )
    if source_result.missing_keys or source_result.unexpected_keys:
        raise RuntimeError(f"Strict timm state-dict load failed for {tag}.")
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
        f"Wrote {output} ({repo_id}@{revision}, sha256={actual_sha256}, "
        f"nc=1000, task=classify, imgsz={IMGSZ[size]})"
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
