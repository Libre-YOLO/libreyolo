"""Convert FeyNobg background-removal weights into LibreYOLO format.

LibreFeyNobg mirrors the upstream FeyNobg tensor names (which are BiRefNet's
key schema with a 24-block stage 3), so a released checkpoint converts by
extracting its state dict and wrapping it in the LibreYOLO v1.0 checkpoint
schema (metadata-wrap; learned parameters unchanged). This script does not
download or redistribute upstream weights.

Usage::

    python weights/convert_feynobg_weights.py model.safetensors weights/LibreFeyNobgl-matte.pt --verify

FeyNobg code and weights are Apache-2.0 (https://huggingface.co/feyninc/FeyNobg,
https://github.com/feyninc/nobg), Copyright (c) 2026 Feyn Inc. See
weights/LICENSE_NOTICE.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _conversion_utils import add_repo_root_to_path, load_checkpoint, save_checkpoint

_FEYNOBG_MARKER = "bb.layers.2.blocks.23.norm1.weight"
_IMGSZ = 1024


def _load_state_dict(input_path: str) -> dict:
    if str(input_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return dict(load_file(input_path))
    raw = load_checkpoint(input_path)
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "params", "net"):
            value = raw.get(key)
            if isinstance(value, dict):
                return dict(value)
        return dict(raw)
    if hasattr(raw, "state_dict"):
        return dict(raw.state_dict())
    raise TypeError(f"Unsupported checkpoint object: {type(raw)!r}")


def convert_weights(input_path: str, output_path: str, *, imgsz: int = _IMGSZ) -> dict:
    print(f"Loading FeyNobg weights from {input_path}")
    state_dict = _load_state_dict(input_path)
    print(f"Found {len(state_dict)} parameter entries")

    if _FEYNOBG_MARKER not in state_dict:
        raise ValueError(
            "This does not look like a FeyNobg checkpoint (no 24-block stage-3 "
            "marker). BiRefNet checkpoints convert with "
            "weights/convert_birefnet_weights.py."
        )

    add_repo_root_to_path()
    from libreyolo.models.feynobg import LibreFeyNobg
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    model = LibreFeyNobg(model_path=None, size="l", device="cpu")
    result = model.model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "FeyNobg state dict did not load strictly: "
            f"missing={result.missing_keys[:8]}, unexpected={result.unexpected_keys[:8]}"
        )

    checkpoint = wrap_libreyolo_checkpoint(
        model.model.state_dict(),
        model_family="feynobg",
        size="l",
        task="matte",
        nc=1,
        names={0: "matte"},
        supported_tasks=("matte",),
        default_task="matte",
        imgsz=imgsz,
    )
    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(checkpoint, tmp)
    tmp.rename(out)  # atomic
    print(f"Saved LibreYOLO-format checkpoint to {out}")
    return checkpoint


def verify_conversion(converted_path: str) -> bool:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    validate_checkpoint_metadata(converted_path)
    print(f"\nLoading converted weights via LibreYOLO({converted_path})...")
    model = LibreYOLO(converted_path, device="cpu")
    print(f"  family={model.FAMILY} size={model.size} task={model.task} nc={model.nb_classes} names={model.names}")
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, _IMGSZ, _IMGSZ))
    print(f"  forward pass OK - output shape: {tuple(out.shape)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FeyNobg weights to LibreYOLO format")
    parser.add_argument("input", help="Upstream FeyNobg checkpoint (.safetensors/.pth/.pt)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument("--imgsz", type=int, default=_IMGSZ, help="Native input size recorded in metadata")
    parser.add_argument("--verify", action="store_true", help="Verify round-trip + metadata after conversion")
    args = parser.parse_args()

    convert_weights(args.input, args.output, imgsz=args.imgsz)
    if args.verify:
        verify_conversion(args.output)
