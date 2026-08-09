"""Convert BiRefNet background-removal weights into LibreYOLO format.

LibreBiRefNet mirrors the upstream BiRefNet tensor names, so a supported
checkpoint converts by extracting its state dict and wrapping it in the
LibreYOLO v1.0 checkpoint schema (metadata-wrap; learned parameters unchanged).
This script does not download or redistribute upstream weights.

Usage::

    # general (Swin-L tier, quality default)
    python weights/convert_birefnet_weights.py BiRefNet.safetensors weights/LibreBiRefNetl-matte.pt --size l --verify
    # lite (Swin-T tier, fast)
    python weights/convert_birefnet_weights.py BiRefNet_lite.safetensors weights/LibreBiRefNett-matte.pt --size t --verify
FeyNobg checkpoints belong to the sibling ``feynobg`` family; use
weights/convert_feynobg_weights.py for those.

BiRefNet source code is MIT (https://github.com/ZhengPeng7/BiRefNet). The
released ``BiRefNet`` (general) weights are tagged MIT on Hugging Face; verify
each variant's weights license before rehosting (see weights/LICENSE_NOTICE.txt).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from _conversion_utils import add_repo_root_to_path, load_checkpoint, save_checkpoint

_EMBED_DIM_TO_SIZE = {96: "t", 192: "l"}
_FEYNOBG_MARKER = "bb.layers.2.blocks.23.norm1.weight"
_IMGSZ = 1024


def _load_state_dict(input_path: str) -> Dict[str, torch.Tensor]:
    """Load a state dict from a .safetensors or torch checkpoint."""
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


def detect_size(state_dict: Dict[str, torch.Tensor]) -> str | None:
    proj = state_dict.get("bb.patch_embed.proj.weight")
    if proj is None or getattr(proj, "ndim", 0) != 4:
        return None
    if _FEYNOBG_MARKER in state_dict:
        raise ValueError(
            "This looks like a FeyNobg checkpoint (24 stage-3 blocks); "
            "use weights/convert_feynobg_weights.py instead."
        )
    return _EMBED_DIM_TO_SIZE.get(int(proj.shape[0]))


def convert_weights(input_path: str, output_path: str, *, size: str | None = None, imgsz: int = _IMGSZ) -> dict:
    print(f"Loading BiRefNet weights from {input_path}")
    state_dict = _load_state_dict(input_path)
    print(f"Found {len(state_dict)} parameter entries")

    if size is None:
        size = detect_size(state_dict)
        if size is None:
            raise ValueError("Could not auto-detect BiRefNet size; pass --size {t,l}.")
        print(f"Auto-detected size: {size}")

    add_repo_root_to_path()
    from libreyolo.models.birefnet import LibreBiRefNet
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    model = LibreBiRefNet(model_path=None, size=size, device="cpu")
    result = model.model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "BiRefNet state dict did not load strictly: "
            f"missing={result.missing_keys[:8]}, unexpected={result.unexpected_keys[:8]}"
        )

    checkpoint = wrap_libreyolo_checkpoint(
        model.model.state_dict(),
        model_family="birefnet",
        size=size,
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
    parser = argparse.ArgumentParser(description="Convert BiRefNet weights to LibreYOLO format")
    parser.add_argument("input", help="Upstream BiRefNet checkpoint (.safetensors/.pth/.pt)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument("--size", choices=["t", "l"], default=None, help="Size code (default: auto-detect)")
    parser.add_argument("--imgsz", type=int, default=_IMGSZ, help="Native input size recorded in metadata")
    parser.add_argument("--verify", action="store_true", help="Verify round-trip + metadata after conversion")
    args = parser.parse_args()

    convert_weights(args.input, args.output, size=args.size, imgsz=args.imgsz)
    if args.verify:
        verify_conversion(args.output)
