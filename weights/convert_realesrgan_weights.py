"""Convert Real-ESRGAN super-resolution weights into LibreYOLO format.

LibreRealESRGAN mirrors the upstream RRDBNet / SRVGGNetCompact tensor names, so
supported checkpoints convert by extracting the model state dict and wrapping it
in the LibreYOLO v1.0 checkpoint schema. This script does not download or
redistribute upstream weights.

Upstream release filename -> LibreYOLO size:
    RealESRGAN_x4plus.pth        -> x4   (RRDBNet, scale 4)
    RealESRGAN_x2plus.pth        -> x2   (RRDBNet, scale 2)
    realesr-general-x4v3.pth     -> x4t  (SRVGGNetCompact, scale 4)

Usage::

    python weights/convert_realesrgan_weights.py RealESRGAN_x4plus.pth \
        weights/LibreRealESRGANx4-restore.pt --size x4 --verify
    python weights/convert_realesrgan_weights.py realesr-general-x4v3.pth \
        weights/LibreRealESRGANx4t-restore.pt --size x4t --verify

Real-ESRGAN code and released weights are BSD-3-Clause; the RRDBNet / SRVGG
architecture lineage is BasicSR (Apache-2.0). Convert only weights you have the
right to use and redistribute.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    load_checkpoint,
    save_checkpoint,
    strip_state_dict_prefix,
)

_STATE_KEYS = ("params_ema", "params", "state_dict", "model", "net", "network")
_PREFIXES = ("module.", "net_g.", "generator.", "model.")
# LR nominal patch size recorded in checkpoint metadata (native-res at runtime).
_IMGSZ = 64
_SCALE_BY_SIZE = {"x4": 4, "x2": 2, "x4t": 4}


def _materialize_state_dict(candidate: Any) -> dict[str, torch.Tensor]:
    state_dict = getattr(candidate, "state_dict", None)
    if callable(state_dict):
        candidate = state_dict()
    if not isinstance(candidate, dict):
        raise TypeError(f"Expected a state_dict-like object, got {type(candidate)!r}.")
    return dict(candidate)


def extract_realesrgan_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract and normalize a Real-ESRGAN state dict from common layouts."""
    if isinstance(checkpoint, dict):
        for key in _STATE_KEYS:
            value = checkpoint.get(key)
            if isinstance(value, dict) or hasattr(value, "state_dict"):
                state_dict = _materialize_state_dict(value)
                break
        else:
            state_dict = _materialize_state_dict(checkpoint)
    else:
        state_dict = _materialize_state_dict(checkpoint)

    for prefix in _PREFIXES:
        if any(key.startswith(prefix) for key in state_dict):
            state_dict = strip_state_dict_prefix(state_dict, prefix)
    return state_dict


def detect_size(state_dict: dict[str, torch.Tensor]) -> str | None:
    """Infer LibreRealESRGAN size from the state dict."""
    add_repo_root_to_path()
    from libreyolo.models.realesrgan.model import LibreRealESRGAN

    return LibreRealESRGAN.detect_size(state_dict)


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    size: str | None = None,
    imgsz: int = _IMGSZ,
    dataset: str | None = None,
) -> dict:
    """Convert one Real-ESRGAN checkpoint to a LibreYOLO metadata-wrapped file."""
    print(f"Loading Real-ESRGAN weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_realesrgan_state_dict(raw)
    print(f"Found {len(state_dict)} parameter entries")

    if size is None:
        size = detect_size(state_dict)
        if size is None:
            raise ValueError("Could not auto-detect size; pass --size {x4,x2,x4t}.")
        print(f"Auto-detected size: {size}")

    add_repo_root_to_path()
    from libreyolo.models.realesrgan import LibreRealESRGAN
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    model = LibreRealESRGAN(model_path=None, size=size, device="cpu")
    result = model.model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Real-ESRGAN state dict did not load strictly: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )

    scale = _SCALE_BY_SIZE[size]
    checkpoint = wrap_libreyolo_checkpoint(
        model.model.state_dict(),
        model_family="realesrgan",
        size=size,
        task="restore",
        nc=1,
        names={0: "image"},
        imgsz=imgsz,
        scale=scale,
        degradation="super-resolution",
        dataset=dataset,
    )
    save_checkpoint(checkpoint, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path} (scale={scale})")
    return checkpoint


def verify_conversion(converted_path: str) -> bool:
    """Load through LibreYOLO and run one small smoke forward."""
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    print(f"\nLoading converted weights via LibreYOLO({converted_path})...")
    model = LibreYOLO(converted_path, device="cpu")
    print(
        f"  family={model.FAMILY} size={model.size} task={model.task} "
        f"scale={model.restore_scale} nc={model.nb_classes} names={model.names}"
    )
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, 32, 32))
    print(f"  forward 32x32 -> {tuple(out.shape)} (expect {model.restore_scale}x)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Real-ESRGAN super-resolution weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream Real-ESRGAN checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument(
        "--size",
        choices=["x4", "x2", "x4t"],
        default=None,
        help="Size code (default: auto-detect from state dict)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=_IMGSZ,
        help="Nominal LR patch size recorded in checkpoint metadata",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset/provenance label recorded in checkpoint metadata",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify round-trip after conversion"
    )
    args = parser.parse_args()

    convert_weights(
        args.input,
        args.output,
        size=args.size,
        imgsz=args.imgsz,
        dataset=args.dataset,
    )
    if args.verify:
        verify_conversion(args.output)
