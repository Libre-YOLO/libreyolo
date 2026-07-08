"""Convert NAFNet restoration weights into LibreYOLO format.

The LibreNAFNet module mirrors the upstream NAFNet tensor names, so supported
checkpoints convert by extracting the model state dict and wrapping it in the
LibreYOLO v1.0 checkpoint schema. This script does not download or redistribute
upstream weights.

Usage::

    python weights/convert_nafnet_weights.py input.pth weights/LibreNAFNets-restore.pt
    python weights/convert_nafnet_weights.py input.pth weights/LibreNAFNetl-restore.pt --size l --verify

NAFNet source code is MIT licensed. Some published GoPro-trained checkpoint
files do not carry an explicit standalone weights license; convert only weights
that you have the right to use and redistribute.
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
_IMGSZ = 256


def _materialize_state_dict(candidate: Any) -> dict[str, torch.Tensor]:
    state_dict = getattr(candidate, "state_dict", None)
    if callable(state_dict):
        candidate = state_dict()
    if not isinstance(candidate, dict):
        raise TypeError(f"Expected a state_dict-like object, got {type(candidate)!r}.")
    return dict(candidate)


def extract_nafnet_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract and normalize a NAFNet state dict from common checkpoint layouts."""
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
    """Infer LibreNAFNet size from the first convolution width."""
    intro = state_dict.get("intro.weight")
    if intro is None or getattr(intro, "ndim", 0) != 4:
        return None
    return {32: "s", 64: "l"}.get(int(intro.shape[0]))


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    size: str | None = None,
    imgsz: int = _IMGSZ,
    degradation: str | None = None,
    dataset: str | None = None,
) -> dict:
    """Convert one NAFNet checkpoint to a LibreYOLO metadata-wrapped checkpoint."""
    print(f"Loading NAFNet weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_nafnet_state_dict(raw)
    print(f"Found {len(state_dict)} parameter entries")

    if size is None:
        size = detect_size(state_dict)
        if size is None:
            raise ValueError("Could not auto-detect NAFNet size; pass --size {s,l}.")
        print(f"Auto-detected size: {size}")

    add_repo_root_to_path()
    from libreyolo.models.nafnet.model import infer_nafnet_config
    from libreyolo.models.nafnet.nn import NAFNetLocal
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    # Build the architecture that matches the checkpoint's block layout (GoPro
    # and SIDD NAFNet checkpoints share tensor names but differ in block counts).
    cfg = infer_nafnet_config(state_dict)
    if cfg is None:
        raise ValueError("Could not infer NAFNet block config from the state dict.")
    model = NAFNetLocal(
        img_channel=3,
        width=int(cfg["width"]),
        middle_blk_num=int(cfg["middle_blk_num"]),
        enc_blk_nums=cfg["enc_blk_nums"],
        dec_blk_nums=cfg["dec_blk_nums"],
        train_size=(1, 3, imgsz, imgsz),
    )
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "NAFNet state dict did not load strictly: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )

    checkpoint = wrap_libreyolo_checkpoint(
        model.state_dict(),
        model_family="nafnet",
        size=size,
        task="restore",
        nc=1,
        names={0: "image"},
        imgsz=imgsz,
        degradation=degradation,
        dataset=dataset,
    )
    save_checkpoint(checkpoint, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}")
    return checkpoint


def verify_conversion(converted_path: str) -> bool:
    """Load through LibreYOLO and run one fixed-size smoke forward."""
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    print(f"\nLoading converted weights via LibreYOLO({converted_path})...")
    model = LibreYOLO(converted_path, device="cpu")
    print(
        f"  family={model.FAMILY} size={model.size} task={model.task} "
        f"nc={model.nb_classes} names={model.names}"
    )
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, _IMGSZ, _IMGSZ))
    print(f"  forward pass OK - output shape: {tuple(out.shape)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert NAFNet restoration weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream NAFNet checkpoint (.pth/.pt)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument(
        "--size",
        choices=["s", "l"],
        default=None,
        help="Size code (default: auto-detect from intro.weight width)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=_IMGSZ,
        help="Nominal training patch size recorded in checkpoint metadata",
    )
    parser.add_argument(
        "--degradation",
        default=None,
        help="Optional degradation label, e.g. deblur or denoise",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset label recorded in checkpoint metadata",
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
        degradation=args.degradation,
        dataset=args.dataset,
    )
    if args.verify:
        verify_conversion(args.output)
