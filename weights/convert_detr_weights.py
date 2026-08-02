"""Wrap official DETR checkpoints in LibreYOLO metadata without remapping.

The four official Apache-2.0 checkpoints are published by
``facebookresearch/detr`` as ``{"model": state_dict}``. LibreYOLO's native
module names intentionally match that state dict, so conversion does not
rename or alter learned tensors.

DC5 dilation is runtime configuration and leaves every checkpoint tensor shape
unchanged. Consequently ``--size`` is mandatory. The converter cross-checks
the official filename when it is available and strict-loads the selected
architecture; for a renamed source file the caller's DC5 choice is necessarily
authoritative.

Usage:
    python weights/convert_detr_weights.py detr-r50-e632da11.pth weights/LibreDETRr50.pt --size r50
    python weights/convert_detr_weights.py detr-r50-dc5-f0fb7ef5.pth weights/LibreDETRr50dc5.pt --size r50dc5
    python weights/convert_detr_weights.py detr-r101-2c7b67e5.pth weights/LibreDETRr101.pt --size r101
    python weights/convert_detr_weights.py detr-r101-dc5-a2e86def.pth weights/LibreDETRr101dc5.pt --size r101dc5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

SIZES = ("r50", "r50dc5", "r101", "r101dc5")
COCO_PUBLIC_CLASSES = 80
COCO_ARCH_CLASSES = 91


def _load_official_checkpoint(input_path: str | Path) -> dict:
    """Load the tensor-only official checkpoint with PyTorch's safe loader."""
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
    state_dict = extract_state_dict(checkpoint, prefer_ema=False)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            "DETR checkpoint does not contain a non-empty model state dict"
        )
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("DETR state dict must contain only string keys and tensors")
    return state_dict


def convert_weights(
    input_path: str,
    output_path: str,
    size: str,
) -> dict:
    """Validate and atomically wrap one official DETR state dict."""
    if size not in SIZES:
        raise ValueError(f"Unsupported DETR size {size!r}; choose one of {SIZES}")

    print(f"Loading official DETR weights from {input_path}")
    state_dict = _load_official_checkpoint(input_path)
    print(f"Found {len(state_dict)} tensor entries")

    add_repo_root_to_path()
    from libreyolo.models.detr.model import LibreDETR
    from libreyolo.models.detr.nn import LibreDETRModel

    if not LibreDETR.can_load(state_dict):
        raise ValueError(
            "Checkpoint does not match the official vanilla DETR signature"
        )

    filename_size = LibreDETR.detect_size_from_filename(Path(input_path).name)
    if filename_size is not None and filename_size != size:
        raise ValueError(
            f"--size {size} conflicts with source filename, which identifies "
            f"DETR {filename_size}"
        )

    head = state_dict["class_embed.weight"]
    if tuple(head.shape) != (COCO_ARCH_CLASSES + 1, 256):
        raise ValueError(
            "Expected the official COCO class head shape (92, 256), got "
            f"{tuple(head.shape)}"
        )

    probe = LibreDETRModel(size=size, nc=COCO_ARCH_CLASSES)
    result = probe.load_state_dict(state_dict, strict=True)
    if (
        result.missing_keys or result.unexpected_keys
    ):  # pragma: no cover - strict raises first
        raise ValueError(
            f"State dict mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    print(f"Strict-load verified for DETR {size}")

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="detr",
        size=size,
        nc=COCO_PUBLIC_CLASSES,
        task="detect",
        imgsz=800,
    )

    output = Path(output_path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        save_checkpoint(checkpoint, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved LibreYOLO checkpoint to {output}")
    return checkpoint


def verify(output_path: str, size: str) -> None:
    """Reload the wrapped checkpoint and run a small native forward pass."""
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    model = LibreYOLO(output_path, device="cpu")
    if model.family != "detr" or model.size != size or model.nb_classes != 80:
        raise RuntimeError(
            "Round-trip metadata mismatch: "
            f"family={model.family}, size={model.size}, nc={model.nb_classes}"
        )
    model.model.eval()
    with torch.inference_mode():
        output = model.model(torch.zeros(1, 3, 64, 64))
    if tuple(output["pred_logits"].shape) != (1, 100, 92):
        raise RuntimeError("Unexpected DETR logits shape after conversion")
    if tuple(output["pred_boxes"].shape) != (1, 100, 4):
        raise RuntimeError("Unexpected DETR boxes shape after conversion")
    print("Round-trip load and forward pass verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Official detr-r*.pth checkpoint")
    parser.add_argument("output", help="Destination LibreDETR*.pt checkpoint")
    parser.add_argument("--size", required=True, choices=SIZES)
    parser.add_argument(
        "--verify", action="store_true", help="Reload and smoke-test the result"
    )
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size)
    if args.verify:
        verify(args.output, args.size)


if __name__ == "__main__":
    main()
