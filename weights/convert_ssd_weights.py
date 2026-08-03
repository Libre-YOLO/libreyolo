"""Wrap official torchvision SSD300-VGG16 weights in LibreYOLO metadata.

The torchvision v0.26.0 module tree and LibreYOLO's native port use identical
parameter names, so conversion does not alter tensor data. The official COCO
head retains 91 outputs (background plus the sparse COCO category-id space),
while checkpoint metadata advertises LibreYOLO's contiguous COCO-80 interface.

Code upstream: https://github.com/pytorch/vision/tree/v0.26.0
Commit: 336d36e8db990a905498c73933e35231876e28bc
Code license: BSD-3-Clause

The publisher does not attach a separate license file to each checkpoint. Its
VGG16 feature initialization also traces to Oxford's pretrained VGG weights.
LibreYOLO's redistribution basis and attribution caveats are recorded in
``docs/provenance/ssd.md``; this converter makes no broader license claim.

Usage:
    python weights/convert_ssd_weights.py upstream.pth \
        weights/LibreSSD300.pt --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


def convert_weights(
    input_path: str,
    output_path: str,
    size: str = "300",
    nc: int = 80,
) -> dict:
    """Validate and metadata-wrap one SSD300-VGG16 state dict."""
    print(f"Loading upstream weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw, prefer_ema=True)
    if not isinstance(state_dict, dict):
        raise TypeError("SSD checkpoint did not contain a state dict")

    add_repo_root_to_path()
    from libreyolo.models.ssd.model import LibreSSD
    from libreyolo.models.ssd.nn import LibreSSDModel

    if not LibreSSD.can_load(state_dict):
        raise ValueError(
            "Not an SSD300-VGG16 checkpoint: required VGG extra or MultiBox "
            "head keys are missing or have incompatible shapes."
        )

    head_width = (
        int(state_dict["head.classification_head.module_list.0.weight"].shape[0]) // 4
    )
    expected_width = 91 if nc == 80 else nc + 1
    if head_width != expected_width:
        raise ValueError(
            f"Checkpoint classification head has {head_width} outputs but --nc "
            f"{nc} expects {expected_width} (including background)."
        )

    detected_size = LibreSSD.detect_size(state_dict)
    if detected_size is not None and detected_size != size:
        raise ValueError(
            f"--size {size} does not match the checkpoint architecture "
            f"(detected '{detected_size}')."
        )

    probe = LibreSSDModel(num_classes=head_width)
    result = probe.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {sorted(result.missing_keys)}")
    print(f"Unexpected keys: {sorted(result.unexpected_keys)}")
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("State dict does not match SSD300-VGG16 exactly.")

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="ssd",
        size=size,
        nc=nc,
        task="detect",
        imgsz=LibreSSD.INPUT_SIZES[size],
        supported_tasks=("detect",),
        default_task="detect",
    )
    output = Path(output_path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
    temporary.replace(output)
    print(f"Saved LibreYOLO-format checkpoint to {output}")
    return checkpoint


def verify(output_path: str) -> None:
    """Reload the result through the unified factory."""
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    model = LibreYOLO(output_path, device="cpu")
    print(f"Loaded back: family={model.family} size={model.size} nc={model.nb_classes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Official torchvision .pth checkpoint")
    parser.add_argument("output", help="Destination LibreSSD300.pt")
    parser.add_argument("--size", default="300", choices=["300"])
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output, args.size, args.nc)
    if args.verify:
        verify(args.output)


if __name__ == "__main__":
    main()
