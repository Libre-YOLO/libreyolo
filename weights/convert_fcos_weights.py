"""Wrap official torchvision FCOS weights in LibreYOLO metadata.

The torchvision v0.26.0 module tree and LibreYOLO's native port use identical
parameter names, so conversion does not alter tensor data. The official COCO
head retains 91 outputs in the sparse COCO category-id space, while checkpoint
metadata advertises LibreYOLO's contiguous COCO-80 interface.

Code upstream: https://github.com/pytorch/vision/tree/v0.26.0
Commit: 336d36e8db990a905498c73933e35231876e28bc
Code license: BSD-3-Clause

The publisher does not attach a separate license file to each checkpoint.
LibreYOLO's redistribution basis and the upstream pretrained-model caveat are
recorded in ``docs/provenance/fcos.md``; this converter makes no broader
license claim.

Usage:
    python weights/convert_fcos_weights.py upstream.pth \
        weights/LibreFCOSr50.pt --verify
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
    nc: int = 80,
) -> dict:
    """Validate and metadata-wrap one FCOS ResNet-50/FPN state dict."""
    print(f"Loading upstream weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw, prefer_ema=True)
    if not isinstance(state_dict, dict):
        raise TypeError("FCOS checkpoint did not contain a state dict")

    add_repo_root_to_path()
    from libreyolo.models.fcos.model import LibreFCOS
    from libreyolo.models.fcos.nn import LibreFCOSModel

    if not LibreFCOS.can_load(state_dict):
        raise ValueError(
            "Not an FCOS ResNet-50/FPN checkpoint: the centerness, "
            "classification, or P6/P7 keys are missing."
        )

    head_width = int(state_dict["head.classification_head.cls_logits.weight"].shape[0])
    expected_width = 91 if nc == 80 else nc
    if head_width != expected_width:
        raise ValueError(
            f"Checkpoint classification head has {head_width} outputs but "
            f"--nc {nc} expects {expected_width}."
        )

    detected_size = LibreFCOS.detect_size(state_dict)
    if detected_size != "r50":
        raise ValueError(
            "Checkpoint architecture is not the supported FCOS ResNet-50/FPN variant."
        )

    probe = LibreFCOSModel(num_classes=head_width)
    probe.load_state_dict(state_dict, strict=True)
    print("Strict architecture probe passed (zero missing/unexpected keys)")

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="fcos",
        size="r50",
        nc=nc,
        task="detect",
        imgsz=LibreFCOS.INPUT_SIZES["r50"],
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
    parser.add_argument("output", help="Destination LibreFCOSr50.pt")
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output, args.nc)
    if args.verify:
        verify(args.output)


if __name__ == "__main__":
    main()
