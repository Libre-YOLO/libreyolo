"""Wrap official torchvision Mask R-CNN weights in LibreYOLO metadata.

The torchvision v0.26.0 module tree and LibreYOLO's native port use identical
parameter names, so conversion does not alter tensor data. The official COCO
head retains 91 outputs (background plus the sparse COCO category-id space),
while checkpoint metadata advertises LibreYOLO's contiguous COCO-80 interface.

Code upstream: https://github.com/pytorch/vision/tree/v0.26.0
Commit: 336d36e8db990a905498c73933e35231876e28bc
Code license: BSD-3-Clause

The publisher does not attach a separate license file to the checkpoint.
LibreYOLO's redistribution basis and the pretrained-model caveat are recorded
in ``docs/provenance/mask_rcnn.md``; this converter makes no broader claim.

Usage:
    python weights/convert_mask_rcnn_weights.py upstream.pth \
        weights/LibreMaskRCNNr50.pt --size r50 --verify
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
    size: str,
    nc: int = 80,
) -> dict:
    """Validate and metadata-wrap one Mask R-CNN state dict."""
    print(f"Loading upstream weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw, prefer_ema=True)
    if not isinstance(state_dict, dict):
        raise TypeError("Mask R-CNN checkpoint did not contain a state dict")

    add_repo_root_to_path()
    from libreyolo.models.mask_rcnn.model import LibreMaskRCNN
    from libreyolo.models.mask_rcnn.nn import LibreMaskRCNNModel

    if not LibreMaskRCNN.can_load(state_dict):
        raise ValueError(
            "Not a Mask R-CNN checkpoint: the RPN, box, or mask head keys "
            "are missing."
        )

    head_width = int(
        state_dict["roi_heads.box_predictor.cls_score.weight"].shape[0]
    )
    mask_width = int(
        state_dict["roi_heads.mask_predictor.mask_fcn_logits.weight"].shape[0]
    )
    expected_width = 91 if nc == 80 else nc + 1
    if head_width != expected_width or mask_width != expected_width:
        raise ValueError(
            "Checkpoint box/mask heads have "
            f"{head_width}/{mask_width} outputs but --nc {nc} expects "
            f"{expected_width} (including background)."
        )

    detected_size = LibreMaskRCNN.detect_size(state_dict)
    if detected_size != size:
        raise ValueError(
            f"--size {size} does not match the checkpoint architecture "
            f"(detected {detected_size!r})."
        )

    probe = LibreMaskRCNNModel(size=size, num_classes=head_width)
    result = probe.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {sorted(result.missing_keys)}")
    print(f"Unexpected keys: {sorted(result.unexpected_keys)}")
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"State dict does not match Mask R-CNN size '{size}' exactly."
        )

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="mask_rcnn",
        size=size,
        nc=nc,
        task="segment",
        imgsz=LibreMaskRCNN.INPUT_SIZES[size],
        supported_tasks=("detect", "segment"),
        default_task="segment",
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
    temporary.replace(output)
    print(f"Saved LibreYOLO-format checkpoint to {output}")
    return checkpoint


def verify(output_path: str) -> None:
    """Validate metadata and reload the result through the unified factory."""
    add_repo_root_to_path()
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    loaded = load_checkpoint(output_path)
    validate_checkpoint_metadata(loaded, strict=True)
    model = LibreYOLO(output_path, device="cpu")
    print(
        f"Loaded back: family={model.family} size={model.size} "
        f"task={model.task} nc={model.nb_classes}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Official torchvision .pth checkpoint")
    parser.add_argument("output", help="Destination LibreMaskRCNNr50.pt")
    parser.add_argument("--size", required=True, choices=["r50"])
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output, args.size, args.nc)
    if args.verify:
        verify(args.output)


if __name__ == "__main__":
    main()
