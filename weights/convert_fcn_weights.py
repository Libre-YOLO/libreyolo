"""Wrap official torchvision FCN weights in LibreYOLO metadata.

The torchvision v0.26.0 module tree and LibreYOLO's native port use identical
parameter names, so conversion does not alter tensor data. The converted
checkpoints retain both the primary and auxiliary 21-class VOC-style heads.

Code upstream: https://github.com/pytorch/vision/tree/v0.26.0
Commit: 336d36e8db990a905498c73933e35231876e28bc
Code license: BSD-3-Clause

The publisher does not attach a separate license file to each checkpoint.
LibreYOLO's redistribution basis and the upstream pretrained-model caveat are
recorded in ``docs/provenance/fcn.md``; this converter makes no broader license
claim.

Usage:
    python weights/convert_fcn_weights.py upstream.pth \
        weights/LibreFCNr50.pt --size r50 --verify
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
    size: str | None = None,
) -> dict:
    """Validate and metadata-wrap one official FCN state dict."""
    print(f"Loading upstream FCN weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw, prefer_ema=True)
    if not isinstance(state_dict, dict):
        raise TypeError("FCN checkpoint did not contain a state dict")

    add_repo_root_to_path()
    from libreyolo.models.fcn.model import VOC_NAMES, LibreFCN
    from libreyolo.models.fcn.nn import LibreFCNModel

    if not LibreFCN.can_load(state_dict):
        raise ValueError(
            "Not an FCN checkpoint: the classifier, auxiliary classifier, and "
            "embedded ResNet fingerprint are required."
        )

    detected_size = LibreFCN.detect_size(state_dict)
    if detected_size is None:
        raise ValueError("Could not determine whether the FCN backbone is r50 or r101.")
    if size is None:
        size = detected_size
        print(f"Auto-detected size: {size}")
    elif size != detected_size:
        raise ValueError(
            f"--size {size} does not match the checkpoint architecture "
            f"(detected '{detected_size}')."
        )

    head_width = LibreFCN.detect_nb_classes(state_dict)
    aux_width = int(state_dict["aux_classifier.4.weight"].shape[0])
    if head_width != len(VOC_NAMES) or aux_width != len(VOC_NAMES):
        raise ValueError(
            "Official FCN conversion expects matching 21-class primary and "
            f"auxiliary heads; got {head_width} and {aux_width}."
        )

    probe = LibreFCNModel(size=size, num_classes=head_width)
    probe.load_state_dict(state_dict, strict=True)

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="fcn",
        size=size,
        nc=head_width,
        names=dict(enumerate(VOC_NAMES)),
        task="semantic",
        imgsz=LibreFCN.INPUT_SIZES[size],
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
    print(
        f"Loaded back: family={model.family} size={model.size} "
        f"task={model.task} nc={model.nb_classes}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Official torchvision .pth checkpoint")
    parser.add_argument("output", help="Destination LibreFCNr<size>.pt")
    parser.add_argument("--size", choices=["r50", "r101"])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output, args.size)
    if args.verify:
        verify(args.output)


if __name__ == "__main__":
    main()
