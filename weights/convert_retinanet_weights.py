"""Wrap official torchvision RetinaNet weights in LibreYOLO metadata.

The torchvision v0.26.0 module tree and LibreYOLO's native port use identical
parameter names, so conversion does not alter learned tensor data. Official
COCO heads retain the sparse 91-way category-id space while checkpoint
metadata advertises LibreYOLO's contiguous COCO-80 public interface.

Code upstream: https://github.com/pytorch/vision/tree/v0.26.0
Commit: 336d36e8db990a905498c73933e35231876e28bc
Code license: BSD-3-Clause

The publisher does not attach a separate license file to each checkpoint.
LibreYOLO's disclosed redistribution basis is documented in
``docs/provenance/retinanet.md``; this converter makes no broader claim.

Usage:
    python weights/convert_retinanet_weights.py upstream.pth \
        weights/LibreRetinaNetr50.pt --size r50 --verify
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
    """Validate and metadata-wrap one RetinaNet state dict."""
    print(f"Loading upstream weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw, prefer_ema=True)
    if not isinstance(state_dict, dict):
        raise TypeError("RetinaNet checkpoint did not contain a state dict")

    add_repo_root_to_path()
    from libreyolo.models.retinanet.model import LibreRetinaNet
    from libreyolo.models.retinanet.nn import LibreRetinaNetModel

    if not LibreRetinaNet.can_load(state_dict):
        raise ValueError(
            "Not a RetinaNet checkpoint: classification or regression head "
            "keys are missing, or an FCOS centerness head is present."
        )

    detected_size = LibreRetinaNet.detect_size(state_dict)
    if detected_size != size:
        raise ValueError(
            f"--size {size} does not match the checkpoint architecture "
            f"(detected {detected_size!r})."
        )

    classification_key = "head.classification_head.cls_logits.weight"
    head_out_channels = int(state_dict[classification_key].shape[0])
    architecture_classes = 91 if nc == 80 else nc
    expected_out_channels = 9 * architecture_classes
    if head_out_channels != expected_out_channels:
        raise ValueError(
            f"Checkpoint classification head has {head_out_channels} outputs "
            f"but --nc {nc} expects {expected_out_channels} "
            "(nine anchors per location)."
        )

    probe = LibreRetinaNetModel(size=size, num_classes=architecture_classes)
    result = probe.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {sorted(result.missing_keys)}")
    print(f"Unexpected keys: {sorted(result.unexpected_keys)}")
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"State dict does not match RetinaNet size {size!r} exactly."
        )

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="retinanet",
        size=size,
        nc=nc,
        task="detect",
        imgsz=LibreRetinaNet.INPUT_SIZES[size],
        supported_tasks=("detect",),
        default_task="detect",
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
    temporary.replace(output)
    print(f"Saved LibreYOLO-format checkpoint to {output}")
    return checkpoint


def verify(output_path: str) -> None:
    """Reload the converted checkpoint through the unified factory."""
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    model = LibreYOLO(output_path, device="cpu")
    print(
        f"Loaded back: family={model.family} size={model.size} "
        f"nc={model.nb_classes} task={model.task}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Official torchvision .pth checkpoint")
    parser.add_argument("output", help="Destination LibreRetinaNet<size>.pt")
    parser.add_argument(
        "--size", required=True, choices=["r50", "r50v2"]
    )
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output, args.size, args.nc)
    if args.verify:
        verify(args.output)


if __name__ == "__main__":
    main()
