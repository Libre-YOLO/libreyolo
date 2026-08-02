"""Wrap the official torchvision AlexNet checkpoint in LibreYOLO metadata.

The native graph preserves torchvision's state-dict keys, so conversion does
not remap or modify learned tensors. It validates the complete state dict with
``strict=True`` and writes the result atomically.

Code upstream: https://github.com/pytorch/vision/tree/v0.26.0
Commit: 336d36e8db990a905498c73933e35231876e28bc
Code license: BSD-3-Clause

The checkpoint has no separate per-object license file. LibreYOLO's disclosed
implied-BSD redistribution basis and torchvision's pretrained-model caveat are
recorded in ``docs/provenance/alexnet.md``.

Usage:
    python weights/convert_alexnet_weights.py alexnet-owt-7be5be79.pth \
        weights/LibreAlexNetb-cls.pt --size b --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    imagenet1k_names,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

_CANONICAL_FILENAME = "LibreAlexNetb-cls.pt"
_NUM_CLASSES = 1000


def convert_weights(input_path: str, output_path: str, size: str = "b") -> dict:
    """Validate and metadata-wrap the official AlexNet state dict."""
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw, prefer_ema=True)
    if not isinstance(state_dict, dict):
        raise TypeError("AlexNet checkpoint did not contain a state dict")

    add_repo_root_to_path()
    from libreyolo.models.alexnet.model import LibreAlexNet
    from libreyolo.models.alexnet.nn import AlexNet

    detected_size = LibreAlexNet.detect_size(state_dict)
    if detected_size != size:
        raise ValueError(
            f"--size {size} does not match the checkpoint architecture "
            f"(detected {detected_size!r})."
        )
    detected_nc = LibreAlexNet.detect_nb_classes(state_dict)
    if detected_nc != _NUM_CLASSES:
        raise ValueError(
            "The official AlexNet checkpoint must have 1000 classifier outputs; "
            f"found {detected_nc!r}."
        )

    probe = AlexNet(num_classes=_NUM_CLASSES)
    result = probe.load_state_dict(state_dict, strict=True)
    print(f"Missing keys: {sorted(result.missing_keys)}")
    print(f"Unexpected keys: {sorted(result.unexpected_keys)}")

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="alexnet",
        size=size,
        nc=_NUM_CLASSES,
        names=imagenet1k_names(),
        task="classify",
        imgsz=LibreAlexNet.INPUT_SIZES[size],
        supported_tasks=("classify",),
        default_task="classify",
    )

    output = Path(output_path)
    if output.name != _CANONICAL_FILENAME:
        raise ValueError(
            f"Canonical AlexNet output must be named {_CANONICAL_FILENAME}, "
            f"got {output.name}."
        )
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
    temporary.replace(output)
    print(f"Saved LibreYOLO-format checkpoint to {output}")
    return checkpoint


def verify(output_path: str) -> None:
    """Strict-load the converted checkpoint through the unified factory."""
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    model = LibreYOLO(output_path, device="cpu")
    print(
        f"Loaded back: family={model.family} size={model.size} "
        f"task={model.task} nc={model.nb_classes}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Official alexnet-owt-7be5be79.pth file")
    parser.add_argument("output", help=f"Destination {_CANONICAL_FILENAME}")
    parser.add_argument("--size", required=True, choices=["b"])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size)
    if args.verify:
        verify(args.output)


if __name__ == "__main__":
    main()
