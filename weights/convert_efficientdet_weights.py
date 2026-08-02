"""Convert official EfficientDet checkpoints to LibreYOLO metadata format.

The source checkpoints are the TensorFlow-ported D0-D4 state dictionaries from
``rwightman/efficientdet-pytorch`` 0.4.1, Apache-2.0. LibreYOLO preserves every
learned tensor and the upstream parameter names. The official 90-output sparse
COCO head remains in the graph while metadata exposes LibreYOLO's contiguous
COCO-80 interface.

Usage::

    python weights/convert_efficientdet_weights.py \
        tf_efficientdet_d0_34-f153e0cf.pth LibreEfficientDetd0.pt --size d0 --verify
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

SIZES = ("d0", "d1", "d2", "d3", "d4")
COCO_PUBLIC_CLASSES = 80
COCO_ARCH_CLASSES = 90
ANCHORS_PER_LOCATION = 9


def _load_official_state_dict(input_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
    state_dict = extract_state_dict(checkpoint, prefer_ema=False)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("EfficientDet checkpoint does not contain a state dict")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("EfficientDet state dict must contain only string keys and tensors")
    return state_dict


def convert_weights(input_path: str, output_path: str, size: str) -> dict:
    """Strict-load and atomically metadata-wrap one official checkpoint."""
    if size not in SIZES:
        raise ValueError(f"Unsupported EfficientDet size {size!r}; choose one of {SIZES}")

    add_repo_root_to_path()
    from libreyolo.models.efficientdet.model import LibreEfficientDet
    from libreyolo.models.efficientdet.nn import LibreEfficientDetModel

    output = Path(output_path)
    expected_name = f"{LibreEfficientDet.FILENAME_PREFIX}{size}.pt"
    if output.name != expected_name:
        raise ValueError(
            f"Canonical output filename for size {size!r} is {expected_name!r}, "
            f"not {output.name!r}."
        )

    print(f"Loading official EfficientDet weights from {input_path}")
    state_dict = _load_official_state_dict(input_path)
    print(f"Found {len(state_dict)} tensor entries")
    if not LibreEfficientDet.can_load(state_dict):
        raise ValueError("Checkpoint does not match the EfficientDet BiFPN signature")

    filename_size = LibreEfficientDet.detect_size_from_filename(Path(input_path).name)
    detected_size = LibreEfficientDet.detect_size(state_dict)
    for source, candidate in (("filename", filename_size), ("architecture", detected_size)):
        if candidate is not None and candidate != size:
            raise ValueError(
                f"--size {size} conflicts with checkpoint {source}, which identifies {candidate}"
            )

    head_width = int(state_dict["class_net.predict.conv_pw.weight"].shape[0])
    if head_width != COCO_ARCH_CLASSES * ANCHORS_PER_LOCATION:
        raise ValueError(
            "Expected the official COCO class-head width "
            f"{COCO_ARCH_CLASSES * ANCHORS_PER_LOCATION}, got {head_width}."
        )

    probe = LibreEfficientDetModel(size=size, num_classes=COCO_ARCH_CLASSES)
    result = probe.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:  # pragma: no cover
        raise ValueError(
            f"State dict mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    print(f"Strict-load verified for EfficientDet {size}")

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="efficientdet",
        size=size,
        nc=COCO_PUBLIC_CLASSES,
        task="detect",
        imgsz=LibreEfficientDet.INPUT_SIZES[size],
        supported_tasks=("detect",),
        default_task="detect",
    )

    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        save_checkpoint(checkpoint, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved LibreYOLO checkpoint to {output}")
    return checkpoint


def verify(output_path: str, size: str) -> None:
    """Validate metadata and reload through the unified factory."""
    add_repo_root_to_path()
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    validate_checkpoint_metadata(checkpoint)
    model = LibreYOLO(output_path, device="cpu")
    actual = (model.family, model.size, model.nb_classes, model.input_size)
    expected = ("efficientdet", size, COCO_PUBLIC_CLASSES, model.INPUT_SIZES[size])
    if actual != expected:
        raise RuntimeError(f"Round-trip metadata mismatch: expected {expected}, got {actual}")
    print(
        f"Round-trip verified: family={model.family} size={model.size} "
        f"nc={model.nb_classes} imgsz={model.input_size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Official tf_efficientdet_d*.pth checkpoint")
    parser.add_argument("output", help="Canonical LibreEfficientDetd*.pt destination")
    parser.add_argument("--size", required=True, choices=SIZES)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size)
    if args.verify:
        verify(args.output, args.size)


if __name__ == "__main__":
    main()
