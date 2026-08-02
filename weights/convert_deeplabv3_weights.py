"""Convert official torchvision DeepLabv3 weights to LibreYOLO checkpoints.

The upstream checkpoints include a training-only auxiliary FCN classifier.
LibreYOLO's inference graph deliberately omits that branch, so conversion
removes only ``aux_classifier.*`` and preserves every runtime tensor exactly.

Usage:
    python weights/convert_deeplabv3_weights.py \
        deeplabv3_resnet50_coco-cd0a2569.pth \
        weights/LibreDeepLabv3r50-sem.pt --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


_IMGSZ = 520


def _strict_runtime_check(
    state_dict: dict[str, torch.Tensor],
    *,
    size: str,
    nc: int,
) -> None:
    add_repo_root_to_path()
    from libreyolo.models.deeplabv3.nn import LibreDeepLabv3Net

    target = LibreDeepLabv3Net(size=size, num_classes=nc).eval()
    target_state = target.state_dict()
    missing = sorted(set(target_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(target_state))
    mismatched = sorted(
        key
        for key in set(target_state) & set(state_dict)
        if tuple(target_state[key].shape) != tuple(state_dict[key].shape)
    )
    print(f"Missing runtime keys: {missing}")
    print(f"Unexpected runtime keys: {unexpected}")
    print(f"Shape-mismatched runtime keys: {mismatched}")
    if missing or unexpected or mismatched:
        raise ValueError("DeepLabv3 runtime state dict is not a strict graph match")
    target.load_state_dict(state_dict, strict=True)


def convert_weights(
    input_path: str | Path,
    output_path: str | Path,
    *,
    size: str | None = None,
) -> Path:
    """Convert one official checkpoint and write it atomically."""
    raw = load_checkpoint(input_path)
    extracted = extract_state_dict(raw, prefer_ema=False)
    if not isinstance(extracted, dict):
        raise TypeError("DeepLabv3 checkpoint does not contain a state dict")
    upstream_count = len(extracted)
    add_repo_root_to_path()
    from libreyolo.models.deeplabv3.convert import (
        convert_upstream_deeplabv3_state_dict,
    )

    state_dict = convert_upstream_deeplabv3_state_dict(extracted)
    if state_dict is None:
        raise ValueError(
            "This is not an official-layout torchvision DeepLabv3 checkpoint "
            "with both ASPP and auxiliary FCN fingerprints."
        )
    print(
        f"Extracted {upstream_count} entries; retained {len(state_dict)} "
        f"runtime entries and removed {upstream_count - len(state_dict)} auxiliary entries"
    )

    from libreyolo.models.deeplabv3.model import LibreDeepLabv3, VOC_NAMES

    if not LibreDeepLabv3.can_load(state_dict):
        raise ValueError(
            "This is not a recognized DeepLabv3 checkpoint with the ASPP "
            "branch/project fingerprint."
        )
    detected_size = LibreDeepLabv3.detect_size(state_dict)
    if size is None:
        size = detected_size
        if size is None:
            raise ValueError("Could not infer DeepLabv3 size; pass --size explicitly")
        print(f"Auto-detected size: {size}")
    elif detected_size is not None and detected_size != size:
        raise ValueError(
            f"--size {size!r} conflicts with checkpoint size {detected_size!r}"
        )

    nc = LibreDeepLabv3.detect_nb_classes(state_dict)
    if nc != 21:
        raise ValueError(
            "Official DeepLabv3 conversion expects 21 COCO-with-VOC-label "
            f"classes, got {nc!r}."
        )
    _strict_runtime_check(state_dict, size=size, nc=nc)

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="deeplabv3",
        size=size,
        task="semantic",
        nc=nc,
        names=VOC_NAMES,
        imgsz=_IMGSZ,
        supported_tasks=("semantic",),
        default_task="semantic",
    )

    output = Path(output_path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
    temporary.replace(output)
    print(f"Wrote {output}")
    return output


def verify_conversion(converted_path: str | Path) -> None:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    payload = load_untrusted_torch_file(
        converted_path,
        map_location="cpu",
        context="converted DeepLabv3 checkpoint",
    )
    errors = validate_checkpoint_metadata(payload, strict=True)
    if errors:
        raise ValueError("Invalid converted metadata: " + "; ".join(errors))
    model = LibreYOLO(str(converted_path), device="cpu")
    with torch.no_grad():
        output = model.model(torch.zeros(1, 3, 32, 32))
    if tuple(output.shape) != (1, 21, 32, 32):
        raise RuntimeError(f"Unexpected converted forward shape: {tuple(output.shape)}")
    print(
        f"Verified family={model.FAMILY} size={model.size} task={model.task} "
        f"nc={model.nb_classes} shape={tuple(output.shape)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", choices=("r50", "r101", "mv3"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    converted = convert_weights(args.input, args.output, size=args.size)
    if args.verify:
        verify_conversion(converted)


if __name__ == "__main__":
    main()
