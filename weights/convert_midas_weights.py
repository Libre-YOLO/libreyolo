"""Convert official MiDaS weights into LibreYOLO checkpoint format.

The official MIT checkpoints from ``isl-org/MiDaS`` already use the module
names preserved by LibreYOLO's native port, so conversion is a metadata wrap:
learned tensors and keys are unchanged.

Canonical variants:

* ``midas_v21_small_256.pt`` -> ``LibreMiDaSs-depth.pt``
* ``dpt_large_384.pt`` -> ``LibreMiDaSl-depth.pt``

Usage::

    python weights/convert_midas_weights.py \
        midas_v21_small_256.pt LibreMiDaSs-depth.pt --verify
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

_INPUT_SIZES = {"s": 256, "l": 384}


def _detect_size(state_dict: dict) -> str | None:
    cls_token = state_dict.get("pretrained.model.cls_token")
    if cls_token is not None and tuple(cls_token.shape) == (1, 1, 1024):
        return "l"
    stem = state_dict.get("pretrained.layer1.0.weight")
    if stem is not None and tuple(stem.shape[:2]) == (32, 3):
        return "s"
    return None


def _dry_load(state_dict: dict, size: str) -> None:
    """Print the exact state-dict compatibility result before writing."""
    add_repo_root_to_path()
    from libreyolo.models.midas.nn import build_midas_model

    model = build_midas_model(size)
    result = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {result.missing_keys}")
    print(f"Unexpected keys: {result.unexpected_keys}")
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "MiDaS checkpoint is not an exact match for the native architecture."
        )


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    size: str | None = None,
) -> dict:
    """Wrap one official MiDaS state dict with strict v1.0 metadata."""
    print(f"Loading official MiDaS weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw)
    print(f"Found {len(state_dict)} parameter entries")

    detected = _detect_size(state_dict)
    if detected is None:
        raise ValueError(
            "This does not look like a supported MiDaS checkpoint. Expected "
            "the official DPT-Large ViT token or v2.1 Small EfficientNet stem."
        )
    if size is None:
        size = detected
        print(f"Auto-detected size: {size}")
    elif size != detected:
        raise ValueError(
            f"--size {size!r} contradicts the checkpoint signature ({detected!r})."
        )

    canonical_name = f"LibreMiDaS{size}-depth.pt"
    if Path(output_path).name != canonical_name:
        raise ValueError(
            f"MiDaS output must use the canonical filename {canonical_name!r}; "
            f"got {Path(output_path).name!r}."
        )

    _dry_load(state_dict, size)
    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="midas",
        size=size,
        nc=1,
        names={0: "depth"},
        task="depth",
        imgsz=_INPUT_SIZES[size],
        supported_tasks=("depth",),
        default_task="depth",
    )

    output = Path(output_path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(wrapped, temporary)
    temporary.rename(output)
    print(f"Wrote {output}")
    return wrapped


def verify_conversion(output_path: str) -> None:
    """Validate metadata, load through the public factory, and smoke-forward."""
    add_repo_root_to_path()
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    errors = validate_checkpoint_metadata(checkpoint, strict=True)
    if errors:
        raise ValueError("Invalid checkpoint metadata: " + "; ".join(errors))

    model = LibreYOLO(output_path, device="cpu")
    model.model.eval()
    with torch.inference_mode():
        output = model.model(torch.zeros(1, 3, 64, 64))
    if tuple(output.shape) != (1, 1, 64, 64):
        raise AssertionError(f"Unexpected MiDaS output shape: {tuple(output.shape)}")
    print(
        f"Verified {output_path}: family={model.family}, size={model.size}, "
        f"task={model.task}, output={tuple(output.shape)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="Official MiDaS .pt checkpoint")
    parser.add_argument("output", help="Canonical LibreYOLO .pt output")
    parser.add_argument(
        "--size",
        choices=sorted(_INPUT_SIZES),
        default=None,
        help="Optional size override; the checkpoint signature is authoritative.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Validate metadata and run a public-factory forward smoke.",
    )
    args = parser.parse_args()
    convert_weights(args.input, args.output, size=args.size)
    if args.verify:
        verify_conversion(args.output)
