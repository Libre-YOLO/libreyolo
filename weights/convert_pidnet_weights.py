"""Convert official PIDNet Cityscapes weights into LibreYOLO format.

Upstream PIDNet checkpoints are MIT-licensed and usually contain a state dict
with PIDNet module keys, sometimes under a leading ``model.`` or ``module.``
prefix. Training checkpoints may also include auxiliary heads
(``seghead_p.*`` and ``seghead_d.*``). LibreYOLO ships inference-only PIDNet, so
the converter keeps the main network and final semantic head only.

Usage:

    python weights/convert_pidnet_weights.py \
        PIDNet_S_Cityscapes_val.pt weights/LibrePIDNets-sem.pt --verify
"""

from __future__ import annotations

import argparse

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
)


_AUX_HEAD_PREFIXES = ("seghead_p.", "seghead_d.")
_COMMON_PREFIXES = ("module.", "model.", "net.")
_IMGSZ = 1024


def _strip_known_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _COMMON_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def normalize_pidnet_state_dict(state_dict: dict) -> dict[str, torch.Tensor]:
    """Strip wrappers and drop PIDNet training-only auxiliary heads."""
    normalized = {}
    for raw_key, value in state_dict.items():
        key = _strip_known_prefix(str(raw_key))
        if key.startswith(_AUX_HEAD_PREFIXES):
            continue
        normalized[key] = value
    return normalized


def _filter_to_runtime_state(state_dict: dict, *, size: str, nc: int) -> dict:
    add_repo_root_to_path()
    from libreyolo.models.pidnet.nn import LibrePIDNetNet

    target = LibrePIDNetNet(size=size, num_classes=nc).state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in target and getattr(value, "shape", None) == target[key].shape
    }
    missing = [
        key
        for key in target
        if key not in filtered and not key.endswith("num_batches_tracked")
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"PIDNet checkpoint is missing runtime keys: {preview}")
    return filtered


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    size: str | None = None,
) -> dict:
    print(f"Loading upstream PIDNet weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = normalize_pidnet_state_dict(extract_state_dict(raw))
    print(f"Found {len(state_dict)} runtime parameter entries")

    add_repo_root_to_path()
    from libreyolo.models.pidnet.model import CITYSCAPES_NAMES, LibrePIDNet
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    if not LibrePIDNet.can_load(state_dict):
        raise ValueError(
            "This does not look like a PIDNet checkpoint "
            "(expected PIDNet stem, fusion modules, and final_layer keys)."
        )

    detected_size = LibrePIDNet.detect_size(state_dict)
    if size is None:
        size = detected_size
        if size is None:
            raise ValueError("Could not auto-detect PIDNet size. Pass --size.")
        print(f"Auto-detected size: {size}")
    elif detected_size is not None and detected_size != size:
        raise ValueError(f"--size {size!r} conflicts with detected size {detected_size!r}")

    nc = LibrePIDNet.detect_nb_classes(state_dict)
    if nc != 19:
        raise ValueError(
            f"PIDNet v1 conversion is Cityscapes-only; expected 19 classes, got {nc}."
        )

    runtime_state = _filter_to_runtime_state(state_dict, size=size, nc=nc)
    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        runtime_state,
        model_family="pidnet",
        size=size,
        task="semantic",
        nc=19,
        names=CITYSCAPES_NAMES,
        imgsz=_IMGSZ,
    )

    save_checkpoint(libreyolo_ckpt, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}")
    return libreyolo_ckpt


def verify_conversion(converted_path: str) -> bool:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    print(f"\nLoading converted weights via LibreYOLO({converted_path})...")
    model = LibreYOLO(converted_path, device="cpu")
    print(
        f"  family={model.FAMILY} size={model.size} task={model.task} "
        f"nc={model.nb_classes}"
    )

    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, 64, 64))
    print(f"  forward pass OK - output shape: {tuple(out.shape)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PIDNet Cityscapes weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream PIDNet checkpoint (.pt)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument(
        "--size",
        choices=["s", "m", "l"],
        default=None,
        help="Size code (default: auto-detect from checkpoint)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify round-trip after conversion"
    )
    args = parser.parse_args()

    convert_weights(args.input, args.output, size=args.size)
    if args.verify:
        verify_conversion(args.output)
