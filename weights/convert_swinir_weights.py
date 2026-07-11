"""Convert official SwinIR super-resolution weights to LibreYOLO format.

Supported upstream checkpoints are the official Apache-2.0 SwinIR-S x4
lightweight generator and SwinIR-M/L x4 real-world generators. The source
checkpoint is not modified or redistributed by this script.

Usage::

    python weights/convert_swinir_weights.py upstream.pth \
        weights/LibreSwinIRm-restore.pt --verify
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch

from _conversion_utils import add_repo_root_to_path, load_checkpoint, save_checkpoint

_STATE_KEYS = ("params_ema", "params", "state_dict", "model", "net")


def extract_swinir_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract the generator tensor dictionary from an official checkpoint."""

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary, got {type(checkpoint)!r}.")
    for key in _STATE_KEYS:
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break
    state = {
        key: value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }
    if not state:
        raise ValueError("The checkpoint does not contain a SwinIR tensor state dict.")
    return state


def detect_size(state_dict: dict[str, torch.Tensor]) -> str | None:
    add_repo_root_to_path()
    from libreyolo.models.swinir import LibreSwinIR

    return LibreSwinIR.detect_size(state_dict)


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    size: str | None = None,
    imgsz: int = 64,
    dataset: str | None = None,
) -> dict:
    """Strictly load and wrap one official SwinIR generator checkpoint."""

    raw = load_checkpoint(input_path)
    state_dict = extract_swinir_state_dict(raw)
    detected = detect_size(state_dict)
    if size is None:
        size = detected
    elif detected is not None and detected != size:
        raise ValueError(
            f"Requested size {size!r}, but checkpoint tensors identify {detected!r}."
        )
    if size is None:
        raise ValueError("Could not detect SwinIR size; pass --size {s,m,l}.")

    add_repo_root_to_path()
    from libreyolo.models.swinir import LibreSwinIR
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    model = LibreSwinIR(model_path=None, size=size, device="cpu")
    model.model.load_state_dict(state_dict, strict=True)
    checkpoint = wrap_libreyolo_checkpoint(
        model.model.state_dict(),
        model_family="swinir",
        size=size,
        task="restore",
        nc=1,
        names={0: "image"},
        imgsz=imgsz,
        scale=4,
        degradation="super-resolution",
        dataset=dataset,
    )
    saved = save_checkpoint(checkpoint, output_path)
    digest = hashlib.sha256(Path(saved).read_bytes()).hexdigest()
    print(f"Saved {saved}")
    print(f"sha256: {digest}")
    return checkpoint


def verify_conversion(converted_path: str) -> bool:
    """Load through the public factory and run a small shape smoke test."""

    add_repo_root_to_path()
    from libreyolo import LibreSwinIR, LibreYOLO

    model = LibreYOLO(converted_path, device="cpu")
    if not isinstance(model, LibreSwinIR):
        raise TypeError(f"Factory selected {type(model).__name__}, not LibreSwinIR.")
    model.model.eval()
    with torch.no_grad():
        output = model.model(torch.zeros(1, 3, 8, 8))
    if tuple(output.shape) != (1, 3, 32, 32):
        raise RuntimeError(f"Unexpected SwinIR output shape: {tuple(output.shape)}.")
    print(f"Verified family=swinir size={model.size} task={model.task} scale=4")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert official SwinIR weights")
    parser.add_argument("input", help="Official SwinIR checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument("--size", choices=("s", "m", "l"), default=None)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    convert_weights(
        arguments.input,
        arguments.output,
        size=arguments.size,
        imgsz=arguments.imgsz,
        dataset=arguments.dataset,
    )
    if arguments.verify:
        verify_conversion(arguments.output)
