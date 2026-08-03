"""Convert official IDEA DINO checkpoints into LibreYOLO checkpoint format.

Accepted inputs are native checkpoints from ``IDEA-Research/DINO`` at commit
``d84a491d41898b3befd8294d1cf2614661fc0953`` (Apache-2.0). The released COCO
heads retain their sparse 91 category-id columns; LibreYOLO exposes contiguous
COCO-80 labels after inference.

Examples:
    python weights/convert_dinodetr_weights.py checkpoint0011_4scale.pth weights/LibreDINODETRr50.pt --size r50
    python weights/convert_dinodetr_weights.py checkpoint0011_5scale.pth weights/LibreDINODETRr50s5.pt --size r50s5
    python weights/convert_dinodetr_weights.py checkpoint0027_5scale_swin.pth weights/LibreDINODETRswinl.pt --size swinl
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

SIZES = ("r50", "r50s5", "swinl")
COCO91_HEAD_WIDTH = 91


def _canonical_filename(size: str) -> str:
    return f"LibreDINODETR{size}.pt"


def convert_weights(
    input_path: str,
    output_path: str,
    size: str,
    nc: int = 80,
) -> dict:
    """Wrap one native upstream state dict after a strict architecture load."""
    if size not in SIZES:
        raise ValueError(f"Invalid size {size!r}; expected one of {', '.join(SIZES)}")
    output = Path(output_path)
    expected_name = _canonical_filename(size)
    if output.name != expected_name:
        raise ValueError(
            f"DINO-DETR output must use canonical filename {expected_name!r}, "
            f"not {output.name!r}"
        )

    print(f"Loading upstream weights from {input_path}")
    state_dict = extract_state_dict(load_checkpoint(input_path), prefer_ema=False)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint did not contain a state-dict mapping")
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }

    add_repo_root_to_path()
    from libreyolo.models.dinodetr.model import LibreDINODETR
    from libreyolo.models.dinodetr.nn import LibreDINODETRModel

    if not LibreDINODETR.can_load(state_dict):
        raise ValueError("Input is not a recognized native IDEA DINO checkpoint")

    head = state_dict.get("class_embed.0.weight")
    if head is None:
        raise ValueError("State dict is missing class_embed.0.weight")
    architecture_classes = int(head.shape[0])
    if architecture_classes not in (COCO91_HEAD_WIDTH, nc):
        raise ValueError(
            f"Checkpoint head has {architecture_classes} columns but --nc is {nc}. "
            "Pass a matching --nc, or use an unmodified COCO checkpoint."
        )

    detected_size = LibreDINODETR.detect_size(state_dict)
    if detected_size != size:
        raise ValueError(
            f"--size {size} does not match the checkpoint architecture "
            f"(detected {detected_size})."
        )

    probe = LibreDINODETRModel(size=size, nc=architecture_classes)
    try:
        probe.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"State dict does not strictly match the {size} architecture: {exc}"
        ) from exc
    print(
        f"Strict dry-load passed ({len(state_dict)} entries, "
        f"head width {architecture_classes})"
    )

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="dinodetr",
        size=size,
        nc=nc,
        task="detect",
        imgsz=800,
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


def verify(output_path: str) -> None:
    """Reload through the public factory and run a native forward smoke."""
    import torch

    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    model = LibreYOLO(
        output_path, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.model.eval()
    with torch.inference_mode():
        output = model.model(torch.zeros(1, 3, 256, 256, device=model.device))
    print(
        f"Verified family={model.family} size={model.size} nc={model.nb_classes}; "
        f"logits={tuple(output['pred_logits'].shape)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Official IDEA DINO .pth checkpoint")
    parser.add_argument("output", help="Destination canonical LibreDINODETR*.pt")
    parser.add_argument("--size", required=True, choices=SIZES)
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output, args.size, args.nc)
    if args.verify:
        verify(args.output)


if __name__ == "__main__":
    main()
