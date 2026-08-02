"""Convert Deformable DETR weights into LibreYOLO checkpoint format.

Accepted inputs:

- native checkpoints from ``fundamentalvision/Deformable-DETR`` (Apache-2.0),
- official ``SenseTime/deformable-detr*`` Transformers safetensors
  (Apache-2.0).

The released COCO heads have 91 category-id columns. They remain unchanged in
the architecture; LibreYOLO exposes contiguous COCO-80 labels after inference.

Examples:
    python weights/convert_deformable_detr_weights.py model.safetensors weights/LibreDeformableDETRr50.pt --size r50
    python weights/convert_deformable_detr_weights.py checkpoint.pth weights/LibreDeformableDETRr50ssdc5.pt --size r50ssdc5
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

SIZES = ("r50ss", "r50ssdc5", "r50", "r50refine", "r50twostage")
COCO91_HEAD_WIDTH = 91


def _load_state_dict(input_path: str) -> dict:
    path = Path(input_path)
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Converting .safetensors requires the safetensors package"
            ) from exc
        return load_file(str(path), device="cpu")
    return extract_state_dict(load_checkpoint(path), prefer_ema=True)


def convert_weights(
    input_path: str,
    output_path: str,
    size: str,
    nc: int = 80,
) -> dict:
    """Convert one native or Transformers checkpoint and validate it strictly."""
    if size not in SIZES:
        raise ValueError(f"Invalid size {size!r}; expected one of {', '.join(SIZES)}")

    print(f"Loading upstream weights from {input_path}")
    state_dict = _load_state_dict(input_path)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint did not contain a state-dict mapping")

    add_repo_root_to_path()
    from libreyolo.models.deformable_detr.conversion import (
        is_hf_deformable_detr_state_dict,
        convert_hf_deformable_detr_state_dict,
    )
    from libreyolo.models.deformable_detr.model import LibreDeformableDETR
    from libreyolo.models.deformable_detr.nn import LibreDeformableDETRModel

    if is_hf_deformable_detr_state_dict(state_dict):
        print("Detected official Transformers key layout; remapping to native keys")
        state_dict = convert_hf_deformable_detr_state_dict(state_dict)
    elif not LibreDeformableDETR.can_load(state_dict):
        raise ValueError("Input is not a recognized Deformable DETR checkpoint")

    head = state_dict.get("class_embed.0.weight")
    if head is None:
        raise ValueError("Converted state dict is missing class_embed.0.weight")
    arch_nc = int(head.shape[0])
    if arch_nc != COCO91_HEAD_WIDTH and arch_nc != nc:
        raise ValueError(
            f"Checkpoint head has {arch_nc} columns but --nc is {nc}. Pass a "
            "matching --nc, or use an unmodified COCO checkpoint."
        )

    detected_size = LibreDeformableDETR.detect_size(state_dict)
    if detected_size is not None and detected_size != size:
        raise ValueError(
            f"--size {size} does not match the checkpoint architecture "
            f"(detected {detected_size})."
        )

    probe = LibreDeformableDETRModel(size=size, nc=arch_nc)
    try:
        probe.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"State dict does not strictly match the {size} architecture: {exc}"
        ) from exc
    print(f"Strict dry-load passed ({len(state_dict)} entries, head width {arch_nc})")

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="deformable_detr",
        size=size,
        nc=nc,
        task="detect",
        imgsz=800,
        supported_tasks=("detect",),
        default_task="detect",
    )
    output = Path(output_path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
    temporary.replace(output)
    print(f"Saved LibreYOLO checkpoint to {output}")
    return checkpoint


def verify(output_path: str) -> None:
    """Reload through the public factory and run a lightweight forward smoke."""
    import torch

    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    model = LibreYOLO(output_path, device="cpu")
    model.model.eval()
    with torch.inference_mode():
        output = model.model(torch.zeros(1, 3, 128, 128))
    print(
        f"Verified family={model.family} size={model.size} nc={model.nb_classes}; "
        f"logits={tuple(output['pred_logits'].shape)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Native .pth/.pt or official model.safetensors")
    parser.add_argument("output", help="Destination LibreDeformableDETR*.pt")
    parser.add_argument("--size", required=True, choices=SIZES)
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size, args.nc)
    if args.verify:
        verify(args.output)


if __name__ == "__main__":
    main()
