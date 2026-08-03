"""Convert official xingyizhou/CenterNet detection checkpoints.

The accepted files are the MIT project's released ``ctdet_coco_resdcn18``
and ``ctdet_coco_dla_2x`` checkpoints. Conversion removes the data-parallel
``module.`` prefix and adds LibreYOLO checkpoint metadata; learned tensors are
otherwise unchanged.

Examples:
    python weights/convert_centernet_weights.py ctdet_coco_resdcn18.pth weights/LibreCenterNetresdcn18.pt --size resdcn18
    python weights/convert_centernet_weights.py ctdet_coco_dla_2x.pth weights/LibreCenterNetdla34.pt --size dla34
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

SIZES = ("resdcn18", "dla34")


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(key.startswith("module.") for key in state_dict):
        return dict(state_dict)
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _detect_size(state_dict: dict) -> str | None:
    if (
        "conv1.weight" in state_dict
        and "deconv_layers.0.conv_offset_mask.weight" in state_dict
        and "hm.2.weight" in state_dict
    ):
        return "resdcn18"
    if (
        "base.base_layer.0.weight" in state_dict
        and "dla_up.ida_0.proj_1.conv.conv_offset_mask.weight" in state_dict
        and "hm.2.weight" in state_dict
    ):
        return "dla34"
    return None


def convert_weights(
    input_path: str,
    output_path: str,
    size: str,
    nc: int = 80,
) -> dict:
    """Validate, metadata-wrap, and atomically save one official checkpoint."""
    if size not in SIZES:
        raise ValueError(f"Invalid size {size!r}; expected one of {', '.join(SIZES)}")

    raw = load_checkpoint(input_path)
    state_dict = _strip_module_prefix(extract_state_dict(raw, prefer_ema=False))
    detected_size = _detect_size(state_dict)
    if detected_size is None:
        raise ValueError("Input is not a recognized CenterNet detection checkpoint")
    if detected_size != size:
        raise ValueError(
            f"--size {size} does not match the checkpoint architecture "
            f"(detected {detected_size})"
        )

    heatmap = state_dict.get("hm.2.weight")
    if heatmap is None:
        raise ValueError("Checkpoint is missing hm.2.weight")
    checkpoint_nc = int(heatmap.shape[0])
    if checkpoint_nc != nc:
        raise ValueError(
            f"Checkpoint heatmap has {checkpoint_nc} classes but --nc is {nc}"
        )

    add_repo_root_to_path()
    from libreyolo.models.centernet.nn import build_centernet

    probe = build_centernet(size, num_classes=checkpoint_nc)
    try:
        probe.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"State dict does not strictly match CenterNet {size}: {exc}"
        ) from exc

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="centernet",
        size=size,
        task="detect",
        nc=nc,
        imgsz=512,
        supported_tasks=("detect",),
        default_task="detect",
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
    temporary.replace(output)
    print(
        f"Converted CenterNet {size}: {len(state_dict)} tensors, "
        f"{checkpoint_nc} classes -> {output}"
    )
    return checkpoint


def verify(output_path: str) -> None:
    """Strictly reload the converted state into its native architecture."""
    import torch

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    size = checkpoint["size"]
    add_repo_root_to_path()
    from libreyolo.models.centernet.nn import build_centernet

    model = build_centernet(size, num_classes=int(checkpoint["nc"])).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 128, 128))
    shapes = {key: tuple(value.shape) for key, value in output.items()}
    print(f"Verified family=centernet size={size}: {shapes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Official ctdet_coco_*.pth checkpoint")
    parser.add_argument("output", help="Destination LibreCenterNet*.pt")
    parser.add_argument("--size", required=True, choices=SIZES)
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output, args.size, args.nc)
    if args.verify:
        verify(args.output)


if __name__ == "__main__":
    main()
