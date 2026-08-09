"""Convert official RT-DETRv2 detection or OBB weights to LibreYOLO format.

Detection source weights are Apache-2.0 from lyuwenyu/RT-DETR. OBB source
weights are Apache-2.0 from RicePasteM/RiO-DETR commit
22d5232a4e0df6ac4bc26ed1c8aac8b4060449c7.

Adaptation steps (same as the existing HGNetv2 converter):
  1. Unwrap the EMA wrapper: ckpt["ema"]["module"] -> raw state_dict.
  2. Remap encoder/decoder input_proj and decoder.enc_output keys from
     v2's named-submodule style (.conv./.norm./.proj.) to LibreYOLO's
     Sequential numeric style (.0./.1.).
  3. Strict-load every converted tensor into the selected LibreYOLO graph.
  4. Wrap with model_family="rtdetrv2" metadata so the factory routes to
     LibreRTDETRv2 instead of LibreRTDETR.
  5. Atomically save the metadata-wrapped checkpoint.

Usage::

    python weights/convert_rtdetrv2_weights.py downloads/v2_ckpts/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \\
        weights/LibreRTDETRv2r18.pt --size r18
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

# v2 registers ``decoder.anchors`` / ``decoder.valid_mask`` and
# ``cross_attn.num_points_scale`` as buffers; ``convert_to_v2`` keeps them all
# so the strict load overrides our init-time values with the upstream-saved
# tensors. (Initial values differ by ~3e-4 due to torch-version/precision
# drift.)


DETECT_SIZES = ("r18", "r34", "r50", "r50m", "r101")
OBB_SIZES = ("n", "s", "m", "l", "x")

DOTA_NAMES = {
    0: "plane",
    1: "baseball-diamond",
    2: "bridge",
    3: "ground-track-field",
    4: "small-vehicle",
    5: "large-vehicle",
    6: "ship",
    7: "tennis-court",
    8: "basketball-court",
    9: "storage-tank",
    10: "soccer-ball-field",
    11: "roundabout",
    12: "harbor",
    13: "swimming-pool",
    14: "helicopter",
}


def _atomic_save(checkpoint: dict, output_path: str) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        save_checkpoint(checkpoint, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def convert_weights(
    input_path: str,
    output_path: str,
    size: str,
    nc: int | None = None,
    task: str = "detect",
) -> dict:
    add_repo_root_to_path()
    from libreyolo.models.rtdetr.convert import convert_to_v2
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    valid_sizes = OBB_SIZES if task == "obb" else DETECT_SIZES
    if size not in valid_sizes:
        raise ValueError(
            f"RT-DETRv2 task={task!r} supports sizes {valid_sizes}, got {size!r}"
        )

    print(f"Loading upstream RT-DETRv2 weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw)
    print(f"Found {len(state_dict)} parameter entries (EMA-preferred)")

    out = {
        key: (value.float().clone() if value.is_floating_point() else value.clone())
        for key, value in convert_to_v2(state_dict).items()
    }
    detected_nc = LibreRTDETRv2.detect_nb_classes(out)
    if detected_nc is None:
        raise ValueError("Could not infer the RT-DETRv2 checkpoint class count")
    if nc is None:
        nc = detected_nc
    elif nc != detected_nc:
        raise ValueError(
            f"--nc={nc} conflicts with the checkpoint head ({detected_nc} classes)"
        )

    wrapper = LibreRTDETRv2(
        model_path=None,
        size=size,
        task=task,
        nb_classes=nc,
        device="cpu",
    )
    expected = wrapper.model.state_dict()
    missing = sorted(expected.keys() - out.keys())
    unexpected = sorted(out.keys() - expected.keys())
    mismatched = sorted(
        key
        for key in expected.keys() & out.keys()
        if expected[key].shape != out[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Converted tensor consumption is not strict: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatches={mismatched[:5]}"
        )
    wrapper.model.load_state_dict(out, strict=True)

    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        out,
        model_family="rtdetrv2",
        size=size,
        nc=nc,
        names=DOTA_NAMES if task == "obb" and nc == 15 else None,
        task=task,
        imgsz=1024 if task == "obb" else 640,
    )
    _atomic_save(libreyolo_ckpt, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}  ({len(out)} tensors)")
    return libreyolo_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RT-DETRv2 weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream RT-DETRv2 checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument(
        "--size",
        required=True,
        help="Size code matching the upstream backbone",
    )
    parser.add_argument(
        "--task", choices=["detect", "obb"], default="detect", help="Checkpoint task"
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=None,
        help="Expected class count (inferred by default)",
    )
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size, args.nc, args.task)
