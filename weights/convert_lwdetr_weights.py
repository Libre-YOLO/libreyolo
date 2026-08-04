"""Convert upstream LW-DETR COCO weights into LibreYOLO format.

Upstream (https://github.com/Atten4Vis/LW-DETR, Apache-2.0) ships
``LWDETR_<size>_60e_coco.pth`` as ``{"model": state_dict, "optimizer": ...,
"args": Namespace}``. The released files were saved with ``--use_ema``, so the
``model`` entry already holds the EMA weights that produce the published mAP.

LibreYOLO's module names mirror upstream exactly, so this is a metadata wrap —
no key remapping. The 91-column classification head is kept as-is (one column
per COCO category id); ``LibreLWDETR`` maps it down to the contiguous COCO-80
interface at postprocess time, so ``nc``/``names`` metadata is the 80-class set.

Usage:
    python weights/convert_lwdetr_weights.py LWDETR_tiny_60e_coco.pth   weights/LibreLWDETRt.pt --size t
    python weights/convert_lwdetr_weights.py LWDETR_small_60e_coco.pth  weights/LibreLWDETRs.pt --size s
    python weights/convert_lwdetr_weights.py LWDETR_medium_60e_coco.pth weights/LibreLWDETRm.pt --size m
    python weights/convert_lwdetr_weights.py LWDETR_large_60e_coco.pth  weights/LibreLWDETRl.pt --size l
    python weights/convert_lwdetr_weights.py LWDETR_xlarge_60e_coco.pth weights/LibreLWDETRx.pt --size x

Add ``--verify`` to load the converted file back through the LibreYOLO factory
and run a smoke inference.
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

# Upstream head width for COCO: one column per category id (max_obj_id + 1).
COCO91_HEAD_WIDTH = 91


def convert_weights(
    input_path: str,
    output_path: str,
    size: str,
    nc: int = 80,
) -> dict:
    print(f"Loading upstream weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw, prefer_ema=True)
    print(f"Found {len(state_dict)} parameter entries")

    head = state_dict.get("class_embed.weight")
    if head is None:
        raise ValueError(
            "Not an LW-DETR checkpoint: 'class_embed.weight' is missing."
        )
    arch_nc = int(head.shape[0])
    if arch_nc != COCO91_HEAD_WIDTH and arch_nc != nc:
        raise ValueError(
            f"Checkpoint head has {arch_nc} columns but --nc is {nc}. Pass a "
            f"matching --nc, or convert an unmodified COCO checkpoint."
        )
    print(f"Classification head width: {arch_nc} (user-facing nc={nc})")

    add_repo_root_to_path()
    from libreyolo.models.lwdetr.model import LibreLWDETR
    from libreyolo.models.lwdetr.nn import LibreLWDETRModel

    detected_size = LibreLWDETR.detect_size(state_dict)
    if detected_size is not None and detected_size != size:
        raise ValueError(
            f"--size {size} does not match the checkpoint architecture "
            f"(detected '{detected_size}'). Re-run with --size {detected_size}."
        )

    # Dry-load into a fresh model so a silent structural mismatch fails here
    # rather than at inference time.
    probe = LibreLWDETRModel(size=size, nc=arch_nc)
    result = probe.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"State dict does not match the {size} architecture.\n"
            f"  missing:    {sorted(result.missing_keys)[:10]}\n"
            f"  unexpected: {sorted(result.unexpected_keys)[:10]}"
        )
    print("Dry-load into a fresh model: no missing or unexpected keys")

    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="lwdetr",
        size=size,
        nc=nc,
        task="detect",
        imgsz=640,
        supported_tasks=("detect",),
        default_task="detect",
    )

    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(libreyolo_ckpt, tmp)
    tmp.replace(out)  # atomic within the same filesystem
    print(f"Saved LibreYOLO-format checkpoint to {out}")
    return libreyolo_ckpt


def verify(output_path: str) -> None:
    import numpy as np

    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    model = LibreYOLO(output_path)
    print(f"Loaded back: family={model.family} size={model.size} nc={model.nb_classes}")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    result = model.predict(dummy, conf=0.01)
    n = 0 if result.boxes is None else len(result.boxes)
    print(f"Smoke inference OK ({n} detections on a blank image)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Upstream LWDETR_<size>_60e_coco.pth")
    parser.add_argument("output", help="Destination LibreLWDETR<size>.pt")
    parser.add_argument("--size", required=True, choices=["t", "s", "m", "l", "x"])
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument(
        "--verify", action="store_true", help="Reload and smoke-test the result"
    )
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size, args.nc)
    if args.verify:
        verify(args.output)


if __name__ == "__main__":
    main()
