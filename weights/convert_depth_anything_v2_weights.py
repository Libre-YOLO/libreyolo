"""Convert official Depth Anything V2 weights into LibreYOLO format.

The released relative-depth checkpoints (e.g. ``depth_anything_v2_vitl.pth``)
ship as a bare state dict whose keys are ``pretrained.*`` (DINOv2 encoder) and
``depth_head.*`` (DPT head). The architecture is vendored under
``libreyolo/models/depth_anything/_vendor`` (Apache-2.0), so this script only
wraps the upstream state dict in the LibreYOLO v1.0 checkpoint schema; the keys
are unchanged, so the load is 0-missing / 0-unexpected.

Weight licensing: the Small (ViT-S) checkpoint is Apache-2.0; Base/Large/Giant
(ViT-B/L/G) are CC-BY-NC-4.0 (non-commercial). LibreYOLO does not redistribute
these weights - you convert your own copy and comply with its license.

Usage::

    python weights/convert_depth_anything_v2_weights.py \
        depth_anything_v2_vitl.pth weights/LibreDepthAnythingV2l-depth.pt

Size is auto-detected from the encoder width; pass ``--size`` to override. Add
``--verify`` to load the converted file through the normal LibreYOLO API and run
a smoke forward pass.
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
)

# DINOv2 cls-token / embedding width -> LibreYOLO size code (ViT-S/B/L/G).
_EMBED_DIM_TO_SIZE = {384: "s", 768: "b", 1024: "l", 1536: "g"}
_IMGSZ = 518


def detect_size(state_dict: dict) -> str | None:
    """Infer the size code from the DINOv2 encoder width."""
    cls_token = state_dict.get("pretrained.cls_token")
    if cls_token is not None and getattr(cls_token, "ndim", 0) >= 1:
        return _EMBED_DIM_TO_SIZE.get(int(cls_token.shape[-1]))
    return None


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    size: str | None = None,
) -> dict:
    # Metric checkpoints share the exact same state-dict keys as the relative
    # ones (the head differs only by a parameter-free Sigmoid * max_depth), so
    # they cannot be told apart by tensors and would load silently wrong into
    # the relative ReLU head. Reject by filename - metric depth is out of scope.
    if "metric" in Path(input_path).name.lower():
        raise ValueError(
            f"{input_path} looks like a metric Depth Anything V2 checkpoint. "
            "LibreYOLO's depth_anything family is relative-depth only; the metric "
            "head (Sigmoid * max_depth) would load but produce wrong values. "
            "Metric depth is out of scope."
        )

    print(f"Loading upstream Depth Anything V2 weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw)
    print(f"Found {len(state_dict)} parameter entries")

    if not any(k.startswith("pretrained.") for k in state_dict) or not any(
        k.startswith("depth_head.") for k in state_dict
    ):
        raise ValueError(
            "This does not look like a Depth Anything V2 checkpoint "
            "(expected 'pretrained.*' encoder and 'depth_head.*' head keys)."
        )

    if size is None:
        size = detect_size(state_dict)
        if size is None:
            raise ValueError(
                "Could not auto-detect size from the encoder width. "
                "Pass --size {s,b,l,g} explicitly."
            )
        print(f"Auto-detected size: {size}")

    add_repo_root_to_path()
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="depth_anything",
        size=size,
        task="depth",
        nc=1,
        names={0: "depth"},
        imgsz=_IMGSZ,
    )

    save_checkpoint(libreyolo_ckpt, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}")
    return libreyolo_ckpt


def verify_conversion(converted_path: str) -> bool:
    """Load through the normal LibreYOLO API and run a smoke forward pass."""
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    print(f"\nLoading converted weights via LibreYOLO({converted_path})...")
    m = LibreYOLO(converted_path, device="cpu")
    print(
        f"  family={m.FAMILY} size={m.size} task={m.task} "
        f"nc={m.nb_classes} names={m.names}"
    )

    m.model.eval()
    with torch.no_grad():
        out = m.model(torch.zeros(1, 3, _IMGSZ, _IMGSZ))
    print(f"  forward pass OK - output shape: {tuple(out.shape)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Depth Anything V2 weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream Depth Anything V2 checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument(
        "--size",
        default=None,
        choices=["s", "b", "l", "g"],
        help="Size code (default: auto-detect from encoder width)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify round-trip after conversion"
    )
    args = parser.parse_args()

    convert_weights(args.input, args.output, size=args.size)
    if args.verify:
        verify_conversion(args.output)
