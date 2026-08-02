"""Prove exact raw-output parity with the pinned official DETR implementation.

This is a manual, external-source test. It requires a checkout of the
Apache-2.0 ``facebookresearch/detr`` repository at the pinned commit and the
four official checkpoints. No upstream source or weights are vendored.

Example:
    python weights/parity_detr.py \
        --upstream-dir ../detr \
        --checkpoint-dir ../detr-checkpoints \
        --device cuda --input-size 800
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

UPSTREAM_COMMIT = "29901c51d7fe8712168b8d0d64351170bc0f83e0"
CHECKPOINTS = {
    "r50": "detr-r50-e632da11.pth",
    "r50dc5": "detr-r50-dc5-f0fb7ef5.pth",
    "r101": "detr-r101-2c7b67e5.pth",
    "r101dc5": "detr-r101-dc5-a2e86def.pth",
}


def _verify_upstream_pin(upstream_dir: Path) -> None:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Could not verify the upstream checkout at {upstream_dir}"
        ) from exc
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Expected facebookresearch/detr {UPSTREAM_COMMIT}, got {revision}"
        )


def _upstream_args(size: str, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_file="coco",
        device=device,
        backbone="resnet101" if size.startswith("r101") else "resnet50",
        dilation=size.endswith("dc5"),
        position_embedding="sine",
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        dim_feedforward=2048,
        enc_layers=6,
        dec_layers=6,
        pre_norm=False,
        num_queries=100,
        aux_loss=False,
        masks=False,
        frozen_weights=None,
        lr_backbone=0.0,
        set_cost_class=1,
        set_cost_bbox=5,
        set_cost_giou=2,
        mask_loss_coef=1,
        dice_loss_coef=1,
        bbox_loss_coef=5,
        giou_loss_coef=2,
        eos_coef=0.1,
    )


def run_parity(
    upstream_dir: Path,
    checkpoint_dir: Path,
    variants: list[str],
    device: str,
    input_size: int,
) -> None:
    _verify_upstream_pin(upstream_dir)
    sys.path.insert(0, str(upstream_dir))

    # Import only after verifying the independently licensed source pin. The
    # official builder otherwise downloads an ImageNet backbone that its DETR
    # checkpoint immediately replaces, so suppress that unnecessary download.
    import models.backbone as upstream_backbone
    from models.detr import build as build_upstream

    upstream_backbone.is_main_process = lambda: False

    from libreyolo.models.detr.nn import LibreDETRModel

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA parity requested but no CUDA device is available")
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    values = torch.linspace(
        -2.0,
        2.0,
        steps=3 * input_size * input_size,
        dtype=torch.float32,
        device=device,
    )
    input_tensor = values.reshape(1, 3, input_size, input_size)

    for size in variants:
        checkpoint_path = checkpoint_dir / CHECKPOINTS[size]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model"]

        reference, _, _ = build_upstream(_upstream_args(size, device))
        port = LibreDETRModel(size=size, nc=91)
        reference.load_state_dict(state_dict, strict=True)
        port.load_state_dict(state_dict, strict=True)
        reference.eval().to(device)
        port.eval().to(device)

        with torch.inference_mode():
            expected = reference(input_tensor)
            actual = port(input_tensor)

        for key in ("pred_logits", "pred_boxes"):
            difference = (expected[key] - actual[key]).abs()
            max_abs_diff = float(difference.max().item())
            nonzero = int(torch.count_nonzero(difference).item())
            print(
                f"{size:8s} {key:11s} shape={tuple(actual[key].shape)} "
                f"max_abs_diff={max_abs_diff:.1f} nonzero={nonzero}"
            )
            if max_abs_diff != 0.0 or nonzero != 0:
                raise AssertionError(
                    f"DETR {size} {key} parity failed: "
                    f"max_abs_diff={max_abs_diff}, nonzero={nonzero}"
                )

        del actual, expected, port, reference, state_dict, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(CHECKPOINTS),
        default=list(CHECKPOINTS),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--input-size", type=int, default=64)
    args = parser.parse_args()

    run_parity(
        args.upstream_dir.resolve(),
        args.checkpoint_dir.resolve(),
        args.variants,
        args.device,
        args.input_size,
    )


if __name__ == "__main__":
    main()
