"""Convert official HRNet pose checkpoints to LibreYOLO metadata format.

Code upstream: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
Commit: 6f69e4676ad8d43d0d61b64b1b9726f0c369e7b1
Code license: MIT

The official project distributes pretrained checkpoints without a separate
per-file license. LibreYOLO records the redistribution basis and that limited
license conclusion in ``docs/provenance/hrnet.md``; this script makes no
broader claim about the weights.

No tensor is transformed for ``--source original``. The ``mmpose`` source
option only removes known container prefixes so compatible HRNet pose tensors
can be validated against the native LibreYOLO module tree.

Usage:
    python weights/convert_hrnet_weights.py pose_hrnet_w32_256x192.pth \
        weights/LibreHRNetw32-pose.pt --size w32 --source original --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


INPUT_SIZES = {
    "w32": (256, 192),
    "w48": (384, 288),
}


def _tensor_state_dict(candidate: Any) -> dict[str, torch.Tensor]:
    """Validate that a checkpoint payload is a string-to-tensor mapping."""
    if not isinstance(candidate, dict):
        raise TypeError("HRNet checkpoint did not contain a state dict")
    invalid = [
        key
        for key, value in candidate.items()
        if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    ]
    if invalid:
        preview = ", ".join(repr(key) for key in invalid[:3])
        raise TypeError(
            "HRNet state dict must map string keys to tensors; invalid entries: "
            f"{preview}"
        )
    return dict(candidate)


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove a distributed-training ``module.`` prefix when present."""
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _remap_mmpose_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map common MMPose HRNet backbone/head prefixes to native names."""
    remapped: dict[str, torch.Tensor] = {}
    ignored_prefixes = ("data_preprocessor.",)
    head_prefixes = (
        "keypoint_head.final_layer.0.",
        "keypoint_head.final_layer.",
        "head.final_layer.0.",
        "head.final_layer.",
    )

    for key, value in state_dict.items():
        if key.startswith(ignored_prefixes):
            continue
        if key.startswith("backbone."):
            mapped = key.removeprefix("backbone.")
        else:
            mapped = key
            for prefix in head_prefixes:
                if key.startswith(prefix):
                    mapped = f"final_layer.{key.removeprefix(prefix)}"
                    break

        if mapped in remapped:
            raise ValueError(f"MMPose remapping produced duplicate key {mapped!r}")
        remapped[mapped] = value
    return remapped


def normalize_state_dict(raw: Any, source: str) -> dict[str, torch.Tensor]:
    """Extract and normalize one supported upstream checkpoint layout."""
    state_dict = _tensor_state_dict(extract_state_dict(raw, prefer_ema=False))
    state_dict = _strip_module_prefix(state_dict)
    if source == "mmpose":
        state_dict = _remap_mmpose_state_dict(state_dict)
    return state_dict


def convert_weights(
    input_path: str,
    output_path: str,
    size: str,
    source: str = "original",
) -> dict[str, Any]:
    """Strictly validate and metadata-wrap one HRNet pose state dict."""
    if size not in INPUT_SIZES:
        raise ValueError(f"Unsupported HRNet size {size!r}; choose one of {sorted(INPUT_SIZES)}")
    if source not in {"original", "mmpose"}:
        raise ValueError("source must be 'original' or 'mmpose'")

    print(f"Loading {source} HRNet weights from {input_path}")
    state_dict = normalize_state_dict(load_checkpoint(input_path), source)

    add_repo_root_to_path()
    from libreyolo.data.pose_metadata import COCO17_OKS_SIGMAS
    from libreyolo.models.hrnet.model import LibreHRNet
    from libreyolo.models.hrnet.nn import HRNetPoseModel

    if not LibreHRNet.can_load(state_dict):
        raise ValueError(
            "Not a supported COCO-17 HRNet pose checkpoint: required stem, "
            "stage, or final heatmap-head tensors are missing or incompatible."
        )

    detected_size = LibreHRNet.detect_size(state_dict)
    if detected_size != size:
        raise ValueError(
            f"--size {size} does not match the checkpoint architecture "
            f"(detected {detected_size!r})."
        )

    width = 32 if size == "w32" else 48
    probe = HRNetPoseModel(width=width, num_keypoints=17)
    probe.load_state_dict(state_dict, strict=True)
    print(f"Strict load succeeded for HRNet-{width} ({len(state_dict)} tensors).")

    height, width_px = INPUT_SIZES[size]
    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="hrnet",
        size=size,
        task="pose",
        nc=1,
        names={0: "person"},
        imgsz=height,
        imgsz_h=height,
        imgsz_w=width_px,
        num_keypoints=17,
        keypoint_dim=3,
        oks_sigmas=list(COCO17_OKS_SIGMAS),
        supported_tasks=("pose",),
        default_task="pose",
    )

    output = Path(output_path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
    temporary.replace(output)
    print(f"Saved LibreYOLO-format checkpoint to {output}")
    return checkpoint


def verify(output_path: str) -> None:
    """Reload the converted checkpoint through the unified factory and run it."""
    add_repo_root_to_path()
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    loaded = load_checkpoint(output_path)
    validate_checkpoint_metadata(loaded, strict=True)
    model = LibreYOLO(output_path, device="cpu")
    expected_size = INPUT_SIZES[model.size]
    with torch.inference_mode():
        heatmaps = model.model(torch.zeros((1, 3, *expected_size)))
    expected_shape = (1, 17, expected_size[0] // 4, expected_size[1] // 4)
    if tuple(heatmaps.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected HRNet output shape {tuple(heatmaps.shape)}; "
            f"expected {expected_shape}."
        )
    print(
        f"Loaded back: family={model.family} size={model.size} task={model.task} "
        f"nc={model.nb_classes} output={tuple(heatmaps.shape)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Trusted upstream HRNet .pth checkpoint")
    parser.add_argument("output", help="Destination LibreHRNet<size>-pose.pt")
    parser.add_argument("--size", required=True, choices=sorted(INPUT_SIZES))
    parser.add_argument(
        "--source",
        default="original",
        choices=("original", "mmpose"),
        help="Checkpoint key layout (default: original)",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output, args.size, args.source)
    if args.verify:
        verify(args.output)


if __name__ == "__main__":
    main()
