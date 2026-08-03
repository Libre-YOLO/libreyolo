"""Prove exact raw-logit parity with torchvision's official DeepLabv3 models.

This manual acceptance test requires the three official torchvision v0.26.0
checkpoints. It verifies their SHA-256 digests, loads the upstream graphs and
LibreYOLO inference graphs strictly, and compares the semantic logits before
any task postprocessing.

Example:
    python weights/parity_deeplabv3.py \
        --checkpoint-dir ../deeplabv3-checkpoints --device cuda
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path

import torch
import torchvision
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet101,
    deeplabv3_resnet50,
)

from _conversion_utils import add_repo_root_to_path


CHECKPOINTS = {
    "r50": (
        "deeplabv3_resnet50_coco-cd0a2569.pth",
        "cd0a25694c4a0f7106b38f4938bf90a874f2f241cc410b8f63c7024399538f06",
        deeplabv3_resnet50,
    ),
    "r101": (
        "deeplabv3_resnet101_coco-586e9e4e.pth",
        "586e9e4e203fcbf17e1ad45533d8d33ab133fc762bf03101c5dd743995c08c0d",
        deeplabv3_resnet101,
    ),
    "mv3": (
        "deeplabv3_mobilenet_v3_large-fc3c493d.pth",
        "fc3c493d68e89cc31ef488c803d5d7dd2f3190fb570598faa49fef69be8e5e70",
        deeplabv3_mobilenet_v3_large,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_parity(
    checkpoint_dir: Path,
    variants: list[str],
    device: str,
    input_size: int,
) -> None:
    if not torchvision.__version__.split("+", 1)[0].startswith("0.26.0"):
        raise RuntimeError(
            f"Parity is pinned to torchvision 0.26.0; got {torchvision.__version__}."
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA parity requested but no CUDA device is available")

    add_repo_root_to_path()
    from libreyolo.models.deeplabv3.convert import (
        convert_upstream_deeplabv3_state_dict,
    )
    from libreyolo.models.deeplabv3.nn import LibreDeepLabv3Net

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
        filename, expected_sha256, builder = CHECKPOINTS[size]
        checkpoint_path = checkpoint_dir / filename
        actual_sha256 = _sha256(checkpoint_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{filename}: expected SHA-256 {expected_sha256}, got {actual_sha256}"
            )
        upstream_state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        runtime_state = convert_upstream_deeplabv3_state_dict(upstream_state)
        if runtime_state is None:
            raise RuntimeError(f"{filename} was not recognized by the converter")

        reference = builder(
            weights=None,
            weights_backbone=None,
            num_classes=21,
            aux_loss=True,
        ).eval()
        port = LibreDeepLabv3Net(size=size, num_classes=21).eval()
        reference.load_state_dict(upstream_state, strict=True)
        port.load_state_dict(runtime_state, strict=True)
        reference.to(device)
        port.to(device)

        with torch.inference_mode():
            expected = reference(input_tensor)["out"]
            actual = port(input_tensor)

        difference = (expected - actual).abs()
        max_abs_diff = float(difference.max().item())
        nonzero = int(torch.count_nonzero(difference).item())
        print(
            f"{size:4s} shape={tuple(actual.shape)} "
            f"max_abs_diff={max_abs_diff:.1f} nonzero={nonzero} "
            f"sha256={actual_sha256}"
        )
        if max_abs_diff != 0.0 or nonzero != 0:
            raise AssertionError(
                f"DeepLabv3 {size} parity failed: "
                f"max_abs_diff={max_abs_diff}, nonzero={nonzero}"
            )

        del actual, expected, port, reference, runtime_state, upstream_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--input-size", type=int, default=520)
    args = parser.parse_args()
    run_parity(
        args.checkpoint_dir.resolve(),
        args.variants,
        args.device,
        args.input_size,
    )


if __name__ == "__main__":
    main()
