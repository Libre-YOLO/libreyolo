"""Compare LibreYOLO MoGe-2 normals against the pinned Microsoft source.

This is a development-time provenance/parity check, not a weight converter.
It imports the official checkout supplied with ``--upstream-repo`` and refuses
to run it unless that checkout is at the audited commit.

Example::

    python weights/parity_moge2.py \
        --upstream-repo ../MoGe \
        --checkpoint model.pt \
        --image image.jpg
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

UPSTREAM_COMMIT = "925b8ed835a7a9cdb7578ba15c658a0afc969030"


def _verify_upstream_checkout(path: Path) -> None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(
            "MoGe parity must use the audited upstream commit "
            f"{UPSTREAM_COMMIT}; got {actual}."
        )


def _install_utils3d_import_stub() -> None:
    """Allow importing the normal forward without optional geometry helpers."""
    if "utils3d" in sys.modules:
        return
    module = types.ModuleType("utils3d")
    module.np = types.SimpleNamespace()
    module.pt = types.SimpleNamespace()
    sys.modules["utils3d"] = module


def _angular_error(reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    reference = torch.nn.functional.normalize(reference.double(), dim=-1)
    actual = torch.nn.functional.normalize(actual.double(), dim=-1)
    cosine = (reference * actual).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def run_parity(
    upstream_repo: Path,
    checkpoint: Path,
    image_path: Path,
    *,
    device: str,
    input_size: int,
    size: str | None,
    fp16: bool,
    threshold: float,
) -> None:
    _verify_upstream_checkout(upstream_repo)
    _install_utils3d_import_stub()
    sys.path.insert(0, str(upstream_repo))

    from libreyolo.models.moge2.model import LibreMoGe2
    from libreyolo.models.moge2.utils import preprocess_numpy
    from moge.model.v2 import MoGeModel

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    chw, _ = preprocess_numpy(rgb, input_size)
    input_tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
    token_count = (input_tensor.shape[-2] // 14) * (input_tensor.shape[-1] // 14)

    if size is None:
        checkpoint_config = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )["model_config"]
        backbone = checkpoint_config["encoder"]["backbone"]
        size = {
            "dinov2_vits14": "s",
            "dinov2_vitb14": "b",
            "dinov2_vitl14": "l",
        }.get(backbone)
        if size is None:
            raise ValueError(f"Unsupported MoGe-2 checkpoint backbone: {backbone!r}")

    official = MoGeModel.from_pretrained(checkpoint).eval().to(device)
    native = LibreMoGe2(checkpoint, size=size, device=device).model.eval()
    autocast = (
        torch.autocast("cuda", dtype=torch.float16)
        if fp16 and str(device).startswith("cuda")
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        expected = official(input_tensor, num_tokens=token_count)["normal"]
        actual = native(input_tensor).permute(0, 2, 3, 1)

    errors = _angular_error(expected.float().cpu(), actual.float().cpu())
    component_error = (expected.float().cpu() - actual.float().cpu()).abs()
    metrics = {
        "mean_deg": float(errors.mean()),
        "median_deg": float(errors.median()),
        "p99_deg": float(torch.quantile(errors, 0.99)),
        "max_deg": float(errors.max()),
        "mean_abs_component": float(component_error.mean()),
        "max_abs_component": float(component_error.max()),
    }
    print(
        "MoGe-2 parity "
        f"size={size} shape={tuple(actual.shape)} tokens={token_count} device={device} "
        f"fp16={fp16}"
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.9g}")
    if metrics["mean_deg"] >= threshold:
        raise SystemExit(
            f"Parity failed: mean angular error {metrics['mean_deg']:.6g} "
            f">= {threshold:.6g} degrees."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--size", choices=["s", "b", "l"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()
    run_parity(
        args.upstream_repo.resolve(),
        args.checkpoint.resolve(),
        args.image.resolve(),
        device=args.device,
        input_size=args.input_size,
        size=args.size,
        fp16=args.fp16,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
