"""Prove exact LibreYOLO parity with the pinned official MiDaS releases.

This development-time check imports, but does not vendor, the audited MIT MiDaS
checkout and its Apache-2.0 EfficientNet dependency. It refuses moving source
checkouts and checksum-verifies both official release checkpoints before any
model construction.

Example::

    python weights/parity_midas.py \
        --upstream-repo ../MiDaS \
        --gen-efficientnet-repo ../gen-efficientnet-pytorch \
        --checkpoint-dir ../midas-weights
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MIDAS_COMMIT = "454597711a62eabcbf7d1e89f3fb9f569051ac9b"
GEN_EFFICIENTNET_COMMIT = "771ce082b2ce6d033f55b3d47c1f77389ad3c180"
GEN_EFFICIENTNET_HUB_ID = "rwightman/gen-efficientnet-pytorch"

CHECKPOINTS = {
    "s": (
        "midas_v21_small_256.pt",
        "70d6b9c891758c67f974a6097fb0c608c7ee67fb81ac3e5588847d5596d56fca",
        256,
    ),
    "l": (
        "dpt_large_384.pt",
        "2f21e586477d90cb9624c7eef5df7891edca49a1c4795ee2cb631fd4daa6ca69",
        384,
    ),
}


def _verify_checkout(path: Path, expected: str, label: str) -> None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != expected:
        raise RuntimeError(f"{label} must be at {expected}; got {actual}.")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError(f"{label} checkout has local modifications; refusing it.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _pinned_official_imports(
    upstream_repo: Path,
    gen_efficientnet_repo: Path,
) -> Iterator[None]:
    """Make the official small model use only the pinned local Hub checkout."""
    original_hub_load = torch.hub.load

    def local_hub_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir != GEN_EFFICIENTNET_HUB_ID:
            raise RuntimeError(f"Unexpected Torch Hub dependency: {repo_or_dir!r}")
        kwargs["pretrained"] = False
        return original_hub_load(
            str(gen_efficientnet_repo),
            model,
            *args,
            source="local",
            **kwargs,
        )

    torch.hub.load = local_hub_load
    sys.path.insert(0, str(upstream_repo))
    try:
        yield
    finally:
        sys.path.pop(0)
        torch.hub.load = original_hub_load


def _build_official(
    size: str,
    upstream_repo: Path,
    gen_efficientnet_repo: Path,
) -> torch.nn.Module:
    with _pinned_official_imports(upstream_repo, gen_efficientnet_repo):
        if size == "s":
            from midas.midas_net_custom import MidasNet_small

            return MidasNet_small(
                path=None,
                features=64,
                backbone="efficientnet_lite3",
                non_negative=True,
                exportable=True,
                blocks={"expand": True},
            )

        from midas.dpt_depth import DPTDepthModel

        return DPTDepthModel(
            path=None,
            backbone="vitl16_384",
            non_negative=True,
        )


def _clear_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _assert_preprocess_parity(size: str) -> None:
    from libreyolo.models.midas.utils import preprocess_numpy

    from midas.transforms import Resize

    filename, _digest, input_size = CHECKPOINTS[size]
    del filename
    image_u8 = np.random.default_rng(20260803).integers(
        0, 256, size=(321, 517, 3), dtype=np.uint8
    )
    image = image_u8.astype(np.float32) / 255.0
    resize = Resize(
        input_size,
        input_size,
        resize_target=None,
        keep_aspect_ratio=True,
        ensure_multiple_of=32,
        resize_method="upper_bound" if size == "s" else "minimal",
        image_interpolation_method=cv2.INTER_CUBIC,
    )
    expected = resize({"image": image})["image"]
    actual, _ = preprocess_numpy(image_u8, input_size, size)
    actual = actual.transpose(1, 2, 0)
    difference = float(np.max(np.abs(expected - actual)))
    if difference != 0.0:
        raise AssertionError(
            f"size={size} preprocess max_abs_diff={difference}; expected exact"
        )
    print(f"size={size} preprocess_shape={actual.shape} max_abs_diff=0")


def _run_size(
    size: str,
    upstream_repo: Path,
    gen_efficientnet_repo: Path,
    checkpoint_dir: Path,
    device: torch.device,
) -> None:
    from libreyolo.models.midas.nn import build_midas_model

    filename, expected_digest, input_size = CHECKPOINTS[size]
    checkpoint = checkpoint_dir / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_digest = _sha256(checkpoint)
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"{filename} SHA-256 mismatch: expected {expected_digest}, "
            f"got {actual_digest}."
        )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    official = _build_official(size, upstream_repo, gen_efficientnet_repo)
    official.load_state_dict(state, strict=True)
    official.eval().to(device)

    generator = torch.Generator(device="cpu").manual_seed(911 + input_size)
    input_tensor = (
        torch.rand(1, 3, input_size, input_size, generator=generator)
        .mul_(2.0)
        .sub_(1.0)
    )
    with torch.inference_mode():
        expected = official(input_tensor.to(device)).cpu()
    del official
    _clear_device(device)

    native = build_midas_model(size)
    native.load_state_dict(state, strict=True)
    native.eval().to(device)
    with torch.inference_mode():
        actual = native.forward_normalized(input_tensor.to(device)).cpu()

    if actual.shape != expected.shape:
        raise AssertionError(
            f"size={size} output shape mismatch: {actual.shape} != {expected.shape}"
        )
    difference = float((expected - actual).abs().max())
    exact = torch.equal(expected, actual)
    print(
        f"size={size} output_shape={tuple(actual.shape)} "
        f"max_abs_diff={difference:.9g} tensor_equal={exact}"
    )
    if difference != 0.0 or not exact:
        raise AssertionError(
            f"size={size} max_abs_diff={difference}; expected bitwise parity"
        )
    del native
    _clear_device(device)
    del state, expected, actual
    gc.collect()


def run_parity(
    upstream_repo: Path,
    gen_efficientnet_repo: Path,
    checkpoint_dir: Path,
    *,
    sizes: list[str],
    device: str,
) -> None:
    _verify_checkout(upstream_repo, MIDAS_COMMIT, "MiDaS")
    _verify_checkout(
        gen_efficientnet_repo,
        GEN_EFFICIENTNET_COMMIT,
        "gen-efficientnet-pytorch",
    )
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA parity requested but CUDA is not available.")

    with _pinned_official_imports(upstream_repo, gen_efficientnet_repo):
        for size in sizes:
            _assert_preprocess_parity(size)
    for size in sizes:
        _run_size(
            size,
            upstream_repo,
            gen_efficientnet_repo,
            checkpoint_dir,
            torch_device,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--gen-efficientnet-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--sizes",
        choices=sorted(CHECKPOINTS),
        nargs="+",
        default=sorted(CHECKPOINTS),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    run_parity(
        args.upstream_repo.resolve(),
        args.gen_efficientnet_repo.resolve(),
        args.checkpoint_dir.resolve(),
        sizes=args.sizes,
        device=args.device,
    )


if __name__ == "__main__":
    main()
