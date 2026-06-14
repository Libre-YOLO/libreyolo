"""Run LibreYOLO nightly e2e tests on Modal.

GitHub Actions uses this as a controller: the action runs on ubuntu-latest,
then this script clones the requested LibreYOLO ref inside Modal and runs the
GPU test target there.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import modal


APP_NAME = "libreyolo-nightly-dev"
CACHE_VOLUME = os.getenv("LIBREYOLO_MODAL_CACHE_VOLUME", "libreyolo-nightly-cache-v2")
GPU = os.getenv("LIBREYOLO_MODAL_GPU", "L4")
REPO_URL = "https://github.com/LibreYOLO/libreyolo.git"
WORKDIR = Path("/workspace/libreyolo")

WEIGHT_SUFFIXES = {".pt", ".pth", ".safetensors"}
GPU_USD_PER_S = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
}

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "ffmpeg",
        "git",
        "git-lfs",
        "libgl1",
        "libglib2.0-0",
        "make",
    )
    .pip_install("uv")
)


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def replace_with_symlink(link: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.is_symlink():
            if link.resolve() == target:
                return
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(target, target_is_directory=True)


def prepare_home_cache_links() -> None:
    cache_root = Path("/cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    home_cache = Path.home() / ".cache"
    home_cache.mkdir(parents=True, exist_ok=True)

    for name in ("libreyolo", "huggingface"):
        replace_with_symlink(home_cache / name, cache_root / name)

    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["LIBREYOLO_DATASETS_DIR"] = str(cache_root / "datasets")


def is_weight_file(path: Path) -> bool:
    return path.is_file() and path.suffix in WEIGHT_SUFFIXES


def link_cached_weights() -> None:
    """Expose cached weight files without replacing tracked weights/ helpers."""
    cache_weights = Path("/cache/weights")
    worktree_weights = WORKDIR / "weights"
    cache_weights.mkdir(parents=True, exist_ok=True)
    worktree_weights.mkdir(parents=True, exist_ok=True)

    for src in cache_weights.rglob("*"):
        if not is_weight_file(src):
            continue
        rel = src.relative_to(cache_weights)
        dest = worktree_weights / rel
        if dest.exists() or dest.is_symlink():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(src)


def sync_downloaded_weights_to_cache() -> None:
    cache_weights = Path("/cache/weights")
    worktree_weights = WORKDIR / "weights"
    if not worktree_weights.exists():
        return

    for src in worktree_weights.rglob("*"):
        if not is_weight_file(src) or src.is_symlink():
            continue
        rel = src.relative_to(worktree_weights)
        dest = cache_weights / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)


def prepare_worktree_cache_links() -> None:
    link_cached_weights()
    replace_with_symlink(WORKDIR / "downloads", Path("/cache/downloads"))


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(32).startswith(b"version https://git-lfs")


def prepare_marbles_dataset() -> None:
    """Hydrate RF1's marbles dataset; older tests only checked existence."""
    dataset_root = Path.home() / ".cache" / "libreyolo" / "marbles"
    if dataset_root.exists():
        if (dataset_root / "data.yaml").exists():
            sample = next(dataset_root.rglob("*.jpg"), None)
            if sample is not None and not is_lfs_pointer(sample):
                return
        shutil.rmtree(dataset_root)

    run(["git", "lfs", "install"])
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "https://huggingface.co/datasets/LibreYOLO/marbles",
            str(dataset_root),
        ]
    )


def checkout_ref(ref: str) -> None:
    if re.fullmatch(r"[0-9a-f]{7,40}", ref):
        run(["git", "init", str(WORKDIR)])
        run(["git", "remote", "add", "origin", REPO_URL], cwd=WORKDIR)
        run(["git", "fetch", "--depth", "1", "origin", ref], cwd=WORKDIR)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=WORKDIR)
        return

    run(["git", "clone", "--depth", "1", "--branch", ref, REPO_URL, str(WORKDIR)])


def run_test_target(target: str) -> None:
    if target == "test_nightly":
        has_target = subprocess.run(
            ["grep", "-q", "^test_nightly:", "Makefile"], cwd=WORKDIR, check=False
        ).returncode == 0
        if not has_target:
            print(
                "test_nightly target not present; running legacy pre-v1.0 e2e command",
                flush=True,
            )
            run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "pytest",
                    "tests/e2e",
                    "-v",
                    "-m",
                    "not export_backend",
                ],
                cwd=WORKDIR,
            )
            return

    run(["make", target], cwd=WORKDIR)


@app.function(
    image=image,
    gpu=GPU,
    timeout=3 * 60 * 60,
    volumes={"/cache": cache},
)
def nightly(ref: str, target: str = "test_nightly") -> dict[str, object]:
    started = time.monotonic()
    timings: dict[str, float] = {}
    status = "failed"
    error = None
    prepare_home_cache_links()

    try:
        if WORKDIR.exists():
            shutil.rmtree(WORKDIR)
        WORKDIR.parent.mkdir(parents=True, exist_ok=True)

        step = time.monotonic()
        checkout_ref(ref)
        timings["clone_s"] = time.monotonic() - step

        prepare_worktree_cache_links()

        step = time.monotonic()
        prepare_marbles_dataset()
        timings["marbles_s"] = time.monotonic() - step

        step = time.monotonic()
        run(["uv", "venv", "--python", "3.10"], cwd=WORKDIR)
        timings["venv_s"] = time.monotonic() - step

        step = time.monotonic()
        run(
            [
                "uv",
                "pip",
                "install",
                "--no-sources",
                "--torch-backend",
                "cu128",
                "--group",
                "dev",
                "-e",
                ".[rfdetr,onnx]",
            ],
            cwd=WORKDIR,
        )
        timings["install_s"] = time.monotonic() - step

        run(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-c",
                (
                    "import torch; "
                    "print('CUDA:', torch.cuda.is_available(), "
                    "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
                ),
            ],
            cwd=WORKDIR,
        )

        step = time.monotonic()
        run_test_target(target)
        timings[f"{target}_s"] = time.monotonic() - step
        status = "passed"
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        cache_error = None
        cache_status = "skipped"
        step = time.monotonic()
        try:
            if WORKDIR.exists():
                sync_downloaded_weights_to_cache()
                cache.commit()
                cache_status = "committed"
            else:
                cache_status = "workdir-missing"
        except Exception as exc:
            cache_error = repr(exc)
            cache_status = "failed"
        timings["cache_s"] = time.monotonic() - step

        total_s = time.monotonic() - started
        gpu_usd_per_s = float(
            os.getenv("LIBREYOLO_MODAL_GPU_USD_PER_S", GPU_USD_PER_S.get(GPU, 0.0))
        )
        result = {
            "ref": ref,
            "gpu": GPU,
            "target": target,
            "status": status,
            "total_s": total_s,
            "estimated_gpu_cost_usd": round(total_s * gpu_usd_per_s, 4),
            "gpu_cost_rate_usd_per_s": gpu_usd_per_s,
            "cost_note": "GPU runtime estimate only; exact total billing is authoritative in Modal.",
            "cache_status": cache_status,
            "timings": timings,
        }
        if error:
            result["error"] = error
        if cache_error:
            result["cache_error"] = cache_error
        print("MODAL_NIGHTLY_RESULT", json.dumps(result, sort_keys=True), flush=True)

    return result


@app.local_entrypoint()
def main(ref: str = "dev", target: str = "test_nightly"):
    print(nightly.remote(ref, target))
