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
CACHE_ROOT = Path("/cache")
CACHE_WEIGHTS = CACHE_ROOT / "weights"

WEIGHT_SUFFIXES = {".pt", ".pth", ".safetensors"}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
# Written by the model loaders once a Hugging Face snapshot is verified complete
# (see libreyolo/models/openvocab/base.py). Its absence means the loader will
# download the snapshot again.
SNAPSHOT_MARKER = ".libreyolo_snapshot_complete"
# Requests-based hub downloads hang forever by default; xet transfers may not
# honour this, which is why the e2e per-test timeout is the real backstop.
HF_DOWNLOAD_TIMEOUT_S = os.getenv("LIBREYOLO_HF_DOWNLOAD_TIMEOUT", "60")
GPU_USD_PER_S = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
}

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)


def hf_secrets() -> list[modal.Secret]:
    """Authenticate to the Hugging Face Hub when the workflow supplies a token.

    Anonymous pulls from a shared datacenter egress are the ones the Hub
    throttles. A missing token keeps the previous anonymous behaviour instead of
    failing the run, so this stays optional.
    """
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        return []
    return [modal.Secret.from_dict({"HF_TOKEN": token})]


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
    cache_root = CACHE_ROOT
    cache_root.mkdir(parents=True, exist_ok=True)
    home_cache = Path.home() / ".cache"
    home_cache.mkdir(parents=True, exist_ok=True)

    for name in ("libreyolo", "huggingface"):
        replace_with_symlink(home_cache / name, cache_root / name)

    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["LIBREYOLO_DATASETS_DIR"] = str(cache_root / "datasets")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", HF_DOWNLOAD_TIMEOUT_S)


def is_weight_file(path: Path) -> bool:
    return path.is_file() and path.suffix in WEIGHT_SUFFIXES


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def snapshot_dirs(root: Path) -> list[Path]:
    """Snapshot directories under ``root`` that the loaders consider complete."""
    if not root.exists():
        return []
    return sorted(marker.parent for marker in root.glob(f"*/{SNAPSHOT_MARKER}"))


def link_cached_snapshots() -> None:
    """Restore Hugging Face snapshots (open-vocab, SAM, VLM) as whole directories.

    A loader only skips the download when the marker file and the config sit
    beside the weights, and neither is a weight file, so file-level caching left
    every snapshot family re-downloading its full snapshot on every run.
    """
    worktree_weights = WORKDIR / "weights"
    worktree_weights.mkdir(parents=True, exist_ok=True)

    for src in snapshot_dirs(CACHE_WEIGHTS):
        dest = worktree_weights / src.name
        if dest.exists() or dest.is_symlink():
            continue
        dest.symlink_to(src, target_is_directory=True)


def link_cached_weights() -> None:
    """Expose cached weight files without replacing tracked weights/ helpers."""
    cache_weights = CACHE_WEIGHTS
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


def sync_snapshots_to_cache() -> None:
    """Persist snapshots that carry the completion marker.

    The marker is only written after the loader verifies the snapshot, so these
    are safe to keep even when the suite fails. Publishing through a temporary
    directory keeps a half-copied snapshot from ever being seen as complete.
    """
    cache_weights = CACHE_WEIGHTS
    worktree_weights = WORKDIR / "weights"
    cache_weights.mkdir(parents=True, exist_ok=True)

    for snapshot in snapshot_dirs(worktree_weights):
        if snapshot.is_symlink():
            continue
        dest = cache_weights / snapshot.name
        if dest.exists():
            continue
        staging = cache_weights / f".{snapshot.name}.partial"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(snapshot, staging)
        staging.rename(dest)


def sync_downloaded_weights_to_cache() -> None:
    cache_weights = CACHE_WEIGHTS
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
    link_cached_snapshots()
    link_cached_weights()
    replace_with_symlink(WORKDIR / "downloads", CACHE_ROOT / "downloads")


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(32).startswith(b"version https://git-lfs")


def prepare_marbles_dataset() -> None:
    """Hydrate RF1's marbles dataset; older tests only checked existence."""
    dataset_root = Path.home() / ".cache" / "libreyolo" / "marbles"
    if dataset_root.exists():
        if (dataset_root / "data.yaml").exists():
            images = [path for path in dataset_root.rglob("*") if is_image_file(path)]
            if images and not any(is_lfs_pointer(image) for image in images):
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
    secrets=hf_secrets(),
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
            if not WORKDIR.exists():
                cache_status = "workdir-missing"
            elif status == "passed":
                sync_snapshots_to_cache()
                sync_downloaded_weights_to_cache()
                cache.commit()
                cache_status = "committed"
            else:
                # Keep verified snapshots even when the suite fails, so a run
                # that dies mid-suite does not leave the next one downloading
                # them cold again. Loose weight files stay out: only snapshots
                # carry a completeness marker, so only they are provably whole.
                sync_snapshots_to_cache()
                cache.commit()
                cache_status = "committed-snapshots-failed-run"
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
