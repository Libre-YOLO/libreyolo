"""Patch published classification checkpoints to embed real ImageNet-1k names.

The classification checkpoints were originally wrapped with placeholder
``class_i`` names. The checkpoint metadata schema (``docs/checkpoint_schema.md``)
requires real per-class names, and prediction results expose
``names[probs.top1]`` as the predicted label, so the published weights must carry
the canonical ImageNet-1k label set.

This rewrites ONLY the ``names`` metadata field. The model ``state_dict`` and
every other metadata key are preserved byte-for-byte, so inference stays
bit-identical -- only the human-readable labels change.

Usage::

    python weights/patch_classify_names.py --dry-run        # download + verify, no upload
    HF_TOKEN=... python weights/patch_classify_names.py      # patch + upload every repo
    HF_TOKEN=... python weights/patch_classify_names.py --repos LibreResNet18-cls
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from _conversion_utils import add_repo_root_to_path, imagenet1k_names

ORG = "LibreYOLO"
REPOS = [
    "LibreMobileNetV4s-cls",
    "LibreMobileNetV4m-cls",
    "LibreMobileNetV4l-cls",
    "LibreConvNeXtt-cls",
    "LibreConvNeXts-cls",
    "LibreConvNeXtb-cls",
    "LibreEfficientNetV2b0-cls",
    "LibreEfficientNetV2b1-cls",
    "LibreEfficientNetV2b2-cls",
    "LibreEfficientNetV2b3-cls",
    "LibreResNet18-cls",
    "LibreResNet34-cls",
    "LibreResNet50-cls",
    "LibreResNet101-cls",
]
COMMIT_MESSAGE = "Embed ImageNet-1k class names in checkpoint metadata"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", nargs="+", default=REPOS,
                        help="Repo names under the org (default: all classify repos).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Download and verify the patch without uploading.")
    args = parser.parse_args()

    add_repo_root_to_path()
    import torch
    from huggingface_hub import hf_hub_download, upload_file

    from libreyolo.utils.serialization import validate_checkpoint_metadata

    names = imagenet1k_names()
    token = os.environ.get("HF_TOKEN")
    if not args.dry_run and not token:
        print("HF_TOKEN env var is required to upload (or pass --dry-run).", file=sys.stderr)
        return 2

    failures: list[str] = []
    for repo in args.repos:
        repo_id = f"{ORG}/{repo}"
        filename = f"{repo}.pt"
        print(f"[{repo}] downloading {filename} ...")
        try:
            local = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
            ckpt = torch.load(local, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  ERROR: download/load failed: {exc}")
            failures.append(repo)
            continue

        if not isinstance(ckpt, dict) or "names" not in ckpt:
            print("  SKIP: not a metadata-wrapped checkpoint.")
            failures.append(repo)
            continue
        nc = ckpt.get("nc")
        if nc != len(names):
            print(f"  SKIP: nc={nc!r} does not match ImageNet-1k ({len(names)}).")
            failures.append(repo)
            continue

        old = ckpt["names"]
        before = old.get(258) if isinstance(old, dict) else None
        ckpt["names"] = dict(names)
        errors = validate_checkpoint_metadata(ckpt, strict=False)
        if errors:
            print(f"  ERROR: checkpoint invalid after patch: {errors}")
            failures.append(repo)
            continue
        print(f"  names[258]: {before!r} -> {ckpt['names'][258]!r}")

        if args.dry_run:
            print("  dry-run: not uploading.")
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / filename
            torch.save(ckpt, out)
            upload_file(
                path_or_fileobj=str(out),
                path_in_repo=filename,
                repo_id=repo_id,
                token=token,
                commit_message=COMMIT_MESSAGE,
            )
        print("  uploaded.")

    if failures:
        print(f"\nCompleted with {len(failures)} failure(s): {failures}")
        return 1
    print(f"\nDone: {len(args.repos)} checkpoint(s) processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
