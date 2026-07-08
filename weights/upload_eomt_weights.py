"""Upload LibreEoMT weights to the LibreYOLO HuggingFace organisation.

Creates each repo if it does not exist yet, then uploads the local .pt file.
All 7 EoMT checkpoints are covered (semantic, instance-segment, and panoptic).

Usage::

    python weights/upload_eomt_weights.py --dry-run      # verify local files, no upload
    HF_TOKEN=... python weights/upload_eomt_weights.py   # create repos + upload all
    HF_TOKEN=... python weights/upload_eomt_weights.py --weights weights/LibreEoMTb-seg.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Catalogue: (local filename, HF repo name, short description for the card)
# ---------------------------------------------------------------------------
ORG = "LibreYOLO"

WEIGHTS = [
    {
        "filename": "LibreEoMTl-sem.pt",
        "repo":     "LibreEoMTl-sem",
        "desc":     "EoMT-L — ADE20K 150-class semantic segmentation, 512 px",
    },
    {
        "filename": "LibreEoMTs-seg.pt",
        "repo":     "LibreEoMTs-seg",
        "desc":     "EoMT-S — COCO 80-class instance segmentation, 640 px",
    },
    {
        "filename": "LibreEoMTb-seg.pt",
        "repo":     "LibreEoMTb-seg",
        "desc":     "EoMT-B — COCO 80-class instance segmentation, 640 px",
    },
    {
        "filename": "LibreEoMTl-seg.pt",
        "repo":     "LibreEoMTl-seg",
        "desc":     "EoMT-L — COCO 80-class instance segmentation, 640 px",
    },
    {
        "filename": "LibreEoMTl-seg-1280.pt",
        "repo":     "LibreEoMTl-seg-1280",
        "desc":     "EoMT-L — COCO 80-class instance segmentation, 1280 px (high-res variant)",
    },
    {
        "filename": "LibreEoMTs-panoptic.pt",
        "repo":     "LibreEoMTs-panoptic",
        "desc":     "EoMT-S — COCO panoptic 133-class (80 things + 53 stuff), 640 px",
    },
    {
        "filename": "LibreEoMTb-panoptic.pt",
        "repo":     "LibreEoMTb-panoptic",
        "desc":     "EoMT-B — COCO panoptic 133-class (80 things + 53 stuff), 640 px",
    },
]

_REPO_CARD_TEMPLATE = """\
---
license: apache-2.0
library_name: libreyolo
---

# {repo}

{desc}.

Part of the [LibreYOLO](https://github.com/LibreYOLO/libreyolo) model hub.

## Usage

```python
from libreyolo import LibreYOLO
model = LibreYOLO("{filename}")
results = model.predict("image.jpg")
```

## Licence

Weights are released under **Apache 2.0**.
The EoMT architecture is from [tue-mps/eomt](https://github.com/tue-mps/eomt).
"""


def _weights_dir() -> Path:
    return Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Specific weight file(s) to upload (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify local files and metadata without uploading.",
    )
    parser.add_argument(
        "--skip-repo-create",
        action="store_true",
        help="Skip creating repos (use when repos already exist).",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch
    from huggingface_hub import HfApi, upload_file
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    token = os.environ.get("HF_TOKEN")
    if not args.dry_run and not token:
        print("HF_TOKEN env var is required (or pass --dry-run).", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    weights_dir = _weights_dir()

    # Filter to requested subset if --weights was given.
    catalogue = WEIGHTS
    if args.weights:
        requested = {Path(w).name for w in args.weights}
        catalogue = [e for e in catalogue if e["filename"] in requested]
        missing = requested - {e["filename"] for e in catalogue}
        if missing:
            print(f"ERROR: unknown filename(s): {missing}", file=sys.stderr)
            return 2

    failures: list[str] = []
    for entry in catalogue:
        filename = entry["filename"]
        repo = entry["repo"]
        repo_id = f"{ORG}/{repo}"
        local_path = weights_dir / filename

        print(f"\n[{repo}]")
        if not local_path.exists():
            print(f"  SKIP: {local_path} not found locally.")
            failures.append(repo)
            continue

        print(f"  loading {local_path} ...")
        try:
            ckpt = torch.load(str(local_path), map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"  ERROR: could not load: {exc}")
            failures.append(repo)
            continue

        errors = validate_checkpoint_metadata(ckpt, strict=False)
        if errors:
            print(f"  ERROR: metadata validation failed: {errors}")
            failures.append(repo)
            continue
        print(
            f"  metadata OK — task={ckpt.get('task')!r}, size={ckpt.get('size')!r}, "
            f"nc={ckpt.get('nc')}, imgsz={ckpt.get('imgsz')}"
        )

        if args.dry_run:
            print("  dry-run: skipping upload.")
            continue

        # Create repo (ignore if it already exists).
        if not args.skip_repo_create:
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    exist_ok=True,
                    private=False,
                )
                print(f"  repo {repo_id} ready.")
            except Exception as exc:
                print(f"  WARNING: could not create repo: {exc}")

        # Upload model card.
        card = _REPO_CARD_TEMPLATE.format(
            repo=repo,
            filename=filename,
            desc=entry["desc"],
        )
        try:
            api.upload_file(
                path_or_fileobj=card.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add model card",
            )
        except Exception as exc:
            print(f"  WARNING: could not upload README: {exc}")

        # Upload weight file.
        print(f"  uploading {filename} ({local_path.stat().st_size / 1e6:.0f} MB)...")
        try:
            upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message=f"Add {filename}",
            )
            print("  uploaded.")
        except Exception as exc:
            print(f"  ERROR: upload failed: {exc}")
            failures.append(repo)

    print()
    if failures:
        print(f"Completed with {len(failures)} failure(s): {failures}")
        return 1
    print(f"Done: {len(catalogue)} checkpoint(s) processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
