"""Build or publish FCOS's five-file Hugging Face weight repository.

The official checkpoint has no per-object license file. The maintainer
approved rehosting under BSD-3-Clause implied by the releasing torchvision
project, with the upstream pretrained-model caveat disclosed on the card.
A real upload therefore requires ``--confirm-implied-license``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from weights.upload_faster_rcnn_hf import _BSD_LICENSE, _GITATTRIBUTES  # noqa: E402

_UPSTREAM_COMMIT = "336d36e8db990a905498c73933e35231876e28bc"
_UPSTREAM_FILE = "fcos_resnet50_fpn_coco-99b0c9b7.pth"
_UPSTREAM_SHA256 = "99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7"
_CONVERTED_SHA256 = "51d79292895816fc09e4a7b159331f8bbc1243e0cee96f991f8e5865f4920788"
_NAME = "LibreFCOSr50"
_FILENAME = f"{_NAME}.pt"
_COLLECTION = "LibreYOLO/libreyolo-models-698875bf2b5f695708415169"


def _readme() -> str:
    source_url = f"https://download.pytorch.org/models/{_UPSTREAM_FILE}"
    return f"""---
license: bsd-3-clause
library_name: libreyolo
pipeline_tag: object-detection
datasets:
  - detection-datasets/coco
tags:
  - object-detection
  - fcos
  - torchvision
  - libreyolo
---

# {_NAME}

FCOS with a ResNet-50 FPN backbone, repackaged for LibreYOLO.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{_FILENAME}")
results = model.predict("image.jpg")
```

## Source

Derived from [pytorch/vision](https://github.com/pytorch/vision) at commit
[`{_UPSTREAM_COMMIT}`](https://github.com/pytorch/vision/commit/{_UPSTREAM_COMMIT}).
Copyright (c) Soumith Chintala 2016 and torchvision contributors. The source
implementation is BSD-3-Clause.

Official checkpoint: [{_UPSTREAM_FILE}]({source_url})
SHA-256: `{_UPSTREAM_SHA256}`
Published COCO val2017 box mAP: 39.2.

## Modifications

Checkpoint metadata was added for LibreYOLO's v1.0 schema. Learned tensors and
state-dict keys are unchanged. The native LibreYOLO graph loads all 319
official state entries strictly and matches the pinned source at raw heads,
anchors, preprocessing, and final detections. See
`weights/convert_fcos_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## Benchmarks

Independent accuracy and speed benchmarks:
[visionanalysis.org/model/fcos-r50](https://www.visionanalysis.org/model/fcos-r50)

## License

The checkpoint publisher did not attach a separate per-object license file.
This mirror applies the releasing project's BSD-3-Clause license on an
**implied**, not publisher-confirmed, basis. Torchvision warns that pretrained
models may have their own licenses or terms derived from training data and
that users must determine whether they have permission for their use case.
COCO annotations are CC BY 4.0; source images retain their individual Flickr
terms. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
"""


def _notice() -> str:
    return f"""{_NAME} weights
----------------------

This product contains weights released by torchvision
(https://github.com/pytorch/vision) at commit {_UPSTREAM_COMMIT}.
Official checkpoint: {_UPSTREAM_FILE}
Copyright (c) Soumith Chintala 2016 and torchvision contributors.

The checkpoint has no separate per-object license file. Redistribution uses
the releasing project's BSD-3-Clause license on an explicitly disclosed
implied basis; this is not a publisher-confirmed checkpoint-specific grant.
Torchvision warns that pretrained-model terms may derive from training data
and users must determine permission for their use case.

Conversion only adds LibreYOLO metadata. Learned tensors are unchanged.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint(path: Path) -> None:
    from libreyolo import LibreYOLO
    from libreyolo.models.fcos.model import LibreFCOS
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    if path.name != _FILENAME:
        raise ValueError(f"Expected canonical filename {_FILENAME}, got {path.name}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != _CONVERTED_SHA256:
        raise ValueError(f"Converted checkpoint SHA-256 mismatch: {actual_sha256}")

    whitelist = (
        _REPO_ROOT / "skills" / "libreyolo-upload-hf-model" / "SKILL.md"
    ).read_text(encoding="utf-8")
    if _FILENAME not in whitelist:
        raise ValueError(
            f"Canonical filename is absent from upload whitelist: {_FILENAME}"
        )
    expected_url = f"https://huggingface.co/LibreYOLO/{_NAME}/resolve/main/{_FILENAME}"
    actual_url = LibreFCOS.get_download_url(_FILENAME)
    if actual_url != expected_url:
        raise ValueError(
            f"Loader URL mismatch for {_FILENAME}: {actual_url!r} != {expected_url!r}"
        )

    checkpoint = load_untrusted_torch_file(path, context="converted checkpoint")
    errors = validate_checkpoint_metadata(checkpoint, strict=True)
    if errors:
        raise ValueError(f"Invalid checkpoint metadata: {'; '.join(errors)}")
    expected = {
        "model_family": "fcos",
        "size": "r50",
        "task": "detect",
        "nc": 80,
        "imgsz": 800,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint metadata mismatch: {mismatches}")

    model = LibreYOLO(str(path), device="cpu")
    loaded = (model.family, model.size, model.nb_classes, model.task)
    expected_loaded = ("fcos", "r50", 80, "detect")
    if loaded != expected_loaded:
        raise ValueError(
            f"Factory load mismatch: expected {expected_loaded}, got {loaded}"
        )


def build_repo_dir(pt_path: Path, out_dir: Path) -> Path:
    """Build and validate exactly five files for the FCOS weight repository."""
    if not pt_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {pt_path}")
    _validate_checkpoint(pt_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / ".gitattributes").write_text(
        _GITATTRIBUTES, encoding="utf-8", newline="\n"
    )
    (out_dir / "README.md").write_text(_readme(), encoding="utf-8", newline="\n")
    (out_dir / "LICENSE").write_text(_BSD_LICENSE, encoding="utf-8", newline="\n")
    (out_dir / "NOTICE").write_text(_notice(), encoding="utf-8", newline="\n")
    shutil.copy2(pt_path, out_dir / _FILENAME)

    expected = {".gitattributes", "README.md", "LICENSE", "NOTICE", _FILENAME}
    actual = {path.name for path in out_dir.iterdir()}
    if actual != expected:
        raise RuntimeError(
            f"Five-file contract mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    return out_dir


def _upload(repo_dir: Path) -> str:
    from huggingface_hub import HfApi

    repo_id = f"LibreYOLO/{_NAME}"
    api = HfApi()
    if api.repo_exists(repo_id=repo_id, repo_type="model"):
        raise FileExistsError(
            f"Refusing to overwrite existing Hugging Face repository: {repo_id}"
        )
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=False)
    api.upload_folder(
        folder_path=str(repo_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial upload: LibreFCOSr50 (BSD-3-Clause implied)",
    )
    api.add_collection_item(
        collection_slug=_COLLECTION,
        item_id=repo_id,
        item_type="model",
        exists_ok=True,
    )
    return f"https://huggingface.co/{repo_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the five files without changing Hugging Face.",
    )
    parser.add_argument(
        "--confirm-implied-license",
        action="store_true",
        help="Confirm the maintainer-approved implied BSD redistribution basis.",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.confirm_implied_license:
        parser.error("a real upload requires --confirm-implied-license")

    repo_dir = build_repo_dir(args.pt, args.out)
    print(f"Built five-file repository: {repo_dir}")
    if args.dry_run:
        print("Dry run complete; no external state changed.")
        return 0

    print(f"Uploaded: {_upload(repo_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
