"""Build or publish Mask R-CNN's five-file Hugging Face weight repository.

The official checkpoint has no per-object license file. The maintainer approved
rehosting under BSD-3-Clause implied by the releasing torchvision project, with
the pretrained-model caveat disclosed on the card. A real upload therefore
requires ``--confirm-implied-license``; ``--dry-run`` only builds and validates
the local repository.
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

_CANONICAL_NAME = "LibreMaskRCNNr50"
_UPSTREAM_COMMIT = "336d36e8db990a905498c73933e35231876e28bc"
_UPSTREAM_FILE = "maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
_UPSTREAM_SHA256 = "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e"
_COLLECTION = "LibreYOLO/libreyolo-models-698875bf2b5f695708415169"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readme(converted_sha256: str) -> str:
    source_url = f"https://download.pytorch.org/models/{_UPSTREAM_FILE}"
    return f"""---
license: bsd-3-clause
library_name: libreyolo
pipeline_tag: image-segmentation
datasets:
  - detection-datasets/coco
tags:
  - instance-segmentation
  - object-detection
  - mask-rcnn
  - torchvision
  - libreyolo
---

# {_CANONICAL_NAME}

Mask R-CNN with a ResNet-50-FPN v2 backbone, repackaged for LibreYOLO. The
checkpoint supports instance segmentation by default and box-only detection
with `task="detect"`.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{_CANONICAL_NAME}.pt")
result = model.predict("image.jpg")
print(result.boxes.xyxy, result.masks.data)
```

## Source

Derived from [pytorch/vision](https://github.com/pytorch/vision) at commit
[`{_UPSTREAM_COMMIT}`](https://github.com/pytorch/vision/commit/{_UPSTREAM_COMMIT}).
Copyright (c) Soumith Chintala 2016 and torchvision contributors. The source
implementation is BSD-3-Clause.

Official checkpoint: [{_UPSTREAM_FILE}]({source_url})

- Official SHA-256: `{_UPSTREAM_SHA256}`
- Converted SHA-256: `{converted_sha256}`
- Published COCO val2017 box mAP: 47.4
- Published COCO val2017 mask mAP: 41.8

## Modifications

LibreYOLO checkpoint metadata was added. Learned tensors and state-dict keys
are unchanged. The native graph loads the official state dict strictly and has
exact eager parity at the RPN head, box head, final boxes, raw mask logits, and
full-image masks. The batch-1 opset-18 ONNX graph is also covered by ONNX
Runtime parity. See `weights/convert_mask_rcnn_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## License

The checkpoint publisher did not attach a separate per-object license file.
This mirror applies the releasing project's BSD-3-Clause license on an
**implied**, not publisher-confirmed, basis. Torchvision warns that pretrained
models may have their own licenses or terms derived from training data and
that users must determine whether they have permission for their use case.
COCO annotations are CC BY 4.0; source images retain their individual Flickr
terms. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
"""


def _notice(converted_sha256: str) -> str:
    return f"""{_CANONICAL_NAME} weights
{'-' * (len(_CANONICAL_NAME) + 8)}

This product contains weights released by torchvision
(https://github.com/pytorch/vision) at commit {_UPSTREAM_COMMIT}.
Official checkpoint: {_UPSTREAM_FILE}
Official SHA-256: {_UPSTREAM_SHA256}
Converted SHA-256: {converted_sha256}
Copyright (c) Soumith Chintala 2016 and torchvision contributors.

The checkpoint has no separate per-object license file. Redistribution uses
the releasing project's BSD-3-Clause license on an explicitly disclosed
implied basis; this is not a publisher-confirmed checkpoint-specific grant.
Torchvision warns that pretrained-model terms may derive from training data
and users must determine permission for their use case.

Conversion only adds LibreYOLO metadata. Learned tensors are unchanged.
"""


def _validate_checkpoint(path: Path) -> None:
    from libreyolo import LibreYOLO
    from libreyolo.models.mask_rcnn.model import LibreMaskRCNN
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    filename = f"{_CANONICAL_NAME}.pt"
    if path.name != filename:
        raise ValueError(f"Expected canonical filename {filename}, got {path.name}")

    whitelist = (
        _REPO_ROOT / "skills" / "libreyolo-upload-hf-model" / "SKILL.md"
    ).read_text(encoding="utf-8")
    if filename not in whitelist:
        raise ValueError(f"Canonical filename is absent from upload whitelist: {filename}")

    expected_url = (
        f"https://huggingface.co/LibreYOLO/{_CANONICAL_NAME}"
        f"/resolve/main/{filename}"
    )
    actual_url = LibreMaskRCNN.get_download_url(filename)
    if actual_url != expected_url:
        raise ValueError(
            f"Loader URL mismatch for {filename}: {actual_url!r} != {expected_url!r}"
        )

    checkpoint = load_untrusted_torch_file(path, context="converted checkpoint")
    errors = validate_checkpoint_metadata(checkpoint, strict=True)
    if errors:
        raise ValueError(f"Invalid checkpoint metadata: {'; '.join(errors)}")
    expected = {
        "model_family": "mask_rcnn",
        "size": "r50",
        "task": "segment",
        "nc": 80,
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
    expected_loaded = ("mask_rcnn", "r50", 80, "segment")
    if loaded != expected_loaded:
        raise ValueError(
            f"Factory load mismatch: expected {expected_loaded}, got {loaded}"
        )


def build_repo_dir(pt_path: Path, out_dir: Path) -> Path:
    """Build and validate exactly five files for the weight repository."""
    if not pt_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {pt_path}")
    _validate_checkpoint(pt_path)
    converted_sha256 = _sha256(pt_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / ".gitattributes").write_text(
        _GITATTRIBUTES, encoding="utf-8", newline="\n"
    )
    (out_dir / "README.md").write_text(
        _readme(converted_sha256), encoding="utf-8", newline="\n"
    )
    (out_dir / "LICENSE").write_text(
        _BSD_LICENSE, encoding="utf-8", newline="\n"
    )
    (out_dir / "NOTICE").write_text(
        _notice(converted_sha256), encoding="utf-8", newline="\n"
    )
    shutil.copy2(pt_path, out_dir / f"{_CANONICAL_NAME}.pt")

    expected = {
        ".gitattributes",
        "README.md",
        "LICENSE",
        "NOTICE",
        f"{_CANONICAL_NAME}.pt",
    }
    actual = {path.name for path in out_dir.iterdir()}
    if actual != expected:
        raise RuntimeError(
            f"Five-file contract mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    return out_dir


def _upload(repo_dir: Path) -> str:
    from huggingface_hub import HfApi

    repo_id = f"LibreYOLO/{_CANONICAL_NAME}"
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
        commit_message=(
            "Initial upload: LibreMaskRCNNr50 "
            "(Mask R-CNN, BSD-3-Clause implied)"
        ),
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
