"""Build or publish RetinaNet's five-file Hugging Face weight repositories.

The official checkpoints have no per-object license file. The maintainer
approved rehosting under BSD-3-Clause implied by the releasing torchvision
project, with the pretrained-model caveat disclosed on every card. A real
upload requires ``--confirm-implied-license``; ``--dry-run`` changes no remote
state.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_UPSTREAM_COMMIT = "336d36e8db990a905498c73933e35231876e28bc"
_COLLECTION = "LibreYOLO/libreyolo-models-698875bf2b5f695708415169"
_GITATTRIBUTES = "*.pt filter=lfs diff=lfs merge=lfs -text\n"

_VARIANTS = {
    "r50": {
        "description": "ResNet-50 FPN v1 with FrozenBatchNorm",
        "upstream_file": "retinanet_resnet50_fpn_coco-eeacb38b.pth",
        "sha256": (
            "eeacb38b7cec8cf93c57867e05eaab621047f19b0d2ec5accaa405f690da15b7"
        ),
        "bytes": 136_595_076,
        "box_map": "36.4",
        "parameters": "34,014,999",
    },
    "r50v2": {
        "description": "ResNet-50 FPN v2 with GroupNorm heads",
        "upstream_file": "retinanet_resnet50_fpn_v2_coco-5905b1c5.pth",
        "sha256": (
            "5905b1c544219215e544dbe319720397bc4e68de61a733a59350d7976645b769"
        ),
        "bytes": 153_130_989,
        "box_map": "41.5",
        "parameters": "38,198,935",
    },
}


def _canonical_name(size: str) -> str:
    return f"LibreRetinaNet{size}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bsd_license() -> str:
    notice = (
        _REPO_ROOT / "libreyolo" / "models" / "retinanet" / "NOTICE"
    ).read_text(encoding="utf-8")
    marker = "BSD 3-Clause License"
    if marker not in notice:
        raise RuntimeError("RetinaNet NOTICE does not contain the BSD license text")
    return notice[notice.index(marker) :]


def _readme(size: str, converted_sha256: str, converted_bytes: int) -> str:
    variant = _VARIANTS[size]
    name = _canonical_name(size)
    source_url = f"https://download.pytorch.org/models/{variant['upstream_file']}"
    return f"""---
license: bsd-3-clause
library_name: libreyolo
pipeline_tag: object-detection
datasets:
  - detection-datasets/coco
tags:
  - object-detection
  - retinanet
  - torchvision
  - libreyolo
---

# {name}

RetinaNet ({variant['description']}), repackaged for LibreYOLO. This is an
inference-only model with {variant['parameters']} parameters.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{name}.pt")
results = model.predict("image.jpg")
```

## Source

Derived from [pytorch/vision](https://github.com/pytorch/vision) at commit
[`{_UPSTREAM_COMMIT}`](https://github.com/pytorch/vision/commit/{_UPSTREAM_COMMIT}).
Copyright (c) Soumith Chintala 2016 and torchvision contributors. The source
implementation is BSD-3-Clause.

Official checkpoint: [{variant['upstream_file']}]({source_url})

- Official file bytes: {variant['bytes']}
- Official SHA-256: `{variant['sha256']}`
- Converted file bytes: {converted_bytes}
- Converted SHA-256: `{converted_sha256}`
- Published COCO val2017 box mAP: {variant['box_map']}

## Model contract

- Input: RGB image, normalized with ImageNet mean/std.
- Resize: short side 800, long side capped at 1333, then bottom/right padding
  to a multiple of 32.
- Output: contiguous COCO-80 boxes, scores, and class ids after per-level
  candidate selection and class-aware NMS.
- Training: not implemented in LibreYOLO; `train()` raises.
- Export: dynamic-spatial, batch-one ONNX is validated.

## Modifications

Checkpoint metadata was added for LibreYOLO's v1.0 schema. Learned tensors and
state-dict keys are unchanged. The native LibreYOLO graph strictly loads the
official state dict and has exact eager parity at every FPN feature, raw head,
and final detection. See `weights/convert_retinanet_weights.py` in the
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


def _notice(size: str, converted_sha256: str) -> str:
    name = _canonical_name(size)
    variant = _VARIANTS[size]
    return f"""{name} weights
{'-' * (len(name) + 8)}

This product contains weights released by torchvision
(https://github.com/pytorch/vision) at commit {_UPSTREAM_COMMIT}.
Official checkpoint: {variant['upstream_file']}
Official SHA-256: {variant['sha256']}
Converted SHA-256: {converted_sha256}
Copyright (c) Soumith Chintala 2016 and torchvision contributors.

The checkpoint has no separate per-object license file. Redistribution uses
the releasing project's BSD-3-Clause license on an explicitly disclosed
implied basis; this is not a publisher-confirmed checkpoint-specific grant.
Torchvision warns that pretrained-model terms may derive from training data
and users must determine permission for their use case.

Conversion only adds LibreYOLO metadata. Learned tensors are unchanged.
"""


def _validate_checkpoint(size: str, path: Path) -> None:
    from libreyolo import LibreYOLO
    from libreyolo.models.retinanet.model import LibreRetinaNet
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    filename = f"{_canonical_name(size)}.pt"
    if path.name != filename:
        raise ValueError(f"Expected canonical filename {filename}, got {path.name}")
    whitelist = (
        _REPO_ROOT / "skills" / "libreyolo-upload-hf-model" / "SKILL.md"
    ).read_text(encoding="utf-8")
    if filename not in whitelist:
        raise ValueError(f"Canonical filename is absent from whitelist: {filename}")

    expected_url = (
        f"https://huggingface.co/LibreYOLO/{_canonical_name(size)}"
        f"/resolve/main/{filename}"
    )
    actual_url = LibreRetinaNet.get_download_url(filename)
    if actual_url != expected_url:
        raise ValueError(
            f"Loader URL mismatch for {filename}: {actual_url!r} != {expected_url!r}"
        )

    checkpoint = load_untrusted_torch_file(path, context="converted checkpoint")
    errors = validate_checkpoint_metadata(checkpoint, strict=False)
    if errors:
        raise ValueError(f"Invalid checkpoint metadata: {'; '.join(errors)}")
    expected = {
        "model_family": "retinanet",
        "size": size,
        "task": "detect",
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
    expected_loaded = ("retinanet", size, 80, "detect")
    if loaded != expected_loaded:
        raise ValueError(
            f"Factory load mismatch: expected {expected_loaded}, got {loaded}"
        )


def build_repo_dir(size: str, pt_path: Path, out_dir: Path) -> Path:
    """Build and validate exactly five files for one weight repository."""
    if not pt_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {pt_path}")
    _validate_checkpoint(size, pt_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    converted_sha256 = _sha256(pt_path)
    converted_bytes = pt_path.stat().st_size
    (out_dir / ".gitattributes").write_text(
        _GITATTRIBUTES, encoding="utf-8", newline="\n"
    )
    (out_dir / "README.md").write_text(
        _readme(size, converted_sha256, converted_bytes),
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "LICENSE").write_text(
        _bsd_license(), encoding="utf-8", newline="\n"
    )
    (out_dir / "NOTICE").write_text(
        _notice(size, converted_sha256), encoding="utf-8", newline="\n"
    )
    shutil.copy2(pt_path, out_dir / f"{_canonical_name(size)}.pt")

    expected = {
        ".gitattributes",
        "README.md",
        "LICENSE",
        "NOTICE",
        f"{_canonical_name(size)}.pt",
    }
    actual = {path.name for path in out_dir.iterdir()}
    if actual != expected:
        raise RuntimeError(
            f"Five-file contract mismatch: expected {sorted(expected)}, "
            f"got {sorted(actual)}"
        )
    return out_dir


def _upload(size: str, repo_dir: Path) -> str:
    from huggingface_hub import HfApi

    repo_name = _canonical_name(size)
    repo_id = f"LibreYOLO/{repo_name}"
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
            f"Initial upload: {repo_name} (RetinaNet, BSD-3-Clause implied)"
        ),
    )
    api.add_collection_item(
        collection_slug=_COLLECTION,
        item_id=repo_id,
        item_type="model",
        exists_ok=True,
    )
    expected = {
        ".gitattributes",
        "README.md",
        "LICENSE",
        "NOTICE",
        f"{repo_name}.pt",
    }
    actual = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    if actual != expected:
        raise RuntimeError(
            f"Published five-file contract mismatch: expected {sorted(expected)}, "
            f"got {sorted(actual)}"
        )
    if api.model_info(repo_id=repo_id).private:
        raise RuntimeError(f"Published repository is unexpectedly private: {repo_id}")
    return f"https://huggingface.co/{repo_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, choices=sorted(_VARIANTS))
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

    repo_dir = build_repo_dir(args.size, args.pt, args.out)
    print(f"Built five-file repository: {repo_dir}")
    if args.dry_run:
        print("Dry run complete; no external state changed.")
        return 0

    print(f"Uploaded: {_upload(args.size, repo_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
