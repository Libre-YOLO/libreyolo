"""Build or publish CenterNet's five-file Hugging Face weight repositories.

The official checkpoint objects have no standalone license files. The
maintainer approved rehosting under MIT implied by the releasing CenterNet
project, with that limitation disclosed on every card. A real upload requires
``--confirm-implied-license``; ``--dry-run`` only builds and validates files.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_UPSTREAM_COMMIT = "4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c"
_COLLECTION = "LibreYOLO/libreyolo-models-698875bf2b5f695708415169"

_VARIANTS = {
    "resdcn18": {
        "description": "ResNet-18 with deformable upsampling",
        "object_id": "1RtFps3kQAyLjQyzCao7pPDclOBQ64Vyp",
        "upstream_file": "ctdet_coco_resdcn18.pth",
        "upstream_sha256": "f9e413f91cdb235adbcb41c5c4052b8f7ff53999374048949789c29d6df18eaa",
        "canonical_sha256": "490a6c98c08510194f89416bde0d684e10f46d679f859b2e1a9e8117c9dc0095",
        "box_ap": "28.1",
    },
    "dla34": {
        "description": "DLA-34 with deformable aggregation",
        "object_id": "18Q3fzzAsha_3Qid6mn4jcIFPeOGUaj1d",
        "upstream_file": "ctdet_coco_dla_2x.pth",
        "upstream_sha256": "43bf4cc2efe00e02c1ae8484035b062a35543872d276c7dcfeb4db3e64203e4f",
        "canonical_sha256": "0818769746f56bffbed9d22be8f1a5896465cd44826ac5c69333a1121205b6e9",
        "box_ap": "37.4",
    },
}

_GITATTRIBUTES = "*.pt filter=lfs diff=lfs merge=lfs -text\n"

_MIT_LICENSE = """MIT License

Copyright (c) 2019 Xingyi Zhou
All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


def _canonical_name(size: str) -> str:
    return f"LibreCenterNet{size}"


def _readme(size: str) -> str:
    variant = _VARIANTS[size]
    name = _canonical_name(size)
    source_url = f"https://drive.google.com/file/d/{variant['object_id']}/view"
    benchmark_slug = f"centernet-{size}"
    return f"""---
license: mit
library_name: libreyolo
pipeline_tag: object-detection
datasets:
  - detection-datasets/coco
tags:
  - object-detection
  - centernet
  - libreyolo
---

# {name}

CenterNet {variant["description"]} COCO detector, repackaged for LibreYOLO.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{name}.pt")
results = model.predict("image.jpg")
```

## Source

Derived from [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)
at commit
[`{_UPSTREAM_COMMIT}`](https://github.com/xingyizhou/CenterNet/commit/{_UPSTREAM_COMMIT}).
Copyright (c) 2019 Xingyi Zhou. The source implementation is MIT licensed.

- Official checkpoint: [{variant["upstream_file"]}]({source_url})
- Official SHA-256: `{variant["upstream_sha256"]}`
- Published COCO test-dev AP without test-time augmentation: {variant["box_ap"]}.

## Modifications

The data-parallel `module.` prefix was removed and LibreYOLO v1 checkpoint
metadata was added. Learned tensors are unchanged. The native graph strictly
loads the official state dict and its `hm`, `wh`, and `reg` outputs are
bit-exact against the pinned implementation. LibreYOLO replaces the legacy
DCNv2 extension with torchvision deformable convolution.

See `weights/convert_centernet_weights.py` and
`docs/provenance/centernet.md` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## Benchmarks

Independent accuracy and speed benchmarks:
[visionanalysis.org/model/{benchmark_slug}](https://www.visionanalysis.org/model/{benchmark_slug})

## License

The checkpoint publisher did not attach a standalone per-object license file.
This mirror applies the releasing project's MIT license on an **implied**, not
publisher-confirmed, basis. COCO annotations are CC BY 4.0; source images
retain their individual Flickr terms. See [`LICENSE`](./LICENSE) and
[`NOTICE`](./NOTICE).
"""


def _notice(size: str) -> str:
    variant = _VARIANTS[size]
    name = _canonical_name(size)
    return f"""{name} weights
{"-" * (len(name) + 8)}

This product contains weights released by xingyizhou/CenterNet
(https://github.com/xingyizhou/CenterNet) at commit {_UPSTREAM_COMMIT}.
Official checkpoint: {variant["upstream_file"]}
Official Google Drive object: {variant["object_id"]}
Official SHA-256: {variant["upstream_sha256"]}
Copyright (c) 2019 Xingyi Zhou.

The checkpoint has no standalone per-object license file. Redistribution uses
the releasing project's MIT license on an explicitly disclosed implied basis;
this is not a publisher-confirmed checkpoint-specific grant. COCO annotations
are CC BY 4.0 and source images retain their individual Flickr terms.

Conversion removes the data-parallel module. key prefix and adds LibreYOLO
checkpoint metadata. Learned tensors are unchanged.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint(size: str, path: Path) -> None:
    from libreyolo import LibreYOLO
    from libreyolo.models.centernet.model import LibreCenterNet
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    filename = f"{_canonical_name(size)}.pt"
    if path.name != filename:
        raise ValueError(f"Expected canonical filename {filename}, got {path.name}")
    actual_sha = _sha256(path)
    expected_sha = _VARIANTS[size]["canonical_sha256"]
    if actual_sha != expected_sha:
        raise ValueError(
            f"Canonical SHA-256 mismatch for {filename}: {actual_sha} != {expected_sha}"
        )

    whitelist = (
        _REPO_ROOT / "skills" / "libreyolo-upload-hf-model" / "SKILL.md"
    ).read_text(encoding="utf-8")
    if filename not in whitelist:
        raise ValueError(
            f"Canonical filename is absent from upload whitelist: {filename}"
        )

    expected_url = (
        f"https://huggingface.co/LibreYOLO/{_canonical_name(size)}"
        f"/resolve/main/{filename}"
    )
    actual_url = LibreCenterNet.get_download_url(filename)
    if actual_url != expected_url:
        raise ValueError(
            f"Loader URL mismatch for {filename}: {actual_url!r} != {expected_url!r}"
        )

    checkpoint = load_untrusted_torch_file(path, context="converted checkpoint")
    errors = validate_checkpoint_metadata(checkpoint, strict=False)
    if errors:
        raise ValueError(f"Invalid checkpoint metadata: {'; '.join(errors)}")
    expected = {
        "model_family": "centernet",
        "size": size,
        "task": "detect",
        "nc": 80,
        "imgsz": 512,
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
    expected_loaded = ("centernet", size, 80, "detect")
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
    (out_dir / ".gitattributes").write_text(
        _GITATTRIBUTES, encoding="utf-8", newline="\n"
    )
    (out_dir / "README.md").write_text(_readme(size), encoding="utf-8", newline="\n")
    (out_dir / "LICENSE").write_text(_MIT_LICENSE, encoding="utf-8", newline="\n")
    (out_dir / "NOTICE").write_text(_notice(size), encoding="utf-8", newline="\n")
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
        commit_message=(f"Initial upload: {repo_name} (CenterNet, MIT implied)"),
    )
    files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    expected = {path.name for path in repo_dir.iterdir()}
    if files != expected:
        raise RuntimeError(
            f"Published five-file contract mismatch: expected {sorted(expected)}, "
            f"got {sorted(files)}"
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
        help="Confirm the maintainer-approved implied MIT redistribution basis.",
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
