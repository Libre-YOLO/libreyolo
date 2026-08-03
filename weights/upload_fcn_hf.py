"""Build or publish FCN's five-file Hugging Face weight repositories.

The checkpoints have no per-object license file. The maintainer approved
rehosting under BSD-3-Clause implied by the releasing torchvision project,
with the upstream pretrained-model caveat disclosed on every card. A real
upload therefore requires ``--confirm-implied-license``; ``--dry-run`` only
builds and validates the local repository.

Examples::

    python weights/upload_fcn_hf.py --size r50 \
        --pt weights/LibreFCNr50.pt --out ./LibreFCNr50 --dry-run

    python weights/upload_fcn_hf.py --size r50 \
        --pt weights/LibreFCNr50.pt --out ./LibreFCNr50 \
        --confirm-implied-license

If a previous attempt created the repository but did not finish::

    python weights/upload_fcn_hf.py --size r50 \
        --pt weights/LibreFCNr50.pt --out ./LibreFCNr50-retry \
        --confirm-implied-license --resume-partial
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_UPSTREAM_COMMIT = "336d36e8db990a905498c73933e35231876e28bc"
_COLLECTION = "LibreYOLO/libreyolo-models-698875bf2b5f695708415169"

_VARIANTS = {
    "r50": {
        "description": "dilated ResNet-50 backbone",
        "upstream_file": "fcn_resnet50_coco-1167a1af.pth",
        "bytes": 141_567_418,
        "sha256": "1167a1affa42e1e62858f8d3fac12d109e0108327ffc91c5855a324b11683c36",
        "miou": "60.5",
        "pixel_acc": "91.4",
    },
    "r101": {
        "description": "dilated ResNet-101 backbone",
        "upstream_file": "fcn_resnet101_coco-7ecb50ca.pth",
        "bytes": 217_800_805,
        "sha256": "7ecb50ca17844860a70d5ed0c748d997cf8adb62932abaa0233430c68594d749",
        "miou": "63.7",
        "pixel_acc": "91.9",
    },
}

_GITATTRIBUTES = """*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*.tar.* filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.wasm filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zst filter=lfs diff=lfs merge=lfs -text
*tfevents* filter=lfs diff=lfs merge=lfs -text
"""

_BSD_LICENSE = """BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
""".replace("2016,\n", "2016, \n", 1)

_VOC_NAMES = (
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def _canonical_name(size: str) -> str:
    return f"LibreFCN{size}"


def _readme(size: str) -> str:
    variant = _VARIANTS[size]
    name = _canonical_name(size)
    source_url = f"https://download.pytorch.org/models/{variant['upstream_file']}"
    categories = ", ".join(f"`{category}`" for category in _VOC_NAMES)
    return f"""---
license: bsd-3-clause
library_name: libreyolo
pipeline_tag: image-segmentation
datasets:
  - detection-datasets/coco
tags:
  - semantic-segmentation
  - fcn
  - torchvision
  - libreyolo
---

# {name}

Torchvision FCN with a {variant["description"]}, repackaged for LibreYOLO.
The 2015 FCN work established end-to-end pixels-to-pixels prediction, but this
checkpoint uses torchvision's later ResNet graph. It is **not** the original
paper's VGG-based FCN-8s skip-fusion architecture.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{name}.pt")
result = model.predict("image.jpg")
mask = result.semantic_mask.data
```

## Source

Derived from [pytorch/vision](https://github.com/pytorch/vision) at commit
[`{_UPSTREAM_COMMIT}`](https://github.com/pytorch/vision/commit/{_UPSTREAM_COMMIT}).
Copyright (c) Soumith Chintala 2016 and the torchvision contributors. The
source implementation is BSD-3-Clause.

Official checkpoint: [{variant["upstream_file"]}]({source_url})
Official checkpoint bytes: {variant["bytes"]:,}
SHA-256: `{variant["sha256"]}`

Torchvision reports COCO-val2017-VOC-labels mIoU {variant["miou"]} and pixel
accuracy {variant["pixel_acc"]}.

## Categories

The 21 output channels are {categories}.

## Modifications

Checkpoint metadata was added for LibreYOLO's v1.0 schema. Learned tensors and
state-dict keys are unchanged, including the primary and auxiliary heads. The
native LibreYOLO graph strict-loads the official state dict and both dense-logit
outputs are bit-exact against torchvision. See
`weights/convert_fcn_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## License

The checkpoint publisher did not attach a separate per-object license file.
This mirror applies the releasing project's BSD-3-Clause license on an
**implied**, not publisher-confirmed, basis. Torchvision warns that pretrained
models may have their own licenses or terms derived from training data and
that users must determine whether they have permission for their use case.
See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
"""


def _notice(size: str) -> str:
    name = _canonical_name(size)
    variant = _VARIANTS[size]
    return f"""{name} weights
{"-" * (len(name) + 8)}

This product contains weights released by torchvision
(https://github.com/pytorch/vision) at commit {_UPSTREAM_COMMIT}.
Official checkpoint: {variant["upstream_file"]}
Copyright (c) Soumith Chintala 2016 and the torchvision contributors.

The checkpoint has no separate per-object license file. Redistribution uses
the releasing project's BSD-3-Clause license on an explicitly disclosed
implied basis; this is not a publisher-confirmed checkpoint-specific grant.
Torchvision warns that pretrained-model terms may derive from training data
and users must determine permission for their use case.

Conversion only adds LibreYOLO metadata. Learned tensors are unchanged.
"""


def _validate_checkpoint(size: str, path: Path) -> None:
    from libreyolo import LibreYOLO
    from libreyolo.models.fcn.model import VOC_NAMES, LibreFCN
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
        raise ValueError(
            f"Canonical filename is absent from upload whitelist: {filename}"
        )
    expected_url = (
        f"https://huggingface.co/LibreYOLO/{_canonical_name(size)}"
        f"/resolve/main/{filename}"
    )
    actual_url = LibreFCN.get_download_url(filename)
    if actual_url != expected_url:
        raise ValueError(
            f"Loader URL mismatch for {filename}: {actual_url!r} != {expected_url!r}"
        )

    checkpoint = load_untrusted_torch_file(path, context="converted checkpoint")
    errors = validate_checkpoint_metadata(checkpoint, strict=False)
    if errors:
        raise ValueError(f"Invalid checkpoint metadata: {'; '.join(errors)}")
    expected = {
        "model_family": "fcn",
        "size": size,
        "task": "semantic",
        "nc": len(VOC_NAMES),
        "imgsz": 520,
        "names": dict(enumerate(VOC_NAMES)),
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
    expected_loaded = ("fcn", size, len(VOC_NAMES), "semantic")
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
    (out_dir / "LICENSE").write_text(_BSD_LICENSE, encoding="utf-8", newline="\n")
    (out_dir / "NOTICE").write_text(_notice(size), encoding="utf-8", newline="\n")
    shutil.copy2(pt_path, out_dir / f"{_canonical_name(size)}.pt")

    expected = {
        ".gitattributes",
        "README.md",
        "LICENSE",
        "NOTICE",
        f"{_canonical_name(size)}.pt",
    }
    actual = {repo_path.name for repo_path in out_dir.iterdir()}
    if actual != expected:
        raise RuntimeError(
            f"Five-file contract mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    return out_dir


def _upload(
    size: str,
    repo_dir: Path,
    *,
    resume_partial: bool = False,
) -> str:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    repo_name = _canonical_name(size)
    repo_id = f"LibreYOLO/{repo_name}"
    api = HfApi()
    expected = {
        ".gitattributes",
        "README.md",
        "LICENSE",
        "NOTICE",
        f"{repo_name}.pt",
    }
    local = {path.name for path in repo_dir.iterdir()}
    if local != expected:
        raise RuntimeError(
            f"Five-file contract mismatch before upload: expected "
            f"{sorted(expected)}, got {sorted(local)}"
        )

    exists = api.repo_exists(repo_id=repo_id, repo_type="model")
    remote_files: set[str] = set()
    if exists:
        if not resume_partial:
            raise FileExistsError(
                f"Refusing to overwrite existing Hugging Face repository: {repo_id}. "
                "Pass --resume-partial only for an interrupted publication."
            )
        try:
            remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        except HfHubHTTPError as exc:
            # A newly created repository has no revision until its first file
            # commit and may therefore return 404 from list_repo_files.
            if exc.response is None or exc.response.status_code != 404:
                raise
        unexpected = remote_files - expected
        if unexpected:
            raise FileExistsError(
                f"Refusing to resume {repo_id}: unexpected remote files "
                f"{sorted(unexpected)}"
            )
    else:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=False)

    if remote_files != expected:
        action = "Resume" if exists else "Initial"
        api.upload_folder(
            folder_path=str(repo_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=(
                f"{action} upload: {repo_name} (FCN, BSD-3-Clause implied)"
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
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help=(
            "Resume an interrupted publication only when the existing remote "
            "contains no files outside the five-file contract."
        ),
    )
    args = parser.parse_args()

    if not args.dry_run and not args.confirm_implied_license:
        parser.error("a real upload requires --confirm-implied-license")

    repo_dir = build_repo_dir(args.size, args.pt, args.out)
    print(f"Built five-file repository: {repo_dir}")
    if args.dry_run:
        print("Dry run complete; no external state changed.")
        return 0

    print(
        f"Uploaded: {_upload(args.size, repo_dir, resume_partial=args.resume_partial)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
