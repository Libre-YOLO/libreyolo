"""Build or publish AlexNet's five-file Hugging Face weight repository.

The official torchvision checkpoint has no per-object license file. The
maintainer approved rehosting under BSD-3-Clause implied by the releasing
project, with torchvision's pretrained-model and ImageNet caveat disclosed.
A real upload therefore requires ``--confirm-implied-license``.

Examples::

    python weights/upload_alexnet_hf.py --pt LibreAlexNetb-cls.pt \
        --out ./LibreAlexNetb-cls --dry-run

    python weights/upload_alexnet_hf.py --pt LibreAlexNetb-cls.pt \
        --out ./LibreAlexNetb-cls --confirm-implied-license
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_REPO_NAME = "LibreAlexNetb-cls"
_FILENAME = f"{_REPO_NAME}.pt"
_UPSTREAM_COMMIT = "336d36e8db990a905498c73933e35231876e28bc"
_UPSTREAM_FILE = "alexnet-owt-7be5be79.pth"
_UPSTREAM_URL = f"https://download.pytorch.org/models/{_UPSTREAM_FILE}"
_UPSTREAM_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
_UPSTREAM_BYTES = 244_408_911
_CONVERTED_SHA256 = "95f6996b7b4c5526e7e47ad99cf78b2a3643baa3ba1d4107ab840a05e73d1f5e"
_CONVERTED_BYTES = 244_431_825
_COLLECTION = "LibreYOLO/libreyolo-classification-6a4164414d64a10aa8576885"

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


def _readme() -> str:
    return f"""---
license: bsd-3-clause
library_name: libreyolo
pipeline_tag: image-classification
datasets:
  - imagenet-1k
tags:
  - image-classification
  - alexnet
  - torchvision
  - imagenet
  - libreyolo
---

# {_REPO_NAME}

AlexNet image classification weights repackaged for LibreYOLO. This is
torchvision's single-tower, 64-channel-stem variant: no local response
normalization and no grouped convolutions. It is not the two-GPU 2012 graph.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{_FILENAME}")
result = model.predict("image.jpg")
print(result.probs.top1, result.probs.top5)
```

## Source

Derived from [pytorch/vision](https://github.com/pytorch/vision) at commit
[`{_UPSTREAM_COMMIT}`](https://github.com/pytorch/vision/commit/{_UPSTREAM_COMMIT}).
Copyright (c) Soumith Chintala 2016 and torchvision contributors. The source
implementation is BSD-3-Clause.

Official checkpoint: [{_UPSTREAM_FILE}]({_UPSTREAM_URL})
Official checkpoint SHA-256: `{_UPSTREAM_SHA256}`
Official checkpoint bytes: `{_UPSTREAM_BYTES}`
Published ImageNet-1K accuracy: 56.522% top-1, 79.066% top-5.

## Modifications

LibreYOLO metadata and the canonical filename were added. Learned tensors and
state-dict keys are unchanged. The native graph strict-loads the official state
dict and produces bit-identical logits (`max_abs_diff == 0.0`).

Converted checkpoint SHA-256: `{_CONVERTED_SHA256}`
Converted checkpoint bytes: `{_CONVERTED_BYTES}`.

The converter and parity tests are in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## License

The checkpoint publisher did not attach a separate per-object license file.
This mirror applies the releasing project's BSD-3-Clause license on an
**implied**, not publisher-confirmed, basis. Torchvision warns that pretrained
models may have their own licenses or terms derived from training data and
that users must determine whether they have permission for their use case.
The weights were trained on ImageNet-1K; ImageNet's dataset and image-source
terms remain the downstream user's responsibility. See [`LICENSE`](./LICENSE)
and [`NOTICE`](./NOTICE).
"""


def _notice() -> str:
    return f"""{_REPO_NAME} weights
{"-" * (len(_REPO_NAME) + 8)}

This product contains weights released by torchvision
(https://github.com/pytorch/vision) at commit {_UPSTREAM_COMMIT}.
Official checkpoint: {_UPSTREAM_FILE}
Copyright (c) Soumith Chintala 2016 and torchvision contributors.

The checkpoint has no separate per-object license file. Redistribution uses
the releasing project's BSD-3-Clause license on an explicitly disclosed
implied basis; this is not a publisher-confirmed checkpoint-specific grant.
Torchvision warns that pretrained-model terms may derive from training data.
The ImageNet dataset and source-image terms remain the user's responsibility.

Conversion only adds LibreYOLO metadata. Learned tensors are unchanged.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint(path: Path) -> None:
    from libreyolo import LibreYOLO
    from libreyolo.models.alexnet.model import LibreAlexNet
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    if path.name != _FILENAME:
        raise ValueError(f"Expected canonical filename {_FILENAME}, got {path.name}")
    whitelist = (
        _REPO_ROOT / "skills" / "libreyolo-upload-hf-model" / "SKILL.md"
    ).read_text(encoding="utf-8")
    if _FILENAME not in whitelist:
        raise ValueError(
            f"Canonical filename is absent from upload whitelist: {_FILENAME}"
        )

    expected_url = (
        f"https://huggingface.co/LibreYOLO/{_REPO_NAME}/resolve/main/{_FILENAME}"
    )
    actual_url = LibreAlexNet.get_download_url(_FILENAME)
    if actual_url != expected_url:
        raise ValueError(
            f"Loader URL mismatch for {_FILENAME}: {actual_url!r} != {expected_url!r}"
        )
    if path.stat().st_size != _CONVERTED_BYTES:
        raise ValueError(
            f"Converted checkpoint byte count changed: {path.stat().st_size}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != _CONVERTED_SHA256:
        raise ValueError(f"Converted checkpoint SHA-256 changed: {actual_sha256}")

    checkpoint = load_untrusted_torch_file(path, context="converted checkpoint")
    errors = validate_checkpoint_metadata(checkpoint, strict=True)
    if errors:
        raise ValueError(f"Invalid checkpoint metadata: {'; '.join(errors)}")
    expected = {
        "model_family": "alexnet",
        "size": "b",
        "task": "classify",
        "nc": 1000,
        "imgsz": 224,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint metadata mismatch: {mismatches}")
    if len(checkpoint.get("names", {})) != 1000:
        raise ValueError("Checkpoint must contain all 1000 ImageNet class names")

    model = LibreYOLO(str(path), device="cpu")
    loaded = (model.family, model.size, model.nb_classes, model.task)
    expected_loaded = ("alexnet", "b", 1000, "classify")
    if loaded != expected_loaded:
        raise ValueError(
            f"Factory load mismatch: expected {expected_loaded}, got {loaded}"
        )


def build_repo_dir(pt_path: Path, out_dir: Path) -> Path:
    """Build and validate exactly five files for the weight repository."""
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

    repo_id = f"LibreYOLO/{_REPO_NAME}"
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
        commit_message="Initial upload: AlexNet ImageNet-1K weights",
    )
    expected = {".gitattributes", "README.md", "LICENSE", "NOTICE", _FILENAME}
    actual = {item.rfilename for item in api.model_info(repo_id).siblings}
    if actual != expected:
        raise RuntimeError(
            f"Remote five-file contract mismatch: expected {sorted(expected)}, got {sorted(actual)}"
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
