"""Build or publish SSD300's five-file Hugging Face weight repository.

The checkpoint has no per-object license file. The maintainer approved
rehosting under BSD-3-Clause implied by the releasing torchvision project,
with the pretrained-model caveat and Oxford VGG-16 CC BY 4.0 initialization
lineage disclosed. A real upload requires ``--confirm-implied-license``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_UPSTREAM_COMMIT = "336d36e8db990a905498c73933e35231876e28bc"
_UPSTREAM_FILE = "ssd300_vgg16_coco-b556d3b4.pth"
_UPSTREAM_SHA256 = (
    "b556d3b43ab6c3f63d81bfb8835fe8756ac22da664357da100dccf96b6a6b42d"
)
_CANONICAL_NAME = "LibreSSD300"
_COLLECTION = "LibreYOLO/libreyolo-models-698875bf2b5f695708415169"

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
    source_url = f"https://download.pytorch.org/models/{_UPSTREAM_FILE}"
    return f"""---
license: bsd-3-clause
library_name: libreyolo
pipeline_tag: object-detection
datasets:
  - detection-datasets/coco
tags:
  - object-detection
  - ssd
  - torchvision
  - libreyolo
---

# {_CANONICAL_NAME}

SSD300 with a VGG16 backbone, repackaged for LibreYOLO as an inference-only
historic model. Input is fixed at 300 x 300.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{_CANONICAL_NAME}.pt")
results = model.predict("image.jpg")
```

## Source

Derived from [pytorch/vision](https://github.com/pytorch/vision) at commit
[`{_UPSTREAM_COMMIT}`](https://github.com/pytorch/vision/commit/{_UPSTREAM_COMMIT}).
Copyright (c) Soumith Chintala 2016 and torchvision contributors. The source
implementation is BSD-3-Clause.

Official checkpoint: [{_UPSTREAM_FILE}]({source_url})
SHA-256: `{_UPSTREAM_SHA256}`
Published COCO val2017 box mAP: 25.1.

### VGG-16 initialization lineage

The torchvision SSD recipe initializes its backbone from VGG-16 feature
weights released by the [Visual Geometry Group, University of
Oxford](https://www.robots.ox.ac.uk/~vgg/research/very_deep/) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution:
Karen Simonyan and Andrew Zisserman, "Very Deep Convolutional Networks for
Large-Scale Image Recognition," ICLR 2015.

## Modifications

Torchvision modified the VGG graph for SSD and trained the detector on COCO.
LibreYOLO adds v1.0 checkpoint metadata; learned tensors and state-dict keys
are unchanged. The native graph has exact eager parity for preprocessing, both
raw heads, default boxes, and final detections. ONNX Runtime prediction parity
is also verified. See `weights/convert_ssd_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## License

The checkpoint publisher did not attach a separate per-object license file.
This mirror applies the releasing project's BSD-3-Clause license on an
**implied**, not publisher-confirmed, basis. Torchvision warns that pretrained
models may have licenses or terms derived from training data and that users
must determine whether they have permission for their use case. COCO
annotations are CC BY 4.0; source images retain their individual Flickr terms.
The Oxford attribution above records the VGG initialization lineage and does
not claim that Oxford licensed the complete SSD checkpoint. See
[`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
"""


def _notice() -> str:
    return f"""{_CANONICAL_NAME} weights
-----------------------

This product contains weights released by torchvision
(https://github.com/pytorch/vision) at commit {_UPSTREAM_COMMIT}.
Official checkpoint: {_UPSTREAM_FILE}
Copyright (c) Soumith Chintala 2016 and torchvision contributors.

The checkpoint has no separate per-object license file. Redistribution uses
the releasing project's BSD-3-Clause license on an explicitly disclosed
implied basis; this is not a publisher-confirmed checkpoint-specific grant.
Torchvision warns that pretrained-model terms may derive from training data
and users must determine permission for their use case.

The SSD backbone was initialized from VGG-16 feature weights released by the
Visual Geometry Group, University of Oxford, under Creative Commons
Attribution 4.0 International:
https://www.robots.ox.ac.uk/~vgg/research/very_deep/
https://creativecommons.org/licenses/by/4.0/
Creators: Karen Simonyan and Andrew Zisserman. Work: "Very Deep Convolutional
Networks for Large-Scale Image Recognition," ICLR 2015.

Torchvision modified the VGG graph for SSD and trained the detector on COCO.
LibreYOLO preserves the learned tensors unchanged and adds checkpoint metadata
only. The Oxford attribution records initialization lineage and does not claim
that Oxford licensed the complete SSD checkpoint.
"""


def _validate_checkpoint(path: Path) -> None:
    from libreyolo import LibreYOLO
    from libreyolo.models.ssd.model import LibreSSD
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
    actual_url = LibreSSD.get_download_url(filename)
    if actual_url != expected_url:
        raise ValueError(
            f"Loader URL mismatch for {filename}: {actual_url!r} != {expected_url!r}"
        )

    checkpoint = load_untrusted_torch_file(path, context="converted checkpoint")
    errors = validate_checkpoint_metadata(checkpoint, strict=False)
    if errors:
        raise ValueError(f"Invalid checkpoint metadata: {'; '.join(errors)}")
    expected = {
        "model_family": "ssd",
        "size": "300",
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
    expected_loaded = ("ssd", "300", 80, "detect")
    if loaded != expected_loaded:
        raise ValueError(
            f"Factory load mismatch: expected {expected_loaded}, got {loaded}"
        )


def build_repo_dir(pt_path: Path, out_dir: Path) -> Path:
    """Build and validate exactly five files for SSD300."""
    if not pt_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {pt_path}")
    _validate_checkpoint(pt_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / ".gitattributes").write_text(
        _GITATTRIBUTES, encoding="utf-8", newline="\n"
    )
    (out_dir / "README.md").write_text(_readme(), encoding="utf-8", newline="\n")
    (out_dir / "LICENSE").write_text(
        _BSD_LICENSE, encoding="utf-8", newline="\n"
    )
    (out_dir / "NOTICE").write_text(_notice(), encoding="utf-8", newline="\n")
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
            "Initial upload: LibreSSD300 (BSD-3-Clause implied, VGG attribution)"
        ),
    )
    expected = {
        ".gitattributes",
        "README.md",
        "LICENSE",
        "NOTICE",
        f"{_CANONICAL_NAME}.pt",
    }
    remote = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    if remote != expected:
        raise RuntimeError(
            f"Remote five-file contract mismatch: expected {sorted(expected)}, "
            f"got {sorted(remote)}"
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
