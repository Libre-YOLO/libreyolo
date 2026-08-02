"""Build or publish Faster R-CNN's five-file Hugging Face weight repos.

The checkpoints have no per-object license file. The maintainer approved
rehosting under BSD-3-Clause implied by the releasing torchvision project,
with the upstream pretrained-model caveat disclosed on every card. A real
upload therefore requires ``--confirm-implied-license``; ``--dry-run`` only
builds and validates the local repository.

Examples::

    python weights/upload_faster_rcnn_hf.py --size n \
        --pt LibreFasterRCNNn.pt --out ./LibreFasterRCNNn --dry-run

    python weights/upload_faster_rcnn_hf.py --size n \
        --pt LibreFasterRCNNn.pt --out ./LibreFasterRCNNn \
        --confirm-implied-license
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
    "n": {
        "description": "MobileNetV3-Large 320-FPN low-resolution variant",
        "upstream_file": "fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
        "sha256": "907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5",
        "box_map": "22.8",
    },
    "s": {
        "description": "MobileNetV3-Large FPN variant",
        "upstream_file": "fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
        "sha256": "fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533",
        "box_map": "32.8",
    },
    "m": {
        "description": "ResNet-50 FPN v1 recipe variant",
        "upstream_file": "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        "sha256": "258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384",
        "box_map": "37.0",
    },
    "l": {
        "description": "ResNet-50 FPN v2 enhanced-recipe variant",
        "upstream_file": "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
        "sha256": "dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf",
        "box_map": "46.7",
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


def _canonical_name(size: str) -> str:
    return f"LibreFasterRCNN{size}"


def _readme(size: str) -> str:
    variant = _VARIANTS[size]
    name = _canonical_name(size)
    source_url = f"https://download.pytorch.org/models/{variant['upstream_file']}"
    benchmark_slug = f"faster_rcnn-{size}"
    return f"""---
license: bsd-3-clause
library_name: libreyolo
pipeline_tag: object-detection
datasets:
  - detection-datasets/coco
tags:
  - object-detection
  - faster-rcnn
  - torchvision
  - libreyolo
---

# {name}

Modernized Faster R-CNN ({variant['description']}), repackaged for LibreYOLO.
This is a torchvision COCO recipe, not the original 2015 VGG16 architecture.

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
SHA-256: `{variant['sha256']}`
Published COCO val2017 box mAP: {variant['box_map']}.

## Modifications

Checkpoint metadata was added for LibreYOLO's v1.0 schema. Learned tensors and
state-dict keys are unchanged. The native LibreYOLO graph loads the official
state dict strictly and has exact eager parity at the RPN head, RoI predictor,
and final detections. See `weights/convert_faster_rcnn_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## Benchmarks

Independent accuracy and speed benchmarks:
[visionanalysis.org/model/{benchmark_slug}](https://www.visionanalysis.org/model/{benchmark_slug})

## License

The checkpoint publisher did not attach a separate per-object license file.
This mirror applies the releasing project's BSD-3-Clause license on an
**implied**, not publisher-confirmed, basis. Torchvision warns that pretrained
models may have their own licenses or terms derived from training data and
that users must determine whether they have permission for their use case.
COCO annotations are CC BY 4.0; source images retain their individual Flickr
terms. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
"""


def _notice(size: str) -> str:
    name = _canonical_name(size)
    variant = _VARIANTS[size]
    return f"""{name} weights
{'-' * (len(name) + 8)}

This product contains weights released by torchvision
(https://github.com/pytorch/vision) at commit {_UPSTREAM_COMMIT}.
Official checkpoint: {variant['upstream_file']}
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
    from libreyolo.models.faster_rcnn.model import LibreFasterRCNN
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
        raise ValueError(f"Canonical filename is absent from upload whitelist: {filename}")
    expected_url = (
        f"https://huggingface.co/LibreYOLO/{_canonical_name(size)}"
        f"/resolve/main/{filename}"
    )
    actual_url = LibreFasterRCNN.get_download_url(filename)
    if actual_url != expected_url:
        raise ValueError(
            f"Loader URL mismatch for {filename}: {actual_url!r} != {expected_url!r}"
        )

    checkpoint = load_untrusted_torch_file(path, context="converted checkpoint")
    errors = validate_checkpoint_metadata(checkpoint, strict=False)
    if errors:
        raise ValueError(f"Invalid checkpoint metadata: {'; '.join(errors)}")
    expected = {
        "model_family": "faster_rcnn",
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
    expected_loaded = ("faster_rcnn", size, 80, "detect")
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
    (out_dir / "README.md").write_text(
        _readme(size), encoding="utf-8", newline="\n"
    )
    (out_dir / "LICENSE").write_text(
        _BSD_LICENSE, encoding="utf-8", newline="\n"
    )
    (out_dir / "NOTICE").write_text(
        _notice(size), encoding="utf-8", newline="\n"
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
            f"Five-file contract mismatch: expected {sorted(expected)}, got {sorted(actual)}"
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
        commit_message=f"Initial upload: {repo_name} (Faster R-CNN, BSD-3-Clause implied)",
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
