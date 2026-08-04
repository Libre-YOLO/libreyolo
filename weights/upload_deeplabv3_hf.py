"""Build or publish DeepLabv3's five-file Hugging Face weight repos.

The checkpoints have no per-object license file. Rehosting uses BSD-3-Clause
implied by the releasing torchvision project, with the upstream pretrained-
model caveat disclosed on every card. A real upload therefore requires
``--confirm-implied-license``; ``--dry-run`` only builds and validates locally.

Example::

    python weights/upload_deeplabv3_hf.py --size r50 \
        --pt LibreDeepLabv3r50-sem.pt --out ./LibreDeepLabv3r50-sem \
        --confirm-implied-license
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
_BSD_LICENSE_SHA256 = "6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d"
_COLLECTION = "LibreYOLO/libreyolo-models-698875bf2b5f695708415169"
_MARKDOWN_LINE_BREAK = "  "

_VARIANTS = {
    "r50": {
        "description": "dilated ResNet-50, output stride 8",
        "upstream_file": "deeplabv3_resnet50_coco-cd0a2569.pth",
        "upstream_bytes": 168_312_152,
        "upstream_sha256": "cd0a25694c4a0f7106b38f4938bf90a874f2f241cc410b8f63c7024399538f06",
        "converted_bytes": 158_900_443,
        "converted_sha256": "a8910db2cb2827ec19fce65a051f4d651bee73f5a46ba8d1c431c0d7042dca7c",
        "miou": "66.4",
        "pixel_accuracy": "92.4",
        "onnx_max_abs": "1.53e-5",
        "openvino_mask": "99.9994%",
        "tensorrt_mask": "99.9981%",
    },
    "r101": {
        "description": "dilated ResNet-101, output stride 8",
        "upstream_file": "deeplabv3_resnet101_coco-586e9e4e.pth",
        "upstream_bytes": 244_545_539,
        "upstream_sha256": "586e9e4e203fcbf17e1ad45533d8d33ab133fc762bf03101c5dd743995c08c0d",
        "converted_bytes": 235_177_707,
        "converted_sha256": "4575b7d5b1b70e9c67225ae76c00f552b29c2e54b07d55cfee8da218a9f41429",
        "miou": "67.4",
        "pixel_accuracy": "92.4",
        "onnx_max_abs": "1.34e-5",
        "openvino_mask": "99.9994%",
        "tensorrt_mask": "99.9986%",
    },
    "mv3": {
        "description": "dilated MobileNetV3-Large, output stride 16",
        "upstream_file": "deeplabv3_mobilenet_v3_large-fc3c493d.pth",
        "upstream_bytes": 44_356_159,
        "upstream_sha256": "fc3c493d68e89cc31ef488c803d5d7dd2f3190fb570598faa49fef69be8e5e70",
        "converted_bytes": 44_325_189,
        "converted_sha256": "fb83a67bca845817d816d139af6fb6a4b9d809c0a813ebcfcb1e2a5fbd222682",
        "miou": "60.3",
        "pixel_accuracy": "91.2",
        "onnx_max_abs": "3.06e-5",
        "openvino_mask": "99.9876%",
        "tensorrt_mask": "99.9851%",
    },
}

_GITATTRIBUTES = "*.pt filter=lfs diff=lfs merge=lfs -text\n"


def _canonical_name(size: str) -> str:
    return f"LibreDeepLabv3{size}-sem"


def _bsd_license() -> str:
    notice = (_REPO_ROOT / "libreyolo/models/deeplabv3/NOTICE").read_text(
        encoding="utf-8"
    )
    marker = "BSD 3-Clause License\n"
    if marker not in notice:
        raise RuntimeError("DeepLabv3 NOTICE does not contain the BSD license text")
    license_text = marker + notice.split(marker, 1)[1]
    # The upstream file has one intentional trailing space after ``2016,``.
    # Add it at runtime so the generated LICENSE is byte-for-byte identical
    # without committing trailing whitespace to this source file.
    license_text = license_text.replace("2016,\n", "2016, \n", 1)
    digest = hashlib.sha256(license_text.encode("utf-8")).hexdigest()
    if digest != _BSD_LICENSE_SHA256:
        raise RuntimeError(
            "DeepLabv3 NOTICE no longer reproduces the pinned torchvision "
            f"license: {digest} != {_BSD_LICENSE_SHA256}"
        )
    return license_text


def _readme(size: str) -> str:
    variant = _VARIANTS[size]
    name = _canonical_name(size)
    source_url = f"https://download.pytorch.org/models/{variant['upstream_file']}"
    return f"""---
license: bsd-3-clause
library_name: libreyolo
pipeline_tag: image-segmentation
datasets:
  - detection-datasets/coco
tags:
  - semantic-segmentation
  - deeplabv3
  - torchvision
  - pascal-voc
  - libreyolo
---

# {name}

DeepLabv3 semantic segmentation with {variant["description"]}, repackaged for
LibreYOLO. It predicts background plus 20 Pascal VOC-named foreground classes
from a checkpoint trained on the matching COCO subset. This is DeepLabv3, not
DeepLabv3+; there is no decoder or CRF.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{name}.pt")
result = model.predict("image.jpg")
mask = result.semantic_mask.data
```

## Source

Derived from [pytorch/vision](https://github.com/pytorch/vision) at commit
[`{_UPSTREAM_COMMIT}`](https://github.com/pytorch/vision/commit/{_UPSTREAM_COMMIT})
(torchvision v0.26.0). Copyright (c) Soumith Chintala 2016 and torchvision
contributors. The source implementation is BSD-3-Clause.

Official checkpoint: [{variant["upstream_file"]}]({source_url}){_MARKDOWN_LINE_BREAK}
Bytes: `{variant["upstream_bytes"]}`{_MARKDOWN_LINE_BREAK}
SHA-256: `{variant["upstream_sha256"]}`{_MARKDOWN_LINE_BREAK}
Published mIoU / pixel accuracy: {variant["miou"]} / {variant["pixel_accuracy"]}.

The published metrics use torchvision's aspect-preserving evaluation preset.
LibreYOLO uses a fixed 520x520 stretch deployment contract, followed by
ImageNet normalization and restoration of the output mask to the source
canvas, so end-to-end metrics can differ.

## Modifications and verification

Conversion removes only the training-time `aux_classifier.*` tensors and adds
LibreYOLO v1.0 checkpoint metadata. Every retained runtime tensor and state-dict
key is unchanged. The native 520x520 logits are bit-exact against the pinned
torchvision implementation before postprocessing (`max_abs_diff == 0.0`).

The fixed-shape deployment graph was also tested through LibreYOLO's unified
backend:

- ONNX Runtime CPU: 100% identical public mask pixels; maximum logit difference
  `{variant["onnx_max_abs"]}`.
- TorchScript: bit-exact logits and 100% identical public mask pixels.
- OpenVINO CPU: {variant["openvino_mask"]} identical public mask pixels using
  the runtime's default reduced-precision execution hint.
- TensorRT 10.16 FP32 on RTX 5070 Ti: {variant["tensorrt_mask"]} identical public
  mask pixels.

The converted file has `{variant["converted_bytes"]}` bytes and SHA-256
`{variant["converted_sha256"]}`. See `docs/provenance/deeplabv3.md` and
`weights/convert_deeplabv3_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## License

The checkpoint publisher did not attach a separate per-object license file.
This mirror applies the releasing project's BSD-3-Clause license on an
**implied, not publisher-confirmed**, basis. Torchvision warns that pretrained
models may have licenses or terms derived from training data and that users
must determine whether they have permission for their use case. COCO
annotations are CC BY 4.0; source images retain their individual Flickr terms.
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
Copyright (c) Soumith Chintala 2016 and torchvision contributors.

The checkpoint has no separate per-object license file. Redistribution uses
the releasing project's BSD-3-Clause license on an explicitly disclosed
implied basis; this is not a publisher-confirmed checkpoint-specific grant.
Torchvision warns that pretrained-model terms may derive from training data
and users must determine permission for their use case. COCO annotations are
CC BY 4.0 and source images retain their individual Flickr terms.

Conversion removes only the training-time auxiliary classifier and adds
LibreYOLO metadata. Every retained learned tensor is unchanged.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint(size: str, path: Path) -> None:
    from libreyolo import LibreYOLO
    from libreyolo.models.deeplabv3.model import LibreDeepLabv3, VOC_NAMES
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    variant = _VARIANTS[size]
    filename = f"{_canonical_name(size)}.pt"
    if path.name != filename:
        raise ValueError(f"Expected canonical filename {filename}, got {path.name}")
    whitelist = (_REPO_ROOT / "skills/libreyolo-upload-hf-model/SKILL.md").read_text(
        encoding="utf-8"
    )
    if filename not in whitelist:
        raise ValueError(
            f"Canonical filename is absent from upload whitelist: {filename}"
        )
    expected_url = (
        f"https://huggingface.co/LibreYOLO/{_canonical_name(size)}"
        f"/resolve/main/{filename}"
    )
    actual_url = LibreDeepLabv3.get_download_url(filename)
    if actual_url != expected_url:
        raise ValueError(
            f"Loader URL mismatch for {filename}: {actual_url!r} != {expected_url!r}"
        )
    if path.stat().st_size != variant["converted_bytes"]:
        raise ValueError(
            f"Converted byte count mismatch for {filename}: "
            f"{path.stat().st_size} != {variant['converted_bytes']}"
        )
    digest = _sha256(path)
    if digest != variant["converted_sha256"]:
        raise ValueError(
            f"Converted SHA-256 mismatch for {filename}: "
            f"{digest} != {variant['converted_sha256']}"
        )

    checkpoint = load_untrusted_torch_file(path, context="converted checkpoint")
    validate_checkpoint_metadata(checkpoint, strict=True)
    expected = {
        "model_family": "deeplabv3",
        "size": size,
        "task": "semantic",
        "nc": 21,
        "imgsz": 520,
        "names": VOC_NAMES,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint metadata mismatch: {mismatches}")

    model = LibreYOLO(str(path), device="cpu")
    loaded = (model.family, model.size, model.nb_classes, model.task, model.input_size)
    expected_loaded = ("deeplabv3", size, 21, "semantic", 520)
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
    (out_dir / "LICENSE").write_text(_bsd_license(), encoding="utf-8", newline="\n")
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
        commit_message=(
            f"Initial upload: {repo_name} (DeepLabv3, BSD-3-Clause implied)"
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
        help="Confirm the disclosed implied BSD redistribution basis.",
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
