"""Stage the LibreMoGe2 mirror repos for the LibreYOLO Hugging Face org.

    .venv/Scripts/python.exe scripts/mirror_moge2.py <staging-dir>

Why
---
LibreMoGe2 currently fetches from Ruicheng's Hugging Face repos at a pinned
revision, so a first run depends on a third party staying put. The weights are
MIT, which permits redistribution, so there is no reason for that dependency.

This stages one directory per size following the 5-file contract in
skills/libreyolo-upload-hf-model. It does not upload; upload is a separate,
deliberate step.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

from libreyolo.utils.serialization import validate_checkpoint_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "weights"
ASSETS = Path(
    "C:/Users/Usuario/AppData/Local/Temp/claude/C--Users-Usuario/"
    "c6dfe57f-09e9-40c8-afb6-aaa1e472f308/scratchpad/mirror-assets"
)

# Upstream repo and the exact revision LibreYOLO pins today, so the mirror
# records which bytes it came from rather than "latest".
SIZES = {
    "l": {
        "converted": "LibreMoGe2l-normal-LibreMoGe2l-normal.pt",
        "upstream": "Ruicheng/moge-2-vitl-normal",
        "revision": "b135031bae30b5ac2ae141a0e68717795ce38340",
        "arch": "MoGe-2 ViT-L/14",
    },
    "s": {
        "converted": "LibreMoGe2s-normal-LibreMoGe2s-normal.pt",
        "upstream": "Ruicheng/moge-2-vits-normal",
        "revision": "679230677b4d282c6f304189a93e98e14f085902",
        "arch": "MoGe-2 ViT-S/14",
    },
}

README = """---
license: mit
library_name: libreyolo
tags:
  - image-to-image
  - surface-normals
  - moge
---

# {repo}

{arch} surface-normal estimator, repackaged in LibreYOLO checkpoint format.

## Source

Derived from [{upstream}](https://huggingface.co/{upstream})
at revision `{revision}`, the exact commit LibreYOLO pins.
Copyright (c) Microsoft Corporation. Licensed under the MIT License.

The DINOv2 encoder MoGe-2 builds on is separately licensed Apache-2.0 by
Meta AI.

## Modifications

State-dict key remapping only. Learned parameters are unchanged.
See `weights/convert_moge2_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## Usage

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{repo}.pt")
result = model.predict("image.jpg")[0]
normals = result.normals
```

## License

MIT License. See the [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE) files in
this repository.
"""

NOTICE = """LibreMoGe2 weights
------------------

This product contains weights derived from MoGe-2
(https://github.com/microsoft/MoGe).
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

The DINOv2 encoder these weights build on is separately licensed under the
Apache License, Version 2.0, by Meta AI
(https://github.com/facebookresearch/dinov2).
"""


def stage(size: str, spec: dict, out_root: Path) -> Path:
    repo = f"LibreMoGe2{size}-normal"
    src = WEIGHTS / spec["converted"]
    if not src.exists():
        raise FileNotFoundError(f"converted checkpoint missing: {src}")

    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    errors = validate_checkpoint_metadata(checkpoint, strict=False)
    if errors:
        raise SystemExit(f"{repo}: checkpoint metadata invalid: {errors}")

    out = out_root / repo
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(ASSETS / ".gitattributes", out / ".gitattributes")
    shutil.copy2(ASSETS / "LICENSE", out / "LICENSE")
    (out / "NOTICE").write_text(NOTICE, encoding="utf-8")
    (out / "README.md").write_text(
        README.format(repo=repo, arch=spec["arch"], upstream=spec["upstream"],
                      revision=spec["revision"]),
        encoding="utf-8",
    )
    # Canonical filename, not the doubled name the converter leaves behind.
    shutil.copy2(src, out / f"{repo}.pt")

    files = sorted(p.name for p in out.iterdir())
    size_mb = (out / f"{repo}.pt").stat().st_size / 1e6
    print(f"  {repo}: {files} ({size_mb:.0f} MB)", flush=True)
    if len(files) != 5:
        raise SystemExit(f"{repo}: expected exactly 5 files, got {len(files)}")
    return out


def main() -> int:
    out_root = Path(sys.argv[1] if len(sys.argv) > 1 else "mirror-staging")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"staging MoGe-2 mirrors under {out_root}", flush=True)
    for size, spec in SIZES.items():
        stage(size, spec, out_root)
    print("staged. nothing uploaded yet.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
