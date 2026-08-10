"""Mirror Meta's SAM-1 transformers snapshots onto the LibreYOLO org.

    .venv/Scripts/python.exe scripts/mirror_sam.py base [large huge]

Why
---
LibreSAM1 loads through ``transformers`` from facebook/sam-vit-{base,large,huge}.
Those weights are Apache-2.0, which permits redistribution, so the dependency
on Meta's repos is a choice rather than a constraint.

Shape
-----
This family ships snapshot directories rather than a single ``.pt``, the same
as LibreGroundingDINO and LibreOWLv2, so the 5-file contract does not apply
verbatim; the repo mirrors the upstream layout plus a card, LICENSE and NOTICE.

Only the PyTorch-relevant files are copied. Upstream also carries
``pytorch_model.bin`` (a duplicate of the safetensors weights) and
``tf_model.h5``; mirroring those would triple the repo for nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ASSETS = Path(
    "C:/Users/Usuario/AppData/Local/Temp/claude/C--Users-Usuario/"
    "c6dfe57f-09e9-40c8-afb6-aaa1e472f308/scratchpad/mirror-assets"
)
STAGING = Path(
    "C:/Users/Usuario/AppData/Local/Temp/claude/C--Users-Usuario/"
    "c6dfe57f-09e9-40c8-afb6-aaa1e472f308/scratchpad/mirror-staging"
)

UPSTREAM = {
    "base": "facebook/sam-vit-base",
    "large": "facebook/sam-vit-large",
    "huge": "facebook/sam-vit-huge",
}

# Everything transformers needs to rebuild the model and its processor.
KEEP = ["config.json", "preprocessor_config.json", "model.safetensors"]

README = """---
license: apache-2.0
library_name: libreyolo
tags:
  - mask-generation
  - image-segmentation
  - sam
---

# {repo}

Segment Anything (SAM-1) ViT-{size_title}, mirrored for LibreYOLO.

## Source

Mirrored from [{upstream}](https://huggingface.co/{upstream}).
Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the Apache License, Version 2.0.

## Modifications

None. The weights and configuration are Meta's, byte for byte. This repository
exists so LibreYOLO can serve them from its own organisation rather than
depending on a third-party repository remaining available.

Upstream also publishes `pytorch_model.bin` (a duplicate of the safetensors
weights) and `tf_model.h5`. Neither is mirrored, because LibreYOLO loads the
safetensors file.

## Usage

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreSAM{size}.pt")
```

## License

Apache License 2.0. See the [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE)
files in this repository.
"""

NOTICE = """LibreSAM weights
----------------

This product contains weights from Segment Anything (SAM)
(https://github.com/facebookresearch/segment-anything).
Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the Apache License, Version 2.0.

The weights are redistributed unmodified.
"""


def stage(size: str) -> Path:
    upstream = UPSTREAM[size]
    repo_name = f"LibreSAM{size}"
    out = STAGING / repo_name
    out.mkdir(parents=True, exist_ok=True)

    print(f"  downloading {upstream} ({', '.join(KEEP)}) ...", flush=True)
    snapshot_download(
        repo_id=upstream,
        local_dir=str(out),
        allow_patterns=KEEP,
    )

    (out / "LICENSE").write_text(
        (ASSETS / "LICENSE-sam").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out / "NOTICE").write_text(NOTICE, encoding="utf-8")
    (out / "README.md").write_text(
        README.format(repo=repo_name, size=size, size_title=size.title(), upstream=upstream),
        encoding="utf-8",
    )

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"  staged {repo_name}: {sorted(p.name for p in out.iterdir())} "
          f"({total / 1e6:.0f} MB)", flush=True)
    return out


def upload(size: str, folder: Path) -> None:
    repo_id = f"LibreYOLO/LibreSAM{size}"
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    url = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Mirror SAM-1 ViT-{size} from {UPSTREAM[size]} (Apache-2.0)",
    )
    print(f"  uploaded {repo_id}: {url}", flush=True)


def main() -> int:
    sizes = sys.argv[1:] or ["base"]
    for size in sizes:
        if size not in UPSTREAM:
            raise SystemExit(f"unknown size {size!r}; expected one of {list(UPSTREAM)}")
        print(f"\n=== {size} ===", flush=True)
        upload(size, stage(size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
