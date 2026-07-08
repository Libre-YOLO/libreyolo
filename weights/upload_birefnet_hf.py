"""Build and upload a LibreBiRefNet weight repo to the LibreYOLO HF org.

Follows skills/libreyolo-upload-hf-model (5-file contract: .gitattributes,
README.md, LICENSE, NOTICE, Libre<Family><size>-matte.pt).

Usage::

    # general (l): weights are MIT-tagged upstream -> host
    python weights/upload_birefnet_hf.py --size l --pt weights/LibreBiRefNetl-matte.pt

    # lite (t): the BiRefNet_lite HF repo has no explicit license tag; hosting is
    # a maintainer decision. --confirm-lite-license is required to proceed.
    python weights/upload_birefnet_hf.py --size t --pt weights/LibreBiRefNett-matte.pt --confirm-lite-license

Add --dry-run to build the 5 files locally without creating/uploading the repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LICENSE_MIT = """MIT License

Copyright (c) 2024 ZhengPeng

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

_GITATTRIBUTES = "*.pt filter=lfs diff=lfs merge=lfs -text\n"

_NOTICE = """LibreBiRefNet{size}-matte weights
------------------------------------

This product contains weights derived from BiRefNet
(https://github.com/ZhengPeng7/BiRefNet).
Copyright (c) 2024 ZhengPeng (Peng Zheng).
Licensed under the MIT License.

Conversion is a state-dict metadata-wrap: learned parameters are unchanged.
See weights/convert_birefnet_weights.py in the LibreYOLO source repository.
"""

_SIZE_DESC = {
    "l": "BiRefNet general (Swin-L tier), the quality default",
    "t": "BiRefNet_lite (Swin-T tier), the fast variant",
}


def _readme(size: str) -> str:
    return f"""---
license: mit
library_name: libreyolo
pipeline_tag: image-segmentation
tags:
  - background-removal
  - matte
  - dichotomous-image-segmentation
  - birefnet
  - libreyolo
---

# LibreBiRefNet{size}-matte

BiRefNet background removal ({_SIZE_DESC[size]}), repackaged for LibreYOLO's
`matte` task. Predicts a soft alpha matte at a fixed native 1024x1024.

```python
from libreyolo import LibreYOLO

m = LibreYOLO("LibreBiRefNet{size}-matte.pt")
res = m.predict("product.jpg")
res[0].matte            # (H, W) float alpha in [0, 1]
res[0].save("cut.png")  # transparent-background PNG
```

## Source

Derived from [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
at commit d83f355.
Copyright (c) 2024 ZhengPeng (Peng Zheng). Licensed under the MIT License.

Backbone: Swin Transformer v1 ({'Swin-L' if size == 'l' else 'Swin-T'}).
Training data provenance (upstream): the BiRefNet DIS/General checkpoints are
trained on dichotomous-image-segmentation datasets (e.g. DIS5K) under their own
academic terms; this repo hosts the author's released weights and does not
redistribute training data.

## Modifications

State-dict key remapping only (metadata-wrap into the LibreYOLO v1.0 checkpoint
schema). Learned parameters are unchanged. Our fp32 forward matches the upstream
released weights with `max_abs_diff == 0`. See
`weights/convert_birefnet_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## License

MIT License. See the [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE) files.
"""


def build_repo_dir(size: str, pt_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitattributes").write_text(_GITATTRIBUTES, encoding="utf-8")
    (out_dir / "README.md").write_text(_readme(size), encoding="utf-8")
    (out_dir / "LICENSE").write_text(_LICENSE_MIT, encoding="utf-8")
    (out_dir / "NOTICE").write_text(_NOTICE.format(size=size), encoding="utf-8")
    target = out_dir / f"LibreBiRefNet{size}-matte.pt"
    if pt_path.resolve() != target.resolve():
        import shutil

        shutil.copy(pt_path, target)
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=["t", "l"])
    ap.add_argument("--pt", required=True, help="Path to the converted LibreBiRefNet<size>-matte.pt")
    ap.add_argument("--out", default=None, help="Local build dir (default: temp)")
    ap.add_argument("--dry-run", action="store_true", help="Build files only; do not create/upload the repo")
    ap.add_argument(
        "--confirm-lite-license",
        action="store_true",
        help="Required for --size t: confirm the BiRefNet_lite weights license permits rehosting.",
    )
    args = ap.parse_args()

    if args.size == "t" and not args.confirm_lite_license and not args.dry_run:
        print(
            "REFUSING to upload the lite (t) weights: the BiRefNet_lite Hugging Face "
            "repo has no explicit license metadata (MIT badge in the card body only). "
            "Re-verify the weights license and pass --confirm-lite-license to proceed. "
            "See weights/LICENSE_NOTICE.txt.",
            file=sys.stderr,
        )
        return 2

    pt_path = Path(args.pt)
    if not pt_path.exists():
        print(f"Weight file not found: {pt_path}", file=sys.stderr)
        return 1

    repo = f"LibreYOLO/LibreBiRefNet{args.size}-matte"
    out_dir = Path(args.out) if args.out else Path(f"./_hf_build_LibreBiRefNet{args.size}-matte")
    build_repo_dir(args.size, pt_path, out_dir)
    print(f"Built 5-file repo in {out_dir}")

    if args.dry_run:
        print("--dry-run: not uploading.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo, repo_type="model",
                      commit_message=f"Initial upload: LibreBiRefNet{args.size}-matte (BiRefNet, MIT)")
    print(f"Uploaded to https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
