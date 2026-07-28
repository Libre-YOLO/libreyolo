"""Build and upload a LibreFeyNobg weight repo to the LibreYOLO HF org.

Follows skills/libreyolo-upload-hf-model (5-file contract: .gitattributes,
README.md, LICENSE, NOTICE, LibreFeyNobgl-matte[...].pt).

Usage::

    # default precision
    python weights/upload_feynobg_hf.py --pt weights/LibreFeyNobgl-matte.pt

    # quantized variants: the repo name gains a -<recipe> suffix and the model
    # card declares base_model feyninc/FeyNobg with
    # base_model_relation: quantized, so the repo appears in the
    # "Quantizations" sidebar of the upstream FeyNobg model page.
    python weights/upload_feynobg_hf.py --recipe fp8 --pt weights/LibreFeyNobgl-matte-fp8.pt
    python weights/upload_feynobg_hf.py --recipe nvfp4 --pt weights/LibreFeyNobgl-matte-nvfp4.pt

Add --dry-run to build the 5 files locally without creating/uploading the repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LICENSE_APACHE = """Apache License 2.0

Copyright 2026 Feyn Inc. (FeyNobg weights, https://huggingface.co/feyninc/FeyNobg)
FeyNobg builds on BiRefNet, Copyright (c) 2024 ZhengPeng (Peng Zheng), MIT License.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

_GITATTRIBUTES = "*.pt filter=lfs diff=lfs merge=lfs -text\n"

_NOTICE = """LibreFeyNobgl-matte{recipe_suffix} weights
------------------------------------

This product contains weights derived from FeyNobg
(https://huggingface.co/feyninc/FeyNobg, https://github.com/feyninc/nobg).
Copyright (c) 2026 Feyn Inc.
Licensed under the Apache License, Version 2.0.

FeyNobg is built on BiRefNet
(https://github.com/ZhengPeng7/BiRefNet).
Copyright (c) 2024 ZhengPeng (Peng Zheng).
Licensed under the MIT License.

{transform_note}
See weights/convert_feynobg_weights.py in the LibreYOLO source repository.
"""

_TRANSFORM_WRAP = (
    "Conversion is a state-dict metadata-wrap: learned parameters are unchanged."
)
_TRANSFORM_QUANT = (
    "Weights are post-training quantized ({recipe}) with LibreYOLO's quantize "
    "API and stored in the packed finalized checkpoint format (see "
    "docs/quantization.md and docs/checkpoint_schema.md)."
)

_RECIPE_DESC = {
    "fp8": "fp8 (E4M3 weights+activations on Conv2d and Linear, calibrated)",
    "nvfp4": "NVFP4 (E2M1 Linear weights in 16-element blocks with FP8 block "
    "scales; non-Linear layers stay in higher precision)",
}


def _readme(recipe: str | None) -> str:
    name = "LibreFeyNobgl-matte" + (f"-{recipe}" if recipe else "")
    lines = [
        "---",
        "license: apache-2.0",
        "library_name: libreyolo",
        "pipeline_tag: image-segmentation",
        "base_model: feyninc/FeyNobg",
    ]
    if recipe:
        lines += ["base_model_relation: quantized"]
    lines += [
        "tags:",
        "  - background-removal",
        "  - matte",
        "  - dichotomous-image-segmentation",
        "  - feynobg",
        "  - birefnet",
    ]
    if recipe:
        lines += [f"  - {recipe}", "  - quantized"]
    lines += ["  - libreyolo", "---", ""]
    head = "\n".join(lines)

    quant_para = (
        f"\nThis repo hosts the **{_RECIPE_DESC[recipe]}** post-training-quantized "
        "variant. The default-precision weights auto-download; quantized "
        "variants are opt-in: download the `.pt` and pass its path as the "
        "weights argument (the checkpoint's `quant` manifest rebuilds the "
        "quantized structure at load time).\n"
        if recipe
        else ""
    )

    if recipe:
        modifications = f"""## Modifications

State-dict metadata-wrap into the LibreYOLO v1.0 checkpoint schema, then
post-training quantization with LibreYOLO's `quantize` API
({_RECIPE_DESC[recipe]}), stored in the packed finalized format documented in
`docs/quantization.md` and `docs/checkpoint_schema.md` of the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo)."""
    else:
        modifications = """## Modifications

State-dict key remapping only (metadata-wrap into the LibreYOLO v1.0 checkpoint
schema). Learned parameters are unchanged. Our fp32 forward matches the upstream
released weights with `max_abs_diff == 0` (weights/parity_feynobg.py). See
`weights/convert_feynobg_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo)."""

    return f"""{head}
# {name}

FeyNobg background removal, repackaged for LibreYOLO's `matte` task. Predicts
a soft alpha matte at a fixed native 1024x1024.
{quant_para}
```python
from libreyolo import LibreYOLO

m = LibreYOLO("{name}.pt")
res = m.predict("product.jpg")
res[0].matte            # (H, W) float alpha in [0, 1]
res[0].save("cut.png")  # transparent-background PNG
```

## Source

Derived from [feyninc/FeyNobg](https://huggingface.co/feyninc/FeyNobg)
([nobg library](https://github.com/feyninc/nobg)), Apache-2.0,
Copyright (c) 2026 Feyn Inc. FeyNobg builds on
[ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (MIT,
Copyright (c) 2024 ZhengPeng).

Backbone: Swin Transformer v1, Swin-L tier with stage 3 deepened from 18 to
24 blocks (263M parameters). Training data provenance (upstream): not
disclosed by Feyn Inc.; this repo redistributes the author's released
weights under their Apache-2.0 grant and does not redistribute training data.

{modifications}

## License

Apache License 2.0. See the [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE) files.
"""


def build_repo_dir(pt_path: Path, out_dir: Path, recipe: str | None = None) -> Path:
    name = "LibreFeyNobgl-matte" + (f"-{recipe}" if recipe else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitattributes").write_text(_GITATTRIBUTES, encoding="utf-8")
    (out_dir / "README.md").write_text(_readme(recipe), encoding="utf-8")
    (out_dir / "LICENSE").write_text(_LICENSE_APACHE, encoding="utf-8")
    transform = _TRANSFORM_QUANT.format(recipe=recipe) if recipe else _TRANSFORM_WRAP
    (out_dir / "NOTICE").write_text(
        _NOTICE.format(recipe_suffix=f"-{recipe}" if recipe else "", transform_note=transform),
        encoding="utf-8",
    )
    target = out_dir / f"{name}.pt"
    if pt_path.resolve() != target.resolve():
        import shutil

        shutil.copy(pt_path, target)
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to the converted LibreFeyNobgl-matte[...].pt")
    ap.add_argument(
        "--recipe",
        default=None,
        choices=sorted(_RECIPE_DESC),
        help="Quantized variant: repo gains a -<recipe> suffix and "
        "base_model_relation: quantized metadata pointing at feyninc/FeyNobg.",
    )
    ap.add_argument("--out", default=None, help="Local build dir (default: temp)")
    ap.add_argument("--dry-run", action="store_true", help="Build files only; do not create/upload the repo")
    args = ap.parse_args()

    pt_path = Path(args.pt)
    if not pt_path.exists():
        print(f"Weight file not found: {pt_path}", file=sys.stderr)
        return 1

    name = "LibreFeyNobgl-matte" + (f"-{args.recipe}" if args.recipe else "")
    repo = f"LibreYOLO/{name}"
    out_dir = Path(args.out) if args.out else Path(f"./_hf_build_{name}")
    build_repo_dir(pt_path, out_dir, recipe=args.recipe)
    print(f"Built 5-file repo in {out_dir}")

    if args.dry_run:
        print("--dry-run: not uploading.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo, repo_type="model",
                      commit_message=f"Initial upload: {name} (FeyNobg, Apache-2.0)")
    print(f"Uploaded to https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
