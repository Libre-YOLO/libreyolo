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

# Verbatim upstream LICENSE from https://github.com/feyninc/nobg (Apache-2.0,
# "Copyright 2026 Feyn"), per the upload skill: copy verbatim, never synthesize.
_LICENSE_APACHE = """                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or Derivative
          Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2026 Feyn

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
    "Conversion is a deterministic state-dict key remap (fused qkv, renamed "
    "modules) into the LibreYOLO checkpoint schema: learned parameters are "
    "unchanged."
)
_TRANSFORM_QUANT = (
    "Weights are post-training quantized ({recipe}) with LibreYOLO's quantize "
    "API and stored in the packed finalized checkpoint format (see "
    "docs/quantization.md and docs/checkpoint_schema.md)."
)

_RECIPE_DESC = {
    "fp16": "fp16 (half-precision cast, float32 I/O contract; near-lossless, "
    "intended for GPU inference - on CPU use the fp32 default)",
    "fp8": "fp8 (E4M3 weights, calibrated static scales, fp16 remainder; on "
    "Ada/Hopper/Blackwell GPUs the Linear layers execute natively on the fp8 "
    "tensor cores via torch._scaled_mm, matching the fp16 checkpoint's speed "
    "at half its size - pass cuda_graph=True to predict for the full effect)",
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

State-dict key remapping only (fused qkv, renamed modules, wrapped into the
LibreYOLO v1.0 checkpoint schema). Learned parameters are unchanged. Our fp32
forward matches the upstream released weights with `max_abs_diff == 0`
(weights/parity_feynobg.py). See `weights/convert_feynobg_weights.py` in the
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
