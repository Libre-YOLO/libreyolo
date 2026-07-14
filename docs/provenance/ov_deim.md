# ov_deim

- **LibreYOLO module:** `libreyolo/models/openvocab/ov_deim.py` plus vendored
  architecture modules in `libreyolo/models/openvocab/ovdeim/`; the DINOv3
  backbone adapter is reused from `libreyolo/models/deimv2/`.
- **Upstream:** https://github.com/wleilei/OV-DEIM (paper: arXiv 2603.07022,
  "OV-DEIM: Real-time DETR-Style Open-Vocabulary Object Detection with
  GridSynthetic Augmentation"), pinned at commit
  `dfbf394672407b7f837ec08e7d68e8127548b254` (the commit that added the
  LICENSE files).
- **Verification status:** Phase 0 audit verified 2026-07-11 (blocked, no
  license). **Unblocked 2026-07-14**: upstream added LICENSE (Apache-2.0),
  MODEL_LICENSE (CC BY-NC 4.0) and THIRD_PARTY.md; ported per option B. See
  "Resolution" below.

## Resolution (2026-07-14)

The repository owner answered the license request
(https://github.com/wleilei/OV-DEIM/issues/4#issuecomment-4967463647)
verbatim:

> Hi, thank you for the thoughtful note and for bringing this licensing issue
> to our attention.
>
> We have now added the following files to the repository:
>
> * `LICENSE` — The OV-DEIM code is released under the Apache License 2.0.
> * `MODEL_LICENSE` — The released S/M/L checkpoints are licensed under
>   CC BY-NC 4.0. Redistribution and format conversion (e.g., ONNX, TensorRT)
>   are permitted with attribution for non-commercial use.
> * `THIRD_PARTY.md` — Documents the relationship and licenses of upstream
>   projects.
>
> To clarify the GPL/AGPL concern: OV-DEIM does not directly redistribute
> source files copied from YOLO-World (GPL-3.0) or YOLOE (AGPL-3.0). These
> projects were used as research references and architectural inspiration.
>
> The codebase includes components adapted from RT-DETR (Apache-2.0) and
> DEIMv2 (Apache-2.0). MobileCLIP components and the DINOv3 backbone remain
> under their respective original license terms, as documented in
> `THIRD_PARTY.md`.
>
> We hope this clarifies the licensing status and makes OV-DEIM easier to
> adopt and integrate. Thank you again for helping us improve the repository
> documentation.

Ship decisions taken on that basis:

- **Code (port):** the vendored modules (`ovdeim/encoder.py`,
  `ovdeim/decoder.py`, the `TextAdapter` in `ovdeim/nn.py`) are taken from
  OV-DEIM under Apache-2.0, RT-DETR headers preserved. The GPL-derived
  training dataloader identified in the Phase 0 map is **not** ported
  (inference-only v1); the vendored `dinov3/` tree is avoided by reusing
  dev's existing DEIMv2 backbone adapter.
- **Detector weights:** converted S/M/L checkpoints are rehosted on
  `LibreYOLO/LibreOVDEIM{s,m,l}` under CC BY-NC 4.0 with attribution, as the
  MODEL_LICENSE explicitly permits. The family is therefore non-commercial;
  the license notice is logged at model construction.
- **Text tower:** the online text encoder is MobileCLIP-B(LT)'s text
  transformer, weights taken unchanged from Apple's own release
  (`apple/MobileCLIP-B-LT-OpenCLIP`). The Apple Machine Learning Research
  Model license **permits redistribution** of the model and derivatives with
  a copy of the agreement, the attribution notice, and disclosure of
  modifications, but restricts use to research purposes. The weight repos
  carry the license text verbatim, the attribution notice, and identify the
  slice (text tower only, unmodified tensors) as required. This is stricter
  than CC BY-NC; the combined artifact is research-use.
- **L backbone:** the fine-tuned DINOv3-S weights inside the L checkpoint
  remain subject to Meta's DINOv3 License; the weight repo documents this,
  matching the existing DEIMv2 precedent.
- **Parity evidence:** detector outputs are bit-exact against upstream for
  all three sizes on identical inputs; the online text tower reproduces
  upstream's released embedding caches to 2.4e-7 (fp32 storage); the full
  predict pipeline matches upstream's evaluation flow on real images with
  100% label agreement across all 300 queries.

The Phase 0 audit below is retained unchanged for the record.

# Phase 0 audit (2026-07-11, superseded by the resolution above)

## License status of the upstream repository

Checked 2026-07-11 via the GitHub API and the repository tree:

- The repository has **no LICENSE file** and GitHub reports `license: null`
  (last upstream push 2026-07-01). Under copyright default this means all
  rights reserved: the code cannot be ported and the released checkpoints
  (Google Drive / Baidu) cannot be redistributed or converted.
- A license clarification issue was opened upstream on 2026-07-11:
  https://github.com/wleilei/OV-DEIM/issues/4 (asks for a code license, a
  checkpoint license, and a map of which files derive from which upstream).
  No answer yet; record the answer here when it arrives.

## Component licenses (verified against each upstream, 2026-07-11)

| Component | License | Notes |
|---|---|---|
| DEIMv2 (Intellindust-AI-Lab/DEIMv2) | Apache-2.0 | already ported on dev at `libreyolo/models/deimv2/` |
| RT-DETR (lyuwenyu/RT-DETR) | Apache-2.0 | |
| YOLO-World (AILab-CVC/YOLO-World) | GPL-3.0 | |
| YOLOE (THU-MIG/yoloe) | AGPL-3.0 | |
| DINOv3 code + weights (facebookresearch/dinov3) | DINOv3 License (Meta custom) | not Apache; redistribution carries Meta's terms |
| MobileCLIP code (apple/ml-mobileclip) | MIT | code only |
| MobileCLIP **weights** (LICENSE_MODELS) | Apple ML Research Model license | **research-only**; use, derivatives and redistribution restricted to scientific research |
| MobileCLIP training data terms (LICENSE_DATA) | CC BY-NC-ND 4.0 | non-commercial, no derivatives |

## File-level provenance map

Method: mechanical line-containment analysis over every `.py` file in the
OV-DEIM repository against shallow clones of DEIMv2, RT-DETR, YOLO-World,
YOLOE and dinov3 (fraction of a file's normalized non-trivial source lines
present in the best-matching reference file, refined with a difflib ratio),
plus a manual pass over file headers and docstrings. 146 Python files total.

Findings by directory:

- **`dinov3/` (110 files):** wholesale vendored copy of Meta's dinov3
  repository, containment 0.95 to 1.00 for nearly every file. Governed by the
  DINOv3 License regardless of what license the OV-DEIM authors adopt. A
  subset of `dinov3/dinov3/layers/` and `utils/` matches DEIMv2's own vendored
  `engine/backbone/dinov3/` copy exactly (same Meta lineage via DEIMv2).
- **`model/` (18 files):** Apache-2.0 lineage plus author-original changes.
  Backbones are near-identical to DEIMv2/RT-DETR (`hgnetv2.py` 1.00,
  `vit_tiny.py` 0.98, `presnet.py` 0.97 vs RT-DETR, `dinov3_adapter.py` 0.92),
  encoder and matcher are DEIMv2-derived (0.88 and 0.81), decoder and
  criterion are heavier author modifications of the same Apache base (0.44 to
  0.64). The open-vocab classification head `model/decoder/cls_embed.py` and
  the top-level `model/ovdeim.py` match no reference (author-original).
  Copyright headers in these files credit lyuwenyu (RT-DETR, Apache-2.0) and
  the DEIMv2 authors, consistent with the diff.
  **No file in `model/` shows meaningful similarity to YOLO-World or YOLOE
  (all containment at or below 0.03).**
- **`dataloader/` (3 files):** `transforms.py` states in docstrings that
  several classes are "adapted from" MMYolo (GPL-3.0) and that
  `MultiModalMosaic` is "a modified version of" YOLO-World's implementation
  (GPL-3.0). Line containment vs YOLO-World is low (0.06) because the code was
  rewritten, but the stated derivation makes these classes derivative works of
  GPL code. **The GPL surface is confined to this training-only dataloader**;
  it does not touch the model or inference path.
- **Training/eval scripts, configs, `optim_tools/`, `dist_tools/`:** original
  or thin RT-DETR derivations (`optim_tools/ema.py` 0.79 vs RT-DETR,
  Apache-2.0).

## Checkpoint (weights) analysis

Released S/M/L checkpoints were trained on Objects365v1 + GoldG with text
embeddings precomputed by MobileCLIP-B(LT):

1. With no upstream license, the checkpoints cannot be redistributed at all
   today. This alone blocks any HF upload.
2. Even if the authors add a permissive code license, MobileCLIP **weights**
   are research-only (Apple ML Research Model license). Shipping an online
   text tower for arbitrary prompts would mean converting and redistributing
   MobileCLIP text-encoder weights, which that license does not permit for a
   commercially usable MIT project. Whether the detector checkpoints
   themselves count as "Model Derivatives" of MobileCLIP under Apple's terms
   is an open legal question; treat as needs-check.
3. Training data terms (Objects365v1 research terms, GoldG mixture including
   Flickr30k entities): needs-check if weights ever become redistributable.

## Verdict (Phase 0 go/no-go)

- **Option A (blocked): in effect as of 2026-07-11.** No upstream license.
  Do not port code, do not redistribute or convert weights. Parked pending an
  answer to upstream issue #4.
- **Option B (port if licensed):** becomes viable for the inference-side code
  if the authors add a permissive license: the inference surface is
  Apache-lineage plus author-original code, and the GPL-derived surface is
  confined to the training dataloader, which a v1 inference port would not
  take. Two residual blockers would remain even then: the vendored `dinov3/`
  tree (avoidable, our port would reuse dev's existing DEIMv2 backbones) and
  the MobileCLIP weights license for the online text tower (not avoidable
  without substituting the text encoder and retraining, see point 2 above).
- **Option C (from-paper reimplementation with our own training):** possible
  but large (Objects365+GoldG scale training) and still needs a
  permissively-licensed text tower substitute. Maintainer decision.
