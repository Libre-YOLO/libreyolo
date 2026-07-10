# ov_deim (pre-integration audit, family NOT shipped)

- **LibreYOLO module:** none. This is a Phase 0 licensing audit for a proposed
  port; no OV-DEIM code or weights have entered this repository.
- **Candidate upstream:** https://github.com/wleilei/OV-DEIM (paper: arXiv
  2603.07022, "OV-DEIM: Real-time DETR-Style Open-Vocabulary Object Detection
  with GridSynthetic Augmentation").
- **Verification status:** verified 2026-07-11. Decision: **blocked (option A)**,
  see verdict below.

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
