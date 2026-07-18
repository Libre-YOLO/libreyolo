# ADR 0012: Unified Multimodal Families In The LibreVLM Tier

- Status: Proposed
- Date: 2026-07-17
- Scope: First multi-task generative family (SenseNova-Vision) and the tier
  contract changes it introduces

## Context

ADR 0002 established the LibreVLM tier for generative models used as
open-vocabulary detectors, and LocateAnything already stretched it to two
tasks (`detect`, `point`). A new class of unified multimodal models goes much
further: SenseNova-Vision (SenseTime, arXiv:2607.06560) casts vision as
prompted generation on a Bagel-MoT backbone, with no task heads at all.
Symbolic outputs (boxes, points, keypoints, OCR words, camera poses) are
generated as tagged text; dense outputs (depth maps, binary and panoptic
segmentation masks, surface normals) are generated as images that a frozen
VAE decodes. One checkpoint covers more than a dozen tasks with competitive
accuracy (56.6 COCO-Com mAP, 4.0 AbsRel NYUv2 depth, 81.3 cIoU RefCOCO).

The question was whether this needs a new tier, new task names, or new result
types. It needs none of them: the capability set maps onto tasks LibreYOLO
already defines, with the same `Results` payloads specialist families return.

## Decision

1. **Unified multimodal models join the LibreVLM tier as multi-task
   families.** `LibreVLM("sensenova-vision", task="depth")` loads the same
   class as `task="detect"`; `SUPPORTED_TASKS` declares what the family
   serves. No new tier, no parallel task vocabulary.

2. **Tasks keep their canonical LibreYOLO names and Results payloads.**
   SenseNova-Vision v1 serves `detect`, `point`, `pose`, `ocr`, `depth`,
   `segment` (referring segmentation: the "class" list is free text such as
   "person furthest to the right"), and `panoptic`. Each returns exactly the
   payload the specialist family for that task returns (`Boxes`, `Points`,
   `Keypoints`, `OCRRegions`, `DepthMap`, `Masks`, `PanopticSegmentation`),
   so swapping a specialist for the generalist is a one-line change.

3. **`set_task()` joins the tier base.** Prompt-driven families serve every
   task from one set of weights, so the task is sticky state like the
   vocabulary, switchable without reloading. Checkpoint families outside the
   tier keep task fixed at load time, unchanged.

4. **Capabilities without a canonical task stay behind the raw escape
   hatches.** `chat()` (visual QA) and `generate()` (any upstream mode:
   surface normals, grounded-conversation segmentation, editing, generation)
   expose the full model. Surface normals, 3D reconstruction, and camera pose
   get public task names only when a second family or real demand justifies
   the shared surface.

5. **The architecture is vendored, not remote-loaded.** Unlike previous VLM
   families, the model is not loadable through transformers and its Hugging
   Face repo carries no remote code, so the Apache-2.0 implementation is
   ported into `libreyolo/models/sensenova/modeling/` with per-file
   provenance (Bagel/ByteDance, Qwen2 and SigLIP/Hugging Face, FLUX
   autoencoder/Black Forest Labs). The port is inference-only, and flash-attn
   is optional behind a numerically equivalent SDPA fallback. The one
   CC BY-NC upstream file (DiT-derived helpers) is not ported; its standard
   components are re-derived from the original permissive sources
   (transformers ViT-MAE, openai/guided-diffusion) in `modeling/layers.py`.

6. **Weights are mirrored under their original license.** The checkpoint is
   CC BY-NC 4.0 (non-commercial); by maintainer decision (2026-07-18,
   following the OV-DEIM precedent) LibreYOLO hosts a byte-identical,
   attribution-carrying mirror at `LibreYOLO/SenseNovaVision7b` with the
   revision and SHA-256 pins on the card. Mirroring does not change the
   license, and the loader prints the non-commercial notice before every
   automatic download.

## Consequences

- The tier's honest-confidence rules extend unchanged: generated outputs
  carry the placeholder score, `val()` stays unsupported, and `track()`
  degrades as documented in ADR 0002.
- Dense tasks pay one diffusion decode (50 denoising steps) per image; this
  is a capability model, not a real-time one. `dtype="auto"` picks bf16 on
  large GPUs and NF4 quantization (bitsandbytes) on consumer GPUs; the
  14.7B-parameter checkpoint is ~29.6 GB in bf16 and ~9 GB in NF4.
- Referring segmentation arrives as a `segment`-task capability with one mask
  per query rather than per-instance masks; the contract is documented on the
  family.
- Panoptic output is open-vocabulary: phrases the model returns beyond the
  configured category list register new entries in `Results.names` instead of
  being dropped.
- The vendored port adds ~5k lines under `models/sensenova/`. Structural
  parity with the released checkpoint was verified tensor-by-tensor (1223
  names and shapes) against the safetensors header.
