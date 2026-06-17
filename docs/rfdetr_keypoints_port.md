# RF-DETR keypoints: adopting the official (Roboflow v1.8.0) architecture + weights

Status: in progress · Tracking: issue #346 (keypoints/pose for YOLO9 and RF-DETR)

## Why this replaces the previous RF-DETR pose head

The previous RF-DETR pose support in LibreYOLO was a clean-room "pose-as-instance" head
(`keypoint_head = MLP(hidden, hidden, K*3, 3)`, 3 channels/keypoint) written before any official
RF-DETR keypoint model existed. It had no public pretrained weights and could only be trained from
the detection backbone.

On 2026-06-16 Roboflow shipped **official RF-DETR keypoints** (v1.8.0) under **Apache-2.0 for both
code and weights**: `rf-detr-keypoint-preview-xlarge.pth`, COCO person 17-keypoint, **71.8 AP50:95**
which their benchmark shows beating AGPL-3.0 YOLO11-pose-X (68.6) and YOLO26-pose-X (71.0) at
comparable latency. A permissively licensed, SOTA, AGPL-beating pose model is exactly the
"match then leapfrog, permissive license is the wedge" play, so LibreYOLO adopts it.

The official model is a **GroupPose-style** architecture (not a simple per-query MLP), so its weights
cannot load into the old clean-room head. Adopting the weights therefore means porting the official
keypoint architecture and **removing the old clean-room RF-DETR pose head**.

## Feasibility: the detection trunk already matches upstream 1:1

LibreYOLO's `libreyolo/models/rfdetr` is a near-verbatim port of the standard RF-DETR detection
trunk. The shared trunk is **identical in module structure, attribute names, and math** to upstream:
DINOv2 windowed backbone (`backbone.0.encoder.*`), `MultiScaleProjector` (`backbone.0.projector.*`),
`MSDeformAttn` (`sampling_offsets/attention_weights/value_proj/output_proj`), the transformer decoder
layers, `class_embed`/`bbox_embed`, and the Group-DETR `refpoint_embed`/`query_feat` packing. Standard
RF-DETR detection checkpoints already load as-is. This means the **same keypoint weights load into the
same trunk and must reproduce the same outputs** once the keypoint-specific modules are added.

## What must be added (the port surface)

1. **Dual projector** — `backbone.0.cross_attn_projector.*`, a second `MultiScaleProjector`, gated on
   `dual_projector=True` (`dual_projector_kp_only=True` in the preview config: keypoint-only branch).
2. **Keypoint head** — `keypoint_embed = MLP(hidden, hidden, 8, 3)` (8 channels/keypoint:
   x, y, findable logit, visible logit, 3 precision-Cholesky params, 1 class-logit contribution) plus
   `keypoint_head.keypoint_proj`.
3. **Keypoint decoder token stream** (per decoder layer): `kp_inst_self_attn`, `kp_inst_norm`,
   `kp_norm`, `kp_cross_attn` (MSDeformAttn), `kp_cross_attn_norm`, `kp_linear1`/`kp_linear3`,
   `kp_norm5`, `instance_kp_layer_scale`; decoder buffers `keypoint_pos_embed`, `keypoint_class_mask`;
   `ConditionalQueryInitializer` (`keypoint_query_initializer` + `_enc`); two-stage
   `enc_out_keypoint_embed`; `_kp_active_mask` schema buffer.
4. **Decode** — `xy = delta * ref_wh + ref_xy` (box-relative, no sigmoid); channels 2-7 passthrough;
   class-logit fusion adds keypoint contribution to `class_embed`.
5. **Postprocess** — keypoint topk gather, denormalize to pixels, `findable.sigmoid()` confidence,
   precision-Cholesky → pixel covariance, uncertainty trace-fusion scoring (`trace_alpha`).
6. **Loss / matcher** — area-normalized L1 + findable BCE + visible BCE + Gaussian NLL; keypoint costs
   in the Hungarian matcher.
7. **Schema / reinit** — per-class `num_keypoints_per_class` schema, `reinitialize_keypoint_head` for
   fine-tuning with a different K.

The trunk modules (backbone encoder, MSDeformAttn, transformer decoder base, queries) are **reused
unchanged** — only the keypoint additions are ported.

## Hard requirement: exact parity with upstream

The ported model must produce **the same outputs as the official `RFDETRKeypointPreview`** on the same
input, verified tensor-for-tensor (boxes, keypoints, scores) within float tolerance, plus a COCO-val
OKS spot-check reproducing ~71.8 AP50:95. Golden parity fixtures are added as tests.

## Weights + distribution

The converted checkpoint carries LibreYOLO metadata (`model_family="rfdetr"`, `task="pose"`, `nc`,
`num_keypoints`, `keypoint_dim`, `oks_sigmas`, `names`, `args`) and is published to LibreYOLO's
HuggingFace with explicit Roboflow Apache-2.0 attribution. `THIRD_PARTY_NOTICES.txt` records the
provenance.

## Licensing / hygiene

Apache-2.0 (code + weights). Retain Roboflow copyright headers on ported files and the
`THIRD_PARTY_NOTICES` entry (same posture as existing `rfdetr/**` and `ec/**`). Any text adopted from
upstream docs is scrubbed of third-party-library names per project policy.
