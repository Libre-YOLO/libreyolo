# ADR 0013: gazetarget task contract

## Status

Accepted.

## Context

The `gaze` task (L2CS) predicts gaze *direction* — per-face pitch/yaw
angles. Gaze-*target* estimation answers a different question: given a
person in a scene, where in the image are they looking? Its outputs are
spatial (a target point / probability heatmap on the image canvas plus an
in/out-of-frame score), its label format is GazeFollow-style (head box +
annotated target points), and its metrics are heatmap AUC and L2 distance.
None of the existing task contracts can represent that honestly, so
`gazetarget` is a separate task (the FOMO/`point` reasoning, ADR 0003).

## Decision

- Task name `gazetarget`; filename suffix `-gazetarget`; aliases
  `gaze-target`, `gaze_target`, `gaze-target-estimation`, `gazefollow`.
- Results contract: `Results.boxes` carries the per-person head boxes
  (class 0 = "person"; confidence is the head-detector score, 1.0 for
  user-supplied boxes). `Results.gazetarget` is a `GazeTargets` payload
  aligned row-by-row with the boxes:
  - `data` is `(N, 3)` — target x, target y in **original-image pixel
    coordinates** (original-canvas coordinates are canonical), and the
    in-frame probability in `[0, 1]`.
  - `heatmaps` optionally carries the raw `(N, H, W)` sigmoid probability
    grids (64x64 for PaGE). Heatmap cell centers map to the canvas by
    plain scaling: `x = (col + 0.5) / W * img_w`.
- Two-stage inference: head boxes come from the caller
  (`head_boxes=[...]`, pixel or normalized xyxy) or from any detector
  implementing the shared face-detector protocol
  (`libreyolo/models/l2cs/face.py`). Detector face boxes are expanded by
  `head_expand` (default 1.4) into head boxes, because gaze-target models
  are trained on full-head crops. The default fallback detector prefers
  **YuNet** (OpenCV `FaceDetectorYN`, 4.5.4+; far fewer false positives
  than Haar) and falls back to the offline Haar cascade when YuNet's API
  or one-time model download is unavailable. A frontal-face detector
  cannot localize people who face away from the camera; for those scenes
  supply `head_boxes=` or a person/head detector via `head_detector=`
  (a small trained head detector is the intended future default).
- JSON/summary rows add a `gaze_target: {x, y, in_frame}` entry per box.
- Launch family: `page` (LibrePAGE, PaGE arXiv:2607.04860), sizes
  s/sp/b/hp mirroring the DINOv3 tower (ViT-S / S+ / B / H+).

## Out of scope (v1)

- Training and ground-truth validation: there is no dataset-file contract
  and no `GazeTargetValidator`; `model.train()` / `model.val()` raise
  with pointers upstream. This mirrors how `gaze` shipped. The metric
  contract (GazeFollow AUC / Avg-L2 / Min-L2) is deferred until a
  permissively-licensed evaluation set is settled.
- Video-temporal smoothing of targets (per-frame independent inference
  only; `predict(video)` works frame-by-frame).
- Multi-scene batching: `SUPPORTS_BATCHED_PREDICT = False`; a scene with
  N people is one forward pass with an N-sized people axis.

## Consequences

- Every axis of the task checklist that ships is wired: tasks.py
  registration + suffix, `GazeTargets` payload with slicing/device
  moves, drawing (`draw_gaze_targets`), UI summary, CLI names
  (`page-s` ... `page-hp`), HF auto-download, checkpoint `task` metadata,
  and cross-task load rejection via the family's unique state-dict keys.
- Weight redistribution carries Meta's DINOv3 License alongside MIT (the
  towers are DINOv3 derivatives); see `libreyolo/models/page/NOTICE`.
