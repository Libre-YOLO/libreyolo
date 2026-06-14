# ADR 0006: Depth Task Contract

## Status

Accepted.

## Context

Monocular depth estimation predicts a dense per-pixel depth map from one image.
Single-image metric depth in meters does not transfer cleanly across cameras:
focal length, sensor size, and scene scale are entangled. A model that emits
meters for one camera can silently overclaim on another camera.

RF-DETR's DINOv2 backbone is a good fit for dense relative depth because the
encoder already provides global image context and multi-scale projected
features. The depth task should still have a family-agnostic public contract so
other model families can implement it later.

## Decision

LibreYOLO defines a canonical `depth` task whose prediction primitive is
`Results.depth_map`: a dense `(H, W)` float map on the original image canvas.
Values are relative inverse depth, where higher values mean closer to the
camera. No metric unit is implied.

Training targets are plain depth maps in any dataset-consistent unit. Pixels
with `0`, negative, NaN, or inf values are invalid. The RF-DETR depth head uses
a scale-and-shift-invariant objective in inverse-depth space with trimmed
residuals and multi-scale gradient matching.

Validation aligns predictions to ground truth with a per-image positive scale
and shift in inverse-depth space. Non-positive fitted scales fall back to a
median shift so inverted predictions cannot validate as perfect. Reported
metrics are AbsRel, RMSE, and delta1/2/3; best-checkpoint fitness is delta1.

Depth checkpoints use `task: "depth"`, `nc: 1`, and
`names: {0: "depth"}` for checkpoint-schema compatibility. The class slot does
not represent a semantic class.

Export, tiled inference, tracking, TTA, augmented validation, and LoRA are
explicitly rejected until each has a depth-aware runtime contract.

## Consequences

- Depth results never fabricate boxes; `Results.boxes` is `None`.
- Metric depth requires user-side calibration against known distances.
- Hosted depth weights must be trained only on data whose license permits the
  intended redistribution and commercial use.
- Future model families can implement `depth` without redefining dataset,
  results, validator, or checkpoint behavior.
