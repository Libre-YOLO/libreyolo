# ADR 0011: Anomaly Task Contract

## Status

Accepted.

## Context

One-class visual anomaly detection does not emit object instances. A model is
fitted for one product or scene using normal images, then produces an
image-level deviation score and a dense localization heatmap.

## Decision

LibreYOLO defines the canonical `anomaly` task. A result exposes:

- `anomaly_score`: a float image score, higher means more anomalous;
- `anomaly_map`: a float `(H, W)` map on the original image canvas;
- `is_anomalous`: the score compared with the checkpoint threshold, or `None`
  when no calibrated threshold is available.

Anomaly checkpoints use `task: "anomaly"`, `nc: 1`, and
`names: {0: "anomaly"}` for checkpoint-schema compatibility. Training is a
family-defined fit operation and need not use gradients or epochs.

The dataset contract follows the category-local `train/good`, `test/good`,
`test/<defect>`, and optional `ground_truth/<defect>` directory layout.
Validation reports image AUROC and maximum F1, plus pixel equivalents when
masks exist.

Tracking, tiled inference, test-time augmentation, and export are rejected
until anomaly-specific contracts exist for them.

## Consequences

- Results do not fabricate boxes.
- A fitted checkpoint is category-specific, not a universal detector.
- Thresholds are checkpoint metadata and may be overridden through the public
  prediction threshold argument.
- Benchmark datasets and derived artifacts remain subject to their own data
  licenses and are never implicitly redistributed.
