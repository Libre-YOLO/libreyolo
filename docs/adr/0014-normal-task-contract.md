# ADR 0014: Surface-normal task contract

Status: accepted
Date: 2026-07-29

Numbering note: the repository currently contains two ADRs numbered 0013
(`embed` and `mesh`). This ADR uses 0014 as planned, but the duplicate 0013
still needs a maintainer numbering decision.

## Context

Surface-normal estimation answers, per image pixel, which direction the
visible surface faces. It is the orientation half of single-image geometry,
complementary to depth's distance field. It needs its own task because the
payload, coordinate convention, and angular validation metrics are different
from depth.

Existing normal estimators do not share a public convention. Arrays with the
same `(H, W, 3)` shape can differ in axis direction, camera frame, vector
orientation, value encoding, and resolution. Treating a colorful normal PNG
as the contract makes consumers infer those choices from a visualization.

## Decision

### One public convention

Add the canonical task `normal`, filename suffix `-normal`, and aliases
`normals`, `surface-normal`, `surface_normal`, `surface-normals`, and
`surface_normals`.

`Results.normal_map` contains float32 data with shape `(H, W, 3)` in `[-1, 1]`
on the original image canvas at the original resolution. Every pixel is a
unit vector in the OpenCV camera frame:

- `+x` points right in the image;
- `+y` points down in the image;
- `+z` points into the scene;
- normals face the camera, so `n . ray < 0` for each visible surface;
- a fronto-parallel wall facing the viewer is `(0, 0, -1)`.

Each model family converts axes and orientation at its own output boundary,
then resizes to the original canvas and renormalizes after interpolation.
Per-family meanings are ruled out because they would make downstream geometry
depend on checkpoint identity.

### Float vectors are the payload

The stored payload is the vector field, not an RGB image. Plotting and CLI
saving render `rgb = (normal + 1) / 2`; that mapping is only a visualization.
Storing PNG values is ruled out because it loses the semantic distinction
between vector data and rendering, introduces quantization, and encourages
code to consume colors as geometry.

### This is a task, not a depth post-process

Native surface-normal predictions populate `Results.normal_map` and validate
with angular error. Normals derived from a depth gradient are a different
quality tier and must not silently populate this payload. Treating normals as
a depth plotting option is ruled out because it erases their distinct learned
output, convention conversion, validity mask, and metrics.

### Validation data

Ground truth uses same-stem three-channel `uint16` RGB PNGs. It decodes as
`normal = png / 65535 * 2 - 1` and is then renormalized. An optional
single-channel mask uses nonzero pixels as valid. Bilinear target or prediction
resizing is always followed by vector renormalization.

Validation reports mean and median angular error in degrees and percentages
within 11.25, 22.5, and 30 degrees over valid ground-truth pixels.

## Consequences

Dense-result plumbing, visualization, dataset decoding, and validation can
exist independently of a model family. A future family has one stable boundary
to target, and exported runtimes must emit the same convention rather than an
upstream-native one.

Three channels at float32 cost more than an RGB preview, but retain the
precision required for geometry. Bilinear resizing also costs a normalization
pass; omitting it would violate the unit-vector invariant.

The first family and exported-runtime contract remain deferred until both
source code and pretrained weights pass LibreYOLO's licensing audit. A
permissive source license alone is insufficient when weight terms or training
data impose restrictions.

## Non-goals

- Training a normal estimator is not part of the v1 contract.
- World, gravity-aligned, or object-local coordinate frames are not implied.
- Depth-gradient normals do not become native normal predictions.
- RGB previews are not accepted as floating-point result payloads.
- No model family, checkpoint, or export format is approved by this ADR.
