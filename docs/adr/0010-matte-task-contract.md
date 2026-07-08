# ADR 0010: Matte Task Contract

## Status

Accepted.

## Context

Background removal (product cutouts, portraits, asset extraction) is one of the
most-used vision operations in the wild and is absent from LibreYOLO's task
list. The model output that serves it best is a soft alpha matte: a per-pixel
foreground opacity in `[0, 1]` that keeps the anti-aliased edges (hair, fur,
motion blur) that a binary mask discards. This is also the output kind used for
dichotomous image segmentation (DIS) and salient-object detection.

No existing task's contract fits it honestly. `segment` and `semantic` produce
classed, hard masks (instances or class IDs); a class-free soft matte would make
their labels and metrics lies. `restore` is image-in/image-out. A soft matte is
a distinct output primitive, so a new task is warranted (the same test that
promoted `point` in ADR 0003 and `depth` in ADR 0006).

BiRefNet (MIT, code and the general weights) is the launch family. It is a
single-purpose model, which is the accepted exception to "new features cover the
flagships": matte is carried by a dedicated family (`birefnet`), like `gaze`
(L2CS) and `restore` (NAFNet).

## Decision

LibreYOLO defines a canonical `matte` task (suffix `-matte`; aliases `matting`,
`background-removal`, `rembg`, `dis`). Its prediction primitive is
`Results.matte`: a dense `(H, W)` float32 map in `[0, 1]` on the original image
canvas, where `1` is fully foreground (opaque) and `0` is fully background
(transparent). A soft matte subsumes binary background removal (threshold at
0.5).

Convenience surface on `Results`:

- `results.cutout(image=None)` returns an RGBA `(H, W, 4)` uint8 array: source
  RGB plus the matte as the alpha channel. The RGB is taken from the argument
  or reloaded from `results.path`.
- `results.save(path)` writes a transparent-background RGBA PNG cutout, the
  canonical background-removal deliverable.
- `Results.plot()` / CLI `--save` render a checkerboard preview so the
  transparency and soft edges are visible.

The matte is produced at the family's fixed native resolution, then resized back
to the original canvas with bilinear interpolation (not nearest, which gives
jagged hair edges) and clamped to `[0, 1]`.

Validation runs on paired image/matte folders (see `docs/dataset_schema.md`) and
reports two resolution-independent metrics, both standard for DIS/SOD:

- **MAE** (mean absolute error), lower is better.
- **S-measure** (structure measure, Fan et al., ICCV 2017), higher is better;
  best-checkpoint fitness is S-measure.

Matte checkpoints use `task: "matte"`, `nc: 1`, and `names: {0: "matte"}` for
checkpoint-schema compatibility. The class slot does not represent a semantic
class; cross-task loads (loading a `-matte` checkpoint as `detect`) fail loudly.

ONNX export uses a fixed-resolution contract (the native square). The exported
graph emits a single-channel logit map named `matte`; apply sigmoid downstream.

Out of scope in v1: training / fine-tuning (`model.train()` raises with a
pointer to the paired-data schema and the upstream MIT training code), tiled
inference, tracking, TTA, and augmented validation.

## Consequences

- Matte results never fabricate boxes; `Results.boxes` is `None`.
- `Results.save()` writes a transparent PNG for matte results and raises for
  other tasks (which use the annotated-image save path).
- New model families can implement `matte` without redefining the Results
  payload, validator, dataset layout, or checkpoint behavior.
- Hosted matte weights must be trained only on data whose license permits the
  intended redistribution; the launch weights host the upstream author's MIT
  checkpoints and state training-data provenance on the model card.
