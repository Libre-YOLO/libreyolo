# ADR 0007: LibreSAM Contract For Promptable Segmentation

- Status: Proposed
- Date: 2026-06-14
- Scope: New model tier (promptable segmentation models — the SAM family)

## Context

LibreYOLO has two model entry points:

- `LibreYOLO(...)` — a weight-sniffing factory over faithful detectors. Every
  member runs one **promptless** forward and returns *all* objects with
  calibrated scores; members register via `can_load` into `BaseModel._registry`.
  `LibreEC` proves masks (`segment` task) live happily here, because its masks
  are produced automatically with no prompt.
- `LibreVLM(...)` — a parallel tier (ADR 0002) for generative open-vocabulary
  detectors. It is separated by **contract fidelity**, not architecture, and
  loads through the permissive `transformers` model API so LibreYOLO ships no
  model source and stays MIT.

Promptable segmentation (the SAM family) fits neither:

- It is **promptable**: a forward is meaningless without a per-image *spatial*
  prompt (a point or box) supplied at call time. "A detection" becomes "the
  thing you pointed at", not "everything found".
- It is **interactive/stateful**: the heavy image encoder runs once
  (`set_image`), then many cheap prompts reuse the cached embedding.
- Its output is **masks**, and its scores are real mask-quality (predicted-IoU)
  values, not detection confidences.

Forcing this through the promptless `InferenceRunner` (preprocess → forward →
postprocess → NMS) would misrepresent the call shape. So, as with LibreVLM, the
line is drawn on contract, and the tier owns its own `predict` surface.

## Decision

Add a third tier, `LibreSAM`, for promptable segmentation. It mirrors LibreVLM's
shape:

- A base class `LibreSAMModel(BaseModel)` that does **not** define `can_load`,
  keeping the family out of the detector `_registry` and the `LibreYOLO`
  factory.
- SAM-1 and SAM-2 load through the permissive `transformers` APIs and ship no
  model source. MobileSAM uses a native Apache-2.0 port because its TinyViT
  image encoder is not representable as a `transformers` SAM-1/2 checkpoint.
- Returns the same `Results` (with `masks`, plus tight `boxes` derived from the
  masks via `masks_to_boxes`, class id `0` = `"object"`), so downstream code is
  unchanged.

The default family remains **SAM-1** (`facebook/sam-vit-base` / `-large` /
`-huge`), autodownloaded on first use.

| Family | API entry | Weight source | Notes |
|---|---|---|---|
| SAM-1 | `LibreSAM("base")`, `LibreSAM1("base")` | `facebook/sam-vit-*` | Default promptable family. |
| SAM-2 image | `LibreSAM("sam2-tiny")`, `LibreSAM2("tiny")` | `facebook/sam2.1-hiera-*` | Image segmentation only in v1. |
| MobileSAM | `LibreSAM("mobilesam")`, `LibreMobileSAM()` | `LibreYOLO/LibreMobileSAM` | Native TinyViT port with converted weights. |

## Public API

The surface mirrors the de-facto-standard promptable interface (sourced from
public documentation, clean-room), expressed with LibreYOLO's own loading idiom
(size aliases + autodownload, as LibreVLM does — not checkpoint-filename
dispatch):

```python
from libreyolo import LibreSAM

model = LibreSAM("base")                                   # autodownloads (Apache-2.0)
model.predict("img.jpg", points=[900, 370], labels=[1])    # point  -> mask
model.predict("img.jpg", bboxes=[100, 100, 200, 200])      # box    -> mask
model.predict("img.jpg")                                   # segment everything (grid AMG)

model.set_image("img.jpg")                                 # encode once...
a = model.predict(points=[500, 375], labels=[1])           # ...prompt cheaply
b = model.predict(bboxes=[100, 100, 200, 200])
model.reset_image()

r.masks.xy        # polygons
r.boxes.xyxy      # tight boxes derived from masks
```

- Points/boxes accept the documented flexible nesting (`[x, y]` = one object;
  `[[x, y], ...]` = N objects; `[[[x, y], ...], ...]` = grouped per object), and
  numpy arrays. Labels are `1` positive / `0` negative, default all positive.
- `multimask=True` returns *all* of SAM's ambiguity masks per prompt (whole vs
  part); the default returns the single best by predicted IoU.
- `conf` filters by predicted mask-IoU (mask quality, **not** a detection
  confidence). `None` keeps all in the prompted path and applies the family grid
  threshold in "segment everything"; `0.0` disables filtering in either mode.
- `device=` on `predict` moves the model and invalidates the cached embedding.

## Internal Contract

`LibreSAMModel` satisfies `BaseModel`'s abstract hooks but overrides `predict()`
/ `__call__` directly rather than driving `InferenceRunner` — the promptless
preprocess/forward/postprocess hooks have no meaning here and raise. The
encode-once lifecycle lives in `set_image()` (caches image embeddings) and is
reused by every later `predict()` until `reset_image()`. A `device=` switch moves
cached embeddings when possible so interactive sessions survive device changes.

| Field             | Meaning                                              |
|-------------------|------------------------------------------------------|
| `FAMILY`          | family id (`sam`, `sam2`, `mobilesam`)               |
| `FILENAME_PREFIX` | `Libre`-prefixed weights-dir prefix                  |
| `HF_REPOS`        | `{size: hf_repo_id}`; drives autodownload            |
| `INPUT_SIZES`     | `{size: nominal_px}` (1024; the processor owns resize)|

## Confidence

Returned `conf` is SAM's predicted mask-IoU (mask quality), surfaced honestly as
a soft score, not a calibrated detection confidence. `val()` (mAP) is
unsupported — promptable masks have no fixed class set to score against.

## Licensing

SAM-1 and SAM-2 code and weights are Apache-2.0 and are loaded from their
upstream Hugging Face repositories. MobileSAM code and weights are Apache-2.0;
LibreYOLO carries a native port plus a NOTICE, and the converted checkpoint is
hosted separately as `LibreMobileSAM.pt`.

SAM-3's custom "SAM License" is gated (download requires accepting Meta's
terms) and would follow the existing LibreVLM license-notice pattern when added;
the tier must not vendor SAM-licensed modeling code.

## Out Of Scope (v1)

- SAM-2 video/memory and SAM-3 (concept/open-vocab seg, gated). Both slot onto
  this same tier later.
- Mask prompts (`masks=`), `train()`, `val()`, `export()`, and `track()` raise.
- "Segment everything" is a simplified grid AMG (predicted-IoU threshold +
  box-IoU dedup); it omits stability-score filtering, multi-crop, and mask-IoU
  dedup, and is documented as approximate. The prompted path is the precise API.

## Consequences

### Positive

- Promptable, interactive segmentation behind a familiar predict surface and the
  standard `Results`.
- No change to the detector factory; the family is fully isolated.
- A new SAM variant is a small adapter (repos, sizes).

### Negative

- `BaseModel`'s abstract surface is detector-shaped, so SAM stubs four unused
  hooks (the same tax LibreVLM pays). A future slim `transformers`-backed
  intermediate base could de-duplicate the weight-acquisition/dtype logic the
  two tiers now share.
- The simplified AMG under-segments crowded scenes versus the reference
  generator.
