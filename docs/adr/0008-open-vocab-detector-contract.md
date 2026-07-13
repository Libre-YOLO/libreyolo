# ADR 0008: Open-Vocabulary Detector Contract

- Status: Proposed
- Date: 2026-07-01
- Scope: New model tier for discriminative open-vocabulary detectors

## Context

LibreYOLO already has two open-vocabulary surfaces:

- `LibreCLIP`, for zero-shot image classification.
- `LibreVLM`, for generative vision-language models used as detectors.

Grounding DINO, OWLv2, and OMDet-Turbo are different from both. They are
purpose-built object detectors conditioned on text labels. They return boxes
with real model scores, but they load as multi-file Hugging Face snapshots
rather than single-file LibreYOLO checkpoints.

They also do not belong in the generic VLM tier. VLM models generate text and
the VLM wrapper parses that text into boxes. Open-vocabulary detector families
expose detector heads and processor postprocessing functions.

## Decision

Add a separate `LibreOpenVocab` tier:

- The base class is `LibreOpenVocabDetector(BaseModel)`.
- It does not define `can_load`, so it stays out of `BaseModel._registry` and
  the `LibreYOLO(...)` checkpoint factory.
- It downloads Hugging Face snapshots into `weights/<FILENAME_PREFIX><size>/`,
  using LibreYOLO-hosted mirror repositories whose cards attribute the original
  upstream source and license.
- It exposes `set_classes([...])` as the sticky open-vocabulary class list.
- It returns standard detection `Results` with `boxes`, `scores`, and `classes`.

The first families are:

- `LibreGroundingDINO`, backed by `GroundingDinoForObjectDetection`.
- `LibreOWLv2`, backed by `Owlv2ForObjectDetection`.
- `LibreOMDetTurbo`, backed by `OmDetTurboForObjectDetection`. This is the
  real-time member of the tier; it decouples class embeddings from a task
  prompt, so post-processing returns labels that map directly to the queried
  classes and it runs its own NMS. It does not accept `text_threshold=`.

## Public API

```python
from libreyolo import LibreOpenVocab

model = LibreOpenVocab("grounding-dino-tiny")
model.set_classes(["person", "dog", "remote control"])
results = model.predict("image.jpg")
```

The default vocabulary is COCO-80, matching the detector tier. Calling
`set_classes()` replaces it until called again.

`conf=` maps to the model's box-score threshold. Grounding DINO also accepts
`text_threshold=` on prediction calls; OWLv2 and OMDet-Turbo do not. OMDet-Turbo
uses its processor's fixed NMS threshold of `0.5`, so `iou=` is ignored.

## Validation

`val()` raises in v1. The standard detection validator calls `_forward(images)`
with a stacked image tensor, while this tier's `_forward()` needs a
text-conditioned Hugging Face input payload. A future
`OpenVocabDetectionValidator` should subclass the existing detection validator
and override only the incompatible inference/data path, then reuse LibreYOLO's
existing metric assembly.

## Out Of Scope

- Training and fine-tuning.
- Export to ONNX or other runtimes.
- Custom validation in v1.
- OWLv2 image-guided detection.
- True batched forward speedups.
- Tracking.
- CLI alias resolution.

## Licensing

LibreYOLO consumes the Apache-2.0 `transformers` implementations and downloads
Apache-2.0 model weights from LibreYOLO-owned Hugging Face mirror repositories.
Those repos preserve the upstream snapshot files needed by `transformers`, add
LibreYOLO-specific README/LICENSE/NOTICE files, and attribute the original
upstream model repos. LibreYOLO does not vendor upstream source code.
