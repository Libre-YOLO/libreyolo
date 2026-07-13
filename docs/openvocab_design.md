# LibreOpenVocab Design Decisions

This document records the contract for discriminative open-vocabulary
detectors in LibreYOLO. The formal ADR is
[`adr/0008-open-vocab-detector-contract.md`](adr/0008-open-vocab-detector-contract.md).

## What LibreOpenVocab is

`LibreOpenVocab` loads purpose-built text-conditioned detectors from Hugging
Face and adapts their outputs to LibreYOLO `Results`.

These models are not generic VLMs. They do not generate text that LibreYOLO
parses. Instead, their processors and detector heads return boxes, scores, and
labels directly.

## Available models

| Aliases | Family | Default size | License |
|---|---|---|---|
| `grounding-dino`, `grounding-dino-tiny`, `grounding-dino-base` | Grounding DINO | tiny | Apache-2.0 |
| `owlv2`, `owlv2-base`, `owlv2-large` | OWLv2 | base-patch16 ensemble | Apache-2.0 |
| `omdet-turbo`, `omdet`, `omdet-turbo-swin-tiny` | OMDet-Turbo | Swin-T | Apache-2.0 |

The authoritative alias table is `_ALIASES` in
`libreyolo/models/openvocab/__init__.py`.

OMDet-Turbo is the real-time member of the tier. It is an RT-DETR-based
open-vocabulary detector that decouples class embeddings from a task prompt and
runs its own NMS in post-processing. Its post-processing returns labels that
map directly back to the queried class list (no phrase disambiguation like
Grounding DINO). It does not expose `text_threshold=`.

Public `LibreOMDetTurbo.predict()` calls the maintained `transformers`
implementation, including its processor and post-processing. The
parity-verified native `OmDetTurboDetectionModel` under
`libreyolo/models/omdet_turbo/` is groundwork for a possible future
fixed-vocabulary export path; it is not called by `predict()`. Export remains
unsupported under ADR 0008 and would require a separate contract change plus
tensor, decoded-detection, and metric parity verification.

Weights are loaded from LibreYOLO-owned Hugging Face mirror repositories:

- `LibreYOLO/LibreGroundingDINOt`
- `LibreYOLO/LibreGroundingDINOb`
- `LibreYOLO/LibreOWLv2b16`
- `LibreYOLO/LibreOWLv2l14`
- `LibreYOLO/LibreOMDetTurbot`

The model cards in those repos record the original upstream source repos.

## Class Vocabulary

The vocabulary is sticky:

```python
model = LibreOpenVocab("owlv2")
model.set_classes(["cat", "dog", "remote control"])
model.predict("a.jpg")
model.predict("b.jpg")  # same class list
```

If `set_classes()` is never called, the vocabulary defaults to COCO-80 labels.

`set_classes()` accepts a list or tuple of label strings. It rejects a bare
string, empty lists, blank labels, and case-insensitive duplicates.

## Thresholds

`conf=` is the box-score threshold for all three families.

Grounding DINO also has a text-token threshold. LibreYOLO exposes it as
`text_threshold=` on prediction calls:

```python
model.predict("image.jpg", conf=0.25, text_threshold=0.25)
```

OWLv2 and OMDet-Turbo do not have this threshold and reject `text_threshold=`.

## Grounding DINO Phrase Mapping

Grounding DINO postprocessing returns decoded text phrases. LibreYOLO maps
those phrases back to the current `set_classes()` vocabulary.

Grounding DINO's text encoder has a shorter model limit than the tokenizer's
nominal maximum. LibreYOLO chunks long vocabularies, including the default
COCO-80 vocabulary, into multiple text prompts that fit `max_text_len`, runs one
forward per chunk, and merges the detections before applying the global
`max_det` cap.

The mapping is deliberately conservative:

- Exact normalized match wins.
- Otherwise, whole-token containment is allowed.
- Raw substring matching is not allowed.
- Ambiguous or unmatched phrases are dropped.

This avoids mistakes such as mapping `carpet` to `car`, or choosing between
`bus` and `school` for the phrase `school bus`.

## Validation

`val()` raises in v1. These models need a text-conditioned processor payload in
`_forward()`, but LibreYOLO's standard detection validator calls `_forward()`
with an image tensor batch.

A future validator should set the model vocabulary from the dataset names, run
the Hugging Face processor path per image, and then reuse LibreYOLO's existing
detection metric machinery.

## Weight Loading

Weights are downloaded on construction into:

```text
weights/<FILENAME_PREFIX><size>/
```

The snapshot completeness check validates the marker, `config.json`, and either
a single weight file or every shard listed by a safetensors/bin index. The
download uses `local_dir` so Windows does not depend on Hugging Face cache
symlinks.
