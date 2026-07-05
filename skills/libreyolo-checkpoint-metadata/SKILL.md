---
name: libreyolo-checkpoint-metadata
description: >-
  Inspect, validate, produce, and debug LibreYOLO .pt checkpoint metadata.
  Use when a checkpoint won't load ("not a LibreYOLO checkpoint", wrong
  family/task/class-count errors), when writing or reviewing a conversion
  script, when deciding what a converter must emit, when preparing weights
  for HF upload, or when a schema change is proposed. Covers the inspection
  commands, the serialization helpers that are the only sanctioned
  writer/reader, the lean-vs-training checkpoint distinction, the
  legacy/foreign-weights paths and auto-conversion, and the
  contract-change protocol. The schema itself lives in
  docs/checkpoint_schema.md; this skill is how to work with it.
---

# Work with LibreYOLO checkpoint metadata

Metadata is the model-loading source of truth (REVIEW.md axiom): a `.pt` is
a `torch.save()` dict whose top-level keys identify family, size, task,
classes, and input size, so loading never depends on filename parsing or
state-dict sniffing. The contract is `docs/checkpoint_schema.md` (v1.0
required keys, task-specific extras, training-checkpoint fields, export
runtime metadata). **Read that doc for the schema; do not trust a from-memory
key list, including this skill's examples.**

## Inspect (first move for any load problem)

```bash
libreyolo metadata path=weights/LibreYOLO9t.pt      # raw embedded metadata
libreyolo info model=LibreYOLO9t.pt                 # resolved family/size/task/nc/device
```

Python, when you need it programmatic:

```python
from libreyolo.utils.serialization import load_untrusted_torch_file, validate_checkpoint_metadata
ckpt = load_untrusted_torch_file("file.pt")   # weights_only-safe load for unknown files
validate_checkpoint_metadata(ckpt)            # raises CheckpointMetadataError with specifics
```

## Produce (converters, trainers, tools)

The helpers in `libreyolo/utils/serialization.py` are the only sanctioned
writer/reader; hand-assembled metadata dicts drift from the schema:

- `wrap_libreyolo_checkpoint(...)`: builds the v1.0 wrapper around a
  state dict. Every `weights/convert_*_weights.py` script ends in this.
- `validate_checkpoint_metadata(...)`: strict check; call it in the
  converter *after* writing and in tests.
- `normalize_checkpoint_names(...)` / `build_class_names(...)`: names-dict
  hygiene (`dict[int, str]`, keys `0..nc-1`).
- `load_untrusted_torch_file` vs `load_trusted_torch_file`: use the
  untrusted loader for anything a user hands you.

Two checkpoint shapes, one schema core:

- **Lean/published**: state dict + required metadata only. What HF repos
  ship. No optimizer, epoch, config, or EMA-resume keys.
- **Training**: adds `epoch`, `optimizer`, `config`, EMA fields
  (`is_ema_weights` declares whether `model` is the EMA weights). `best.pt`
  and `last.pt` are these. Strip to lean before publishing
  (`libreyolo-upload-hf-model` validates this).

## Debug a load failure (the decision tree)

1. **"Foreign/unrecognized checkpoint"**: it is an upstream file, not a
   LibreYOLO one. Route: the family's `weights/convert_*_weights.py`, or the
   runtime auto-converter (`libreyolo/models/autoconvert.py` + per-family
   recognizers) which handles known upstream layouts transparently. If a
   known upstream format is rejected, the recognizer is the bug.
2. **Cross-family or cross-task rejection** ("checkpoint is family X, model
   is Y"): working as designed; these must fail loudly. Never "fix" by
   loosening the check; load with the right class/task instead.
3. **Metadata invalid** (missing keys, bad `names` range, wrong
   `schema_version`): the producer is at fault; fix the converter/trainer
   and regenerate, do not hand-patch the file (a hand-patched checkpoint in
   the wild is a provenance mystery later).
4. **Legacy LibreYOLO checkpoint** (pre-v1.0, loads with a warning): the
   compatibility path works but is legacy; recommend re-converting. State-
   dict sniffing exists only on this path and must not spread to new code.
5. **Class-count surprises** (nc=90 vs 80 RF-DETR-style): read the
   "Legacy And Foreign Weights" section of the schema doc before touching
   anything; the COCO-normalization rules there are subtle, deliberate, and
   test-covered.

## Change the contract (rarely, and never quietly)

Adding/removing/reinterpreting a metadata key is a contract change.
CONTRIBUTING.md hard-requires: update `docs/checkpoint_schema.md` **and**
the helpers in `libreyolo/utils/serialization.py` in the same PR. Also:

- New required keys break every existing published weight; prefer optional
  keys with reader defaults (the `crop_pct`/`interpolation` classify keys
  are the pattern).
- Exported-runtime metadata (ONNX props etc.) is a parallel surface with
  its own section in the schema doc; exporter-writes and backend-reads must
  change together (`libreyolo-export-formats`).
- Task-specific extras (pose keypoint fields, restore `degradation`) belong
  in the task's schema section, defined when the task lands
  (`libreyolo-add-task` step 9).
- The unit tests around serialization and per-family load/reject
  (`tests/unit/test_serialization*.py`, `test_*_can_load*`-style) are the
  gate; extend them with the change.

## Related

- `docs/checkpoint_schema.md`: the contract (always read it, it moves).
- `skills/libreyolo-port-model/`: converter templates that emit v1.0.
- `skills/libreyolo-upload-hf-model/`: pre-upload validation of the file.
- `skills/libreyolo-export-formats/`: the runtime-metadata twin surface.
