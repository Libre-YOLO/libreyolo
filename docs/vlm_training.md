# VLM detection fine-tuning

Fine-tune a LibreVLM detector on a standard detect dataset. First trainable
cohort: Qwen3-VL 2B and 4B. Everything below is the shipped contract; the wider
design rationale lives in `docs/adr/0002-librevlm-contract.md` (tier contract)
and the training ADR draft.

## Use

```python
from libreyolo import LibreVLM

model = LibreVLM("qwen3-vl-2b")
results = model.train(data="strawberries.yaml", epochs=10)

model = LibreVLM(results["best"])       # load the fine-tune
model.predict("field.jpg")              # vocabulary comes from the dataset
```

The same verified path is available from the CLI:

```bash
libreyolo train --model qwen3-vl-2b --data strawberries.yaml
libreyolo predict --model runs/vlm/train/weights/best --source field.jpg
```

Prediction accepts a VLM alias, a local VLM checkpoint directory, or an
immutable `hf+vlm://owner/repo@<commit>` publication URI. Training starts from
a verified base alias; remote publication artifacts are inference-only. To
continue a LoRA adapter, keep the base alias and pass the local checkpoint
through `--resume`:

```bash
libreyolo train --model qwen3-vl-2b --data strawberries.yaml \
  --resume runs/vlm/train/weights/last
```

Training an inference-loaded checkpoint is rejected because its adapter is
already merged. The explicit base-plus-resume form keeps the saved adapter
relative to the base recorded in its contract. For zero-shot prediction,
`--names` sets the open vocabulary before inference:

```bash
libreyolo predict --model qwen3-vl-4b --source field.jpg \
  --names '["ripe strawberry", "person wearing a helmet"]'
```

`--names` is VLM-only. `--classes` remains the numeric class-id filter applied
to model output. Explicit prediction `--imgsz` is rejected because the family
processor owns image resizing.

Needs the optional extra:

```
pip install "libreyolo[vlm-train]"
```

The v1 training and publication writer contract pins `peft==0.19.1` and
`transformers==5.12.1`; training preflight rejects other writer versions. The
broader `libreyolo[vlm]` dependency range applies to inference, not artifact
creation.

- `data` is the standard detect dataset (`docs/dataset_schema.md`): YAML with
  `train`/`val` image sources and either normalized txt label rows or native
  COCO JSON annotations. COCO category names must match the YAML vocabulary;
  crowd and ignored annotations are excluded from training targets.
- The `names` in the YAML become the training vocabulary. Names may be any
  phrases ("ripe strawberry", "person wearing a helmet"); they are rendered
  into the model's own grounding prompt and answer format automatically, using
  the same `BBOX_KEY` / `COORD_DIVISOR` / `BOX_FORMAT` convention the family's
  `predict()` parser uses. Train and predict cannot drift apart, by
  construction.
- `lora=True` is the default and the recommended path. The recipe (rank 16,
  alpha 32, language-model attention and MLP projections, vision tower frozen)
  is fixed per family in `libreyolo/models/vlm/training/recipes.py`, not a
  user-facing knob. `lora=False` full fine-tuning exists behind a VRAM
  preflight.
- Defaults reflect VLM reality: `batch=1`, `accumulate=8` (effective batch 8),
  gradient checkpointing on, cosine schedule with warmup, bf16 autocast on
  CUDA, `epochs=10`, and output under `runs/vlm/train`. There is no `imgsz`;
  the family's processor owns resolution. These defaults apply to the CLI too;
  detector-command defaults are not forwarded into the VLM trainer.
- CUDA fine-tuning requires a BF16-capable GPU. The trainer rejects unscaled
  FP16 rather than silently running an unstable optimization path. `device`
  accepts the standard LibreYOLO forms (`auto`, `cpu`, `mps`, `0`, `cuda:0`),
  and the wrapper remains on the selected training device afterward.
- Augmentation is geometric-safe only (`hflip=0.5`), because every geometric
  transform re-renders the target text. Mosaic-style compositing does not
  apply to generative targets.
- `resume=` continues from a prior adapter checkpoint (weights only; the
  optimizer state starts fresh, and the log says so). Resume validates the
  adapter payload plus family, size, base revision, prompt, and box convention
  before loading the base model.
- `callbacks=` and `loggers=` are the standard training layers; TensorBoard,
  MLflow, and W&B loggers work unchanged. The generic Hugging Face Hub logger
  is rejected because it only publishes detector `.pt` files, while VLM
  checkpoints are directories with a different reload and licensing contract.

The CLI accepts the VLM trainer's small option set and rejects explicitly
requested detector-only optimizer, scheduler, image-size, and augmentation
options before loading model weights. Use `--dry-run` to inspect the resolved
VLM configuration without downloading the base model. Training is verified
only for Qwen3-VL 2B and 4B; other families and Qwen3-VL 8B fail before load.
The Python `train()` surface also rejects unknown keywords instead of ignoring
typos or detector-only options. Dataset YAML, class names, local train/val
sources, resume identity, adapter payloads, and optional dependencies are
preflighted before base weights are loaded where the reference permits it.

## Local checkpoints

A VLM checkpoint is a directory, not a `.pt`:

```
runs/vlm/train/weights/best/
  adapter_model.safetensors    # LoRA adapter (megabytes)
  adapter_config.json
  ...processor files...
  libreyolo_vlm.json           # the contract file
```

`libreyolo_vlm.json` records family, size, the exact base repo and revision
the adapter was trained on, the ordered vocabulary, the coordinate convention,
and the run metrics. `LibreVLM(path)` recognizes the contract, downloads the
recorded base and immutable revision, merges the adapter for
inference speed, and pre-applies the saved vocabulary, prompt, and coordinate
convention. An explicit `prompt=` may override the saved prompt, while parsing
continues to use the checkpoint's saved box key, coordinate divisor, and box
layout. LoRA checkpoints reference their pinned base and do not copy its
weights.

Current Qwen checkpoints record an immutable upstream revision. Older schema-1
checkpoints may contain a null `base_revision`; they remain loadable, but cannot
prove bit-for-bit base-model reproducibility and resolve the recorded repository
without inventing a newer pin.

Full fine-tunes (`lora=False`) save a self-contained model directory instead;
the same `LibreVLM(path)` call loads either kind. Unlike a LoRA checkpoint, a
full fine-tune necessarily contains model weights and is not accepted by the
v1 Hub artifact builder.
Each `best` or `last` write is staged and replaces the complete prior directory,
so switching between LoRA and full-model runs cannot leave stale files that
change how the checkpoint is interpreted.

The inherited generic `save()` and `push_to_hub()` methods are deliberately
disabled for LibreVLM. They would create a monolithic detector-style `.pt`
artifact that the LibreVLM factory cannot reload and that lacks the required
dataset, benchmark, and upstream-license evidence. The generic Hugging Face
logger is disabled for the same reason.

## Reviewed publication artifacts

A local `best` or `last` directory is not automatically publishable. For a
Qwen3-VL 2B/4B LoRA checkpoint, the strict publication workflow is:

1. Generate a create-only, unapproved evidence template with
   `create_vlm_publication_evidence_template()`.
2. Have a human review the bound data, license, privacy, evaluation, code, base,
   adapter, contract, and processor evidence. The library never creates an
   approval.
3. Build and validate `libreyolo.vlm-artifact.v1` with
   `build_vlm_artifact()` and `validate_vlm_artifact()`.
4. Explicitly upload it with `push_vlm_artifact()`, which returns an immutable
   `hf+vlm://` URI.

Base weights remain reference-only. The artifact does include the exact Qwen
processor, tokenizer, and chat-template assets under Apache-2.0, with generated
license and notice files. Evidence hashes bind reviewed bytes and reports for
integrity; they do not authenticate the reviewer or prove that a report is
true. Full details and Python examples are in
[`vlm_hub_artifact.md`](vlm_hub_artifact.md).

## Best/last selection

`best` tracks validation loss when the dataset has a `val` split, else
training loss. CLI results report that metric by name together with the
`best` and `last` checkpoint directories; they do not report detection mAP.
Only `predict` and verified `train` are VLM CLI execution surfaces. Other
model-execution commands, including `val`, `export`, `quantize`, `info`,
`compare`, and face-detector roles, fail before loading weights. Validation mAP
remains blocked on real per-box confidence; see the ADR.

## Local benchmark preparation

The internal confidence gate has a no-download COCO metadata builder:

```text
python -m libreyolo.validation.vlm_benchmark_dataset build \
  --annotations /data/coco/annotations/instances_val2017.json \
  --images-dir /data/coco/val2017 \
  --output-root /data/libreyolo-vlm-benchmark
```

It accepts only the pinned COCO val2017 annotation content, selects image
license id 4 (CC BY 2.0), decodes the local image files, and requires their
aggregate identity to match members independently hashed from the official
val2017 archive. It writes no image bytes and emits self-verifying holdout100,
train400, and promotion500 metadata. The filtered annotation artifacts are
identified separately as modified COCO annotations under CC BY 4.0.

This is machine-checkable provenance evidence, not legal or privacy clearance.
Image attribution sufficiency, annotation redistribution, privacy/PII, visual
quality, selection stability, benchmark suitability, and publication approval
remain explicit human-review gates before a run or upload.

Generate an unapproved, manifest-bound review template after preparing the
bundle:

```text
python -m libreyolo.validation.vlm_benchmark_dataset review-template \
  --manifest /data/libreyolo-vlm-benchmark/manifest.json \
  --annotations /data/coco/annotations/instances_val2017.json \
  --images-dir /data/coco/val2017 \
  --output /data/reviews/vlm-promotion500-review.json
```

The command re-verifies the bundle and selected image bytes, then writes a
create-only template with status `unapproved`, no reviewer or review time, and
every manual check set to `false`. It cannot authorize a run. A human reviewer
must complete the external file; the library never creates an approved
attestation.

The manifest records category coverage per partition. In the current pin,
holdout100 and promotion500 represent 79 of 80 COCO categories (raw category
89 is unavailable after the eligibility filter); train400 represents 76 of 80
(raw categories 21, 38, 87, and 89 are absent). Train400 must therefore not be
described as full 80-class supervised coverage.

The confidence runner accepts only the verified `promotion500` bundle. It does
not accept an arbitrary dataset YAML. Before any model construction it requires
the manifest, original pinned annotations, local image root, and a separate
human review attestation:

```text
PYTHONHASHSEED=0 python -m libreyolo.validation.vlm_confidence_benchmark run \
  --manifest /data/libreyolo-vlm-benchmark/manifest.json \
  --annotations /data/coco/annotations/instances_val2017.json \
  --images-dir /data/coco/val2017 \
  --review-attestation /data/reviews/vlm-promotion500-review.json \
  --output-root /data/runs/qwen-confidence-1 \
  --device cuda
```

The external attestation uses schema
`libreyolo.vlm-benchmark-dataset-review.v1`, binds the manifest SHA256 and
`zero_shot_confidence_promotion` role, names the reviewer and UTC review time,
and must explicitly approve every manual gate listed by the manifest. The
library records the attestation digest and checks the COCO artifact, category
mapping, image order, dimensions, and selected image bytes again before
generation.

Before renting or starting the full run, use the same evidence with
`preflight`:

```text
PYTHONHASHSEED=0 python -m libreyolo.validation.vlm_confidence_benchmark preflight \
  --manifest /data/libreyolo-vlm-benchmark/manifest.json \
  --annotations /data/coco/annotations/instances_val2017.json \
  --images-dir /data/coco/val2017 \
  --review-attestation /data/reviews/vlm-promotion500-review.json \
  --output-root /data/runs/qwen-confidence-1 \
  --device cuda
```

Preflight creates no run directory, lock, model, or network request. It checks
the clean code revision, process-start determinism, offline dependency state,
target device, native COCO loading and metric backend, all verified dataset
evidence, and the exact official Qwen3-VL-2B config, safetensors, and processor
bytes already present under `weights/LibreQwen3VL2b`. The output directory must
not exist and must be outside the worktree. `run` repeats these checks and does
not trust a saved preflight result.

A ready preflight is a point-in-time local readiness result. It does not prove
that the model will fit in memory, that a human attestation is truthful, or
that confidence quality meets the promotion thresholds.

Run twice in fresh processes and compare the persisted reports:

```text
python -m libreyolo.validation.vlm_confidence_benchmark compare \
  /data/runs/qwen-confidence-1/vlm_confidence_report.json \
  /data/runs/qwen-confidence-2/vlm_confidence_report.json
```

This runner remains an internal promotion gate. A reproducible result does not
activate public Qwen confidence or `val()` by itself; ranking, mAP, coverage,
default-threshold retention, and calibration still require maintainer review.

## Trainable families

| Family | train() | Why |
|---|---|---|
| qwen3-vl 2B/4B | yes | grounding-pretrained, Apache-2.0; first verified training cohort |
| qwen3-vl 8B | not yet | outside the first verified training cohort |
| lfm2-vl 450M | not yet | activation is blocked pending maintainer licensing and provenance review; no recipe is enabled |
| lfm2-vl 1.6B/3B, florence-2, north-micro-vision, internvl3, gemma-4, moondream | not yet | planned; see the training ADR draft |
| smolvlm2 | no | not grounding-pretrained; cannot reliably learn box emission |
| kosmos-2 | no | no established recipe; 224px 32x32 grid caps localization |
| locate-anything | no | research-only non-commercial upstream weights |
| sensenova-vision | no | multi-node-scale training only, no parameter-efficient path |
| libremodus | no | upstream ships a pretraining pipeline; gated research-only weights |

The refusal messages on the untrainable families state these reasons.

## Implementation notes

- Trainer: `libreyolo/models/vlm/training/` (dataset rendering, chat-template
  collation with longest-common-prefix label masking, PEFT LoRA injection with
  scope assertions, compact loop). Not a `BaseTrainer` subclass: stacked-tensor
  batching, EMA, and mAP-based epoch logic do not apply to a generative
  fine-tune.
- Label masking tolerates boundary retokenization (a template ending in a bare
  newline can merge that newline into the first answer token) but fails loudly
  if a chat template rewrites the user turn when an answer is appended.
- Tests: `tests/unit/test_vlm_training.py` (offline: serialization round-trip
  through the inference parser, collator masking, contract, gating) and
  `tests/e2e/test_vlm_train_qwen3vl.py` (CPU end-to-end on a pinned tiny-random
  Qwen3-VL: train, checkpoint, factory reload, predict).
