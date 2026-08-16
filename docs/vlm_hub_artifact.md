# VLM Hub artifact contract

`libreyolo.vlm-artifact.v1` is the publication format for the verified
Qwen3-VL 2B/4B detection LoRA cohort. It is a strict, flat directory artifact,
not a detector `.pt` file and not a general-purpose PEFT repository format.
Full-model fine-tunes and other VLM families are outside this v1 publication
contract.

## Writer contract

Install the training extra before producing a publishable checkpoint:

```bash
pip install "libreyolo[vlm-train]"
```

The v1 writer contract is exactly `peft==0.19.1` and
`transformers==5.12.1`. Training preflight enforces both versions. The
`libreyolo[vlm]` inference extra remains broader; it is not a publication
writer contract.

Add the Hub transport extra only when uploading or downloading an immutable
artifact:

```bash
pip install "libreyolo[vlm-train,hf]"
```

The artifact contains:

```text
.gitattributes
LICENSE
NOTICE
README.md
adapter_config.json
adapter_model.safetensors
chat_template.jinja
libreyolo_vlm.json
libreyolo_vlm_artifact.json
processor_config.json
publication_evidence.json
tokenizer.json
tokenizer_config.json
```

The manifest binds the exact file inventory, sizes, roles, and SHA-256
digests. The adapter must match the fixed Qwen3-VL detection LoRA recipe and
contain only the expected language-model LoRA tensors in safetensors form.

Base weights are not included. The artifact records an immutable Qwen Hub
revision plus the complete expected base-snapshot inventory. The exact Qwen
processor, tokenizer, and chat-template assets are included and redistributed
under Apache-2.0; the generated `LICENSE`, `NOTICE`, and model card record that
distinction.

## Publication evidence

Publication begins with a create-only, deliberately unapproved template:

```python
from libreyolo.models.vlm import create_vlm_publication_evidence_template

template = create_vlm_publication_evidence_template(
    "runs/vlm/train/weights/best",
    "weights/LibreQwen3VL2b",
    "reviews/strawberry-vlm.unapproved.json",
    training_data=training_data_record,
    code=code_record,
    confidence_report="runs/vlm/confidence/vlm_confidence_report.json",
    repeatability_receipt="reviews/qwen-confidence-repeatability.json",
)
```

The two plain JSON input records contain:

- `training_data`: `source`, `version`, `split`, `license_spdx`,
  `license_evidence_url`, and `manifest_sha256`.
- `code`: a clean 40-character `revision`, boolean `clean`, and exact
  `libreyolo`, `peft`, `torch`, and `transformers` dependency versions.

`confidence_report` must name a strict
`libreyolo.vlm-confidence-report.v2` report produced by the v3 confidence
runner against that exact checkpoint. Its sibling
`vlm_confidence_run.json` envelope is mandatory. Checkpoint-backed publication
uses only `holdout100` with role `fine_tune_validation`; `promotion500` is not
valid publication evidence because it includes the `train400` fine-tuning
partition. The helper derives and binds the raw report and envelope SHA-256
digests, the exact benchmark identity
`libreyolo.vlm-confidence-report.v2:libreyolo.vlm-confidence-benchmark-context.v3:holdout100:fine_tune_validation`,
and exactly 17 finite mAP, ranking, calibration, coverage, and
threshold-retention metrics. Probability metrics must be in `[0, 1]`, mAP
deltas must be in `[-1, 1]`, and each delta must equal the candidate metric
minus its constant-score counterpart. Timing and internal counter metrics are
not publication evaluation claims.

`repeatability_receipt` must name canonical
`libreyolo.vlm-confidence-repeatability-receipt.v1` JSON produced by the
confidence runner's `compare --receipt` command. Its ordered `runs[0]` entry
must exactly match `confidence_report` and its sibling envelope, including the
run and process identifiers plus both raw SHA-256 digests. `runs[1]` must have
different run and process identifiers. Publication accepts only a receipt
whose strict comparison is reproducible with `score_atol`, `metric_atol`, and
`map_atol` all equal to zero.

The report and envelope must agree on the complete execution context. The
helper rejects any difference in the full path-free checkpoint identity,
including its pinned base, adapter weights, adapter configuration, checkpoint
contract, processor, and exact file records. It also binds a canonical hash of
the evaluation claim itself so a reviewer cannot edit its benchmark, report,
envelope, checkpoint, or metric fields without invalidating the review.
Publication evidence v2 also carries a path-free repeatability claim and binds
both the raw receipt SHA-256 and the canonical comparison SHA-256.

The helper validates the checkpoint and complete pinned base snapshot, derives
the recipe, adapter, contract, processor, evaluation, and base bindings,
validates the declared training-data and code record shapes, rechecks the
checkpoint, benchmark run, and repeatability receipt, and writes canonical
JSON outside both input directories. The training-data and code claims still
require human review. The template sets legal and redistribution decisions to
`unreviewed`, evaluation and review approval to `false`, reviewer fields to
blank, and every human gate to `false`. It never creates an approval, and the
artifact builder rejects the untouched template.

The first receipt run supplies the published validation metrics. `holdout100`
is fine-tune validation evidence, not an untouched test set. The second
fresh-process run and complete strict comparison are machine-bound through the
receipt. This is structural byte-integrity evidence, not publisher or reviewer
authentication, proof that the source reports are truthful, or a substitute
for human judgment about quality and publication suitability.

A human reviewer must verify the underlying data, license, privacy,
evaluation, and code-provenance evidence. The reviewed file must retain its
derived bindings and explicitly record:

- artifact and processor redistribution as `approved`, while base weights
  remain `reference-only`;
- training data as `approved-for-derived-weights`;
- a passed evaluation against the bound adapter;
- a clean code revision, reviewer identity, RFC 3339 UTC review time, overall
  approval, and all six review gates as true.

SHA-256 values make accidental or substituted byte changes detectable. They
are integrity and consistency records, not signatures, proof that a report is
truthful, publisher or reviewer authentication, or legal advice. The
repeatability claim means only that the strict comparator accepted the exact
bound inputs with zero tolerances.

## Build, validate, and upload

After human approval, build into a path that does not exist:

```python
from libreyolo.models.vlm import build_vlm_artifact, validate_vlm_artifact

info = build_vlm_artifact(
    "runs/vlm/train/weights/best",
    "artifacts/strawberry-vlm",
    publication_evidence="reviews/strawberry-vlm.approved.json",
)
validate_vlm_artifact(info.root)
```

The builder revalidates every source and evidence binding, regenerates the
license, notice, model card, Hub configuration, and manifest, then publishes
the result create-only. `validate_vlm_artifact()` is offline and checks the
complete directory and payload again.

Upload is a separate explicit step:

```python
from libreyolo.models.vlm import push_vlm_artifact

uri = push_vlm_artifact(info.root, "someuser/strawberry-vlm")
```

The repository must not already exist. Upload starts private, writes one
commit, verifies the exact committed tree through a fresh download, and
returns `hf+vlm://owner/repo@<40-character-commit>`. Passing `private=False`
makes the repository public only after verification. The generic detector Hub
logger and `LibreVLMModel.push_to_hub()` are not VLM publication paths.

Load only through the immutable URI:

```python
from libreyolo import LibreVLM

model = LibreVLM(uri)
```

`LibreVLM` verifies the artifact and the exact referenced base snapshot before
constructing the model. Remote artifacts are inference inputs; training still
starts from a verified Qwen base alias and uses `resume=<local checkpoint>`.

See [`hf_hub.md`](hf_hub.md) for transport details and
[`vlm_training.md`](vlm_training.md) for the local checkpoint workflow.
