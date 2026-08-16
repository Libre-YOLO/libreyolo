# ADR 0002: LibreVLM Contract For Vision-Language Detectors

- Status: Accepted
- Date: 2026-06-05 (updated 2026-08-16)
- Scope: New model tier (vision-language models used as open-vocab detectors)

## Context

LibreYOLO's detector families are loaded by the `LibreYOLO(...)` factory, which
sniffs a `.pt` state dict (`can_load`), detects size from keys, and runs a
single forward pass producing calibrated `(boxes, scores, classes)`. RF-DETR
proves a transformer detector fits this path.

Vision-language models (Qwen3-VL, LFM2-VL, and others) do not fit it:

- They are multi-file Hugging Face repos, not a single sniffable state dict.
- They are autoregressive: image plus text prompt in, generated text out. A box
  is text the model types, not a tensor from a detection head.
- They have no per-box confidence, and the class list is open vocabulary (any
  words), not a fixed head.

Forcing them through `LibreYOLO(...)` would misrepresent what `conf` means and
the latency profile. But the boxes a VLM produces are exactly what LibreYOLO
already renders, so the user-facing experience can and should stay familiar.

## Decision

Add a parallel tier, `LibreVLM`, for generative open-vocabulary models. The line
is drawn on **contract fidelity**, not architecture:

- Faithful detector (real scores, closed-set-able, single forward) stays in the
  `LibreYOLO(...)` factory. This includes transformer detectors, exactly as
  RF-DETR already does.
- Generative VLM (soft confidence, prompt-driven, open vocabulary) is loaded by
  `LibreVLM(...)`.

Both return the same `Results`, so downstream code is unchanged. They are
separated because their *contract* differs, not because the network differs.

The model defaults to **Qwen3-VL-4B** (Apache-2.0), autodownloaded on first use.

## Public API

Two layers, intentionally:

1. The raw model (`chat`): the honest truth, an image-plus-text chat model.
2. The detection convenience (`set_classes` + `predict`/`track`): a cached
   detection prompt and a per-family parser, returning `Results`.

```python
from libreyolo import LibreVLM

model = LibreVLM()                          # Qwen3-VL-4B by default, autodownloads
model.set_classes(["pink car", "wheel"])    # open vocabulary, sticky, any words
result = model.predict("image.jpg")          # same Results as a YOLO model
results = model.predict("folder/")           # folders, video, stream, track()
result.boxes.xyxy        # pixel xyxy
result.boxes.cls         # ids into the vocabulary set above
result.plot(); result.save()

text = model.chat("image.jpg", "How many cars are pink?")  # raw escape hatch
```

- `set_classes(labels)` is the primary way to set the vocabulary. It is sticky:
  set once, reused by every later `predict()`/`track()` until set again. This
  keeps `predict()` signature-compatible with the closed-vocab detectors.
- `names=[...]` at construction is a convenience that calls `set_classes` for you.
- `chat(image, prompt)` exposes the underlying model for anything the detection
  wrapper does not cover (free-form questions, custom formats, counting). It is
  available on the chat-template families; the task-prompt families (Florence-2,
  Kosmos-2) are not chat models and their `chat()` raises `NotImplementedError`.
  `predict()` (the detection layer) is supported on every family.
- `prompt="..."` overrides the detection prompt on the chat-template families;
  `max_new_tokens`, `device` as usual. Florence-2 and Kosmos-2 build their prompt
  from a fixed task / grounding token plus the class list, so `prompt=` is ignored
  for those two.

The supported VLM CLI surface is intentionally smaller than the detector CLI:

```bash
libreyolo predict --model qwen3-vl-4b --source image.jpg \
  --names '["pink car", "wheel"]'
libreyolo train --model qwen3-vl-2b --data dataset.yaml
```

- `predict` resolves VLM aliases, schema-valid checkpoint directories, and
  immutable `hf+vlm://owner/repo@<commit>` artifacts. `--names` is a VLM-only
  JSON vocabulary; `--classes` remains a numeric output filter. Explicit
  `imgsz=` is rejected because the family processor owns image resizing.
  Directory chunks still generate one image at a time.
- `train` is verified only for the Qwen3-VL 2B and 4B base aliases. It uses
  VLM-native defaults rather than detector defaults, and selects `best` by
  validation loss when a validation split exists, otherwise training loss.
  Continue an adapter with the base alias plus `resume=<checkpoint directory>`;
  training an inference-loaded checkpoint wrapper is rejected.
- Detector-only training options, unsupported families/sizes, and standalone
  `val`, `export`, or `quantize` requests fail before loading VLM weights.

### Publication artifacts

Local training checkpoints and published artifacts are different contracts.
The v1 publication builder accepts only Qwen3-VL 2B/4B detection LoRA output
written with `peft==0.19.1` and `transformers==5.12.1`. It requires external,
human-approved evidence that binds the adapter, checkpoint contract, processor,
complete immutable base snapshot, training-data manifest, evaluation report,
clean code revision, and fixed recipe. The library can generate an unapproved
template with derived hashes, but it cannot manufacture an approval.

The artifact includes the adapter and exact Qwen processor, tokenizer, and
chat-template assets under Apache-2.0. Base weights remain reference-only.
`push_vlm_artifact()` refuses existing repositories, starts private, creates
one commit, verifies that immutable tree through a fresh download, and returns
`hf+vlm://owner/repo@<40-character-commit>`. The detector `hf://` transport and
generic Hub logger are not used. Hashes are integrity bindings, not signatures
or proof that human claims are true. See
[`../vlm_hub_artifact.md`](../vlm_hub_artifact.md).

## Internal Contract

`LibreVLMModel(BaseModel)` is the shared base. It does NOT define `can_load`, so
`BaseModel.__init_subclass__` never registers VLM families into the detector
`_registry`; they stay out of the weight-sniffing factory.

To support a new model, subclass it and declare the adapter:

| Field             | Meaning                                                  |
|-------------------|----------------------------------------------------------|
| `FAMILY`          | family id (e.g. `qwen3vl`)                               |
| `FILENAME_PREFIX` | upstream brand-cased weights dir prefix                  |
| `HF_REPOS`        | `{size: hf_repo_id}`; drives autodownload                |
| `HF_REVISIONS`    | `{size: commit_sha}`; required for remote-code families  |
| `INPUT_SIZES`     | `{size: nominal_px}`; nominal, the processor owns resize |
| `_detection_prompt()` | how to ask THIS model for boxes (override if needed)  |
| `BBOX_KEY`        | JSON key holding the box (`bbox`, `bbox_2d`, ...)        |
| `COORD_DIVISOR`   | scale of the coords (1.0 for [0,1], 1000.0 for 0-1000)  |
| `BOX_FORMAT`      | box layout: `xyxy` (default), `xywh`, `cxcywh`, or `yxyx` |
| `_LICENSE_NOTICE` | text logged once before loading/downloading (if needed)  |

The base implements the predict/track surface by satisfying the four hooks the
shared `InferenceRunner` drives:

- `_get_input_size()` returns the nominal `imgsz`.
- `_preprocess(image, ...)` builds the chat-template inputs from the image plus
  the detection prompt; returns `(inputs, pil_image, (W, H), ratio=1.0)`. Boxes
  come back normalized to the image, so there is no letterbox/unpad math.
- `_forward(inputs)` runs `model.generate(...)` greedily and returns the newly
  generated tokens. A family-gated scoring path can also attach one
  selected-token log-probability per step without retaining vocabulary-sized
  score tensors.
- `_postprocess(output, conf, ...)` decodes, tolerantly parses the JSON, scales
  the coordinates per `BBOX_KEY`/`COORD_DIVISOR`, and returns the standard
  detection dict `{boxes, scores, classes, num_detections}` that
  `InferenceRunner._wrap_results` converts to `Results`.

Parsing lives in `libreyolo/models/vlm/parsing.py` (pure, unit-tested offline):
it tolerates markdown fences, prose, single quotes, and truncated arrays; clamps
boxes and orders corners; dedupes identical/high-IoU boxes (a generative loop
can repeat one box); and maps labels case-insensitively to class ids, dropping
out-of-vocabulary labels. That label mapping is what makes an open-vocab
generator behave as a closed-set detector against `set_classes`.

### Coordinate conventions differ per model

Each model writes boxes in its own scheme, learned from its training labels, so
the convention must be verified empirically (feed a known box, read the output)
and declared via `BBOX_KEY`/`COORD_DIVISOR`. The verified per-model table lives
in [`../librevlm_design.md`](../librevlm_design.md).

## Confidence

Generated detections carry no calibrated per-box score. The generic VLM families
currently assign a constant placeholder (`DEFAULT_SCORE = 1.0`). A bounded-memory
candidate for Qwen3-VL can derive a ranking signal from the geometric mean of
generated label-token and coordinate-token probabilities. It records one
selected-token log-probability after the configured generation processors per
step, rather than retaining a vocabulary-sized score tensor for every token.
The candidate remains disabled until its real-data gate passes, so ordinary
`predict()` keeps the established constant-score behavior. Candidate scores are
available only to the internal validator, not public prediction, the CLI, or
integrations. LibreMODUS separately uses the minimum constrained-token
probability for each detection.

`model.confidence_method` reports the configured source (`constant` today for
Qwen3-VL, and `constrained_token_min` for LibreMODUS).

Consequences:

- On constant-score families, `conf=` filtering is mechanical and ByteTrack's
  score-stratified association remains inert (no separate low-confidence
  recovery stage).
- `val()` (mAP) remains unsupported until the candidate score orders correct
  detections better than the constant baseline, behaves safely with the public
  confidence threshold, and is reproducible. Unit tests establish plumbing, not
  score quality.
- Candidate-only Brier error, fixed-bin ECE, and reliability bins are diagnostic
  outputs of the internal gate, not a calibration claim. Missing scores never
  inherit the constant fallback in those statistics; fallback safety and score
  coverage are reported separately.
- Persisted gate reports use schema v2 and can be compared across processes only
  after strict reconstruction validates their hashes and duplicated metrics.
  Timing and plot artifacts do not define reproducibility. The hashes provide
  consistency checks, not report authentication.

`_score_detections(items)` remains the scalar fallback for custom generation
paths. Scored greedy generations use the additive per-item scoring path.

## Licensing

Family aliases either load through the Apache-2.0 `transformers` API or, when a
model genuinely requires Hugging Face remote code, download that upstream
repository at runtime under its terms. LibreYOLO does not mirror those base
weights or remote-code repositories. The strict Qwen LoRA artifact is the
narrow exception for non-weight assets: it redistributes the exact processor,
tokenizer, and chat-template files under Apache-2.0 while retaining the base
weights as an immutable external reference.

The default model (Qwen3-VL-4B) is Apache-2.0, so it needs no notice. When a
model's weights or required model-repository code are under a non-permissive
license (for example LFM2-VL under the LFM Open License v1.0 with a revenue
threshold, InternVL3 whose `-hf` weights carry the Qwen License, or a remote-code
model repository under its own terms), loading/downloading logs a one-time
license notice. This follows the existing download-notice pattern in
`libreyolo/utils/download.py` and `libreyolo/models/l2cs/model.py`.

Any family that sets `TRUST_REMOTE_CODE = True` must also pin `HF_REVISIONS` to
a commit SHA for every supported size. This keeps a LibreYOLO release from
executing mutable upstream model-repository code under the same alias.

## Out Of Scope (v1)

- Training outside the verified Qwen3-VL 2B/4B detection cohort. Supported
  fine-tuning is documented in [`../vlm_training.md`](../vlm_training.md).
- Dataset validation / mAP (`val()` and the CLI `val` command reject VLMs; see
  "Confidence").
- Export to ONNX/TensorRT/etc. and post-training quantization. The CLI `export`
  and `quantize` commands reject VLM references before loading weights.
- Broad CLI parity. The documented VLM CLI contract is prediction plus the
  verified Qwen training cohort; tier-specific methods such as `chat()` remain
  Python API surfaces.

## Consequences

### Positive

- Open-vocabulary, zero-setup detection behind a familiar predict/track surface.
- No change to the detector factory; VLM families are fully isolated.
- A new model is a small adapter class (repos, prompt, coordinate convention).

### Negative

- Generic-family confidence remains constant until each score path passes its
  real-data quality gate; LibreMODUS's constrained-token score is uncalibrated.
- Generation is slower and less deterministic than a detector forward.
- Adds `transformers` (already an optional extra) to the `vlm` extra.

## Implementation Status

- `LibreVLMModel` base with `set_classes()` and `chat()`.
- Chat-template families parse JSON boxes (Qwen3-VL, LFM2-VL, InternVL3,
  SmolVLM2, Gemma 4). Florence-2 and Kosmos-2 use task / grounding tokens.
  Moondream uses native detect/point skills. See the Available-models table in
  [`../librevlm_design.md`](../librevlm_design.md).
- CLI alias/checkpoint/immutable-artifact routing for VLM prediction and
  verified Qwen3-VL 2B/4B training, with pre-load guards for remote training
  and unsupported validation, export, and quantization.
- Strict, reviewed Qwen3-VL LoRA artifact construction and immutable Hub
  transport, separate from detector checkpoint publication.
- Offline parser unit tests plus a `vlm`-marked end-to-end smoke test.
