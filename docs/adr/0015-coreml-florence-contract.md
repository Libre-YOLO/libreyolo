# ADR 0015: Florence-2 Core ML Contract

- Status: Accepted (experimental runtime)
- Date: 2026-07-29
- Scope: Florence-2-base open-vocabulary detection on Core ML

## Context

Florence-2 detection is encoder-decoder generation, not a conventional
one-shot detector graph. Deployment requires the exact image processor and
tokenizer, a multimodal DaViT/BART encoder, autoregressive BART decoding,
three-beam search, cache reordering, detokenization, and Florence's coordinate
post-processor.

A bare `.mlpackage` cannot preserve that behavior by itself. The deployment
artifact must carry a pinned offline processor and a host contract. It must
also isolate state between requests: reusing a decoder state would leak
self-attention history from one image into the next.

## Provenance decision

The first public profile accepts exactly:

- family/size/task: `florence2` / `base` / `detect`;
- checkpoint and processor:
  `florence-community/Florence-2-base` at
  `00921df66db728a9ceb750f5eca43e5c203a2051`;
- checkpoint license: MIT, original copyright Microsoft Corporation;
- checkpoint payload: `model.safetensors`, 463,178,864 bytes,
  SHA-256
  `62f3e696da74f8869a68ddb529a9b3e14eb25b21c592cb3dea6179bf944df6a0`;
- unique in-memory parameter count: 231,443,968;
- processor: exactly ten byte-for-byte SHA-256-pinned files from the same
  revision, loaded with `local_files_only=True` and
  `trust_remote_code=False`;
- graph and beam reference: Hugging Face Transformers 5.12.1, commit
  `ddb849abe009d1089e6c691bfc897f27211c663c`, Apache-2.0;
- adapted reference files:
  `src/transformers/models/florence2/modeling_florence2.py`,
  `src/transformers/models/bart/modeling_bart.py`,
  `src/transformers/generation/utils.py`,
  `src/transformers/generation/logits_process.py`, and
  `src/transformers/generation/stopping_criteria.py`;
- converter: Core ML Tools 9.x;
- deployment target: iOS 18 or macOS 15 and later.

Unknown revisions, floating revisions, alternate processor bytes, remote
checkpoint code, architecture drift, untied BART embeddings, changed
in-memory tensor values, and unpinned Transformers versions are rejected
before conversion.

The converted bundle carries the Microsoft MIT license, the canonical
Apache-2.0 license, and a deterministic notice naming both exact upstream
revisions. The repository-level third-party notice must receive the same
Transformers adaptation declaration before integration.

## Fixed deployment profile

The base ABI is finite and versioned:

- image input: one RGB image processed to FP16 `[1, 3, 768, 768]`;
- image tokens: exactly 577 contiguous placeholders at the encoder prefix;
- encoder context: 1024;
- decoder context and maximum generated-token budget: 1024;
- text width/layers/heads/head width: 768 / 6 / 12 / 64;
- vocabulary: 51,328;
- beams: exactly three;
- precision: FP32 encoder compute, FP16 decoder compute and function I/O, and
  FP16-declared state from an FP32 source model. Apple's runtime materializes
  writable state through FP32 host arrays.

The final token at a 1024-token generation budget is forced EOS and is never
fed back into the decoder, so decoder positions 0 through 1023 are sufficient.

This bounded graph contract is not an Apple hardware execution profile. Every
package records `coreml_execution_profile_status=experimental`. Public export
and runtime default to `compute_units="validated"` and reject before conversion
or native model-proxy creation. Experimental use requires an explicit native
planner; `compute_units="cpu_only"` is the recommended discovery setting.
Explicit accelerator planners remain unvalidated opt-ins and do not imply
operator placement or numerical parity.

Four decoder state buffers are each shaped as an aggregate layer/beam cache.
The two self-attention and two cross-attention buffers have shape
`[6, 3, 12, 1024, 64]`. Together they occupy 113,246,208 bytes (108 MiB) at
FP16. The encoder emits one-beam cross K/V tensors, each
`[6, 1, 12, 1024, 64]` (9 MiB).

## Named-function ABI

One multifunction ML Program exposes exactly two functions:

1. `encode`, stateless
   - inputs:
     - `pixel_values`, FP16 `[1, 3, 768, 768]`;
     - `encoder_input_ids`, INT32 `[1, 1024]`;
     - `encoder_attention_mask`, FP16 `[1, 1, 1, 1024]`;
   - outputs:
     - `cross_key_values`, FP16 `[6, 1, 12, 1024, 64]`;
     - `cross_value_values`, FP16 `[6, 1, 12, 1024, 64]`.
2. `decode`, stateful, one token for three beams
   - inputs:
     - `decoder_input_ids`, INT32 `[3, 1]`;
     - `causal_mask`, FP16 `[3, 1, 1, E]`, `1 <= E <= 1024`;
     - `cross_attention_mask`, FP16 `[3, 1, 1, 1024]`;
     - `position_ids`, INT32 `[3, 1]`;
     - `beam_parent_indices`, INT32 `[3]`;
   - output: `last_logits`, FP16 `[3, 51328]`;
   - state:
     `self_key_cache`, `self_value_cache`, `cross_key_cache`, and
     `cross_value_cache`.

The static graph retains the native DaViT/projector, BART encoder, BART
decoder, and language-head equations. The decoder reorders all three
self-attention cache rows from `beam_parent_indices` before appending the
selected token.

## Cross-state initialization

`encode` is stateless, while `decode` reads cross K/V from state. The host
therefore owns one explicit dataflow step:

1. validate both encoder outputs by exact name, nominal FP16 contract,
   representable finite values, and `[6, 1, 12, 1024, 64]` shape; Apple's
   runtime may materialize nominal FP16 outputs as NumPy FP32;
2. repeat only the beam axis to `[6, 3, 12, 1024, 64]`;
3. allocate a new decoder state with `MLModel.make_state()`;
4. widen the repeated arrays to contiguous FP32, as required by Apple's
   writable-state materialization, and write them by the exact names
   `cross_key_cache` and `cross_value_cache` using
   `MLState.write_state(name=..., value=...)`;
5. begin decoding only after both writes succeed.

This is a supported host operation, not a captured PyTorch initializer.
Apple's
[Core ML Tools stateful-model guide](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
documents Python `read_state`/`write_state`, and Apple's
[MLState API](https://developer.apple.com/documentation/coreml/mlstate)
documents direct state-buffer access and requires predictions using one state
to be serialized.

Every request gets a fresh state and a fresh append-only position cursor.
The runtime serializes the entire processor/encode/state-seed/decode/postprocess
sequence. A failed encoder call, either failed state write, a failed decode
call, invalid output, or failed postprocessing discards the request state.
State and cursor cannot be independently reset or reused.

## Host generation contract

The host performs:

- ordered class-prompt construction with
  `<OPEN_VOCABULARY_DETECTION>`;
- fixed padding and additive encoder/cross-mask construction;
- cross-cache repeat and named state writes;
- deterministic FP32 log-softmax and exact three-beam scoring;
- six candidate continuations per step so EOS candidates do not remove all
  three live beams;
- no-repeat trigram processing;
- forced BOS at the first generated position and forced EOS at the budget;
- early stopping and length penalty 1.0;
- beam-parent propagation into the next decoder call;
- `batch_decode(skip_special_tokens=False)`;
- `post_process_generation(..., task="<OPEN_VOCABULARY_DETECTION>")`;
- mapping returned labels back to the ordered LibreYOLO class IDs.

Florence provides no calibrated detection confidence for this task. Each
valid parsed box receives the existing placeholder score 1.0, so confidence
filtering is intentionally all-or-nothing. Boxes with unknown labels,
non-finite coordinates, malformed coordinates, or non-positive area are
discarded.

## Portable artifact

The portable artifact is a directory ending in `.coremlvlm`:

```text
name.coremlvlm/
  manifest.json
  Model.mlpackage/
  Processor/                    # exactly ten pinned assets
  LICENSES/MIT-Florence.txt
  LICENSES/Apache-2.0.txt
  NOTICE.txt
```

`manifest.json` binds every payload path, byte length, and SHA-256, the exact
profile, processor provenance, and Core ML contract hash. Validation also
checks Apple's package manifest, the two-function Core ML specification, and
embedded metadata.

Symbolic links, special files, path traversal, duplicate manifest keys,
unmanifested files, missing files, processor extras, renamed source weights,
and source `model.safetensors` are rejected. The 463 MB source safetensors file
is conversion input, not a deployment payload. Publication is staged beside
the destination and uses a same-filesystem atomic no-replace rename. Existing
destinations are never overwritten.

## Deliberate limitations

This contract supports only `florence2/base` open-vocabulary detection. It
does not claim support for:

- Florence caption, OCR, dense caption, grounding, or segmentation prompts;
- Florence-2-large or fine-tuned/community variants;
- dynamic image sizes, batch sizes, beam counts, or contexts;
- model accuracy, processor-to-source accuracy, or application-level quality
  merely from graph parity.

The graph wrappers are dimension-driven where the native architecture permits
it, but the public validators, metadata, source provenance, conversion
entrypoint, and runtime fail closed to the base profile.

The audited future `florence2/large` source has width 1024, 12 encoder and
12 decoder layers, 16 heads, and a source maximum position count of 4096. A
strict 1024 encoder/decoder deployment profile remains source-valid and would
use about 288 MiB of four-state FP16 cache. A full 4096 profile would require
about 1.125 GiB for state alone. Large needs its own pinned checkpoint,
processor, parameter-count, parity, memory, latency, bundle, and hardware
contract; it is not implicitly covered here.

## Recorded Linux conversion evidence

The exact base checkpoint passed full Core ML Tools 9 conversion and strict
bundle validation on Linux:

- all 665 source FP16 tensors matched the model's exact lossless FP32 widening;
- the merged multifunction package is 464,766,282 bytes with tree SHA-256
  `edc4c4baa9a42cbbdfcb2f8b08e3c827017f3122adaa36f2c4d7a5b58a053dcc`;
- the deduplicated `weight.bin` is 464,111,040 bytes with SHA-256
  `63c7008ac7f9e78e2370f43a8109e28a71139fac0987d10c080ee3e07795345b`;
- the portable bundle is 470,159,338 bytes with tree SHA-256
  `48faba1341be70f381e6fa649ade0385b17cf1219ad76ff78b2c7dbe12561b3b`;
- the bundle contains only the package, ten exact processor assets, the MIT
  and Apache-2.0 licenses, notice, and manifest; it does not contain source
  safetensors;
- conversion took 47.32 seconds with peak RSS 5,945,364 KiB and no swap.

This proves source admission, graph conversion, package structure, metadata,
weight deduplication, and bundle integrity. It is not Apple runtime, numerical
runtime parity, latency, task accuracy, or application-level evidence.

## Apple hardware evidence

The exact base bundle passed real CPU-only execution on Apple M4:

- named encoder outputs reached both decoder cross-state buffers;
- two deterministic probes measured 0.0416% worst encoder-cache error and
  0.2923% worst stateful-decoder error against the prepared PyTorch graphs;
- every output had a meaningful input-sensitivity margin;
- parent-beam reorder and recurrent state progression passed;
- repeated public requests allocated fresh states and were deterministic.

This evidence covers only the exact checkpoint, ABI, and explicit CPU-only
planner. No execution-profile-v2 identity is registered yet, so omitting
`compute_units` intentionally continues to fail closed.
