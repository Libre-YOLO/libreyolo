# ADR 0017: Kosmos-2 Core ML Contract

- Status: Accepted (experimental runtime)
- Date: 2026-07-30
- Scope: Kosmos-2-patch14-224 open-vocabulary detection on Core ML

## Context

Kosmos-2 grounding combines a vision encoder, learned image-to-text projection,
causal language generation, special grounding tokens, and processor-side box
decoding. A single image-to-output graph does not preserve that contract. The
deployment artifact must carry the exact offline processor and an explicit host
generation contract.

## Provenance and source admission

The first profile accepts exactly:

- `microsoft/kosmos-2-patch14-224` at revision
  `e91cfbcb4ce051b6a55bfb5f96165a3bbf5eb82c`;
- MIT checkpoint code and weights;
- `model.safetensors`, 6,658,052,808 bytes, SHA-256
  `051bf4b62a25429f4d542d11ec0c07a4ac1aac91003d3bf301133c6913008cbf`;
- 1,664,485,376 FP32 parameters;
- eight byte-for-byte SHA-256-pinned processor assets from the same revision;
- Hugging Face Transformers 5.12.1 at commit
  `ddb849abe009d1089e6c691bfc897f27211c663c`, Apache-2.0;
- Core ML Tools 9.x and CPU-only execution.

The exporter rejects other revisions, file bytes, processor contents,
architectures, parameter counts, floating-point dtypes, Transformers versions,
compute planners, contexts, or source sizes before conversion.

## Fixed component ABI

The `.coremlvlm` bundle contains three separate FP32 ML Program packages:

1. `Vision.mlpackage`
   - input: `pixel_values`, `[1, 3, 224, 224]`;
   - output: `image_embeddings`, `[1, 64, 2048]`.
2. `TokenEmbedding.mlpackage`
   - input: `input_ids`, INT32 `[1, 128]`;
   - output: `token_embeddings`, `[1, 128, 2048]`.
3. `Decoder.mlpackage`
   - inputs:
     - `input_embeddings`, `[1, 128, 2048]`;
     - `attention_mask`, `[1, 128]`;
     - `position_ids`, INT32 `[1, 128]`;
   - output: `last_logits`, `[1, 65037]`.

The host left-pads every active prefix to 128 tokens, inserts the 64 projected
image embeddings at the processor-declared positions, and recomputes the
stateless decoder for each generated token. Generation is capped at 48 new
tokens and uses greedy selection, no-repeat trigrams, and EOS token 2. This is
a small, auditable fidelity profile, not a throughput or latency profile.

The host loads the pinned processor locally, validates every named output,
detokenizes with special tokens preserved, delegates grounding-token parsing to
the processor, maps labels to the ordered LibreYOLO class vocabulary, and
returns the existing synthetic score 1.0. Malformed, unknown-label, non-finite,
or non-positive-area boxes are discarded.

## Portable artifact and runtime

The portable directory contains:

```text
name.coremlvlm/
  manifest.json
  Decoder.mlpackage/
  TokenEmbedding.mlpackage/
  Vision.mlpackage/
  Processor/
  LICENSES/MIT-Kosmos-2.txt
  LICENSES/Apache-2.0.txt
  NOTICE.txt
```

The manifest binds every deployment file by path, byte length, and SHA-256.
Validation rejects symbolic links, special files, traversal, duplicate keys,
unmanifested files, processor extras, source safetensors, invalid Core ML
specifications, or mismatched embedded metadata. Publication is staged beside
the destination and never overwrites an existing bundle.

The bundle stays on the `LibreVLM` surface. Only `predict()` is supported;
Kosmos-2 is task-token driven and does not gain a chat API from export.

## Apple hardware evidence

The exact profile passed a fresh real-device campaign on Apple M4/macOS 27:

- all three saved packages executed through Core ML with CPU_ONLY;
- two deterministic probes passed the maximum-relative-error `3e-4`,
  minimum-relative-sensitivity `1e-6`, and sensitivity/error margin `100x`
  gates for vision and decoder outputs;
- token embeddings passed saved-package PyTorch parity;
- the native public predictor returned a non-empty grounded result;
- the bundle reloaded through `LibreVLM`, matched native boxes, classes, and
  scores within the explicit test tolerances, and repeated bit-exactly.

This validates the exact checkpoint, component graphs, processor bundle, and
host result contract. It does not prove detection accuracy, arbitrary source
geometry, other checkpoints or contexts, Neural Engine placement, latency, or
device performance.

No execution-profile-v2 identity is registered yet. Export and loading default
to `compute_units="validated"` and fail closed; experimental use requires the
explicit `compute_units="cpu_only"` planner.
