# ADR 0014: Stateful Core ML VLM Bundle Contract

- Status: Accepted (experimental runtime)
- Date: 2026-07-29
- Scope: SmolVLM2-500M Core ML export and deployment

## Context

A generative VLM is not a one-image/one-output graph. Deployment also needs a
tokenizer, image processor, prompt construction, image-token merging,
autoregressive decoding, and a request-local KV cache. A bare `.mlpackage`
therefore cannot preserve the public `LibreVLM` contract.

Core ML state is available only to ML Program models targeting iOS 18 or
macOS 15 and later. Its state is mutated by prediction, so reuse across
requests would leak one prompt into another.

## Decision

LibreYOLO supports one narrow, versioned profile:

- family/size: `smolvlm2` / `500m`;
- checkpoint:
  `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`;
- source license: Apache-2.0;
- conversion toolchain: Core ML Tools 9.x and Transformers 5.12.1;
- public contexts: 2048 and 4096 tokens;
- deployment target: iOS 18 or macOS 15 and later;
- precision: mixed compute with FP32 vision/decoder, FP16 token embedding,
  FP16 function I/O, and FP16 KV state.

The 8192-token conversion profile remains an internal specification and is
rejected by the public exporter/runtime until peak prefill memory is measured
on Apple hardware. SmolVLM2-2.2B and other VLM architectures require separate
contracts.

This bounded graph contract is not an Apple hardware execution profile. Every
package records `coreml_execution_profile_status=experimental`. Public export
and runtime therefore default to `compute_units="validated"` and reject before
conversion or native model-proxy creation. A caller must pass an explicit
native planner to opt into the experimental route; `cpu_only` is the
recommended discovery planner. Explicit accelerator planners remain
experimental and carry no placement or numerical-parity claim.

## Portable artifact

The public artifact is a directory ending in `.coremlvlm`:

```text
name.coremlvlm/
  manifest.json
  Model.mlpackage/
  Processor/                 # exactly 11 hash-pinned assets
  LICENSES/Apache-2.0.txt
  NOTICE.txt
```

`manifest.json` binds every payload path, byte length, and SHA-256; the model
package is also checked against Apple's package manifest and the embedded Core
ML metadata/ABI. Symbolic links, special files, unmanifested payloads,
duplicate JSON keys, path traversal, source `model.safetensors`, and overwrite
publication are rejected. The full canonical Apache-2.0 text travels with the
artifact, together with a deterministic model/processor/Transformers provenance
notice. Bundle publication is staged beside the destination and uses a
same-filesystem no-replace rename.

The original 2.03 GB safetensors file is conversion input, not a deployment
payload. Converted Core ML weights remain inside `Model.mlpackage`.

## Named-function ABI

One multifunction package exposes:

1. `encode_image`
   - input: `pixel_values`, FP16 `[1, 17, 3, 512, 512]`;
   - output: `image_embeddings`, FP16 `[1, 1088, 960]`.
2. `embed_tokens`
   - input: `input_ids`, INT32 `[1, Q]`, with finite `Q`;
   - output: `token_embeddings`, FP16 `[1, Q, 960]`.
3. `decode`
   - inputs: token embeddings, an append-only causal mask, and position IDs;
   - output: `last_logits`, FP16 `[1, 49280]`;
   - state: key/value caches, each
     `[32, 1, 5, context_length, 64]`.

The decoder supports bounded multi-token prefill and single-token decode.
Every request creates exactly one fresh Core ML state paired with one
append-only host cursor. A failed request discards both; cursors cannot be
reset independently of state.

## Host contract

The host performs the operations that are not part of the three functions:

- fixed-stretch RGB preprocessing to 2048x2048;
- the pinned processor's 17-crop, all-valid image expansion;
- tokenization and validation of exactly 1088 image placeholders;
- image/token embedding merge;
- causal-mask and position-ID construction;
- greedy decoding with repetition penalty and EOS termination;
- detokenization and the family result parser.

All flexible axes have finite upper bounds. The host validates token ranges,
shapes, dtypes, finite values, context budget, output names, processor hashes,
metadata, and state/cursor progression before committing a decode step.

## Public API

```python
from libreyolo import LibreVLM

source = LibreVLM("smolvlm2-500m", device="cpu")
bundle = source.export(
    format="coreml",
    context_length=2048,
    output_path="smol.coremlvlm",
    compute_units="cpu_only",  # explicit experimental opt-in
)

deployed = LibreVLM("smol.coremlvlm", compute_units="cpu_only")
deployed.set_classes(["cat", "dog"])
result = deployed.predict("image.jpg")
text = deployed.chat("image.jpg", "Describe the scene.")
deployed.close()
```

The bundle stays in the `LibreVLM` tier; it is not routed through
`LibreYOLO` or the generic one-shot `CoreMLBackend`.

## Validation status

The exact 500M source has:

- source-wrapper parity for the full fixed-grid vision path and Llama decoder;
- complete Core ML Tools 9 conversions for both public context profiles;
- multifunction spec, metadata, processor, license, and portable-bundle
  validation on Linux;
- fake-runtime coverage for named-function loading, fresh state, cursor
  progression, failure cleanup, generation, parsing, and concurrent-request
  serialization;
- real named-function and request-local-state execution on Apple M4 for the
  2048- and 4096-token packages;
- two-probe PyTorch/Core ML parity with 0.0258% worst vision error, exact token
  embeddings, 0.0586% worst stateful-decoder error, and meaningful
  input-sensitivity margins;
- repeated public `chat()` and `predict()` inference with fresh state.

This hardware evidence validates only the explicit CPU-only planner and exact
profiles above. No execution-profile-v2 identity is registered yet, so omitting
`compute_units` intentionally continues to fail closed.
