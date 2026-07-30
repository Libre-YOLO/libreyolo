# ADR 0018: Qwen3-VL-2B Core ML Contract

- Status: Accepted (experimental runtime)
- Date: 2026-07-30
- Scope: Qwen3-VL-2B open-vocabulary detection on Core ML

## Context

Qwen3-VL detection is multimodal autoregressive generation. Correct deployment
requires its processor and chat template, vision tower, three DeepStack feature
injections, tied token embedding, causal text decoder, multimodal 3D positions,
interleaved MRoPE, repetition penalty, greedy generation, detokenization, and
JSON box parsing. Conversion success for any one graph is not evidence that
this complete contract works.

The first deployment profile deliberately favors a small, auditable ABI over
latency. It fixes one image and recomputes a stateless, left-padded decoder
prefix for every generated token.

## Provenance and source admission

The profile accepts exactly:

- `Qwen/Qwen3-VL-2B-Instruct` at revision
  `89644892e4d85e24eaac8bacfd4f463576704203`;
- Apache-2.0 checkpoint code and weights;
- `model.safetensors`, 4,255,140,312 bytes, SHA-256
  `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`;
- 2,127,532,032 FP32 parameters;
- nine byte-for-byte size- and SHA-256-pinned processor assets;
- Hugging Face Transformers 5.12.1 at commit
  `ddb849abe009d1089e6c691bfc897f27211c663c`, Apache-2.0;
- Core ML Tools 9.x and `CPU_ONLY`.

The exporter validates the checkpoint bytes, processor bytes, architecture,
parameter count, source floating-point dtype, Transformers version, context,
and compute planner before conversion. Qwen3-VL-4B, Qwen3-VL-8B, arbitrary
checkpoints, and other toolchain versions are not admitted by this contract.

LibreYOLO composes public Transformers modules from the pinned Apache-2.0
implementation. The repository notice identifies the upstream repository,
commit, files, checkpoint, license, and checkpoint hash.

## Fixed component ABI

The `.coremlvlm` bundle contains three ML Program packages:

1. `Vision.mlpackage`, FP32
   - input: `patch_values`, `[784, 1536]`;
   - outputs: `image_embeddings`, `deepstack_0`, `deepstack_1`, and
     `deepstack_2`, each `[196, 2048]`.
2. `TokenEmbedding.mlpackage`, FP16
   - input: `input_ids`, INT32 `[1, 512]`;
   - output: `token_embeddings`, `[1, 512, 2048]`.
3. `Decoder.mlpackage`, FP16
   - inputs:
     - `input_embeddings`, `[1, 512, 2048]`;
     - `causal_mask`, `[1, 1, 512, 512]`;
     - `rope_cos` and `rope_sin`, each `[1, 512, 128]`;
     - `deepstack_embeddings`, `[3, 1, 512, 2048]`;
   - output: `last_logits`, `[1, 151936]`.

The host stretches one RGB image to 448 by 448 pixels and requires the pinned
processor to emit exactly a `[1, 28, 28]` image grid, 784 patches, and one
contiguous 196-token image group. It constructs Qwen's one-image 3D positions,
interleaves temporal/height/width rotary frequencies with sections
`[24, 20, 20]`, builds finite `-1e4` causal and invalid-key masks, left-pads the
active prefix, and scatters the main plus three DeepStack image embeddings.

Generation is greedy with repetition penalty 1.1, both pinned EOS token IDs,
and at most 48 new tokens inside the fixed 512-token context. The runtime
returns only generated tokens. The public adapter detokenizes and reuses the
existing Qwen `bbox_2d` 0-to-1000 parsing contract, maps labels to the active
class vocabulary, and preserves the original image geometry for result boxes.

This full-prefix CPU profile prioritizes conversion fidelity. It is not a
latency, batching, video, or Neural Engine profile.

## Portable artifact and validation

The portable directory is:

```text
name.coremlvlm/
  manifest.json
  Decoder.mlpackage/
  TokenEmbedding.mlpackage/
  Vision.mlpackage/
  Processor/
  LICENSES/Apache-2.0.txt
  NOTICE.txt
```

The exact manifest is compared structurally at load time. The runtime rejects
symbolic-link bundle roots, missing components, modified processor bytes,
wrong package input/output shapes, mismatched component metadata, unsupported
compute planners, and incompatible Transformers versions. The source
`model.safetensors` is not copied into the deployment bundle.

The bundle stays on the `LibreVLM` surface and supports `predict()`, `track()`,
and bounded `chat()` generation. Exported bundles cannot be re-exported.

## Apple hardware evidence

The exact profile passed a real-device campaign on Apple M4/macOS 27:

- FP16 vision was rejected after a 0.6706 relative-error failure;
- the retained FP32 vision package measured `7.78485e-5` worst relative error
  for final image embeddings and at most `2.56273e-5` across the three
  DeepStack outputs, with meaningful two-probe sensitivity;
- the retained FP16 decoder measured `0.00176033` worst relative logit error,
  `0.0611670` relative sensitivity, a `34.747x` sensitivity/error margin, and
  preserved both probe top tokens;
- host-generated 3D positions and FP16 MRoPE tables matched the pinned
  Transformers implementation exactly on the production request;
- the production component sizes were 3,442,104,385 bytes for the decoder,
  1,622,367,743 bytes for vision, and 622,332,995 bytes for token embedding;
- the final portable bundle was 5,698,319,484 bytes and loaded in 17.724
  seconds with `CPU_ONLY`;
- PyTorch and Core ML generated the exact same 48-token detection prefix,
  class, and box on a fixed real image; box IoU was `0.99999993`;
- Core ML generation took 27.005 seconds, and a second request through public
  `predict()` took 24.716 seconds and repeated boxes/classes exactly.

This proves the exact checkpoint, saved component graphs, processor contract,
host MRoPE/scatter/generation logic, portable reload, and public result path.
It does not prove model accuracy, calibrated confidence, arbitrary prompts,
other images or checkpoints, other contexts, accelerator placement, or
production performance.

No execution-profile-v2 identity is registered. Export defaults to
`compute_units="validated"` and fails closed; experimental export and loading
require explicit `compute_units="cpu_only"`.
