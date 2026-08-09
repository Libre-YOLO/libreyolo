# ADR 0016: LibreMODUS Analysis-Only Any-to-Any Contract

- Status: Accepted
- Date: 2026-07-29
- Scope: MODUS 14B-A7B inference integration

## Context

MODUS is a decoder-only mixture-of-transformers model trained across image,
dense-prediction, and structured-token modalities. It does not fit the
single-file detector factory: the released checkpoint is a multi-file external
snapshot, combines autoregressive and flow-matching decoders, and can consume
several conditions in one context.

LibreYOLO needs the model's analysis and composition capability without
turning the library into a general content-generation surface. It must also
preserve the distinction between the Apache-2.0 source repository and the
separately distributed checkpoint.

## Decision

Add `LibreMODUS` to the `LibreVLM` tier with two public surfaces:

1. Standard `predict()` tasks: `depth`, `normal`, `edge`, and `detect`.
2. `any2any()` for one to three image-derived inputs, optional text, chaining,
   and best-of-N self-verification.

Every request must include an image-derived condition. RGB is not an output
target. Text-to-image, captioning, VQA, feature tokens, training, and export are
not exposed.

The canonical class spelling is `LibreMODUS`; `LibreModus` remains an import
alias. The only size is `14b-a7b`, and the canonical factory alias is
`libremodus-14b-a7b`.

## Result mapping

| Public task/target | Decoder | Standard result |
| --- | --- | --- |
| `depth` | flow + VAE | `DepthMap` |
| `normal` | flow + VAE | `NormalMap` |
| `edge` / `canny` / `samedge` | flow + VAE | `EdgeMap` |
| `detect` / `cocodet` | constrained autoregressive tokens | COCO-80 `Boxes` |
| `grounding` / `det` | phrase-conditioned constrained coordinates | `Boxes` |

Generated coordinates are constrained to valid token slots. `x2` and `y2`
cannot precede `x1` and `y1`; postprocessing still clamps coordinates, sorts by
confidence, applies class-aware NMS, and returns pixel `xyxy`.

The standard single-image flow recipe keeps the released text/image guidance
split (`4.0` / `2.0`) and ten updates by default. The compositional
`any2any(cfg=...)` surface deliberately applies its one explicit scale to both
guidance channels.

MODUS normal rasters are reoriented at the public boundary. `NormalMap` follows
LibreYOLO's camera frame (`+x` right, `+y` down, `+z` into the scene) with
vectors facing the camera; a fronto-parallel surface is `(0, 0, -1)`. All
image-derived `any2any()` inputs must share one aligned canvas.

## Checkpoint and licensing boundary

The Apache-2.0 code port is pinned to
`EPFL-VILAB/Modus@c299ef0fbba1cfe7c93336c45d7085afd770c0fa`.
The upstream CC BY-NC 4.0 `modeling_utils.py` file is excluded. The few standard
components needed at that boundary use the already audited clean,
permissively sourced implementations documented by the SenseNova NOTICE.

Weights are not bundled, mirrored, renamed, or published in another precision.
The loader accepts local files or fetches directly from
`EPFL-VILAB/MODUS@8428a81602c19141e422b1e1795dddcb5d2bc14b`.
The upstream model card declares custom `license: other` / `bagel-derived`
terms and requests research-only use. A load-time notice makes clear that the
Apache source license does not grant checkpoint rights.
A fresh upstream download requires credentials from the user's own Hugging
Face account; a complete local snapshot remains usable without authentication.

The released `llm_config.json` retains a stale base vocabulary size. The
safetensor embedding row count and rebuilt tokenizer are authoritative and
must agree exactly before dispatch. No remote Python code is executed.

The pinned Stage 2/3 training and inference configs disable generic target
instructions and per-modality latent normalization. Runtime follows those
released settings; phrase grounding keeps its dedicated
`[start grounding the phrase]` prompt.

## Precision decision

BF16 is the reference tier. `dtype="fp8"` performs a local-only, streaming
E4M3 weight conversion with per-output-channel scales for eligible linear
layers in both decoder experts. Numerically sensitive and non-decoder modules
stay BF16.

The cache key includes the immutable upstream revision or local source hash
plus the full module recipe. FP8 weights are dequantized for GEMM. This first
tier targets storage and VRAM; it makes no tensor-core speed claim.

## Testing decision

PR-gate tests use a two-layer random model and fake tokenizer. They validate
state-dict shape, the released tokenizer order, constrained token grammars,
Accelerate dispatch, result mapping, chaining, verification, and FP8 cache
arithmetic without network, external weights, or GPU.

Full checkpoint parity and metric thresholds remain explicit manual GPU gates.
Passing CPU tests does not imply those gates passed.

## Consequences

The integration provides one familiar result contract across dense prediction,
closed-set detection, grounding, and composed conditions. Its external
checkpoint remains isolated from LibreYOLO distribution, and users can audit
the precision derivative locally.

The tradeoffs are a large optional runtime, no generic validation/export path,
slower autoregressive/flow inference, and hardware-dependent quality gates that
cannot run in ordinary CI.
