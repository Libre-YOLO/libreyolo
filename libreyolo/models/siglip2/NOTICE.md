# LibreSigLIP2 - third-party notices & weight provenance

LibreSigLIP2 is a native (pure-`torch`) zero-shot, open-vocabulary image
classifier. It does **not** depend on `transformers` at runtime.

## Vendored code and assets (Apache-2.0)

- **Image + text towers** (`nn.py`) are a clean-room re-implementation of the
  SigLIP architecture. The module structure matches the reference
  implementation so the upstream `state_dict` loads directly (0 missing / 0
  unexpected keys).
- **SentencePiece tokenizer model** (`siglip2_tokenizer.model`) is shipped
  verbatim from the Apache-2.0 `google/siglip2-*` release (the multilingual
  Gemma vocabulary, 256k pieces). See the root `THIRD_PARTY_NOTICES.txt` for the
  license text.

## Architecture note

The SigLIP 2 release ships two model families. The fixed-resolution checkpoints
converted here (`siglip2-base-patch16-256` -> `b16` and
`siglip2-so400m-patch14-384` -> `so400m`) carry `model_type: "siglip"` and reuse
the original SigLIP transformer (Conv2d patch embedding, learned position
embedding, multi-head attention pooling head, last-token text pooling). Only the
NaFlex variants use the patchified SigLIP 2 vision embeddings, which are out of
scope for v1.

## Weights

The shipped checkpoints (`LibreSigLIP2b16-cls`, `LibreSigLIP2so400m-cls`) are
converted from the **Apache-2.0** Google SigLIP 2 weights:

| LibreSigLIP2 | Upstream repo                              | Resolution |
|--------------|--------------------------------------------|------------|
| `b16`        | `google/siglip2-base-patch16-256`          | 256        |
| `so400m`     | `google/siglip2-so400m-patch14-384`        | 384        |

Convert your own copy with `weights/convert_siglip2_weights.py` (needs the
`libreyolo[siglip2-convert]` extra). Conversion is a metadata-wrap: learned
parameters are unchanged; only the LibreYOLO v1.0 checkpoint metadata is added.

## Data provenance

The SigLIP 2 weights were trained by Google on the WebLI dataset, which is not
redistributed here (only the Apache-2.0 weights are rehosted). SigLIP is
multilingual; the tokenizer and text tower support many languages.
