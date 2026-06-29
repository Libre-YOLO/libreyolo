# LibreCLIP — third-party notices & weight provenance

LibreCLIP is a native (pure-`torch`) zero-shot, open-vocabulary image
classifier. It does **not** depend on `open_clip` at runtime.

## Vendored code (MIT)

- **CLIP BPE tokenizer** — `tokenizer.py` and the merge table
  `bpe_simple_vocab_16e6.txt.gz` are vendored from OpenAI CLIP / OpenCLIP
  (both MIT). See the root `THIRD_PARTY_NOTICES.txt` for the license text.
- **Image + text towers** (`nn.py`) are a clean-room re-implementation of the
  standard CLIP architecture; the module structure matches OpenCLIP so the
  upstream `state_dict` loads directly.

## Weights

The shipped checkpoints (`LibreCLIPb32-cls`, `LibreCLIPb16-cls`) are converted
from **OpenCLIP LAION-2B** weights, which are **MIT-redistributable**:

| LibreCLIP | OpenCLIP arch | Pretrained tag         |
|-----------|---------------|------------------------|
| `b32`     | `ViT-B-32`    | `laion2b_s34b_b79k`    |
| `b16`     | `ViT-B-16`    | `laion2b_s34b_b88k`    |

Convert your own copy with `weights/convert_clip_weights.py` (needs the
`libreyolo[clip-convert]` extra).

## ⚠️ LAION data-provenance note

The LAION-2B dataset these weights were trained on had a **documented
CSAM-content history** (Stanford Internet Observatory, December 2023). LAION
subsequently released **Re-LAION**, a cleaned re-release. Anyone re-hosting or
publishing LibreCLIP weights should:

- prefer **Re-LAION-derived** checkpoints where available, and
- carry this provenance note in the model card / `pipeline_tag:
  zero-shot-image-classification` README.

LibreCLIP does **not** ship OpenAI-WIT (undisclosed-data) weights or any
non-commercial CLIP weights.
