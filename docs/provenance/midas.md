# MiDaS provenance

- Upstream code: `isl-org/MiDaS`
- Pinned code commit: `454597711a62eabcbf7d1e89f3fb9f569051ac9b`
- Code license: MIT, copyright Intel ISL
- Runtime encoder dependency: `timm>=1.0.28,<1.1`, Apache-2.0
- LibreYOLO module: `libreyolo/models/midas/`
- Converter: `weights/convert_midas_weights.py`
- Parity runner: `weights/parity_midas.py`

LibreYOLO ports two complementary official inference variants. Module names
and learned tensors are unchanged; conversion adds checkpoint schema v1.0
metadata only (`task="depth"`, `nc=1`, `names={0: "depth"}`).

## Official checkpoints

Downloaded from official `isl-org/MiDaS` GitHub releases on 2026-08-03:

| Size | Official variant and file | Source URL | Bytes | SHA-256 |
|---|---|---|---:|---|
| `s` | MiDaS v2.1 Small, `midas_v21_small_256.pt` | `https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt` | 85,761,505 | `70d6b9c891758c67f974a6097fb0c608c7ee67fb81ac3e5588847d5596d56fca` |
| `l` | DPT-Large, `dpt_large_384.pt` | `https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt` | 1,376,378,527 | `2f21e586477d90cb9624c7eef5df7891edca49a1c4795ee2cb631fd4daa6ca69` |

The runtime download hook checks these hashes before loading the third-party
pickle with PyTorch's restricted `weights_only=True` path, confirms the
architecture signature, adds strict LibreYOLO metadata, and atomically
publishes the local canonical file. The same raw state dict can be converted
offline to `LibreMiDaSs-depth.pt` or `LibreMiDaSl-depth.pt`.

## Weight distribution decision

The official checkpoints are assets of the MIT-licensed MiDaS repository, but
they were trained across a mixture of depth datasets whose redistribution and
commercial-use terms have not been cleared as a set. ADR 0006 requires that
clearance before LibreYOLO hosts a depth checkpoint. LibreYOLO therefore does
not mirror these files on its Hugging Face organization. Auto-download links
the official release assets directly, verifies the exact bytes above, and
wraps them locally. Hosting can be reconsidered only after a maintainer records
the training-data clearance required by ADR 0006.

## Architecture and preprocessing

- `s`: EfficientNet-Lite3 encoder, 256-pixel upper-bound aspect resize,
  ImageNet mean/std normalization.
- `l`: ViT-L/16 DPT encoder/decoder, 384-pixel minimal aspect resize, mean and
  std 0.5.

Both models emit relative inverse depth. Higher values mean closer, but the
map is defined only up to an unknown per-image scale and shift and has no
metric unit. Zero-shot validation fits scale and shift independently per image
as required by the shared depth contract.

## Numerical evidence

The parity runner imports only clean checkouts at the commits above and at
`rwightman/gen-efficientnet-pytorch` commit
`771ce082b2ce6d033f55b3d47c1f77389ad3c180` (Apache-2.0) for the official
v2.1 Small reference. It checksum-verifies both checkpoints. Official
preprocessing and native preprocessing have `max_abs_diff=0`; the full native
outputs are bit-exact for both variants:

```text
size=s output_shape=(1, 256, 256) max_abs_diff=0 tensor_equal=True
size=l output_shape=(1, 384, 384) max_abs_diff=0 tensor_equal=True
```

Trained-checkpoint fixed-canvas export tests reload each artifact through the
public factory and compare two input-sensitive images. TorchScript reached at
least 82.49 dB PSNR (`l` was bit-exact); ONNX Runtime reached at least 70.58
dB. Every measured native signal was over three million times the conversion
error, above the required 20x guard.
