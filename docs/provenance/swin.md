# Swin Transformer V1 classification

- **LibreYOLO module:** `libreyolo/models/swin/`, classifier postprocessing in
  `libreyolo/postprocess/swin.py`, and converter at
  `weights/convert_swin_weights.py`.
- **Reference code:** `huggingface/pytorch-image-models` v1.0.28, commit
  `8ef73809f622e0031bd7f4940265734aef8b9978`, Apache-2.0.
- **Model lineage:** `microsoft/Swin-Transformer`, commit
  `f82860bfb5225915aca09c3227159ee9e1df874d`, MIT.
- **Released-weight lineage:** `SwinTransformer/storage` v1.0.0, commit
  `3cc359915d3a6079b176a871f68d5fb0d8dfdea2`, MIT.
- **Scope:** image classification inference, ImageNet-style top-1/top-5
  validation, and fixed-224 export. The upstream training recipe is not
  implemented and `train()` raises `NotImplementedError`.

The Swin tower was already a shared LibreYOLO component used by Grounding DINO
and OMDet-Turbo. This port leaves that shared source unchanged. It adds a
standalone classifier wrapper whose final-stage shift behavior and parameter
names match timm's fixed-224 Swin V1 graph.

## Source and converted checkpoints

The converter downloads the pinned timm Hugging Face snapshots below. Each
source card declares `license: mit`. Learned tensors are loaded strictly and
saved unchanged; conversion adds only the LibreYOLO checkpoint schema and the
canonical ImageNet-1k class names.

| Size | timm tag | Pinned HF revision | Source bytes | Source SHA-256 | LibreYOLO bytes | LibreYOLO SHA-256 |
|---|---|---|---:|---|---:|---|
| `t` | `swin_tiny_patch4_window7_224.ms_in1k` | `4cc3a7275b50b53a7bec45f32c236ebe64227cff` | 114,286,722 | `fb01861f793143135fa0d6cd97b1631e4b33eaa3ee162bbea9e62de1c76ebac1` | 113,242,787 | `81844e9bbe8edba1d0e0c204341126d2ed77fd0905eb96aec1fad188c02c71fe` |
| `s` | `swin_small_patch4_window7_224.ms_in1k` | `b59828b00f6fac85d99c8eac48094b605b6e7ec3` | 200,037,522 | `8fa31b116680e02e4ad6ad06eb29a1b9ca56bd93a2e88510a0bc7e82e0e2024f` | 198,574,595 | `6eaf6826fb084a708d7ecb10620d6689d6c568901dd93646752dcf3fd8bcca19` |
| `b` | `swin_base_patch4_window7_224.ms_in1k` | `160443c7878650977f11a3a89d4ed685b001a304` | 352,685,652 | `6544e46498082f24e90b3e5269d909dad03aa016db8711777313c162e136420c` | 351,223,491 | `3e5172822afef8c813a944617e97cc9402016744dcba2bbda4a923b6e16f8c29` |
| `l` | `swin_large_patch4_window7_224.ms_in22k_ft_in1k` | `e05e58ff5362edd212120dffaff3206a633c5534` | 787,742,820 | `ccdcb5b425de65ed85875d5897681a72f7406c3fc07e087e933bace0c83807fd` | 786,279,875 | `ad4d6efdd55b04c255482fa775240fd88a4097a8588be0be0633c6596ffbad88` |

Tiny, Small, and Base are ImageNet-1k releases. Large was pretrained on
ImageNet-22k and fine-tuned on ImageNet-1k. All four hosted LibreYOLO weight
repositories carry the verbatim MIT license and a NOTICE that records this
lineage.

## Published LibreYOLO repositories

| Size | Repository | Initial revision |
|---|---|---|
| `t` | `LibreYOLO/LibreSwint-cls` | `547aca44a0126fa33a401baafba461f0911a12e1` |
| `s` | `LibreYOLO/LibreSwins-cls` | `7713f34331661200ddd323a6de980a3cecd94efb` |
| `b` | `LibreYOLO/LibreSwinb-cls` | `9f67301f3cc91119e6dec348be1797f475b80df7` |
| `l` | `LibreYOLO/LibreSwinl-cls` | `caa2c8bd8b0c73e36d602cd445dc213993e7af50` |

Each public repository was verified server-side with exactly five files:
`.gitattributes`, `README.md`, `LICENSE`, `NOTICE`, and its canonical `.pt`.
The LFS object hashes match the converted hashes above, every card declares
MIT and `pipeline_tag: image-classification`, and all four are members of the
LibreYOLO Models collection. A clean local-weight test downloaded each bare
canonical filename, rechecked its SHA-256, strict-loaded it, and ran prediction
through `LibreYOLO`.

## Measured evidence

- Strict state-dict loading succeeds for `t`, `s`, `b`, and `l` with no missing
  or unexpected keys.
- Pinned timm pretrained-logit parity is bit-exact for all four sizes:
  `max_abs_diff == 0.0` on a seeded CUDA input at 224.
- The existing shared 640px Swin backbone parity gate remains bit-exact at all
  four feature stages.
- On the bundled parkour image, the converted Tiny checkpoint returns 1,000
  normalized probabilities and top-1 ImageNet class 880 (`unicycle`).
- Trained Tiny public-runtime probability parity on the same image:
  ONNX maximum absolute difference `2.09e-7`, TorchScript `0.0`, OpenVINO
  cosine similarity `0.9999986`, and TensorRT FP32 cosine similarity
  `0.9999994`. Every runtime preserves top-1 class 880.

## Compatibility boundary

LibreSwin accepts only Swin V1 patch-4/window-7 classifier checkpoints with a
fixed classification head. It rejects Swin V2 (`cpb_mlp` / `logit_scale`),
window-12 releases, backbone-only checkpoints, and the unrelated SwinIR family.
Bidirectional discriminator tests also keep it disjoint from ResNet, ConvNeXt,
EfficientNetV2, MobileNetV4, CLIP, DINOv2, and SAM-style ViT checkpoints.
