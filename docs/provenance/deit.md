# DeiT

- **LibreYOLO module:** `libreyolo/models/deit/`, centralized classification
  postprocessing in `libreyolo/postprocess/deit.py`, and converter at
  `weights/convert_deit_weights.py`.
- **Original architecture:** `facebookresearch/deit` at commit
  `7e160fe43f0252d17191b71cbb5826254114ea5b` (Apache-2.0).
- **Native implementation source:** `huggingface/pytorch-image-models` at
  commit `e98c05a5a15e81188ec62dd5380b8f5c3251075a` (Apache-2.0), specifically
  `timm/models/vision_transformer.py` and
  `timm/layers/{attention,patch_embed,mlp}.py`.
- **Scope:** inference and top-1/top-5 validation for the plain tiny, small,
  and base patch-16 classifiers at fixed 224px. Distillation-token, DeiT III,
  384px, and training surfaces are intentionally excluded.

The complete attribution notice is in `libreyolo/models/deit/NOTICE`. The
Apache-2.0 license text is in `licenses/Apache-2.0.txt`. No code or weights
from an incompatible or unknown-license source were used.

## Source checkpoints

The source model repositories declare `license: apache-2.0` and
`pipeline_tag: image-classification`. The converter downloads
`model.safetensors` from the immutable revisions below, verifies its SHA-256,
loads it strictly into timm 1.0.28, then strictly loads the same state
dictionary into LibreYOLO. It does not rename or alter a learned tensor. The
only conversion is a LibreYOLO v1.0 metadata wrapper with the canonical
ImageNet-1k class names.

| Size | LibreYOLO filename | Source repository | Revision | Source bytes | Source SHA-256 |
|---|---|---|---|---:|---|
| `t` | `LibreDeiTt-cls.pt` | `timm/deit_tiny_patch16_224.fb_in1k` | `80e968688553f219e4a86f940ed945a23709c16f` | 22,883,348 | `21d4764d94f6c3ffdb6da3581115a0a1ee2d505537d96883b540e54766407c9e` |
| `s` | `LibreDeiTs-cls.pt` | `timm/deit_small_patch16_224.fb_in1k` | `91327a9c99f98fe6b524cd4d397b7226b80e1365` | 88,216,496 | `1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b` |
| `b` | `LibreDeiTb-cls.pt` | `timm/deit_base_patch16_224.fb_in1k` | `b78cc5532a69df6bcad9c3a8d76653fd20b31ac6` | 346,284,714 | `cd2da27b74ed7f68b599f16c77af3e1e80f01c75f9ad96029d22ce747a247e8e` |

The generated checkpoint inventory is:

| Size | Output bytes | Output SHA-256 | Parameters |
|---|---:|---|---:|
| `t` | 22,950,331 | `8228b6c94f0b28f700be6a8206937fa4935184c0ac5f6253a3ee83d69841c99c` | 5,717,416 |
| `s` | 88,283,579 | `57481624aef7ed8da37bbe911a2fbecfdf3780767aba0fb52609fe001786f3f0` | 22,050,664 |
| `b` | 346,351,547 | `e85d98a69815d40fe3275ad602c64de4ec049f92015a0c6ca0e7a01ea51c444d` | 86,567,656 |

The public mirrors are
[`t`](https://huggingface.co/LibreYOLO/LibreDeiTt-cls),
[`s`](https://huggingface.co/LibreYOLO/LibreDeiTs-cls), and
[`b`](https://huggingface.co/LibreYOLO/LibreDeiTb-cls). Each repository is
public, contains exactly the canonical checkpoint plus `.gitattributes`,
`README.md`, `LICENSE`, and `NOTICE`, and belongs to the LibreYOLO
Classification collection. The Hub LFS SHA-256 and byte count match the table
above. All three bare filenames were then downloaded into a new empty working
directory and strict-loaded through the public `LibreYOLO(...)` factory.

## Ported surface

| LibreYOLO surface | Pinned timm source |
|---|---|
| `nn.py` patch embedding and tokens | `timm/layers/patch_embed.py`, `timm/models/vision_transformer.py` |
| `nn.py` attention, MLP, and transformer blocks | `timm/layers/attention.py`, `timm/layers/mlp.py`, `timm/models/vision_transformer.py` |
| `nn.py` classifier graph and initialization-compatible state layout | `timm/models/vision_transformer.py` |
| `utils.py` evaluation preprocessing | the pinned checkpoint `pretrained_cfg` contract |
| ImageNet WNID validation mapping | `timm/data/_info/imagenet_synsets.txt` |

LibreYOLO implements this graph natively with PyTorch and SDPA. There is no
runtime dependency on timm, Hugging Face Hub, or safetensors. Those packages
are used only by the external-data parity test and the conversion script.

## Preprocessing and published accuracy

All three checkpoints use a 224x224 center crop after a bicubic shorter-edge
resize to `floor(224 / 0.9) = 248`, followed by RGB conversion and ImageNet
normalization with mean `(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`.

The pinned timm results table reports the following 50,000-image ImageNet-1k
validation metrics with that contract:

| Size | Top-1 | Top-5 |
|---|---:|---:|
| `t` | 72.190 | 91.100 |
| `s` | 79.856 | 95.056 |
| `b` | 81.980 | 95.740 |

The gated ImageNet validation set is not distributed with LibreYOLO and was
not present in the porting environment, so those full-dataset figures were not
claimed as a local rerun. Instead, every released checkpoint was compared
directly with the pinned timm graph on the same input and produced
`max_abs_diff == 0.0`. The unchanged learned tensors and exact logits establish
that the native graph does not change the source model's accuracy under the
same evaluation transform.

The bundled synset index also lets `model.val()` consume the conventional
`train/<wnid>/` and `val/<wnid>/` layout, including proper absolute head
indices for ImageNet subsets such as Imagenette. Checkpoint results continue
to expose human-readable names.

## Measured evidence

- Strict state-dictionary loading and exact eager logits for all three pinned
  checkpoints: `max_abs_diff == 0.0`.
- Public prediction smoke on the bundled sample image, including readable
  ImageNet labels and batched classification output.
- Public checkpoints on the 3,925-image Imagenette validation split (top-1 /
  top-5): `t` `0.704713 / 0.923057`, `s` `0.802803 / 0.968153`, and `b`
  `0.844586 / 0.985478`. This exercises `model.val()` with WNID folders and is
  a subset smoke, not a replacement for the published ImageNet-1k figures.
- Tiny checkpoint ONNX Runtime parity: raw-logit maximum absolute difference
  `5.722e-6`, probability maximum absolute difference `2.384e-7`, and exact
  top-1/top-5 indices.
- Tiny checkpoint TorchScript parity: bit-identical raw logits and
  probabilities, with exact top-1/top-5 indices.
- Tiny checkpoint OpenVINO 2026.2 parity: raw cosine similarity `0.99999833`,
  probability cosine similarity `0.99999994`, and exact top-1/top-5 indices.
- Tiny checkpoint TensorRT 10.16 FP16 parity on an NVIDIA RTX 5070 Ti: raw
  cosine similarity `0.99998593`, probability cosine similarity `0.999997735`,
  finite outputs, and exact top-1/top-5 indices. TensorRT FP32 was not promoted
  because tactic selection exceeded the available workspace on this device.
