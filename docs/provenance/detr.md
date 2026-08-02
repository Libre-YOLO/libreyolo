# detr

- **LibreYOLO module:** `libreyolo/models/detr/`, centralized postprocessing
  in `libreyolo/postprocess/detr.py`, converter at
  `weights/convert_detr_weights.py`, and parity runner at
  `weights/parity_detr.py`.
- **Upstream:** https://github.com/facebookresearch/detr, the original
  Detection Transformer implementation. Ported from commit
  `29901c51d7fe8712168b8d0d64351170bc0f83e0` (2024-03-12).
- **Code license:** Apache-2.0. The upstream root license and source headers
  were verified before any implementation source was inspected. DETR's
  `models/transformer.py` also declares PyTorch `torch.nn.Transformer`
  lineage; PyTorch is BSD-3-Clause and that attribution is preserved in
  `NOTICE`, `THIRD_PARTY_NOTICES.txt`, and `libreyolo/models/detr/NOTICE`.
- **Weights license:** Apache-2.0. These are the official checkpoints linked
  from the Apache-2.0 repository's model zoo; the official Hugging Face DETR
  card also declares Apache-2.0. Converted repositories retain that license
  verbatim and include a NOTICE.
- **Scope:** detection inference, validation, ONNX, and TorchScript. The
  upstream 500-epoch Hungarian-matching training recipe is not ported;
  `LibreDETR.train()` raises `NotImplementedError`.

## Official checkpoints

Downloaded from `https://dl.fbaipublicfiles.com/detr/` on 2026-08-02:

| Size | Official file | Bytes | SHA-256 | Upstream COCO box AP |
|---|---|---:|---|---:|
| `r50` | `detr-r50-e632da11.pth` | 166,618,694 | `e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc` | 42.0 |
| `r50dc5` | `detr-r50-dc5-f0fb7ef5.pth` | 166,618,694 | `f0fb7ef52a6b0bbd87dce4a4b8569b509f03ad082f60f936dc5b7009940295d3` | 43.3 |
| `r101` | `detr-r101-2c7b67e5.pth` | 242,846,568 | `2c7b67e52d2e687cefa3c4a5e701a148664159c96725c62984600d2d9d5c4104` | 43.5 |
| `r101dc5` | `detr-r101-dc5-a2e86def.pth` | 242,846,568 | `a2e86defc9f49cfca7df75523d8745c6aa15482a5184e8dc62a0a19119c0286e` | 44.9 |

The canonical converted files are `LibreDETRr50.pt`,
`LibreDETRr50dc5.pt`, `LibreDETRr101.pt`, and `LibreDETRr101dc5.pt`.
Their SHA-256 values at upload are, respectively:

- `4cecc1ce6fb932f174f9c7ccbd6e5bc4dfbafe479bb083994beae98bbfc49071`
- `2dd4f8c0102e68fd62d0ed22fae5adcf9b581cfe014d26c83075afc2a30ae1b2`
- `7aefe0a2f4a9078fa4a651e40121019846d55dd3b36c655e075fa816f1bae7f0`
- `0b3bfb5f872234afac784a1b8cb92745ab305dd7b28974e6c16afeaabdea00b4`

Conversion is a metadata-only wrap: every learned tensor is bitwise identical
to the official state dict. `--size` is mandatory because DC5 changes runtime
dilation but no serialized tensor shape. An official filename is cross-checked
against that argument; a renamed file cannot be distinguished as DC5 from its
state dict alone and is never guessed by runtime auto-conversion.

## Ported surface

| LibreYOLO surface | Pinned upstream source |
|---|---|
| `models/detr/nn.py` — detector, MLP | `models/detr.py` |
| `models/detr/nn.py` — ResNet backbone, frozen batch norm, joiner | `models/backbone.py` |
| `models/detr/nn.py` — sine position encoding | `models/position_encoding.py` |
| `models/detr/nn.py` — encoder and decoder | `models/transformer.py` |
| `models/detr/nn.py` — `NestedTensor` | `util/misc.py` |
| `postprocess/detr.py` | `models/detr.py` (`PostProcess`) |
| `models/detr/utils.py` | inference normalization derived from `datasets/transforms.py` |

LibreYOLO combines the native inference graph into one file, suppresses the
wasted ImageNet backbone download that the detector checkpoint immediately
overwrites, accepts a fixed same-sized tensor batch with an all-false padding
mask, and exposes a tuple export wrapper. Module names remain unchanged, so
all four official state dicts load with `strict=True` and no remapping.

## Preprocessing contract

Official COCO evaluation preserves aspect ratio: short side 800, long side at
most 1333, with a padding mask for batched images. LibreYOLO's exported-model
contract is one fixed canvas, so this port uses one PIL bilinear stretch to
800×800 followed by the same RGB ImageNet normalization. The resulting batch
has no padding and therefore uses an all-false mask. This deliberate deployment
mapping is shared by native predict/val, ONNX, and TorchScript; it is not a
claim that LibreYOLO reproduces the official aspect-preserving evaluation
recipe.

## Numerical evidence

`weights/parity_detr.py` verified the pinned upstream implementation against
the native port on an RTX 5070 Ti at both 64×64 and the shipped 800×800 canvas.
Every row was exact:

```text
r50      pred_logits max_abs_diff=0.0  pred_boxes max_abs_diff=0.0
r50dc5   pred_logits max_abs_diff=0.0  pred_boxes max_abs_diff=0.0
r101     pred_logits max_abs_diff=0.0  pred_boxes max_abs_diff=0.0
r101dc5  pred_logits max_abs_diff=0.0  pred_boxes max_abs_diff=0.0
```

Trained-checkpoint FP32 exports were then run at 800×800 through ONNX Runtime
1.26.0 CPU and TorchScript. TorchScript raw outputs were bit-exact for all four
sizes. ONNX and public-result maxima were:

| Size | Logits max abs | Normalized boxes max abs | Public boxes max abs (pixels) | Scores max abs |
|---|---:|---:|---:|---:|
| `r50` | 5.341e-4 | 2.152e-5 | 9.156e-5 | 2.384e-7 |
| `r50dc5` | 9.728e-5 | 7.570e-6 | 6.867e-5 | 2.384e-7 |
| `r101` | 2.099e-4 | 1.547e-5 | 1.221e-4 | 2.384e-7 |
| `r101dc5` | 2.432e-4 | 3.398e-5 | 6.104e-5 | 3.577e-7 |

All runtimes returned the same five classes on COCO image `000000039769`.
COCO128 validation with the fixed-square contract produced:

| Size | Batch | mAP50-95 | mAP50 |
|---|---:|---:|---:|
| `r50` | 4 | 0.5411 | 0.7541 |
| `r50dc5` | 1 | 0.5665 | 0.7645 |
| `r101` | 4 | 0.5800 | 0.7937 |
| `r101dc5` | 1 | 0.6183 | 0.8252 |

## Class-index convention

The official head has 91 COCO category-id slots plus one explicit no-object
class (`class_embed.weight` shape `(92, 256)`). Postprocessing performs
softmax over all 92 logits, removes no-object, excludes the 11 unused COCO ids
before per-query ranking, and maps the remaining ids through
`COCO91_TO_COCO80`. Both native and exported backends expose contiguous
classes 0–79 and apply no NMS.
