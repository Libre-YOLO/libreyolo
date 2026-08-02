# deformable_detr

- **LibreYOLO surface:** `libreyolo/models/deformable_detr/`, postprocessing
  in `libreyolo/postprocess/deformable_detr.py`, and the standalone converter
  at `weights/convert_deformable_detr_weights.py`.
- **Architecture source:**
  https://github.com/fundamentalvision/Deformable-DETR at commit
  `11169a60c33333af00a4849f1808023eba96a931` (Apache-2.0). The audited
  canonical-LF `LICENSE` SHA-256 is
  `068413f8cf5e42e34d6e171e45a779df6dbea0a18249c0a223a3878e6de3cb27`.
- **Converter mapping source:**
  https://github.com/huggingface/transformers at commit
  `4a224b1e2182d1f8f27d1d76fb8de6ab40b7ff62` (tag `v4.25.1`, Apache-2.0).
  Only its Transformers-to-original checkpoint key mapping is adapted.
- **Copyright notices:** Copyright (c) 2020 SenseTime. All Rights Reserved.
  The upstream implementation is modified from DETR, Copyright (c) Facebook,
  Inc. and its affiliates. Both notices remain in the ported source headers and
  the repository `NOTICE`.
- **Scope:** inference only. The Hungarian matcher, criterion, auxiliary
  training losses, optimizer groups, data augmentation, and custom CUDA
  extension are not ported. Multi-scale deformable attention always uses the
  upstream pure-PyTorch `grid_sample` reference core.

## Official checkpoints and redistribution basis

The five inputs are the SenseTime Hugging Face mirrors listed below. Each model
card declares `license: apache-2.0`, which is the explicit redistribution basis
for the weights; the original README's Google Drive links are not used as a
licensing source.

| Size | SenseTime repository | Pinned revision | Source `model.safetensors` SHA-256 | Converted LibreYOLO SHA-256 | COCO AP (upstream) |
|---|---|---|---|---|---:|
| `r50ss` | https://huggingface.co/SenseTime/deformable-detr-single-scale | `e880a4ca7bbe47b33d37ed90e2948efbbdad0d44` | `82eeb57bbcdd02408afc53d5f5c874e3a7f27b5034194ae2c4475d06fceaa59b` | `a09f5f4a995f1ea74bb6e2d0d2e4c6388ae8f8072d91b569f9a1f38e42f4b419` | 39.4 |
| `r50ssdc5` | https://huggingface.co/SenseTime/deformable-detr-single-scale-dc5 | `c23332913d0ae1a8c98725e308eccba65a5933cc` | `e71afa5f5900e2e769275156494195508efcadaab4275b0cd4c80f10369dc090` | `777237ddcb4cfe84d72a1daad5e447cd1a070b212def10102981779c405eb5bf` | 41.5 |
| `r50` | https://huggingface.co/SenseTime/deformable-detr | `83ecd26945199939cb82806f988debdb71e6f43e` | `caf1e3e61283c6ce35cd2d9adaa7033cf40997d4dfe434003bcdb9085cc8cf9b` | `1f8499d1ddf0e03e999ad4f821a68375144b814d765707df015a4373941b398b` | 44.5 |
| `r50refine` | https://huggingface.co/SenseTime/deformable-detr-with-box-refine | `2e9e461623a8fdc296e19666c46c8a4389a3a6fe` | `4113700fe8aade398808424b7c5c1304cfbf886adc6450a6ca5d50a702be3373` | `84e4044553a306e07817bff9f147a50af3a51d16e43056160243a5a0d46d48c9` | 46.2 |
| `r50twostage` | https://huggingface.co/SenseTime/deformable-detr-with-box-refine-two-stage | `e74bff70d69f3e825f6cefaf179bfba707f92054` | `411bb4238a834d40fff651b1b5b7d6dd80c2dd28be1747eec7b6918674e85de6` | `7250d79708f53f2f5b8b0daec94011def1c262c4fc723993c1f58128c8002670` | 46.9 |

Conversion reconstructs the original shared-head aliases, concatenates the
Transformers query/key/value projections into upstream's `in_proj_*` tensors,
and removes only non-parameter batch-normalization tracking counters absent
from the original state dict. Learned tensor values are otherwise unchanged.
The converted checkpoints preserve the released 91-column COCO category-id
head; native and exported postprocessing remove the 11 unused ids before top-K
and expose contiguous COCO-80 classes.

## Ported surface

| LibreYOLO file | Pinned upstream source |
|---|---|
| `common.py` | `util/misc.py`, `models/backbone.py`, `models/position_encoding.py` |
| `ms_deform_attn.py` | `models/ops/modules/ms_deform_attn.py`, `models/ops/functions/ms_deform_attn_func.py` |
| `transformer.py` | `models/deformable_transformer.py` |
| `nn.py` | `models/deformable_detr.py` |
| `postprocess/deformable_detr.py` | `models/deformable_detr.py` (`PostProcess`) |
| `conversion.py` | Transformers `convert_deformable_detr_to_pytorch.py` at the pinned commit above |

The original validation transform resizes the short side to 800 with a
1333-pixel long-side cap. LibreYOLO's fixed-shape deployment contract instead
uses a direct PIL-bilinear 800 x 800 resize with ImageNet normalization and
independent x/y box scaling. Inference, validation, and exported backends share
that documented transform.

## Parity evidence

`tests/unit/test_deformable_detr_parity.py` verifies the upstream commit and
license digest, replaces only the unavailable compiled operator with upstream's
own pure-PyTorch core, and strictly loads every official checkpoint into both
implementations. At 128 x 128, every tensor leaf is bit-exact for all five
variants (`max_abs_diff == 0.0`), including auxiliary decoder outputs and the
two-stage encoder outputs.

`tests/e2e/test_deformable_detr_onnx.py` exports every official variant at the
native 800 x 800 canvas and checks the ONNX graph, raw tensors, metadata, and
`OnnxBackend.predict()` against native PyTorch on the same real image. All five
retain identical detection counts and classes; matched box IoU is above 0.95
and mean score error is below 0.01. The two-stage graph is traced on CPU because
PyTorch 2.11's legacy exporter can terminate while lowering its CUDA top-K
graph; the live model is restored to its original device after export.

The catalog gate `tests/e2e/test_val_coco128.py -k deformable` was run on all
five canonical checkpoints at batch 1. The measured COCO128 mAP50-95 values
were `0.4500` (`r50ss`), `0.4613` (`r50ssdc5`), `0.4887` (`r50`), `0.5510`
(`r50refine`), and `0.5732` (`r50twostage`); every variant clears the shared
`0.18` catastrophic-regression floor.
