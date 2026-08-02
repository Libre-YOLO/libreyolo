# dinodetr

- **LibreYOLO surface:** `libreyolo/models/dinodetr/`, postprocessing in
  `libreyolo/postprocess/dinodetr.py`, and the standalone converter at
  `weights/convert_dinodetr_weights.py`.
- **Architecture source:** https://github.com/IDEA-Research/DINO at commit
  `d84a491d41898b3befd8294d1cf2614661fc0953` (Apache-2.0). The audited
  canonical-LF `LICENSE` SHA-256 is
  `465ed5f2f9d61880f2f37c7b8de6c7342813c80c92bc3542ea7dd55422c4637c`.
- **Copyright and lineage:** Copyright (c) 2022 IDEA. The pinned source retains
  Apache-2.0 notices for Conditional DETR (Microsoft), DETR (Facebook), and
  Deformable DETR (SenseTime), identifies DAB-DETR and DN-DETR as architectural
  predecessors, and derives its Swin-L backbone from Microsoft's MIT-licensed
  Swin Transformer. These notices remain in the model `NOTICE` and the root
  `THIRD_PARTY_NOTICES.txt`.
- **Scope:** inference only. Denoising training, Hungarian matching, criteria,
  auxiliary training losses, optimizers, data augmentation, and the compiled
  CUDA extension are not ported. Multi-scale deformable attention uses the
  pure-PyTorch reference core already audited for LibreDeformableDETR.

## Official checkpoints and redistribution basis

The inputs are the three official checkpoints linked from the pinned DINO
README and downloaded from the authors' Google Drive release folder. The
upstream repository releases DINO under Apache-2.0 but does not put a separate
license file or license metadata beside each checkpoint. The redistribution
basis is therefore the license declaration of the releasing repository, not a
publisher-confirmed checkpoint-specific grant. Each LibreYOLO mirror ships the
verbatim upstream Apache-2.0 license, a notice explaining this basis, and the
source and converted hashes below.

| Size | Official file | Source SHA-256 | Converted LibreYOLO SHA-256 | COCO AP (upstream) |
|---|---|---|---|---:|
| `r50` | `checkpoint0011_4scale.pth` | `0bcd6b0c33d60ed33461ce6f02ce5797a819c7c02eb7e15b76adfb6df307955a` | `8b2243075a086e17c898d80ceb939784b1b56d44c5aca26256b4914f3b8d5d03` | 49.0 |
| `r50s5` | `checkpoint0011_5scale.pth` | `1ccc1b6b7139813e4d3bfbeecfcf88347ebc226829769a0bf16c4a114c275cc0` | `8dd59b36fff9750835fac7eb14c07a00f244bc0ec3f205dceac74907f0ef723a` | 49.4 |
| `swinl` | `checkpoint0027_5scale_swin.pth` | `17ddce1592816a0c63a2edc94d4a0877ffeb086f397a6657e151c703a4c850b5` | `1532135001dff0fa6ba688eac52df9d92af83c2c6bb13a06139fbfcd81574118` | 58.5 |

The public LibreYOLO mirrors are pinned at revisions
`462f5afabb53146d933827814199564a9bd6ed93` (`r50`),
`7d04c21564296ed31385c2f93db749a568940ab1` (`r50s5`), and
`c73afa85c93fced1cc31e23a9925c96638d140b0` (`swinl`). The external ONNX
test downloads only these immutable revisions.

Conversion does not rename or transform learned tensors. It strips an optional
distributed `module.` prefix, strictly loads the complete native state dict,
then wraps it with LibreYOLO checkpoint-schema metadata. The released 91-column
COCO category-id head is retained. Native and exported postprocessing remove
the 11 unused category ids before top-K selection and expose contiguous COCO-80
classes.

## Ported surface

| LibreYOLO file | Pinned DINO source |
|---|---|
| `common.py` | `models/dino/backbone.py`, `models/dino/position_encoding.py`, `util/misc.py` |
| `swin.py` | `models/dino/swin_transformer.py` |
| `transformer.py` | `models/dino/deformable_transformer.py`, `models/dino/utils.py` |
| `nn.py` | `models/dino/dino.py` and the export adapter |
| `model.py` | LibreYOLO loading, preprocessing, validation, and prediction adapter |
| `postprocess/dinodetr.py` | the equivalent DINO inference selection, shared with LibreDeformableDETR |

The original validation transform resizes the short side to 800 with a
1333-pixel long-side cap. LibreYOLO's fixed-shape deployment contract instead
uses a direct PIL-bilinear 800 x 800 resize with ImageNet normalization and
independent x/y box scaling. Native inference, validation, and exported
backends share that transform.

## Parity and export evidence

`tests/unit/test_dinodetr_parity.py` verifies the upstream commit, license
digest, and all three source checkpoint hashes before importing the reference.
It replaces only the unavailable compiled attention operator with upstream's
own pure-PyTorch reference function, then strictly loads every checkpoint into
both implementations. At 240 x 240 for `r50` and 128 x 128 for the larger
variants, every output tensor leaf is bit-exact (`max_abs_diff == 0.0`).

ONNX Runtime was checked against native PyTorch for all three converted
checkpoints. Maximum absolute differences were `3.00e-5` logits and `3.96e-6`
boxes (`r50`), `9.78e-5` and `4.17e-6` (`r50s5`), and `1.55e-4` and `1.29e-5`
(`swinl`). The public `r50` backend comparison on a real image retained the
same 85 detections and classes with maximum matched box/score difference below
`3.7e-4`.
