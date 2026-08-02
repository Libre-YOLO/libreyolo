# RetinaNet

- **LibreYOLO module:** `libreyolo/models/retinanet/`, postprocessing in
  `libreyolo/postprocess/retinanet.py`, and converter at
  `weights/convert_retinanet_weights.py`.
- **Upstream code:** `pytorch/vision` at commit
  `336d36e8db990a905498c73933e35231876e28bc` (torchvision v0.26.0).
- **Code license:** BSD-3-Clause, Copyright (c) Soumith Chintala 2016. The full
  notice is reproduced in `libreyolo/models/retinanet/NOTICE`.
- **Scope:** box detection inference only. Focal-loss target assignment and
  training losses are intentionally excluded; `train()` raises
  `NotImplementedError`.
- **Verification:** both released ResNet-50 variants have exact eager-mode
  parity with the pinned upstream implementation at every FPN feature, raw
  classification/regression head, and final box/class/score output.

## Source checkpoints

The official files below were downloaded from PyTorch's model host and checked
by SHA-256. Conversion preserves every learned tensor and state-dict key and
only adds LibreYOLO checkpoint metadata. The files are not included in this
repository.

| Size | LibreYOLO filename | Upstream variant | Bytes | SHA-256 |
|---|---|---|---:|---|
| `r50` | `LibreRetinaNetr50.pt` | `retinanet_resnet50_fpn_coco-eeacb38b.pth` | 136,595,076 | `eeacb38b7cec8cf93c57867e05eaab621047f19b0d2ec5accaa405f690da15b7` |
| `r50v2` | `LibreRetinaNetr50v2.pt` | `retinanet_resnet50_fpn_v2_coco-5905b1c5.pth` | 153,130,989 | `5905b1c544219215e544dbe319720397bc4e68de61a733a59350d7976645b769` |

Official URLs are the filename above appended to
`https://download.pytorch.org/models/`. The pinned weight enums report COCO
val2017 box AP of 36.4 and 41.5 for `r50` and `r50v2`, respectively.

### Weight-license status

The torchvision source is BSD-3-Clause. The publisher does not attach a
separate license file to these checkpoint objects, so the redistribution basis
is **BSD-3-Clause implied by the releasing project**, not an explicit
checkpoint-specific grant. Torchvision's README separately warns that
pretrained models may have their own terms derived from their training data and
leaves use-case permission to the user. The COCO annotations are CC BY 4.0; the
source images retain their individual Flickr terms.

The project maintainer approved rehosting on that disclosed implied-license
basis. Every LibreYOLO weight repository therefore carries the verbatim
torchvision BSD-3-Clause license, an attribution notice, and the caveat above;
the cards must never describe the checkpoint license as publisher-confirmed.
`weights/upload_retinanet_hf.py` builds and validates the exact five-file
repositories before publication. The public mirrors are
[`r50`](https://huggingface.co/LibreYOLO/LibreRetinaNetr50) and
[`r50v2`](https://huggingface.co/LibreYOLO/LibreRetinaNetr50v2). Each is
public, contains exactly `.gitattributes`, `README.md`, `LICENSE`, `NOTICE`,
and its canonical checkpoint, belongs to the LibreYOLO Models collection, and
has passed a strict-load CUDA prediction from a fresh bare-filename download.

## Ported surface

| LibreYOLO surface | Pinned torchvision source |
|---|---|
| `nn.py` -- RetinaNet orchestration and heads | `models/detection/retinanet.py` |
| `nn.py` -- P3-P7 anchors and box decode | `models/detection/anchor_utils.py`, `models/detection/_utils.py` |
| `nn.py` -- ResNet-FPN construction | `models/detection/backbone_utils.py` |
| `utils.py` and postprocessing | `models/detection/transform.py`, `models/detection/retinanet.py` |

The runtime imports torchvision's BSD-licensed low-level ResNet, FPN,
FrozenBatchNorm, and box-operation primitives. The one-stage orchestration,
heads, anchors, decode, preprocessing, and candidate selection live in
LibreYOLO, so the family is not a wrapper around an upstream model class.

## Variant and class contracts

- `r50`: ResNet-50 FPN v1 with FrozenBatchNorm and convolution-only heads.
- `r50v2`: ResNet-50 FPN v2 with BatchNorm in the backbone and GroupNorm in
  the deeper classification/regression heads.
- Both variants resize the short side to 800, cap the long side at 1333,
  normalize with ImageNet statistics, and pad the bottom/right edge to a
  multiple of 32.
- Released heads use sparse 91-way COCO category ids. Public results map those
  ids to LibreYOLO's contiguous COCO-80 convention.

## Measured evidence

- Eager parity for `r50` and `r50v2`: `max_abs_diff == 0.0` for every P3-P7
  feature, raw classification/regression tensor, and final detection on the
  bundled parkour image.
- COCO128 mAP50-95 / mAP50: `r50` `0.4994 / 0.7177`; `r50v2`
  `0.5196 / 0.7316`.
- Dynamic-spatial, batch-one ONNX Runtime parity for `r50`: the graph was
  exported with an 800x800 dummy and run at 800x1216; decoded boxes differ by
  at most `3.7e-4`, scores by `8.1e-7`, final boxes by `1.3e-4`, and classes
  are identical.
- ONNX emits decoded boxes and contiguous class scores. Thresholding and
  per-level top-K selection run before the unified backend's class-aware NMS;
  RetinaNet is deliberately not classified as an NMS-free family.
