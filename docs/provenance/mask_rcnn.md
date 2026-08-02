# Mask R-CNN

- **LibreYOLO module:** `libreyolo/models/mask_rcnn/`, with shared two-stage
  inference code from `libreyolo/models/faster_rcnn/` and conversion at
  `weights/convert_mask_rcnn_weights.py`.
- **Upstream code:** `pytorch/vision` at commit
  `336d36e8db990a905498c73933e35231876e28bc` (torchvision v0.26.0).
- **Code license:** BSD-3-Clause, Copyright (c) Soumith Chintala 2016. The full
  notice is reproduced in `libreyolo/models/mask_rcnn/NOTICE`.
- **Scope:** detection and instance-segmentation inference. RPN/RoI training,
  mask sampling, and mask loss are intentionally excluded; `train()` raises
  `NotImplementedError`.

## Source checkpoint

The official file below was downloaded from PyTorch's model host, checked by
SHA-256, and strict-loaded into the native graph. Conversion preserves every
tensor and only adds LibreYOLO checkpoint metadata. The file is not included
in this repository.

| Size | LibreYOLO filename | Upstream variant | Bytes | SHA-256 |
|---|---|---|---:|---|
| `r50` | `LibreMaskRCNNr50.pt` | `maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth` | 185,828,065 | `73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e` |

Official URL:
`https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth`.
The pinned weight enum records COCO box AP 47.4, mask AP 41.8, and 46,359,409
parameters.

### Weight-license status

The torchvision source is BSD-3-Clause. The publisher does not attach a
separate license file to this checkpoint object, so the redistribution basis
is **BSD-3-Clause implied by the releasing project**, not an explicit
checkpoint-specific grant. Torchvision separately warns that pretrained models
may have their own terms derived from training data and leaves use-case
permission to the user. COCO annotations are CC BY 4.0; source images retain
their individual Flickr terms.

The weight repository therefore carries the verbatim torchvision BSD-3-Clause
license, an attribution notice, and this caveat. Its card must never describe
the checkpoint license as publisher-confirmed.

## Ported surface

| LibreYOLO surface | Pinned torchvision source |
|---|---|
| shared backbone, FPN, RPN, box head, and transform | `models/detection/faster_rcnn.py`, `generalized_rcnn.py`, `backbone_utils.py`, `rpn.py`, `transform.py` |
| mask RoI head and class-specific mask selection | `models/detection/mask_rcnn.py`, `roi_heads.py` |

The runtime continues to import torchvision's BSD-licensed low-level ResNet,
FPN, anchor, RoIAlign, and box operations. It does not wrap or instantiate an
upstream Mask R-CNN model at runtime.

## Measured evidence

- Eager parity for `r50` on the bundled parkour image: `max_abs_diff == 0.0`
  for RPN outputs, box-head outputs, final boxes/labels/scores, and raw
  pre-sigmoid mask logits.
