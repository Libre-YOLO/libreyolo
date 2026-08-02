# Faster R-CNN

- **LibreYOLO module:** `libreyolo/models/faster_rcnn/`, postprocessing in
  `libreyolo/postprocess/faster_rcnn.py`, and converter at
  `weights/convert_faster_rcnn_weights.py`.
- **Upstream code:** `pytorch/vision` at commit
  `336d36e8db990a905498c73933e35231876e28bc` (torchvision v0.26.0).
- **Code license:** BSD-3-Clause, Copyright (c) Soumith Chintala 2016. The full
  notice is reproduced in `libreyolo/models/faster_rcnn/NOTICE`.
- **Scope:** box detection inference only. RPN/RoI training matching, sampling,
  and losses are intentionally excluded; `train()` raises
  `NotImplementedError`.
- **Verification:** all four released variants have exact eager-mode parity
  with the pinned upstream implementation, including RPN head tensors, RoI
  classifier/regressor tensors, and final boxes/labels/scores.

## Source checkpoints

The official files below were downloaded from PyTorch's model host and checked
by SHA-256. Conversion preserves every tensor and only adds LibreYOLO checkpoint
metadata. The files are not included in this repository.

| Size | LibreYOLO filename | Upstream variant | Bytes | SHA-256 |
|---|---|---|---:|---|
| `n` | `LibreFasterRCNNn.pt` | `fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth` | 77,844,807 | `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5` |
| `s` | `LibreFasterRCNNs.pt` | `fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth` | 77,844,807 | `fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533` |
| `m` | `LibreFasterRCNNm.pt` | `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` | 167,502,836 | `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384` |
| `l` | `LibreFasterRCNNl.pt` | `fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth` | 175,221,657 | `dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf` |

Official URLs are the filename above appended to
`https://download.pytorch.org/models/`. The pinned weight enum records COCO
box AP of 22.8 / 32.8 / 37.0 / 46.7 for `n` / `s` / `m` / `l` respectively.

### Weight-license status

The torchvision source is unambiguously BSD-3-Clause. Its README separately
states that pretrained models may have their own licenses or terms derived
from the training data and places the permission check on the user. No
checkpoint-specific permissive license was found for these four COCO files.
LibreYOLO therefore does **not** rehost them or claim they are BSD-licensed.
The converter is available for users who have separately established their
right to use a local checkpoint. A permissive checkpoint license must be
established before adding LibreYOLO Hugging Face mirrors or public nightly
catalog rows.

## Ported surface

| LibreYOLO surface | Pinned torchvision source |
|---|---|
| `nn.py` -- Faster R-CNN orchestration and builders | `models/detection/faster_rcnn.py`, `generalized_rcnn.py`, `backbone_utils.py` |
| `nn.py` -- proposal generation | `models/detection/rpn.py`, `anchor_utils.py` |
| `nn.py` -- RoI box head and final class-wise NMS | `models/detection/roi_heads.py` |
| `nn.py` -- transforms, box coder, FPN wrapper | `models/detection/transform.py`, `_utils.py`, `backbone_utils.py` |

The runtime still imports torchvision's BSD-licensed low-level building blocks
for the MobileNetV3/ResNet backbones, anchor generation, FPN, RoIAlign, and box
operations. The two-stage orchestration and inference-only RPN/RoI logic live
in LibreYOLO so the family is not a wrapper around an upstream model class.

## Variant and class contracts

- `n`: MobileNetV3-Large FPN, internal min/max resize 320/640.
- `s`: MobileNetV3-Large FPN, internal min/max resize 800/1333.
- `m`: ResNet-50 FPN v1 with FrozenBatchNorm, internal resize 800/1333.
- `l`: ResNet-50 FPN v2 with batch-normalized FPN and a deeper RPN/box head,
  internal resize 800/1333.

The `n` and `s` state dictionaries have indistinguishable tensor structures;
upstream or canonical filename hints resolve that pair. The released heads use
background plus sparse COCO category ids in 91 outputs. Public results map
those ids to LibreYOLO's contiguous 80-class convention; custom heads use
background plus `nc` classes and subtract the background index.

## Measured evidence

- Eager parity for `n`, `s`, `m`, and `l`: `max_abs_diff == 0.0` at the RPN
  head, RoI classifier/regressor, and final detections on the bundled parkour
  image.
- Official `n` checkpoint on COCO128: mAP50-95 `0.34931876998581457`, mAP50
  `0.5067003520164212`.
- Fixed 320x320, batch-one ONNX Runtime parity for `n`: identical output shapes
  and labels, boxes within `5e-4`, scores within `1e-5`; unified backend
  prediction returns the same five detections on the parity image.
- ONNX emits final boxes/scores/labels after the model's class-wise RoI NMS.
  Exported-backend parsing deliberately bypasses generic NMS.
