# SSD300

- **LibreYOLO module:** `libreyolo/models/ssd/`, postprocessing in
  `libreyolo/postprocess/ssd.py`, and converter at
  `weights/convert_ssd_weights.py`.
- **Upstream code:** `pytorch/vision` at commit
  `336d36e8db990a905498c73933e35231876e28bc` (torchvision v0.26.0).
- **Code license:** BSD-3-Clause, Copyright (c) Soumith Chintala 2016. The full
  notice is reproduced in `libreyolo/models/ssd/NOTICE`.
- **Scope:** fixed-300 box detection inference only. MultiBox matching,
  hard-negative mining, and training losses are intentionally excluded;
  `train()` raises `NotImplementedError`.
- **Verification:** the released COCO variant has exact eager-mode parity with
  the pinned upstream implementation at preprocessing, regression head,
  classification head, default boxes, and final detections.

## Source checkpoint

The official file below was downloaded from PyTorch's model host and checked
by SHA-256. Conversion preserves every learned tensor and only adds LibreYOLO
checkpoint metadata. The file is not included in this repository.

| LibreYOLO filename | Upstream file | Bytes | SHA-256 |
|---|---|---:|---|
| `LibreSSD300.pt` | `ssd300_vgg16_coco-b556d3b4.pth` | 142,594,222 | `b556d3b43ab6c3f63d81bfb8835fe8756ac22da664357da100dccf96b6a6b42d` |

Official URL:
`https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth`.
The pinned weight enum records COCO box AP of 25.1.

### Weight-license status

The torchvision source is BSD-3-Clause. The publisher does not attach a
separate license file to the checkpoint object, so the redistribution basis is
**BSD-3-Clause implied by the releasing project**, not an explicit
checkpoint-specific grant. Torchvision's README separately warns that
pretrained models may have their own terms derived from their training data
and leaves use-case permission to the user. COCO annotations are CC BY 4.0;
the source images retain their individual Flickr terms.

The separate LibreYOLO weight repository therefore carries the verbatim
torchvision BSD-3-Clause license, attribution notice, and caveat above; it must
never describe the checkpoint license as publisher-confirmed. The public
mirror is [`LibreSSD300`](https://huggingface.co/LibreYOLO/LibreSSD300).

### VGG-16 feature-weight lineage

Torchvision's SSD recipe initializes the backbone from the Oxford Visual
Geometry Group's VGG-16 features-only weights before training the detector on
COCO. Oxford publishes the model under CC BY 4.0 and requests citation of
Karen Simonyan and Andrew Zisserman's "Very Deep Convolutional Networks for
Large-Scale Image Recognition" (ICLR 2015). The source and license are:

- `https://www.robots.ox.ac.uk/~vgg/research/very_deep/`
- `https://creativecommons.org/licenses/by/4.0/`

Torchvision modified the VGG graph for SSD and trained the detector;
LibreYOLO preserves the released SSD learned tensors and only adds metadata.
This records the initialization lineage and does not claim that Oxford
licensed the complete SSD checkpoint.

## Ported surface

| LibreYOLO surface | Pinned torchvision source |
|---|---|
| `nn.py` -- SSD orchestration, VGG extractor, extras, and MultiBox heads | `models/detection/ssd.py`, `models/vgg.py` |
| `nn.py` and `postprocess/ssd.py` -- default boxes and box decoding | `models/detection/anchor_utils.py`, `models/detection/_utils.py` |
| `utils.py` -- normalization and fixed-size resize | `models/detection/transform.py` |
| `postprocess/ssd.py` -- score filtering and class-wise NMS | `models/detection/ssd.py` |

The inference graph is native LibreYOLO code with the upstream state-dict
layout. It is not a wrapper around torchvision's SSD model class.

## Runtime contracts

- Canonical family and filename: `ssd`, size `300`, `LibreSSD300.pt`.
- Input is always a direct RGB stretch to 300 x 300 followed by torchvision's
  SSD normalization. Arbitrary export resolutions are rejected.
- The released head has background plus sparse COCO category ids in 91
  outputs. Public results map those ids to LibreYOLO's contiguous 80 classes.
- Native defaults match the pinned source: score threshold 0.01, NMS IoU 0.45,
  400 candidates per class, and 200 final detections. Public `conf` and `iou`
  arguments remain authoritative; `max_det` can request a lower limit while
  retaining the source model's fixed 200-detection ceiling.
- ONNX emits one decoded YOLO-grid tensor shaped `(B, 84, 8732)`. The shared
  LibreYOLO backend performs score filtering and class-aware NMS. Only the
  batch axis is dynamic; spatial dimensions remain fixed at 300.

## Measured evidence

- Eager reference parity on an RGB fixture: preprocessing, regression head,
  classification head, default boxes, final boxes, final scores, and labels
  all have `max_abs_diff == 0.0`; both implementations retain 200 detections.
- ONNX Runtime raw parity: maximum absolute error `6.1035e-05`, mean absolute
  error `5.3951e-08`, and p99 absolute error `1.1176e-08`.
- Unified backend prediction parity at `conf=0.25`: five detections with
  identical classes, boxes within `6.1035e-05`, and scores within
  `5.9605e-07`. At `conf=0.01, max_det=200`, classes remain identical, boxes
  are within `1.2207e-04`, and scores are within `5.9605e-07`.
- Public checkpoint on COCO128: mAP50-95 `0.3322` and mAP50 `0.5171`.
