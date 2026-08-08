# FCOS

- **LibreYOLO module:** `libreyolo/models/fcos/`, postprocessing in
  `libreyolo/postprocess/fcos.py`, and converter at
  `weights/convert_fcos_weights.py`.
- **Upstream code:** `pytorch/vision` at commit
  `336d36e8db990a905498c73933e35231876e28bc` (torchvision v0.26.0).
  The pinned `torchvision/models/detection/fcos.py` blob is
  `ccbd2496517c33b74a1a1581e0cbf3b3f173bfed`.
- **Code license:** BSD-3-Clause, Copyright (c) Soumith Chintala 2016. The
  full notice is reproduced in `libreyolo/models/fcos/NOTICE`.
- **Scope:** ResNet-50/FPN box-detection inference and validation. Dense
  target assignment and focal, box-regression, and centerness losses are
  excluded; `train()` raises `NotImplementedError`.

No source from the original FCOS repository or from AdelaiDet was inspected
or used. The implementation and its numerical reference both come from the
pinned BSD-3-Clause torchvision release.

## Source checkpoint

The official file was downloaded from PyTorch's model host and checked by
SHA-256. Conversion preserves every learned tensor and state-dict key and only
adds LibreYOLO checkpoint metadata. Neither checkpoint is included in this
source repository.

| LibreYOLO filename | Official file | Official bytes | Official SHA-256 | COCO box AP |
|---|---|---:|---|---:|
| `LibreFCOSr50.pt` | `fcos_resnet50_fpn_coco-99b0c9b7.pth` | 129,612,099 | `99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7` | 39.2 |

The converted checkpoint is 129,617,833 bytes with SHA-256
`51d79292895816fc09e4a7b159331f8bbc1243e0cee96f991f8e5865f4920788`.
It is published at
[`LibreYOLO/LibreFCOSr50`](https://huggingface.co/LibreYOLO/LibreFCOSr50).
The remote repository was verified to contain exactly `.gitattributes`,
`README.md`, `LICENSE`, `NOTICE`, and `LibreFCOSr50.pt`; the LFS object hash
matches the converted checkpoint.

### Weight-license status

The torchvision source is BSD-3-Clause. The publisher does not attach a
separate license file to this checkpoint object, so the redistribution basis
is **BSD-3-Clause implied by the releasing project**, not an explicit
checkpoint-specific grant. Torchvision's README separately warns that
pretrained models may have their own terms derived from training data and
leaves use-case permission to the user. COCO annotations are CC BY 4.0; source
images retain their individual Flickr terms.

The project maintainer approved rehosting on that disclosed implied-license
basis. The weight repository carries the verbatim torchvision BSD-3-Clause
license, an attribution notice, and the caveat above; its card does not call
the grant publisher-confirmed. `weights/upload_fcos_hf.py` validates the
checkpoint schema, canonical loader URL, converted hash, exact five-file
contract, and collection membership workflow before publication. The bare
`LibreFCOSr50.pt` autodownload route was tested from an empty local directory,
strict-loaded, and used for a real GPU prediction before the downloaded copy
and upload staging directories were removed.

## Ported surface

| LibreYOLO surface | Pinned torchvision source |
|---|---|
| `models/fcos/nn.py` - detector and FCOS heads | `models/detection/fcos.py` |
| `models/fcos/nn.py` - anchor grid and box decoding | `models/detection/anchor_utils.py`, `models/detection/_utils.py` |
| `models/fcos/utils.py` - normalization, aspect resize, stride padding | `models/detection/transform.py` |
| `postprocess/fcos.py` - threshold, per-level top-k, decode, clip, class-wise NMS | `models/detection/fcos.py` |

The runtime imports torchvision's BSD-licensed ResNet-50, FPN,
`FrozenBatchNorm2d`, and box operations. LibreYOLO owns the inference-only
orchestration, factory, result contract, validator, and exported-runtime
adapter. The official module names remain unchanged, so all 319 state entries
load with `strict=True` and no remapping.

## Input, score, and class contracts

- RGB pixels are converted to float in `[0, 1]`, normalized with ImageNet
  mean/std, resized so the short side is 800 and long side is at most 1333,
  then bottom/right padded to a multiple of 32.
- Public score defaults are `conf=0.2`, `iou=0.6`, and `max_det=100`. Scores
  are `sqrt(sigmoid(classification) * sigmoid(centerness))`; candidates are
  thresholded and capped to 1000 per pyramid level before class-wise NMS.
- The released head has 91 sparse COCO category-id columns. Public results and
  exported graphs map the 80 used ids to LibreYOLO's contiguous 0-79 classes.
- Prediction and validation are batch-one because source images create
  variable aspect-preserving padded canvases.

## Measured evidence

- Raw-head parity against the pinned reference is exact for all five feature
  levels: classification logits, box regression, centerness, anchors, and
  level sizes have `max_abs_diff == 0` on the same preprocessed tensor.
- The 800/1333 transform tensor is exact. On the bundled dog image, native
  postprocessing returns the same 64 classes, boxes, and scores as the pinned
  reference with maximum absolute difference 0.
- COCO128 at batch 1, `conf=0.001`, and `iou=0.6` returns mAP50-95 `0.5527`,
  mAP50 `0.7621`, and AR100 `0.6590` over all 128 images.
- At the native 800 short side, ONNX Runtime returns the same 64 detections and
  classes as native PyTorch; maximum box drift is `1.221e-4` pixels and score
  drift is `1.103e-6`. TorchScript is bit-exact for the same input.
- OpenVINO CPU FP32 passes raw-output tolerances and high-confidence public
  prediction parity. Small numerical drift can reorder low-confidence NMS
  survivors, so validation is limited to raw outputs and high-confidence
  predictions.
- TensorRT is blocked for this family because the current LibreYOLO TensorRT
  runtime profiles dynamic batch only, while FCOS requires dynamic padded
  height and width to preserve its aspect-resize contract.
