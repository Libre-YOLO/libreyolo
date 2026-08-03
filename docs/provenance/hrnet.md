# HRNet pose

- **LibreYOLO module:** `libreyolo/models/hrnet/`, heatmap decoding and OKS
  suppression in `libreyolo/postprocess/hrnet.py`, and converter at
  `weights/convert_hrnet_weights.py`.
- **Upstream:** [`leoxiaobin/deep-high-resolution-net.pytorch`](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
  at commit `6f69e4676ad8d43d0d61b64b1b9726f0c369e7b1` (2022-12-13).
- **Code license:** MIT. The root license is Copyright (c) 2019 Leo Xiao. The
  adapted model, geometry, decoding, and NMS source files also carry their
  original `Copyright (c) Microsoft. Written by Bin Xiao.` headers. The full
  license and both attributions are preserved in
  `libreyolo/models/hrnet/NOTICE` and `THIRD_PARTY_NOTICES.txt`.
- **Scope:** inference and pose validation only. LibreYOLO provides the two
  official COCO-17 variants, native top-down composition with a configurable
  person detector, and fixed person-crop exports. The upstream training recipe
  is not ported; `LibreHRNet.train()` raises `NotImplementedError`.

## Official checkpoints

The two files selected from the official upstream model zoo were downloaded
and hashed before conversion. Conversion is a metadata-only wrap for these
`--source original` files: all learned tensor keys, values, and dtypes remain
unchanged and load with `strict=True`.

| Size | Official file | Crop HxW | Source bytes | Source SHA-256 | Converted file | Converted SHA-256 |
|---|---|---:|---:|---|---|---|
| `w32` | `pose_hrnet_w32_256x192.pth` | 256x192 | 114,721,619 | `19bc083708bb8d873211e50d85d56344c10290c6e8b564c813fdde09645c4c1c` | `LibreHRNetw32-pose.pt` | `c9d8eb383e63ce795f87a3ee2f2b0c8aa0df9bae981573ec4b3909d63815a825` |
| `w48` | `pose_hrnet_w48_384x288.pth` | 384x288 | 255,061,287 | `95e0fec3194826d5e3f806ea89be68bbb84517b114c3a32b3058c56610b5ef61` | `LibreHRNetw48-pose.pt` | `ab85504f323c43babdcbed6550f759d323efd0ffc284a0593043e108cd3c4df5` |

The upstream repository is MIT-licensed and distributes these checkpoints
from its official model zoo, but it does not attach a separate per-file weight
license. LibreYOLO's redistribution basis is therefore the MIT license implied
by the releasing project, not a claim of a distinct checkpoint-specific grant.
Every mirror carries the verbatim upstream MIT `LICENSE` and a `NOTICE` that
states this limitation. Training-data rights remain separate and users remain
responsible for their use case.

## Ported surface

| LibreYOLO surface | Pinned upstream source |
|---|---|
| `models/hrnet/nn.py` | `lib/models/pose_hrnet.py` |
| `models/hrnet/utils.py` affine geometry | `lib/utils/transforms.py`, `lib/dataset/coco.py`, and the official demo box-to-center/scale path |
| `postprocess/hrnet.py` heatmap maxima and quarter-pixel decode | `lib/core/inference.py` |
| `postprocess/hrnet.py` flip restoration | `lib/utils/transforms.py` |
| `postprocess/hrnet.py` OKS IoU and NMS | `lib/nms/nms.py` |

`models/hrnet/model.py`, `inference.py`, and `detector.py` are LibreYOLO
integration code. They connect the fixed crop head to `Results.boxes` and
`Results.keypoints`, explicit `person_boxes`, `cropped=True`, callable detector
adapters, and native LibreYOLO detectors. The default is `LibreYOLO9t.pt`;
`person_detector="rfdetr"` selects the RF-DETR adapter. The selected detector
is not part of the HRNet checkpoint or export graph.

## Numerical evidence

`tests/unit/test_hrnet_parity.py` loads the pinned upstream Python modules and
official checkpoints directly. For W32 and W48 it verifies exact equality of:

- box-to-center/scale conversion, affine matrices, RGB crop pixels, and
  ImageNet normalization;
- every model state-dict key and the final FP32 heatmap tensor
  (`max_abs_diff = 0.0`);
- decoded original-image coordinates and peak responses; and
- horizontal flip restoration plus the official one-pixel shift.

The two-stage unit suite also covers empty detections, explicit boxes, ready
person crops, YOLO9 and RF-DETR adapters, COCO-17 result shapes, score
composition, OKS suppression, fixed input contracts, validation routing, and
the intentional training error.

The real-checkpoint export gate ran all eight combinations, one isolated
process at a time, and finished with `8 passed`:

| Runtime | W32 | W48 | Maximum raw/public absolute tolerance |
|---|---|---|---:|
| ONNX Runtime 1.26.0 | passed | passed | `3e-6` |
| TorchScript (PyTorch 2.11.0+cu128) | passed | passed | `0` |
| OpenVINO 2026.2.1 | passed | passed | `3e-3` |
| TensorRT 10.16.1.11, CUDA 12.8 | passed | passed | `3e-3` |

All exports are batch-one FP32 person-crop heads at their canonical fixed
canvas. ONNX, TorchScript, OpenVINO, and TensorRT are the only validated HRNet
formats. Full-image detector composition remains Python-only.

## Accuracy interpretation

The upstream model zoo reports COCO validation AP 0.744 for W32 256x192 and
0.763 for W48 384x288 with flip testing and its stated detector (human AP
56.4). Those figures describe an end-to-end detector-plus-pose recipe.
LibreYOLO does not claim to reproduce them with its default detector. For a
meaningful comparison, validation must use the same person boxes, detector
scores, flip setting, and COCO evaluation inputs.
