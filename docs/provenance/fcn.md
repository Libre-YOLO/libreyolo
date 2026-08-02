# FCN semantic segmentation

- **LibreYOLO module:** `libreyolo/models/fcn/`, postprocessing in
  `libreyolo/postprocess/fcn.py`, and converter at
  `weights/convert_fcn_weights.py`.
- **Upstream code:** `pytorch/vision` at commit
  `336d36e8db990a905498c73933e35231876e28bc` (torchvision v0.26.0).
- **Code license:** BSD-3-Clause, Copyright (c) Soumith Chintala 2016 and the
  torchvision contributors. The full notice is reproduced in
  `libreyolo/models/fcn/NOTICE`.
- **Scope:** 21-class semantic inference and validation. Training losses are
  intentionally excluded; `train()` raises `NotImplementedError`.
- **Verification:** the `r50` and `r101` primary and auxiliary dense logits are
  bit-exact against the pinned upstream implementation on two independent
  inputs (`max_abs_diff == 0`).

## Architecture identity

The 2015 FCN work established end-to-end pixels-to-pixels semantic prediction.
LibreFCN preserves that historical family name, but the shipped graph is
torchvision's later model: a dilated ResNet-50 or ResNet-101 backbone,
`IntermediateLayerGetter`, a primary head on `layer4`, and an auxiliary head on
`layer3`. It is **not** the original paper's VGG-based FCN-8s graph and does not
contain its skip-fusion topology.

The historical Berkeley reference repository is context only and was not a
source for LibreYOLO code. At commit
`1305c7378a9f0ab44b2c936f4d60e4687e3d8743`, its README says the code and models
are "available under the same license as Caffe (BSD-2)", but the repository has
no formal license file. LibreYOLO therefore did not inspect, copy, adapt, or
derive from its implementation. Historical README:
https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/README.md

## Source checkpoints

The official files below were downloaded from PyTorch's model host and checked
by SHA-256. Conversion preserves every learned tensor and state-dict key and
only adds LibreYOLO checkpoint metadata. The files are not included in the
source repository.

| Size | LibreYOLO filename | Upstream checkpoint | Bytes | SHA-256 | Published COCO-val2017-VOC-labels mIoU / pixel accuracy |
|---|---|---|---:|---|---:|
| `r50` | `LibreFCNr50.pt` | `fcn_resnet50_coco-1167a1af.pth` | 141,567,418 | `1167a1affa42e1e62858f8d3fac12d109e0108327ffc91c5855a324b11683c36` | 60.5 / 91.4 |
| `r101` | `LibreFCNr101.pt` | `fcn_resnet101_coco-7ecb50ca.pth` | 217,800,805 | `7ecb50ca17844860a70d5ed0c748d997cf8adb62932abaa0233430c68594d749` | 63.7 / 91.9 |

Official URLs are the filenames above appended to
`https://download.pytorch.org/models/`. Both checkpoints expose background
plus the 20 Pascal VOC object categories while using COCO training data and
the upstream `COCO-val2017-VOC-labels` evaluation protocol.

### Weight-license status

The torchvision source is BSD-3-Clause. The publisher does not attach a
separate license file to either checkpoint object, so the redistribution basis
is **BSD-3-Clause implied by the releasing project**, not an explicit
checkpoint-specific grant. Torchvision's README separately warns that
pretrained models may have their own terms derived from their training data
and leaves use-case permission to the user.

The project maintainer approved rehosting on that disclosed implied-license
basis. Every LibreYOLO weight repository therefore carries the verbatim
torchvision BSD-3-Clause license, an attribution notice, and the caveat above;
the cards must never describe the checkpoint license as publisher-confirmed.
`weights/upload_fcn_hf.py` builds and validates the exact five-file
repositories before publication.

## Ported surface

| LibreYOLO surface | Pinned torchvision source |
|---|---|
| `nn.py` FCN graph and primary/auxiliary heads | `torchvision/models/segmentation/fcn.py` |
| dilated backbone selection and returned feature layers | `torchvision/models/segmentation/fcn.py` |

The runtime imports torchvision's BSD-licensed ResNet builders and
`IntermediateLayerGetter`. LibreYOLO owns factory dispatch, input handling,
semantic result conversion, validation, export wrappers, checkpoint metadata,
and automatic conversion.

## Runtime and export contract

- `r50` and `r101` use a 520-pixel square RGB input and ImageNet normalization.
- The primary 21-channel `out` logits define public masks; the retained
  auxiliary logits are exposed by the raw graph but do not affect inference.
- Native PyTorch, ONNX Runtime, TorchScript, OpenVINO, and TensorRT were checked
  against the same model outputs. Unsupported export formats remain blocked in
  the canonical export-support registry.
