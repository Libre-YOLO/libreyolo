# DeepLabv3

- **LibreYOLO module:** `libreyolo/models/deeplabv3/`, postprocessing in
  `libreyolo/postprocess/deeplabv3.py`, and converter at
  `weights/convert_deeplabv3_weights.py`.
- **Upstream code:** `pytorch/vision` at commit
  `336d36e8db990a905498c73933e35231876e28bc` (torchvision v0.26.0).
- **Code license:** BSD-3-Clause, Copyright (c) Soumith Chintala 2016. The full
  notice is reproduced in `libreyolo/models/deeplabv3/NOTICE`.
- **Scope:** 21-output semantic segmentation inference and validation only.
  The training-only auxiliary FCN head and training recipe are excluded;
  `train()` raises `NotImplementedError`.
- **Verification:** all three released variants produce bit-exact eager logits
  against the pinned upstream implementation before postprocessing.

## Source checkpoints

The official files below were downloaded from PyTorch's model host and checked
by SHA-256. They were trained on the subset of COCO categories that matches the
20 Pascal VOC foreground labels; output zero is background. They were not
trained on Pascal VOC images. The files are not included in this repository.

| Size | Upstream file | Bytes | SHA-256 | Upstream mIoU / pixel accuracy |
|---|---|---:|---|---:|
| `r50` | `deeplabv3_resnet50_coco-cd0a2569.pth` | 168,312,152 | `cd0a25694c4a0f7106b38f4938bf90a874f2f241cc410b8f63c7024399538f06` | 66.4 / 92.4 |
| `r101` | `deeplabv3_resnet101_coco-586e9e4e.pth` | 244,545,539 | `586e9e4e203fcbf17e1ad45533d8d33ab133fc762bf03101c5dd743995c08c0d` | 67.4 / 92.4 |
| `mv3` | `deeplabv3_mobilenet_v3_large-fc3c493d.pth` | 44,356,159 | `fc3c493d68e89cc31ef488c803d5d7dd2f3190fb570598faa49fef69be8e5e70` | 60.3 / 91.2 |

Official URLs are the filenames above appended to
`https://download.pytorch.org/models/`. The metrics are those recorded by the
pinned torchvision weight enums. Conversion removes only the unused
`aux_classifier.*` tensors, preserves every runtime tensor, and adds LibreYOLO
checkpoint metadata.

| Size | LibreYOLO filename | Bytes | SHA-256 |
|---|---|---:|---|
| `r50` | `LibreDeepLabv3r50-sem.pt` | 158,900,443 | `a8910db2cb2827ec19fce65a051f4d651bee73f5a46ba8d1c431c0d7042dca7c` |
| `r101` | `LibreDeepLabv3r101-sem.pt` | 235,177,707 | `4575b7d5b1b70e9c67225ae76c00f552b29c2e54b07d55cfee8da218a9f41429` |
| `mv3` | `LibreDeepLabv3mv3-sem.pt` | 44,325,189 | `fb83a67bca845817d816d139af6fb6a4b9d809c0a813ebcfcb1e2a5fbd222682` |

### Weight-license status

The torchvision source is BSD-3-Clause. The publisher does not attach a
separate license file to these three checkpoint objects, so the redistribution
basis is **BSD-3-Clause implied by the releasing project**, not an explicit
checkpoint-specific grant. Torchvision's README separately warns that
pretrained models may have their own terms derived from their training data
and leaves use-case permission to the user.

COCO annotations are CC BY 4.0. The source images retain their individual
Flickr terms. Every LibreYOLO weight repository therefore carries the verbatim
torchvision BSD-3-Clause license, an attribution notice, and this caveat; the
cards must never describe the checkpoint license as publisher-confirmed.

## Ported surface

| LibreYOLO surface | Pinned torchvision source |
|---|---|
| `nn.py` -- DeepLabv3 orchestration and full-canvas interpolation | `models/segmentation/_utils.py` |
| `nn.py` -- ASPP branches, pooling, projection, and classifier | `models/segmentation/deeplabv3.py` |
| `nn.py` -- ResNet and MobileNetV3 backbone selection | `models/segmentation/deeplabv3.py` |
| `convert.py` -- strict upstream recognition and auxiliary-head removal | Upstream checkpoint layout from the three pinned weight enums |

The runtime imports torchvision's BSD-licensed ResNet, MobileNetV3, and
`IntermediateLayerGetter` building blocks. The dense head and inference
orchestration live in LibreYOLO, so the family is not a wrapper around an
upstream segmentation model class.

## Variant, class, and preprocessing contracts

- `r50`: dilated ResNet-50 backbone, output stride 8, 42,004,074 upstream
  parameters before the auxiliary head is removed.
- `r101`: dilated ResNet-101 backbone, output stride 8, 60,996,202 upstream
  parameters before the auxiliary head is removed.
- `mv3`: dilated MobileNetV3-Large backbone, output stride 16, 11,029,328
  upstream parameters before the auxiliary head is removed.
- All variants emit 21 full-canvas logits: background plus aeroplane, bicycle,
  bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse,
  motorbike, person, pottedplant, sheep, sofa, train, and tvmonitor.
- The canonical filename suffix is required: `-sem.pt`. Task resolution is
  semantic-only and bidirectionally rejects checkpoints from other semantic
  families.

The upstream evaluation transform preserves aspect ratio and resizes the short
side to 520 before ImageNet normalization. LibreYOLO deliberately uses its
fixed deployment contract instead: RGB is stretched to 520x520, normalized by
the same ImageNet mean and standard deviation outside the graph, and the final
mask is resized back to the original canvas. This makes the supported exported
graphs fixed-shape and is a documented behavior difference from upstream's
evaluation preset. Exact native parity is measured on identical already-sized
520x520 normalized tensors, before either library's external preprocessing or
postprocessing.

## Measured evidence

- Eager logits for `r50`, `r101`, and `mv3` at 520x520 are bit-exact against
  torchvision v0.26.0: `max_abs_diff == 0.0` for every variant.
- ONNX Runtime CPU agrees with native CPU masks at 100% of pixels for all
  variants. Maximum logit differences are `1.53e-5`, `1.34e-5`, and `3.06e-5`
  for `r50`, `r101`, and `mv3` respectively.
- TorchScript is bit-exact for every variant and preserves 100% of public mask
  pixels.
- OpenVINO CPU preserves 99.9994%, 99.9994%, and 99.9876% of public mask pixels
  for `r50`, `r101`, and `mv3`; its default reduced-precision execution hint
  produces maximum logit differences of 0.180, 0.196, and 0.390.
- TensorRT 10.16 FP32 on an RTX 5070 Ti preserves 99.9981%, 99.9986%, and
  99.9851% of public mask pixels; maximum logit differences are 0.0178, 0.0119,
  and 0.0349.
- ONNX, TorchScript, OpenVINO, and TensorRT artifacts all strict-reload through
  the unified backend with family, task, size, names, input-size, and
  normalization metadata intact. Other export formats remain unsupported.
