# EfficientDet

- **LibreYOLO module:** `libreyolo/models/efficientdet/`, centralized
  postprocessing in `libreyolo/postprocess/efficientdet.py`, and converter at
  `weights/convert_efficientdet_weights.py`.
- **Primary upstream:**
  [`rwightman/efficientdet-pytorch`](https://github.com/rwightman/efficientdet-pytorch)
  at commit `c6dff775a36cea0bf9b76c58e59f936411c5ce01`, released as
  `effdet` 0.4.1.
- **Code license:** Apache-2.0, Copyright 2020 Ross Wightman. The upstream
  repository license and the sdist's verbatim `LICENSE` were verified before
  implementation source was inspected.
- **Additional permissive sources:**
  [`pytorch-image-models`](https://github.com/huggingface/pytorch-image-models)
  v1.0.28 at commit `8ef73809f622e0031bd7f4940265734aef8b9978`
  (Apache-2.0), and the original
  [`google/automl`](https://github.com/google/automl/tree/master/efficientdet)
  EfficientDet implementation at BiFPN commit
  `56815c9986ffd4b508fe1d68508e268d129715c1` and anchor commit
  `6f6694cec1a48cdb33d5d1551a2d5db8ad227798` (Apache-2.0).
- **Scope:** D0-D4 detection inference, validation, ONNX, TorchScript,
  OpenVINO, and TensorRT. Focal-loss training and anchor assignment are not
  ported; `LibreEfficientDet.train()` raises `NotImplementedError`.
- **Exclusion:** no source from the LGPL-licensed
  `zylo117/Yet-Another-EfficientDet-Pytorch` repository was consulted or used.

## Official checkpoints

Downloaded from the upstream v0.1 GitHub release on 2026-08-03:

| Size | Input | Official file | Bytes | SHA-256 |
|---|---:|---|---:|---|
| `d0` | 512 | `tf_efficientdet_d0_34-f153e0cf.pth` | 15,839,413 | `f153e0cfe3c987981ab2d02ce32c8a5fa0b4092bc97abcf95c887e2e05bd4003` |
| `d1` | 640 | `tf_efficientdet_d1_40-a30f94af.pth` | 26,957,295 | `a30f94afc3326a6ef7a61c1657baefe3ea7168139006fbf0fe41807260b885b8` |
| `d2` | 768 | `tf_efficientdet_d2_43-8107aa99.pth` | 32,896,078 | `8107aa9988942c88b909158e14de3319870d64e75bacbe4806c3b84aa8c700e3` |
| `d3` | 896 | `tf_efficientdet_d3_47-0b525f35.pth` | 48,799,941 | `0b525f352fea3c768fd1a3cc885f1016d24d1b99cf299e8861e784a4ec2f0eff` |
| `d4` | 1024 | `tf_efficientdet_d4_49-f56376d9.pth` | 83,812,262 | `f56376d93b7f0a5e75eb7d6a54ab6a767c230d4bcb2d44510392f53c781f40a0` |

The release URLs are the filenames above appended to
`https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/`.

### Weight-license status

The source repository and packaged `effdet` code are Apache-2.0. The publisher
does not attach a separate license object to these checkpoint assets, so the
redistribution basis is **Apache-2.0 implied by the releasing project**, not a
checkpoint-specific grant. The Hugging Face packages reproduce the upstream
Apache license and this caveat. COCO annotations are CC BY 4.0, while source
images retain their individual Flickr terms.

## Conversion

Conversion validates the full BiFPN/backbone/class-head/box-head signature,
cross-checks the requested size against both filename and FPN width, and
strict-loads all tensors into the native graph. It then adds checkpoint-schema
metadata. Learned tensors, their names, dtypes, and values remain unchanged.

| Size | Canonical file | Bytes | Converted SHA-256 |
|---|---|---:|---|
| `d0` | `LibreEfficientDetd0.pt` | 15,967,401 | `89553d315e12f7d6543bd5755e686408a621fffadafa402fbe126da3b775a795` |
| `d1` | `LibreEfficientDetd1.pt` | 27,108,633 | `5e6b5d5e4ae96771424ded21a33cca5e3666bd97db6c39ec3f072756ad1f8183` |
| `d2` | `LibreEfficientDetd2.pt` | 33,069,769 | `0f78ce5e9c832919ea6cd5b2201909ecba0ede2134d610639704e30b9aa57a85` |
| `d3` | `LibreEfficientDetd3.pt` | 49,003,877 | `2077a387e88da0fcfd3f2d9267a6c02bb6829e43d56473899e20079ed24bde51` |
| `d4` | `LibreEfficientDetd4.pt` | 84,050,573 | `e2b301f82add4a0adee2e0914be13e08e1816fb80fd8071bcf5b2d9d17ebb936` |

## Ported surface

| LibreYOLO surface | Pinned upstream source |
|---|---|
| `models/efficientdet/nn.py` - BiFPN and heads | `effdet/efficientdet.py`, `effdet/config/fpn_config.py` |
| `models/efficientdet/nn.py` - EfficientNet backbone and SAME padding | timm EfficientNet builder/blocks used by `tf_efficientnet_b0` through `b4` |
| `models/efficientdet/config.py` | `effdet/config/model_config.py` (`tf_efficientdet_d0` through `d4`) |
| `postprocess/efficientdet.py` | `effdet/anchors.py`, `effdet/bench.py` |
| `models/efficientdet/utils.py` | `effdet/data/transforms.py` (`ResizePad`) |

Module names remain compatible with the upstream state dictionaries, so every
official checkpoint loads with `strict=True` and no key remapping.

## Inference contract

The upstream evaluation transform uses PIL bilinear aspect-preserving resize,
places the image at the top-left of its fixed square canvas, fills the bottom
and right with rounded ImageNet mean pixels, and applies ImageNet mean/std
normalization. Native prediction, validation, and exported backends share that
single transform.

The official head has 90 output slots corresponding to COCO category ids
1-90. Ten ids have no category. LibreYOLO filters those gaps and maps the
remaining ids to contiguous classes 0-79 before class-aware NMS. Custom heads
without the official 90-slot layout expose their classes directly.

## Numerical evidence

For D0-D4, fixed random tensors at each official resolution produced
`max_abs_diff == 0.0` for every class and box tensor against `effdet` 0.4.1.
Generated anchors are bit-exact, and all 5,000 decoded xyxy/score/class
candidates are also bit-exact. The external-data parity test records these
checks in `tests/unit/test_efficientdet_parity.py`.

The official D0 checkpoint detected six objects on LibreYOLO's bundled parkour
image at confidence 0.25. Its saved visualization was inspected for aligned
person and skateboard boxes. Trained-checkpoint FP32 runtime parity at
confidence 0.30 was:

| Runtime | Candidate output | Detections/classes | Max box drift | Max score drift |
|---|---|---|---:|---:|
| TorchScript CPU | `(1, 5000, 6)` | exact | 0 px | 0 |
| ONNX Runtime CPU | `(1, 5000, 6)` | exact | 0.000366 px | 8.94e-6 |
| OpenVINO 2026.2 CPU | `(1, 5000, 6)` | exact | 0.413208 px | 6.09e-4 |
| TensorRT 10.16 FP32, RTX 5070 Ti | `(1, 3840, 6)` | exact | 0.056274 px | 3.57e-4 |

TensorRT's `ITopK` layer rejects `K > 3840`, so only that export uses its
3,840-point maximum. Native, ONNX, TorchScript, and OpenVINO retain upstream's
5,000-point budget. The trained-checkpoint export tests live in
`tests/e2e/test_efficientdet_export.py`.
