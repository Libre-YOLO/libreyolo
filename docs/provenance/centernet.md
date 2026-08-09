# CenterNet

- **LibreYOLO module:** `libreyolo/models/centernet/`, postprocessing in
  `libreyolo/postprocess/centernet.py`, and conversion/parity tools at
  `weights/convert_centernet_weights.py` and `weights/parity_centernet.py`.
- **Upstream code:** `xingyizhou/CenterNet` at commit
  `4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c`.
- **Code license:** MIT, Copyright (c) 2019 Xingyi Zhou. The upstream NOTICE
  additionally records MIT or BSD-3-Clause lineage from tf-faster-rcnn,
  Microsoft's human-pose-estimation.pytorch, CornerNet, DCNv2, and DLA. The
  complete texts are reproduced in `libreyolo/models/centernet/NOTICE`.
- **Scope:** box-detection inference and validation only. CenterNet training
  and multi-scale/flip test augmentation are not implemented; `train()` and
  TTA requests raise.

## Ported surface

| LibreYOLO surface | Pinned CenterNet source |
|---|---|
| ResDCN-18 detector and heads | `src/lib/models/networks/resnet_dcn.py` |
| DLA-34 detector, DLAUp, and IDAUp | `src/lib/models/networks/pose_dla_dcn.py`, `src/lib/models/networks/dla.py` |
| Fixed-512 BGR affine preprocessing | `src/lib/detectors/base_detector.py`, `src/lib/utils/image.py` |
| Heatmap peak selection and top-K decode | `src/lib/models/decode.py` |
| Inverse-affine box projection | `src/lib/utils/image.py`, `src/lib/detectors/ctdet.py` |

Both graphs preserve the official `hm`, `wh`, and `reg` parameter layout and
produce stride-4 heads. Postprocessing applies sigmoid, a 3x3 local-maximum
filter, and the top 100 centers across all classes. CenterNet deliberately does
not apply NMS.

The pinned project used a compiled DCNv2 extension. LibreYOLO does not copy,
build, or distribute that extension. Eager inference substitutes
`torchvision.ops.deform_conv2d` from `pytorch/vision` commit
`336d36e8db990a905498c73933e35231876e28bc` (torchvision v0.26.0,
BSD-3-Clause). ONNX and TorchScript export use a LibreYOLO-authored equivalent
made from portable grid sampling and convolution. NCNN is rejected before
conversion because this portable graph is outside the supported NCNN contract.

## Source checkpoints

The two official files were downloaded from the CenterNet model-zoo links and
verified before conversion. Google Drive object ids are included because the
download service does not expose stable descriptive URLs.

| Size | Official file and object id | Bytes | Official SHA-256 | Canonical file | Canonical SHA-256 |
|---|---|---:|---|---|---|
| `resdcn18` | `ctdet_coco_resdcn18.pth`, `1RtFps3kQAyLjQyzCao7pPDclOBQ64Vyp` | 57,825,323 | `f9e413f91cdb235adbcb41c5c4052b8f7ff53999374048949789c29d6df18eaa` | `LibreCenterNetresdcn18.pt` | `490a6c98c08510194f89416bde0d684e10f46d679f859b2e1a9e8117c9dc0095` |
| `dla34` | `ctdet_coco_dla_2x.pth`, `18Q3fzzAsha_3Qid6mn4jcIFPeOGUaj1d` | 80,911,783 | `43bf4cc2efe00e02c1ae8484035b062a35543872d276c7dcfeb4db3e64203e4f` | `LibreCenterNetdla34.pt` | `0818769746f56bffbed9d22be8f1a5896465cd44826ac5c69333a1121205b6e9` |

The official objects are available at
`https://drive.google.com/uc?export=download&id=<object-id>`. The ResDCN-18
checkpoint records epoch 140 and the DLA-34 checkpoint records epoch 230.
Conversion removes the data-parallel `module.` key prefix and adds LibreYOLO
v1 checkpoint metadata. Learned tensors are unchanged.

The CenterNet model zoo reports COCO test-dev AP for no augmentation / flip /
multi-scale inference of 28.1 / 30.0 / 33.2 for ResDCN-18 and
37.4 / 39.2 / 41.7 for DLA-34. LibreYOLO implements the no-augmentation path.

### Weight-license status

The official checkpoint objects have no standalone license files or explicit
checkpoint-specific grant. They were published by the MIT-licensed CenterNet
project, so the redistribution basis is **MIT implied by the releasing
project**, not publisher-confirmed. The project maintainer approved rehosting
on that disclosed basis. Each LibreYOLO weight repository therefore contains
exactly the canonical checkpoint, `.gitattributes`, a model card, the verbatim
CenterNet MIT `LICENSE`, and a `NOTICE` that repeats this caveat. COCO
annotations are CC BY 4.0; source images retain their individual Flickr terms,
and users remain responsible for their use case.

Public mirrors are
[`resdcn18`](https://huggingface.co/LibreYOLO/LibreCenterNetresdcn18) and
[`dla34`](https://huggingface.co/LibreYOLO/LibreCenterNetdla34), verified at
revisions `a54c4568201fb54449bfef58a91728acfcb2ee95` and
`7d48e7e8ac8c07a4891e5ebaa5462ce6f37a3668`, respectively. Each repository
was verified public with exactly five files, added to the LibreYOLO Models
collection, downloaded through its bare-filename route from an isolated empty
cache, hash-checked against the canonical SHA-256 above, and used for a CUDA
prediction before the e2e and general-nightly catalog rows were enabled.

## Measured evidence

- Strict loading covers all 177 ResDCN-18 and 400 DLA-34 state entries.
- On the same synthetic 512 input, every `hm`, `wh`, and `reg` output is
  bit-exact against the pinned upstream graph (`max_abs_diff == 0.0`) for both
  variants.
- Affine preprocessing and heatmap decoding are separately bit-exact. The
  complete synthetic image-to-original-canvas path has identical classes and
  scores, with maximum box difference `1.52587890625e-05` pixels.
- Hosted checkpoints on COCO128 at `conf=0.001`, `iou=0.6`, batch one:
  ResDCN-18 mAP50-95 / mAP50 is `0.3931 / 0.5845`; DLA-34 is
  `0.5344 / 0.7682`.
- On the public sample image, native PyTorch and ONNX Runtime return identical
  detection counts and classes. ResDCN-18 boxes differ by at most
  `0.000122` pixels and scores by `5.4e-7`; DLA-34 boxes differ by at most
  `0.000336` pixels and scores by `1.16e-6`.
- TorchScript returns the same detections. Its measured maximum box difference
  is `0.000122` pixels for ResDCN-18 and `0.000671` pixels for DLA-34; maximum
  score difference is `5.4e-7` and `2.89e-6`, respectively.
- Native CUDA and the portable export graph return the same public detections,
  with boxes within `0.022` pixels and scores within `2.2e-4`.

ONNX Runtime and TorchScript were executed for both official variants.
TensorRT and OpenVINO were not installed in the validation environment and are
not claimed as runtime-validated backends.
