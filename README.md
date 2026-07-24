# LibreYOLO

[English](README.md) | [简体中文](README.zh-CN.md)

> ⭐ **Support LibreYOLO.** The best way to help is to **star the repo**. Feel free to [open an issue](https://github.com/LibreYOLO/libreyolo/issues/new) if you encounter problems or have suggestions, and code contributions are very welcome (see [CONTRIBUTING.md](CONTRIBUTING.md)).

[![Documentation](https://img.shields.io/badge/docs-libreyolo.com-blue)](https://www.libreyolo.com/docs)
[![PyPI](https://img.shields.io/pypi/v/libreyolo)](https://pypi.org/project/libreyolo/)
[![PyPI Downloads](https://static.pepy.tech/badge/libreyolo)](https://pepy.tech/projects/libreyolo)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LibreYOLO-yellow)](https://huggingface.co/LibreYOLO)
[![Benchmarks](https://img.shields.io/badge/benchmarks-visionanalysis.org-purple)](https://www.visionanalysis.org/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-LibreYOLO-blue?logo=linkedin)](https://www.linkedin.com/company/libreyolo/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

MIT-licensed computer vision library with inference and training support for a variety of models. It provides a familiar high-level Python and CLI interface and reads common YOLO-format datasets, so existing workflows port over with minimal changes.

![LibreYOLO Detection Example](libreyolo/assets/parkour_result.jpg)

## Installation & Quick start

`pip install libreyolo` covers most users. It comes with the YOLOv9 flagship
and the other detection models, plus training and inference. Now and then you'll
add an extra: for a model family with a heavier dependency (for example RF-DETR,
which needs the large `transformers` library), or for an export backend when you
need to export a model:

```bash
pip install libreyolo

# Add an extra in brackets when you need one (comma-separate to combine),
# e.g. pip install "libreyolo[rfdetr,onnx]":
#   export:    onnx, tensorrt, openvino, ncnn, tflite (alias: litert), coreml
#   models:    rfdetr, vlm, sam, openvocab, clip, gaze
#   training:  lora, plots, tensorboard, mlflow, wandb
#   or all:    pip install "libreyolo[all]"
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreYOLO9t.pt")
result = model(SAMPLE_IMAGE, save=True)
```

For the full list of extras and per-backend notes, see the [docs](https://www.libreyolo.com/docs#installation).

To install from source in editable mode (for development or to track unreleased changes):

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
pip install -e .
```

A plain clone checks out `release`, the stable branch whose code matches these
docs. For the latest unreleased work, switch to the integration branch with
`git checkout dev`.

## Flagship models

LibreYOLO recommends these model families because they offer the best balance
and receive the heaviest testing:

- **YOLOv9** for CNN-based YOLO models.
- **RF-DETR** for transformer-based detection and segmentation.

## Compatibility

Training capabilities are documented per family in
[`docs/nomenclature.md`](docs/nomenclature.md).

`✓` parity-validated, `exp` experimental. Empty cells are blocked before export.
<!-- export-support:start -->
| Family | Task | onnx | torchscript | tensorrt | openvino | ncnn | tflite | coreml |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo9 | detect | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | exp |
| rfdetr | detect | ✓ | ✓ | ✓ | ✓ |  | exp | exp |
| rfdetr | segment | ✓ | ✓ | exp | exp |  |  |  |
| rfdetr | pose | ✓ | ✓ | exp | exp |  |  |  |
| rfdetr | obb | ✓ | ✓ | exp | exp |  |  |  |
| ec | detect | ✓ | ✓ | exp | exp |  |  |  |
| ec | pose | ✓ | ✓ | exp | exp |  |  |  |
| ec | segment | ✓ | ✓ | exp | exp |  |  |  |
| yolonas | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolonas | pose | ✓ | ✓ | exp | exp | ✓ |  |  |
| dfine | detect | ✓ | ✓ | exp | exp |  |  |  |
| dfine | segment | ✓ | ✓ | exp | exp |  |  |  |
| yolox | detect | ✓ | ✓ | exp | exp | ✓ | ✓ | exp |
| picodet | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolo1 | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolo2 | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolo3 | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolo4 | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolo7 | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolo9_e2e | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolo9_p2 | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| rtdetr | detect | ✓ | ✓ | exp | exp |  |  | exp |
| rtmdet | detect | ✓ | ✓ | exp | exp |  |  |  |
| rtmdet | segment |  |  |  |  |  |  |  |
| deim | detect | exp | ✓ | exp | exp |  |  |  |
| deimv2 | detect | exp | ✓ | exp | exp |  |  |  |
| rtdetrv2 | detect | exp | ✓ | exp | exp |  |  |  |
| rtdetrv4 | detect | exp | ✓ | exp | exp |  |  |  |
| florence2 | detect |  |  |  |  |  |  |  |
| grounding_dino | detect |  |  |  |  |  |  |  |
| internvl3 | detect |  |  |  |  |  |  |  |
| kosmos2 | detect |  |  |  |  |  |  |  |
| lfm2vl | detect |  |  |  |  |  |  |  |
| locateanything | detect |  |  |  |  |  |  |  |
| locateanything | point |  |  |  |  |  |  |  |
| omdet_turbo | detect |  |  |  |  |  |  |  |
| ov_deim | detect |  |  |  |  |  |  |  |
| owlv2 | detect |  |  |  |  |  |  |  |
| qwen3vl | detect |  |  |  |  |  |  |  |
| smolvlm2 | detect |  |  |  |  |  |  |  |
| eomt | semantic | ✓ | ✓ | exp | exp |  |  |  |
| eomt | segment |  |  |  |  |  |  |  |
| eomt | panoptic |  |  |  |  |  |  |  |
| picosam3 | segment | ✓ |  |  |  |  |  |  |
| edgetam | segment |  |  |  |  |  |  |  |
| mobilesam | segment |  |  |  |  |  |  |  |
| sam | segment |  |  |  |  |  |  |  |
| sam2 | segment |  |  |  |  |  |  |  |
| sam3 | segment |  |  |  |  |  |  |  |
| fomo | point | ✓ | ✓ | exp | exp | ✓ |  |  |
| convnext | classify | ✓ | ✓ | exp | exp | ✓ | ✓ |  |
| efficientnetv2 | classify | ✓ | ✓ | exp | exp | ✓ | ✓ |  |
| mobilenetv4 | classify | ✓ | ✓ | exp | exp | ✓ | ✓ |  |
| resnet | classify | ✓ | ✓ | exp | exp | ✓ | ✓ |  |
| dinov2 | semantic | ✓ | ✓ | exp | exp |  |  |  |
| dinov2 | classify | ✓ |  |  |  |  |  |  |
| clip | classify | ✓ |  |  |  |  |  |  |
| siglip2 | classify | ✓ |  |  |  |  |  |  |
| pidnet | semantic | ✓ | ✓ | exp | exp | ✓ | ✓ |  |
| segformer | semantic |  |  |  |  |  |  |  |
| zipdepth | depth | ✓ | ✓ | exp | exp | ✓ |  |  |
| depth_anything | depth | ✓ | ✓ | exp | exp |  |  |  |
| depth_anything3 | depth |  |  |  |  |  |  |  |
| realesrgan | restore | ✓ | ✓ | exp | exp | ✓ | ✓ |  |
| nafnet | restore | ✓ | ✓ | exp | exp | ✓ |  |  |
| swinir | restore | exp | exp | exp | exp | exp |  |  |
| birefnet | matte | exp | ✓ | exp | exp |  |  |  |
| ppocr | ocr |  |  |  |  |  |  |  |
| l2cs | gaze | ✓ |  |  |  |  |  |  |
<!-- export-support:end -->

## Depth estimation

Depth Anything 3 mono-large is the recommended quality default for relative
monocular depth:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDepthAnything3l-depth.pt")
result = model("image.jpg")[0]
inverse_depth = result.depth_map.data  # (H, W), higher means closer
```

The checkpoint is Apache-2.0 and downloads from the LibreYOLO Hugging Face
organization. Depth Anything V2 remains available for compatibility, while
ZipDepth provides the lightweight edge tier.

## License

- **Code:** MIT License
- **Weights:** Pre-trained weights may inherit licensing from the original source. Check the license in the specific HF repo of weights that you are interested in. LibreYOLO HF models always have a license.
