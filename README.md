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

Image classification works the same way. Load a pretrained ImageNet-1k
classifier (`MobileNetV4`, `ConvNeXt`, `EfficientNetV2`, or `ResNet`), then
predict or fine-tune on your own folder-per-class dataset:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreResNet50-cls.pt")   # weights auto-download on first use
result = model("image.jpg")                  # a single image -> one Results
print(result.probs.top1, float(result.probs.top1conf))  # class index + confidence
print(result.probs.top5)                     # indices of the 5 most likely classes

# Fine-tune on an ImageFolder dataset (train/ and val/, one sub-folder per
# class). The classifier head resizes to your class count automatically.
model.train(data="path/to/dataset", epochs=5)
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

`✓` parity-validated, `exp` experimental. Empty cells are blocked before export.
<!-- export-support:start -->
| Family | Task | onnx | torchscript | tensorrt | openvino | ncnn | tflite | coreml |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| birefnet | matte | exp | exp | exp | exp | exp |  |  |
| clip | classify | ✓ | exp | exp | exp | exp |  |  |
| convnext | classify | ✓ | ✓ | exp | exp | exp |  |  |
| deim | detect | exp | exp | exp | exp |  |  |  |
| deimv2 | detect | exp | exp | exp | exp |  |  |  |
| depth_anything | depth |  |  |  |  |  |  |  |
| dfine | detect | exp | exp | exp | exp |  |  |  |
| dfine | segment | exp | exp | exp | exp |  |  |  |
| dinov2 | semantic |  |  |  |  |  |  |  |
| dinov2 | classify | exp | exp | exp | exp | exp |  |  |
| ec | detect | exp | exp | exp | exp |  |  |  |
| ec | pose | exp | exp | exp | exp |  |  |  |
| ec | segment | exp | exp | exp | exp |  |  |  |
| efficientnetv2 | classify | ✓ | ✓ | exp | exp | exp |  |  |
| eomt | semantic |  |  |  |  |  |  |  |
| eomt | segment |  |  |  |  |  |  |  |
| eomt | panoptic |  |  |  |  |  |  |  |
| florence2 | detect |  |  |  |  |  |  |  |
| fomo | point |  |  |  |  |  |  |  |
| grounding_dino | detect |  |  |  |  |  |  |  |
| internvl3 | detect |  |  |  |  |  |  |  |
| kosmos2 | detect |  |  |  |  |  |  |  |
| l2cs | gaze |  |  |  |  |  |  |  |
| lfm2vl | detect |  |  |  |  |  |  |  |
| locateanything | detect |  |  |  |  |  |  |  |
| locateanything | point |  |  |  |  |  |  |  |
| mobilenetv4 | classify | ✓ | ✓ | exp | exp | exp |  |  |
| mobilesam | segment |  |  |  |  |  |  |  |
| nafnet | restore | exp | exp | exp | exp | exp |  |  |
| owlv2 | detect |  |  |  |  |  |  |  |
| picodet | detect | exp | exp | exp | exp | exp |  |  |
| pidnet | semantic |  |  |  |  |  |  |  |
| qwen3vl | detect |  |  |  |  |  |  |  |
| realesrgan | restore | exp | exp | exp | exp | exp |  |  |
| resnet | classify | ✓ | ✓ | exp | exp | exp |  |  |
| rfdetr | detect | ✓ | ✓ | ✓ | ✓ |  | ✓ | exp |
| rfdetr | segment | exp | exp | exp | exp |  | exp |  |
| rfdetr | pose | exp | exp | exp | exp |  | exp |  |
| rfdetr | obb | exp | exp | exp | exp |  |  |  |
| rtdetr | detect | exp | exp | exp | exp |  |  | exp |
| rtdetrv2 | detect | exp | exp | exp | exp |  |  |  |
| rtdetrv4 | detect | exp | exp | exp | exp |  |  |  |
| rtmdet | detect | exp | exp | exp | exp | exp |  |  |
| sam | segment |  |  |  |  |  |  |  |
| sam2 | segment |  |  |  |  |  |  |  |
| siglip2 | classify | ✓ | exp | exp | exp | exp |  |  |
| smolvlm2 | detect |  |  |  |  |  |  |  |
| yolo1 | detect | exp | exp | exp | exp | exp |  |  |
| yolo2 | detect | exp | exp | exp | exp | exp | exp |  |
| yolo3 | detect | exp | exp | exp | exp | exp | ✓ |  |
| yolo4 | detect | exp | exp | exp | exp | exp | exp |  |
| yolo7 | detect | exp | exp | exp | exp | exp | exp |  |
| yolo9 | detect | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | exp |
| yolo9_e2e | detect | exp | exp | exp | exp | exp |  |  |
| yolo9_p2 | detect | ✓ | exp | exp | exp | exp |  |  |
| yolonas | detect | exp | exp | exp | exp | exp |  |  |
| yolonas | pose | exp | exp | exp | exp | exp |  |  |
| yolox | detect | exp | exp | exp | exp | exp |  | exp |
| zipdepth | depth | exp | exp | exp | exp | exp |  |  |
<!-- export-support:end -->

YOLOv9-P2 is a small-object variant of YOLOv9 with an extra stride-4 detection
scale, built for aerial/tiny-object imagery where objects fall below ~16 px
(on regular datasets like COCO, prefer stock YOLOv9). A VisDrone-trained
research preview is available as
[`LibreYOLO9P2s-visdrone.pt`](https://huggingface.co/LibreYOLO/LibreYOLO9P2s-visdrone)
(non-commercial license); train your own with
`LibreYOLO9P2(None, size="s").train(..., pretrained="LibreYOLO9s.pt")`.

## License

- **Code:** MIT License
- **Weights:** Pre-trained weights may inherit licensing from the original source. Check the license in the specific HF repo of weights that you are interested in. LibreYOLO HF models always have a license.
