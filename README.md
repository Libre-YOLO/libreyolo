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

`✓` supported, `exp` experimental. Empty cells are not currently supported.
All trainable families in the Training column accept universal training
hooks via `callbacks=` and built-in experiment loggers via `loggers=`
(`tensorboard`, `mlflow`, `wandb`).
<table>
  <thead>
    <tr>
      <th rowspan="2">Model family</th>
      <th colspan="7">Inference</th>
      <th rowspan="2">Training</th>
      <th colspan="6">Export formats</th>
    </tr>
    <tr>
      <th>Detection</th>
      <th>Segmentation</th>
      <th>Semantic</th>
      <th>Classification</th>
      <th>Pose</th>
      <th>OBB</th>
      <th>Gaze</th>
      <th>ONNX</th>
      <th>TorchScript</th>
      <th>TensorRT</th>
      <th>OpenVINO</th>
      <th>NCNN</th>
      <th>TFLite (LiteRT)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>⭐ YOLOv9</strong></td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td></tr>
    <tr><td><strong>⭐ RF-DETR</strong></td><td>✓</td><td>✓</td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>exp</td></tr>
    <tr><td>YOLOX</td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td></td></tr>
    <tr><td>YOLOv9-E2E</td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td></td><td></td><td></td></tr>
    <tr><td>YOLOv9-P2</td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td>✓</td><td>exp</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>YOLO-NAS</td><td>✓</td><td></td><td></td><td></td><td>✓</td><td></td><td></td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td></td></tr>
    <tr><td>D-FINE</td><td>✓</td><td>exp</td><td></td><td></td><td></td><td></td><td></td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td></td><td></td></tr>
    <tr><td>DEIM</td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td></td><td></td></tr>
    <tr><td>DEIMv2</td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td></td><td></td></tr>
    <tr><td>RT-DETR</td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td>exp</td><td></td><td></td></tr>
    <tr><td>RT-DETRv2</td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td>exp</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>RT-DETRv4</td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td>exp</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>PicoDet</td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td>exp</td><td>exp</td><td>exp</td><td></td><td></td><td></td><td></td></tr>
    <tr><td>RTMDet</td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td>exp</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>EC</td><td>✓</td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td>exp</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>MobileNetV4</td><td></td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>ConvNeXt</td><td></td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>EfficientNetV2</td><td></td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>ResNet</td><td></td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>L2CS</td><td></td><td></td><td></td><td></td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
  </tbody>
</table>

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
