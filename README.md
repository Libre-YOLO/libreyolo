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

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreYOLO9t.pt")
result = model(SAMPLE_IMAGE, save=True)
```

<details>
<summary><b>Optional pip extras</b></summary>

<br>

The base install covers YOLOv9 and the other detection models, plus training
and inference. Add an extra in brackets when you need a heavier model family
(for example RF-DETR, which needs `transformers`) or an export backend.
Comma-separate to combine, e.g. `pip install "libreyolo[rfdetr,onnx]"`:

| Group | Extras |
| --- | --- |
| Export | `onnx`, `tensorrt`, `openvino`, `ncnn`, `tflite` (alias: `litert`), `coreml` |
| Models | `rfdetr`, `vlm`, `sam`, `openvocab`, `clip`, `gaze` |
| Training | `lora`, `plots`, `tensorboard`, `mlflow`, `wandb` |
| Everything | `pip install "libreyolo[all]"` |

For the full list of extras and per-backend notes, see the [docs](https://www.libreyolo.com/docs#installation).

</details>

To install from source (development, or to track unreleased changes):

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

## Detection models

LibreYOLO is a YOLO library first. The table below covers the main detection
families. `✓` supported. Empty cells are not currently supported.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model family</th>
      <th colspan="3">Inference</th>
      <th rowspan="2">Training</th>
      <th colspan="6">Export formats</th>
    </tr>
    <tr>
      <th>Detection</th>
      <th>Segmentation</th>
      <th>Pose</th>
      <th>ONNX</th>
      <th>TorchScript</th>
      <th>TensorRT</th>
      <th>OpenVINO</th>
      <th>NCNN</th>
      <th>TFLite (LiteRT)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>⭐ YOLOv9</strong></td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr>
    <tr><td><strong>⭐ RF-DETR</strong></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>✓</td></tr>
    <tr><td>YOLOX</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr>
    <tr><td>YOLO-NAS</td><td>✓</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td></tr>
    <tr><td>YOLOv9-E2E</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td></tr>
    <tr><td>YOLOv9-P2</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td></tr>
    <tr><td>YOLOv7</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td></tr>
    <tr><td>D-FINE</td><td>✓</td><td>✓</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr>
    <tr><td>DEIM</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr>
    <tr><td>DEIMv2</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr>
    <tr><td>RT-DETR</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr>
    <tr><td>RT-DETRv2</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr>
    <tr><td>RT-DETRv4</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr>
    <tr><td>DETR</td><td>✓</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr>
    <tr><td>LW-DETR</td><td>✓</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr>
    <tr><td>Deformable DETR</td><td>✓</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr>
    <tr><td>Faster R-CNN</td><td>✓</td><td></td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>EC</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr>
    <tr><td>RTMDet</td><td>✓</td><td>✓</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr>
    <tr><td>PicoDet</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td></tr>
    <tr><td>MiDaS (depth)</td><td></td><td></td><td></td><td></td><td>✓</td><td>✓</td><td></td><td></td><td></td><td></td></tr>
  </tbody>
</table>

## Beyond detection

The same three-line API covers a lot more than boxes: the checkpoint name
selects the task. These families are extras on top of the core library.

<details>
<summary><b>All other tasks and their models</b></summary>

<br>

- **Instance segmentation (promptable):** SAM, SAM 2, SAM 3, MobileSAM, EdgeTAM, PicoSAM3, EoMT
- **Semantic segmentation:** SegFormer, PIDNet, EoMT, DINOv2
- **Panoptic segmentation:** EoMT
- **Oriented boxes (OBB):** RF-DETR
- **Classification:** ResNet, ConvNeXt, MobileNetV4, EfficientNetV2, DINOv2, CLIP, SigLIP2
- **Point detection:** FOMO, LocateAnything
- **Depth estimation:** Depth Anything 3, Depth Anything V2, ZipDepth
- **Surface-normal estimation:** MoGe-2 (ViT-S/B/L)
- **Image restoration & super-resolution:** NAFNet, Real-ESRGAN, SwinIR
- **Background removal (matting):** BiRefNet, FeyNobg
- **OCR:** PP-OCR
- **Gaze estimation:** L2CS
- **Facial recognition (face embedding):** LibreFaceRec (verification + gallery identification)
- **Open-vocabulary & VLM detection:** Grounding DINO, OWLv2, OmDet-Turbo, OV-DEIM, Florence-2, Kosmos-2, Qwen3-VL, InternVL3, LFM2-VL, SmolVLM2, LocateAnything

</details>

## License

- **Code:** MIT License
- **Weights:** Pre-trained weights may inherit licensing from the original source. Check the license in the specific HF repo of weights that you are interested in. LibreYOLO HF models always have a license.
