---
name: use-libreyolo
description: >-
  Use LibreYOLO as a computer vision library: run inference, train, validate,
  and export object-detection / segmentation models on your own images. This is
  the guide for *using* the `libreyolo` pip package — not for contributing to or
  developing it. Use whenever someone wants to detect or segment with a YOLO9 or
  RF-DETR model, train on a YOLO-format dataset, measure mAP, or export to ONNX
  / TensorRT / OpenVINO / CoreML / NCNN / TFLite. Covers both the `libreyolo`
  CLI and the `from libreyolo import LibreYOLO` Python API.
---

# Use LibreYOLO

LibreYOLO is an MIT-licensed CV library. Its API follows the **Ultralytics YOLO
standard**, which means two things you can rely on:

1. **CLI and Python mirror each other** — same verbs (`predict`, `train`,
   `val`, `export`), same argument names. Use whichever the user prefers.
2. The CLI is **self-describing**. Never guess a flag — ask the binary (see
   *Exact options* below). This is also why this skill stays short: it teaches
   the shape, the tool supplies the details for the installed version.

Flagship models: **YOLO9** (CNN) and **RF-DETR** (transformer). Weights
auto-download on first use — pass a name like `LibreYOLO9t.pt` / `LibreRFDETRn.pt`,
or a path to the user's own `.pt`.

## Setup

```bash
pip install libreyolo
libreyolo checks      # verify install, CUDA/MPS, and optional export backends
```

## The four verbs

Arguments take either Ultralytics `key=value` **or** `--key value`. Examples use
`key=value`.

**Predict — run inference**
```bash
libreyolo predict model=LibreYOLO9t.pt source=path/to/img_or_dir conf=0.25 save=true
```
```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE
model = LibreYOLO("LibreYOLO9t.pt")
results = model(SAMPLE_IMAGE, save=True)   # equivalently: model.predict(source=...)
```

**Train — needs a YOLO-format dataset YAML**
```bash
libreyolo train model=LibreYOLO9t.pt data=coco8.yaml epochs=100 imgsz=640 batch=16 device=0
```
```python
model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

**Validate — mAP on a split**
```bash
libreyolo val model=runs/train/exp/weights/best.pt data=coco8.yaml save_json=true save_plots=true
```

**Export — onnx · torchscript · tensorrt · openvino · ncnn · tflite · coreml**
```bash
libreyolo export model=runs/train/exp/weights/best.pt format=onnx half=true
```

## Exact, version-correct options

The CLI is the source of truth for the installed version. Prefer these over
recalling flags from memory:

```bash
libreyolo --help                 # list every command
libreyolo train --help-json      # full argument schema for one command, as JSON
libreyolo models                 # list supported models / families / tasks
libreyolo predict ... --json     # machine-readable results to stdout
libreyolo ... --quiet            # suppress progress output (good for scripting)
```

In Python the same kwargs apply; `help(LibreYOLO.train)` and `model.info()`
describe the loaded model.

## Notes

- **Tasks:** detection — and segmentation on YOLO9 / RF-DETR — are the
  well-supported core. Pose, OBB, classify, and semantic are experimental;
  confirm with `libreyolo models` before relying on them.
- **Datasets** are standard YOLO format, so existing Ultralytics dataset YAMLs
  (e.g. `coco8.yaml`) work unchanged.
- **Outputs** land under `runs/` (`runs/detect`, `runs/train`, `runs/val`).
- **Stuck or an import/CUDA error?** Run `libreyolo checks` first - it diagnoses
  the environment and export-backend problems before you debug anything else.
