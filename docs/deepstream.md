# DeepStream export

`deepstream=True` on the ONNX export produces artifacts ready for NVIDIA
DeepStream's `nvinfer` element (Jetson and x86 dGPU):

```python
from libreyolo import LibreYOLO9

model = LibreYOLO9("libreyolo9s.pt", size="s")
model.export(format="onnx", deepstream=True)
```

This writes three files next to each other:

- `libreyolo9s.onnx`: the detection graph with a single output tensor of
  shape `(batch, num_detections, 6)`, rows `[x1, y1, x2, y2, score,
  class_id]` in network-input pixel coordinates.
- `config_infer_primary_libreyolo9s.txt`: an `nvinfer` configuration with
  the family's preprocessing constants, class count, clustering (NMS)
  thresholds, and parser wiring filled in.
- `libreyolo9s_labels.txt`: one class name per line.

DeepStream builds the TensorRT engine from the ONNX on first run and caches
it next to the model.

## The parser library

`nvinfer` needs a custom bounding-box parser for this output layout. The
generated config targets `NvDsInferParseYolo` from the MIT-licensed
[DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)
project. Build it once per device:

```bash
git clone https://github.com/marcoslucianops/DeepStream-Yolo
cd DeepStream-Yolo
# CUDA_VER: see /usr/local/cuda/version.json (Jetson and x86 differ)
CUDA_VER=12.8 make -C nvdsinfer_custom_impl_Yolo
```

Adjust `custom-lib-path` in the generated config to the built
`libnvdsinfer_custom_impl_Yolo.so`. No NMS is embedded in the ONNX graph;
the parser applies the confidence threshold and DeepStream's clustering
stage (`cluster-mode=2`) suppresses using `nms-iou-threshold`.

## Supported families

Detection task only. CNN families: yolo9, yolox, yolonas, rtmdet, picodet,
yolo1, yolo2, yolo3, yolo4, yolo7. DETR families: rfdetr, dfine, deim,
deimv2, ec, rtdetr, rtdetrv2, rtdetrv4.

Families whose native preprocessing cannot be expressed by `nvinfer`'s
scalar `net-scale-factor` (per-channel std: rfdetr, ec, DINO-backboned
deimv2 sizes, rtmdet, picodet) have the normalization baked into the
exported graph; the generated config feeds the graph the matching raw
input space, so no manual preprocessing configuration is needed.

## Preprocessing approximations

Two known deviations from the native Python pipelines, both small and
documented here for benchmark accounting:

- Letterbox families (yolo9, yolox, yolonas, rtmdet, yolo2/3/4/7) pad with
  gray natively; `nvinfer` pads black.
- yolonas natively resizes the longest side to 636 inside its 640 canvas;
  `nvinfer`'s `maintain-aspect-ratio` uses the full 640.

For exact-parity workloads, validate on your data before deploying; all
other math is parity-tested against each family's native postprocess.

## Options

- `conf` / `iou` (defaults 0.25 / 0.45) seed `pre-cluster-threshold` and
  `nms-iou-threshold` in the generated config.
- `dynamic=True` exports a dynamic batch axis; set `batch-size` in the
  config to the engine batch you want DeepStream to build.
- `half=True` marks the config `network-mode=2` (fp16 engine build).
- `deepstream=True` and `nms=True` are mutually exclusive.
