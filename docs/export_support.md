# Export support

This document is generated from `libreyolo/export/support.py`.
Do not edit the matrix by hand.

`✓` means parity-validated, `exp` means conversion is available without a
numeric parity guarantee, and an empty cell is blocked in preflight.

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

## Parity thresholds

- Detection and OBB: matched box IoU above 0.95 and score MAE below 0.01.
- Segmentation and panoptic: mask IoU above 0.95.
- Pose: keypoint L2 below 2 pixels at native resolution.
- Classification: logits cosine above 0.999 and equal top-1 class.
- Depth and restoration: PSNR above 40 dB against native output.
- Point: peak locations equal within one output cell.

## Blocked combinations

- `birefnet` / `matte` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `birefnet` / `matte` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `clip` / `classify` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `clip` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `convnext` / `classify` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `convnext` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `deim` / `detect` / `ncnn`: NCNN export is not supported for DEIM: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `deim` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `deim` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `deimv2` / `detect` / `ncnn`: NCNN export is not supported for DEIMv2: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `deimv2` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `deimv2` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `depth_anything` / `depth` / `onnx`: Depth Anything V2 has not been validated against the depth export contract.
- `depth_anything` / `depth` / `torchscript`: Depth Anything V2 has not been validated against the depth export contract.
- `depth_anything` / `depth` / `tensorrt`: Depth Anything V2 has not been validated against the depth export contract.
- `depth_anything` / `depth` / `openvino`: Depth Anything V2 has not been validated against the depth export contract.
- `depth_anything` / `depth` / `ncnn`: Depth Anything V2 has not been validated against the depth export contract.
- `depth_anything` / `depth` / `tflite`: Depth Anything V2 has not been validated against the depth export contract.
- `depth_anything` / `depth` / `coreml`: Depth Anything V2 has not been validated against the depth export contract.
- `dfine` / `detect` / `ncnn`: NCNN export is not supported for D-FINE: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `dfine` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `dfine` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `dfine` / `segment` / `ncnn`: NCNN export is not supported for D-FINE: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `dfine` / `segment` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `dfine` / `segment` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `dinov2` / `semantic` / `onnx`: Export for semantic-segmentation models is not implemented yet. Semantic export needs a dense-logits output and backend argmax contract.
- `dinov2` / `semantic` / `torchscript`: Export for semantic-segmentation models is not implemented yet. Semantic export needs a dense-logits output and backend argmax contract.
- `dinov2` / `semantic` / `tensorrt`: Export for semantic-segmentation models is not implemented yet. Semantic export needs a dense-logits output and backend argmax contract.
- `dinov2` / `semantic` / `openvino`: Export for semantic-segmentation models is not implemented yet. Semantic export needs a dense-logits output and backend argmax contract.
- `dinov2` / `semantic` / `ncnn`: Export for semantic-segmentation models is not implemented yet. Semantic export needs a dense-logits output and backend argmax contract.
- `dinov2` / `semantic` / `tflite`: Export for semantic-segmentation models is not implemented yet. Semantic export needs a dense-logits output and backend argmax contract.
- `dinov2` / `semantic` / `coreml`: Export for semantic-segmentation models is not implemented yet. Semantic export needs a dense-logits output and backend argmax contract.
- `dinov2` / `classify` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `dinov2` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `ec` / `detect` / `ncnn`: NCNN export is not supported for EC: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `ec` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `ec` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `ec` / `pose` / `ncnn`: NCNN export is not supported for EC: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `ec` / `pose` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `ec` / `pose` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `ec` / `segment` / `ncnn`: NCNN export is not supported for EC: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `ec` / `segment` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `ec` / `segment` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `efficientnetv2` / `classify` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `efficientnetv2` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `eomt` / `semantic` / `onnx`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `semantic` / `torchscript`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `semantic` / `tensorrt`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `semantic` / `openvino`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `semantic` / `ncnn`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `semantic` / `tflite`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `semantic` / `coreml`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `segment` / `onnx`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `segment` / `torchscript`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `segment` / `tensorrt`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `segment` / `openvino`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `segment` / `ncnn`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `segment` / `tflite`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `segment` / `coreml`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `panoptic` / `onnx`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `panoptic` / `torchscript`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `panoptic` / `tensorrt`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `panoptic` / `openvino`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `panoptic` / `ncnn`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `panoptic` / `tflite`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `eomt` / `panoptic` / `coreml`: EoMT export does not yet have semantic, instance, or panoptic runtime parsing.
- `florence2` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `fomo` / `point` / `onnx`: Export for point-task models is not implemented yet. Point export needs a raw heatmap output and backend peak-decoding contract.
- `fomo` / `point` / `torchscript`: Export for point-task models is not implemented yet. Point export needs a raw heatmap output and backend peak-decoding contract.
- `fomo` / `point` / `tensorrt`: Export for point-task models is not implemented yet. Point export needs a raw heatmap output and backend peak-decoding contract.
- `fomo` / `point` / `openvino`: Export for point-task models is not implemented yet. Point export needs a raw heatmap output and backend peak-decoding contract.
- `fomo` / `point` / `ncnn`: Export for point-task models is not implemented yet. Point export needs a raw heatmap output and backend peak-decoding contract.
- `fomo` / `point` / `tflite`: Export for point-task models is not implemented yet. Point export needs a raw heatmap output and backend peak-decoding contract.
- `fomo` / `point` / `coreml`: Export for point-task models is not implemented yet. Point export needs a raw heatmap output and backend peak-decoding contract.
- `grounding_dino` / `detect` / `onnx`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `torchscript`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `tensorrt`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `openvino`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `ncnn`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `tflite`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `coreml`: Open-vocabulary runtime export is out of scope for v1.
- `internvl3` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `l2cs` / `gaze` / `onnx`: L2CS export awaits the gaze two-head runtime contract.
- `l2cs` / `gaze` / `torchscript`: L2CS export awaits the gaze two-head runtime contract.
- `l2cs` / `gaze` / `tensorrt`: L2CS export awaits the gaze two-head runtime contract.
- `l2cs` / `gaze` / `openvino`: L2CS export awaits the gaze two-head runtime contract.
- `l2cs` / `gaze` / `ncnn`: L2CS export awaits the gaze two-head runtime contract.
- `l2cs` / `gaze` / `tflite`: L2CS export awaits the gaze two-head runtime contract.
- `l2cs` / `gaze` / `coreml`: L2CS export awaits the gaze two-head runtime contract.
- `lfm2vl` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `onnx`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `torchscript`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `tensorrt`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `openvino`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `ncnn`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `tflite`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `coreml`: Generative VLM export is out of scope for v1.
- `mobilenetv4` / `classify` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `mobilenetv4` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `mobilesam` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `coreml`: Promptable model export is out of scope for the v1 runtime contract.
- `nafnet` / `restore` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `nafnet` / `restore` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `owlv2` / `detect` / `onnx`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `torchscript`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `tensorrt`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `openvino`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `ncnn`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `tflite`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `coreml`: Open-vocabulary runtime export is out of scope for v1.
- `picodet` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `picodet` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `pidnet` / `semantic` / `onnx`: PIDNet export awaits the semantic dense-logits runtime contract.
- `pidnet` / `semantic` / `torchscript`: PIDNet export awaits the semantic dense-logits runtime contract.
- `pidnet` / `semantic` / `tensorrt`: PIDNet export awaits the semantic dense-logits runtime contract.
- `pidnet` / `semantic` / `openvino`: PIDNet export awaits the semantic dense-logits runtime contract.
- `pidnet` / `semantic` / `ncnn`: PIDNet export awaits the semantic dense-logits runtime contract.
- `pidnet` / `semantic` / `tflite`: PIDNet export awaits the semantic dense-logits runtime contract.
- `pidnet` / `semantic` / `coreml`: PIDNet export awaits the semantic dense-logits runtime contract.
- `qwen3vl` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `realesrgan` / `restore` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `realesrgan` / `restore` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `resnet` / `classify` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `resnet` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rfdetr` / `detect` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `segment` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `segment` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rfdetr` / `pose` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `pose` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rfdetr` / `obb` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `obb` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `rfdetr` / `obb` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rtdetr` / `detect` / `ncnn`: NCNN export is not supported for RT-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rtdetr` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `rtdetrv2` / `detect` / `ncnn`: NCNN export is not supported for RT-DETRv2: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rtdetrv2` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `rtdetrv2` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rtdetrv4` / `detect` / `ncnn`: NCNN export is not supported for RT-DETRv4: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rtdetrv4` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `rtdetrv4` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rtmdet` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `rtmdet` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `sam` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `coreml`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `coreml`: Promptable model export is out of scope for the v1 runtime contract.
- `siglip2` / `classify` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `siglip2` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `smolvlm2` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `yolo1` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolo1` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo2` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo3` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo4` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo7` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo9_e2e` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolo9_e2e` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo9_p2` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolo9_p2` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolonas` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolonas` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolonas` / `pose` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolonas` / `pose` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolox` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `zipdepth` / `depth` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `zipdepth` / `depth` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
