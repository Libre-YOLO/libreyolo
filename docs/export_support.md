# Export support

This document is generated from `libreyolo/export/support.py`.
Do not edit the matrix by hand.

`✓` means parity-validated, `exp` means conversion is available without a
numeric parity guarantee, and an empty cell is blocked in preflight.

| Family | Task | onnx | torchscript | tensorrt | openvino | ncnn | tflite | coreml |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| birefnet | matte | exp | ✓ | exp | exp |  |  |  |
| clip | classify | ✓ |  |  |  |  |  |  |
| convnext | classify | ✓ | ✓ | exp | exp | ✓ | ✓ |  |
| deim | detect | exp | ✓ | exp | exp |  |  |  |
| deimv2 | detect | exp | ✓ | exp | exp |  |  |  |
| depth_anything | depth | ✓ | ✓ | exp | exp |  |  |  |
| depth_anything3 | depth |  |  |  |  |  |  |  |
| dfine | detect | ✓ | ✓ | exp | exp |  |  |  |
| dfine | segment | ✓ | ✓ | exp | exp |  |  |  |
| dinov2 | semantic | ✓ | ✓ | exp | exp |  |  |  |
| dinov2 | classify | ✓ |  |  |  |  |  |  |
| ec | detect | ✓ | ✓ | exp | exp |  |  |  |
| ec | pose | ✓ | ✓ | exp | exp |  |  |  |
| ec | segment | ✓ | ✓ | exp | exp |  |  |  |
| edgetam | segment |  |  |  |  |  |  |  |
| efficientnetv2 | classify | ✓ | ✓ | exp | exp | ✓ | ✓ |  |
| eomt | semantic | ✓ | ✓ | exp | exp |  |  |  |
| eomt | segment |  |  |  |  |  |  |  |
| eomt | panoptic |  |  |  |  |  |  |  |
| feynobg | matte | exp | ✓ | exp | exp |  |  |  |
| florence2 | detect |  |  |  |  |  |  |  |
| fomo | point | ✓ | ✓ | exp | exp | ✓ |  |  |
| grounding_dino | detect |  |  |  |  |  |  |  |
| internvl3 | detect |  |  |  |  |  |  |  |
| kosmos2 | detect |  |  |  |  |  |  |  |
| l2cs | gaze | ✓ |  |  |  |  |  |  |
| lfm2vl | detect |  |  |  |  |  |  |  |
| lingbotvision | semantic | ✓ | ✓ | exp | exp |  |  |  |
| locateanything | detect |  |  |  |  |  |  |  |
| locateanything | point |  |  |  |  |  |  |  |
| mobilenetv4 | classify | ✓ | ✓ | exp | exp | ✓ | ✓ |  |
| mobilesam | segment |  |  |  |  |  |  |  |
| nafnet | restore | ✓ | ✓ | exp | exp | ✓ |  |  |
| omdet_turbo | detect |  |  |  |  |  |  |  |
| ov_deim | detect |  |  |  |  |  |  |  |
| owlv2 | detect |  |  |  |  |  |  |  |
| picodet | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| picosam3 | segment | ✓ |  |  |  |  |  |  |
| pidnet | semantic | ✓ | ✓ | exp | exp | ✓ | ✓ |  |
| ppocr | ocr |  |  |  |  |  |  |  |
| qwen3vl | detect |  |  |  |  |  |  |  |
| realesrgan | restore | ✓ | ✓ | exp | exp | ✓ | ✓ |  |
| resnet | classify | ✓ | ✓ | exp | exp | ✓ | ✓ |  |
| rfdetr | detect | ✓ | ✓ | ✓ | ✓ |  | exp | exp |
| rfdetr | segment | ✓ | ✓ | exp | exp |  |  |  |
| rfdetr | pose | ✓ | ✓ | exp | exp |  |  |  |
| rfdetr | obb | ✓ | ✓ | exp | exp |  |  |  |
| rtdetr | detect | ✓ | ✓ | exp | exp |  |  | exp |
| rtdetrv2 | detect | exp | ✓ | exp | exp |  |  |  |
| rtdetrv4 | detect | exp | ✓ | exp | exp |  |  |  |
| rtmdet | detect | ✓ | ✓ | exp | exp |  |  |  |
| rtmdet | segment |  |  |  |  |  |  |  |
| sam | segment |  |  |  |  |  |  |  |
| sam2 | segment |  |  |  |  |  |  |  |
| sam3 | segment |  |  |  |  |  |  |  |
| segformer | semantic |  |  |  |  |  |  |  |
| siglip2 | classify | ✓ |  |  |  |  |  |  |
| smolvlm2 | detect |  |  |  |  |  |  |  |
| swinir | restore | exp | exp | exp | exp | exp |  |  |
| yolo1 | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolo2 | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolo3 | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolo4 | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolo7 | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolo9 | detect | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | exp |
| yolo9_e2e | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolo9_p2 | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolonas | detect | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolonas | pose | ✓ | ✓ | exp | exp | ✓ |  |  |
| yolox | detect | ✓ | ✓ | exp | exp | ✓ | ✓ | exp |
| zipdepth | depth | ✓ | ✓ | exp | exp | ✓ |  |  |

## Parity thresholds

- Detection and OBB: matched box IoU above 0.95 and score MAE below 0.01.
- Segmentation and panoptic: mask IoU above 0.95.
- Pose: keypoint L2 below 2 pixels at native resolution.
- Classification: logits cosine above 0.999 and equal top-1 class.
- Depth and restoration: PSNR above 40 dB against native output.
- Point: peak locations equal within one output cell.

## Blocked combinations

- `birefnet` / `matte` / `ncnn`: BiRefNet's decoder requires torchvision deformable convolution, which PNNX/NCNN cannot lower to a runnable graph.
- `birefnet` / `matte` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `birefnet` / `matte` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `clip` / `classify` / `torchscript`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `clip` / `classify` / `tensorrt`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `clip` / `classify` / `openvino`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `clip` / `classify` / `ncnn`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `clip` / `classify` / `tflite`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `clip` / `classify` / `coreml`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `convnext` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `deim` / `detect` / `ncnn`: NCNN export is not supported for DEIM: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `deim` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `deim` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `deimv2` / `detect` / `ncnn`: NCNN export is not supported for DEIMv2: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `deimv2` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `deimv2` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `depth_anything` / `depth` / `ncnn`: PNNX 20260526 reports unsupported batch-index reshapes in the DINOv2 transformer graph; the produced NCNN artifact fails numeric parity.
- `depth_anything` / `depth` / `tflite`: onnx2tf 2.4.x converts the DINOv2 depth graph, but LiteRT rejects a generated FILL node because its dimensions are invalid.
- `depth_anything` / `depth` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `depth_anything3` / `depth` / `onnx`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `torchscript`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `tensorrt`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `openvino`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `ncnn`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `tflite`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `coreml`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `dfine` / `detect` / `ncnn`: NCNN export is not supported for D-FINE: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `dfine` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `dfine` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `dfine` / `segment` / `ncnn`: NCNN export is not supported for D-FINE: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `dfine` / `segment` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `dfine` / `segment` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `dinov2` / `semantic` / `ncnn`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `dinov2` / `semantic` / `tflite`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `dinov2` / `semantic` / `coreml`: The CoreML wrapper does not implement the dense semantic-logits contract.
- `dinov2` / `classify` / `torchscript`: LibreDINOv2 classify export currently supports ONNX only.
- `dinov2` / `classify` / `tensorrt`: LibreDINOv2 classify export currently supports ONNX only.
- `dinov2` / `classify` / `openvino`: LibreDINOv2 classify export currently supports ONNX only.
- `dinov2` / `classify` / `ncnn`: LibreDINOv2 classify export currently supports ONNX only.
- `dinov2` / `classify` / `tflite`: LibreDINOv2 classify export currently supports ONNX only.
- `dinov2` / `classify` / `coreml`: LibreDINOv2 classify export currently supports ONNX only.
- `ec` / `detect` / `ncnn`: NCNN export is not supported for EC: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `ec` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `ec` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `ec` / `pose` / `ncnn`: NCNN export is not supported for EC: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `ec` / `pose` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `ec` / `pose` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `ec` / `segment` / `ncnn`: NCNN export is not supported for EC: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `ec` / `segment` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `ec` / `segment` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `edgetam` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `coreml`: Promptable model export is out of scope for the v1 runtime contract.
- `efficientnetv2` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `eomt` / `semantic` / `ncnn`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `eomt` / `semantic` / `tflite`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `eomt` / `semantic` / `coreml`: The CoreML wrapper does not implement the dense semantic-logits contract.
- `eomt` / `segment` / `onnx`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `torchscript`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `tensorrt`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `openvino`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `ncnn`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `tflite`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `coreml`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `onnx`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `torchscript`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `tensorrt`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `openvino`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `ncnn`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `tflite`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `coreml`: EoMT instance and panoptic export do not yet have runtime parsing.
- `feynobg` / `matte` / `ncnn`: BiRefNet's decoder requires torchvision deformable convolution, which PNNX/NCNN cannot lower to a runnable graph.
- `feynobg` / `matte` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `feynobg` / `matte` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `florence2` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `fomo` / `point` / `tflite`: onnx2tf 2.4.x produces an invalid depthwise-convolution graph for the static SAME-padded FOMO backbone on this toolchain.
- `fomo` / `point` / `coreml`: The CoreML wrapper does not implement the raw point-heatmap contract.
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
- `l2cs` / `gaze` / `torchscript`: The v1 L2CS gaze export contract supports ONNX only.
- `l2cs` / `gaze` / `tensorrt`: The v1 L2CS gaze export contract supports ONNX only.
- `l2cs` / `gaze` / `openvino`: The v1 L2CS gaze export contract supports ONNX only.
- `l2cs` / `gaze` / `ncnn`: The v1 L2CS gaze export contract supports ONNX only.
- `l2cs` / `gaze` / `tflite`: The v1 L2CS gaze export contract supports ONNX only.
- `l2cs` / `gaze` / `coreml`: The v1 L2CS gaze export contract supports ONNX only.
- `lfm2vl` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `lingbotvision` / `semantic` / `ncnn`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `lingbotvision` / `semantic` / `tflite`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `lingbotvision` / `semantic` / `coreml`: The CoreML wrapper does not implement the dense semantic-logits contract.
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
- `mobilenetv4` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `mobilesam` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `coreml`: Promptable model export is out of scope for the v1 runtime contract.
- `nafnet` / `restore` / `tflite`: onnx2tf 2.4.x converts the fixed-canvas graph, but LiteRT fails at invoke time because an internal input tensor lacks data.
- `nafnet` / `restore` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `omdet_turbo` / `detect` / `onnx`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `torchscript`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `tensorrt`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `openvino`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `ncnn`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `tflite`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `coreml`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `onnx`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `torchscript`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `tensorrt`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `openvino`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `ncnn`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `tflite`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `coreml`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `onnx`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `torchscript`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `tensorrt`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `openvino`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `ncnn`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `tflite`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `coreml`: Open-vocabulary runtime export is out of scope for v1.
- `picodet` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `picodet` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `picosam3` / `segment` / `torchscript`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `picosam3` / `segment` / `tensorrt`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `picosam3` / `segment` / `openvino`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `picosam3` / `segment` / `ncnn`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `picosam3` / `segment` / `tflite`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `picosam3` / `segment` / `coreml`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `pidnet` / `semantic` / `coreml`: The CoreML wrapper does not implement the dense semantic-logits contract.
- `ppocr` / `ocr` / `onnx`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `torchscript`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `tensorrt`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `openvino`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `ncnn`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `tflite`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `coreml`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `qwen3vl` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `realesrgan` / `restore` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `resnet` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rfdetr` / `detect` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `segment` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `segment` / `tflite`: onnx2tf 2.4.x assigns an invalid NHWC layout to the segmentation-head Einsum (78 channels versus the required 256), so conversion fails.
- `rfdetr` / `segment` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rfdetr` / `pose` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `pose` / `tflite`: RF-DETR pose-x TFLite conversion exceeded the CPU timebox and 8 GB working memory without producing an artifact on this toolchain.
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
- `rtmdet` / `detect` / `ncnn`: PNNX 20260526 reports an unregistered nn.Conv2d layer and leaves the RTMDet NCNN graph without usable input blobs.
- `rtmdet` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `rtmdet` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rtmdet` / `segment` / `onnx`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `torchscript`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `tensorrt`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `openvino`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `ncnn`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `tflite`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `coreml`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
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
- `sam3` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `coreml`: Promptable model export is out of scope for the v1 runtime contract.
- `segformer` / `semantic` / `onnx`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `torchscript`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `tensorrt`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `openvino`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `ncnn`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `tflite`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `coreml`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `siglip2` / `classify` / `torchscript`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `siglip2` / `classify` / `tensorrt`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `siglip2` / `classify` / `openvino`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `siglip2` / `classify` / `ncnn`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `siglip2` / `classify` / `tflite`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `siglip2` / `classify` / `coreml`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `smolvlm2` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `swinir` / `restore` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `swinir` / `restore` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo1` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolo1` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo2` / `detect` / `tflite`: onnx2tf 2.4.x leaves an unresolved ONNX_CONCAT custom operation; LiteRT cannot prepare the converted detector graph.
- `yolo2` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo3` / `detect` / `tflite`: onnx2tf 2.4.x leaves an unresolved ONNX_CONCAT custom operation; LiteRT cannot prepare the converted detector graph.
- `yolo3` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo4` / `detect` / `tflite`: onnx2tf 2.4.x produces an invalid CONV_2D channel layout for YOLO4; LiteRT fails while allocating tensors.
- `yolo4` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo7` / `detect` / `tflite`: The converted LiteRT graph changes decoded box coordinates beyond the detector parity tolerance.
- `yolo7` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo9_e2e` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolo9_e2e` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo9_p2` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolo9_p2` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolonas` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolonas` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolonas` / `pose` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolonas` / `pose` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `zipdepth` / `depth` / `tflite`: onnx2tf 2.4.x flatbuffer-direct conversion does not support the edge-mode Pad operation in ZipDepth's convex upsampler.
- `zipdepth` / `depth` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
