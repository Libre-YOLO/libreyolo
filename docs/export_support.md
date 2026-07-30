# Export support

This document is generated from `libreyolo/export/support.py`.
Do not edit the matrix by hand.

`✓` means parity-validated, `exp` means conversion is available without a
numeric parity guarantee, and an empty cell is blocked in preflight.

| Family | Task | onnx | torchscript | executorch | tensorrt | openvino | ncnn | tflite | coreml | coreai |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| birefnet | matte | exp | ✓ | exp |  |  |  |  |  |  |
| clip | classify | ✓ |  |  |  |  |  |  |  | ✓ |
| clip | embed |  |  |  |  |  |  |  |  |  |
| convnext | classify | ✓ | ✓ | exp | ✓ | ✓ | ✓ | ✓ |  | ✓ |
| deim | detect | ✓ | ✓ |  | exp | exp |  |  |  | ✓ |
| deimv2 | detect | exp | ✓ |  | exp | exp |  |  |  | ✓ |
| depth_anything | depth | ✓ | ✓ | exp | ✓ | ✓ |  |  |  | ✓ |
| depth_anything3 | depth |  |  |  |  |  |  |  |  |  |
| dexined | edge | ✓ |  | exp |  |  |  |  |  |  |
| dfine | detect | ✓ | ✓ |  | exp | ✓ |  |  |  | ✓ |
| dfine | segment | ✓ | ✓ | exp | exp | ✓ |  |  |  |  |
| dinov2 | semantic | ✓ | ✓ |  | ✓ | ✓ |  |  |  |  |
| dinov2 | classify | ✓ | ✓ |  |  |  |  |  |  | exp |
| dinov2 | embed |  |  |  |  |  |  |  |  |  |
| ec | detect | ✓ | ✓ | ✓ | exp | ✓ |  |  |  | ✓ |
| ec | pose | ✓ | ✓ | exp | exp | exp |  |  |  |  |
| ec | segment | ✓ | ✓ | exp | exp | ✓ |  |  |  |  |
| edgetam | segment |  |  |  |  |  |  |  |  |  |
| efficientnetv2 | classify | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ |
| eomt | semantic | ✓ | ✓ |  | ✓ | ✓ |  |  |  |  |
| eomt | segment |  |  |  |  |  |  |  |  |  |
| eomt | panoptic |  |  |  |  |  |  |  |  |  |
| feynobg | matte | exp | ✓ | exp |  |  |  |  |  |  |
| florence2 | detect |  |  |  |  |  |  |  |  |  |
| fomo | point | ✓ | ✓ | exp | ✓ | ✓ | ✓ |  |  | ✓ |
| grounding_dino | detect |  |  |  |  |  |  |  |  |  |
| internvl3 | detect |  |  |  |  |  |  |  |  |  |
| kosmos2 | detect |  |  |  |  |  |  |  |  |  |
| l2cs | gaze | ✓ | ✓ |  |  |  |  |  |  |  |
| lfm2vl | detect |  |  |  |  |  |  |  |  |  |
| lingbotvision | semantic | ✓ | ✓ |  | exp | ✓ |  |  |  | ✓ |
| locateanything | detect |  |  |  |  |  |  |  |  |  |
| locateanything | point |  |  |  |  |  |  |  |  |  |
| mobilenetv4 | classify | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ |
| mobilesam | segment |  |  |  |  |  |  |  |  |  |
| moge2 | normal | ✓ |  |  |  |  |  |  |  |  |
| nafnet | restore | ✓ | ✓ | exp | ✓ | ✓ | ✓ |  |  | ✓ |
| omdet_turbo | detect |  |  |  |  |  |  |  |  |  |
| ov_deim | detect |  |  |  |  |  |  |  |  |  |
| owlv2 | detect |  |  |  |  |  |  |  |  |  |
| picodet | detect | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  | ✓ |
| picosam3 | segment | ✓ |  |  |  |  |  |  |  |  |
| pidnet | semantic | ✓ | ✓ | ✓ | exp | ✓ | ✓ | ✓ |  | ✓ |
| ppocr | ocr |  |  |  |  |  |  |  |  |  |
| qwen3vl | detect |  |  |  |  |  |  |  |  |  |
| realesrgan | restore | ✓ | ✓ | exp | ✓ | ✓ | ✓ | ✓ |  | ✓ |
| resnet | classify | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ |
| rfdetr | detect | ✓ | ✓ | ✓ | ✓ | ✓ |  |  | exp | ✓ |
| rfdetr | segment | ✓ | ✓ | exp | exp | exp |  |  |  |  |
| rfdetr | pose | ✓ | ✓ | exp | exp | exp |  |  |  |  |
| rfdetr | obb | ✓ | ✓ | exp | exp | exp |  |  |  |  |
| rtdetr | detect | ✓ | ✓ | ✓ | exp | ✓ |  |  | exp | ✓ |
| rtdetrv2 | detect | ✓ | ✓ | ✓ | exp | exp |  |  |  | ✓ |
| rtdetrv4 | detect | ✓ | ✓ | ✓ | exp | ✓ |  |  |  | ✓ |
| rtmdet | detect | ✓ | ✓ |  | ✓ | ✓ |  |  |  | ✓ |
| rtmdet | segment |  |  |  |  |  |  |  |  |  |
| sam | segment |  |  |  |  |  |  |  |  |  |
| sam2 | segment |  |  |  |  |  |  |  |  |  |
| sam3 | segment |  |  |  |  |  |  |  |  |  |
| sam3dbody | mesh |  |  |  |  |  |  |  |  |  |
| segformer | semantic | ✓ | ✓ |  |  | ✓ |  |  |  |  |
| siglip2 | classify | ✓ |  |  |  |  |  |  |  | ✓ |
| siglip2 | embed |  |  |  |  |  |  |  |  |  |
| smolvlm2 | detect |  |  |  |  |  |  |  |  |  |
| swinir | restore | ✓ | ✓ | exp | ✓ | ✓ |  | ✓ |  |  |
| teed | edge | ✓ |  | exp |  |  |  |  |  |  |
| yolo1 | detect | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  | ✓ |
| yolo2 | detect | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  | ✓ |
| yolo3 | detect | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  | ✓ |
| yolo4 | detect | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  | ✓ |
| yolo7 | detect | ✓ | ✓ | ✓ | exp | ✓ | ✓ |  |  | ✓ |
| yolo9 | detect | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | exp | ✓ |
| yolo9_e2e | detect | ✓ | ✓ | ✓ | exp | ✓ | ✓ |  |  | ✓ |
| yolo9_p2 | detect | ✓ | ✓ | exp | exp | ✓ | ✓ |  |  | ✓ |
| yolonas | detect | ✓ | ✓ | exp | exp | ✓ | ✓ | ✓ |  | ✓ |
| yolonas | pose | ✓ | ✓ | exp | exp | ✓ | ✓ |  |  |  |
| yolox | detect | ✓ | ✓ | ✓ | exp | ✓ | ✓ | ✓ | exp | ✓ |
| zipdepth | depth | ✓ | ✓ | exp | exp | ✓ | ✓ |  |  | ✓ |

## Parity thresholds

- Detection and OBB: matched box IoU above 0.95 and score MAE below 0.01.
- Segmentation and panoptic: mask IoU above 0.95.
- Pose: keypoint L2 below 2 pixels at native resolution.
- Classification: logits cosine above 0.999 and equal top-1 class.
- Depth and restoration: PSNR above 40 dB against native output.
- Surface normals: mean angular error below 0.1 degree.
- Point: peak locations equal within one output cell.

## Validated constraints

A check mark applies only under any constraint listed here.

- `birefnet` / `matte` / `torchscript`: fixed 1024x1024 input
- `clip` / `classify` / `onnx`: frozen-class labels and fixed input resolution
- `clip` / `classify` / `coreai`: frozen class set and fixed export canvas; permissively licensed trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `convnext` / `classify` / `tensorrt`: FP32 with fixed family-native input resolution
- `convnext` / `classify` / `openvino`: fixed family-native input resolution
- `convnext` / `classify` / `ncnn`: PNNX/NCNN 20260526 CPU FP32 at the family-native input resolution; two-input raw parity, factory reload, metadata, and public predict parity
- `convnext` / `classify` / `coreai`: fixed export canvas; a representative published trained ImageNet checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `deim` / `detect` / `onnx`: DETR query rows are aligned as an unordered set for parity
- `deim` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `deimv2` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `depth_anything` / `depth` / `tensorrt`: FP32 with a fixed input resolution divisible by 14
- `depth_anything` / `depth` / `openvino`: fixed input resolution divisible by 14
- `depth_anything` / `depth` / `coreai`: fixed export canvas; permissively licensed trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `dexined` / `edge` / `onnx`: fixed-resolution batch-1 edge-probability canvas
- `dfine` / `detect` / `openvino`: fixed export canvas
- `dfine` / `detect` / `coreai`: fixed export canvas; trained LibreDFINEn weights are covered on macOS 27 by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `dfine` / `segment` / `openvino`: fixed export canvas
- `dinov2` / `semantic` / `onnx`: fixed 518x518 input
- `dinov2` / `semantic` / `torchscript`: fixed 518x518 input
- `dinov2` / `semantic` / `tensorrt`: FP32 with a fixed family-native export canvas
- `dinov2` / `semantic` / `openvino`: fixed family-native export canvas
- `dinov2` / `classify` / `onnx`: fixed 224x224 input
- `dinov2` / `classify` / `torchscript`: fixed 224x224 input
- `ec` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `ec` / `detect` / `openvino`: fixed export canvas
- `ec` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `ec` / `pose` / `onnx`: fixed 640x640 input
- `ec` / `pose` / `torchscript`: fixed 640x640 input
- `ec` / `segment` / `onnx`: fixed 640x640 input
- `ec` / `segment` / `torchscript`: fixed 640x640 input
- `ec` / `segment` / `openvino`: fixed 640x640 input
- `efficientnetv2` / `classify` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `efficientnetv2` / `classify` / `tensorrt`: FP32 with fixed family-native input resolution
- `efficientnetv2` / `classify` / `openvino`: fixed family-native input resolution
- `efficientnetv2` / `classify` / `ncnn`: PNNX/NCNN 20260526 CPU FP32 at the family-native input resolution; two-input raw parity, factory reload, metadata, and public predict parity
- `efficientnetv2` / `classify` / `coreai`: fixed export canvas; a representative published trained ImageNet checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `eomt` / `semantic` / `onnx`: fixed 512x512 input
- `eomt` / `semantic` / `torchscript`: fixed 512x512 input
- `eomt` / `semantic` / `tensorrt`: FP32 with a fixed family-native export canvas
- `eomt` / `semantic` / `openvino`: fixed family-native export canvas
- `feynobg` / `matte` / `torchscript`: fixed 1024x1024 input
- `fomo` / `point` / `tensorrt`: FP32 with a fixed 96x96 input
- `fomo` / `point` / `openvino`: fixed square input
- `fomo` / `point` / `ncnn`: PNNX/NCNN 20260526 CPU FP32 with a fixed 96x96 input; two-input raw parity, factory reload, metadata, and public predict parity
- `fomo` / `point` / `coreai`: native 96 canvas; a deterministic model state trained from scratch for eight steps on synthetic tensors is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; this validates conversion and the existing heatmap contract, not point-localization accuracy
- `l2cs` / `gaze` / `onnx`: head-only contract: each input image is one face crop
- `l2cs` / `gaze` / `torchscript`: head-only contract: each input image is one face crop
- `lingbotvision` / `semantic` / `onnx`: fixed 512x512 input
- `lingbotvision` / `semantic` / `torchscript`: fixed 512x512 input
- `lingbotvision` / `semantic` / `openvino`: fixed family-native export canvas
- `lingbotvision` / `semantic` / `coreai`: fixed family-native canvases (PIDNet 1024, LingBotVision 512); trained LibrePIDNets-sem and LibreLingBotVisions-sem checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; exported backends already implement the shared dense-logit resize and argmax contract
- `mobilenetv4` / `classify` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `mobilenetv4` / `classify` / `tensorrt`: FP32 with fixed family-native input resolution
- `mobilenetv4` / `classify` / `openvino`: fixed family-native input resolution
- `mobilenetv4` / `classify` / `ncnn`: PNNX/NCNN 20260526 CPU FP32 at the family-native input resolution; two-input raw parity, factory reload, metadata, and public predict parity
- `mobilenetv4` / `classify` / `coreai`: fixed export canvas; a representative published trained ImageNet checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `moge2` / `normal` / `onnx`: fixed square batch-1 export canvas divisible by 14; exported inference rejects non-square sources rather than stretching image-plane geometry; the official MIT ViT-S/B/L normal checkpoints are covered by FP32 same-canvas native-versus-ONNX angular parity below 0.1 degree
- `nafnet` / `restore` / `onnx`: fixed-resolution export canvas
- `nafnet` / `restore` / `torchscript`: fixed-resolution export canvas
- `nafnet` / `restore` / `tensorrt`: FP32 with a fixed-resolution export canvas
- `nafnet` / `restore` / `openvino`: fixed-resolution export canvas
- `nafnet` / `restore` / `ncnn`: PNNX/NCNN 20260526 CPU FP32 with a fixed-resolution export canvas; two-input raw parity, factory reload, metadata, and public predict parity
- `nafnet` / `restore` / `coreai`: fixed export canvas; permissively licensed trained restoration checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `picodet` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `picodet` / `detect` / `tensorrt`: TensorRT 10.16 FP32 with a fixed canvas; YOLO1 requires 448x448
- `picodet` / `detect` / `openvino`: fixed export canvas; YOLO1 requires 448x448
- `picodet` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `picosam3` / `segment` / `onnx`: raw fixed-96 ROI contract: roi_image -> mask_logits
- `pidnet` / `semantic` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `pidnet` / `semantic` / `openvino`: fixed square input
- `pidnet` / `semantic` / `ncnn`: PNNX/NCNN 20260526 CPU FP32 with a fixed export canvas; two-input raw parity, factory reload, metadata, and public predict parity
- `pidnet` / `semantic` / `coreai`: fixed family-native canvases (PIDNet 1024, LingBotVision 512); trained LibrePIDNets-sem and LibreLingBotVisions-sem checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; exported backends already implement the shared dense-logit resize and argmax contract
- `realesrgan` / `restore` / `onnx`: dynamic spatial input
- `realesrgan` / `restore` / `torchscript`: fixed-resolution export canvas
- `realesrgan` / `restore` / `tensorrt`: FP32 with a fixed-resolution export canvas
- `realesrgan` / `restore` / `openvino`: fixed-resolution export canvas
- `realesrgan` / `restore` / `ncnn`: PNNX/NCNN 20260526 CPU FP32 with a fixed-resolution export canvas; two-input raw parity, factory reload, metadata, and public predict parity
- `realesrgan` / `restore` / `tflite`: fixed-resolution export canvas
- `realesrgan` / `restore` / `coreai`: fixed export canvas; permissively licensed trained restoration checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `resnet` / `classify` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `resnet` / `classify` / `tensorrt`: FP32 with fixed family-native input resolution
- `resnet` / `classify` / `openvino`: fixed family-native input resolution
- `resnet` / `classify` / `ncnn`: PNNX/NCNN 20260526 CPU FP32 at the family-native input resolution; two-input raw parity, factory reload, metadata, and public predict parity
- `resnet` / `classify` / `coreai`: fixed export canvas; a representative published trained ImageNet checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `rfdetr` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `rfdetr` / `detect` / `coreai`: fixed export canvas; trained LibreRFDETRn weights are covered on macOS 27 against the graph the exporter itself prepares, using direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin. Conversion needed _rebake_rfdetr_pos_embed in export/coreai.py: the backbone bakes its position embedding for its configured 384 canvas, so exporting at any other size left an antialiased bicubic in the graph and the converter has no lowering for aten._upsample_bicubic2d_aa. The rebake re-runs the model's OWN baking path for the actual canvas, so the interpolation happens eagerly, outside the graph, computing exactly what it computed before. NOTE the reference. This family is verified against the exporter's prepared graph, not against ONNX, and the difference is not a detail: at a 640 canvas the rfdetr ONNX artifact disagrees with that same prepared graph by 9.3e-01. Core AI's rebake preserves the antialiased resize the eager model performs, whereas the ONNX path disables antialiasing (the model checks torch.onnx.is_in_onnx_export). Which artifact is right is an ONNX question and is not settled here, but ONNX cannot be used as the reference for this family at a non-native canvas.
- `rfdetr` / `segment` / `onnx`: fixed task-native input resolution
- `rfdetr` / `segment` / `torchscript`: fixed task-native input resolution
- `rfdetr` / `pose` / `onnx`: fixed task-native input resolution
- `rfdetr` / `pose` / `torchscript`: fixed task-native input resolution
- `rfdetr` / `obb` / `onnx`: fixed task-native input resolution
- `rfdetr` / `obb` / `torchscript`: fixed task-native input resolution
- `rtdetr` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `rtdetr` / `detect` / `openvino`: fixed export canvas
- `rtdetr` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `rtdetrv2` / `detect` / `onnx`: fixed export canvas; same-device CPU raw parity after one shared unordered-query permutation; published Apache-2.0 trained checkpoint covered by non-square public predict parity
- `rtdetrv2` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `rtdetrv2` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `rtdetrv4` / `detect` / `onnx`: fixed export canvas; same-device CPU raw parity after one shared unordered-query permutation; published Apache-2.0 trained checkpoint covered by non-square public predict parity
- `rtdetrv4` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `rtdetrv4` / `detect` / `openvino`: fixed export canvas
- `rtdetrv4` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `rtmdet` / `detect` / `tensorrt`: TensorRT 10.16 FP32 with a fixed canvas; YOLO1 requires 448x448
- `rtmdet` / `detect` / `openvino`: fixed export canvas; YOLO1 requires 448x448
- `rtmdet` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `segformer` / `semantic` / `onnx`: fixed square input divisible by 32
- `segformer` / `semantic` / `torchscript`: fixed square input divisible by 32
- `segformer` / `semantic` / `openvino`: fixed square input divisible by 32
- `siglip2` / `classify` / `onnx`: frozen-class labels and fixed input resolution
- `siglip2` / `classify` / `coreai`: frozen class set and fixed export canvas; permissively licensed trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `swinir` / `restore` / `onnx`: fixed export canvas; raw-output and predict parity are validated when the source dimensions exactly match that canvas. Smaller sources are padded to the canvas before the exported transformer and can diverge from native variable-size inference.
- `swinir` / `restore` / `torchscript`: fixed export canvas; raw-output and predict parity are validated when the source dimensions exactly match that canvas. Smaller sources are padded to the canvas before the exported transformer and can diverge from native variable-size inference.
- `swinir` / `restore` / `tensorrt`: FP32 with a fixed export canvas; raw-output and predict parity are validated when the source dimensions exactly match that canvas.
- `swinir` / `restore` / `openvino`: fixed export canvas; raw-output and predict parity are validated when the source dimensions exactly match that canvas. Smaller sources are padded to the canvas before the exported transformer and can diverge from native variable-size inference.
- `swinir` / `restore` / `tflite`: fixed export canvas; raw-output and predict parity are validated when the source dimensions exactly match that canvas. Smaller sources are padded to the canvas before the exported transformer and can diverge from native variable-size inference.
- `teed` / `edge` / `onnx`: fixed-resolution batch-1 edge-probability canvas
- `yolo1` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `yolo1` / `detect` / `tensorrt`: TensorRT 10.16 FP32 with a fixed canvas; YOLO1 requires 448x448
- `yolo1` / `detect` / `openvino`: fixed export canvas; YOLO1 requires 448x448
- `yolo1` / `detect` / `ncnn`: fixed 448x448 input
- `yolo1` / `detect` / `coreai`: fixed family-native canvases (YOLO1 448, YOLO2 608, YOLO3 416, YOLO4 608); representative published trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; Core AI graph preparation exactly folds Darknet inference batch normalization into the preceding convolutions because Core AI 0.4.1 does not preserve Darknet's epsilon-after-square-root formula
- `yolo2` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `yolo2` / `detect` / `tensorrt`: FP32 with a fixed export canvas
- `yolo2` / `detect` / `openvino`: fixed export canvas; YOLO1 requires 448x448
- `yolo2` / `detect` / `coreai`: fixed family-native canvases (YOLO1 448, YOLO2 608, YOLO3 416, YOLO4 608); representative published trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; Core AI graph preparation exactly folds Darknet inference batch normalization into the preceding convolutions because Core AI 0.4.1 does not preserve Darknet's epsilon-after-square-root formula
- `yolo3` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `yolo3` / `detect` / `tensorrt`: FP32 with a fixed export canvas
- `yolo3` / `detect` / `openvino`: fixed export canvas; YOLO1 requires 448x448
- `yolo3` / `detect` / `coreai`: fixed family-native canvases (YOLO1 448, YOLO2 608, YOLO3 416, YOLO4 608); representative published trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; Core AI graph preparation exactly folds Darknet inference batch normalization into the preceding convolutions because Core AI 0.4.1 does not preserve Darknet's epsilon-after-square-root formula
- `yolo4` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `yolo4` / `detect` / `tensorrt`: FP32 with a fixed export canvas
- `yolo4` / `detect` / `openvino`: fixed export canvas; YOLO1 requires 448x448
- `yolo4` / `detect` / `coreai`: fixed family-native canvases (YOLO1 448, YOLO2 608, YOLO3 416, YOLO4 608); representative published trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; Core AI graph preparation exactly folds Darknet inference batch normalization into the preceding convolutions because Core AI 0.4.1 does not preserve Darknet's epsilon-after-square-root formula
- `yolo7` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `yolo7` / `detect` / `openvino`: fixed export canvas; YOLO1 requires 448x448
- `yolo7` / `detect` / `coreai`: fixed 640x640 export canvas; trained LibreYOLO7b weights are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; the export decoder uses direct arange grids because Core AI 0.4.1 mislowers the equivalent cumulative-sum expression
- `yolo9` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `yolo9` / `detect` / `ncnn`: PNNX/NCNN 20260526 CPU FP32 with a fixed export canvas; trained MIT checkpoint covered by two-input raw parity, factory reload, metadata, and non-square public predict parity
- `yolo9` / `detect` / `coreai`: fixed export canvas; trained LibreYOLO9t weights are covered on macOS 27 by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `yolo9_e2e` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `yolo9_e2e` / `detect` / `openvino`: fixed export canvas; YOLO1 requires 448x448
- `yolo9_e2e` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `yolo9_p2` / `detect` / `openvino`: fixed export canvas; YOLO1 requires 448x448
- `yolo9_p2` / `detect` / `coreai`: fixed 640x640 export canvas; a deterministic YOLO9-P2-T model initialized from the SHA-256-pinned, permissively licensed trained LibreYOLO9t checkpoint is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; this validates conversion, not P2 task accuracy, and does not depend on the restricted VisDrone research-preview checkpoint
- `yolonas` / `detect` / `openvino`: fixed export canvas
- `yolonas` / `detect` / `tflite`: fixed export canvas
- `yolonas` / `detect` / `coreai`: fixed 96x96 export canvas with pre-shaped canonical RGB tensors; a deterministic, license-clean synthetic YOLO-NAS-S state is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; the state receives 12 native training steps and a 20x regression-head scale to make both exported outputs non-degenerate; this validates conversion, not detection accuracy, raw-image preprocessing, or native-640 behavior, and does not convert restricted official weights
- `yolonas` / `pose` / `openvino`: fixed export canvas
- `yolox` / `detect` / `executorch`: ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape
- `yolox` / `detect` / `openvino`: fixed export canvas; YOLO1 requires 448x448
- `yolox` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `zipdepth` / `depth` / `onnx`: fixed-resolution export canvas
- `zipdepth` / `depth` / `torchscript`: fixed-resolution export canvas
- `zipdepth` / `depth` / `openvino`: fixed-resolution export canvas
- `zipdepth` / `depth` / `ncnn`: PNNX/NCNN 20260526 CPU FP32 with a fixed-resolution export canvas; two-input raw parity, factory reload, metadata, and public predict parity
- `zipdepth` / `depth` / `coreai`: fixed export canvas; permissively licensed trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin

## Blocked combinations

- `birefnet` / `matte` / `tensorrt`: TensorRT 10.16 reaches the shared ONNX DeformConv node but cannot parse it because ModulatedDeformConv2d is absent from the plugin registry.
- `birefnet` / `matte` / `openvino`: OpenVINO 2026.2 cannot lower the shared matte decoder's standard ONNX DeformConv-19 operation.
- `birefnet` / `matte` / `ncnn`: BiRefNet's decoder requires torchvision deformable convolution, which PNNX/NCNN cannot lower to a runnable graph.
- `birefnet` / `matte` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `birefnet` / `matte` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `birefnet` / `matte` / `coreai`: The decoder needs torchvision deform_conv2d, which the Core AI converter cannot lower ('unable to handle call function op: deform_conv2d.default'). The same operator already blocks the NCNN path. An encoder-only contract is the realistic route, matching the seam the CUDA graph work used.
- `clip` / `classify` / `torchscript`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `clip` / `classify` / `executorch`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `clip` / `classify` / `tensorrt`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `clip` / `classify` / `openvino`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `clip` / `classify` / `ncnn`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `clip` / `classify` / `tflite`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `clip` / `classify` / `coreml`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `clip` / `embed` / `onnx`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `clip` / `embed` / `torchscript`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `clip` / `embed` / `executorch`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `clip` / `embed` / `tensorrt`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `clip` / `embed` / `openvino`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `clip` / `embed` / `ncnn`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `clip` / `embed` / `tflite`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `clip` / `embed` / `coreml`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `clip` / `embed` / `coreai`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `convnext` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `deim` / `detect` / `executorch`: The trained nano model captures, lowers, and serializes, but ExecuTorch 1.2 runtime execution fails with an invalid delegated tensor dimension order.
- `deim` / `detect` / `ncnn`: NCNN export is not supported for DEIM: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `deim` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `deim` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `deimv2` / `detect` / `executorch`: The trained atto model captures, lowers, and serializes, but the ExecuTorch 1.2 runtime process terminates while executing forward.
- `deimv2` / `detect` / `ncnn`: NCNN export is not supported for DEIMv2: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `deimv2` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `deimv2` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `depth_anything` / `depth` / `ncnn`: PNNX 20260526 reports unsupported batch-index reshapes in the DINOv2 transformer graph; the produced NCNN artifact fails numeric parity.
- `depth_anything` / `depth` / `tflite`: onnx2tf 2.6.7 converts the DINOv2 depth graph, but LiteRT 2.1.2 cannot broadcast [1,3,3,32] and [1,72,72,32] in a generated ADD.
- `depth_anything` / `depth` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `depth_anything3` / `depth` / `onnx`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `torchscript`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `executorch`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `tensorrt`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `openvino`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `ncnn`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `tflite`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `coreml`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `coreai`: The model raises NotImplementedError for every format: depth export is out of scope per ADR 0006, the depth task contract. Depth Anything V2 exports and validates at 5.2e-06, so this is specific to the V3 family and not a Core AI limitation.
- `dexined` / `edge` / `torchscript`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `dexined` / `edge` / `tensorrt`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `dexined` / `edge` / `openvino`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `dexined` / `edge` / `ncnn`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `dexined` / `edge` / `tflite`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `dexined` / `edge` / `coreml`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `dexined` / `edge` / `coreai`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `dfine` / `detect` / `executorch`: Strict capture reaches an unsupported ContextVar read in deformable attention. Forcing the manual exported grid-sample path permits serialization, but ExecuTorch 1.2 runtime execution still fails with an invalid delegated tensor dimension order.
- `dfine` / `detect` / `ncnn`: NCNN export is not supported for D-FINE: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `dfine` / `detect` / `tflite`: onnx2tf flatbuffer-direct lowering crashes in GatherElements shape handling with an axis IndexError.
- `dfine` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `dfine` / `segment` / `ncnn`: NCNN export is not supported for D-FINE: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `dfine` / `segment` / `tflite`: onnx2tf flatbuffer-direct lowering crashes in GatherElements shape handling with an axis IndexError.
- `dfine` / `segment` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `dfine` / `segment` / `coreai`: This family and task have not been validated for Core AI export.
- `dinov2` / `semantic` / `executorch`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `dinov2` / `semantic` / `ncnn`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `dinov2` / `semantic` / `tflite`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `dinov2` / `semantic` / `coreml`: The CoreML wrapper does not implement the dense semantic-logits contract.
- `dinov2` / `semantic` / `coreai`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `dinov2` / `classify` / `executorch`: LibreDINOv2 classify export currently supports ONNX and TorchScript only.
- `dinov2` / `classify` / `tensorrt`: LibreDINOv2 classify export currently supports ONNX and TorchScript only.
- `dinov2` / `classify` / `openvino`: LibreDINOv2 classify export currently supports ONNX and TorchScript only.
- `dinov2` / `classify` / `ncnn`: LibreDINOv2 classify export currently supports ONNX and TorchScript only.
- `dinov2` / `classify` / `tflite`: LibreDINOv2 classify export currently supports ONNX and TorchScript only.
- `dinov2` / `classify` / `coreml`: LibreDINOv2 classify export currently supports ONNX and TorchScript only.
- `dinov2` / `embed` / `onnx`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `dinov2` / `embed` / `torchscript`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `dinov2` / `embed` / `executorch`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `dinov2` / `embed` / `tensorrt`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `dinov2` / `embed` / `openvino`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `dinov2` / `embed` / `ncnn`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `dinov2` / `embed` / `tflite`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `dinov2` / `embed` / `coreml`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `dinov2` / `embed` / `coreai`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `ec` / `detect` / `ncnn`: NCNN export is not supported for EC: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `ec` / `detect` / `tflite`: onnx2tf 2.6.7 emits an ONNX_LAYERNORMALIZATION custom operation that LiteRT 2.1.2 cannot prepare.
- `ec` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `ec` / `pose` / `ncnn`: NCNN export is not supported for EC: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `ec` / `pose` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `ec` / `pose` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `ec` / `pose` / `coreai`: This family and task have not been validated for Core AI export.
- `ec` / `segment` / `ncnn`: NCNN export is not supported for EC: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `ec` / `segment` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `ec` / `segment` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `ec` / `segment` / `coreai`: This family and task have not been validated for Core AI export.
- `edgetam` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `executorch`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `coreml`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `coreai`: Promptable model export is out of scope for the v1 runtime contract.
- `efficientnetv2` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `eomt` / `semantic` / `executorch`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `semantic` / `ncnn`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `eomt` / `semantic` / `tflite`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `eomt` / `semantic` / `coreml`: The CoreML wrapper does not implement the dense semantic-logits contract.
- `eomt` / `semantic` / `coreai`: torch.export refuses the graph: GuardOnDataDependentSymNode, 'Could not guard on data-dependent expression Eq(u0, 1)'. Something in the mask path reads a value off a tensor and branches on it, which becomes an unbacked symbol with no hint the tracer can resolve. This is a real capture failure, not a missing operator and not the task gate: it was measured with the gate open. Fixing it means finding the host read and making the shape static for a fixed export canvas, the same shape of fix as the rfdetr torch._assert.
- `eomt` / `segment` / `onnx`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `torchscript`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `executorch`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `tensorrt`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `openvino`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `ncnn`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `tflite`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `coreml`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `coreai`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `onnx`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `torchscript`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `executorch`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `tensorrt`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `openvino`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `ncnn`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `tflite`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `coreml`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `coreai`: EoMT instance and panoptic export do not yet have runtime parsing.
- `feynobg` / `matte` / `tensorrt`: TensorRT 10.16 reaches the shared ONNX DeformConv node but cannot parse it because ModulatedDeformConv2d is absent from the plugin registry.
- `feynobg` / `matte` / `openvino`: OpenVINO 2026.2 cannot lower the shared matte decoder's standard ONNX DeformConv-19 operation.
- `feynobg` / `matte` / `ncnn`: BiRefNet's decoder requires torchvision deformable convolution, which PNNX/NCNN cannot lower to a runnable graph.
- `feynobg` / `matte` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `feynobg` / `matte` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `feynobg` / `matte` / `coreai`: This family and task have not been validated for Core AI export.
- `florence2` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `executorch`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `fomo` / `point` / `tflite`: LiteRT 2.1.2 cannot invoke the onnx2tf 2.6.7 graph because a DEPTHWISE_CONV_2D reports 16 filter channels versus zero input channels.
- `fomo` / `point` / `coreml`: The CoreML wrapper does not implement the raw point-heatmap contract.
- `grounding_dino` / `detect` / `onnx`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `torchscript`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `executorch`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `tensorrt`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `openvino`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `ncnn`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `tflite`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `coreml`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `coreai`: Open-vocabulary runtime export is out of scope for v1.
- `internvl3` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `executorch`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `executorch`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `l2cs` / `gaze` / `executorch`: The L2CS gaze export contract supports ONNX and TorchScript only.
- `l2cs` / `gaze` / `tensorrt`: The L2CS gaze export contract supports ONNX and TorchScript only.
- `l2cs` / `gaze` / `openvino`: The L2CS gaze export contract supports ONNX and TorchScript only.
- `l2cs` / `gaze` / `ncnn`: The L2CS gaze export contract supports ONNX and TorchScript only.
- `l2cs` / `gaze` / `tflite`: The L2CS gaze export contract supports ONNX and TorchScript only.
- `l2cs` / `gaze` / `coreml`: The L2CS gaze export contract supports ONNX and TorchScript only.
- `l2cs` / `gaze` / `coreai`: The model itself refuses: 'LibreL2CS export to coreai is not implemented. The gaze export contract supports ONNX and TorchScript only.' That is a model-side decision, unchanged by opening the support gate, so nothing about Core AI is being tested here.
- `lfm2vl` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `executorch`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `lingbotvision` / `semantic` / `executorch`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `lingbotvision` / `semantic` / `ncnn`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `lingbotvision` / `semantic` / `tflite`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `lingbotvision` / `semantic` / `coreml`: The CoreML wrapper does not implement the dense semantic-logits contract.
- `locateanything` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `executorch`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `onnx`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `torchscript`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `executorch`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `tensorrt`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `openvino`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `ncnn`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `tflite`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `coreml`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `coreai`: Generative VLM export is out of scope for v1.
- `mobilenetv4` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `mobilesam` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `executorch`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `coreml`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `coreai`: Promptable model export is out of scope for the v1 runtime contract.
- `moge2` / `normal` / `torchscript`: This family is not wired to the fixed-canvas dense unit-normal export and backend renormalization contract.
- `moge2` / `normal` / `executorch`: This family is not wired to the fixed-canvas dense unit-normal export and backend renormalization contract.
- `moge2` / `normal` / `tensorrt`: This family is not wired to the fixed-canvas dense unit-normal export and backend renormalization contract.
- `moge2` / `normal` / `openvino`: This family is not wired to the fixed-canvas dense unit-normal export and backend renormalization contract.
- `moge2` / `normal` / `ncnn`: This family is not wired to the fixed-canvas dense unit-normal export and backend renormalization contract.
- `moge2` / `normal` / `tflite`: This family is not wired to the fixed-canvas dense unit-normal export and backend renormalization contract.
- `moge2` / `normal` / `coreml`: This family is not wired to the fixed-canvas dense unit-normal export and backend renormalization contract.
- `moge2` / `normal` / `coreai`: This family is not wired to the fixed-canvas dense unit-normal export and backend renormalization contract.
- `nafnet` / `restore` / `tflite`: onnx2tf 2.6.7 converts the fixed-canvas graph, but LiteRT 2.1.2 fails at invoke time because input tensor 4539 lacks data.
- `nafnet` / `restore` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `omdet_turbo` / `detect` / `onnx`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `torchscript`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `executorch`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `tensorrt`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `openvino`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `ncnn`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `tflite`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `coreml`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `coreai`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `onnx`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `torchscript`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `executorch`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `tensorrt`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `openvino`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `ncnn`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `tflite`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `coreml`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `coreai`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `onnx`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `torchscript`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `executorch`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `tensorrt`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `openvino`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `ncnn`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `tflite`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `coreml`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `coreai`: Open-vocabulary runtime export is out of scope for v1.
- `picodet` / `detect` / `tflite`: LiteRT 2.1.2 cannot prepare the onnx2tf 2.6.7 artifact because a RESHAPE maps 19,200 input elements to 9,600 output elements.
- `picodet` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `picosam3` / `segment` / `torchscript`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `picosam3` / `segment` / `executorch`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `picosam3` / `segment` / `tensorrt`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `picosam3` / `segment` / `openvino`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `picosam3` / `segment` / `ncnn`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `picosam3` / `segment` / `tflite`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `picosam3` / `segment` / `coreml`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `picosam3` / `segment` / `coreai`: PicoSAM3 currently exports its raw ROI CNN through ONNX only.
- `pidnet` / `semantic` / `coreml`: The CoreML wrapper does not implement the dense semantic-logits contract.
- `ppocr` / `ocr` / `onnx`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `torchscript`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `executorch`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `tensorrt`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `openvino`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `ncnn`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `tflite`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `coreml`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `coreai`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `qwen3vl` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `executorch`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `realesrgan` / `restore` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `resnet` / `classify` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rfdetr` / `detect` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `detect` / `tflite`: onnx2tf emits a flatbuffer at the native 384x384 canvas, but LiteRT cannot allocate it because STRIDED_SLICE receives an input above its supported 5-D rank.
- `rfdetr` / `segment` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `segment` / `tflite`: onnx2tf 2.4.x assigns an invalid NHWC layout to the segmentation-head Einsum (78 channels versus the required 256), so conversion fails.
- `rfdetr` / `segment` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rfdetr` / `segment` / `coreai`: This family and task have not been validated for Core AI export.
- `rfdetr` / `pose` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `pose` / `tflite`: RF-DETR pose-x TFLite conversion exceeded the CPU timebox and 8 GB working memory without producing an artifact on this toolchain.
- `rfdetr` / `pose` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rfdetr` / `pose` / `coreai`: This family and task have not been validated for Core AI export.
- `rfdetr` / `obb` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `obb` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `rfdetr` / `obb` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rfdetr` / `obb` / `coreai`: This family and task have not been validated for Core AI export.
- `rtdetr` / `detect` / `ncnn`: NCNN export is not supported for RT-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rtdetr` / `detect` / `tflite`: LiteRT 2.1.2 rejects the onnx2tf 2.6.7 graph because a CONCATENATION receives incompatible 256 and 1 dimensions.
- `rtdetrv2` / `detect` / `ncnn`: NCNN export is not supported for RT-DETRv2: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rtdetrv2` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `rtdetrv2` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rtdetrv4` / `detect` / `ncnn`: NCNN export is not supported for RT-DETRv4: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rtdetrv4` / `detect` / `tflite`: onnx2tf flatbuffer-direct lowering crashes in GatherElements shape handling with an axis IndexError at the native 640x640 canvas.
- `rtdetrv4` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rtmdet` / `detect` / `executorch`: ExecuTorch 1.2 XNNPACK lowering fails in FuseBatchNormPass because the generated graph has a duplicate fused parameter name.
- `rtmdet` / `detect` / `ncnn`: PNNX 20260526 reports an unregistered nn.Conv2d layer and leaves the RTMDet NCNN graph without usable input blobs.
- `rtmdet` / `detect` / `tflite`: onnx2tf 2.6.7 exports, reloads, and preserves raw output parity, but at the native 640x640 canvas public boxes fall to 0.911 IoU with 29.9 px coordinate drift.
- `rtmdet` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `rtmdet` / `segment` / `onnx`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `torchscript`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `executorch`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `tensorrt`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `openvino`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `ncnn`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `tflite`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `coreml`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `rtmdet` / `segment` / `coreai`: RTMDet-Ins export is not supported yet; the dynamic-kernel mask decode has no exported-runtime contract. Use native PyTorch inference for task='segment'.
- `sam` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `executorch`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `coreml`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `coreai`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `executorch`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `coreml`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `coreai`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `executorch`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `coreml`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `coreai`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3dbody` / `mesh` / `onnx`: Body-mesh export is blocked until its graph outputs, metadata, and backend runtime contract are defined.
- `sam3dbody` / `mesh` / `torchscript`: Body-mesh export is blocked until its graph outputs, metadata, and backend runtime contract are defined.
- `sam3dbody` / `mesh` / `executorch`: Body-mesh export is blocked until its graph outputs, metadata, and backend runtime contract are defined.
- `sam3dbody` / `mesh` / `tensorrt`: Body-mesh export is blocked until its graph outputs, metadata, and backend runtime contract are defined.
- `sam3dbody` / `mesh` / `openvino`: Body-mesh export is blocked until its graph outputs, metadata, and backend runtime contract are defined.
- `sam3dbody` / `mesh` / `ncnn`: Body-mesh export is blocked until its graph outputs, metadata, and backend runtime contract are defined.
- `sam3dbody` / `mesh` / `tflite`: Body-mesh export is blocked until its graph outputs, metadata, and backend runtime contract are defined.
- `sam3dbody` / `mesh` / `coreml`: Body-mesh export is blocked until its graph outputs, metadata, and backend runtime contract are defined.
- `sam3dbody` / `mesh` / `coreai`: Body-mesh export is blocked until its graph outputs, metadata, and backend runtime contract are defined.
- `segformer` / `semantic` / `executorch`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `tensorrt`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `ncnn`: PNNX leaves unsupported pnnx.Expression nodes in the SegFormer graph; the generated NCNN network reports 'network graph not ready' and has no runnable input blob.
- `segformer` / `semantic` / `tflite`: onnx2tf 2.6.7 emits a flatbuffer, but LiteRT 2.1.2 cannot prepare its attention reshape (1024 input elements versus 256 output elements).
- `segformer` / `semantic` / `coreml`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `coreai`: The SegFormer Core AI capture path has not been assessed. Its published weights are non-commercial regardless of export format.
- `siglip2` / `classify` / `torchscript`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `siglip2` / `classify` / `executorch`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `siglip2` / `classify` / `tensorrt`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `siglip2` / `classify` / `openvino`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `siglip2` / `classify` / `ncnn`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `siglip2` / `classify` / `tflite`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `siglip2` / `classify` / `coreml`: Frozen-class vision-language export is ONNX-only in v1; re-export the frozen ONNX graph for a different deployment runtime.
- `siglip2` / `embed` / `onnx`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `siglip2` / `embed` / `torchscript`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `siglip2` / `embed` / `executorch`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `siglip2` / `embed` / `tensorrt`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `siglip2` / `embed` / `openvino`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `siglip2` / `embed` / `ncnn`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `siglip2` / `embed` / `tflite`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `siglip2` / `embed` / `coreml`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `siglip2` / `embed` / `coreai`: Embedding export is not implemented in v1; use the native predict()/embed() API.
- `smolvlm2` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `executorch`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `coreml`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `swinir` / `restore` / `ncnn`: PNNX writes NCNN artifacts after reporting unsupported 5-rank Permute operations, but the NCNN runtime process exits while loading or executing the resulting graph.
- `swinir` / `restore` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `swinir` / `restore` / `coreai`: The export process DIES rather than hangs, and the kill point moves between runs, which is the signature of memory exhaustion rather than a stuck loop. One run reached 'Step 3/3: Optimizing and writing the asset' before stopping; a later run of the same graph at the same 128 canvas died inside to_coreai() before returning, in both cases with a leaked-semaphore warning and no traceback. Window attention unrolls into a very large number of small ops, so the converter's peak memory is the prime suspect on a 16 GB machine. Next steps: watch RSS during conversion, try the smallest available size at a 64 canvas, and check the system log for a memory kill. Do NOT assume optimize() is at fault; an earlier note said so on the strength of a single run and the second run contradicted it.
- `teed` / `edge` / `torchscript`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `teed` / `edge` / `tensorrt`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `teed` / `edge` / `openvino`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `teed` / `edge` / `ncnn`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `teed` / `edge` / `tflite`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `teed` / `edge` / `coreml`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `teed` / `edge` / `coreai`: The edge exported-runtime contract is ONNX-only in v1; add runtime parity before enabling another format.
- `yolo1` / `detect` / `tflite`: onnx2tf 2.6.7 emits an ONNX_EINSUM custom operation that LiteRT 2.1.2 cannot prepare at the native 448x448 canvas.
- `yolo1` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo2` / `detect` / `tflite`: LiteRT 2.1.2 cannot prepare the onnx2tf 2.6.7 artifact because a RESHAPE maps 4,225 input elements to one output element.
- `yolo2` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo3` / `detect` / `tflite`: A public-domain trained checkpoint exports, reloads, and preserves normalized raw parity, but public top-k class membership changes.
- `yolo3` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo4` / `detect` / `tflite`: onnx2tf 2.6.7 exports and runs, but public boxes fall to 0 IoU with 176 px coordinate drift on the deterministic full model.
- `yolo4` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo7` / `detect` / `tflite`: The converted LiteRT graph changes decoded box coordinates beyond the detector parity tolerance.
- `yolo7` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo9_e2e` / `detect` / `tflite`: onnx2tf 2.6.7 exports a runnable artifact, but public top-k class membership changes after LiteRT 2.1.2 conversion.
- `yolo9_e2e` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolo9_p2` / `detect` / `tflite`: onnx2tf 2.6.7 exports a runnable artifact, but public top-k class membership changes after LiteRT 2.1.2 conversion.
- `yolo9_p2` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolonas` / `detect` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolonas` / `pose` / `tflite`: LiteRT rejects the converted pose graph because a CONCATENATION input has an unsupported/invalid tensor type.
- `yolonas` / `pose` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
- `yolonas` / `pose` / `coreai`: This family and task have not been validated for Core AI export.
- `zipdepth` / `depth` / `tflite`: onnx2tf 2.6.7 flatbuffer-direct conversion does not support the edge-mode Pad operation in ZipDepth's convex upsampler.
- `zipdepth` / `depth` / `coreml`: This family and task are not covered by the family-aware CoreML wrapper.
