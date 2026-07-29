# Export support

This document is generated from `libreyolo/export/support.py`.
Do not edit the matrix by hand.

`✓` means parity-validated, `exp` means conversion is available without a
numeric parity guarantee, and an empty cell is blocked in preflight.

| Family | Task | onnx | torchscript | tensorrt | openvino | ncnn | tflite | coreml | coreai |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| birefnet | matte | exp | ✓ | exp | exp |  |  | exp |  |
| clip | classify | ✓ |  |  |  |  |  | exp | ✓ |
| convnext | classify | ✓ | ✓ | exp | exp | ✓ | ✓ | exp | ✓ |
| deim | detect | exp | ✓ | exp | exp |  |  | exp | ✓ |
| deimv2 | detect | exp | ✓ | exp | exp |  |  | exp | ✓ |
| depth_anything | depth | ✓ | ✓ | exp | exp |  |  | exp | ✓ |
| depth_anything3 | depth |  |  |  |  |  |  | exp |  |
| dfine | detect | ✓ | ✓ | exp | exp |  |  | exp | ✓ |
| dfine | segment | ✓ | ✓ | exp | exp |  |  | exp |  |
| dinov2 | semantic | ✓ | ✓ | exp | exp |  |  | exp |  |
| dinov2 | classify | ✓ |  |  |  |  |  | exp | exp |
| ec | detect | ✓ | ✓ | exp | exp |  |  | exp | ✓ |
| ec | pose | ✓ | ✓ | exp | exp |  |  | exp |  |
| ec | segment | ✓ | ✓ | exp | exp |  |  | exp |  |
| edgetam | segment |  |  |  |  |  |  | exp |  |
| efficientnetv2 | classify | ✓ | ✓ | exp | exp | ✓ | ✓ | exp | ✓ |
| eomt | semantic | ✓ | ✓ | exp | exp |  |  | exp |  |
| eomt | segment |  |  |  |  |  |  | exp |  |
| eomt | panoptic |  |  |  |  |  |  | exp |  |
| florence2 | detect |  |  |  |  |  |  |  |  |
| fomo | point | ✓ | ✓ | exp | exp | ✓ |  | exp | ✓ |
| grounding_dino | detect |  |  |  |  |  |  | exp |  |
| internvl3 | detect |  |  |  |  |  |  |  |  |
| kosmos2 | detect |  |  |  |  |  |  |  |  |
| l2cs | gaze | ✓ |  |  |  |  |  | exp |  |
| lfm2vl | detect |  |  |  |  |  |  |  |  |
| lingbotvision | semantic | ✓ | ✓ | exp | exp |  |  | exp | ✓ |
| locateanything | detect |  |  |  |  |  |  |  |  |
| locateanything | point |  |  |  |  |  |  |  |  |
| mobilenetv4 | classify | ✓ | ✓ | exp | exp | ✓ | ✓ | exp | ✓ |
| mobilesam | segment |  |  |  |  |  |  | exp |  |
| nafnet | restore | ✓ | ✓ | exp | exp | ✓ |  | exp | ✓ |
| omdet_turbo | detect |  |  |  |  |  |  | exp |  |
| ov_deim | detect |  |  |  |  |  |  |  |  |
| owlv2 | detect |  |  |  |  |  |  | exp |  |
| picodet | detect | ✓ | ✓ | exp | exp | ✓ |  | exp | ✓ |
| picosam3 | segment | ✓ |  |  |  |  |  | exp |  |
| pidnet | semantic | ✓ | ✓ | exp | exp | ✓ | ✓ | exp | ✓ |
| ppocr | ocr |  |  |  |  |  |  | exp |  |
| qwen3vl | detect |  |  |  |  |  |  |  |  |
| realesrgan | restore | ✓ | ✓ | exp | exp | ✓ | ✓ | exp | ✓ |
| resnet | classify | ✓ | ✓ | exp | exp | ✓ | ✓ | exp | ✓ |
| rfdetr | detect | ✓ | ✓ | ✓ | ✓ |  | exp | exp | ✓ |
| rfdetr | segment | ✓ | ✓ | exp | exp |  |  | exp |  |
| rfdetr | pose | ✓ | ✓ | exp | exp |  |  | exp |  |
| rfdetr | obb | ✓ | ✓ | exp | exp |  |  | exp |  |
| rtdetr | detect | ✓ | ✓ | exp | exp |  |  | exp | ✓ |
| rtdetrv2 | detect | exp | ✓ | exp | exp |  |  | exp | ✓ |
| rtdetrv4 | detect | exp | ✓ | exp | exp |  |  | exp | ✓ |
| rtmdet | detect | ✓ | ✓ | exp | exp |  |  | exp | ✓ |
| rtmdet | segment |  |  |  |  |  |  | exp |  |
| sam | segment |  |  |  |  |  |  | exp |  |
| sam2 | segment |  |  |  |  |  |  | exp |  |
| sam3 | segment |  |  |  |  |  |  | exp |  |
| segformer | semantic |  |  |  |  |  |  | exp |  |
| sensenovavision | detect |  |  |  |  |  |  |  |  |
| sensenovavision | segment |  |  |  |  |  |  |  |  |
| sensenovavision | panoptic |  |  |  |  |  |  |  |  |
| sensenovavision | pose |  |  |  |  |  |  |  |  |
| sensenovavision | point |  |  |  |  |  |  |  |  |
| sensenovavision | depth |  |  |  |  |  |  |  |  |
| sensenovavision | ocr |  |  |  |  |  |  |  |  |
| siglip2 | classify | ✓ |  |  |  |  |  | exp | ✓ |
| smolvlm2 | detect |  |  |  |  |  |  |  |  |
| swinir | restore | exp | exp | exp | exp | exp |  | exp |  |
| yolo1 | detect | ✓ | ✓ | exp | exp | ✓ |  | exp | ✓ |
| yolo2 | detect | ✓ | ✓ | exp | exp | ✓ |  | exp | ✓ |
| yolo3 | detect | ✓ | ✓ | exp | exp | ✓ |  | exp | ✓ |
| yolo4 | detect | ✓ | ✓ | exp | exp | ✓ |  | exp | ✓ |
| yolo7 | detect | ✓ | ✓ | exp | exp | ✓ |  | exp | ✓ |
| yolo9 | detect | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | exp | ✓ |
| yolo9_e2e | detect | ✓ | ✓ | exp | exp | ✓ |  | exp | ✓ |
| yolo9_p2 | detect | ✓ | ✓ | exp | exp | ✓ |  | exp | ✓ |
| yolonas | detect | ✓ | ✓ | exp | exp | ✓ |  | exp | ✓ |
| yolonas | pose | ✓ | ✓ | exp | exp | ✓ |  | exp |  |
| yolox | detect | ✓ | ✓ | exp | exp | ✓ | ✓ | exp | ✓ |
| zipdepth | depth | ✓ | ✓ | exp | exp | ✓ |  | exp | ✓ |

## Core ML experimental profile

Every Core ML `exp` row is fixed-canvas, batch-one, and raw-output.
It records an implemented conversion path, not macOS runtime parity,
application preprocessing parity, task accuracy, or device performance.
No Core ML row is parity-validated yet.

## Parity thresholds

- Detection and OBB: matched box IoU above 0.95 and score MAE below 0.01.
- Segmentation and panoptic: mask IoU above 0.95.
- Pose: keypoint L2 below 2 pixels at native resolution.
- Classification: logits cosine above 0.999 and equal top-1 class.
- Depth and restoration: PSNR above 40 dB against native output.
- Point: peak locations equal within one output cell.

## Validated constraints

A check mark applies only under any constraint listed here.

- `birefnet` / `matte` / `torchscript`: fixed 1024x1024 input
- `clip` / `classify` / `onnx`: frozen-class labels and fixed input resolution
- `clip` / `classify` / `coreai`: frozen class set and fixed export canvas; permissively licensed trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `convnext` / `classify` / `coreai`: fixed export canvas; a representative published trained ImageNet checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `deim` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `deimv2` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `depth_anything` / `depth` / `coreai`: fixed export canvas; permissively licensed trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `dfine` / `detect` / `coreai`: fixed export canvas; trained LibreDFINEn weights are covered on macOS 27 by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `dinov2` / `semantic` / `onnx`: fixed 518x518 input
- `dinov2` / `semantic` / `torchscript`: fixed 518x518 input
- `dinov2` / `classify` / `onnx`: fixed 224x224 input
- `ec` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `ec` / `pose` / `onnx`: fixed 640x640 input
- `ec` / `pose` / `torchscript`: fixed 640x640 input
- `ec` / `segment` / `onnx`: fixed 640x640 input
- `ec` / `segment` / `torchscript`: fixed 640x640 input
- `efficientnetv2` / `classify` / `coreai`: fixed export canvas; a representative published trained ImageNet checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `eomt` / `semantic` / `onnx`: fixed 512x512 input
- `eomt` / `semantic` / `torchscript`: fixed 512x512 input
- `fomo` / `point` / `coreai`: native 96 canvas; a deterministic model state trained from scratch for eight steps on synthetic tensors is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; this validates conversion and the existing heatmap contract, not point-localization accuracy
- `l2cs` / `gaze` / `onnx`: head-only contract: each input image is one face crop
- `lingbotvision` / `semantic` / `onnx`: fixed 512x512 input
- `lingbotvision` / `semantic` / `torchscript`: fixed 512x512 input
- `lingbotvision` / `semantic` / `coreai`: fixed family-native canvases (PIDNet 1024, LingBotVision 512); trained LibrePIDNets-sem and LibreLingBotVisions-sem checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; exported backends already implement the shared dense-logit resize and argmax contract
- `mobilenetv4` / `classify` / `coreai`: fixed export canvas; a representative published trained ImageNet checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `nafnet` / `restore` / `onnx`: fixed-resolution export canvas
- `nafnet` / `restore` / `torchscript`: fixed-resolution export canvas
- `nafnet` / `restore` / `ncnn`: fixed-resolution export canvas
- `nafnet` / `restore` / `coreai`: fixed export canvas; permissively licensed trained restoration checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `picodet` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `picosam3` / `segment` / `onnx`: raw fixed-96 ROI contract: roi_image -> mask_logits
- `pidnet` / `semantic` / `coreai`: fixed family-native canvases (PIDNet 1024, LingBotVision 512); trained LibrePIDNets-sem and LibreLingBotVisions-sem checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; exported backends already implement the shared dense-logit resize and argmax contract
- `realesrgan` / `restore` / `onnx`: ONNX supports dynamic spatial input; TorchScript and NCNN are fixed-canvas
- `realesrgan` / `restore` / `torchscript`: ONNX supports dynamic spatial input; TorchScript and NCNN are fixed-canvas
- `realesrgan` / `restore` / `ncnn`: ONNX supports dynamic spatial input; TorchScript and NCNN are fixed-canvas
- `realesrgan` / `restore` / `tflite`: fixed-resolution export canvas
- `realesrgan` / `restore` / `coreai`: fixed export canvas; permissively licensed trained restoration checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `resnet` / `classify` / `coreai`: fixed export canvas; a representative published trained ImageNet checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `rfdetr` / `detect` / `coreai`: fixed export canvas; trained LibreRFDETRn weights are covered on macOS 27 against the graph the exporter itself prepares, using direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin. Conversion needed _rebake_rfdetr_pos_embed in export/coreai.py: the backbone bakes its position embedding for its configured 384 canvas, so exporting at any other size left an antialiased bicubic in the graph and the converter has no lowering for aten._upsample_bicubic2d_aa. The rebake re-runs the model's OWN baking path for the actual canvas, so the interpolation happens eagerly, outside the graph, computing exactly what it computed before. NOTE the reference. This family is verified against the exporter's prepared graph, not against ONNX, and the difference is not a detail: at a 640 canvas the rfdetr ONNX artifact disagrees with that same prepared graph by 9.3e-01. Core AI's rebake preserves the antialiased resize the eager model performs, whereas the ONNX path disables antialiasing (the model checks torch.onnx.is_in_onnx_export). Which artifact is right is an ONNX question and is not settled here, but ONNX cannot be used as the reference for this family at a non-native canvas.
- `rfdetr` / `segment` / `onnx`: fixed task-native input resolution
- `rfdetr` / `segment` / `torchscript`: fixed task-native input resolution
- `rfdetr` / `pose` / `onnx`: fixed task-native input resolution
- `rfdetr` / `pose` / `torchscript`: fixed task-native input resolution
- `rfdetr` / `obb` / `onnx`: fixed task-native input resolution
- `rfdetr` / `obb` / `torchscript`: fixed task-native input resolution
- `rtdetr` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `rtdetrv2` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `rtdetrv4` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `rtmdet` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `siglip2` / `classify` / `onnx`: frozen-class labels and fixed input resolution
- `siglip2` / `classify` / `coreai`: frozen class set and fixed export canvas; permissively licensed trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `yolo1` / `detect` / `ncnn`: fixed 448x448 input
- `yolo1` / `detect` / `coreai`: fixed family-native canvases (YOLO1 448, YOLO2 608, YOLO3 416, YOLO4 608); representative published trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; Core AI graph preparation exactly folds Darknet inference batch normalization into the preceding convolutions because Core AI 0.4.1 does not preserve Darknet's epsilon-after-square-root formula
- `yolo2` / `detect` / `coreai`: fixed family-native canvases (YOLO1 448, YOLO2 608, YOLO3 416, YOLO4 608); representative published trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; Core AI graph preparation exactly folds Darknet inference batch normalization into the preceding convolutions because Core AI 0.4.1 does not preserve Darknet's epsilon-after-square-root formula
- `yolo3` / `detect` / `coreai`: fixed family-native canvases (YOLO1 448, YOLO2 608, YOLO3 416, YOLO4 608); representative published trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; Core AI graph preparation exactly folds Darknet inference batch normalization into the preceding convolutions because Core AI 0.4.1 does not preserve Darknet's epsilon-after-square-root formula
- `yolo4` / `detect` / `coreai`: fixed family-native canvases (YOLO1 448, YOLO2 608, YOLO3 416, YOLO4 608); representative published trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; Core AI graph preparation exactly folds Darknet inference batch normalization into the preceding convolutions because Core AI 0.4.1 does not preserve Darknet's epsilon-after-square-root formula
- `yolo7` / `detect` / `coreai`: fixed 640x640 export canvas; trained LibreYOLO7b weights are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; the export decoder uses direct arange grids because Core AI 0.4.1 mislowers the equivalent cumulative-sum expression
- `yolo9` / `detect` / `coreai`: fixed export canvas; trained LibreYOLO9t weights are covered on macOS 27 by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin
- `yolo9_e2e` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `yolo9_p2` / `detect` / `coreai`: fixed 640x640 export canvas; a deterministic YOLO9-P2-T model initialized from the SHA-256-pinned, permissively licensed trained LibreYOLO9t checkpoint is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; this validates conversion, not P2 task accuracy, and does not depend on the restricted VisDrone research-preview checkpoint
- `yolonas` / `detect` / `coreai`: fixed 96x96 export canvas with pre-shaped canonical RGB tensors; a deterministic, license-clean synthetic YOLO-NAS-S state is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; the state receives 12 native training steps and a 20x regression-head scale to make both exported outputs non-degenerate; this validates conversion, not detection accuracy, raw-image preprocessing, or native-640 behavior, and does not convert restricted official weights
- `yolox` / `detect` / `coreai`: fixed export canvas; a representative published trained checkpoint for each family is covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin; RT-DETRv2 permits one shared whole-query permutation across its box and logit outputs because DETR query rows are an unordered set
- `zipdepth` / `depth` / `onnx`: fixed-resolution export canvas
- `zipdepth` / `depth` / `torchscript`: fixed-resolution export canvas
- `zipdepth` / `depth` / `ncnn`: fixed-resolution export canvas
- `zipdepth` / `depth` / `coreai`: fixed export canvas; permissively licensed trained checkpoints are covered on Apple hardware by direct named-output parity with a 3e-04 tolerance and a 100x input-sensitivity margin

## Experimental constraints

Experimental rows may be narrower than the family-wide size surface.

- `birefnet` / `matte` / `coreml`: Fixed 1024x1024, batch one, raw matte logits. Stable Core ML Tools 9.0 predates the required lowering; export feature-detects the Apple converter implementation instead of trusting its version string. Published `l` weights are MIT; `t` artifacts remain local-user-only until their upstream weight provenance is explicit.
- `clip` / `classify` / `coreml`: CLIP and SigLIP2 freeze the current class set and require their native input resolution. SigLIP2 preserves the exported softmax or sigmoid classification activation.
- `convnext` / `classify` / `coreml`: CLIP and SigLIP2 freeze the current class set and require their native input resolution. SigLIP2 preserves the exported softmax or sigmoid classification activation.
- `deimv2` / `detect` / `coreml`: Only the permissive `atto`, `femto`, `pico`, and `n` variants are accepted. DINOv3-backed `s`, `m`, `l`, `x`, and unknown variants fail in preflight.
- `depth_anything3` / `depth` / `coreml`: Fixed 504x504, batch one. The graph emits positive relative depth and non-negative sky scores. LibreYOLO preserves the native sky-region gate, random-with-replacement sampling, 0.99 quantile, reciprocal, and final align_corners=True resize on the host. Non-square input uses the documented fixed-stretch depth approximation.
- `dinov2` / `classify` / `coreml`: CLIP and SigLIP2 freeze the current class set and require their native input resolution. SigLIP2 preserves the exported softmax or sigmoid classification activation.
- `edgetam` / `segment` / `coreml`: FP32 only. One fixed model-ready image encoder and six named prompt decoders cover points, boxes, points+boxes, and single/multimask modes. Point count is bounded by prompt_max_points (default 16); raw-image preprocessing, prompt transforms, query loops, and mask upscaling remain exact host operations. SAM3 is visual-prompt-only and converted artifacts are local-user-only under its custom license.
- `efficientnetv2` / `classify` / `coreml`: CLIP and SigLIP2 freeze the current class set and require their native input resolution. SigLIP2 preserves the exported softmax or sigmoid classification activation.
- `eomt` / `semantic` / `coreml`: Fixed square DINOv2 S/B/L component. The graph emits compact raw class-query logits and stride-4 mask logits. LibreYOLO preserves EoMT's exact shortest-edge split/stitch geometry for semantic and longest-edge top-left pad/query decoding for instance and panoptic tasks on the host. The functional attention-mask graph has real Core ML Tools 9 conversion evidence; macOS runtime parity is pending.
- `eomt` / `segment` / `coreml`: Fixed square DINOv2 S/B/L component. The graph emits compact raw class-query logits and stride-4 mask logits. LibreYOLO preserves EoMT's exact shortest-edge split/stitch geometry for semantic and longest-edge top-left pad/query decoding for instance and panoptic tasks on the host. The functional attention-mask graph has real Core ML Tools 9 conversion evidence; macOS runtime parity is pending.
- `eomt` / `panoptic` / `coreml`: Fixed square DINOv2 S/B/L component. The graph emits compact raw class-query logits and stride-4 mask logits. LibreYOLO preserves EoMT's exact shortest-edge split/stitch geometry for semantic and longest-edge top-left pad/query decoding for instance and panoptic tasks on the host. The functional attention-mask graph has real Core ML Tools 9 conversion evidence; macOS runtime parity is pending.
- `grounding_dino` / `detect` / `coreml`: Sizes `t` and `b` use a fixed 800x800 batch-one RGB stretch profile, 900 queries, and at most 256 BERT tokens. The exact pre-fusion BERT boundary, prompt tokens, masks, positions, and WordPiece ABI are frozen; changing classes requires re-export. Fixed stretch differs from the native keep-aspect image policy for non-square sources.
- `mobilenetv4` / `classify` / `coreml`: CLIP and SigLIP2 freeze the current class set and require their native input resolution. SigLIP2 preserves the exported softmax or sigmoid classification activation.
- `mobilesam` / `segment` / `coreml`: FP32 only. One fixed model-ready image encoder and six named prompt decoders cover points, boxes, points+boxes, and single/multimask modes. Point count is bounded by prompt_max_points (default 16); raw-image preprocessing, prompt transforms, query loops, and mask upscaling remain exact host operations. SAM3 is visual-prompt-only and converted artifacts are local-user-only under its custom license.
- `nafnet` / `restore` / `coreml`: The source image must exactly match the fixed export canvas. Re-export, crop, or tile instead of padding an arbitrary smaller image to the canvas.
- `omdet_turbo` / `detect` / `coreml`: The released `t` checkpoint uses a fixed 640x640 batch-one FP32 TensorType boundary. The current class vocabulary and task-language embeddings are frozen; changing classes requires re-export. Exact Torchvision-v2 uint8 bilinear-antialias stretch remains on the host; the graph emits 900 normalized boxes and per-class logits for exact top-900 and class-aware NMS decoding.
- `owlv2` / `detect` / `coreml`: Fixed native 960x960 (`b16`) or 1008x1008 (`l14`) FP32 TensorType input with exact pad-before-Gaussian-resize preprocessing. The text tower and tokenizer are intentionally absent at runtime, so changing classes requires re-export. Published mirror provenance must be completed before converted weights are treated as release artifacts.
- `picosam3` / `segment` / `coreml`: Raw fixed batch-one 96x96 ROI component. The host expands each box by 10%, crops and resizes the ROI, then places mask logits back into the source image; point/text/mask prompts remain unsupported.
- `ppocr` / `ocr` / `coreml`: FP32 only. The detector and recognizer use named bounded-flexible TensorType functions; DB contours, perspective crops, reading order, bucketing, and CTC decoding remain exact host operations. Export requires an explicit finite rec_max_width; overflow is an error, never silent truncation.
- `realesrgan` / `restore` / `coreml`: The source image must exactly match the fixed export canvas. Re-export, crop, or tile instead of padding an arbitrary smaller image to the canvas.
- `resnet` / `classify` / `coreml`: CLIP and SigLIP2 freeze the current class set and require their native input resolution. SigLIP2 preserves the exported softmax or sigmoid classification activation.
- `rtmdet` / `segment` / `coreml`: Fixed batch-one canvas divisible by 32. The graph emits three class maps, three box-distance maps, three 169-parameter dynamic-kernel maps, and one stride-8 mask feature map; LibreYOLO performs per-level top-k, class-aware NMS, dynamic mask decoding, and placement on the host.
- `sam` / `segment` / `coreml`: FP32 only. One fixed model-ready image encoder and six named prompt decoders cover points, boxes, points+boxes, and single/multimask modes. Point count is bounded by prompt_max_points (default 16); raw-image preprocessing, prompt transforms, query loops, and mask upscaling remain exact host operations. SAM3 is visual-prompt-only and converted artifacts are local-user-only under its custom license.
- `sam2` / `segment` / `coreml`: FP32 only. One fixed model-ready image encoder and six named prompt decoders cover points, boxes, points+boxes, and single/multimask modes. Point count is bounded by prompt_max_points (default 16); raw-image preprocessing, prompt transforms, query loops, and mask upscaling remain exact host operations. SAM3 is visual-prompt-only and converted artifacts are local-user-only under its custom license.
- `sam3` / `segment` / `coreml`: FP32 only. One fixed model-ready image encoder and six named prompt decoders cover points, boxes, points+boxes, and single/multimask modes. Point count is bounded by prompt_max_points (default 16); raw-image preprocessing, prompt transforms, query loops, and mask upscaling remain exact host operations. SAM3 is visual-prompt-only and converted artifacts are local-user-only under its custom license.
- `segformer` / `semantic` / `coreml`: All b0-b5 eval graphs pass fixed-canvas two-probe TorchScript trace parity. The architecture/source is permissive, but published ADE20K weights remain restricted to research or evaluation.
- `siglip2` / `classify` / `coreml`: CLIP and SigLIP2 freeze the current class set and require their native input resolution. SigLIP2 preserves the exported softmax or sigmoid classification activation.
- `swinir` / `restore` / `coreml`: Sizes `s`, `m`, and `l` are enabled at their native 64x64 canvas. Every full graph has bit-exact two-probe TorchScript parity and Core ML Tools 9 FP16 ML Program conversion evidence. The exact source canvas is required; non-native canvases and Apple runtime parity remain pending.
- `yolonas` / `detect` / `coreml`: Square fixed canvas with the native longest-side cap: 636-centered RGB padding for detect; 640 top-left placement, bottom/right padding, and BGR graph input for pose. Current evidence uses license-clean synthetic graphs, not restricted published weights.
- `yolonas` / `pose` / `coreml`: Square fixed canvas with the native longest-side cap: 636-centered RGB padding for detect; 640 top-left placement, bottom/right padding, and BGR graph input for pose. Current evidence uses license-clean synthetic graphs, not restricted published weights.

## Checkpoint and artifact gates

These notices are independent of technical export status. A family may
accept user-trained weights even when a published checkpoint is restricted.

- `birefnet` / `matte`: The `l` checkpoint is MIT. The `t` checkpoint is not rehosted because its upstream repository has no explicit license metadata or LICENSE file.
- `deimv2` / `detect`: DINOv3-backed variants carry Meta's custom, non-OSI DINOv3 terms. Do not treat conversion of the permissive HGNet variants as evidence for those variants.
- `depth_anything` / `depth`: The `s` checkpoint is Apache-2.0. Published `b`, `l`, and `g` checkpoints are CC-BY-NC-4.0 and are not redistributed by LibreYOLO.
- `internvl3` / `detect`: The published `-hf` weights carry the Qwen License rather than a permissive MIT, Apache-2.0, or BSD license.
- `l2cs` / `gaze`: Published gaze checkpoints are bound by the research/non-commercial Gaze360 dataset terms and are not bundled, mirrored, or auto-downloaded.
- `lfm2vl` / `detect`: Published checkpoints carry the non-permissive LFM Open License v1.0 with a revenue threshold.
- `locateanything` / `detect`: The published LocateAnything checkpoint is NVIDIA non-commercial.
- `locateanything` / `point`: The published LocateAnything checkpoint is NVIDIA non-commercial.
- `nafnet` / `restore`: Some published GoPro checkpoints have no explicit standalone weights license. Convert only checkpoints the user has the right to use.
- `ov_deim` / `detect`: Published detector weights are CC BY-NC 4.0, and the MobileCLIP text tower carries research-only model terms. These are not MIT artifacts.
- `sam3` / `segment`: SAM 3 access is gated by Meta's custom SAM License; LibreYOLO does not redistribute the checkpoint under its MIT license.
- `segformer` / `semantic`: The architecture port is Apache-2.0, but published ADE20K checkpoints are restricted to research or evaluation by NVIDIA's license.
- `sensenovavision` / `depth`: The published SenseNova-Vision checkpoint is CC BY-NC 4.0.
- `sensenovavision` / `detect`: The published SenseNova-Vision checkpoint is CC BY-NC 4.0.
- `sensenovavision` / `ocr`: The published SenseNova-Vision checkpoint is CC BY-NC 4.0.
- `sensenovavision` / `panoptic`: The published SenseNova-Vision checkpoint is CC BY-NC 4.0.
- `sensenovavision` / `point`: The published SenseNova-Vision checkpoint is CC BY-NC 4.0.
- `sensenovavision` / `pose`: The published SenseNova-Vision checkpoint is CC BY-NC 4.0.
- `sensenovavision` / `segment`: The published SenseNova-Vision checkpoint is CC BY-NC 4.0.
- `yolo9_p2` / `detect`: The VisDrone research-preview variant is CC BY-NC-SA. Permissive YOLO9 transfer weights may be used for conversion-only tests.
- `yolonas` / `detect`: Published pretrained weights may carry separate non-commercial terms and are not bundled. Synthetic or user-trained states remain separate.
- `yolonas` / `pose`: Published pretrained weights may carry separate non-commercial terms and are not bundled. Synthetic or user-trained states remain separate.

## Blocked combinations

- `birefnet` / `matte` / `ncnn`: BiRefNet's decoder requires torchvision deformable convolution, which PNNX/NCNN cannot lower to a runnable graph.
- `birefnet` / `matte` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `birefnet` / `matte` / `coreai`: The decoder needs torchvision deform_conv2d, which the Core AI converter cannot lower ('unable to handle call function op: deform_conv2d.default'). The same operator already blocks the NCNN path. An encoder-only contract is the realistic route, matching the seam the CUDA graph work used.
- `clip` / `classify` / `torchscript`: Frozen-class vision-language export is available only for ONNX and the Apple runtimes in v1.
- `clip` / `classify` / `tensorrt`: Frozen-class vision-language export is available only for ONNX and the Apple runtimes in v1.
- `clip` / `classify` / `openvino`: Frozen-class vision-language export is available only for ONNX and the Apple runtimes in v1.
- `clip` / `classify` / `ncnn`: Frozen-class vision-language export is available only for ONNX and the Apple runtimes in v1.
- `clip` / `classify` / `tflite`: Frozen-class vision-language export is available only for ONNX and the Apple runtimes in v1.
- `deim` / `detect` / `ncnn`: NCNN export is not supported for DEIM: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `deim` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `deimv2` / `detect` / `ncnn`: NCNN export is not supported for DEIMv2: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `deimv2` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `depth_anything` / `depth` / `ncnn`: PNNX 20260526 reports unsupported batch-index reshapes in the DINOv2 transformer graph; the produced NCNN artifact fails numeric parity.
- `depth_anything` / `depth` / `tflite`: onnx2tf 2.4.x converts the DINOv2 depth graph, but LiteRT rejects a generated FILL node because its dimensions are invalid.
- `depth_anything3` / `depth` / `onnx`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `torchscript`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `tensorrt`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `openvino`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `ncnn`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `tflite`: Depth Anything 3 currently rejects export for every format; its depth graph has not been added to the exported-runtime contract.
- `depth_anything3` / `depth` / `coreai`: The model raises NotImplementedError for every format: depth export is out of scope per ADR 0006, the depth task contract. Depth Anything V2 exports and validates at 5.2e-06, so this is specific to the V3 family and not a Core AI limitation.
- `dfine` / `detect` / `ncnn`: NCNN export is not supported for D-FINE: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `dfine` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `dfine` / `segment` / `ncnn`: NCNN export is not supported for D-FINE: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `dfine` / `segment` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `dfine` / `segment` / `coreai`: This family and task have not been validated for Core AI export.
- `dinov2` / `semantic` / `ncnn`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `dinov2` / `semantic` / `tflite`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `dinov2` / `semantic` / `coreai`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `dinov2` / `classify` / `torchscript`: LibreDINOv2 classify export is not wired for this runtime; use ONNX, Core AI, or experimental Core ML export.
- `dinov2` / `classify` / `tensorrt`: LibreDINOv2 classify export is not wired for this runtime; use ONNX, Core AI, or experimental Core ML export.
- `dinov2` / `classify` / `openvino`: LibreDINOv2 classify export is not wired for this runtime; use ONNX, Core AI, or experimental Core ML export.
- `dinov2` / `classify` / `ncnn`: LibreDINOv2 classify export is not wired for this runtime; use ONNX, Core AI, or experimental Core ML export.
- `dinov2` / `classify` / `tflite`: LibreDINOv2 classify export is not wired for this runtime; use ONNX, Core AI, or experimental Core ML export.
- `ec` / `detect` / `ncnn`: NCNN export is not supported for EC: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `ec` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `ec` / `pose` / `ncnn`: NCNN export is not supported for EC: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `ec` / `pose` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `ec` / `pose` / `coreai`: This family and task have not been validated for Core AI export.
- `ec` / `segment` / `ncnn`: NCNN export is not supported for EC: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `ec` / `segment` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `ec` / `segment` / `coreai`: This family and task have not been validated for Core AI export.
- `edgetam` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `edgetam` / `segment` / `coreai`: Promptable model export is out of scope for the v1 runtime contract.
- `eomt` / `semantic` / `ncnn`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `eomt` / `semantic` / `tflite`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `eomt` / `semantic` / `coreai`: torch.export refuses the graph: GuardOnDataDependentSymNode, 'Could not guard on data-dependent expression Eq(u0, 1)'. Something in the mask path reads a value off a tensor and branches on it, which becomes an unbacked symbol with no hint the tracer can resolve. This is a real capture failure, not a missing operator and not the task gate: it was measured with the gate open. Fixing it means finding the host read and making the shape static for a fixed export canvas, the same shape of fix as the rfdetr torch._assert.
- `eomt` / `segment` / `onnx`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `torchscript`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `tensorrt`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `openvino`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `ncnn`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `tflite`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `segment` / `coreai`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `onnx`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `torchscript`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `tensorrt`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `openvino`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `ncnn`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `tflite`: EoMT instance and panoptic export do not yet have runtime parsing.
- `eomt` / `panoptic` / `coreai`: EoMT instance and panoptic export do not yet have runtime parsing.
- `florence2` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `florence2` / `detect` / `coreml`: Generative vision-language inference requires tokenizer, prefill, decode, and state/cache runtime components rather than one image graph.
- `florence2` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `fomo` / `point` / `tflite`: onnx2tf 2.4.x produces an invalid depthwise-convolution graph for the static SAME-padded FOMO backbone on this toolchain.
- `grounding_dino` / `detect` / `onnx`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `torchscript`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `tensorrt`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `openvino`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `ncnn`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `tflite`: Open-vocabulary runtime export is out of scope for v1.
- `grounding_dino` / `detect` / `coreai`: Open-vocabulary runtime export is out of scope for v1.
- `internvl3` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `internvl3` / `detect` / `coreml`: Generative vision-language inference requires tokenizer, prefill, decode, and state/cache runtime components rather than one image graph.
- `internvl3` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `kosmos2` / `detect` / `coreml`: Generative vision-language inference requires tokenizer, prefill, decode, and state/cache runtime components rather than one image graph.
- `kosmos2` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `l2cs` / `gaze` / `torchscript`: The v1 L2CS gaze export contract supports ONNX only.
- `l2cs` / `gaze` / `tensorrt`: The v1 L2CS gaze export contract supports ONNX only.
- `l2cs` / `gaze` / `openvino`: The v1 L2CS gaze export contract supports ONNX only.
- `l2cs` / `gaze` / `ncnn`: The v1 L2CS gaze export contract supports ONNX only.
- `l2cs` / `gaze` / `tflite`: The v1 L2CS gaze export contract supports ONNX only.
- `l2cs` / `gaze` / `coreai`: The model itself refuses: 'LibreL2CS export to coreai is not implemented. The v1 gaze export contract supports ONNX only.' That is a model-side decision, unchanged by opening the support gate, so nothing about Core AI is being tested here. Wiring the gaze contract beyond ONNX comes first.
- `lfm2vl` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `lfm2vl` / `detect` / `coreml`: Generative vision-language inference requires tokenizer, prefill, decode, and state/cache runtime components rather than one image graph.
- `lfm2vl` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `lingbotvision` / `semantic` / `ncnn`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `lingbotvision` / `semantic` / `tflite`: The dense-logits runtime contract is implemented, but this transformer graph has not produced a parity-valid edge-runtime artifact.
- `locateanything` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `locateanything` / `detect` / `coreml`: LocateAnything is a generative multi-input pipeline and has no tokenizer, state/cache, or output-decoding Core ML component contract.
- `locateanything` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `onnx`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `torchscript`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `tensorrt`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `openvino`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `ncnn`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `tflite`: Generative VLM export is out of scope for v1.
- `locateanything` / `point` / `coreml`: LocateAnything is a generative multi-input pipeline and has no tokenizer, state/cache, or output-decoding Core ML component contract.
- `locateanything` / `point` / `coreai`: Generative VLM export is out of scope for v1.
- `mobilesam` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `mobilesam` / `segment` / `coreai`: Promptable model export is out of scope for the v1 runtime contract.
- `nafnet` / `restore` / `tflite`: onnx2tf 2.4.x converts the fixed-canvas graph, but LiteRT fails at invoke time because an internal input tensor lacks data.
- `omdet_turbo` / `detect` / `onnx`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `torchscript`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `tensorrt`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `openvino`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `ncnn`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `tflite`: Open-vocabulary runtime export is out of scope for v1.
- `omdet_turbo` / `detect` / `coreai`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `onnx`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `torchscript`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `tensorrt`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `openvino`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `ncnn`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `tflite`: Open-vocabulary runtime export is out of scope for v1.
- `ov_deim` / `detect` / `coreml`: Open-vocabulary detection needs a frozen-vocabulary or bounded text/image component contract before Core ML export can be enabled.
- `ov_deim` / `detect` / `coreai`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `onnx`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `torchscript`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `tensorrt`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `openvino`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `ncnn`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `tflite`: Open-vocabulary runtime export is out of scope for v1.
- `owlv2` / `detect` / `coreai`: Open-vocabulary runtime export is out of scope for v1.
- `picodet` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `picosam3` / `segment` / `torchscript`: PicoSAM3's raw ROI component is currently wired only for ONNX and Core ML.
- `picosam3` / `segment` / `tensorrt`: PicoSAM3's raw ROI component is currently wired only for ONNX and Core ML.
- `picosam3` / `segment` / `openvino`: PicoSAM3's raw ROI component is currently wired only for ONNX and Core ML.
- `picosam3` / `segment` / `ncnn`: PicoSAM3's raw ROI component is currently wired only for ONNX and Core ML.
- `picosam3` / `segment` / `tflite`: PicoSAM3's raw ROI component is currently wired only for ONNX and Core ML.
- `picosam3` / `segment` / `coreai`: PicoSAM3's raw ROI component is currently wired only for ONNX and Core ML.
- `ppocr` / `ocr` / `onnx`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `torchscript`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `tensorrt`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `openvino`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `ncnn`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `tflite`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `ppocr` / `ocr` / `coreai`: OCR uses two networks for detection and recognition with dynamic per-region cropping, so it does not fit the single-graph export contract.
- `qwen3vl` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `qwen3vl` / `detect` / `coreml`: Generative vision-language inference requires tokenizer, prefill, decode, and state/cache runtime components rather than one image graph.
- `qwen3vl` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `rfdetr` / `detect` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `segment` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `segment` / `tflite`: onnx2tf 2.4.x assigns an invalid NHWC layout to the segmentation-head Einsum (78 channels versus the required 256), so conversion fails.
- `rfdetr` / `segment` / `coreai`: This family and task have not been validated for Core AI export.
- `rfdetr` / `pose` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `pose` / `tflite`: RF-DETR pose-x TFLite conversion exceeded the CPU timebox and 8 GB working memory without producing an artifact on this toolchain.
- `rfdetr` / `pose` / `coreai`: This family and task have not been validated for Core AI export.
- `rfdetr` / `obb` / `ncnn`: NCNN export is not supported for RF-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rfdetr` / `obb` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `rfdetr` / `obb` / `coreai`: This family and task have not been validated for Core AI export.
- `rtdetr` / `detect` / `ncnn`: NCNN export is not supported for RT-DETR: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rtdetr` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `rtdetrv2` / `detect` / `ncnn`: NCNN export is not supported for RT-DETRv2: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rtdetrv2` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `rtdetrv4` / `detect` / `ncnn`: NCNN export is not supported for RT-DETRv4: the model requires decoder or sampling operations unavailable in NCNN. Use ONNX, OpenVINO, TorchScript, or TensorRT instead.
- `rtdetrv4` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `rtmdet` / `detect` / `ncnn`: PNNX 20260526 reports an unregistered nn.Conv2d layer and leaves the RTMDet NCNN graph without usable input blobs.
- `rtmdet` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `rtmdet` / `segment` / `onnx`: RTMDet-Ins dynamic-kernel mask decoding has no contract for this exported runtime. Use native PyTorch inference or the Core ML raw-output profile.
- `rtmdet` / `segment` / `torchscript`: RTMDet-Ins dynamic-kernel mask decoding has no contract for this exported runtime. Use native PyTorch inference or the Core ML raw-output profile.
- `rtmdet` / `segment` / `tensorrt`: RTMDet-Ins dynamic-kernel mask decoding has no contract for this exported runtime. Use native PyTorch inference or the Core ML raw-output profile.
- `rtmdet` / `segment` / `openvino`: RTMDet-Ins dynamic-kernel mask decoding has no contract for this exported runtime. Use native PyTorch inference or the Core ML raw-output profile.
- `rtmdet` / `segment` / `ncnn`: RTMDet-Ins dynamic-kernel mask decoding has no contract for this exported runtime. Use native PyTorch inference or the Core ML raw-output profile.
- `rtmdet` / `segment` / `tflite`: RTMDet-Ins dynamic-kernel mask decoding has no contract for this exported runtime. Use native PyTorch inference or the Core ML raw-output profile.
- `rtmdet` / `segment` / `coreai`: RTMDet-Ins dynamic-kernel mask decoding has no contract for this exported runtime. Use native PyTorch inference or the Core ML raw-output profile.
- `sam` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `sam` / `segment` / `coreai`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `sam2` / `segment` / `coreai`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `onnx`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `torchscript`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `tensorrt`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `openvino`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `ncnn`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `tflite`: Promptable model export is out of scope for the v1 runtime contract.
- `sam3` / `segment` / `coreai`: Promptable model export is out of scope for the v1 runtime contract.
- `segformer` / `semantic` / `onnx`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `torchscript`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `tensorrt`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `openvino`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `ncnn`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `tflite`: This family is not wired to the shared dense-logits and backend argmax semantic export contract.
- `segformer` / `semantic` / `coreai`: LibreSegformer implements no export path at all ('Export is not implemented for LibreSegformer yet'), so this is not a Core AI limitation. Note its weights are non-commercial regardless.
- `sensenovavision` / `detect` / `onnx`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `detect` / `torchscript`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `detect` / `tensorrt`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `detect` / `openvino`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `detect` / `ncnn`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `detect` / `tflite`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `detect` / `coreml`: SenseNova-Vision is a generative multimodal pipeline with text and diffusion/VAE outputs; it has no stateful multi-component Core ML contract.
- `sensenovavision` / `detect` / `coreai`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `segment` / `onnx`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `segment` / `torchscript`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `segment` / `tensorrt`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `segment` / `openvino`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `segment` / `ncnn`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `segment` / `tflite`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `segment` / `coreml`: SenseNova-Vision is a generative multimodal pipeline with text and diffusion/VAE outputs; it has no stateful multi-component Core ML contract.
- `sensenovavision` / `segment` / `coreai`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `panoptic` / `onnx`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `panoptic` / `torchscript`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `panoptic` / `tensorrt`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `panoptic` / `openvino`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `panoptic` / `ncnn`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `panoptic` / `tflite`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `panoptic` / `coreml`: SenseNova-Vision is a generative multimodal pipeline with text and diffusion/VAE outputs; it has no stateful multi-component Core ML contract.
- `sensenovavision` / `panoptic` / `coreai`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `pose` / `onnx`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `pose` / `torchscript`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `pose` / `tensorrt`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `pose` / `openvino`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `pose` / `ncnn`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `pose` / `tflite`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `pose` / `coreml`: SenseNova-Vision is a generative multimodal pipeline with text and diffusion/VAE outputs; it has no stateful multi-component Core ML contract.
- `sensenovavision` / `pose` / `coreai`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `point` / `onnx`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `point` / `torchscript`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `point` / `tensorrt`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `point` / `openvino`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `point` / `ncnn`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `point` / `tflite`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `point` / `coreml`: SenseNova-Vision is a generative multimodal pipeline with text and diffusion/VAE outputs; it has no stateful multi-component Core ML contract.
- `sensenovavision` / `point` / `coreai`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `depth` / `onnx`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `depth` / `torchscript`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `depth` / `tensorrt`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `depth` / `openvino`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `depth` / `ncnn`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `depth` / `tflite`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `depth` / `coreml`: SenseNova-Vision is a generative multimodal pipeline with text and diffusion/VAE outputs; it has no stateful multi-component Core ML contract.
- `sensenovavision` / `depth` / `coreai`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `ocr` / `onnx`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `ocr` / `torchscript`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `ocr` / `tensorrt`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `ocr` / `openvino`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `ocr` / `ncnn`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `ocr` / `tflite`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `sensenovavision` / `ocr` / `coreml`: SenseNova-Vision is a generative multimodal pipeline with text and diffusion/VAE outputs; it has no stateful multi-component Core ML contract.
- `sensenovavision` / `ocr` / `coreai`: Generative multimodal export needs tokenizer, state/cache, and diffusion/VAE component contracts; it is out of scope for v1.
- `siglip2` / `classify` / `torchscript`: Frozen-class vision-language export is available only for ONNX and the Apple runtimes in v1.
- `siglip2` / `classify` / `tensorrt`: Frozen-class vision-language export is available only for ONNX and the Apple runtimes in v1.
- `siglip2` / `classify` / `openvino`: Frozen-class vision-language export is available only for ONNX and the Apple runtimes in v1.
- `siglip2` / `classify` / `ncnn`: Frozen-class vision-language export is available only for ONNX and the Apple runtimes in v1.
- `siglip2` / `classify` / `tflite`: Frozen-class vision-language export is available only for ONNX and the Apple runtimes in v1.
- `smolvlm2` / `detect` / `onnx`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `torchscript`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `tensorrt`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `openvino`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `ncnn`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `tflite`: Generative VLM export is out of scope for v1.
- `smolvlm2` / `detect` / `coreml`: Generative vision-language inference requires tokenizer, prefill, decode, and state/cache runtime components rather than one image graph.
- `smolvlm2` / `detect` / `coreai`: Generative VLM export is out of scope for v1.
- `swinir` / `restore` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `swinir` / `restore` / `coreai`: The export process DIES rather than hangs, and the kill point moves between runs, which is the signature of memory exhaustion rather than a stuck loop. One run reached 'Step 3/3: Optimizing and writing the asset' before stopping; a later run of the same graph at the same 128 canvas died inside to_coreai() before returning, in both cases with a leaked-semaphore warning and no traceback. Window attention unrolls into a very large number of small ops, so the converter's peak memory is the prime suspect on a 16 GB machine. Next steps: watch RSS during conversion, try the smallest available size at a 64 canvas, and check the system log for a memory kill. Do NOT assume optimize() is at fault; an earlier note said so on the strength of a single run and the second run contradicted it.
- `yolo1` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolo2` / `detect` / `tflite`: onnx2tf 2.4.x leaves an unresolved ONNX_CONCAT custom operation; LiteRT cannot prepare the converted detector graph.
- `yolo3` / `detect` / `tflite`: onnx2tf 2.4.x leaves an unresolved ONNX_CONCAT custom operation; LiteRT cannot prepare the converted detector graph.
- `yolo4` / `detect` / `tflite`: onnx2tf 2.4.x produces an invalid CONV_2D channel layout for YOLO4; LiteRT fails while allocating tensors.
- `yolo7` / `detect` / `tflite`: The converted LiteRT graph changes decoded box coordinates beyond the detector parity tolerance.
- `yolo9_e2e` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolo9_p2` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolonas` / `detect` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolonas` / `pose` / `tflite`: This family and task have not been validated through the ONNX-to-TFLite path.
- `yolonas` / `pose` / `coreai`: This family and task have not been validated for Core AI export.
- `zipdepth` / `depth` / `tflite`: onnx2tf 2.4.x flatbuffer-direct conversion does not support the edge-mode Pad operation in ZipDepth's convex upsampler.
