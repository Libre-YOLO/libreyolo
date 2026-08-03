# Changelog

All notable changes to LibreYOLO are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases
before 1.4.0 are documented in the
[GitHub Releases](https://github.com/LibreYOLO/libreyolo/releases) only.

## [Unreleased]

### Added

- LibreHRNet W32 and W48, inference-only top-down COCO-17 pose models with
  fixed 256x192 and 384x288 person-crop canvases. Native heatmaps, affine crop
  geometry, flip testing, and decoding are exact against the pinned MIT
  upstream. Full-image inference composes a configurable person detector and
  also accepts explicit boxes or ready-made crops. Official weights and
  fixed-crop ONNX, TorchScript, TensorRT, and OpenVINO exports are validated;
  `train()` raises
- Opt-in `faster_coco_eval` flag (off by default) on `ValidationConfig`,
  `TrainConfig`, `model.val()` / `model.train()` kwargs, and the CLI
  (`--faster-coco-eval` on `libreyolo val` / `libreyolo train`). Routes
  bbox/segm COCO metrics through the faster-coco-eval C++ backend
  (10-50x faster on detection-dense datasets; validated bit-identical to
  pycocotools within float64 summation order across the 100 RF100-VL
  datasets). The `LIBREYOLO_FASTER_COCO_EVAL` env var overrides the flag
  in either direction, the backend actually used (name + version) is
  logged and exposed as `COCOEvaluator.last_backend` for provenance, and
  the evaluator falls back to pycocotools with a warning if the package
  is missing. Install via `pip install libreyolo[fast-eval]`.

- LibreViT, an inference-only classic Vision Transformer classifier in
  patch-16 tiny/small/base/large sizes at 224px. Native pretrained logits are
  bit-exact against the pinned Apache-2.0 timm AugReg checkpoints for all four
  sizes, with top-1/top-5 validation and ONNX Runtime prediction parity
- LibreRetinaNet, an inference-only torchvision RetinaNet port in ResNet-50
  FPN v1 and v2 variants (`r50`, `r50v2`). Both official COCO checkpoints
  have exact native head and detection parity against the pinned BSD-3-Clause
  source. Batch-one ONNX export supports dynamic aspect-preserved inputs with
  class-aware NMS in the unified backend; `train()` raises
- LibreDINODETR, an inference-only museum port of IDEA's Apache-2.0 DINO
  detector in the three released COCO variants (`r50`, `r50s5`, `swinl`).
  Native outputs are bit-exact against the pinned standalone source for every
  checkpoint, and fixed-canvas ONNX Runtime preserves raw and public prediction
  parity. Contrastive-denoising training is not implemented and `train()`
  raises `NotImplementedError`.
- LibreFCN, an inference-only semantic museum family in `r50` and `r101`
  sizes with the official 21-class COCO-trained, VOC-label heads. The shipped
  torchvision dilated-ResNet graph is not the original VGG FCN-8s topology;
  primary and auxiliary logits are bit-exact against pytorch/vision v0.26.0.
  ONNX, TorchScript, OpenVINO, and TensorRT prediction parity are validated,
  and the BSD-3-Clause code attribution and pretrained-weight caveat ship with
  the family (#637)
- LibreDeiT, an inference-only museum port of the plain DeiT patch-16
  ImageNet-1k classifiers in tiny, small, and base sizes at fixed 224px.
  Official Apache-2.0 timm checkpoints load with unchanged tensors and
  bit-exact upstream/native logits; ONNX Runtime, TorchScript, OpenVINO, and
  TensorRT FP16 export parity is covered. Distilled and 384px variants remain
  out of scope (#637)
- LibreCenterNet, an inference-only museum port of the official CenterNet
  `resdcn18` and `dla34` COCO detectors. Native preprocessing, raw heads, and
  top-100 no-NMS decoding match the pinned MIT source; legacy DCNv2 is replaced
  by torchvision deformable convolution. Fixed-512 ONNX and TorchScript exports
  use a portable deformable-convolution graph and return decoded detections
- LibreAlexNet, an inference-only museum port of torchvision's BSD-3-Clause
  AlexNet classifier. The canonical `LibreAlexNetb-cls.pt` checkpoint preserves
  the official ImageNet-1K tensors and produces bit-exact native logits;
  ONNX, TorchScript, OpenVINO, and TensorRT export parity is validated
- LibreVGG, an inference-only image-classification family with VGG-16,
  VGG-19, VGG-16-BN, and VGG-19-BN at fixed 224. Official torchvision
  ImageNet-1k V1 logits are bit-exact for every variant; ONNX, TorchScript,
  OpenVINO, and TensorRT backend parity is verified for VGG-16. The
  BSD-3-Clause code attribution and pretrained-weight caveat ship with the
  family, and `train()` raises `NotImplementedError`.
- LibreSwin, an inference-only Swin Transformer V1 image-classification family
  in Tiny, Small, Base, and Large sizes at 224px. All four released ImageNet-1k
  variants are bit-exact against the pinned timm reference, and trained
  prediction parity is verified for ONNX, TorchScript, OpenVINO, and TensorRT
- LibreMiDaS, an inference-only museum port of MiDaS v2.1 Small (`s`, 256)
  and DPT-Large (`l`, 384). Both native graphs are bit-exact against the
  pinned MIT upstream implementation and official checkpoints. Predictions
  are relative inverse depth with no metric unit; zero-shot depth validation
  and fixed-resolution ONNX, TorchScript, TensorRT, and OpenVINO export are
  supported. Official release assets are downloaded directly and
  checksum-verified rather than rehosted while ADR 0006 training-data
  clearance remains unresolved
- LibreEfficientDet, an inference-only EfficientDet D0-D4 museum family with
  fixed 512/640/768/896/1024 inputs, native EfficientNet backbones and weighted
  BiFPN, exact raw-output and decoded-candidate parity against the pinned
  Apache-2.0 `rwightman/efficientdet-pytorch` source, and validated ONNX,
  TorchScript, OpenVINO, and TensorRT prediction parity. The focal-loss and
  anchor-matching training recipe is not implemented and `train()` raises
  `NotImplementedError`.
- LibreDETR, an inference-only museum port of the original DETR (ECCV 2020)
  in all four released COCO variants (`r50`, `r50dc5`, `r101`, `r101dc5`).
  Native outputs are bit-exact against the pinned facebookresearch/detr
  Apache-2.0 source; TorchScript export is bit-exact and fixed-800 ONNX
  Runtime prediction parity is verified. DC5 dilation is stored explicitly in
  checkpoint metadata because it changes the runtime graph without changing
  any tensor shape, and `size` is required when constructing `LibreDETR`
  directly. The 500-epoch Hungarian-matching training recipe is not
  implemented and `train()` raises
- LibreFasterRCNN, an inference-only museum port of torchvision's Faster
  R-CNN in sizes n/s/m/l (MobileNetV3-Large 320-FPN, MobileNetV3-Large FPN,
  ResNet-50 FPN v1, ResNet-50 FPN v2). Native detections are exact against
  pytorch/vision v0.26.0; batch-one ONNX export keeps the upstream aspect
  resize and final class-wise NMS in-graph with dynamic source H/W. The
  BSD-3-Clause code attribution and the pretrained-weight caveat ship with
  the family, and `train()` raises
- LibreSSD300, an inference-only museum port of torchvision's fixed-resolution
  SSD300 VGG16 COCO detector. Native preprocessing, raw heads, default boxes,
  and final detections are exact against pytorch/vision v0.26.0. Batch-dynamic,
  fixed-300 ONNX export emits decoded predictions for LibreYOLO's shared
  class-aware NMS. The BSD-3-Clause source and implied checkpoint-license
  basis are disclosed alongside the Oxford VGG feature-weight CC BY 4.0
  lineage; `train()` raises
- LibreMaskRCNN, an inference-only Mask R-CNN R50 FPN v2 port with instance
  segmentation by default and detection from the same checkpoint. Native RPN,
  box, raw mask-logit, and final-mask outputs are exact against pinned
  torchvision v0.26.0. Batch-one ONNX for both tasks preserves dynamic source
  H/W. BSD-3-Clause attribution and the pretrained-weight caveat ship with the
  family, and `train()` raises
- LibreFCOS, an inference-only ResNet-50/FPN port of torchvision's FCOS. The
  official COCO checkpoint loads all 319 state entries strictly; raw heads,
  anchors, preprocessing, and native detections are exact against the pinned
  BSD-3-Clause source. ONNX and TorchScript have trained-checkpoint runtime
  parity, OpenVINO is experimental due to low-confidence NMS ordering drift,
  and the published `LibreFCOSr50.pt` mirror carries the explicit
  pretrained-weight license caveat
- LibreDeepLabv3, an inference-only semantic port of torchvision's three
  released COCO-with-VOC-label variants (`r50`, `r101`, `mv3`). Native
  21-class logits are bit-exact against pytorch/vision v0.26.0 before
  postprocessing; fixed-520 ONNX, TorchScript, OpenVINO, and TensorRT exports
  reload through the unified backend. Conversion removes only the upstream
  auxiliary FCN head, the BSD-3-Clause provenance and checkpoint-license
  caveat ship with the family, and `train()` raises
- LibreDeformableDETR, an inference-only museum port of the original
  Apache-2.0 Deformable DETR in all five released ResNet-50 variants
  (`r50ss`, `r50ssdc5`, `r50`, `r50refine`, `r50twostage`). The portable
  `grid_sample` attention path is bit-exact against upstream's pure-PyTorch
  reference, and all variants have fixed-800 ONNX Runtime prediction parity
- LibreLWDETR (LW-DETR), a detect-only family in sizes t/s/m/l/x at 640px:
  plain-ViT encoder with interleaved window/global attention, multi-scale
  projector, and a shallow deformable DETR decoder. Code and weights are
  Apache-2.0 (Atten4Vis/Baidu); ported outputs are bit-exact against the
  official implementation on all five released sizes. Inference-only — the
  Group-DETR one-to-many training recipe is not implemented and `train()`
  raises. LW-DETR is the architecture RF-DETR was forked from, so LibreYOLO
  now ships both the ancestor and its descendant
- Canonical `edge` and `normal` dense-prediction task contracts, including
  original-canvas result payloads and visualization, dataset schemas,
  validators (edge ODS/OIS and normal angular metrics), and public API aliases
- LibreTEED and LibreDexiNed edge specialists with native MIT-licensed
  architectures, local checkpoint converters, and fixed-resolution ONNX
  runtime parity; upstream BIPED-trained checkpoints are not bundled,
  mirrored, or auto-downloaded because the dataset terms are non-commercial
- LibreMODUS 14B-A7B analysis-only inference for depth, normals, edges,
  COCO detection, and phrase grounding, plus image-conditioned `any2any()`
  chaining and self-verification. The Apache-2.0 code port loads the upstream
  custom-term checkpoint directly or from a local directory, never mirrors it,
  and offers BF16 plus a local-only weight-only FP8 cache
- LibreFeyNobg, a new matte (background removal) family: FeyNobg by Feyn Inc., BiRefNet architecture with stage 3 deepened to 24 blocks (263M params), size l at fixed 1024px; code and weights Apache-2.0, converted from feyninc/FeyNobg; reuses the birefnet nn module with a family-local dimension table
- Quantization support for the birefnet and feynobg families (fp16/bf16/fp8/int8/w4a16/w4a8/nvfp4/mxfp4; int2 rejected since these families are inference-only and cannot heal); pre-quantized fp16 and fp8 LibreFeyNobg checkpoints published on the LibreYOLO Hugging Face org, loadable by passing the downloaded .pt as the weights argument (fp16 is GPU-oriented; bf16 is blocked by torchvision's missing BFloat16 deform_conv2d kernel; an nvfp4 variant was built, measured, and withdrawn: no kernel path beats fp16 on these GEMM shapes and 4-bit noise can flip foreground selection on ambiguous scenes)
- Native fp8 execution tier: finalized fp8 QuantLinear runs on the fp8 tensor cores via torch._scaled_mm (Ada/Hopper/Blackwell); optional Triton kernels fuse activation conversion and the per-channel scale/bias epilogue, while validation-selected FeyNobg Swin stage-0 Linears use manifest-recorded tensorwise weight scales for a fully fused cuBLASLt epilogue. Finalized fp8 QuantConv2d convolves in fp16 on cached dequantized weights, and fp16-remainder checkpoints get float32 I/O root hooks. On LibreFeyNobg/RTX 5070 Ti, fp8 is 123.1 vs 129.3 ms for batch-1 graphed predict and 515.4 vs 535.3 ms at batch 4, with a 275 vs 531 MB file.
- CUDA graph capture for the birefnet and feynobg families via encoder-only capture (the deformable decoder replays wrong under capture and stays eager; graphed output is bit-identical to eager); GraphRunner warms up on the capture stream so lazily-allocated cuBLASLt/cuDNN workspaces stop invalidating capture, and quant modules cache the calibration flag as a host bool (the per-forward .item() sync also invalidated capture)

### Fixed

- YOLOX BatchNorm eps=1e-3 / momentum=0.03 (official YOLOX values) is now
  applied by `LibreYOLOXModel` at construction instead of as a post-hoc fixup
  in the `LibreYOLOX` wrapper, so it survives the class-count rebuild
  (`_rebuild_for_new_classes`) that `train()` performs when the dataset `nc`
  differs from the checkpoint. Previously any such fine-tune trained and
  reported in-training validation at torch's default eps=1e-5 but was
  reloaded for inference at 1e-3 — same tensors, different normalization.
  Regular-conv sizes barely move, but depthwise `n` has per-channel
  running_var small enough for eps to dominate: on RF100-VL `ball` the same
  nano checkpoint scores 0.566 mAP50-95 evaluated at its trained eps and
  0.151 after a stock reload. Checkpoints trained before this fix carry
  eps=1e-5 semantics and must be evaluated with BN eps overridden to 1e-5
  (or have `sqrt((var+1e-3)/(var+1e-5))` folded into BN weights) to report
  faithful numbers
- CUDA graph capture no longer races with DataLoader pin-memory threads:
  training capture (`train(..., cuda_graph=True)`) and inference/validation
  capture now run with `capture_error_mode="thread_local"`, so a
  `cudaHostAlloc` from a pin-memory thread staging the next batch can no
  longer invalidate the capture and poison that thread (previously the run
  died with "AcceleratorError ... in pin memory thread" /
  `cudaErrorStreamCaptureUnsupported`; observed twice on an RF100-VL
  campaign with `pin_memory` dataloaders)
- D-FINE training now applies upstream's per-size multi-scale recipe instead
  of a hardcoded `base_size_repeat=3`: n trains at fixed size, s uses 20,
  m 6, l 4, x 3 (Peterande/D-FINE custom fine-tune configs; only X matched
  before). New `DFINEConfig.base_size_repeat` field overrides the per-size
  default when set (#675)

## [1.4.0] - 2026-07-24

LibreYOLO v1.4.0: 15 new model families, 3 new tasks (panoptic, matte, OCR), a quantization stack, two new trackers, and a multi-GPU training correctness overhaul.

### Added

- New model families:
  - LibreSegformer (SegFormer), semantic segmentation, sizes b0-b5 at 512px (b5 640px); code Apache-2.0, converted NVIDIA ADE20K weights non-commercial with a pre-download license notice (#589)
  - LibreSwinIR, x4 super-resolution (restore task), sizes s/m/l, Apache-2.0 code and weights (#571)
  - LibreRealESRGAN, super-resolution, sizes x4/x2 (RRDBNet) and x4t (compact SRVGG) (#549)
  - LibreBiRefNet, background removal with the new matte task, sizes t/l at 1024px; the t (lite) weights are not yet rehosted pending license confirmation (#549)
  - LibreZipDepth, depth, sizes b and bnpu (NPU-friendly decoder), 384px, MIT (#562)
  - LibreDepthAnything3, depth, size l at 504px, Apache-2.0; separate family from LibreDepthAnythingV2 (#577)
  - LibrePPOCR (PP-OCRv5), text detection + recognition with the new ocr task, sizes t/l at 960px, inference and validation only (#575, #587)
  - LibreSigLIP2, open-vocabulary zero-shot classification, sizes b16/so400m, native torch, inference only (#546)
  - LibreYOLO1, a YOLOv1 museum family, detect, sizes t/b, VOC 20 classes, fixed 448px; pretrained weights ship for b only (the tiny-yolov1 weights are lost upstream) (#549)
  - LibreSAM3 (SAM 3), promptable segmentation, size large at 1008px, transformers-backed; weights gated on Hugging Face under the Meta custom SAM License (#576)
  - LibreEdgeTAM, promptable segmentation, size edge at 1024px, image inference only, Apache-2.0 (#602)
  - LibrePicoSAM3, native 96px promptable ROI segmentation, ONNX-only export (#585)
  - LibreOMDetTurbo, open-vocabulary detection, size t, transformers-backed (#600)
  - LibreOVDEIM, open-vocabulary detection, sizes s/m/l, native NMS-free port via LibreOpenVocab("ov-deim"); code Apache-2.0, weights CC BY-NC 4.0, licensing confirmed by the upstream author (#607)
  - LibreSenseNovaVision (experimental), 7B unified multimodal checkpoint serving 7 tasks; weights CC BY-NC 4.0 non-commercial; not yet in __all__, the model inventory, CLI or UI (#618)
- Three new tasks: panoptic, matte, ocr, with result types PanopticSegmentation, Matte, OCRRegions and validators PanopticValidator (+ PanopticQuality), MatteValidator, OCRValidator (#557, #560, #549, #575)
- EoMT instance segmentation and panoptic: sizes s/b/l, 640px, new 1280 weight variant, panoptic checkpoints for s/b/l (#553, #557, #560)
- RTMDet-Ins instance segmentation, inference and validation, sizes t/s/m/l/x (training not implemented) (#572)
- D-FINE segmentation (experimental) with published seg weights and automatic detect-to-segment transfer in CLI train (#537)
- NAFNet SIDD denoise weight variant (LibreNAFNetl-restore-sidd, l size only) (#549)
- Quantization subsystem: libreyolo quantize CLI and model.quantize()/quant_info()/dequantize()/save(); recipes fp16/bf16/fp8/int8/w4a16/w4a8/nvfp4/mxfp4/int2 (research); QAT/QAD via train() on quantized checkpoints; supported families yolo9 and rfdetr; in-tree Triton kernels with a pluggable registry and LIBREYOLO_QUANT_KERNELS override (#619, #623)
- BoT-SORT tracker (model.track(tracker="botsort"), BoTSortTracker/BoTSortConfig exported top-level) (#621)
- Deep OC-SORT ReID tracker with an OSNet-AIN embedder auto-downloaded from LibreYOLO/LibreReID-osnet; custom embedder callables supported (#580)
- YOLOv7 training (SimOTA loss); the family was inference-only in v1.3.1 (#538)
- LoRA fine-tuning extended to D-FINE, DEIM, DEIMv2, RT-DETR v1/v2/v4, EC and ConvNeXt; adapters merged on export (#622)
- DINOv2 foundation-teacher distillation (distill_model="dinov2", feat_mse loss, distill_normalize knob; yolo9 backbones) (#534)
- Test-time augmentation for semantic (PIDNet, SegFormer, EoMT, DINOv2) and panoptic (EoMT) segmentation (#601, #608)
- Multi-class keypoint training for YOLO-NAS pose (#530)
- Augmentations: classification auto_augment/erasing/mixup/cutmix, copy-paste for segmentation, perspective and flipud, rot90 for OBB, vflip+rot90 for restore, HSV jitter for semantic (#532)
- Declarative augmentation spec (libreyolo/data/augment/spec.py): a per-family used/mosaic-gated/ignored matrix for every TrainConfig augmentation knob, pinned to the real pipelines by tests; the CLI now warns for every family when an explicitly-set training parameter is ignored (previously RF-DETR only), and training warns when mixup_prob is set with mosaic_prob=0 in the mosaic-gated pipelines (#635)
- Spawn-path multi-GPU training for ResNet, ConvNeXt, EfficientNetV2, MobileNetV4 and NAFNet (#567)
- Canonical export-support matrix with validated/experimental/blocked tiers, docs page and ADR 0011 (#578, #587)
- TFLite inference backend (LibreYOLO("model.tflite") via ai-edge-litert, Python >= 3.12) (#587)
- "litert" export alias for tflite and libreyolo[litert] extra (#563)
- Semantic, depth and point export unblocked (PIDNet, FOMO, ZipDepth, Depth Anything V2 under a fixed-resolution batch-1 depth contract) (#562, #578, #587)
- CLI: enriched libreyolo models, libreyolo formats --family/--task, libreyolo info export_support, libreyolo predict --json ocr array (#578, #587, #575)
- UI support for gaze, panoptic and open-vocabulary models; non-downloadable models greyed out (#579)
- New optional extras: sensenova, siglip2, siglip2-convert, litert; timm added to sam and openvocab extras (#618, #546, #563, #602, #600)

### Changed

- D-FINE and RT-DETRv4 now evaluate and predict at sizes other than the native 640; v1.3.1 crashed at any non-native imgsz. Known residual: rectangular sizes with the same token count as the native size still reuse a wrong-aspect embedding. DEIM, DEIMv2, EC, RT-DETR and RT-DETRv2 get the same dynamic eval-size support via per-shape regenerated embeddings/anchors (#541, #630)
- PicoDet fine-tune defaults: lr0 0.1 -> 0.01, warmup_lr_start 0.01 -> 0.001; the old default destroyed COCO-pretrained weights (coco128 fine-tune 0.40 -> 0.14 before vs 0.40 -> 0.49 after) (#568)
- DEIM fine-tune defaults: lr0 4e-4 -> 1e-4, min_lr_ratio 0.5 -> 0.05; RT-DETRv4 inherits the min_lr_ratio change; pass the old values to reproduce the upstream COCO recipe (#622)
- DEIMv2-n flat_epochs 7800 -> 78 (iteration count misplaced as epochs; LR schedule shape changes) (#622)
- AdamW no longer applies weight decay to BatchNorm/bias parameter groups in the base trainer (#568)
- Semantic segmentation training applies HSV jitter by default (trained mIoU moves for PIDNet, DINOv2-semantic, RF-DETR-semantic) (#532)
- Restore training adds coupled vertical flip and rot90 (NAFNet training results move) (#532)
- SyncBatchNorm defaults on under multi-GPU DDP for YOLO9, YOLOX, YOLOv7, YOLO-NAS, PicoDet, RTMDet and FOMO (#531, #538, #567)
- DDP now shards correctly for DEIM, D-FINE and YOLO-NAS-pose (previously every rank trained the full dataset at the full batch), and loss normalizers are globally all-reduced to match single-GPU gradients (#605, #567)
- New DDP hard errors: non-divisible global batch, batch < 1 after AutoBatch, and non-sharding custom loaders all raise at setup (#605)
- model.train(profile=True) keeps training after the profiled window; profile_then_stop=True restores the old stop behavior (#590)
- Semantic and panoptic val/predict accept augment=True (previously raised) (#601, #608)
- YOLO-NAS multi-class pose checkpoints load with their real class count and return real class ids (previously forced to single-class person) (#530)
- Export gated by the support matrix: blocked combos raise up front, experimental combos warn (#578, #587)
- RF-DETR imgsz validated early in predict/val/export with suggested valid sizes (#551)
- EC training config augmentation defaults zeroed to match the trainer's actual pass-through path (executed training unchanged) (#551)
- libreyolo models --json schema changed (task-suffixed cli_names, new keys); libreyolo formats/info JSON gained keys (#578)
- Checkpoints using the new task strings or finalized quant state are not loadable by v1.3.1 (#619, #575, #557)

### Removed

- libreyolo/models/omdet_turbo/ native graph (unreachable dead code at v1.3.1); replaced by the transformers-based LibreOMDetTurbo (#600)
- broadcast_ema_buffers internal helper (unused) (#567)
- No public API deprecations were added or removed; pre-existing deprecated aliases still warn

### Fixed

- libreyolo[openvocab] now installs ftfy and regex, required by OV-DEIM's CLIP prompt tokenizer at predict time; a clean openvocab-only install previously failed on the first prediction (#636)
- Results and LibreEoMT keep full v1.3 positional-argument compatibility: new v1.4 parameters (panoptic/matte/ocr/restore_scale; num_queries) moved after the complete v1.3 signatures, with compatibility tests (#636)
- Export never mutates the live model before the request is accepted: LoRA adapters are folded (and finalized int8 models re-prepared) only after format lookup, option preflight, and parameter resolution; quantized format='pt' export folds adapters on the checkpoint copy, leaving the live model trainable (#636)
- RTMDet fine-tune collapse from missing head init (~196,000x loss shock on re-heading; nc=1 rebuild 0.26 -> 0.709 mAP50-95) (#568)
- PicoDet/RTMDet AMP training crashes/NaN on CUDA (fp32 loss under AMP, SimOTA BCE outside autocast) (#568, #551)
- Pose validation under DDP: per-rank file clobbering and collective deadlock (#605)
- YOLO9 multi-GPU convergence degradation from per-rank BatchNorm stats (#531)
- DDP loss under-scaling on sparse batches for PicoDet/RTMDet (#567)
- Segmentation training RAM exhaustion on COCO-scale datasets with multiple workers (uint8 variable-length masks) (#529)
- profile=True silently truncating runs and corrupting resume state; stale stop flag on trainer reuse (#590)
- Atomic weight downloads (.part staging, Content-Length verification) (#587)
- Gaze face detection on OpenCV 5 (YuNet detector added) (#587)
- Depth models crashing on video input (#562)
- libreyolo models advertising names the factory refused to load (#600)
- OMDet-Turbo ignoring iou= (#600)
- RF-DETR loud warning on class-count head reinit; rfdetr accepted by the distillation config (#628)
- Security: NAFNet arch peek switched to torch.load(weights_only=True) (#550, #559)
- YOLOv1 decode guards survive python -O; batched predict loops per image (#550)
- D-FINE seg parity with ONNX/TensorRT; seg training auto-downloads published weights; tiled inference rejects segment models loudly (#537)
- YOLOv7 training color-space and fp16 overflow bugs (fine-tune mAP50 0.0 -> 0.92) (#538)
- YOLOX SimOTA crash when no anchor matches any ground truth (#538)
- SegFormer bit-exact reference parity, re-head init, hsv_prob actually applied (#589)
- Panoptic Quality crowd-region under-counting and bounds guards (#560)
- EoMT panoptic checkpoints validate instead of crashing val() (#553)
- Quantization: calibration device mismatch; NVFP4/MXFP4 scale buffers protected from fp16 cast (#619, #623)
- Deep OC-SORT respects the requested device (#587)
- Concurrent SwinIR tiling race (ContextVar) (#587)
- OV-DEIM device-switch crash on cached text features (#607)
- DINOv2 distillation teacher border cropping at non-14-multiple sizes (#534)
- OCR validator optimal one-to-one assignment (#575)
- Depth Anything 3 per-image sky quantile (#581)
- RTMDet-Ins box clamping (#581)
- Video writer sized from output frames; matte overlays on video (#549)
- SenseNova 16 GiB loading and task-state leaks (#618)
- SAM 3 text-prompt threshold semantics (#576)
- Classification square_resize + augment now raises; SigLIP2 fp32 softmax (#546)
- NCNN export on Windows tolerates the PNNX auxiliary-loader failure (#587)
- Export/inventory metadata corrections (AST-based classification, duplicate-entry errors, SwinIR weights published) (#587, #582, #581)

### Release stats

316 commits, 66 merged pull requests, 560 files changed, +97,139 / -3,130 lines, 58 new test modules. Contributors: datarocks0 (SegFormer fixes via #589), imagra93 (#553, #589, #601, #608), Xuban Ceccon (maintainer, 301 of 316 commits).
