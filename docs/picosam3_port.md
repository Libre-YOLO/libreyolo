# PicoSAM3 Native Port

LibrePicoSAM3 is a native ROI-conditioned segmentation family in the
`LibreSAM` tier. It uses only box prompts and runs a complete 1.37M-parameter
CNN once for each ROI.

## Provenance

- Code: `pbonazzi/picosam3`, commit
  `1b03949e43472953bb0021685c7fc3f5fdf48fde`, Apache-2.0.
- Weights served to users: `LibreYOLO/LibrePicoSAM3` (Apache-2.0), converted
  from upstream `pietrobonazzi/picosam3` revision
  `af49e4322b6b7cf448499fee5c073d4576f59444`, file
  `PicoSAM3_SAM3_student_best.pt`. Tensor values are unchanged; the mirror
  exists so autodownload does not depend on a third-party account.
- Teacher chain recorded by upstream: SAM 2.1 and SAM 3, used for
  distillation. Teacher implementations and checkpoints are not included.

The upstream `LICENSE_cctorch` is not included because this port does not use
or vendor cctorch code.

## Architecture and prompt contract

The encoder has four depthwise-separable stages, a dilated depthwise
bottleneck, additive skip connections, channel attention, and a depthwise
refinement head. It consumes a normalized RGB ROI at 96x96 and returns one
96x96 mask-logit map.

`predict(..., bboxes=[x1, y1, x2, y2])` expands each box by 10%, makes the crop
square, clips it to the image, resizes it to 96x96, and places the predicted
mask back into the original image. Multiple boxes are batched. Point, text,
mask, multimask, and segment-everything modes raise `ValueError`; use
LibreSAM2 or LibreSAM3 for those prompts. `set_image()` caches the image, not
an embedding, because PicoSAM3 has no split encoder/decoder path.

The model has no predicted-IoU head. `Results.boxes.conf` is therefore the mean
foreground pixel probability, a derived mask-certainty score rather than a
calibrated detection confidence.

## Checkpoint correction

At the pinned upstream revision, both `PicoSAM3_student_epoch1.pt` and
`PicoSAM3_epoch1.pt` contain the older 122-key PicoSAM2 architecture
(`output_head.*`). They cannot load PicoSAM3's dilated bottleneck, ECA, and
`refine.*` layers. The converter rejects them. The supported `best` checkpoint
has 141 state-dict entries and strictly loads the advertised architecture.
There is currently no matching supervised-baseline PicoSAM3 checkpoint.

## Parity

The pinned `best` state dict strictly loads into both implementations. On a
seeded batch of three FP32 96x96 inputs, the native port and upstream model
produce shape `(3, 1, 96, 96)` with maximum absolute difference `0.0`.
The raw ONNX graph matches PyTorch on the same three-input batch with maximum
absolute difference `2.27e-6` (within the `1e-5` parity tolerance).

## Measured mask quality

Evaluated on COCO val2017 with ground-truth boxes as ROI prompts and
ground-truth masks as reference, over 2000 randomly sampled non-crowd instances
(seed 0, no area filter):

| Protocol | mIoU |
|---|---|
| Crop space, 96x96 (upstream's evaluation space) | 0.692 |
| Full image, end-to-end via `predict()` | 0.697 |

The paper reports 65.45 mIoU on COCO. Our sampling and filtering may differ from
theirs, so this records the number we measured rather than restating theirs as
reproduced; it exceeds the published figure, which confirms the shipped `best`
checkpoint is the trained artifact it claims to be.

The epoch-1 files are a separate matter (see below) and are not shipped.

## Where it sits against MobileSAM

Same 150 COCO val2017 instances, same ground-truth box prompts, single-threaded
CPU, measured per prompt end-to-end:

| Model | Params | mIoU | CPU latency |
|---|---|---|---|
| PicoSAM3 | 1.37M | 0.691 | **8.6 ms/prompt** |
| MobileSAM | 10.13M | **0.800** | 407 ms/prompt |

MobileSAM is the better segmenter and should stay the default when a GPU is
available or when quality dominates. PicoSAM3 trades roughly 11 mIoU points for
7x fewer parameters and ~47x lower CPU latency, which is what makes CPU-only and
in-sensor use viable: 8.6 ms per prompt is interactive on a laptop CPU, 407 ms is
not. Its natural pairing is a LibreYOLO detector supplying boxes, turning `detect`
into instance segmentation on hardware that cannot host a ViT encoder.

## Conversion and export

`weights/convert_picosam3_weights.py` strictly validates the architecture and
writes checkpoint-schema v1 metadata including source revisions and license.

ONNX export exposes the honest raw contract:

```text
roi_image:  float32 [batch, 3, 96, 96]
mask_logits: float32 [batch, 1, 96, 96]
```

Box-to-ROI cropping and full-image mask placement remain host operations. For
IMX500 deployment, the sensor supplies the ROI crop before inference; `.rpk`
packaging remains a deployment step, not a LibreYOLO export format.
