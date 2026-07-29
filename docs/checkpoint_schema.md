# LibreYOLO Checkpoint Metadata Schema

LibreYOLO `.pt` files are checkpoint wrapper dictionaries saved with
`torch.save()`. The top-level `model` key stores the PyTorch `state_dict`; the
other required top-level keys are metadata used to identify and load the
checkpoint without filename parsing or state-dict sniffing.

## Schema v1.0

Every official LibreYOLO `.pt` checkpoint must contain:

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

Required field meanings:

- `model`: PyTorch state dict for the model weights.
- `schema_version`: metadata contract version. v1.0 uses the string `"1.0"`.
- `libreyolo_version`: LibreYOLO version that produced the checkpoint.
- `model_family`: registered LibreYOLO family, such as `yolo9`, `rfdetr`,
  `dfine`, or `ec`.
- `size`: model variant within the family, such as `t`, `s`, `r18`, or `atto`.
- `task`: canonical task, one of `detect`, `segment`, `semantic`, `panoptic`,
  `pose`, `classify`, `gaze`, `obb`, `point`, `depth`, `normal`, `restore`,
  `matte`, `ocr`, `embed`, or `mesh`.
- `nc`: positive integer class count.
- `names`: `dict[int, str]` with keys in `0..nc-1`. Official checkpoints
  should write every key. Readers may pad missing keys with `class_i` labels for
  legacy sparse mappings, but out-of-range keys are invalid.
- `imgsz`: positive integer square input resolution. Checkpoints trained with
  a rectangular input size (supported detection families only) keep a scalar
  here for legacy readers, set to `max(imgsz_h, imgsz_w)`, and additionally
  dual-write `imgsz_h` and `imgsz_w` with the real dimensions, mirroring the
  export-runtime convention below. Readers that understand the rectangular
  fields must prefer them over the scalar; loading a checkpoint does not set
  the inference input size in either case (pass `imgsz` explicitly at predict
  or validation time, same as for non-default square sizes).

Pose checkpoints additionally include:

- `nc` / `names`: pose is usually single-class (`nc: 1`, `person`), but the
  YOLO-NAS pose head also supports multi-class pose with a single shared
  keypoint skeleton (one `kpt_shape` for every class); `nc` and `names` then
  describe the classes as in detection. Runtime pose exports emit `scores` with
  shape `[batch, anchors, nc]`.
- `num_keypoints`: positive integer keypoint count used by the pose head.
- `keypoint_dim`: pose label dimension from the dataset contract, either `2`
  for `x,y` labels or `3` for `x,y,visibility` labels. Model outputs always
  expose keypoints as `x,y,visibility`.
- `oks_sigmas`: optional list of per-keypoint OKS sigmas. When omitted, loaders
  and validators use the task default for `num_keypoints`.
- `num_keypoints_per_class`: optional list of per-class keypoint counts for
  GroupPose-style heads whose exported keypoint tensor is padded by class. Use
  `0` for classes without keypoints. Runtime backends use this schema to select
  the active keypoints for the predicted class.

Mesh checkpoints use the task string `mesh`, `nc: 1`, and
`names: {0: "person"}`. Because parameter layouts differ between body models,
the dimensions are recorded rather than assumed, the same way pose records
`num_keypoints`:

- `body_model`: the parameterization the checkpoint predicts into, such as
  `mhr`. Required; consumers use it to interpret every field below and to pick
  a body-model decoder.
- `num_betas`: identity/shape coefficient count (45 for MHR).
- `num_body_pose`: width of the body-pose parameter block (130 for MHR). This
  is a flat parameter vector, not one triplet per joint, because rig joints
  carry different degrees of freedom.
- `num_vertices` / `num_joints`: geometry sizes the body-model decoder emits
  (18439 and 127 for MHR), recorded so a payload can be validated before the
  decoder is loaded.
- `rotation_format`: how rotations are encoded, such as `euler_zyx` for MHR or
  `axis_angle`. Never inferred from the tensor shape, since a 3-vector is
  ambiguous between the two.

Depth checkpoints use the task string `depth`, `nc: 1`, and
`names: {0: "depth"}`. The single class-like slot exists only for checkpoint
schema compatibility; depth predictions are dense float maps, not classes.

Restore checkpoints use the task string `restore`, `nc: 1`, and
`names: {0: "image"}`. The single class-like slot exists only for checkpoint
schema compatibility; restoration predictions are dense RGB images, not
classes. Restoration checkpoints may also include:

- `degradation`: optional short label for the corruption type, such as
  `deblur`, `denoise`, or `super-resolution`.
- `dataset`: optional dataset/provenance label, such as `GoPro` or `SIDD`.
- `scale`: optional positive integer output-to-input upscale factor for
  super-resolution checkpoints (for example `4` for Real-ESRGAN x4). Absent or
  `1` means the restored image keeps the input resolution (deblur/denoise). The
  runtime also derives this from the model family and size, so the field is
  provenance metadata rather than a load-time requirement.

OCR checkpoints use the task string `ocr`, `nc: 1`, and `names: {0: "text"}`.
The single class-like slot exists only for checkpoint schema compatibility;
OCR predictions are text quads with transcripts, not classes. The `ppocr`
family ships one composite checkpoint per tier whose `model` state dict holds
two submodels under the `det.*` (DB text detector) and `rec.*` (CTC text
recognizer) key namespaces. OCR checkpoints additionally include:

- `charset`: list of strings, the full CTC alphabet in output-index order
  (index 0 is the CTC blank, then the recognition dictionary, then the space
  character). Embedding it makes the `.pt` self-contained; loaders must read
  the charset from the checkpoint, never from a side file.
- `pipeline`: dict of pipeline defaults baked at conversion time
  (`det_limit_side_len`, `det_db_thresh`, `det_db_box_thresh`,
  `det_db_unclip_ratio`, `rec_image_shape`). Runtime arguments may override
  them per call.
- `components`: reserved dict for optional pipeline stages (document
  orientation classification, image unwarping, textline 0/180 rotation).
  Empty in v1; adding a component later must not break this schema.

The schema is intentionally flat. Existing LibreYOLO checkpoints and loaders
already use top-level keys such as `model_family`, `size`, `nc`, `names`, and
`task`; nesting the metadata would increase migration risk before release.
The top-level `model` value is deliberately a `state_dict`, matching existing
LibreYOLO behavior. Other checkpoint formats may differ.

## Export Runtime Metadata

The checkpoint schema above remains square-only. Exported runtime artifacts may
also carry metadata for graph tracing and backend loading. For rectangular
graph exports, exporters may dual-write `imgsz_h` and `imgsz_w` next to the
legacy scalar `imgsz`; readers that do not understand the rectangular fields
must not silently treat the scalar as a square runtime contract.

Backend support for rectangular runtime metadata is family- and format-scoped.
YOLO9-family and NAFNet exports may use non-square `imgsz_h/imgsz_w` in
supported runtime formats; families or formats without explicit rectangular
support must reject the metadata instead of preprocessing those artifacts as
square inputs.

NAFNet restore runtime exports use a fixed-resolution v1 contract. ONNX exports
emit one dense `restored` output tensor and force `dynamic=false`; backend
prediction pads images that fit inside the exported canvas without resizing,
then crops the restored RGB result back to the original image shape. Dynamic
spatial restore export and tiled exported-runtime inference are deferred for
NAFNet.

Real-ESRGAN restore exports support dynamic spatial dims: the generators are
fully convolutional, so ONNX exports may set dynamic `height`/`width` axes on
both `images` and `restored`. Backend prediction runs at the native image
resolution (reflect-padded only to the network divisibility factor) and crops
the restored output to `scale` times the original image shape. The backend
derives `scale` from the model family and size (`x4`/`x4t` = 4, `x2` = 2).

SwinIR restore exports use a fixed-resolution v1 contract. ONNX exports emit
one dense `restored` tensor and force `dynamic=false` because shifted-window
attention masks are trace-shape-dependent. Backend prediction pads images that
fit inside the exported canvas, crops the output to four times the original
image shape, and reports `restore_scale = 4` for sizes `s`, `m`, and `l`.

Embedded-NMS runtime exports may also write these flat metadata keys:

- `nms`: string boolean. `"true"` means the exported graph includes an
  embedded post-processing output.
- `nms_conf`: confidence threshold baked into the embedded NMS graph output.
- `nms_iou`: IoU threshold baked into the embedded NMS graph output.
- `max_det`: maximum number of post-NMS detection rows emitted by the embedded
  graph output.
- `nms_raw_output`: string boolean. `"true"` means the exported graph also
  exposes an auxiliary raw detector output for LibreYOLO backend parsing.

Pose runtime exports may also write these flat metadata keys:

- `num_keypoints`: positive integer keypoint count used by the exported pose
  head.
- `keypoint_dim`: pose output dimension. Common values are `2` for xy-only
  exports and `3` for xy+visibility. GroupPose-style raw runtime exports may
  use larger values, such as `8`, when the tensor includes precision or
  class-logit fields consumed by LibreYOLO postprocessing.
- `num_keypoints_per_class`: optional JSON-encoded list of per-class keypoint
  counts for GroupPose-style heads. Readers must preserve zero-keypoint class
  slots because they define the class-to-keypoint schema.

Classification runtime exports (MobileNetV4 / ConvNeXt / EfficientNetV2 /
ResNet) may also write these flat metadata keys so that exported-backend
preprocessing reproduces the native model's resize/crop and the logits stay
bit-identical:

- `crop_pct`: float center-crop ratio. The pre-crop resize target is
  `round(imgsz / crop_pct)`. Readers default to `0.875` when the key is absent.
- `interpolation`: resize filter, `"bilinear"` or `"bicubic"`. Readers default
  to `"bilinear"` when the key is absent.

For ONNX YOLO9 detection exports with `nms=true`, output `0` / `output` is the
standalone post-NMS tensor using the export-time `nms_conf`, `nms_iou`, and
`max_det` values. When `nms_raw_output=true`, output `1` / `raw` is reserved for
LibreYOLO backends so they can apply native original-canvas clipping and runtime
`predict(conf=..., iou=..., max_det=...)` semantics. Third-party consumers that
want graph-embedded NMS should use the first output.

## Quantized Checkpoints

Quantized models add one optional flat key, `quant`: a small manifest dict
(`schema`, `recipe`, `keep_high_precision`, `execution`, calibration
provenance, `module_count`, `state`). Loaders that see `quant` rebuild the
quantized module structure before `load_state_dict`. See `quantization.md`.

`state` distinguishes the two artifact forms:

- `"prepared"` (default): fp32 master weights plus `_q_*` scale buffers.
  Trainable (QAT/QAD). Readers without quantization support may ignore the
  `quant` key and load the masters as a float model.
- `"finalized"`: crystallized deployment form written by
  `export(format="pt")`. Masters are stripped; per quantized module the
  state dict instead carries:
  - int8: `weight_packed` (int8, original weight shape) and `_q_w_scale`
    (fp32 per-channel). Dequant: `weight_packed * scale`. Activation range
    buffers (`_q_act_lo`/`_q_act_hi`/`_q_calibrated`) are retained.
  - fp8: `weight_packed` (float8_e4m3fn, original weight shape) and
    `_q_w_scale` (fp32 per-channel). Dequant: `weight_packed * scale`.
  - w4a16 / w4a8: `weight_packed` (uint8, two 4-bit codes per byte, low
    nibble first; code = q + 8) and `_q_w_gscale` (fp32, [out, ngroups],
    group 128 along in_features). int2: four 2-bit codes per byte
    (code = q + 2), group 64.
  - nvfp4: `weight_packed` (uint8, [out, ceil(in/16)*8], two 4-bit codes
    per byte, low nibble first; code = sign<<3 | E2M1 level index),
    `weight_block_scale` (float8_e4m3fn, [out, ceil(in/16)]), and
    `_q_w_amax` (fp32 per-tensor amax). Effective block scale:
    `block_scale * amax / (448 * 6)`.
  - mxfp4: `weight_packed` as nvfp4 but 32-element blocks, and
    `weight_block_exp` (int8, [out, ceil(in/32)]) storing the power-of-two
    block exponent. Effective block scale: `2 ** exponent`.
  The manifest records `remainder` (`"fp16"` or `"fp32"`) for the
  non-quantized tensors. Unpacking reproduces the fake-quant simulation bit
  for bit, so finalized inference matches prepared inference exactly on the
  finalizing device. This layout is the stable contract for external
  exporters and runtimes.

## Training Checkpoints

Trainer checkpoints use the same required metadata core and may also contain
flat training/resume fields:

```python
{
    "model": state_dict,
    "...": "all required v1.0 metadata",
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {...},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights` declares whether the top-level `model` is EMA-smoothed. When
EMA is enabled, `train_model`, `ema`, and `ema_updates` preserve resume state.
Published inference weights should be lean checkpoints and should not include
optimizer, epoch, config, loss, or EMA resume state unless intentionally
distributed as training checkpoints.

For release compatibility, readers accept legacy best-metric aliases such as
`best_mAP50_95`, `best_mAP50`, `best_metric`, and `best_metric_name`.

## Legacy And Foreign Weights

New LibreYOLO writers validate strictly and must emit v1.0 metadata.

When metadata is missing or incomplete:

- Legacy LibreYOLO-looking checkpoints load through the compatibility path with
  a warning and conversion instructions.
- Foreign upstream checkpoints are not loaded by `LibreYOLO(...)` as LibreYOLO
  checkpoints. Convert them with the appropriate `weights/convert_*.py` script
  before loading.

### RF-DETR COCO normalization

Upstream RF-DETR checkpoints expose a 91-output `class_embed` head
(`raw_nc = 90`, COCO's 90 classes + background). Auto-conversion normalizes a
*COCO* RF-DETR to LibreYOLO's COCO-80 convention (`nc = 80`, with the COCO
remap applied at post-process). A checkpoint is treated as COCO when it:

- carries exactly 80 names, **or**
- declares an explicit class count of 80 (`nc` / `args.num_classes`), **or**
- has a `coco` dataset hint, **or**
- has **no** class or dataset metadata at all — a bare upstream state-dict is
  the canonical Roboflow COCO-pretrained checkpoint (the only metadata-less
  91-output RF-DETR in distribution).

A genuine custom 90-class RF-DETR is preserved as `nc = 90`. It is identified
by a `names`/`class_names` list, an explicit non-80 class count, or a non-COCO
dataset hint (e.g. `args.dataset_file`), so the bare-checkpoint COCO fallback
does not fire for it. Empty placeholders (`""`, `{}`, `[]`) are ignored when
deciding whether a dataset hint is present.

Schema helpers live in `libreyolo/utils/serialization.py`:

```python
wrap_libreyolo_checkpoint(...)
unwrap_libreyolo_checkpoint(...)
validate_checkpoint_metadata(...)
```
