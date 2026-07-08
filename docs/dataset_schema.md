# Dataset Schema

This is the dataset-file contract for canonical tasks in `libreyolo/tasks.py`.

Clean-room rule: use public dataset-format docs and YAML examples only. Do not
use third-party source code, tests, or converters.

## Common YAML

Applies to `detect`, `segment`, `pose`, and `obb`.

- `path`: optional dataset root.
- `train`: required for training.
- `val`: required for validation.
- `test`: optional.
- `names`: required list or integer-keyed class mapping.
- `nc`: optional; must match `names` when present.
- `download`: optional; Python download scripts require explicit opt-in.
- `annotations`: optional mapping of split names to native COCO JSON files for
  detection, instance segmentation, and OBB.

`train`, `val`, and `test` may be image directories, image-list `.txt` files,
or lists of those values. Label paths follow:

```text
images/.../image.jpg -> labels/.../image.txt
```

For native COCO JSON detection/instance-segmentation/OBB datasets, `annotations`
maps a split to the JSON file and the split path gives the image root:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

When `names` is present, native COCO JSON category names must match the YAML
class names; those names define the model label IDs. Without `names`, COCO
category IDs are sorted and mapped densely to `0..N-1`.

Do not require `task` in dataset YAML. Explicit model/task selection wins.

Common label rules:

- one `.txt` label file per image;
- missing or empty label file means no objects;
- `class_id` is an integer in `0..nc-1`;
- coordinates are finite normalized floats in `[0, 1]`;
- coordinates are relative to original image width and height;
- rows contain no confidence or track id.

## detect

Canonical row, exactly 5 fields:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h` is a normalized axis-aligned box. `w` and `h` must be positive.

## segment

Polygon row:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N >= 3`. Coordinate count after `class_id` must be even. The polygon must be
non-degenerate.

A 5-field detection row is also accepted and represents a rectangular segment.

## semantic

Semantic segmentation pairs each image with a dense single-channel mask
(lossless format, typically PNG) instead of a `.txt` label file:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

Mask rules:

- single channel; palette-mode PNGs are read as palette indices;
- each pixel value is a class ID in `0..nc-1`;
- pixel value `255` means ignore and is excluded from loss and metrics;
- mask resolution must equal the paired image resolution.

YAML adds two optional keys on top of the common contract:

- `masks_dir`: mask directory name substituted for `images` in each image
  path (default `masks`).
- `label_mapping`: `{source_id: train_id}` remap applied to mask pixel
  values at load time; unmapped source values become ignore. Train IDs must
  fall in `0..nc-1`.

When `masks_dir` is omitted, masks are rasterized at load time from YOLO
`segment` polygon labels resolved through the standard
`images -> labels` convention, and a `background` class is appended after
the object classes (`nc` grows by one).

Canonical loader: `libreyolo.data.SemanticDataset`.

## panoptic

> SCAFFOLD (issue #555): the `panoptic` task is registered and its validator is
> wired, but the dataset loader below is **not implemented yet**. This section
> is the contract a contributor implements against; there is no canonical
> `PanopticDataset` on disk yet.

Panoptic segmentation pairs each image with a dense single-channel segment-id
map and a per-image list describing each segment. The intended contract follows
the COCO-panoptic format:

- A PNG per image whose pixel value (or RGB-encoded value) is a **segment id**.
  Every pixel belongs to exactly one segment; there is no overlap.
- A JSON `segments_info` list, one entry per segment id present in the image:
  `{"id": int, "category_id": int, "iscrowd": 0|1, "area": int}`. `category_id`
  indexes the dataset `names`; a per-category `isthing` flag distinguishes
  countable "things" from amorphous "stuff".

Validation uses Panoptic Quality (PQ = SQ x RQ), matching predicted to
ground-truth segments of the same category at IoU > 0.5. See
`libreyolo/validation/panoptic_validator.py` for the metric plug points.

Canonical loader: *(to be added — `libreyolo.data.PanopticDataset`)*.

## depth

Depth estimation pairs each image with a dense single-channel depth map instead
of a `.txt` label file:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

Depth rules:

- single channel PNG/TIF or `.npy`;
- map resolution must equal the paired image resolution;
- values are plain depth in a dataset-consistent unit;
- `0`, negative, NaN, and inf mark invalid pixels and are excluded from loss
  and metrics.

YAML adds two optional keys on top of the common contract:

- `depths_dir`: depth directory name substituted for `images` in each image
  path (default `depths`).
- `depth_stem_suffix`: optional suffix appended to the image stem before
  depth extension lookup. When omitted, both same-stem files and the common
  `_depth` suffix are tried.
- `depth_mask_suffix`: optional suffix appended to the resolved depth stem to
  find a validity mask (default `_mask`). If the mask exists, mask values
  `<= 0`, NaN, and inf invalidate the corresponding depth pixels.
- `depth_scale`: divisor for integer-typed depth maps (default `256.0`, the
  common 16-bit PNG convention where stored value / 256 is the depth).

Float `.npy` maps are used as-is and do not apply `depth_scale`.

Canonical loader: `libreyolo.data.DepthDataset`.

## restore

Image restoration pairs each degraded input image with a clean RGB target image
instead of a `.txt` label file:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

Restore rules:

- input and target images are RGB-compatible image files;
- input and target resolution must match exactly;
- validation keeps native resolution and pads only enough to stack a batch;
- metrics are computed on the original image canvas;
- training applies coupled crop and horizontal flip to the input/target pair.

YAML adds these optional keys on top of the common split contract:

- `input_dir`: degraded-input directory name used in split paths
  (default `inputs`).
- `target_dir`: clean-target directory name substituted for `input_dir`
  (default `targets`).
- `target_stem_suffix`: optional suffix appended to the input image stem before
  target extension lookup.
- `target_stem_suffixes`: list form of `target_stem_suffix`.
- `degradation`: optional metadata label such as `deblur` or `denoise`.
- `dataset`: optional dataset/provenance label such as `GoPro`.

The class-like YAML fields are schema placeholders: use `nc: 1` and
`names: {0: image}`. Restore models expose `Results.restored`, not detections.

Canonical loader: `libreyolo.data.RestoreDataset`.

## matte

Background removal / dichotomous segmentation pairs each RGB image with a
single-channel ground-truth alpha matte (0 = background, 255 = foreground)
sharing the same stem:

```text
images/subject.jpg -> mattes/subject.png
```

Two layouts are accepted:

- **Directory**: a root containing `images/` and a matte directory, auto-detected
  among `mattes/`, `matte/`, `gt/`, `masks/`, `mask/`, `alpha/`. Pass the root as
  `data=`.
- **YAML**: `path` (root), plus per-split `val_images` / `val_mattes` (and
  optional `train_images` / `train_mattes` for a future fine-tune), each a
  directory relative to `path` or absolute.

Matte rules:

- the matte is grayscale; values are read as alpha in `[0, 1]` (`/255`);
- a matte is resized to the prediction canvas with bilinear interpolation when
  the shapes differ;
- metrics are MAE and S-measure (Fan et al., ICCV 2017), computed on the
  original image canvas; best-checkpoint fitness is S-measure.

The class-like YAML fields are schema placeholders: use `nc: 1` and
`names: {0: matte}`. Matte models expose `Results.matte`, not detections.

Validation is inference-only in v1 (matte training/fine-tuning is a documented
follow-up). Canonical pair resolver: `libreyolo.data.matte_dataset.resolve_matte_pairs`.

## pose

YAML adds:

- `kpt_shape`: required, `[K, 2]` or `[K, 3]`;
- `flip_idx`: optional integer permutation of `0..K-1`.

Label row:

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Field count is exactly `5 + K * D`, where `D` is the second `kpt_shape` value.
Keypoint `x y` values are normalized. Visibility `v`, when present, is `0`,
`1`, or `2`.

## obb

Row, exactly 9 fields:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

The four points are normalized image coordinates in `[0, 1]` and form a
non-degenerate oriented rectangle. No angle is stored in the label file.

The canonical parser is strict by default and rejects out-of-range
coordinates. Dataset and validation ingestion may clip coordinates to `[0, 1]`
for otherwise valid crop-boundary labels, then still reject degenerate boxes.

Parsing is task-aware: 9 fields mean `obb` only in `obb` mode; in `segment`
mode they may be a 4-point polygon.

Canonical row parser: `libreyolo.data.parse_yolo_obb_label_line`.

Internal OBB geometry: parse normalized corners and convert them to canonical
`xywhr`. The angle is in radians and represents rotation of the width side
around the box center. Model families may adapt that canonical geometry to
their own training tensors, but public results should expose OBB detections as
`xywhr, conf, cls` rows.

YOLO9 OBB currently uses a family-private training adapter that stores targets
as `class, x1, y1, x2, y2, angle`, where `xyxy` is a horizontal proxy box for
assignment and DFL, and `angle` is trained with a separate periodic loss. Do
not treat that proxy tensor as the general OBB contract for other families.

Native COCO JSON OBB loading accepts annotations in this priority order:

- `obb: [x1, y1, x2, y2, x3, y3, x4, y4]` pixel-space corners;
- `obb: [cx, cy, w, h, angle]`, with `angle` in radians;
- COCO `segmentation` polygon/RLE, refit to a minimum-area rectangle;
- COCO `bbox: [x, y, w, h]`, interpreted as an axis-aligned rectangle and
  canonicalized to LibreYOLO `xywhr`.

Mosaic and mixup are disabled for OBB training until corner-aware OBB
augmentation is implemented.

## classify

Classification uses an ImageFolder-style directory tree, not label files:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/` is required for training and defines the class-to-index mapping by
sorted folder name. `val/` is required for validation. `test/` may be present
but is not used by the default train/val commands. Non-training splits must
contain the same class folder names as the expected train/checkpoint class set.
Supported image extensions are defined in
`libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`.

## gaze

No LibreYOLO training or validation dataset-file contract is implemented for
`gaze`.

## point

`point` is currently a model-output task, not a canonical dataset-label schema.
Point model families may adapt existing labels internally, for example by
deriving object centers from YOLO box rows, but a point-only text label format
is not defined in this document yet.
