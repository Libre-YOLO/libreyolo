# LibreYOLO Model Nomenclature

This document catalogs the model-naming conventions **currently in use** in
the LibreYOLO repository. It is descriptive — it records what is there today,
not a proposal. Sources of truth are the `FAMILY` and `FILENAME_PREFIX`
class constants in `libreyolo/models/<family>/model.py` and the
task-resolution rules in [`libreyolo/tasks.py`](../libreyolo/tasks.py).

## Filename schema

Every weight file follows:

```
Libre<FAMILY><size>[-<task>].pt
```

- `FAMILY` — family-specific prefix (see table below).
- `<size>` — single-letter or backbone-named size code. Always **lowercase**,
  attached directly to the family prefix with no separator.
- `<task>` — optional task suffix, hyphen-prefixed.
  Detect is **implicit** (no suffix), following the common YOLO naming convention.

## Family prefixes

The model families registered into the model factory (the VLM tier is a
separate category, covered in the note below). Most are detectors; `pidnet` is semantic-only; `eomt` supports semantic and
instance segmentation; the `mobilenetv4` / `convnext` /
`efficientnetv2` / `resnet` families are classify-only:

| Family id (`FAMILY`) | Filename prefix | Casing rule applied |
|---|---|---|
| `yolox`     | `LibreYOLOX`    | All-caps acronym |
| `yolo2`     | `LibreYOLO2`    | All-caps acronym + version digit (YOLOv2 / Darknet, public domain) — inference-only |
| `yolo3`     | `LibreYOLO3`    | All-caps acronym + version digit (YOLOv3 / Darknet, public domain) — inference-only |
| `yolo4`     | `LibreYOLO4`    | All-caps acronym + version digit (YOLOv4 / Darknet, public domain) — inference-only |
| `yolo7`     | `LibreYOLO7`    | All-caps acronym + version digit (YOLOv7 / MIT MultimediaTechLab/YOLO) — inference-only |
| `yolo9`     | `LibreYOLO9`    | All-caps acronym + version digit |
| `yolo9_e2e` | `LibreYOLO9E2E` | All-caps acronym + version + variant |
| `yolo9_p2`  | `LibreYOLO9P2`  | All-caps acronym + version + variant (stride-4 small-object) |
| `yolonas`   | `LibreYOLONAS`  | All-caps acronym (hyphen dropped from `YOLO-NAS`) |
| `dfine`     | `LibreDFINE`    | All-caps acronym (hyphen dropped from `D-FINE`) |
| `deim`      | `LibreDEIM`     | All-caps acronym |
| `deimv2`    | `LibreDEIMv2`   | All-caps acronym + lowercase version |
| `rtdetr`    | `LibreRTDETR`   | All-caps acronym (hyphen dropped from `RT-DETR`) |
| `rtdetrv2`  | `LibreRTDETRv2` | All-caps acronym + lowercase version |
| `rtdetrv4`  | `LibreRTDETRv4` | All-caps acronym + lowercase version |
| `rtmdet`    | `LibreRTMDet`   | Upstream brand casing preserved (`RTMDet`) |
| `rfdetr`    | `LibreRFDETR`   | All-caps acronym (hyphen dropped from `RF-DETR`) |
| `dinov2`    | `LibreDINOv2`   | All-caps acronym + lowercase version (DINOv2 backbone) |
| `eomt`      | `LibreEoMT`     | Mixed-case upstream brand preserved (`EoMT`) - semantic + instance segmentation transformer family |
| `pidnet`    | `LibrePIDNet`   | All-caps acronym + `Net` brand casing - semantic-only real-time family |
| `picodet`   | `LibrePICODET`  | All-caps (`PicoDet` rendered uppercase) |
| `ec`     | `LibreEC`    | Short form of EdgeCrafter — used as the family alias for the three sibling upstream models `ECDet`, `ECPose`, `ECSeg` |
| `l2cs`      | `LibreL2CS`     | All-caps acronym (`L2CS` gaze estimation) — inference-only |
| `fomo`      | `LibreFOMO`     | All-caps acronym (Faster Objects, More Objects) |
| `mobilenetv4` | `LibreMobileNetV4` | CamelCase preserved (MobileNet is not an acronym) — first classify-only family |
| `convnext`  | `LibreConvNeXt`  | CamelCase preserved (upstream brand casing `ConvNeXt`) — classify-only family |
| `efficientnetv2` | `LibreEfficientNetV2` | CamelCase preserved (EfficientNet is not an acronym) — classify-only accuracy tier |
| `resnet`    | `LibreResNet`    | CamelCase preserved (`ResNet` brand casing) — classify-only baseline |
| `clip`      | `LibreCLIP`     | All-caps acronym (`CLIP` zero-shot open-vocab classify) — inference-only |
| `nafnet`    | `LibreNAFNet`   | All-caps acronym + CamelCase `Net`; restore-only image-restoration family |
| `depth_anything` | `LibreDepthAnythingV2` | CamelCase preserved + version (Depth Anything V2), depth-only |

Casing rules observed in the table:

1. **Acronyms remain all-caps** (`YOLOX`, `YOLO9`, `YOLONAS`, `DFINE`, `DEIM`,
   `RTDETR`, `RFDETR`).
2. **Hyphens and dots from upstream branding are dropped**
   (`D-FINE` → `DFINE`, `RT-DETR` → `RTDETR`, `RF-DETR` → `RFDETR`,
   `YOLO-NAS` → `YOLONAS`).
3. **Version suffixes are lowercase** (`DEIMv2`, not `DEIMV2`).
4. **`ec` is a family alias, not a single model name.** The EdgeCrafter
   project ships three sibling upstream models — `ECDet`, `ECPose`, `ECSeg`
   — that share a backbone+encoder and differ only in the head. LibreYOLO
   collapses all three into one family (`FAMILY = "ec"`) with three task
   variants (`SUPPORTED_TASKS = ("detect", "pose", "segment")`); the
   filename prefix `LibreEC` is the short form of EdgeCrafter, with the
   task carried in the `-pose` / `-seg` suffix.

For these checkpoint-emitting detector families the casing rule is uniform:
**every family prefix is all-caps after `Libre`**, with the mixed-case
exceptions being lowercase version suffixes (`DEIMv2`, `RTDETRv2`,
`RTDETRv4`) and preserved upstream brand casing (`RTMDet`).

The VLM and promptable SAM tiers are separate categories and do not follow this
rule. Their weights-directory prefixes (`LibreQwen3VL`, `LibreLFM2VL`,
`LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`, `LibreKosmos2`,
`LocateAnything`, `LibreSAM`, `LibreSAM2`, `LibreMobileSAM`) are not registered
into the detector factory and do not emit `Libre<FAMILY><size>.pt` detector
checkpoints. Their `FILENAME_PREFIX` is only a weights-directory prefix for a
downloaded Hugging Face snapshot or promptable checkpoint, so upstream brand
casing (CamelCase) is intentionally preserved. See
[`librevlm_design.md`](librevlm_design.md) and
[`adr/0007-libresam-contract.md`](adr/0007-libresam-contract.md).

The open-vocabulary detector tier is also separate from the checkpoint factory.
Its weights-directory prefixes (`LibreGroundingDINO`, `LibreOWLv2`) identify
downloaded Hugging Face snapshots, not `Libre<FAMILY><size>.pt` checkpoints.
These models are discriminative text-conditioned detectors with calibrated
scores; they are not VLMs. Upstream brand casing is intentionally preserved.
See [`openvocab_design.md`](openvocab_design.md).

## Size codes

Sizes are family-specific. The table below records what each family currently
ships:

| Family | Size codes |
|---|---|
| `yolox`     | `n`, `t`, `s`, `m`, `l`, `x` |
| `yolo2`     | `t` (yolov2-tiny, 416), `b` (yolov2, 608) |
| `yolo3`     | `t` (yolov3-tiny, 416), `b` (yolov3, 416), `spp` (yolov3-spp, 608) |
| `yolo4`     | `t` (yolov4-tiny, 416), `b` (yolov4, 608) |
| `yolo7`     | `b` (yolov7, 640) |
| `yolo9`     | `t`, `s`, `m`, `c` |
| `yolo9_e2e` | `t`, `s`, `m`, `c` (inherited from yolo9) |
| `yolo9_p2`  | `t`, `s` |
| `yolonas`   | `s`, `m`, `l` |
| `dfine`     | `n`, `s`, `m`, `l`, `x` |
| `deim`      | `n`, `s`, `m`, `l`, `x` |
| `deimv2`    | per-cfg (see `SIZE_CONFIGS`) |
| `rtdetr`    | `r18`, `r34`, `r50`, `r50m`, `r101`, `l`, `x` |
| `rtdetrv2`  | `r18`, `r34`, `r50`, `r50m`, `r101` |
| `rtdetrv4`  | `s`, `m`, `l`, `x` |
| `rtmdet`    | `t`, `s`, `m`, `l`, `x` |
| `rfdetr`    | `n`, `s`, `m`, `l` |
| `dinov2`    | `n`, `s`, `m`, `l` (projector width; all sizes share the DINOv2-S encoder) |
| `eomt`      | `s`, `b`, `l` — semantic: ADE20K 150-class at 512; segment: COCO 80-class at 640 (l also at 1280) |
| `pidnet`    | `s`, `m`, `l` (PIDNet Small/Medium/Large, Cityscapes checkpoints at 1024) |
| `picodet`   | `s`, `m`, `l` (320 / 416 / 640 input) |
| `ec`     | `s`, `m`, `l`, `x` |
| `l2cs`      | `r18`, `r34`, `r50`, `r101`, `r152` (ResNet backbone depth) |
| `fomo`      | `s`, `m`, `l` |
| `mobilenetv4` | `s`, `m`, `l` (conv-Small/Medium/Large) |
| `convnext`  | `t`, `s`, `b` (V1 Tiny/Small/Base) |
| `efficientnetv2` | `b0`, `b1`, `b2`, `b3` (EfficientNetV2-base scaling tiers) |
| `resnet`    | `18`, `34`, `50`, `101` (ResNet depth) |
| `nafnet`    | `s`, `l` (small width-32 / large width-64 restoration models) |
| `depth_anything` | `s`, `b`, `l`, `g` (ViT-S/B/L/G, all at 518) |

Promptable SAM tier size aliases:

| Family | Size codes |
|---|---|
| `sam` | `base`, `large`, `huge` |
| `sam2` | `tiny`, `small`, `base-plus`, `large` |
| `mobilesam` | `tiny` (the default and only shipped size) |

Open-vocabulary detector snapshot families use their own size codes:

| Family | Size codes |
|---|---|
| `grounding_dino` | `t` (Swin-T), `b` (Swin-B) |
| `owlv2` | `b16` (base patch-16 ensemble), `l14` (large patch-14 ensemble) |

Notes:

- Standard codes are `n` (nano), `t` (tiny), `s` (small), `m` (medium),
  `l` (large), `x` (xlarge).
- `yolo9` uses `c` for "compact" instead of `l`.
- `rtdetr` mixes backbone-named codes (`r18`, `r50`, …) with letter codes
  (`l`, `x`).

## Task suffixes

From `libreyolo/tasks.py`:

| Task          | Filename suffix |
|---|---|
| `detect`      | *(none — implicit)* |
| `segment`     | `-seg` |
| `semantic`    | `-sem` |
| `pose`        | `-pose` |
| `classify`    | `-cls` |
| `gaze`        | `-gaze` |
| `obb`         | `-obb` |
| `point`       | `-point` |
| `depth`       | `-depth` |
| `restore`     | `-restore` |

The factory accepts selected upstream-style aliases (`detection`, `det`,
`segmentation`, `keypoints`, `cls`, …) at the API boundary; only the canonical
names above appear in filenames.

`point` is the task for object-localization models whose learned output is a
single image coordinate per detection, exposed as `(x, y, class, confidence)`.
This keeps box detection under `detect` while allowing centroid-style models to
use point-specific result and validation contracts.

`semantic` is the task for dense semantic segmentation: one class label per
pixel with no instance separation. `segment` remains the task for
instance segmentation (per-object masks). Semantic models expose
`Results.semantic_mask` and use per-pixel validation metrics (mIoU,
pixel accuracy) instead of box/mask mAP.

`depth` is the task for dense monocular depth estimation. Models expose
`Results.depth_map`, a float `(H, W)` relative inverse-depth map on the
original image canvas. Higher values mean closer to the camera; no metric unit
is implied without user-side calibration.

`restore` is the task for paired image restoration, including deblurring and
denoising. Models expose `Results.restored`, a uint8 RGB `(H, W, 3)` image on
the original image canvas. Canonical restore filenames must carry the
`-restore` suffix; task aliases such as `deblur`, `denoise`, and `restoration`
resolve to `restore` at the API boundary.

Dataset and label contracts are documented in
[`dataset_schema.md`](dataset_schema.md). A task is supported by a model family
only when it appears in that family's `SUPPORTED_TASKS`.

## Per-family task support

| Family    | `SUPPORTED_TASKS`                   | Default | Notes |
|---|---|---|---|
| `yolox`     | `("detect",)` (default)             | detect | detect-only |
| `yolo2`     | `("detect",)` (default)             | detect | YOLOv2/YOLO9000 (Darknet, public domain); inference-only in LibreYOLO |
| `yolo3`     | `("detect",)` (default)             | detect | YOLOv3 (Darknet, public domain); inference-only in LibreYOLO |
| `yolo4`     | `("detect",)` (default)             | detect | YOLOv4 (Darknet, public domain); inference-only in LibreYOLO |
| `yolo7`     | `("detect",)` (default)             | detect | YOLOv7 (MIT MultimediaTechLab/YOLO); inference-only in LibreYOLO |
| `yolo9`     | `("detect",)`                       | detect | detect-only (non-detect flagship variants removed in #436) |
| `yolo9_e2e` | `("detect",)` (default)             | detect | detect-only |
| `yolo9_p2`  | `("detect",)`                       | detect | detect-only |
| `dfine`     | `("detect", "segment")`             | detect | segment uses the D-FINE-seg mask head; same sizes as detect; COCO `-seg` weights on HF (detect-to-segment fine-tune needs an explicit transfer flag) |
| `deim`      | `("detect",)` (default)             | detect | detect-only |
| `deimv2`    | `("detect",)` (default)             | detect | detect-only |
| `rtdetr`    | `("detect",)` (default)             | detect | detect-only |
| `rtdetrv2`  | `("detect",)` (default)             | detect | detect-only |
| `rtdetrv4`  | `("detect",)` (default)             | detect | detect-only |
| `rtmdet`    | `("detect",)` (default)             | detect | detect-only; training is gated experimental (`allow_experimental=True`) |
| `picodet`   | `("detect",)` (default)             | detect | detect-only |
| `rfdetr`    | `("detect", "segment", "pose", "obb")` | detect | seg uses smaller sizes; pose/OBB use detect sizes |
| `dinov2`    | `("semantic", "classify")`          | semantic | DINOv2 backbone + task head (semantic dense head at 518 / classify linear probe at 224); NOT the RF-DETR detector |
| `eomt`      | `("semantic", "segment")`           | semantic | DINOv2 backbone; sizes s/b/l. Semantic: ADE20K 150-class at 512. Instance segment: COCO 80-class at 640 (l also at 1280). DINOv3 variants excluded |
| `pidnet`    | `("semantic",)`                     | semantic | real-time PIDNet semantic segmentation; s/m/l at 1024; Cityscapes 19-class checkpoints; inference + `val`; not trainable in LibreYOLO |
| `yolonas`   | `("detect", "pose")`                | detect | pose adds size `n` |
| `ec`     | `("detect", "pose", "segment")`     | detect | all three tasks |
| `l2cs`      | `("gaze",)`                         | gaze   | inference-only; two-stage (face detector + gaze head); not trainable in LibreYOLO |
| `fomo`      | `("point",)`                        | point  | point-only localizer model |
| `depth_anything` | `("depth",)`                   | depth  | Depth Anything V2 (DINOv2 + DPT); sizes `s`/`b`/`l`/`g` all at 518; predict + zero-shot `val`; not trainable in LibreYOLO |
| `nafnet`    | `("restore",)`                      | restore | NAFNet RGB restoration; sizes `s`/`l`; native predict runs at original resolution with reflect padding; paired PSNR/SSIM train+val; fixed-resolution ONNX v1 |
| `mobilenetv4` | `("classify",)`                | classify | MobileNetV4-conv image classifier; s/m/l at 224/224/256; predict + top-1/top-5 `val` + CE fine-tune train + ONNX |
| `convnext`  | `("classify",)`                | classify | ConvNeXt V1 image classifier; t/s/b at 224; predict + top-1/top-5 `val` + CE fine-tune train + ONNX |
| `efficientnetv2` | `("classify",)`             | classify | EfficientNetV2-base image classifier; b0/b1/b2/b3 at 224/240/260/300; predict + top-1/top-5 `val` + CE fine-tune train + ONNX |
| `resnet`    | `("classify",)`             | classify | vanilla ResNet image classifier (v1.5); 18/34/50/101 at 224; predict + top-1/top-5 `val` + CE fine-tune train + ONNX |

Families that override `SUPPORTED_TASKS` also declare `TASK_INPUT_SIZES` so
each task can use a different per-size input resolution (relevant for RF-DETR).
LibreFOMO uses `SUPPORTED_TASKS = ("point",)`. No pretrained weights are auto-downloadable for this family; see `libreyolo/models/fomo/model.py`. Other point-localization families must opt into `SUPPORTED_TASKS = ("point",)` or an equivalent multi-task tuple.

## Examples by family + task

### Detection only

```text
LibreYOLOXn.pt
LibreYOLO9s.pt
LibreYOLO9E2Es.pt
LibreYOLONASm.pt
LibreDFINEl.pt
LibreDEIMx.pt
LibreDEIMv2s.pt
LibreRTDETRr50.pt
LibreRFDETRn.pt
LibrePICODETs.pt
LibreECs.pt
```

### Multi-task families

```text
# yolonas — detect + pose
LibreYOLONASs.pt           # detect (default)
LibreYOLONASn-pose.pt      # pose (note: size n only ships for pose)
LibreYOLONASs-pose.pt
LibreYOLONASm-pose.pt
LibreYOLONASl-pose.pt

# dfine - detect + segment
LibreDFINEn.pt            # detect (default)
LibreDFINEn-seg.pt        # segment

# rfdetr - detect + segment + pose + obb
LibreRFDETRn.pt            # detect
LibreRFDETRn-seg.pt        # segment
LibreRFDETRx-pose.pt       # pose (preview; only size x ships)
LibreRFDETRn-obb.pt        # obb

# dinov2 — DINOv2 backbone + task head (NOT the RF-DETR detector)
LibreDINOv2n.pt            # semantic (default task; dense head at 518)
LibreDINOv2n-cls.pt        # classify (linear probe at 224)

# eomt - semantic (ADE20K), instance segmentation (COCO), and panoptic (COCO things+stuff)
LibreEoMTl-sem.pt          # EoMT-L, ADE20K 150-class semantic, DINOv2 backbone, 512px
LibreEoMTs-seg.pt          # EoMT-S, COCO 80-class instance segment, DINOv2 backbone, 640px
LibreEoMTb-seg.pt          # EoMT-B, COCO 80-class instance segment, DINOv2 backbone, 640px
LibreEoMTl-seg.pt          # EoMT-L, COCO 80-class instance segment, DINOv2 backbone, 640px
LibreEoMTl-seg-1280.pt     # EoMT-L, COCO 80-class instance segment, DINOv2 backbone, 1280px
LibreEoMTs-panoptic.pt     # EoMT-S, COCO 133-class panoptic (80 things + 53 stuff), DINOv2 backbone, 640px
LibreEoMTb-panoptic.pt     # EoMT-B, COCO 133-class panoptic (80 things + 53 stuff), DINOv2 backbone, 640px

# pidnet - real-time semantic segmentation
LibrePIDNets-sem.pt        # PIDNet-S, Cityscapes 19-class semantic
LibrePIDNetm-sem.pt        # PIDNet-M, Cityscapes 19-class semantic
LibrePIDNetl-sem.pt        # PIDNet-L, Cityscapes 19-class semantic

# ec — detect + pose + segment
LibreECs.pt             # detect (default)
LibreECs-pose.pt        # pose
LibreECs-seg.pt         # segment

# depth_anything — Depth Anything V2 (depth-only)
LibreDepthAnythingV2s-depth.pt   # ViT-S (Apache-2.0 weights)
LibreDepthAnythingV2b-depth.pt   # ViT-B (CC-BY-NC-4.0 weights)
LibreDepthAnythingV2l-depth.pt   # ViT-L (CC-BY-NC-4.0 weights)
LibreDepthAnythingV2g-depth.pt   # ViT-G (CC-BY-NC-4.0 weights)

# nafnet — NAFNet restoration (restore-only)
LibreNAFNets-restore.pt
LibreNAFNetl-restore.pt
```

### Zero-shot / open-vocabulary classify (inference-only)

```text
# clip — CLIP zero-shot, open-vocabulary (set_classes); no fixed label set.
# Defaults to ImageNet-1k labels; classify task suffix `-cls`.
LibreCLIPb32-cls.pt       # OpenCLIP ViT-B/32, LAION-2B (MIT weights)
LibreCLIPb16-cls.pt       # OpenCLIP ViT-B/16, LAION-2B (MIT weights)
LibreCLIPl14-cls.pt       # OpenCLIP ViT-L/14, LAION-2B (config + converter ready; weights not yet published)
```

### Open-vocabulary detection (inference-only snapshot tier)

```text
# grounding_dino - Hugging Face snapshot, no .pt checkpoint filename
weights/LibreGroundingDINOt/
weights/LibreGroundingDINOb/

# owlv2 - Hugging Face snapshot, no .pt checkpoint filename
weights/LibreOWLv2b16/
weights/LibreOWLv2l14/
```

### Gaze (inference-only)

```text
LibreL2CSr50.pt           # L2CS gaze estimation (ResNet-50, Gaze360 weights)
```

### Point (object-localizer)

```text
LibreFOMOs-point.pt       # FOMO point-localizer (size s, point task)
LibreFOMOm-point.pt       # FOMO point-localizer (size m, point task)
LibreFOMOl-point.pt       # FOMO point-localizer (size l, point task)
```

These are the canonical filenames for LibreFOMO checkpoints. Pretrained weights
are not currently auto-downloadable; pass a local checkpoint path or train from
scratch. See `libreyolo/models/fomo/model.py` for details.

`gaze` is L2CS's only task, so — like `detect` for the detection families —
it carries no suffix in the canonical filename; `-gaze` is accepted but
redundant. L2CS weights are not hosted by LibreYOLO (the Gaze360 dataset
license forbids redistribution); see `libreyolo/models/l2cs/model.py`.

### Classification (classifier-only)

```text
LibreMobileNetV4s-cls.pt   # MobileNetV4-conv-Small  (224, ImageNet-1k)
LibreMobileNetV4m-cls.pt   # MobileNetV4-conv-Medium (224, ImageNet-1k)
LibreMobileNetV4l-cls.pt   # MobileNetV4-conv-Large  (256, ImageNet-1k)

LibreConvNeXtt-cls.pt      # ConvNeXt-V1-Tiny        (224, ImageNet-1k)
LibreConvNeXts-cls.pt      # ConvNeXt-V1-Small       (224, ImageNet-1k)
LibreConvNeXtb-cls.pt      # ConvNeXt-V1-Base        (224, ImageNet-1k)

LibreEfficientNetV2b0-cls.pt   # EfficientNetV2-base-b0 (224, ImageNet-1k)
LibreEfficientNetV2b1-cls.pt   # EfficientNetV2-base-b1 (240, ImageNet-1k)
LibreEfficientNetV2b2-cls.pt   # EfficientNetV2-base-b2 (260, ImageNet-1k)
LibreEfficientNetV2b3-cls.pt   # EfficientNetV2-base-b3 (300, ImageNet-1k)

LibreResNet18-cls.pt       # ResNet-18  (224, ImageNet-1k, a1 recipe)
LibreResNet34-cls.pt       # ResNet-34  (224, ImageNet-1k, a1 recipe)
LibreResNet50-cls.pt       # ResNet-50  (224, ImageNet-1k, a1 recipe)
LibreResNet101-cls.pt      # ResNet-101 (224, ImageNet-1k, a1 recipe)
```

Unlike `gaze`/`point` (which carry their suffix despite being single-task),
`classify` keeps its `-cls` suffix to match the ecosystem-wide convention. The
`mobilenetv4` family is a native port of MobileNetV4 (the speed tier); the
`convnext` family is a native port of ConvNeXt V1; the `efficientnetv2` family
is a native port of EfficientNetV2-base (the accuracy tier). All are derived
from timm (Apache-2.0); weights are Apache-2.0 ImageNet-1k and load
bit-identically (see each family's `NOTICE`, e.g.
`libreyolo/models/efficientnetv2/NOTICE`, `libreyolo/models/convnext/NOTICE`).
Only ConvNeXt **V1** ships — ConvNeXt-V2's small checkpoints are CC-BY-NC and
are excluded; EfficientNetV2 ships only the ImageNet-1k checkpoints, as the
`.in21k`/JFT variants carry extra-data terms.

**Eval resolution is a deliberate choice.** The classify families evaluate at a
real-time-friendly default (224 for MobileNetV4 s/m, ConvNeXt, ResNet; 256 for
MobileNetV4-l; 224/240/260/300 for EfficientNetV2 b0–b3) rather than timm's
larger *test* resolutions (e.g. 256/288/320), which trade ~1.6–2× compute for a
few tenths of a percent top-1. This does **not** affect parity — given the same
input tensor the logits are bit-identical to timm — only the headline ImageNet
number, which sits a hair below the test-size figure. Each family threads its
`crop_pct`/`interpolation` through `predict()`, `val()`, and exported-backend
inference so all three agree.

## Resolution precedence

When loading via `LibreYOLO("...")`, the task is resolved with this priority
(see `libreyolo/tasks.py:resolve_task` and the factory in
`libreyolo/models/__init__.py`):

```
explicit task=    →    checkpoint["task"]    →    filename suffix    →    family DEFAULT_TASK
```

Official LibreYOLO v1.0 checkpoints must carry `task` metadata; see
[`checkpoint_schema.md`](checkpoint_schema.md). State-dict key inspection is a
legacy compatibility path for old LibreYOLO checkpoints, not the standard for
new artifacts.

## Filename regex

`BaseModel._filename_regex` builds the canonical pattern as:

```
<prefix>(?P<size>{size_alternation})(?P<task>{task_suffixes})?(?P<variant>-{variants})?\.pt
```

with `task_suffixes` derived from `SUPPORTED_TASKS` via
`libreyolo.tasks.task_suffix_pattern`. This is the single source of truth for
parsing a filename back into `(family, size, task, variant)`.

The `variant` group only exists for families that declare `WEIGHT_VARIANTS`, a
dataset suffix for published checkpoints trained on a non-default dataset.
Example: `yolo9_p2` declares `("visdrone",)`, so `LibreYOLO9P2s-visdrone.pt`
resolves the Hugging Face repo `LibreYOLO/LibreYOLO9P2s-visdrone` (a research
preview under VisDrone's CC BY-NC-SA license, announced by a download notice).
Plain COCO-default weights never carry a variant suffix.
