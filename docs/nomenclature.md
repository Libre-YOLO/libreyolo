# LibreYOLO Model Nomenclature

This document catalogs the model-naming conventions **currently in use** in
the LibreYOLO repository. It is descriptive — it records what is there today,
not a proposal. Sources of truth are the `FAMILY` and `FILENAME_PREFIX`
class constants in `libreyolo/models/<family>/model.py` and the
task-resolution rules in [`libreyolo/tasks.py`](../libreyolo/tasks.py).

## Model groups

Families are enrolled in rollout groups that decide where a cross-cutting
feature lands first and how hard it is validated. The single source of truth
is `MODEL_GROUPS` in
[`libreyolo/models/registry.py`](../libreyolo/models/registry.py); the CLI
inventory exposes each family's group, and
`tests/unit/test_model_registry.py` fails when a registered family is not
enrolled.

| Group | Meaning |
|---|---|
| `g0` | Flagship (`yolo9`, `rfdetr`). Features are designed and fully GPU-validated here first. |
| `g1` | Core trainable detectors. Features follow `g0` in the same release wave, GPU-smoked per family. |
| `g2` | Supporting trainables. Kept green in CI; features land opportunistically or on request. |
| `g3` | Inference-only specialists. Predict/val/export surface only; training features do not apply. |
| `g4` | Museum (`yolo1`-`yolo4`). Frozen exhibits; bug fixes only. |
| `s`  | Sibling tiers (SAM, open-vocab, VLM, zero-shot). Separate product surfaces, excluded from group rollouts. |

Groups classify **families, not tasks**: "add validation loss to `g1`" means
the `g1` families, and a task-scoped rollout says so explicitly ("`g1`
detect"). A group states the default priority, not an obligation; a family
may sit out a feature its architecture cannot support, documented at the
feature.

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
separate category, covered in the note below). Most are detectors; `pidnet`, `segformer`, and `lingbotvision` are semantic-only; `eomt` supports semantic,
instance, and panoptic segmentation; the `mobilenetv4` / `convnext` /
`efficientnetv2` / `resnet` families are classify-only:

| Family id (`FAMILY`) | Filename prefix | Casing rule applied |
|---|---|---|
| `yolox`     | `LibreYOLOX`    | All-caps acronym |
| `yolo1`     | `LibreYOLO1`    | All-caps acronym + version digit (YOLOv1 / Darknet, public domain; VOC-20, fixed 448) — inference-only |
| `yolo2`     | `LibreYOLO2`    | All-caps acronym + version digit (YOLOv2 / Darknet, public domain) — inference-only |
| `yolo3`     | `LibreYOLO3`    | All-caps acronym + version digit (YOLOv3 / Darknet, public domain) — inference-only |
| `yolo4`     | `LibreYOLO4`    | All-caps acronym + version digit (YOLOv4 / Darknet, public domain) — inference-only |
| `yolo7`     | `LibreYOLO7`    | All-caps acronym + version digit (YOLOv7 / MIT MultimediaTechLab/YOLO) |
| `yolo9`     | `LibreYOLO9`    | All-caps acronym + version digit |
| `yolo9_e2e` | `LibreYOLO9E2E` | All-caps acronym + version + variant |
| `yolo9_p2`  | `LibreYOLO9P2`  | All-caps acronym + version + variant (stride-4 small-object) |
| `yolonas`   | `LibreYOLONAS`  | All-caps acronym (hyphen dropped from `YOLO-NAS`) |
| `dfine`     | `LibreDFINE`    | All-caps acronym (hyphen dropped from `D-FINE`) |
| `deim`      | `LibreDEIM`     | All-caps acronym |
| `deimv2`    | `LibreDEIMv2`   | All-caps acronym + lowercase version |
| `detr`      | `LibreDETR`     | All-caps acronym |
| `rtdetr`    | `LibreRTDETR`   | All-caps acronym (hyphen dropped from `RT-DETR`) |
| `rtdetrv2`  | `LibreRTDETRv2` | All-caps acronym + lowercase version |
| `rtdetrv4`  | `LibreRTDETRv4` | All-caps acronym + lowercase version |
| `rtmdet`    | `LibreRTMDet`   | Upstream brand casing preserved (`RTMDet`) |
| `rfdetr`    | `LibreRFDETR`   | All-caps acronym (hyphen dropped from `RF-DETR`) |
| `lwdetr`    | `LibreLWDETR`   | All-caps acronym (hyphen dropped from `LW-DETR`) |
| `faster_rcnn` | `LibreFasterRCNN` | Upstream acronym and underscore retained in the family id; canonical filename drops punctuation; inference-only two-stage detector |
| `ssd`       | `LibreSSD`       | All-caps acronym; fixed-300 inference-only single-shot detector |
| `deformable_detr` | `LibreDeformableDETR` | Upstream name rendered as `DeformableDETR`; the family id retains the separator |
| `dinov2`    | `LibreDINOv2`   | All-caps acronym + lowercase version (DINOv2 backbone) |
| `eomt`      | `LibreEoMT`     | Mixed-case upstream brand preserved (`EoMT`) - semantic + instance + panoptic segmentation transformer family |
| `pidnet`    | `LibrePIDNet`   | All-caps acronym + `Net` brand casing - semantic-only real-time family |
| `segformer` | `LibreSegformer` | CamelCase preserved (upstream brand casing) — semantic-only transformer family; weights are non-commercial |
| `lingbotvision` | `LibreLingBotVision` | Upstream brand casing preserved (`LingBot-Vision`, hyphen dropped) — semantic-only ViT family; Apache-2.0 backbone weights |
| `picodet`   | `LibrePICODET`  | All-caps (`PicoDet` rendered uppercase) |
| `ec`     | `LibreEC`    | Short form of EdgeCrafter — used as the family alias for the three sibling upstream models `ECDet`, `ECPose`, `ECSeg` |
| `l2cs`      | `LibreL2CS`     | All-caps acronym (`L2CS` gaze estimation) — inference-only |
| `fomo`      | `LibreFOMO`     | All-caps acronym (Faster Objects, More Objects) |
| `mobilenetv4` | `LibreMobileNetV4` | CamelCase preserved (MobileNet is not an acronym) — first classify-only family |
| `convnext`  | `LibreConvNeXt`  | CamelCase preserved (upstream brand casing `ConvNeXt`) — classify-only family |
| `efficientnetv2` | `LibreEfficientNetV2` | CamelCase preserved (EfficientNet is not an acronym) — classify-only accuracy tier |
| `resnet`    | `LibreResNet`    | CamelCase preserved (`ResNet` brand casing) — classify-only baseline |
| `clip`      | `LibreCLIP`     | All-caps acronym (`CLIP` zero-shot classify + image/text embed) — inference-only |
| `siglip2`   | `LibreSigLIP2`  | Upstream brand casing preserved (`SigLIP`) + version (`SigLIP 2` zero-shot classify + image/text embed); inference-only |
| `nafnet`    | `LibreNAFNet`   | All-caps acronym + CamelCase `Net`; restore-only image-restoration family |
| `realesrgan` | `LibreRealESRGAN` | Upstream brand casing (`RealESRGAN`); restore-only super-resolution family |
| `swinir`    | `LibreSwinIR`    | Upstream brand casing (`SwinIR`); restore-only transformer super-resolution family |
| `depth_anything` | `LibreDepthAnythingV2` | CamelCase preserved + version (Depth Anything V2), depth-only |
| `depth_anything3` | `LibreDepthAnything3` | CamelCase preserved + version (Depth Anything 3), depth-only |
| `zipdepth`  | `LibreZipDepth` | CamelCase preserved (`ZipDepth` brand casing); depth-only lightweight CNN (speed/edge tier) |
| `moge2`     | `LibreMoGe2` | Upstream brand casing preserved (`MoGe`) + version; surface-normal-only |
| `teed`      | `LibreTEED` | All-caps acronym (`TEED`); edge-only tiny CNN specialist |
| `dexined`   | `LibreDexiNed` | Upstream brand casing preserved (`DexiNed`); edge-only base CNN specialist |
| `birefnet`  | `LibreBiRefNet` | CamelCase preserved (Bilateral Reference); matte-only background-removal family |
| `feynobg`   | `LibreFeyNobg` | CamelCase preserved (FeyNobg); matte-only background-removal family built on the BiRefNet architecture |
| `ppocr`     | `LibrePPOCR`    | All-caps acronym (PP-OCR brand, hyphen dropped); ocr-only two-stage text detection + recognition family |
| `facerec`   | `LibreFaceEmbedder` | Descriptive family name (no upstream brand): embed-only two-stage face detection + identity-embedding family, inference-only |
| `sam3dbody` | `LibreSAM3DBody` | All-caps acronym plus CamelCase `Body` (hyphens dropped); mesh-only family. Named in full rather than shortened so it does not collide with the `LibreSAM` promptable-segmentation tier. Sizes are backbone codes: `d3` (DINOv3 ViT-H/16+) and `h` (ViT-H). This family wraps an optional third-party package rather than porting it; see ADR 0013 |

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
`LocateAnything`, `LibreMODUS`, `LibreSAM`, `LibreSAM2`, `LibreSAM3`,
`LibreMobileSAM`, `LibrePicoSAM3`, `LibreEdgeTAM`) are not registered
into the detector factory and do not emit `Libre<FAMILY><size>.pt` detector
checkpoints. Their `FILENAME_PREFIX` is only a weights-directory prefix for a
downloaded Hugging Face snapshot or promptable checkpoint, so upstream brand
casing (CamelCase) is intentionally preserved. See
[`librevlm_design.md`](librevlm_design.md) and
[`adr/0007-libresam-contract.md`](adr/0007-libresam-contract.md).

The open-vocabulary detector tier is also separate from the checkpoint factory.
Its weights-directory prefixes (`LibreGroundingDINO`, `LibreOWLv2`,
`LibreOMDetTurbo`) identify
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
| `yolo1`     | `t` (tiny-yolov1, 448), `b` (yolov1, 448) (both fixed 448; the FC head forbids dynamic shapes) |
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
| `detr`      | `r50`, `r50dc5`, `r101`, `r101dc5` (ResNet depth plus optional dilated C5; all use a fixed 800 square) |
| `rtdetr`    | `r18`, `r34`, `r50`, `r50m`, `r101`, `l`, `x` |
| `rtdetrv2`  | `r18`, `r34`, `r50`, `r50m`, `r101` |
| `rtdetrv4`  | `s`, `m`, `l`, `x` |
| `rtmdet`    | `t`, `s`, `m`, `l`, `x` |
| `rfdetr`    | `n`, `s`, `m`, `l` |
| `lwdetr`    | `t`, `s`, `m`, `l`, `x` (upstream tiny / small / medium / large / xlarge; all at 640, which must stay a multiple of 64) |
| `faster_rcnn` | `n`, `s`, `m`, `l` (MobileNetV3 320-FPN / MobileNetV3 FPN / ResNet-50 FPN v1 / ResNet-50 FPN v2; public input sizes 320 / 800 / 800 / 800) |
| `ssd`       | `300` (SSD300 VGG16; input is always fixed at 300 x 300) |
| `deformable_detr` | `r50ss`, `r50ssdc5`, `r50`, `r50refine`, `r50twostage` (single-scale, single-scale DC5, multi-scale base, iterative refinement, and refinement plus two-stage; all fixed at 800) |
| `dinov2`    | `n`, `s`, `m`, `l` (projector width; all sizes share the DINOv2-S encoder) |
| `eomt`      | `s`, `b`, `l` — semantic: ADE20K 150-class at 512 (l only); segment: COCO 80-class at 640 (l only, also 1280); panoptic: COCO 133-class at 640 (s/b/l) |
| `pidnet`    | `s`, `m`, `l` (PIDNet Small/Medium/Large, Cityscapes checkpoints at 1024) |
| `segformer` | `b0`, `b1`, `b2`, `b3`, `b4`, `b5` (MiT-b0..b5 encoder depth/width tiers; ADE20K at 512, b5 at 640) |
| `lingbotvision` | `s`, `b`, `l`, `g` (ViT-S/B/L distilled from the ViT-g teacher; g is the 1.1B teacher, loadable but no hosted weights; all at 512) |
| `picodet`   | `s`, `m`, `l` (320 / 416 / 640 input) |
| `ec`     | `s`, `m`, `l`, `x` |
| `l2cs`      | `r18`, `r34`, `r50`, `r101`, `r152` (ResNet backbone depth) |
| `fomo`      | `s`, `m`, `l` |
| `mobilenetv4` | `s`, `m`, `l` (conv-Small/Medium/Large) |
| `convnext`  | `t`, `s`, `b` (V1 Tiny/Small/Base) |
| `efficientnetv2` | `b0`, `b1`, `b2`, `b3` (EfficientNetV2-base scaling tiers) |
| `resnet`    | `18`, `34`, `50`, `101` (ResNet depth) |
| `nafnet`    | `s`, `l` (small width-32 / large width-64 restoration models). Weight variants select the degradation: `LibreNAFNetl-restore.pt` (GoPro deblur) and `LibreNAFNetl-restore-sidd.pt` (SIDD denoise, the model behind the `denoise` alias) |
| `realesrgan` | `x4`, `x2`, `x4t` (size code encodes scale + tier: `x4` = RealESRGAN_x4plus RRDBNet 4x quality default, `x2` = RealESRGAN_x2plus RRDBNet 2x, `x4t` = realesr-general-x4v3 SRVGG compact 4x fast/video tier) |
| `swinir`    | `s`, `m`, `l` (all 4x: lightweight SwinIR-S, real-world SwinIR-M, and real-world SwinIR-L) |
| `depth_anything` | `s`, `b`, `l`, `g` (ViT-S/B/L/G, all at 518) |
| `depth_anything3` | `l` (DA3MONO-LARGE ViT-L, native upper-bound 504) |
| `zipdepth`  | `b` (base, GPU/CPU convex upsampling), `bnpu` (base capacity with the separately trained unfold-free upsampling head for NPU/edge compilers); both at short-side 384 |
| `moge2`     | `s`, `b`, `l` (official MoGe-2 ViT-S/B/L-14 normal checkpoints; all at native short side 518, `l` quality default) |
| `teed`      | `t` (tiny, 58,910 parameters; fixed 352 square) |
| `dexined`   | `b` (base, 35.2M parameters; fixed 352 square) |
| `birefnet`  | `t` (BiRefNet_lite, Swin-T tier), `l` (BiRefNet general, Swin-L tier); both at fixed 1024 |
| `feynobg`   | `l` (single released variant: Swin-L tier with stage 3 deepened to 24 blocks, 263M params) at fixed 1024 |
| `ppocr`     | `t` (PP-OCRv5 mobile det + mobile rec, CPU tier), `l` (PP-OCRv5 server det + server rec, quality tier); detection long side 960 |
| `clip`      | `b32`, `b16`, `l14` (ViT patch size baked in, all at 224) |
| `siglip2`   | `b16` (base patch-16 at 256), `so400m` (shape-optimized 400M patch-14 at 384) |

VLM snapshot families use model-specific size names:

| Family | Size codes |
|---|---|
| `libremodus` | `14b-a7b` (14B total parameters, approximately 7B active; external MODUS snapshot) |

Promptable SAM tier size aliases:

| Family | Size codes |
|---|---|
| `sam` | `base`, `large`, `huge` |
| `sam2` | `tiny`, `small`, `base-plus`, `large` |
| `edgetam` | `edge` (the default and only shipped size; 1024px input) |
| `sam3` | `large` (the default and only shipped size) |
| `mobilesam` | `tiny` (the default and only shipped size) |
| `picosam3` | `pico` (the default and only shipped size; 96px ROI input) |

Open-vocabulary detector snapshot families use their own size codes:

| Family | Size codes |
|---|---|
| `grounding_dino` | `t` (Swin-T), `b` (Swin-B) |
| `owlv2` | `b16` (base patch-16 ensemble), `l14` (large patch-14 ensemble) |
| `omdet_turbo` | `t` (Swin-T, the only released checkpoint) |
| `ov_deim` | `s`, `m`, `l` (ViT-tiny / ViT-tinyplus / DINOv3-S backbones) |

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
| `panoptic`    | `-panoptic` |
| `pose`        | `-pose` |
| `classify`    | `-cls` |
| `gaze`        | `-gaze` |
| `obb`         | `-obb` |
| `point`       | `-point` |
| `depth`       | `-depth` |
| `edge`        | `-edge` |
| `normal`      | `-normal` |
| `restore`     | `-restore` |
| `matte`       | `-matte` |
| `ocr`         | `-ocr` |
| `embed`       | `-embed` |
| `mesh`        | `-mesh` |

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

`panoptic` is the task for panoptic segmentation: every pixel gets exactly one
non-overlapping label, unifying `semantic` "stuff" (amorphous regions) with
`segment` "things" (countable instances). Panoptic models expose
`Results.panoptic` (a `(H, W)` segment-id map plus `segments_info`) and are
scored with Panoptic Quality (PQ = SQ x RQ) rather than mIoU or mask mAP. The
canonical suffix is the full word `-panoptic` (not an abbreviation), so
`LibreEoMTs-panoptic.pt` is a first-class panoptic checkpoint, not a `segment`
checkpoint in disguise. Ground truth follows the COCO-panoptic format
(`PanopticDataset`) and `model.val()` reports PQ / SQ / RQ split into things and
stuff. The per-family panoptic postprocess is what a model family provides (a
family without it raises from `_postprocess`); the `eomt` family implements it
as a Mask2Former-style non-overlapping thing+stuff merge.

`depth` is the task for dense monocular depth estimation. Models expose
`Results.depth_map`, a float `(H, W)` relative inverse-depth map on the
original image canvas. Higher values mean closer to the camera; no metric unit
is implied without user-side calibration.

`edge` is the task for dense edge detection. Models expose `Results.edges`, a
float32 `(H, W)` probability map in `[0, 1]` on the original image canvas.
`EdgeMap.binary(threshold)` produces a boolean mask. Plotting renders the
continuous map as inverted grayscale (high-confidence edges are black), but
does not alter the stored probabilities. Validation reports BSDS-style ODS and
OIS F-measures after non-maximum thinning with one-to-one pixel matching.

`normal` is the task for dense surface-normal estimation. Models expose
`Results.normal_map` (also available as `Results.normals`), a float32
`(H, W, 3)` unit-vector field in `[-1, 1]` on
the original image canvas at its original resolution. Vectors use the OpenCV
camera frame: `+x` points right, `+y` points down, and `+z` points into the
scene. Normals face the camera (`n . ray < 0` for each visible surface), so a
fronto-parallel wall facing the viewer is `(0, 0, -1)`. The RGB mapping
`(normal + 1) / 2` is a visualization produced by plotting and saving; it is
never the stored payload. Each family converts its upstream convention at the
family boundary before constructing `NormalMap`.

`restore` is the task for paired image restoration, including deblurring,
denoising, and super-resolution. Models expose `Results.restored`, a uint8 RGB
`(H, W, 3)` image. For deblur/denoise the restored canvas equals the input; for
super-resolution it is `Results.restore_scale` times larger on each axis
(`restore_scale` is `1` for deblur/denoise and every non-super-resolution
result). Canonical restore filenames must carry the `-restore` suffix; task
aliases such as `deblur`, `denoise`, `restoration`, `sr`, `super-resolution`,
and `upscale` resolve to `restore` at the API boundary.

`matte` is the task for background removal / dichotomous image segmentation.
Models expose `Results.matte`, a float `(H, W)` soft alpha map in `[0, 1]` on
the original image canvas (`1` = foreground, `0` = background), plus
`results.cutout()` (RGBA) and a transparent-PNG `results.save()`. Canonical
matte filenames must carry the `-matte` suffix; task aliases such as `matting`,
`background-removal`, `rembg`, and `dis` resolve to `matte` at the API boundary.
See ADR 0010 for the full contract.

`ocr` is the task for located text: detection quads plus transcripts. Models
expose `Results.ocr`, a list of text regions each carrying a 4-point `polygon`
in original-image coordinates, the recognized `text`, and recognition plus
detection confidences, in reading order (top to bottom, then left to right).
Detection quads are genuine polygons (rotated text) and do not populate
`Results.boxes`. Canonical ocr filenames must carry the `-ocr` suffix; task
aliases `text`, `text-recognition`, and `text_recognition` resolve to `ocr` at
the API boundary.

`embed` turns an image, image region, or paired-tower text into a float32,
L2-normalized vector whose dot product measures agreement. Whole-image results
carry `Results.embeddings` with shape `(1, D)` and no boxes; region results use
`(N, D)` rows aligned with `Results.boxes`. Paired image/text families expose
`model.embed_text(texts) -> (M, D)`; a string passed to `model(...)` remains an
image path and is never inferred to be text. `Gallery` stores named references
for any shape, while `FaceGallery` remains its compatibility alias.

Dedicated embed checkpoints use `-embed`. Dual-task CLIP and SigLIP2 reuse
their existing `-cls` two-tower artifact with an explicit `task="embed"`; no
duplicate artifact is published for identical weights. DINOv2 likewise loads
an existing family checkpoint and bypasses its task head. Task aliases
`facial-recognition`, `face-recognition`, `recognition`, `face`, `faceid`,
`embedding`, and `reid` resolve to `embed` at the API boundary. See ADR 0013
for the face-region contract and ADR 0015 for the general contract.
`mesh` is the task for human body mesh recovery: recovering a posed 3D body per
detected person. Models expose `Results.meshes`, row-aligned with
`Results.boxes` exactly as pose keypoints are, carrying the parametric core
(`global_orient`, `body_pose`, `betas`, `transl`) plus decoded `vertices`,
`joints3d` and `joints2d`. Everything is in the camera frame of the original
image, with metric translation in meters and `joints2d` in original-image
pixels; there is no world frame in this version. Parameter layouts vary by body
model, so `Meshes.body_model` names the parameterization and shapes are read
from the tensors rather than assumed. Canonical mesh filenames must carry the
`-mesh` suffix; task aliases `body-mesh`, `hmr`, and `human-mesh-recovery`
resolve to `mesh` at the API boundary. Note that `smpl` is deliberately *not*
an alias: the shipped body model is MHR, and accepting the name would imply an
interoperability that is not provided. See ADR 0013 for the full contract.

Dataset and label contracts are documented in
[`dataset_schema.md`](dataset_schema.md). A task is supported by a model family
only when it appears in that family's `SUPPORTED_TASKS`.

## Per-family task support

The VLM tier is separate from the detector checkpoint factory. LibreMODUS is
the multi-task analysis family in that tier:

| Family | `SUPPORTED_TASKS` | Default | Notes |
|---|---|---|---|
| `libremodus` | `("detect", "depth", "normal", "edge")` | detect | External `14b-a7b` snapshot; standard tasks plus image-conditioned `any2any()`; inference-only; no RGB output |

The promptable SAM tier is separate from the detector checkpoint factory. Its
families all expose the promptable `segment` task:

| Family | `SUPPORTED_TASKS` | Default | Notes |
|---|---|---|---|
| `sam` | `("segment",)` | segment | SAM-1 image prompting |
| `sam2` | `("segment",)` | segment | SAM-2 image prompting; video deferred |
| `edgetam` | `("segment",)` | segment | EdgeTAM image prompting; video deferred |
| `sam3` | `("segment",)` | segment | SAM 3 visual and concept prompting |
| `mobilesam` | `("segment",)` | segment | MobileSAM image prompting |
| `picosam3` | `("segment",)` | segment | PicoSAM3 box prompting |

Detector-factory family support follows:

| Family    | `SUPPORTED_TASKS`                   | Default | Notes |
|---|---|---|---|
| `yolox`     | `("detect",)` (default)             | detect | detect-only |
| `yolo1`     | `("detect",)` (default)             | detect | YOLOv1 (Darknet, public domain); Pascal VOC 20 classes, fixed 448; inference-only in LibreYOLO |
| `yolo2`     | `("detect",)` (default)             | detect | YOLOv2/YOLO9000 (Darknet, public domain); inference-only in LibreYOLO |
| `yolo3`     | `("detect",)` (default)             | detect | YOLOv3 (Darknet, public domain); inference-only in LibreYOLO |
| `yolo4`     | `("detect",)` (default)             | detect | YOLOv4 (Darknet, public domain); inference-only in LibreYOLO |
| `yolo7`     | `("detect",)` (default)             | detect | YOLOv7 (MIT MultimediaTechLab/YOLO); trainable via SimOTA loss |
| `yolo9`     | `("detect",)`                       | detect | detect-only (non-detect flagship variants removed in #436) |
| `yolo9_e2e` | `("detect",)` (default)             | detect | detect-only |
| `yolo9_p2`  | `("detect",)`                       | detect | detect-only |
| `dfine`     | `("detect", "segment")`             | detect | segment uses the D-FINE-seg mask head; same sizes as detect; COCO `-seg` weights on HF (detect-to-segment fine-tune needs an explicit transfer flag) |
| `deim`      | `("detect",)` (default)             | detect | detect-only |
| `deimv2`    | `("detect",)` (default)             | detect | detect-only |
| `detr`      | `("detect",)`                       | detect | original DETR; inference-only (no trainer, `train()` raises); fixed 800 square |
| `rtdetr`    | `("detect",)` (default)             | detect | detect-only |
| `rtdetrv2`  | `("detect",)` (default)             | detect | detect-only |
| `rtdetrv4`  | `("detect",)` (default)             | detect | detect-only |
| `lwdetr`    | `("detect",)`                       | detect | detect-only; inference-only (no trainer, `train()` raises) |
| `faster_rcnn` | `("detect",)`                     | detect | detect-only; inference-only native RPN + RoI graph; official COCO heads map sparse 91-way ids to contiguous COCO-80 |
| `ssd`       | `("detect",)`                     | detect | detect-only; inference-only fixed-300 VGG16 graph; official COCO head maps sparse 91-way ids to contiguous COCO-80 |
| `rtmdet`    | `("detect", "segment")` (default: detect) | detect | RTMDet-Ins uses `-seg`; detect training is gated experimental, segment training is not implemented |
| `picodet`   | `("detect",)` (default)             | detect | detect-only |
| `rfdetr`    | `("detect", "segment", "pose", "obb")` | detect | seg uses smaller sizes; pose/OBB use detect sizes |
| `dinov2`    | `("semantic", "classify", "embed")` | semantic | DINOv2 backbone + task head; embed bypasses heads and returns the 384-d final CLS token at 224 (all sizes share DINOv2-S); no text tower |
| `eomt`      | `("semantic", "segment", "panoptic")` | semantic | DINOv2 backbone; sizes s/b/l. Semantic: ADE20K 150-class at 512. Instance segment: COCO 80-class at 640 (l also at 1280). Panoptic: COCO 133-class at 640. Upstream ships no COCO instance checkpoint at s/b. DINOv3 variants excluded |
| `pidnet`    | `("semantic",)`                     | semantic | real-time PIDNet semantic segmentation; s/m/l at 1024; Cityscapes 19-class checkpoints; inference + `val`; not trainable in LibreYOLO |
| `segformer` | `("semantic",)`                     | semantic | SegFormer MiT-b0..b5 encoder + all-MLP decode head; ADE20K 150-class at 512 (b5 at 640). Pretrained weights are NON-COMMERCIAL (NVIDIA Source Code License, research/evaluation only); also trainable from scratch via `model.train(...)` for unrestricted use |
| `lingbotvision` | `("semantic",)`                 | semantic | LingBot-Vision self-supervised ViT (Apache-2.0, arXiv:2607.05247) + 1x1 dense head (the report's linear probe); s/b/l/g at 512; ADE20K 150-class hosted weights for s/b/l; head-only training by default (`freeze_backbone=False` for full fine-tune) |
| `yolonas`   | `("detect", "pose")`                | detect | pose adds size `n` |
| `ec`     | `("detect", "pose", "segment")`     | detect | all three tasks |
| `l2cs`      | `("gaze",)`                         | gaze   | inference-only; two-stage (face detector + gaze head); not trainable in LibreYOLO |
| `fomo`      | `("point",)`                        | point  | point-only localizer model |
| `depth_anything` | `("depth",)`                   | depth  | Depth Anything V2 (DINOv2 + DPT); sizes `s`/`b`/`l`/`g` all at 518; predict + zero-shot `val`; not trainable in LibreYOLO |
| `depth_anything3` | `("depth",)`                  | depth  | Depth Anything 3 mono (ViT-L + DPT); size `l` at upper-bound 504; recommended quality default; Apache-2.0 code/weights; predict + zero-shot `val`; not trainable in LibreYOLO |
| `zipdepth`  | `("depth",)`                        | depth  | ZipDepth lightweight CNN (RepVGG encoder + FPN decoder, DA2-L distilled); sizes `b`/`bnpu` at short-side 384; predict + zero-shot `val` + fixed-resolution ONNX/TorchScript export; MIT code and weights; not trainable in LibreYOLO |
| `moge2`     | `("normal",)`                       | normal | MoGe-2 ViT-S/B/L-14 normal models; sizes `s`/`b`/`l` at native short-side 518 (`l` quality default); OpenCV camera-frame unit normals; predict + zero-shot `val` + fixed-resolution ONNX export; official code and checkpoints are MIT; not trainable in LibreYOLO |
| `teed`      | `("edge",)`                         | edge   | TEED tiny edge CNN at 352; MIT architecture source; predict + ODS/OIS `val` + fixed-resolution ONNX; local checkpoints only because the released BIPED-trained weights are non-commercial |
| `dexined`   | `("edge",)`                         | edge   | DexiNed base edge CNN at 352; MIT architecture source; predict + ODS/OIS `val` + fixed-resolution ONNX; local checkpoints only because the released BIPED-trained weights are non-commercial |
| `nafnet`    | `("restore",)`                      | restore | NAFNet RGB restoration; sizes `s`/`l`; native predict runs at original resolution with reflect padding; paired PSNR/SSIM train+val; fixed-resolution ONNX v1. Published denoise weights: `LibreNAFNetl-restore-sidd.pt` (SIDD width-64, bit-exact conversion, upstream PSNR 40.3045 dB) |
| `birefnet`  | `("matte",)`                        | matte  | BiRefNet background removal; sizes `t` (lite)/`l` (general), both fixed 1024; predict + `cutout` + transparent-PNG save + zero-shot `val` (MAE/S-measure); inference-only in v1; fixed-resolution ONNX (opset 19 DeformConv) |
| `feynobg`   | `("matte",)`                        | matte  | FeyNobg background removal (BiRefNet architecture, deeper stage 3); size `l`, fixed 1024; same matte surface as birefnet; inference-only; fp8/nvfp4 pre-quantized checkpoints published on HF |
| `ppocr`     | `("ocr",)`                          | ocr    | PP-OCRv5 two-stage text detection + recognition (zh/zh-TW/en/ja/pinyin, one dictionary); sizes `t` (mobile)/`l` (server); one composite checkpoint bundles det.* and rec.* plus the charset; predict + `val` (hmean / e2e F1 / 1-NED); inference-only; export unsupported (two-network pipeline) |
| `realesrgan` | `("restore",)`                     | restore | Real-ESRGAN super-resolution; sizes `x4`/`x2`/`x4t`; native predict at original resolution, `Results.restored` is `restore_scale` x the input; optional seam-free tiling (`predict(..., tile=512)`); inference + PSNR/SSIM `val` only (no training); dynamic-H/W ONNX |
| `swinir`    | `("restore",)`                     | restore | SwinIR transformer super-resolution; sizes `s`/`m`/`l`, all 4x; native predict at original resolution with window padding; optional tiled inference; inference + PSNR/SSIM `val` only (no training); fixed-resolution ONNX |
| `mobilenetv4` | `("classify",)`                | classify | MobileNetV4-conv image classifier; s/m/l at 224/224/256; predict + top-1/top-5 `val` + CE fine-tune train + ONNX |
| `convnext`  | `("classify",)`                | classify | ConvNeXt V1 image classifier; t/s/b at 224; predict + top-1/top-5 `val` + CE fine-tune train + ONNX |
| `efficientnetv2` | `("classify",)`             | classify | EfficientNetV2-base image classifier; b0/b1/b2/b3 at 224/240/260/300; predict + top-1/top-5 `val` + CE fine-tune train + ONNX |
| `resnet`    | `("classify",)`             | classify | vanilla ResNet image classifier (v1.5); 18/34/50/101 at 224; predict + top-1/top-5 `val` + CE fine-tune train + ONNX |
| `clip`      | `("classify", "embed")`     | classify | shared two-tower `-cls` weights; zero-shot classify or whole-image/text embeddings in one space |
| `siglip2`   | `("classify", "embed")`     | classify | shared two-tower `-cls` weights; zero-shot classify or multilingual whole-image/text embeddings in one space |
| `facerec`   | `("embed",)`                | embed | two-stage face-region embeddings; rows align with face boxes; inference-only |

Families that override `SUPPORTED_TASKS` also declare `TASK_INPUT_SIZES` so
each task can use a different per-size input resolution (relevant for RF-DETR).
LibreFOMO uses `SUPPORTED_TASKS = ("point",)`. No pretrained weights are auto-downloadable for this family; see `libreyolo/models/fomo/model.py`. Other point-localization families must opt into `SUPPORTED_TASKS = ("point",)` or an equivalent multi-task tuple.

`segformer` weights are auto-downloadable but **non-commercial**. All six sizes
are converted from NVIDIA's ADE20K checkpoints, whose license permits
redistribution (with the license attached) but restricts use to research or
evaluation only. LibreYOLO hosts them and prints that restriction before each
download, exactly as it does for the VisDrone research-preview weights. They are
not covered by LibreYOLO's permissive license; train from scratch for
unrestricted use. See `libreyolo/models/segformer/NOTICE`.

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
LibreSSD300.pt
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
# Either artifact may be loaded with task="embed"; the head is bypassed and
# the final 384-d DINOv2-S CLS token is returned. No duplicate -embed weights.

# eomt - semantic (ADE20K), instance segmentation (COCO), and panoptic (COCO things+stuff)
LibreEoMTl-sem.pt          # EoMT-L, ADE20K 150-class semantic, DINOv2 backbone, 512px
LibreEoMTl-seg.pt          # EoMT-L, COCO 80-class instance segment, DINOv2 backbone, 640px
LibreEoMTl-seg-1280.pt     # EoMT-L, COCO 80-class instance segment, DINOv2 backbone, 1280px
LibreEoMTs-panoptic.pt     # EoMT-S, COCO 133-class panoptic (80 things + 53 stuff), DINOv2 backbone, 640px
LibreEoMTb-panoptic.pt     # EoMT-B, COCO 133-class panoptic (80 things + 53 stuff), DINOv2 backbone, 640px
LibreEoMTl-panoptic.pt     # EoMT-L, COCO 133-class panoptic (80 things + 53 stuff), DINOv2 backbone, 640px
# NOTE: "-panoptic" is a first-class task suffix (see the task table above), so
# these load with task="panoptic" and nc=133 (things 0-79, stuff 80-132).
# Upstream ships no COCO *instance* checkpoint at s/b; those sizes are panoptic
# only. Slicing a panoptic head down to the 80 things would discard the 53
# stuff classes, so LibreYOLO does not publish LibreEoMT{s,b}-seg.

# pidnet - real-time semantic segmentation
LibrePIDNets-sem.pt        # PIDNet-S, Cityscapes 19-class semantic
LibrePIDNetm-sem.pt        # PIDNet-M, Cityscapes 19-class semantic
LibrePIDNetl-sem.pt        # PIDNet-L, Cityscapes 19-class semantic

# segformer — MiT-b0..b5 ADE20K semantic segmentation; weights are
# NON-COMMERCIAL (see the note above)
LibreSegformerb0-sem.pt
LibreSegformerb1-sem.pt
LibreSegformerb2-sem.pt
LibreSegformerb3-sem.pt
LibreSegformerb4-sem.pt
LibreSegformerb5-sem.pt

# lingbotvision — LingBot-Vision ViT + dense head, ADE20K semantic
# segmentation (Apache-2.0 backbone; head trained by LibreYOLO)
LibreLingBotVisions-sem.pt
LibreLingBotVisionb-sem.pt
LibreLingBotVisionl-sem.pt

# ec — detect + pose + segment
LibreECs.pt             # detect (default)
LibreECs-pose.pt        # pose
LibreECs-seg.pt         # segment

# depth_anything — Depth Anything V2 (depth-only)
LibreDepthAnythingV2s-depth.pt   # ViT-S (Apache-2.0 weights)
LibreDepthAnythingV2b-depth.pt   # ViT-B (CC-BY-NC-4.0 weights)
LibreDepthAnythingV2l-depth.pt   # ViT-L (CC-BY-NC-4.0 weights)
LibreDepthAnythingV2g-depth.pt   # ViT-G (CC-BY-NC-4.0 weights)

# depth_anything3 — Depth Anything 3 mono (recommended depth quality default)
LibreDepthAnything3l-depth.pt    # DA3MONO-LARGE ViT-L (Apache-2.0 weights)

# zipdepth — ZipDepth lightweight CNN (depth-only, MIT weights)
LibreZipDepthb-depth.pt          # base, convex upsampling (GPU/CPU default)
LibreZipDepthbnpu-depth.pt       # base, unfold-free upsampling (NPU/edge export)

# moge2 — MoGe-2 surface normals (normal-only, MIT weights)
LibreMoGe2s-normal.pt            # ViT-S/14, native short side 518
LibreMoGe2b-normal.pt            # ViT-B/14, native short side 518
LibreMoGe2l-normal.pt            # ViT-L/14, native short side 518, quality default

# edge specialists - architecture code is MIT; no released weights are mirrored
LibreTEEDt-edge.pt               # tiny TEED, local compatible checkpoint
LibreDexiNedb-edge.pt            # base DexiNed, local compatible checkpoint

# nafnet — NAFNet restoration (restore-only)
LibreNAFNets-restore.pt
LibreNAFNetl-restore.pt

# swinir: SwinIR super-resolution (restore-only, all 4x)
LibreSwinIRs-restore.pt
LibreSwinIRm-restore.pt
LibreSwinIRl-restore.pt

# birefnet — BiRefNet background removal (matte-only)
LibreBiRefNett-matte.pt          # BiRefNet_lite (Swin-T tier)
LibreBiRefNetl-matte.pt          # BiRefNet general (Swin-L tier), MIT weights

# feynobg — FeyNobg background removal (matte-only)
LibreFeyNobgl-matte.pt           # FeyNobg (Swin-L tier, 24 stage-3 blocks), Apache-2.0 weights
LibreFeyNobgl-matte-fp16.pt      # half-precision cast (HF only, pass path as weights; GPU-oriented)
LibreFeyNobgl-matte-fp8.pt       # pre-quantized fp8 variant (HF only, pass path as weights; native fp8 tensor-core execution on Ada/Hopper/Blackwell)

# ppocr — PP-OCRv5 text detection + recognition (ocr-only)
LibrePPOCRt-ocr.pt               # mobile det + mobile rec (CPU tier), Apache-2.0 weights
LibrePPOCRl-ocr.pt               # server det + server rec (quality tier), Apache-2.0 weights
```

### Zero-shot / open-vocabulary classify (inference-only)

```text
# clip — CLIP zero-shot, open-vocabulary (set_classes); no fixed label set.
# Defaults to ImageNet-1k classify. Use the same -cls artifact with
# task="embed"; no duplicate -embed checkpoint is published.
LibreCLIPb32-cls.pt       # OpenCLIP ViT-B/32, LAION-2B (MIT weights)
LibreCLIPb16-cls.pt       # OpenCLIP ViT-B/16, LAION-2B (MIT weights)
LibreCLIPl14-cls.pt       # OpenCLIP ViT-L/14, LAION-2B (config + converter ready; weights not yet published)

# siglip2: SigLIP 2 zero-shot classify or paired image/text embedding.
# SentencePiece tokenizer, sigmoid-native scoring with a multi_label option.
# Use the same -cls artifact with task="embed"; size codes bake in resolution.
LibreSigLIP2b16-cls.pt    # google/siglip2-base-patch16-256 (Apache-2.0 weights), 256 px
LibreSigLIP2so400m-cls.pt # google/siglip2-so400m-patch14-384 (Apache-2.0 weights), 384 px
```

### VLM analysis (external snapshot tier)

```text
# libremodus - external Hugging Face snapshot, no .pt checkpoint filename.
# The canonical factory alias resolves this directory without renaming or
# mirroring the upstream files.
libremodus-14b-a7b -> weights/LibreMODUS14b-a7b/
```

LibreMODUS is resolved by `LibreVLM("libremodus-14b-a7b")`, not by the
single-file `LibreYOLO(...)` checkpoint factory. A local path may point directly
at an upstream snapshot directory instead. See
[`libremodus.md`](libremodus.md).

### Open-vocabulary detection (inference-only snapshot tier)

```text
# grounding_dino - Hugging Face snapshot, no .pt checkpoint filename
weights/LibreGroundingDINOt/
weights/LibreGroundingDINOb/

# owlv2 - Hugging Face snapshot, no .pt checkpoint filename
weights/LibreOWLv2b16/
weights/LibreOWLv2l14/

# omdet_turbo - Hugging Face snapshot, no .pt checkpoint filename
weights/LibreOMDetTurbot/

# ov_deim - Hugging Face snapshot, no .pt checkpoint filename
weights/LibreOVDEIMs/
weights/LibreOVDEIMm/
weights/LibreOVDEIMl/
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

with `task_suffixes` derived from `WEIGHT_TASKS` when a family declares it,
otherwise from `SUPPORTED_TASKS`, via
`libreyolo.tasks.task_suffix_pattern`. `WEIGHT_TASKS` is used only when
multiple runtime tasks share one artifact (CLIP, SigLIP2, and DINOv2 embed);
it prevents a nonexistent duplicate checkpoint suffix from being advertised.
This is the single source of truth for parsing a filename back into
`(family, size, task, variant)`.

The `variant` group only exists for families that declare `WEIGHT_VARIANTS`, a
dataset suffix for published checkpoints trained on a non-default dataset.
Example: `yolo9_p2` declares `("visdrone",)`, so `LibreYOLO9P2s-visdrone.pt`
resolves the Hugging Face repo `LibreYOLO/LibreYOLO9P2s-visdrone` (a research
preview under VisDrone's CC BY-NC-SA license, announced by a download notice).
Plain COCO-default weights never carry a variant suffix.
