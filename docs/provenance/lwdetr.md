# lwdetr

- **LibreYOLO module:** `libreyolo/models/lwdetr/` (`nn.py` holds the ported
  architecture), postprocessing in `libreyolo/postprocess/lwdetr.py`, converter
  at `weights/convert_lwdetr_weights.py`.
- **Upstream:** https://github.com/Atten4Vis/LW-DETR (paper: arXiv 2406.03459,
  "LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection"),
  Baidu / Atten4Vis, 2024.
- **Upstream code license:** Apache-2.0 (`LICENSE` at the repository root,
  verified 2026-08-01). Source file headers carry
  `Copyright (c) 2024 Baidu. All Rights Reserved.` and preserve the
  Conditional DETR / DETR / Deformable DETR / ViTDet lineage; those headers are
  reproduced in the ported modules.
- **Upstream weights license:** Apache-2.0, declared in the model-card metadata
  of https://huggingface.co/xbsu/LW-DETR (verified 2026-08-01). Rehosted under
  the `LibreYOLO/` org per the port skill's license table.
- **Verification status:** ported and verified 2026-08-01. Numerical parity
  against the official implementation is **exact** (`max_abs_diff == 0.0` on
  both `pred_logits` and `pred_boxes`) for all five released sizes; see
  "Parity evidence" below.

## Source checkpoints

Downloaded 2026-08-01 from
`https://huggingface.co/xbsu/LW-DETR/resolve/main/pretrain_weights/<file>`:

| Size | Upstream file | SHA-256 | COCO AP (upstream README) |
|---|---|---|---|
| `t` | `LWDETR_tiny_60e_coco.pth`   | `382431625cf2aaf81771d8d7f708d9742dc0a4447044c09201b3066bab97e43c` | 42.6 |
| `s` | `LWDETR_small_60e_coco.pth`  | `b5f07c9d73f1a9ac1d8c26e184202c28ca9e01b458f61002789a8843c4591331` | 48.0 |
| `m` | `LWDETR_medium_60e_coco.pth` | `18675ef2d23c7490bf977fc1dccef9f317257c328809de68bbe4f04a7ef89054` | 52.5 |
| `l` | `LWDETR_large_60e_coco.pth`  | `5bef27a8fbef37f1d7b5a6c6669a09ce182a631c699ba2ddf63d0c4af22d4f36` | 56.1 |
| `x` | `LWDETR_xlarge_60e_coco.pth` | `5f116ad0752fdbc2fa1e49b54ab3349594f0091b1834383db0892ebda644ab69` | 58.3 |

Each file is `{"model", "optimizer", "lr_scheduler", "epoch", "args"}`. The
releases were produced with `--use_ema`, and upstream's save path writes
`ema_m.module.state_dict()` into the `model` key (`main.py`), so the `model`
entry already holds the EMA weights behind the published numbers. The stored
`args` namespace was cross-checked against `scripts/lwdetr_<size>_coco_eval.sh`
and is the authority for the per-size configuration in `nn.py`.

## Relationship to the RF-DETR family

`libreyolo/models/rfdetr/` vendors the **Roboflow-modified** descendant of this
architecture, not upstream LW-DETR. The two have diverged: RF-DETR swapped the
plain-ViT encoder for DINOv2, dropped the projector's wide-input channel-trim
branch (which LW-DETR-xlarge does use), and added keypoint and segmentation
heads. `models/lwdetr/nn.py` is therefore an independent port from the
Atten4Vis source, not a subclass — the parity target is Atten4Vis, never
Roboflow. Bidirectional `can_load` rejection between the two families is
covered by `tests/unit/test_lwdetr_can_load.py`.

## Ported surface

| LibreYOLO | Upstream source |
|---|---|
| `nn.py` — `ViT`, `Block`, `Attention`, `PatchEmbed`, `Mlp`, `get_abs_pos` | `models/backbone/vit.py` |
| `nn.py` — `MultiScaleProjector`, `C2f`, `Bottleneck`, `ConvX`, `LayerNorm` | `models/backbone/projector.py` |
| `nn.py` — `Backbone`, `Joiner` | `models/backbone/backbone.py`, `models/backbone/__init__.py` |
| `nn.py` — `PositionEmbeddingSine` | `models/position_encoding.py` |
| `nn.py` — `Transformer`, `TransformerDecoder`, `TransformerDecoderLayer`, `gen_sineembed_for_position`, `gen_encoder_output_proposals` | `models/transformer.py` |
| `nn.py` — `MSDeformAttn`, `ms_deform_attn_core_pytorch` | `models/ops/modules/ms_deform_attn.py`, `models/ops/functions/ms_deform_attn_func.py` |
| `nn.py` — `MultiheadAttention` | `models/attention.py` |
| `nn.py` — `LibreLWDETRModel`, `MLP` | `models/lwdetr.py` (`LWDETR`, `MLP`) |
| `postprocess/lwdetr.py` | `models/lwdetr.py` (`PostProcess`) |
| `models/lwdetr/utils.py` preprocessing | `datasets/coco.py` (`make_coco_transforms_square_div_64`), `datasets/transforms.py` (`SquareResize`) |
| `models/lwdetr/box_ops.py` | `util/box_ops.py` |

Deliberately **not** ported (inference-only scope): `SetCriterion` and the
Group-DETR one-to-many training path, the Hungarian matcher, the IoU-aware /
varifocal / position-supervised classification losses, and the training
dataloader. `LibreLWDETR.train()` raises `NotImplementedError`.

Two upstream dependencies are avoided rather than vendored, so the family adds
no new install requirements: `timm`'s `Mlp` / `DropPath` / `trunc_normal_` (a
local `Mlp`, `nn.Identity` since every released size sets `drop_path_rate=0`,
and `torch.nn.init.trunc_normal_`), and `fairscale`'s `checkpoint_wrapper`
(unused — `use_act_checkpoint=False` throughout). The `MultiScaleDeformableAttention`
CUDA extension is likewise not required: LibreYOLO always takes upstream's own
pure-PyTorch reference core, which upstream itself selects when exporting or
running fp16.

## Parity evidence

`weights/parity_lwdetr.py`, run 2026-08-01 with both env vars set:

```
t/tiny:   missing=0 unexpected=0  pred_logits max_abs_diff=0.0  pred_boxes max_abs_diff=0.0
s/small:  missing=0 unexpected=0  pred_logits max_abs_diff=0.0  pred_boxes max_abs_diff=0.0
m/medium: missing=0 unexpected=0  pred_logits max_abs_diff=0.0  pred_boxes max_abs_diff=0.0
l/large:  missing=0 unexpected=0  pred_logits max_abs_diff=0.0  pred_boxes max_abs_diff=0.0
x/xlarge: missing=0 unexpected=0  pred_logits max_abs_diff=0.0  pred_boxes max_abs_diff=0.0
```

Parameter counts match the paper's table exactly: 12.1M / 14.6M / 28.2M /
46.8M / 118.0M.

Export parity (`t`, same device on both sides, `conf=0.25` and `conf=0.05`):
identical class ids and detection counts, scores within `3e-6` for ONNX and
`6e-8` for TorchScript.

`model.val(data="coco128.yaml")` on the `s` checkpoint reports
`metrics/mAP50-95 = 0.606`.

## Class-index convention

Upstream trains on raw COCO annotations, so the classification head has 91
columns — one per COCO category id (`max_obj_id + 1`), 11 of which are unused.
The converted checkpoints keep that head unchanged; `LibreLWDETR` exposes the
contiguous COCO-80 interface and maps ids through
`libreyolo/utils/coco.py::COCO91_TO_COCO80` at postprocess time, dropping the
unused ids. This matches the existing RF-DETR handling, and exported backends
go through the same map.
