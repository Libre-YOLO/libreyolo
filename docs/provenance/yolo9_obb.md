# YOLO9 OBB design and provenance

Status: approved for implementation. Branch: `yolo9-obb-v2`. Target: `dev`.
Closes the YOLO9 half of issue #320.

## 0. Provenance rules for this plan

Every design element below carries a `Source:` line. Allowed sources:

1. **In-tree code on current dev** (commit `f039d382` or later), cited by file
   and line. The live OBB groundwork (task registry, dataset, augmentations,
   rotated NMS, validator, results container, CLI/export plumbing) was audited
   on 2026-07-14 and is used as-is.
2. **Published papers**, cited by name. Only the math is taken, transcribed
   into formulas in this plan; no code from paper repos is read or ported
   unless the repo is explicitly listed under 3.
3. **Permissively licensed repositories** already attributed in
   `THIRD_PARTY_NOTICES.txt` (MultimediaTechLab/YOLO, MIT; Deci-AI
   super-gradients, Apache-2.0). No new third-party code is ported in this
   plan.
4. **Classical mathematics** (trigonometric identities, multivariate normal
   formulas), cited as such.
5. **Original design decisions**, explicitly marked `Source: original`.

Banned sources: the AGPL detection framework and any code derived from it;
the removed June 2026 YOLO9 task heads (`f2a22821`, `2d5fe2ba` history) and
the withdrawn PR #606 branch. Two exceptions from that branch are re-landed
because they were authored in-repo before any AGPL exposure and are
self-contained: the OBB validator spawn-pickling fix and the export
raw-parity test skeleton (both listed in section 8).

Implementation discipline: the implementer works by transcribing the formulas
and structures written in this plan, not from memory of any external
implementation. After implementation, the diff is checked against the
divergence checklist in section 9.

## 1. Goal and scope

- New task `obb` for the `yolo9` family (sizes t/s/m/c): training, validation,
  prediction, ONNX/TorchScript export.
- `yolo9_e2e` and `yolo9_p2` stay detect-only (out of scope).
- User-facing surface follows the existing task conventions: `task=obb`,
  `-obb` weight suffix, `Results.obb` container, OBB metrics. All of these
  already exist on dev and are consumed today by the RF-DETR OBB task.

## 2. Current-code inventory (what this plan builds on, all live on dev)

| Surface | Location | Status |
|---|---|---|
| Task registry: `obb` name, aliases, `-obb` suffix | `libreyolo/tasks.py` | live |
| YOLO OBB label parsing (9-field rows), canonical `xywhr`, proxy conversion | `libreyolo/data/obb.py` (`parse_yolo_obb_label_line`, `canonicalize_xywhr`, `corners_to_xywhr`, `xywhr_to_proxy_xyxy`, `xywhr_iou`) | live |
| Dataset loading: `load_obb`, rows emitted as `[x1, y1, x2, y2, cls, theta]` (pixel proxy box + radians) | `libreyolo/data/dataset.py` (~line 448) | live |
| Train transforms: 6-column labels, angle-aware hflip/vflip/rot90, mosaic/mixup with the corner-aware affine (`apply_affine_to_obb`; shear and perspective suppressed) | `libreyolo/data/augment/yolo9.py`, `libreyolo/data/augment/geometry.py` | live |
| Training targets contract: `[class, x1, y1, x2, y2, theta]`, normalized | `docs/dataset_schema.md` (OBB section) | live |
| Inference postprocess contract: predictions `(B, 4+1+nc, A)` with rows `[proxy xyxy in input pixels, theta, sigmoided class scores]`, dict flag `obb: True`; exact rotated NMS | `libreyolo/postprocess/yolo9.py:176-321` | live |
| Rotated IoU: exact, vectorized (axis-aligned envelope gate, then convex-polygon intersection); the OpenCV scalar form remains the reference | `libreyolo/utils/box_ops.py` (`rotated_iou_matrix`, `rotated_iou_pairwise`), `libreyolo/data/obb.py:xywhr_iou` | live |
| OBB validator (rotated-IoU AP) | `libreyolo/validation/obb_validator.py` | live |
| Results container `OBB` (xywhr/conf/cls) and drawing | `libreyolo/utils/results.py:846`, `libreyolo/utils/drawing.py:draw_obb` | live |
| CLI train/val/predict OBB plumbing, backends, export metadata | `libreyolo/cli/commands/*.py`, `libreyolo/backends/base.py`, `libreyolo/export/exporter.py` | live |
| Detect head: towers `cv2`/`cv3`, DFL integral, grid helpers `_anchor_grid`/`_grid`/`_decode_inference` | `libreyolo/models/yolo9/nn.py` | live (`f039d382`) |
| Training decode and assignment: `Vec2Box`, `BoxMatcher`, `BCELoss`, `DFLoss` | `libreyolo/models/yolo9/loss.py` | live |

The plan adds exactly four new pieces of substance: an oriented head variant,
a rotated matcher variant, an oriented loss, and the task wiring.

## 3. Geometry representation

### 3.1 Internal box parameterization (unchanged)

The rotated box is carried as the canonical `xywhr` of `data/obb.py`: center
`(cx, cy)`, long side `w`, short side `h` (so `w >= h`), angle
`theta in [-pi/2, pi/2)` measured as the rotation of the `w` side. For the
regression head the box is represented by its **proxy rectangle**: the
axis-aligned rectangle of size `(w, h)` centered at `(cx, cy)`, encoded as
LTRB distances from the anchor and regressed through the existing DFL
machinery, exactly as the dataset and postprocess already expect.

Source: in-tree contract (`data/obb.py`, `docs/dataset_schema.md`,
`postprocess/yolo9.py`).

### 3.2 Orientation encoding: the double-angle vector

The head does not regress `theta` as a scalar. It predicts a 2-channel vector

```
v = (v_c, v_s)   with target   t = w_ar * (cos 2*theta, sin 2*theta)
```

Decode at inference: `theta_hat = 0.5 * atan2(v_s, v_c)`, which lands in
`(-pi/2, pi/2]`, matching the canonical range.

Why the double angle: a rectangle is invariant under rotation by pi, and
`(cos 2t, sin 2t)` is exactly pi-periodic, so the two equivalent descriptions
of the same box map to the same target. The angular boundary discontinuity
(a box at 89 degrees vs -89 degrees) disappears by construction instead of
being patched in the loss.

`w_ar` is an aspect-ratio weight (section 5.3): elongated boxes carry a
full-length target vector, near-squares carry a near-zero one, because a
square's orientation is undefined under the canonical convention. As a side
effect the norm of the predicted vector becomes a learned "orientedness"
signal (not exposed in `Results` for now; the container contract is frozen).

Sources: trigonometric double-angle identities (classical); encoding angles
in trigonometric/phase components for rotated detection appears in the
Phase-Shifting Coder paper (Yu and Da, CVPR 2023) and Circular Smooth Label
(Yang and Yan, ECCV 2020) motivates the square-ambiguity treatment. The
specific combination (2-channel double-angle vector, aspect-weighted target
length, norm as orientedness) is `Source: original`.

### 3.3 Gaussian form of a rotated box

For the joint loss the box `(cx, cy, w, h, theta)` maps to a 2D Gaussian
`N(mu, Sigma)` with `mu = (cx, cy)` and

```
Sigma = R(theta) * diag(w^2/4, h^2/4) * R(theta)^T
```

Written with half-angle identities (`cos^2 t = (1 + cos 2t)/2` etc.) the
covariance components are **linear in the double-angle vector**:

```
p = (w^2 + h^2) / 8        q = (w^2 - h^2) / 8
a = p + q * cos 2*theta    (var x)
b = p - q * cos 2*theta    (var y)
c = q * sin 2*theta        (cov xy)
```

This is the key structural fit of the design: the head's raw `(v_c, v_s)`,
normalized to unit length, plugs directly into `(a, b, c)`. The training loss
never computes `atan2`; it is smooth everywhere, including at the square
degeneracy where `q -> 0` erases the angle from the Gaussian entirely, which
is the mathematically correct behavior.

Sources: box-to-Gaussian mapping from the Gaussian Wasserstein Distance paper
(Yang et al., ICML 2021) and the ProbIoU paper (Murrugarra-Llerena et al.,
2021); the half-angle expansion making `Sigma` linear in the double-angle
vector is `Source: classical identities, derivation original to this plan`.

## 4. Head: `OrientedDDetect(DDetect)`

File: `libreyolo/models/yolo9/nn.py`.

### 4.1 Structure

- Inherits everything from the current `DDetect` (`f039d382`): towers,
  bias init, `_anchor_grid`, `_grid`, `_decode_bboxes`.
- Adds one `nn.ModuleList` named `ang`: per scale, a single
  `nn.Conv2d(box_hidden, 2, 1)` applied to the output of the **first box
  tower conv** (`cv2[i][0]`), i.e. the ungrouped hidden geometry features.
  There is no separate multi-layer angle tower; orientation is read from the
  same representation that produces the distances. Bias initialized to zero
  (weights default init) so the initial orientation claim is near-neutral.
- Checkpoint keys: `head.ang.{i}.weight/bias`. Task inference reads the
  presence of `head.ang.` keys. The `cv2`/`cv3`/`dfl` layout is untouched, so
  detect-to-OBB transfer partial-loads the entire detect head and only the
  `ang` convs start fresh.

Source: `DDetect` is in-tree; the shared-geometry tap (angle from the box
tower's hidden features via one 1x1 conv) is `Source: original`.

### 4.2 Forward

Training with targets: per-level towers run once; `preds` (the usual
`cat(cv2_i, cv3_i)` maps) plus per-level angle maps `ang_i(hidden_i)` of
shape `(B, 2, H, W)` go to `YOLO9OrientedLoss` (section 5). Training targets
are `[B, N, 6]`; a shape guard raises on 5-column labels.

Inference: reuse the base per-level decode (`_decode_inference` from
`f039d382`, our own expression) with one extension: flatten the per-level
angle maps to `(B, 2, A)`, compute `theta_hat = 0.5 * atan2(v_s, v_c)` as a
`(B, 1, A)` row, and emit

```
y = cat([proxy_boxes_pixels, theta_hat, scores.sigmoid()], dim=1)   # (B, 4+1+nc, A)
```

which is exactly the live postprocess contract. Export mode returns `y`
alone. `LibreYOLO9Model.forward` sets `result["obb"] = True` when built with
`obb=True` so the postprocess selects the rotated branch.

Source: decode structure in-tree (`f039d382`); output layout dictated by
in-tree `postprocess/yolo9.py`; `atan2` decode classical.

## 5. Loss: `YOLO9OrientedLoss(YOLO9Loss)`

File: `libreyolo/models/yolo9/loss.py`. Components and weighting follow the
in-tree normalization scheme (`cls_norm`, `box_norm`, weighted sums) of
`YOLO9Loss` verbatim, with the CIoU box loss replaced by the Gaussian loss:

```
total = box_weight * L_kld  +  dfl_weight * L_dfl  +  cls_weight * L_bce  +  vec_weight * L_vec
```

Defaults: `box_weight=7.5`, `dfl_weight=1.5`, `cls_weight=0.5` (in-tree
values), `vec_weight=0.5` (original, tunable). Loss dict keys mirror the
detect loss plus `angle_loss`/`angle` so the trainer logging picks them up.

### 5.1 Assignment: `RotatedBoxMatcher(BoxMatcher)`

- `get_valid_matrix` unchanged: it gates which anchors can represent the
  **proxy** LTRB distances inside `reg_max` bins, which is precisely the
  quantity DFL regresses. Source: in-tree.
- Similarity replaces the CIoU matrix: the Bhattacharyya coefficient between
  the target Gaussian and the (detached) predicted Gaussian,

```
Sigma_m = (Sigma_p + Sigma_t) / 2,   d = mu_p - mu_t
B_D = (1/8) * d^T Sigma_m^{-1} d + (1/2) * ln( det Sigma_m / sqrt(det Sigma_p * det Sigma_t) )
score = exp(-B_D)  in (0, 1]
```

  computed in closed 2x2 form (`det = a*b - c^2` clamped at `1e-7`,
  `Sigma^{-1} = [[b, -c], [-c, a]] / det`). The score is IoU-like (1 at
  identity), so it plugs into the existing task-aligned target
  `score^iou_factor * cls^cls_factor` unchanged.
- `__call__` accepts 6-column targets, splits `[1, 4, 1]`, and additionally
  gathers the matched `theta` (and matched `w, h` implied by the proxy box)
  per anchor, following the same gather pattern the parent uses for boxes.

Sources: matcher body in-tree (MIT-derived); Bhattacharyya distance between
Gaussians classical statistics; its use as a rotated-box similarity is the
ProbIoU paper. Applying it as the TAL metric inside our matcher is
`Source: original` (consequence of reusing in-tree assignment).

### 5.2 Gaussian regression loss (KLD)

For each matched anchor, target Gaussian `(mu_t, Sigma_t)` from the matched
proxy box and `theta_t`; predicted Gaussian `(mu_p, Sigma_p)` from the
decoded proxy box and the **normalized** predicted vector
`(c_hat, s_hat) = v / max(||v||, 1e-6)` via section 3.3. Kullback-Leibler
divergence in the direction that only ever inverts the exact target
covariance (numerically safe; predictions never get inverted):

```
D = 1/2 * [ tr(Sigma_t^{-1} Sigma_p) + d^T Sigma_t^{-1} d - 2 + ln(det Sigma_t / det Sigma_p) ]
L_kld = 1 - 1 / (1 + ln(1 + D))
```

`L_kld in [0, 1)` behaves like an IoU-style loss and takes the in-tree
`box_norm`/`cls_norm` weighting. Computed in fp32 under AMP (in-tree
precedent: the fp32-loss guard added for picodet/rtmdet).

Sources: multivariate normal KL closed form classical; using KL between box
Gaussians with a `1 - 1/(tau + f(D))` normalization is the KLD paper (Yang et
al., NeurIPS 2021); the inversion-direction choice is
`Source: original (numerical-stability argument stated above)`.

### 5.3 Orientation vector loss

```
w_ar = clamp( ln(w_t / h_t) / ln(3), 0, 1 )        (w_t >= h_t canonical)
t = w_ar * (cos 2*theta_t, sin 2*theta_t)
L_vec = mean over matched anchors of || v - t ||^2, weighted like the box terms
```

Near-squares (`w_t ~ h_t`) get `w_ar ~ 0`: their target vector is zero, the
network learns to output short vectors there, and no gradient fights the
undefined orientation. Aspect 3 and beyond gets full supervision. The KLD
term already couples orientation for everything in between, so `L_vec` is an
auxiliary sharpener, not the main angle signal.

Source: original (motivated by the square-ambiguity discussion in the CSL and
KLD papers; threshold `ln(3)` is a tunable original default).

### 5.4 What is deliberately absent

- No tiny-box filtering inside the loss: degenerate rows are already rejected
  or clamped at parse/transform time in-tree.
- No separate scalar-angle regression, no periodic trig penalty on a scalar
  angle, no angle classification bins in v1 (CSL-style distributional angle
  is noted as a possible v2 under the same DFL philosophy).
- No changes to `BoxMatcher`, `Vec2Box`, `DFLoss`, `BCELoss` for the detect
  path.

## 6. Task wiring

All following the current file conventions on dev (each item is glue in the
style of the surrounding code; `Source: in-tree patterns`):

- `nn.LibreYOLO9Model`: `obb=False` kwarg; selects `OrientedDDetect`;
  inference dict gains `"obb": True`.
- `model.LibreYOLO9`: `SUPPORTED_TASKS = ("detect", "obb")`,
  `TASK_INPUT_SIZES["obb"]`, `_is_obb` property, `_init_model(obb=...)`,
  `detect_checkpoint_task` returns `"obb"` iff `head.ang.` keys present,
  `_validate_loaded_state_dict_for_task` requires `head.ang.` for `task=obb`
  direct loads (transfer path exempt). No extra loss-cache resets needed: the
  oriented loss lives in the standard `_loss_fn` slot that the rebuild
  helpers already clear.
- `trainer.YOLO9Trainer.create_transforms`: `output_label_dim=6` when the
  wrapper task is `obb`; `get_loss_components` logs the `angle` component.
- Validator: re-land the `_OBBValPreprocessor.__getattr__` spawn-pickling fix
  (raise `AttributeError` for `base_preprocessor`) plus its regression test.
- Docs: nomenclature family table row, dataset schema pointer already exists.
- Export support (`libreyolo/export/support.py`): `yolo9`/`obb` validated for
  `onnx`, `torchscript` with the raw-parity test; regenerate inventory and
  tables with the existing tools.

## 7. Hyperparameters and numerics

| Knob | Default | Rationale |
|---|---|---|
| `box_weight` (KLD) | 7.5 | in-tree box slot, `L_kld` is IoU-like |
| `dfl_weight` | 1.5 | in-tree |
| `cls_weight` | 0.5 | in-tree |
| `vec_weight` | 0.5 | original default, aux term |
| aspect threshold | `ln(3)` | full angle supervision from aspect 3 |
| det clamp | `1e-7` | 2x2 inversions |
| vector norm floor | `1e-6` | normalization guard |
| KLD normalization | `1 - 1/(1 + ln(1+D))` | KLD paper form |
| AMP | Gaussian terms in fp32 | in-tree precedent |

## 8. Verification ladder

1. **Unit tests** (`tests/unit/test_yolo9_layers.py` + factory/CLI files):
   head forward shapes `(B, 4+1+nc, A)`, theta row bounded in
   `(-pi/2, pi/2]`, export single-tensor mode, encode/decode roundtrip
   `theta -> v -> theta` over a grid of angles, Gaussian helper identities
   (`L_kld(x, x) = 0`, Bhattacharyya score 1 at identity, monotone decrease
   under center offset), square-invariance (rotating a square target changes
   the loss by < 1e-5), 6-column target guard, gradient-flow to `ang`,
   detect-to-OBB transfer stats, checkpoint task inference from `head.ang.`,
   scratch checkpoint round-trip via the factory, CLI dry-run and
   task-architecture selection, validator pickle roundtrip.
2. **Export parity**: ONNX + TorchScript raw outputs vs the eager head
   (adapting the pre-exposure parity test skeleton from 2026-07-14).
3. **Synthetic overfit smoke** (same generator as the 2026-07-14 run:
   rotated bars, 48 train / 12 val, imgsz 320, yolo9-t, detect transfer,
   60 epochs): quality bar is the June design's measured result,
   rotated-IoU mAP50 >= 0.87 on val, angles visually aligned.
4. **In-training validation** on Windows spawn workers (exercises the
   pickling fix), `best.pt` selected on OBB mAP50-95.
5. **Real-data run** (follow-up, not gating this PR): DOTA-v1.0 split at
   1024, compare against the literature deltas for Gaussian losses; weights
   publication is a separate maintainer decision (dataset is academic-use).

## 9. Divergence checklist (checked against the final diff)

The AGPL implementation is characterized by: a separate multi-layer angle
tower on the FPN features with a specific width formula; a single
sigmoid-scalar angle mapped by a fixed affine to a three-quarter-pi range; a
rotated task-aligned assigner ported alongside the loss; a Gaussian-IoU
fast-NMS; corner conversions with while-loop range normalization. The
implementation of this plan must contain none of those: angle as a 2-channel
double-angle vector from a single 1x1 conv on the box tower's hidden
features; `atan2` decode; assignment through the in-tree `BoxMatcher` with a
Bhattacharyya score; the existing exact OpenCV NMS; covariance built linearly
from the vector with no angle normalization code at all. Any overlap found in
review is treated as a bug.

## 10. Implementation map

| File | Change |
|---|---|
| `libreyolo/models/yolo9/nn.py` | `OrientedDDetect`, `obb` kwarg in `LibreYOLO9Model`, obb inference dict flag |
| `libreyolo/models/yolo9/loss.py` | Gaussian helpers (`_rbox_gaussian`, `_gaussian_kld`, `_bhattacharyya_score`), `RotatedBoxMatcher`, `YOLO9OrientedLoss` |
| `libreyolo/models/yolo9/model.py` | task wiring per section 6 |
| `libreyolo/models/yolo9/trainer.py` | 6-column labels, angle logging |
| `libreyolo/validation/obb_validator.py` | pickling fix; per-class cached vectorized IoU |
| `libreyolo/utils/box_ops.py` | `rotated_iou_pairwise` / `rotated_iou_matrix` (section 11) |
| `libreyolo/postprocess/yolo9.py` | rotated NMS rebuilt on the vectorized IoU (section 11) |
| `libreyolo/data/augment/geometry.py` | `apply_affine_to_obb`, `obb` flag on `random_affine` (section 12) |
| `libreyolo/data/augment/yolo9.py`, `libreyolo/training/trainer.py` | mosaic/mixup enabled for OBB (section 12) |
| `libreyolo/export/support.py` + generated tables | obb rows |
| `docs/nomenclature.md` | family table + filename example |
| `tests/unit/...` | per section 8 |

## 11. Vectorized rotated IoU

Rotated NMS evaluated one OpenCV polygon intersection per candidate pair from
a Python loop, and the OBB validator repeated the same per-pair calls once per
mAP threshold. Both now go through an exact vectorized IoU.

Two convex polygons intersect in the convex hull of: each polygon's vertices
that lie inside the other, plus every edge-edge crossing. For two rectangles
that is at most eight of twenty-four candidate points; they are ordered by
angle about their centroid and the area follows from the shoelace formula.
Pairs are gated first on their axis-aligned envelopes, which cannot miss when
the rotated boxes overlap, so on dense imagery only a small fraction of the
matrix reaches the polygon stage.

Greedy suppression order, the threshold comparison and the resulting keep
indices are unchanged; the tests assert bit-identical output against the
OpenCV implementation, which remains the reference.

Source: classical computational geometry (convex-polygon intersection,
shoelace area, axis-aligned rejection). `Source: original` for the
formulation and for reusing the same primitive in both NMS and validation.

## 12. Orientation-aware mosaic

Mosaic was disabled for OBB because the shared affine warps the corners of the
xyxy columns and rewrites them as an envelope, which for the OBB proxy box
(section 3.1) both discards the rotation and corrupts the side lengths.

A rectangle stays a rectangle under rotation, uniform scale and translation,
so the OBB affine is exact: the center rides the matrix, and the two side
directions are carried through the matrix's linear part, which yields the new
orientation and the new side lengths without reading an angle out of the
matrix (and so without depending on any sign or handedness convention).
Shear and perspective are suppressed for OBB, as neither maps a rectangle to a
rectangle. Mosaic tile placement needed no change: uniform scale and
translation of all four proxy coordinates scale the center and the sides
consistently and cannot rotate the box.

Source: classical linear algebra. `Source: original` for the proxy-box
transform and for the decision to suppress shear rather than refit it.

Estimated new code: ~450 lines library, ~400 lines tests.
