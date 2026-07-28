# Body mesh (`mesh`) task: status and follow-ups

Scope: human body mesh recovery. First family is SAM 3D Body on the MHR body
model, integrated as a **wrapper** rather than a port. Investigation and
implementation 2026-07-28/29.

Status: **shipped and working end to end.** Branch `mesh-body-recovery` off
`dev`.

## Why the field is a licensing minefield

Nearly every mesh model predicts into SMPL, and SMPL is unusable for a
permissive library: the model files are non-commercial and non-redistributable,
the `smplx` PyPI package's *code* carries the same non-commercial license (the
most common misconception in this area), and the license forbids using the
model to train networks for commercial deployment, so it reaches into
checkpoints too. Worse, SMPL is registration-gated, so no library can offer
autodownload for it.

MHR (Apache 2.0, code and assets, ungated) fixes the body-model layer. What it
does not fix is the regressor layer: the only strong MHR regressor is SAM 3D
Body, whose **code** is under the SAM License with military and trade-control
field-of-use clauses. That code cannot enter an MIT tree.

## The resolution: wrap, do not port

`LibreSAM3DBody` calls the upstream package's public API and translates its
output into `Meshes`. LibreYOLO ships no SAM-licensed bytes; the upstream
package is an optional dependency the user installs. The SAM License triggers
on *distributing* SAM Materials, and an adapter distributes none of their code.
A user who never touches the mesh task never encounters those terms.

Weights are separate from code: the SAM License permits redistribution with
passthrough, so the checkpoints are mirrored on the LibreYOLO org with the
license included and a gate that records acceptance.

## What is done

* Task registration: `mesh`, suffix `-mesh`, aliases `body-mesh`, `hmr`,
  `human-mesh-recovery`. `smpl` deliberately not aliased.
* `Meshes` payload, row-aligned with `Results.boxes` as pose keypoints are.
  Body-model-agnostic: `body_model` names the parameterization, counts are read
  from the tensors. `save_obj()` writes Wavefront OBJ.
* MHR body model decoder, fetched from the public Apache release, no
  dependency beyond PyTorch.
* Camera helpers: crop-camera lifting and perspective projection.
* `LibreSAM3DBody` adapter with person-detector chaining
  (`person_boxes=` / `person_detector=`), following the gaze family's pattern.
* Metrics: MPJPE, PA-MPJPE with a reflection-safe Procrustes, PVE.
* `draw_mesh`: renderer-free projected-vertex scatter plus the MHR-70 skeleton.
* Gates: export raises an explicit not-implemented error, validation explains
  the dataset-license situation, TTA is rejected.
* Docs: ADR 0013, nomenclature, checkpoint schema.

### Verified, not assumed

* MHR `model_params` is 204 wide (3 translation, 3 global rotation, 130 body
  pose, 68 bone scales); `betas` 45; `expression` 72; outputs 18439 vertices
  and 127 joints in centimeters, translation entering in decimeters. The
  upstream comment saying 127 is stale. Rest-pose height decodes to 1.727 m.
* Real inference on the sample image: 1 person, 18439 vertices, 70 joints,
  metric depth 4.0-4.9 m in front of the camera, 1.7 s on CUDA.
* Projection parity against upstream: **max 0.0001 px** across all 70 joints,
  confirming the camera-frame contract and focal-length handling.

### Weight mirrors

* `LibreYOLO/LibreSAM3DBodyd3-mesh` (DINOv3 ViT-H/16+, 2109 MB)
* `LibreYOLO/LibreSAM3DBodyh-mesh` (ViT-H, 1691 MB)

Both public with `gated="auto"`: the user accepts the SAM License and attests
they are not in a comprehensively sanctioned jurisdiction, then gets immediate
access. Meta's own gate is manual and can take days; auto records the same
acceptance without the wait. Flip to `manual` with one
`update_repo_settings(gated="manual")` call if a human review step is wanted.

The MHR asset is **not** mirrored: it is Apache 2.0 and public, so LibreYOLO
fetches it from the upstream release and caches it locally.

## Known limitations

* The upstream repository has no packaging metadata, so it cannot be
  `pip install`ed. Users clone it and set `SAM_3D_BODY_PATH`. If they
  restructure, the adapter breaks.
* The upstream estimator moves its batch to the GPU unconditionally, so there
  is no CPU path. The adapter raises a clear error rather than failing deep
  inside their code.
* Their loader pulls DINOv3 from `torch.hub` at load time, which carries its
  own license. That lands on the user's machine via their code, not ours.

## Follow-ups

1. **A permissive second family.** Verified survey: `simple-romp` is the
   cleanest option, MIT with a self-contained LBS layer that does not import
   `smplx`, no bundled SMPL bytes, and an existing `prepare_smpl` flow where
   the user converts their own SMPL file. HMR 2.0 is the accuracy pick (clean
   MIT, but `smplx` must be swapped for a permissive LBS layer). Reject HybrIK
   and WHAM: both carry Max Planck proprietary-headered files, and WHAM has a
   hard AGPL `ultralytics` dependency.
2. **NVIDIA GEM-X is worth a serious look.** Apache-2.0 code, redistributable
   commercially-usable weights with **no military or trade-control clause**,
   targeting SOMA (Apache 2.0), with zero SMPL anywhere. The catch is that it
   is video-only with no documented single-image path. If a single-frame
   invocation works, it is a better licensing fit than anything else in the
   field.
3. Export contract, validation dataset story, video and world-frame support,
   SMPL-X whole-body hands and face.
