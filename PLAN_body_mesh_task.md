# PLAN: `mesh` task (human body mesh recovery), Meta stack

Scope: the Meta stack only, SAM 3D Body on the MHR body model. Other families
(ROMP, 4D-Humans, NLF, Multi-HMR, TokenHMR, CameraHMR, SMPLer-X, GVHMR) were
surveyed and are out of scope by decision. Investigation date 2026-07-28.

Status: **task contract and body model shipped; regressor blocked on weight
access.** See "What is done" and "What is blocked" below.

## Why the Meta stack, in one paragraph

The rest of the field predicts into SMPL, and SMPL is unusable here: the model
files are non-commercial and non-redistributable, the `smplx` PyPI package's
*code* carries that same non-commercial license (the most common misconception
in this area), and the license reaches into any checkpoint trained against it.
MHR is Apache 2.0 for code and assets with no registration, and SAM 3D Body's
weights are redistributable under the SAM license with passthrough. It is the
only end-to-end path where the body model, the code and the weights all clear.

## What is done

Landed on branch `mesh-body-recovery`, off `dev`:

* **Task registration.** `mesh` in `TaskType`/`TASKS`, suffix `-mesh`, aliases
  `body-mesh`, `hmr`, `human-mesh-recovery`. `smpl` deliberately not aliased.
* **`Meshes` result payload**, row-aligned with `Results.boxes` the way pose
  keypoints are. Body-model-agnostic: `body_model` names the parameterization
  and all counts are read from the tensors. Slicing selects a person without
  touching shared face topology; device moves cover topology and extras.
  `save_obj()` writes standard Wavefront OBJ.
* **MHR body model** (`libreyolo/models/sam3dbody/mhr_body.py`): TorchScript
  loader, parameter assembly, unit conversion, and a downloader for the public
  Apache-2.0 release asset. Validated against the real 696 MB asset.
* **Camera helpers** (`camera.py`): crop-camera to full-image lifting and
  perspective projection, so `joints2d` is in original-image pixels.
* **Metrics** (`libreyolo/validation/mesh_metrics.py`): MPJPE, PA-MPJPE with a
  reflection-safe Procrustes, PVE.
* **Drawing** (`draw_mesh`): renderer-free projected-vertex scatter plus the
  MHR-70 skeleton. No rasterizer dependency.
* **Gates**: export raises an explicit not-implemented error, validation
  explains the dataset-license situation, TTA is rejected with the
  left/right-swap reason.
* **Docs**: ADR 0013, nomenclature, checkpoint schema.
* **Tests**: 54 unit tests on fabricated payloads, plus 9 `external_data`
  integration tests against the real MHR asset.

Verified numbers, not assumptions: MHR `model_params` is 204 wide (3
translation, 3 global rotation, 130 body pose, 68 bone scales), `betas` 45,
`expression` 72, outputs 18439 vertices and 127 joints in centimeters, with
translation entering the rig in decimeters. The upstream comment saying 127 is
stale; the assertion two functions below it says 136. Rest-pose height decodes
to 1.727 m and a requested 0.1/0.2/0.3 m translation moves the body by exactly
that, which is what pins the unit conventions.

## What is blocked

**SAM 3D Body regressor weights.** `facebook/sam-3d-body-dinov3` and
`facebook/sam-3d-body-vith` are **gated with manual approval** on Hugging Face,
and the token cached on this machine is invalid ("Invalid user token"). Both
must be resolved before the regressor family can be built, because a port that
cannot be run against real weights certifies nothing.

To unblock, in order:

1. Request access at https://huggingface.co/facebook/sam-3d-body-dinov3 and
   wait for approval. Note the upstream warning that comprehensively sanctioned
   jurisdictions are rejected.
2. Generate a fresh token and `hf auth login --force`.
3. Confirm with `huggingface_hub.model_info("facebook/sam-3d-body-dinov3")` and
   an `hf_hub_download` of `model_config.yaml`.

Then the remaining work is the regressor family itself: DINOv3/ViT-H backbone,
promptable decoder, MHR head and camera head; a `weights/convert_sam3dbody.py`
converter into the LibreYOLO checkpoint schema; a top-down runner following the
l2cs chaining pattern (`person_boxes=` / `person_detector=`, never an external
detectron2 dependency); parity evidence on trained weights; and an upload to
the LibreYOLO org carrying the SAM license text, since passthrough is a
condition of redistribution.

Naming when it lands: `LibreSAM3DBody<size>-mesh.pt`, sizes `d3` (DINOv3) and
`h` (ViT-H), chosen to avoid colliding with the existing `LibreSAM` promptable
tier.

## Nothing is mirrored

The MHR asset is fetched from the public upstream release and cached locally
rather than copied to the LibreYOLO org: it is freely reachable, Apache 2.0 and
700 MB, so a second copy serves no one. No SMPL-derived artifact exists
anywhere in this work.

## Scope fences

Out of scope and not started: video tracking, temporal smoothing, world-frame
trajectories (the schema reserves room via future `*_world` fields), SMPL-X
whole-body hands and face, training, mesh export formats beyond OBJ, any
renderer dependency, and exported-graph support.
