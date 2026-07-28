# Proposal: `mesh` task (human body mesh recovery)

Status: investigation complete, API proposal. No code exists yet. Companion surveys behind this document: 13 repositories read (image HMR, video HMR, whole-body, frameworks) plus a licensing deep dive on the SMPL family and its permissive alternatives.

## 1. Is there a standard?

Yes: SMPL, from Max Planck, is the de-facto standard for representing a 3D human body. Every benchmark (3DPW, AGORA, EMDB) and virtually every method speaks it. The parameterization:

| Model | Covers | Vertices | Pose params | Shape | Extra |
|---|---|---|---|---|---|
| SMPL | body | 6,890 | 24 joints x 3 axis-angle = 72 (3 global_orient + 69 body_pose) | 10 betas | |
| SMPL-X | body + hands + face | 10,475 | 3 global + 63 body + 2x45 hands + jaw/eyes | 10 betas | 10 expression |

A body is fully described by `{global_orient, body_pose, betas, transl}`; a fixed differentiable decoder (linear blend skinning over a template mesh) turns that into vertices and joints. The mesh topology (faces) is constant, which is what makes the format an interchange standard: Blender add-ons, retargeting tools and mocap pipelines all consume it.

The catch, and it is the central fact of this proposal: the SMPL and SMPL-X model files are non-commercial AND non-redistributable (one archive copy only, registration-gated), and the `smplx` Python package CODE carries the same non-commercial Max Planck license, not a permissive one. The SMPL license additionally forbids training networks on it "for commercial deployment", so the non-commercial cloud reaches into any checkpoint supervised by or decoding to SMPL. Under our weights policy, non-commercial weights are hostable with a license tag, but no-redistribution artifacts are not mirrorable: SMPL model files are in the cannot-mirror category, full stop. Every OSS repo in the field handles this the same way: the user registers at the MPI site and downloads the file themselves.

Since 2025 there are, for the first time, permissive alternatives:

- MHR / Momentum Human Rig (Meta, Apache 2.0 code AND assets, `pip install mhr`, no registration). It is the output format of SAM 3D Body, whose checkpoints permit commercial use and redistribution with license passthrough.
- Anny (NAVER, Apache 2.0 code, CC0 geometry from MakeHuman). Fully clean, but no strong pretrained image-to-Anny regressor ships with it.

Neither is SMPL-compatible; both ship learned mappings toward SMPL(-X) as separate opt-in components.

## 2. What the field's APIs actually look like

Thirteen repos surveyed. The user experience today is uniformly bad: conda env, pinned torch, compile something, register for SMPL, run demo scripts. Nobody offers `model.predict("img.jpg")` with auto-downloaded weights. No mainstream easy-install library ships this task at all; it is greenfield.

Despite the mess, all of them converge on the same per-person payload: SMPL params (`global_orient`, `body_pose`, `betas`), camera translation, vertices, 3D joints, projected 2D joints, confidence. The three cleanest end-user APIs, for reference:

- NLF: one TorchScript file, `model.detect_smpl_batched(frames)` returns pose, betas, transl, joints3d, vertices3d, joints2d, uncertainties. Detector is inside the graph.
- simple-romp: `pip install simple_romp`, one-liner call, flat numpy dict, single-shot multi-person, permissive code and weights.
- Multi-HMR: single forward on the full frame, list of per-person dicts, optional camera intrinsics input (non-commercial, reference only).

Two architectural camps:

- Single-shot (ROMP, Multi-HMR): full image in, all people out, no detector.
- Top-down (HMR2.0/4D-Humans, TokenHMR, CameraHMR, OSX, SAM 3D Body): external person detector, then per-crop regression. These repos all bolt detectron2 or similar onto their demos, which is exactly the pain we can delete: we already have person detectors in-house.

Video methods (WHAM, GVHMR, TRAM) add tracking, temporal smoothing and world-frame trajectories via SLAM or gravity alignment. A v1 image API can ignore all of it, and should; the schema just needs to name the camera frame explicitly so a world frame can be added later without breaking anything. The cleanest schema in the field is GVHMR's pair of dicts, each exactly `{global_orient, body_pose, betas, transl}`, one per coordinate frame.

## 3. Proposed Python API

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreXXXs-mesh.pt")        # family TBD, task from -mesh suffix
results = model("people.jpg", save=True)      # save=True draws skeleton + vertex overlay

r = results
r.boxes                    # person Boxes (N), same as detect/pose
r.meshes                   # Meshes payload, row-aligned with boxes

# Parametric core: directly splattable into a body-model forward
r.meshes.global_orient     # (N, 3) axis-angle root rotation, camera frame
r.meshes.body_pose         # (N, J_body, 3) axis-angle
r.meshes.betas             # (N, num_betas)
r.meshes.transl            # (N, 3) metric camera-space translation

# Derived geometry, precomputed by the pipeline
r.meshes.vertices          # (N, V, 3) camera-space, metric
r.meshes.faces             # (F, 3) shared topology, constant per body model
r.meshes.joints3d          # (N, J, 3) camera-space
r.meshes.joints2d          # (N, J, 2) pixel coords in the original image
r.meshes.conf              # (N,) per-person confidence

# Identity of the representation, from checkpoint metadata
r.meshes.body_model        # "smpl" | "smplx" | "mhr"

# Conveniences
r.meshes.save_obj("person0.obj", index=0)     # standard mesh interchange
r.summary(); r.to_json()                       # params + joints2d (vertices omitted by default)

# Optional intrinsics, as in the metric-aware repos
results = model("people.jpg", focal_length=1400.0)   # or K=3x3; default: estimated or heuristic
```

Top-down families additionally follow the established gaze pattern for detector chaining:

```python
mesh = LibreYOLO("LibreXXXs-mesh.pt", person_detector="LibreYOLO9s.pt")  # default wired in
r = mesh("people.jpg")
r = mesh("people.jpg", person_boxes=my_boxes)    # BYO boxes, detector skipped
```

CLI: `libreyolo predict model=LibreXXXs-mesh.pt source=people.jpg` behaves like every other task.

### Design decisions baked into the above

- The parametric block IS the API. `{global_orient, body_pose, betas, transl}` is the field's universal denominator and splats directly into a body-model layer. Vertices/joints are derived caches, not the source of truth.
- Camera frame is explicit and singular in v1. No world frame, no track IDs, no temporal smoothing. A future video story adds `r.meshes.global_orient_world` / `transl_world` (or a parallel `meshes_world` slot) without touching v1 fields.
- Row-aligned with `Boxes`, exactly like `Keypoints` in the pose task. `joints2d` reuses the existing keypoint drawing path for `save=True`; full mesh rasterization is not a v1 requirement (a painter's-algorithm vertex overlay is enough, no renderer dependency).
- `body_model` is a first-class metadata field. The schema is body-model-agnostic; SMPL, SMPL-X and MHR families coexist, with `V`, `J`, `num_betas` read from checkpoint metadata (analogous to `num_keypoints` / `keypoint_dim` in pose).
- Single-shot families use the standard `InferenceRunner`; top-down families use a dedicated runner with `person_detector=` / `person_boxes=`, mirroring the existing crop-consumer precedent. Never an external detectron2-style dependency.

## 4. The one decision that matters: which body model we ship

This is a licensing decision, not a technical one, and it gates everything.

Door A, SMPL compatibility: maximum ecosystem value, poisoned supply chain. We cannot host SMPL files, cannot depend on the `smplx` package (its code is non-commercial), and checkpoints that embed SMPL template/blendshape buffers are themselves redistributing derived SMPL data (grey at best). The honest version of Door A is the simple-romp pattern: a `libreyolo mesh prepare --smpl-path ...` step that converts the USER'S own registered download into our cache, a clean-room LBS forward pass of our own (the math is published; the DATA is what is licensed), and hosted checkpoints only for models whose weights are genuinely redistributable.

Door B, MHR: Apache 2.0 body model, pip-installable assets, and SAM 3D Body as a strong pretrained predictor whose weights we may redistribute with license passthrough. Fully consistent with the MIT wedge. Cost: not SMPL, so the Blender/retargeting ecosystem needs a conversion step (MHR ships mappings).

Recommendation: design the schema body-model-agnostic (already done above), ship the first hosted family through Door B, and treat Door A as a follow-up family for users who bring their own SMPL file. That gives a working `pip install` to first-mesh experience with zero license ceremony, which no library on earth offers today, while keeping the door open for SMPL interop.

## 5. Candidate first families

| Candidate | Type | Code | Weights | Verdict |
|---|---|---|---|---|
| SAM 3D Body (MHR) | top-down, promptable | SAM license | redistributable, commercial OK, passthrough | Door B flagship |
| ROMP / simple-romp | single-shot | Apache/MIT | permissive, auto-download | Door A flagship; older accuracy, lightest port |
| 4D-Humans (HMR2.0) | top-down | MIT | unstated; SMPL-entangled | reference architecture, weights unclear |
| NLF | packaged top-down | MIT | non-commercial research | packaging inspiration; weights NC |
| Multi-HMR, TokenHMR, CameraHMR, SMPLer-X, GVHMR | various | NC or gated | NC, registration-gated | schema reference only, no code porting |

## 6. What adding the task touches

The plumbing is the well-trodden ten-task path: `tasks.py` tables (`mesh`, aliases `body-mesh`/`hmr`/`smpl`, suffix `-mesh`), a `Meshes` payload class in `utils/results.py` implementing the tensor-payload protocol, a `_wrap_results` branch, `draw_mesh` in `utils/drawing.py` (skeleton via `joints2d` + vertex overlay), a `MeshValidator` (MPJPE / PA-MPJPE / PVE) with the `val()` dispatch entry, checkpoint metadata (`body_model`, `num_betas`, `num_body_joints`, `num_vertices`), export explicitly gated off in v1 like semantic/depth were, and docs (nomenclature, checkpoint schema, ADR).

Genuinely new work, beyond plumbing:

1. A body-model decoder layer of our own (LBS forward). Small, published math; the licensing lives in the data files, not the equations. For MHR the pip package is Apache and usable directly.
2. Visualization without a renderer dependency (projected-vertex overlay).
3. Evaluation data: 3DPW and EMDB are research-only licenses; the validator ships, hosted eval data does not. Same posture as other license-gated datasets.

## 7. Scope fences (v1)

- Image and video-as-frames inference only. No tracking, no temporal smoothing, no world-frame trajectory, no SLAM. The schema reserves room; the code does not attempt it.
- Body only. SMPL-X hands/face/expression fields are schema-compatible extensions, not v1 deliverables.
- Inference-only family at first (the gaze precedent). Training on mesh data is a separate, later effort with its own dataset-licensing investigation.
- Export gated off with a clear error until the runtime metadata contract is defined.

## 8. Suggested phasing

1. Decide Door A vs Door B in the open (ADR).
2. Contract: docs + `Meshes` schema + checkpoint metadata, reviewed before any model code.
3. Port the chosen family, inference only, with the detector-chaining or single-shot runner as appropriate.
4. Validator + a small permissively-licensed smoke-eval set.
5. Ship; video/world-frame and SMPL-X whole-body as separately scoped follow-ups.

## 9. Verdict

It makes sense, and it is not too hard: the task plumbing is routine and well-precedented by pose (row-aligned payload) and gaze (detector chaining). The genuinely hard part is not code, it is the body-model license, and it is solvable: either the user-supplies-SMPL flow everyone else uses, or the new Apache-licensed MHR path that nobody has productized yet. Shipping `LibreYOLO("...-mesh.pt")` into a per-person parametric body with zero conda ceremony would be a first in the ecosystem.
