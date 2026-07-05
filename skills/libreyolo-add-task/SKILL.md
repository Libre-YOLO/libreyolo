---
name: libreyolo-add-task
description: >-
  Add a new task type to LibreYOLO (a new output kind like detect, segment,
  semantic, pose, classify, gaze, obb, point, depth, restore), as opposed to
  adding a model family to an existing task. Use when someone proposes a new
  capability that no current task's result/label/metric contract fits, e.g.
  "add OCR", "add 3D boxes", "add image enhancement as a task". Covers the
  decision (new task vs existing task), and the full wiring checklist:
  tasks.py registration, the Results payload class, dataset loader + schema,
  validator, trainer support, filename suffix, docs/ADR, and the tests that
  gate each piece. For porting a model into an existing task use
  libreyolo-port-model.
---

# Add a new task to LibreYOLO

A **task** is a contract, not a model: what `Results` exposes, what the label
format is, what the val metric means, and what filename suffix marks the
checkpoint. Families then opt in via `SUPPORTED_TASKS`. Ten tasks exist
(`detect segment semantic pose classify gaze obb point depth restore`); read
`docs/nomenclature.md` and `libreyolo/tasks.py` first, they are the source of
truth this skill points into.

## Step 0: is it actually a new task?

New tasks are expensive (every axis below, forever). Reuse an existing task
when the *output contract* fits, even if the model is exotic:

- CLIP zero-shot classification reused `classify` (same Results.probs
  contract, new `set_classes` capability).
- Open-vocabulary detectors reused `detect` (boxes are boxes; prompts are a
  model capability, not a new output kind).
- FOMO could not reuse `detect` honestly (its learned output is a centroid,
  not a box), so `point` became a task with its own contract (ADR 0003).
- Deblur/denoise did not become two tasks; they are aliases of one `restore`
  task (paired image-in image-out).

The test: **would validation metrics and the label format be lies under every
existing task?** Only then add one. Naming: pick the generic capability name
(`restore`, not `deblur`); add user-friendly aliases in `TASK_ALIASES`.

Precedents to copy end to end: `point` (ADR 0003), `depth` (ADR 0006),
`restore` and `semantic` (git history around their landing PRs). Diffing one
of those landings is the fastest complete file list.

## The wiring checklist

Work through every row; a task missing a row ships a hole users find.

1. **`libreyolo/tasks.py`**: add to `TaskType`, `TASKS`, `TASK_ALIASES`
   (canonical + aliases), `TASK_TO_SUFFIX` (the filename suffix, e.g.
   `-ocr`). Detect is the only suffixless task; every new task carries one.
   Unit tests: `tests/unit/test_tasks.py` (alias resolution, suffix
   round-trip, `resolve_task` precedence).

2. **Results payload** (`libreyolo/utils/results.py`): a payload class on the
   `_TensorPayload` pattern (`SemanticMask`, `DepthMap`, `RestoredImage`,
   `Points` are the models), documented shape **on the original image
   canvas** (original-canvas coordinates are canonical, REVIEW.md axiom),
   a `Results.<field>` attribute, plotting support in `Results.plot()`, and
   export in `to_json`/`to_df` paths. Export the class from
   `libreyolo/__init__.py.__all__`. Keep `Results` flat and API-compatible:
   a new field, never a new nesting level.

3. **Dataset loader + schema** (`libreyolo/data/`): a `<task>_dataset.py`
   loader and a documented layout in `docs/dataset_schema.md` (that doc is a
   contract file; PRs that change label semantics without updating it get
   flagged). Decide the YAML keys (`masks_dir`, `depth_scale`,
   `input_dir`/`target_dir` are precedents) and whether a COCO-JSON
   `annotations:` path exists (only detect/segment/obb have one today;
   others convert offline). Add a small fixture dataset yaml under
   `libreyolo/config/datasets/` if a public tiny set exists.

4. **Validator** (`libreyolo/validation/<task>_validator.py`): the metric
   contract. Reuse the shared preprocessing in `validation/preprocessors.py`
   and follow an adjacent validator (`SemanticValidator` for dense tasks,
   `PointValidator` for sparse). Export it from `validation/__init__.py`
   (and `libreyolo/__init__.py` if user-facing). Define the headline metric
   names carefully; renaming metric keys later is a breaking change users
   notice (it has happened, and needed deprecated aliases).

5. **Trainer support** (only if trainable at launch): loss + target
   assembly in the family's trainer subclass; the generic `BaseTrainer`
   orchestrates. It is legitimate to ship a task inference-only first
   (`semantic` via PIDNet/EoMT and `gaze` did); say so in docs and keep
   `model.train()` failing with a clear message, never silently.

6. **Model family** (at least one, via `libreyolo-port-model`): sets
   `SUPPORTED_TASKS`, `DEFAULT_TASK`, per-task input sizes
   (`TASK_INPUT_SIZES`) if they differ. Per AGENTS.md, new *features* should
   cover the flagships (YOLO9, RF-DETR) where the task is a natural fit;
   single-purpose tasks carried by a dedicated family (gaze, restore) are
   the accepted exception.

7. **CLI + predict surface**: `libreyolo predict` must render/save the new
   output (drawing in `libreyolo/utils/drawing.py`), `--json` must include
   it, and `libreyolo val` must route to the new validator. Check
   `libreyolo/cli/commands/predict.py` and `val.py` for task-specific
   handling to extend.

8. **Docs + ADR**: a short ADR under `docs/adr/` for the task contract
   (0003 point and 0006 depth are the models: result semantics, metric
   definitions, what is out of scope), a row in every `docs/nomenclature.md`
   table that mentions tasks, and `docs/dataset_schema.md` (step 3).

9. **Checkpoint metadata**: new-task checkpoints must carry `task` metadata
   per `docs/checkpoint_schema.md`; the conversion script writes it
   (`weights/convert_*_weights.py` templates in `libreyolo-port-model`).
   Cross-task load rejection must hold: loading a `-ocr` checkpoint as
   `detect` fails loudly (there are per-family `*_can_load`-style unit tests
   to extend).

10. **Tests, minimum set**: tasks.py unit tests (step 1), Results payload
    shape/plot tests, dataset loader tests on a synthetic fixture, validator
    tests with known-answer inputs, one e2e inference case once weights
    exist (`libreyolo-write-e2e-tests`), and the RF1-style training check if
    trainable (`libreyolo-verify-training`).

## Order of work

Land it as reviewable slices, contract-first: (1) tasks.py + Results + docs
+ ADR; (2) dataset loader + validator; (3) the first family + weights; (4)
trainer. Each slice keeps the suite green on its own. One PR per slice
matches the repo's one-problem-per-PR policy; link them to the tracking
issue.

## Traps

- Adding the task but not the suffix (or vice versa): filename parsing and
  task resolution disagree, and autoresolution breaks in ways only
  `test_tasks.py` catches.
- Metric keys named casually. They become API the moment a user's script
  reads them; design them like public function names.
- A dataset layout invented fresh when an ecosystem-standard layout exists.
  Follow the de-facto YOLO ecosystem format for anything box-like; users
  arrive with those exports.
- Skipping the license gate on the launch dataset/weights: run
  `skills/libreyolo-upload-hf-dataset` and `skills/libreyolo-upload-hf-model`
  gates before promising auto-download.
- Announcing the task while its only family is inference-only without
  saying so.

## Related

- `skills/libreyolo-port-model/`: the family-level port this skill hands to.
- `docs/adr/0003-point-task-contract.md`, `docs/adr/0006-depth-task-contract.md`:
  the contract-ADR pattern to copy.
- `skills/libreyolo-api-conventions/`: naming and API-shape rules for the new
  surface.
