---
name: libreyolo-write-model-prd
description: >-
  Write the PRD (port handoff) for adding a new model to LibreYOLO: the
  single markdown document an implementing agent executes end to end. Use when
  someone proposes a model ("should we add X?", "can we port X?", "write a PRD
  for X", "what would it take to add X"), when triaging a model-request issue,
  or when building a batch of candidates for the museum tier. Covers the
  already-in-the-tree check, the license gate, choosing the port source and
  scaffold, the fixed PRD section template, and publishing the result to a
  gist registered in issue #637. This is the planning half; the execution half
  is `libreyolo-port-model`.
---

# Write a PRD for adding a model to LibreYOLO

The deliverable is **one markdown file** that an implementing agent can follow
without further research: what the model is, whether we may legally ship it,
where to port it from, what to clone, what gates to pass, and what will silently
break. If the reader has to go re-derive the license or hunt for a scaffold, the
PRD failed.

This skill does not port anything. When the PRD is approved, the implementer
follows `libreyolo-port-model` (and `libreyolo-upload-hf-model` at the weights
step). Keep the PRD free of process the skills already own: cite the skill
section instead of restating it.

## 1. First question: is it already in the tree?

Do this before any research. It is the cheapest step and the most expensive one
to skip.

```bash
git fetch upstream dev
git ls-tree --name-only upstream/dev:libreyolo/models/
```

**Always inventory `dev`, never a feature branch.** `dev` carries far more
families than any working branch. A campaign that inventoried a feature branch
wrote a complete PRD for YOLOv1, which had already shipped as
`libreyolo/models/yolo1/`, and nearly instructed an agent to create
`libreyolo/models/swin/`, which already exists and holds a shared backbone that
Grounding DINO imports.

Check four things, not one:

1. The family directory (`libreyolo/models/<family>/`).
2. The `FAMILY` id and `FILENAME_PREFIX` you intend to claim, against every
   existing family.
3. The architecture as a **component**. A model can already be in the tree as a
   shared backbone or neck without being a family of its own
   (`git grep -l "<ArchName>" upstream/dev -- libreyolo`).
4. `docs/nomenclature.md` and `weights/LICENSE_NOTICE.txt`, which sometimes
   record a family before or after the code moves.

If it already exists, **stop and write a short note instead of a PRD**: what is
shipped, what is genuinely missing, and the one or two residual tasks worth
doing. Register that note the same way (section 6). A PRD for a shipped model
wastes an entire implementation cycle.

## 2. License gate

Run `libreyolo-license-audit` for the verdict. Do not improvise licensing
judgement here. What this skill adds is what the PRD must *record*.

Two different bars, and collapsing them is the most common error:

- **Code** vendored into core must be MIT, Apache-2.0 or BSD. No GPL, AGPL,
  LGPL, non-commercial or unknown-license code, ever, including rewrites.
- **Weights** only need to be **redistributable**. Non-commercial weights are
  shippable when tagged accordingly. Weights that forbid redistribution are not
  a blocker either: link upstream instead of rehosting (the YOLO-NAS precedent).

Consequences the PRD must state explicitly:

- **A permissive reimplementation rescues a tainted original.** Both original
  FCOS repos open with a non-commercial clause, so neither may be read or
  ported, but torchvision ships a BSD-3 FCOS and that is a clean path. Name the
  source to use *and* the source to stay out of.
- **Never launder.** Rewriting non-permissive code to disguise its origin is
  prohibited regardless of how the result looks.
- **Primary sources only, with evidence URLs in the PRD.** The GitHub license
  API (`https://api.github.com/repos/<org>/<repo>/license`), the raw `LICENSE`
  file, and the HF model card YAML. Never a recollection, never a badge.
- **Say "implied" when it is implied.** Most weights carry no per-artifact
  license and inherit it from the releasing repository. torchvision publishes
  none at all and explicitly disclaims that pretrained models "may have their
  own licenses or terms and conditions derived from the dataset used for
  training". Write that distinction on the model card rather than upgrading an
  inference into a stated grant.
- **Check per-variant, not per-model.** Licenses differ across sizes and
  generations. MiDaS is MIT, but its v2 and v2.1 checkpoints were fine-tuned
  from CC-BY-NC pretraining, so only some variants are clean.
- **Watch the backbone init.** A "trained by us" checkpoint often starts from a
  third-party backbone whose license still applies.
- **Beware the popular decoy.** The most-starred implementation is often the
  unusable one. Name it in the PRD so nobody finds it later and assumes it is
  fine.

## 3. Port source and scaffold

Pick the source that combines permissive code with loadable weights, preferring
the one whose module names match the checkpoints so conversion stays a
metadata-wrap. State the runner-up and why it lost, because the implementer will
otherwise rediscover it.

For the scaffold, read the per-family ledger in `libreyolo-port-model`
(its section 4) and name a concrete directory to clone. Then check the
identifiers you are claiming do not collide, and list the families whose
`can_load` could steal your checkpoints, so the PRD can require bidirectional
rejection tests.

## 4. Scope and maturity

Say which tasks are in scope and declare `SUPPORTED_TASKS` explicitly. Then pick
a maturity target from `libreyolo-port-model` section 2, and remember that
**inference-only is a legitimate ship state**. A PRD that gates delivery on a
working trainer will usually stall. Make the trainer a follow-up unless training
is the point of the port.

## 5. The PRD document

Use these sections, in this order. It is the shape that has survived adversarial
review, and a consistent shape is what lets an implementing agent trust it.

```
# Handoff: add <MODEL> as a LibreYOLO <task> family

**For:** an implementing agent starting fresh.
**Process authority:** skills/libreyolo-port-model/SKILL.md.

## 0. Mandatory gates          (the non-negotiables, see below)
## 1. The model                (architecture, variants/sizes, historic significance)
## 2. License                  (verdict + evidence URLs + rehost-or-link decision)
## 3. Why we did not add it before
## 4. Why we are adding it now
## 5. Head start already in-tree (closest scaffold, and its traps)
## 6. Scope and maturity
## 7. Implementation pointers  (family id, can_load, converter, export contract)
## 8. Definition of done       (checklist mirroring section 0)
```

Section 0 always carries these gates:

1. **Upstream parity**: `max_abs_diff == 0.0` in eval mode against the
   recommended source, for every shipped size, before any postprocess, export or
   trainer code exists. If the port vendors the same implementation it compares
   against, that check is tautological: say so and give a meaningful alternative
   (published metric reproduction, or parity against the reference the weights
   came from).
2. **Export parity**, separately: the exported graph must match our PyTorch
   output on the same image. "The export runs" is not the bar (see section 7).
3. **Weights**: rehost per `libreyolo-upload-hf-model` when redistributable, or
   link upstream when not. Verify auto-download on a cleared cache.
4. **Tests**: the right registration for the task (see section 7).
5. **UI smoke check**: load one converted checkpoint through the UI and confirm
   predict renders.

Close with the branch rule: branch off `dev` and land the work through
`merge-to-dev`. An agent may open the PR but never approves or merges it. The PR
body must contain a filled `## Code provenance` section, which
`provenance-check.yml` enforces by matching a `^#{1,6}\s*code provenance$`
heading and failing when it is missing or empty.

Style: no em dashes or en dashes, no personal names, no machine-specific paths.
State only what you verified, and tell the implementer to verify at port time
where you could not.

## 6. Publish and register

A PRD nobody can find is lost work.

1. Publish it as a **public gist** (`gh gist create <file> --public --desc "..."`).
   Per-file anchors are the filename lowercased with dots turned into hyphens,
   for example `#file-handoff_detr_detect-md`.
2. Register it in **issue #637, "Model candidates to be added to the library"**:
   https://github.com/LibreYOLO/libreyolo/issues/637
   Edit the issue body, put the model under its task category (Detection,
   Instance segmentation, Classification, Semantic segmentation, Keypoint/pose,
   Depth), and follow the existing line format:

   ```
   - <Model>: <paper or repo url> (<license summary>). Handoff: <gist anchor url>
   ```

   Models already shipped go in the `ADDED` section at the bottom with their
   family name, not in the candidate list.
3. Before publishing anything, scan for leakage: no real names, usernames,
   emails, absolute machine paths, or credentials. Repo-relative paths only.

Keep the issue and the gist in sync. The issue is the index; the gist is the
content.

## 7. Facts PRD authors get wrong

Verified against `dev`. Each of these has already shipped in a PRD and had to be
corrected.

- **`from_pretrained` does not exist.** `LibreYOLO` is a factory *function*
  (`libreyolo/models/__init__.py`) that takes a path. The auto-download check is
  `LibreYOLO("Libre<Family><size>.pt")` on a cleared cache with no staged copy
  under `weights/`. Note that `libreyolo-port-model` and
  `libreyolo-upload-hf-model` both still show a stale `from_pretrained` form;
  do not copy it into a PRD.
- **A non-YOLO-grid export needs two backend edits, not one.**
  `_is_nms_free_family()` is a module-level function in
  `libreyolo/backends/base.py` (not a `BaseBackend` method) and only decides
  whether NMS is re-applied *after* parsing. The parse itself is family
  dispatched in `_parse_outputs`, whose final `else` falls through to
  `_parse_yolo9` and reads the graph as a `(4+nc, N)` YOLO tensor. A DETR-shaped
  graph parsed that way returns garbage while appearing to export fine. Route to
  `_parse_dfine` like the shipped DETR families. Never cite line numbers for
  either: they move between branches.
- **`MODEL_CATALOG` is detect-only.** It drives `test_val_coco128.py` and its
  mAP50-95 gate, so a classify, depth or semantic-segmentation row fails by
  construction. Point those families at a per-family unit suite instead, and
  check what the closest merged family actually registers before writing the
  gate.
- **Historic model, modern artifact.** For older models the cleanly licensed
  implementation is often not the historic one: torchvision's AlexNet is the
  "one weird trick" variant, its VGG weights were trained from scratch rather
  than converted from the original release, and its FCN has no skip fusion. Ship
  the modern rebuild if you like, but require the PRD to say so plainly on the
  model card rather than exhibiting a replica under a historic name.
- **Weight hosting rots.** Checkpoints living only on Google Drive or a personal
  academic host need rehosting early in the port, not at the end.
