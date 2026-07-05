---
name: libreyolo-api-conventions
description: >-
  The rules every user-facing LibreYOLO surface must follow (Python API, CLI,
  dataset YAMLs, metric keys, Results fields) and the safe method for
  checking parity with the de-facto YOLO ecosystem conventions. Use when
  designing or reviewing any new public API, CLI command or flag, train/val
  kwargs, metric names, or Results attributes; when someone asks "what should
  this be called?", "should predict return a list?", or "is this compatible
  with standard YOLO tooling?"; or when an API-parity gap is reported.
  Encodes the AGENTS.md rule that user-facing APIs follow the ecosystem
  standard, and the license-safe way to verify that (docs only, never
  source).
---

# LibreYOLO API conventions

AGENTS.md's general constraint: every user-facing API (Python, CLI, YAMLs)
follows the de-facto YOLO ecosystem standard, so users' existing scripts,
datasets, and muscle memory transfer. REVIEW.md enforces it as an axiom.
This skill is what that means concretely, plus how to check parity without
license risk.

## The parity-check method (license-critical)

The dominant YOLO library is AGPL. To keep LibreYOLO clean:

- **Never read its source code** to answer "how does the standard behave?".
  Exposure creates contamination risk (see `libreyolo-license-audit`).
- **Do read its public documentation** (docs pages, API reference) to learn
  the *behavioral contract*: argument names, return shapes, attribute
  names. Facts about an interface, learned from docs, are safe; code is not.
- Verify current LibreYOLO behavior by running LibreYOLO, not from memory.
- Where LibreYOLO deliberately deviates, the deviation must be documented
  where users hit it (docstring + docs page), not discovered.

This doc-only method is established practice here (it is how classify
predict/val parity was specified) and is not yours to relax.

## The core contracts

**CLI and Python mirror each other.** Same verbs (`predict`, `train`, `val`,
`export`, `track`), same argument names, same defaults. A capability added
to one side lands on both sides in the same PR, or the PR says why not.

**CLI argument grammar**: both YOLO-style `key=value` and `--key value` are
accepted (`libreyolo/cli/parsing.py`). Every command supports `--json`
(machine-readable stdout), `--quiet`, and `--help-json` (full argument
schema). New flags must work in both grammars and appear in `--help-json`
automatically by being declared properly, never hand-parsed from argv.

**predict returns**: a single `Results` for a single image; a `list[Results]`
for multi-input (directory, list, batched); a generator with `stream=True`.
Do not change this shape per family or per task; downstream code indexes on
it. (Indexing a single `Results` selects a *detection*, which is exactly why
the shape must stay predictable.)

**Results stays flat.** New task outputs are new flat attributes
(`r.boxes`, `r.masks`, `r.keypoints`, `r.probs`, `r.obb`, `r.gaze`,
`r.semantic_mask`, `r.depth_map`, `r.restored`, `r.points`), each a payload
class with data on the **original image canvas**. Never nest, never return
raw tensors in model-input coordinates.

**val returns a metrics object** whose attribute names are the ecosystem's
(`.top1`/`.top5` for classify, mAP-family keys for detection). Metric keys
are public API: renaming one requires a deprecated alias and a loud
changelog entry ("Changed output semantics" is its own changelog section in
`skills/libreyolo-release` for this reason).

**Explicit user kwargs beat defaults** (REVIEW.md axiom), and **CLI defaults
are family-derived**: the family declares its input size / thresholds, the
CLI reads them from the loaded model rather than hardcoding. The worst
violation class is accept-and-ignore: an argument that parses successfully
but does nothing must instead either work or raise (`distill_model`'s
reserved-arg NotImplementedError guard is the sanctioned pattern for
not-yet-implemented surface).

**Dataset YAMLs and label formats** follow the ecosystem format (a
competitor-exported dataset must load unchanged); extensions (e.g.
`masks_dir`, `input_dir`/`target_dir`) are additive keys documented in
`docs/dataset_schema.md`.

**Weights resolution**: a bare canonical filename (`LibreYOLO9t.pt`)
auto-downloads; a path is a path; extra-dependent families fail with the
exact `pip install libreyolo[<extra>]` line in the error.

## Known sanctioned deviations

Deviations exist where honesty beats imitation; they are documented, not
accidental. The load-bearing example: RF-DETR's train signature keeps its
recipe's native argument names (`batch_size`, absolute `lr`, `output_dir`)
because mapping YOLO-style knobs onto a DETR recipe would accept-and-ignore
or silently reinterpret them, which is worse than a visible difference.
REVIEW.md pins this ("RF-DETR ignores generic YOLO augmentation knobs",
"RF-DETR learning rate is absolute"). Match this pattern: deviate loudly
and document, or conform exactly; nothing in between.

## Naming rules (from docs/nomenclature.md, the contract file)

- Model classes: `Libre<FAMILY>` exported from `libreyolo/__init__.py`.
- Weight files: `Libre<FAMILY><size>[-<task>][-<variant>].pt`, sizes
  lowercase, detect suffixless, task suffixes from `libreyolo/tasks.py`.
- Task aliases: generous on input (`seg`, `segmentation` resolve), exactly
  one canonical name in filenames and metadata.
- New user-visible names (CLI commands, kwargs, metric keys, YAML keys) get
  checked against the ecosystem's documented name first; invent only when
  the ecosystem has no name for it.

## Review checklist for any new public surface

1. Name exists in the ecosystem standard? Use it verbatim (from docs).
2. Both CLI grammars, `--json`, `--quiet`, `--help-json` covered?
3. Python and CLI land together?
4. Return shape consistent with the contracts above?
5. Any accepted-but-ignored input? (Blocking.)
6. Docstring + docs page + `docs/nomenclature.md`/`dataset_schema.md`
   updated if a contract file is affected?
7. Unit tests assert the shape/name, so parity cannot silently regress.

## Related

- `docs/nomenclature.md`, `docs/dataset_schema.md`: the contract files.
- `skills/libreyolo-add-cli-command/`: implementing the CLI side.
- `skills/libreyolo-add-task/`: contracts for new output kinds.
- `skills/libreyolo-license-audit/`: why the docs-only rule is absolute.
