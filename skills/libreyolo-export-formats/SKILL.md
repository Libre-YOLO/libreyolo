---
name: libreyolo-export-formats
description: >-
  Add, validate, or debug export support for a LibreYOLO (family, task,
  format) cell: ONNX, TorchScript, TensorRT, OpenVINO, NCNN, TFLite, CoreML,
  Core AI. Executable guide: read the support matrix, acquire trained
  weights (never validate random init), prove parity with the two-probe
  sensitivity gate and task-aware thresholds, wire predict-back, record the
  cell in export/support.py, regenerate docs. Use for "add <format> export
  to <family>", "the exported model gives different results", export-parity
  campaigns, filling matrix gaps with agents, or adding a new format. For a
  user just running export, see use-libreyolo; for Hailo,
  libreyolo-export-hailo.
---

# LibreYOLO export support (developer guide)

This skill is **executable**. The unit of work is a **cell**: one
`(family, task, format)` combination. A cell carries three separate claims,
and you must know which one you are making:

1. **Converts**: `model.export(format=...)` produces an artifact.
2. **Numeric parity**: the artifact computes the same numbers as the
   reference graph, proven with the ladder in §3.
3. **Predict-back**: `LibreYOLO(artifact).predict(...)` returns the same
   `Results` a `.pt` would.

"Validated" in the matrix means claim 2 with a parity test; a "double tick"
in audits means claims 2 and 3 together. Claim 1 alone is `experimental`.

## Architecture in one table

| Layer | Path | Role |
|---|---|---|
| Support matrix | `libreyolo/export/support.py` | Sole source of truth for cell tiers |
| Exporters | `libreyolo/export/exporter.py` + per-format modules | Serialize the graph **and** embed metadata |
| Backends | `libreyolo/backends/` | Load the artifact, read metadata, reproduce pre/postprocess |
| Docs | `tools/gen_compat_table.py` writes `docs/export_support.md` | Generated, drift-gated, never hand-edited |

The two REVIEW.md axioms that govern all of it: **backends must behave like
models** (same API, same Results, original-canvas coordinates) and
**exported runtimes round-trip metadata** (what the exporter writes, the
backend must read; a model loaded from an export must not need the user to
re-specify imgsz/task/classes).

The tier contract is `docs/adr/0011-export-support-tiers.md`. Three
consequences you will hit: exporters warn for `experimental` and refuse
`blocked` before dependency checks or tracing; adding a `validated` entry
requires a parity test and a `since` field; CoreML conversion without a
macOS prediction run can be `experimental` but can never be `validated`.

## 1. Read the matrix before writing code

Schema (`libreyolo/export/support.py:10-33`): `SupportEntry(tier, reason,
since, constraint)` with `Tier = validated | experimental | blocked`, keyed
by `(family, task, format)` in the `SUPPORT` dict. `_add()`
(`support.py:36-60`) is the **only** writer; it takes cartesian products of
families x tasks x formats and raises on duplicate keys.

`get_support()` resolution order (`support.py:1119-1169`), each step
terminal:

1. Non-canonical task or unknown format: synthesized `blocked`.
2. Exact `SUPPORT[(family, task, fmt)]` entry.
3. `_FAMILY_BLOCKS[family]` (whole-family block, e.g. promptable SAM tier,
   generative VLMs, l2cs).
4. `_TASK_BLOCKS[task]` (task has no shared export contract yet: ocr,
   point, semantic, mesh, normal, panoptic, gaze; families opt in with an
   explicit `_add`).
5. NCNN DETR blocklist `_NCNN_BLOCKS`.
6. **Format defaults**: `tensorrt`/`openvino` fall to `experimental`;
   `tflite`, `coreml`, `coreai` fall to `blocked`; anything else
   (onnx, torchscript) falls to `experimental`.

That last asymmetry is the most important line in the file: an unlisted
family lands `experimental` on ONNX/TorchScript/TensorRT/OpenVINO
**silently**, and `blocked` on TFLite/CoreML/Core AI. So "add tflite
support for family X" always means writing a matrix row; "add onnx support"
may mean the cell already converts and your job is only §3.

Two rules about existing entries:

- **A `blocked` cell with a measured toolchain reason is a fact, not a
  gap.** "PNNX 20260526 reports unsupported batch-index reshapes" does not
  get re-attempted every campaign; it gets re-tested when the named tool
  version changes. Do not delete a blocked row without reproducing the
  failure's absence.
- **Reason wording is tested** (`tests/unit/test_export_support.py`).
  Reasons describe project support, never the developer's machine ("this
  environment" is asserted absent), and blocked reasons for toolchain
  failures must name the mechanism (the op, the converter, the version).

## 2. The weights rule (read this before validating anything)

**Never make a parity claim from randomly initialized weights alone.** This
invalidated a whole round of Apple-format results once: a random-init
detection head emits a near-constant tensor whatever it is shown, because
the constant anchor grid dominates the output. Measured on the ONNX
reference between two very different probes: yolox moved 1.5e-09, rtmdet
8.9e-12, picodet exactly 0. A parity figure of 1e-08 against a reference
that moves 1.5e-09 is two constants agreeing. One family (yolo7) recorded
1.4e-07 on random init and turned out to be 2.9e-01 on trained weights.

The three qualifiers, permanently recorded at `support.py:822-851` next to
the numbers they govern:

1. **Non-degenerate weights.** Use published trained weights wherever a
   permissive checkpoint exists: `LibreYOLO("Libre<FAM><size>.pt")`
   auto-downloads via `get_download_url` (`models/base/model.py:635`), or
   call `download_weights` (`libreyolo/utils/download.py:263`); RF-DETR
   loads pretrained even with `model_path=None`. For families with no
   redistributable weights (FOMO, TEED/DexiNed, YOLO-NAS official), build a
   deterministic, license-clean non-degenerate state and disclose it in the
   `constraint`: the YOLO-NAS Core AI row runs 12 native training steps
   plus a 20x regression-head scale; the FOMO TensorRT harness runs an
   80-step training loop. Such rows must say "validates conversion, not
   accuracy".
2. **Input contract.** Feed each artifact the input its own contract
   expects. `_wrap_for_family` (`export/coreml.py:87-96`) embeds
   preprocessing for the Apple formats (yolox: x255 + RGB-to-BGR; rfdetr:
   ImageNet normalization) that ONNX does not embed. Handing both the same
   tensor compares two different functions and reads ~0.5 no matter how
   correct the conversion is. To learn what a wrapper does, wrap
   `nn.Identity()` in it and look at the output; never restate the math by
   hand.
3. **Sensitivity margin.** A result counts only if parity error is at
   least **100x below** how far the reference itself moves between two
   probes. Otherwise the honest verdict is INCONCLUSIVE, recorded as
   `experimental` with the measurement in the reason, never `validated`.

**Acceptance rule for delegated work**: any parity report (from an agent or
a human) must state (a) weights provenance, (b) the reference's measured
input-sensitivity, and (c) the parity figure. A report missing any of the
three is not evidence; send it back. "Same number of detections" is not a
parity claim at all.

## 3. The parity ladder (the double-tick contract)

Five steps, in order. The reference harnesses are
`tests/unit/test_detr_cpu_export_matrix.py:78-120` (CPU formats) and
`tests/e2e/test_tensorrt_round8.py`-style GPU suites; clone their shape.

1. **Export through the public API**: `model.export(format=..., imgsz=N,
   dynamic=False, simplify=False)` at the family's native canvas.
2. **Reload through the public factory**: `LibreYOLO(artifact)`. Assert
   `backend.model_family`, `backend.task`, `backend.imgsz` survived the
   metadata round trip.
3. **Raw parity against the exporter's prepared graph**, never the naked
   eager model:

   ```python
   exporter = OnnxExporter(model)  # or the format's exporter class
   with exporter._model_context(device, False, False, 1, (N, N)) as (wrapped, _):
       with torch.no_grad():
           expected = wrapped(tensor)
   actual = LibreYOLO(artifact)._run_inference(tensor.numpy())
   ```

   `_model_context` (`exporter.py:729-926`) applies the same per-family
   wrapper or `head.export = True` flag the artifact was traced with; a
   comparison that skips it measures the wrapper, not the conversion.
4. **Two probes + the sensitivity gate.** Run steps 3 with a second probe
   (`second = 1.0 - first`) and assert, as in
   `tests/e2e/test_tensorrt_round8.py:289-302`:

   ```python
   assert expected_signal > 1e-12                        # reference responds to input
   assert actual_signal > max(1e-12, 100.0 * parity_error)  # artifact is not a constant
   ```

   This is the assertion that turns "exported noise matches noise" reports
   into failures instead of false validations.
5. **Public `predict()` parity** with the task criteria in §4, on a real
   or deterministic image, through `backend.predict(...)` vs
   `native.predict(...)`.

Cross-step traps:

- **Output ordering.** Core AI returns a named dict whose order matches
  nothing; names are in metadata under `coreai_output_names`. Never pair
  outputs positionally (`export/coreai.py:30-37`).
- **DETR query order.** Graphs with in-graph top-k reorder near-tied
  queries under tiny float drift. Align queries with
  `scipy.optimize.linear_sum_assignment` on the geometric output before
  comparing (`test_detr_cpu_export_matrix.py::_align_query_outputs`); a
  genuinely wrong box still fails after alignment.
- **Canvas.** Use the family's native or documented-small canvas: FOMO is
  96, restore families crawl at 640 (RealESRGAN's 4x upscale), depth/edge
  families have their own divisors enforced in `_resolve_params`
  (`exporter.py:600-692`).
- **RF-DETR ONNX diverges from eager by design** at 640: ONNX disables the
  antialiased position-embedding resize (`torch.onnx.is_in_onnx_export`).
  Compare Apple-format artifacts against the prepared eager graph, not
  against the ONNX artifact.

## 4. Pass criteria by task

The doc-level thresholds live in `tools/gen_compat_table.py:74-82` and
render into `docs/export_support.md`; the executable versions are hardcoded
per test. Use these:

| Task | Raw-tensor criterion | `predict()` criterion |
|---|---|---|
| detect / obb | allclose (see tolerances below); DETR: after Hungarian query alignment | equal count; Hungarian-match boxes, then per match: box `rtol=2e-3, atol=1px`, score `rtol=2e-3, atol=2e-2`, class exact; each > 95% of matches |
| classify | logits cosine > 0.999 and equal argmax | probs cosine > 0.999 and equal top-1 |
| semantic | logits cosine > 0.999 and per-pixel argmax agreement > 0.95 | mask agreement > 0.95 |
| segment / panoptic | as detect, plus mask IoU > 0.95 | matched mask IoU > 0.95 |
| pose | as detect for boxes; keypoint L2 < 2px at native resolution | same |
| restore / depth | PSNR > 40 dB (peak-relative) | same on the output image / depth map |
| normal | mean angular error < 0.1 degree | same |
| point | heatmap cosine > 0.999 and peak locations within one output cell | point coords within 1px |
| gaze | yaw/pitch logits cosine > 0.999 | angles within task tolerance |

Tolerances by format, from the CPU matrix tests: ONNX `rtol=2e-3,
atol=2e-2`; TorchScript `rtol=1e-3, atol=1e-3`; Core AI worst relative
error `3e-4` with the 100x sensitivity margin; box rows may use "> 95% of
rows within tolerance" instead of allclose.

**Prohibitions**, each of which has produced a wrong verdict at least once:

- Never compare only detection **counts**.
- Never compare **positionally after top-k**: tie groups reorder; DEIMv2
  once read 8.2e-01 positional against 2.4e-07 properly matched.
- Never use **nearest-neighbour matching** to pair detections; it has
  faked failures repeatedly. Use `linear_sum_assignment` on a squared-
  distance cost over box geometry.
- Never "fix" detection order by sorting the outputs.

**Known gaps are recorded, not hidden**: a cell that converts but misses
the bar gets a `strict=True` xfail with a quantified reason ("92.3% of
aligned boxes meet the converted-runtime tolerance"), and stays
`experimental` with that measurement in the matrix reason.

## 5. Recording the result in the matrix

Promote with an `_add` that carries the evidence. Validated with a full
measurement note (`support.py:852-863` is the canonical short form):

```python
_add(
    "validated",
    ("dfine",), ("detect",), ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; trained LibreDFINEn weights are covered on "
        "macOS 27 by direct named-output parity with a 3e-04 tolerance "
        "and a 100x input-sensitivity margin"
    ),
)
```

Blocked with the mechanism named:

```python
_add(
    "blocked",
    ("rfdetr",), ("segment",), ("tflite",),
    reason=(
        "onnx2tf 2.4.x assigns an invalid NHWC layout to the "
        "segmentation-head Einsum (78 channels versus the required 256), "
        "so conversion fails."
    ),
)
```

Then:

1. `python tools/gen_compat_table.py` regenerates `docs/export_support.md`;
   `--check` is the CI gate
   (`tests/unit/test_export_support.py::test_generated_export_docs_are_current`).
   Never hand-edit the generated file. The README landing table is curated
   by hand separately (`libreyolo-update-readme` skill).
2. If you added a family or task, the committed
   `reports/export_inventory.json` must know it: regenerate with
   `tools/dump_model_inventory.py` **in a transformers-provisioned env**
   (it refuses partial overwrites; `--allow-family-removal` is the escape
   hatch).
3. New validated rows must land with the parity test that produced them
   (`support.py:63`: "New validated rows must land with a parity test").

## 6. Checklist: new family into an existing format

1. Matrix row via `_add` (or accept the format default from §1).
2. Tracing: either a family wrapper branch in
   `exporter.py::_model_context` (`:756-844`) with a `finally` restore, or
   rely on the generic `head.export = True` path (`:846-856`). It is one
   or the other, never both.
3. ONNX and ONNX-derived formats (TensorRT, OpenVINO, TFLite): an
   output-name branch in `onnx.py::export_onnx` (`:212-448`) keyed on task
   (preferred) or family. Opset: 13 default, 17 for DETR-tuple families
   and moge2 (`onnx.py:43-45`).
4. CoreML: add to `_SUPPORTED_FAMILIES` (`coreml.py:26-30`), extend
   `_wrap_for_family` if the family needs embedded preprocessing or an
   output adapter, and `_NMS_FREE_FAMILIES` if DETR-style. This allowlist
   is separate from the matrix; both need the row.
5. Core AI: extend `_wrap_coreai_contract` (`coreai.py:268-279`) and the
   membership sets `_ANCHOR_FREEZE_FAMILIES` / `_DARKNET_FAMILIES` /
   `_RTDETR_STATIC_FAMILIES` (`coreai.py:56-80`) as applicable. Core AI
   reuses CoreML's wrappers on purpose; they ARE the per-family contract.
6. TFLite: nothing beyond the matrix row (its allowlist is derived from
   the matrix, `tflite.py:26-53`) unless the graph needs surgery.
7. Shape gates: `_FIXED_SQUARE_EXPORT_FAMILIES` /
   `_RECTANGULAR_EXPORT_FAMILIES` (`exporter.py:142-159`) and
   `_RECTANGULAR_BACKEND_FAMILIES` (`backends/base.py:61`).
8. Backend: family branch in `BaseBackend._preprocess` and
   `_parse_outputs`, plus `_is_nms_free_family` (`backends/base.py:236-254`)
   membership for DETR-style and NMS-free families. Route through the
   shared helpers; do not fork them.
9. The parity test (§3), then flip the row to `validated` + `since`.
10. Regenerate docs (§5).

## 7. Checklist: a whole new format

Subclass `BaseExporter` in `exporter.py` declaring the class-attribute
contract (`exporter.py:236-244`: `format_name`, `suffix`, `requires_onnx`,
`supports_int8`, `supports_fp16`, `apply_model_half`,
`supports_embedded_nms`); registration is automatic via
`__init_subclass__`. Then: the format module under `libreyolo/export/`,
membership in `EXPORT_FORMATS` (`support.py:11`, which drives the docs
columns, `libreyolo formats`, and `get_support` validation), a
format-default branch in `get_support` if it should default blocked, an
alias in `_aliases` if needed, a `tools/export_env_check.py` row, a backend
under `libreyolo/backends/` plus an extension branch in
`libreyolo/models/__init__.py:280-344`, a pyproject extra (never a core
dep), and an e2e file marked `export_backend` + `experimental_backend`. A
new format starts experimental and stays there until parity tests say
otherwise.

Two sanctioned deviations: when the toolchain cannot be a pip dependency
(Hailo's proprietary compiler), ship a skill/doc for the two-stage flow
instead of a `format=` target. And a format may be **export-only by
design**: Core AI has no backend and no dispatch branch, its validated rows
are numeric-parity claims about the exported graph, not a promise that
`predict` will run it (`export/coreai.py:21-28`). If you do that, say so in
the module docstring exactly as coreai.py does.

## 8. Predict-back wiring

Dispatch is by file extension or directory shape in
`libreyolo/models/__init__.py:280-344`: `.onnx`, `.torchscript`,
`.tflite`, `.engine`, OpenVINO dir (`model.xml`), `.mlpackage`, NCNN dir
(`model.ncnn.param` + `.bin`). Anything else falls through to the `.pt`
loader, which is why `.aimodel` "loads" and then fails: no Core AI backend
exists, deliberately.

`BaseBackend.__init__` (`backends/base.py:300-380`) is the runtime
metadata contract: family, size, task, names, imgsz, plus per-task extras
(classify `crop_pct`/`interpolation`, pose keypoint shape, gaze bin
parameters). Defaults exist for absent metadata but exported artifacts
should carry the real values; the exporter side is `_build_metadata` /
`_build_onnx_metadata` (`exporter.py:978-1111`). Metadata transport per
format: ONNX metadata_props, TorchScript `_extra_files`, TFLite sidecar
`<path>.tflite.json`, TensorRT sidecar `<engine>.json` (precision is
rewritten to what the build actually achieved), OpenVINO/NCNN
`metadata.yaml`, CoreML `user_defined_metadata`.

Embedded-NMS ONNX: the backend re-decodes the raw head output by name
(`backends/base.py:884-905`) rather than trusting the post-NMS tensor, so
native clipping parity holds on non-square inputs. If you add an NMS-free
family and skip `_is_nms_free_family`, exported backends will wrongly
apply NMS on top of set predictions: that is a silent wrong-results bug,
not a crash.

## 9. Per-format quirks (the ones that cost hours)

- **ONNX**: Python API defaults `dynamic=True`; static-shape consumers
  need `dynamic=False`. `nms=True` embeds NMS (YOLO9/yolox only, changes
  the output contract). The raw `[1, 4+nc, N]` head tensor is an external
  contract (a C++ consumer and `tests/e2e/test_sam3dbody_contract.py`
  guard it): do not reshape casually. onnxsim is skipped on macOS-arm
  where it crashes (`onnx.py:66-80`).
- **TorchScript**: traces with `check_trace=False` on purpose (wrappers
  cache shape-dependent anchors on first forward; a re-trace check false-
  negatives). Metadata rides in `_extra_files`.
- **TensorRT**: consumes the intermediate ONNX, no family logic of its
  own. Engines are GPU-arch-specific, never redistributable. fp16 is where
  drift appears first; compare fp32 before blaming the model. Repeated
  builds of the same engine can straddle a threshold: promotion needs the
  strict gate to hold across builds, otherwise record an experimental
  parity floor (the PIDNet precedent in the round-8 suite).
- **OpenVINO**: directory artifact + `metadata.yaml`. Request FP32
  execution explicitly when measuring parity on CPU; BF16 execution hints
  can drift results across families.
- **NCNN**: goes through PNNX with an ONNX fallback. DETR families are
  blocked (`topk` not in the op registry). The YOLOX `Focus` rewrite
  (`ncnn.py:31-88`) permutes conv weights at export and restores them
  after; that snapshot/restore pattern is the sanctioned fix whenever an
  export must mutate model state.
- **TFLite**: onnx2tf chain, Python 3.12+. The allowlist derives from the
  matrix. RF-DETR needs the GridSample-to-Gather rewrite because TFLite's
  GatherNd silently accepts the model and produces wrong scores: the worst
  failure class, wrong-but-running. When a converter "succeeds", parity is
  still mandatory.
- **CoreML**: `torch.jit.trace` based, macOS-only end to end
  (`test_coreml_roundtrip.py` skips elsewhere). Hard allowlist +
  `_wrap_for_family` + static-eval preparations that freeze anchor grids
  and position embeddings before tracing (`coreml.py:99-176`). Validation
  requires a macOS prediction run (ADR 0011).
- **Core AI**: `torch.export` pipeline (`export -> run_decompositions ->
  TorchConverter -> optimize -> save_asset`). Read the coreai.py module
  docstring before touching it; it is the spec. Graph preparation is a
  scoped ExitStack that always restores live state
  (`coreai.py:563-587`). Upstream converter bugs get version-pinned shims
  in `coreai_compat.py`, patched from our side, never by editing
  site-packages; note the resolver tables capture functions by value at
  import, so the shim rewrites the tables too. Ruled out, do not retry:
  call-order interpolate replays (a deleted `_bake_bicubic_interpolate`
  broke every export), swapping bicubic for bilinear (silently changes
  outputs), decomposing `aten.as_strided` (fix the model side instead).
- **Apple formats generally**: parity evidence requires real macOS
  hardware. If you drive a remote Mac over SSH, batch all remote work into
  one connection per session, never leave a retry poller running against
  it, and reuse the machine's existing venv and clone.
- **Rectangular imgsz** is only supported by the formats in
  `_RECTANGULAR_EXPORT_FORMATS` (`exporter.py:160-169`); square-check
  errors elsewhere are intentional.

## 10. Testing and CI reality

Markers (`pyproject.toml`): `export_backend`, `supported_backend` (ONNX is
the release-gated one), `experimental_backend`, plus per-format markers
(`onnx`, `tensorrt`/`trt`, `openvino`, `ncnn`, `tflite`, `coreml`).

What actually runs where:

- The **PR gate** (`unit-tests.yml`) runs the unit parity matrices
  (`test_detr_cpu_export_matrix.py`, `test_yolo_edge_export_matrix.py`,
  `test_classifier_edge_export_matrix.py`,
  `test_darknet_edge_export_matrix.py`), the matrix self-gate
  (`test_export_support.py`), and the docs `--check`. Do not pin
  `OMP_NUM_THREADS` there: thread count changes float reduction order and
  flips the strict DEIM xfail to XPASS.
- **Nightly e2e excludes export backends** (`-m "not export_backend"` in
  the nightly workflows and Makefile). e2e export parity is on demand:
  `make test_e2e MARKERS='export_backend and not experimental_backend'`
  (see `libreyolo-run-e2e-tests`).
- Apple-format e2e needs macOS hardware; a green Linux CI says nothing
  about CoreML/Core AI cells.

Consequence: **a validated cell whose test never executes anywhere is a
claim, not evidence.** When you promote a cell, know which workflow (or
which documented manual run) executes its test, and prefer wiring it into
one that runs.

## 11. Campaign mode: filling matrix gaps with agents

The proven shape for coverage campaigns (audit, then rounds):

1. **Baseline**: generate the current matrix view (`docs/export_support.md`
   plus, if auditing evidence, which tests are CI-executed vs manual).
   Pick a round of ~10 cells, grouped by format or by family so one
   environment serves the whole round.
2. **One agent per cell or per small cell-group.** Each agent gets: the
   cell, the §2 weights rule, the §3 ladder, the §4 criteria for its task,
   and the required report format: verdict (promote / experimental with
   measured floor / blocked with mechanism) plus weights provenance,
   sensitivity figure, and parity figure. **Reject any report missing
   those three numbers** (§2 acceptance rule); that rule is what catches
   agents that skip downloading weights and validate noise.
3. **Verify before recording**: re-run at least the sensitivity assert for
   promotions; nearest-neighbour-matched or count-only evidence does not
   promote anything.
4. **Record** every assessed cell in `support.py` (§5): promotions with
   `since` + constraint, misses as `experimental` with the quantified gap,
   hard failures as `blocked` naming the mechanism. Cells you assessed and
   could not conclude stay experimental with the measurement; silence is
   the only forbidden outcome.
5. **Regenerate docs**, run the matrix self-gate, land the round as one
   reviewable unit with its tests.

Round discipline learned the hard way: keep rounds small enough to verify
each cell's evidence; one wrong "validated" costs more than ten unfilled
cells, because the matrix is what users and the README trust.

## 12. Debugging "the exported model is wrong"

Work the pipeline in order; the bug class differs by stage:

1. **Export succeeded but predictions differ from `.pt`**: run both on the
   same image and compare stage by stage. Preprocessing first (the backend
   reimplements letterbox in numpy; a 1px offset here looks like "slightly
   wrong boxes"), then raw tensor closeness against the prepared graph
   (§3, with the sensitivity gate), then postprocess (NMS thresholds,
   top-k count, class map, `_is_nms_free_family` membership).
2. **Export itself fails**: check the extra is installed
   (`tools/export_env_check.py` / `libreyolo checks`), then whether the
   cell is blocked in the matrix (the error message cites the reason and
   validated alternatives), then whether the op is task-specific. Only
   then debug the converter stack, and pin versions before patching code;
   if the fix is an upstream-bug workaround, follow the `coreai_compat.py`
   shim discipline.
3. **Backend load fails on a file that exported fine**: metadata
   round-trip bug. Inspect what was embedded and what
   `BaseBackend.__init__` expects; fix the *pair*, and add the missing key
   to both sides plus a unit test.
4. **Wrong but running** (the TFLite GatherNd class): only §3 catches
   this. A conversion that emits no error is evidence of nothing.

## Related

- `skills/use-libreyolo/`: the user-facing export + exported-inference API.
- `skills/libreyolo-export-hailo/`: the no-format precedent, done right.
- `skills/libreyolo-checkpoint-metadata/`: the runtime-metadata twin
  surface for `.pt` checkpoints.
- `skills/libreyolo-write-e2e-tests/` and `libreyolo-run-e2e-tests`:
  markers and commands for export coverage.
- `docs/adr/0011-export-support-tiers.md`: the tier contract.
- `docs/testing.md`: which backends gate releases vs run experimentally.
