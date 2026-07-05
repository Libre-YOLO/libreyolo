---
name: libreyolo-profiling
description: >-
  Diagnose and fix slow LibreYOLO with the `libreyolo profile` CLI — both
  TRAINING throughput (`profile run`) and INFERENCE latency (`profile infer`).
  Use whenever training feels slow, GPU utilization is low, images/sec is
  disappointing, inference/predict latency is too high, a run is dataloader- /
  host-launch- / NMS- / preprocess-bound, or someone wants to optimize step
  time, batch size, throughput, or p50/p90/p99 latency. Teaches the
  profile → diagnose → change → compare loop an agent runs to push speed to the
  max. This is for SPEED, not accuracy (mAP).
---

# Profile & speed up LibreYOLO (training + inference)

`libreyolo profile` answers one question fast: **where does the time go, and is
the GPU starved?** It is a **measurement tool** built to be driven by an agent
in a loop — every subcommand takes `--json`, and both entry points write a
self-contained `profile.json` you can copy and `compare`.

**The profiler only measures. It never changes your model.** You read the
verdict, *you* change one thing (a config value or code), then you re-measure.
Don't wait for the tool to auto-tune — that's your job.

The CLI is **self-describing**. Never guess a flag — run
`libreyolo profile --help` and `libreyolo profile <sub> --help`. Run
`libreyolo profile get <profile.json>` with no field to list the exact metric
names for the installed version.

## The loop

```
run|infer  →  summary  →  (kernels|ops|phases)  →  change ONE thing  →  compare  →  repeat
```

Both `run` (training) and `infer` (inference) emit the same `profile.json`, so
`summary/get/phases/kernels/ops/compare/what-if` work on either.

## A. Training throughput — `profile run`

```bash
libreyolo profile run <data> --weights LibreYOLO9t.pt --size t --repeat 3 --json
```

- `<data>` is a dataset yaml/name (e.g. `coco128`). `--batch -1` auto-fits ~70% VRAM.
- **Always `--repeat 3`+** — a single run *lies* when launch-bound; `--repeat`
  gives mean ± stdev and is what makes a later `compare` significant. The
  aggregate is `runs/profile/profile_repeat.json` — use **that** path (not a
  per-trial `prof_0/profile.json` sibling).

Read the **verdict**, then pull the matching lever (then re-measure):

| Verdict | Meaning | You change |
|---|---|---|
| **dataloader** | GPU waits on input (~≥20% of step) | more `workers`, `cache='ram'`/`'disk'`, lighter aug, larger batch |
| **host / launch** | GPU only partly busy — fed too slowly | **larger micro-batch** (amortizes launches), fewer per-step `.item()`/syncs, CUDA graphs, op fusion |
| **compute** | GPU saturated (~≥80% busy) | already GPU-bound — AMP/bf16, or accept it / change model size |
| **memory-pressure** | VRAM thrash (util reads >100%) | **lower batch**, reduce activation memory — util/img/s here are *unreliable* |

Highest-value, lowest-effort win: **host/launch-bound → raise the micro-batch.**
(yolov9t on an RTX 5070 Ti → micro-batch 36 = **+142% img/s**, painless.)

## B. Inference latency — `profile infer`

```bash
libreyolo profile infer <image-or-dir> --weights LibreYOLO9t.pt --size t --batch 1 --json
```

- Source defaults to a bundled sample image, so `libreyolo profile infer --weights X`
  works standalone. `--half` uses fp16 (CUDA). `--runs`/`--warmup` size the window.
- Reports **latency p50/p90/p99**, **throughput** (img/s at `--batch`), and a
  **stage split**: preprocess / forward / postprocess(**NMS**).
- `--conf`, `--iou`, `--max-det` change how much NMS work happens — the knobs to
  vary when NMS-bound.

| Verdict | Meaning | You change |
|---|---|---|
| **nms / postprocess** | NMS dominates the step | lower `--max-det`, higher `--conf`, fewer `classes`, or an end-to-end (NMS-free) model |
| **preprocess** | CPU resize/letterbox dominates | larger `--batch`, faster decode, GPU preprocessing |
| **compute** | forward dominates (GPU busy, or CPU) | bigger batch, `--half`, a smaller model, or export (ONNX/TensorRT) |
| **host / launch** | forward dominates but GPU underused (CUDA) | larger `--batch`, or a bigger model to fill the GPU |

## Drill (shared, when the verdict isn't enough)

```bash
libreyolo profile phases  <profile.json>              # stage/phase split
libreyolo profile kernels <profile.json> --top 20     # worst GPU kernels (--category --grep --tensorcore --phase)
libreyolo profile ops     <profile.json> --top 20     # aten ops by host time
libreyolo profile get     <profile.json> latency_p50_ms   # one metric, tight loops
```

Low `tensorcore_pct` on a compute-bound fp32 run → AMP/bf16/`--half` helps. Big
`layout / copy` share → channels_last. Many tiny elementwise kernels → fusion.

## Change one thing, then prove it helped

Change **one** lever, re-run with the same `--repeat`/`--runs`, then:

```bash
libreyolo profile compare <before.json> <after.json>
```

`compare` reports the delta **and a significance call**. "single run — use
--repeat N" means the delta is noise; repeat before trusting it.

## Gotchas

- **The tool won't change your config — you do, then re-measure.**
- **Always `--repeat` for training** (one run is noise, esp. launch-bound).
- **Compare the aggregate**, not a per-trial sibling.
- **Under memory-pressure, trust the verdict, not raw util** (thrash inflates it).
- **This measures speed, not accuracy** — validate mAP with `libreyolo val` after
  changing batch/LR/aug.
