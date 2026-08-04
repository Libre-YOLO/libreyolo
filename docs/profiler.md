# Profiler (training + inference)

A built-in profiler that answers one question fast: **where does the time go, and
is the GPU actually busy?** It runs from a single flag (training) or a single
command (inference), prints a diagnosis, and exposes a CLI an agent can drive to
optimise speed. It is a **measurement tool** — it reports the bottleneck and the
levers; you make the change and re-measure.

Most of this page covers **training** (`model.train(profile=True)` /
`libreyolo profile run`); the [inference profiler](#inference-profiling--libreyolo-profile-infer)
(`libreyolo profile infer`) applies the same loop to the predict path.

## Quick start

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t.pt", size="t")
model.train(data="coco1000", profile=True)
```

`profile=True` profiles a short window of real training steps (a few warmup
steps, then a measured window), prints a report, writes its artifacts, and then
**keeps training**: the hooks are dropped once the window closes, so the rest
of the run pays nothing. It is safe to combine with a real (or resumed)
training run.

To profile *without* training to the end, pass `profile_then_stop=True`: the
run stops right after the window, so it takes seconds instead of a full
training. This is what `libreyolo profile run` uses. A stopped run is
benchmark-only: the partial epoch is neither validated nor checkpointed, and
the log says so instead of reporting a completed training.

It is **zero overhead when off** (the default); the hot loop's hooks short
-circuit to a no-op. Profiling is ignored under distributed (DDP) training.

### Family coverage

**Training** profiling is supported across every trainable family — all of
**g0, g1 and g2** in `libreyolo/models/registry.py`, every task (detect,
point, classify, semantic, restore). That includes the DFINE/DEIM-family
trainers, whose copied training loops carry the same hooks as the base loop.
`tests/e2e/test_profile_training_families.py` runs a real profiled training
per family and is kept in lockstep with the registry, so a family cannot join
a trainable group without profiler coverage. Families whose training is gated
behind `allow_experimental=True` (ec, yolo7, rtmdet, picodet, fomo) need that
flag alongside `profile=True` — or `--allow-experimental` on the CLI.

**Inference** profiling (`libreyolo profile infer`) drives the shared
predict-stage contract, verified per family for every g0/g1 family plus the
g2 detect/point families (`tests/e2e/test_profile_inference_families.py`);
g3 inference-only families use the same stage contract and work through the
same path. Classify/semantic/restore predict paths are best-effort: the
stage split still measures, but the "postprocess = NMS" verdict semantics
are detection-specific.

Out of scope: g4 (museum, frozen) and the s tier (SAM / open-vocab / VLM
siblings), whose predict surfaces differ from the staged contract this tool
measures.

### What it prints

```
GPU util 29%  (89 ms GPU-busy / 304 ms step)  |  8187 kernels/step @ ~11us
peak VRAM 5540 MB  |  Tensor Cores 18% of GPU time  |  memcpy 3.0 ms/step
>> VERDICT: HOST/LAUNCH-BOUND — GPU only ~29% busy — fed too slowly...
   Levers: larger batch, fewer per-step .item()/syncs, CUDA graphs, op fusion.
GPU kernel mix:  gemm/conv 39% · elementwise 26% · batchnorm 18% · layout 11%
top kernels:     cudnn bn-bwd 10% · cutlass conv 5% · nchw->nhwc(x496) 4% ...
```

The **verdict** is one of three:

| verdict | meaning | typical levers |
|---|---|---|
| `dataloader-bound` | the GPU waits on input data | more `workers`, `cache='ram'/'disk'`, lighter aug, larger batch |
| `host / launch-bound` | the GPU is fed too slowly (tiny kernels, per-step syncs) | larger batch, fewer `.item()`/syncs, CUDA graphs, op fusion |
| `compute-bound` | the GPU is saturated | AMP/bf16, a larger model |
| `memory-pressure` | allocator thrash / VRAM at the edge — util & throughput here are unreliable | lower batch, fix fragmentation; don't run at the OOM edge |

GPU utilisation is measured honestly: kernel busy-time over the *real* (unsynced)
step time — not wall-clock hand-waving.

### Artifacts (written to the run directory)

| file | what |
|---|---|
| `timeline.html` | self-contained CPU/GPU timeline (auto-opens) + analysis panel — no external viewer |
| `profile_trace.json` | the raw `torch.profiler` Chrome trace (also loads in Perfetto/Nsight) |
| `profile_summary.json` | the computed metrics + verdict |
| `profile.json` | **self-contained** analysis (metrics + verdict + kernels + phases) — the one file to copy and `compare` |

### Config knobs

| key | default | meaning |
|---|---|---|
| `profile` | `False` | enable the profiler (training continues after the window) |
| `profile_then_stop` | `False` | stop the run right after the profile window (benchmark mode; no checkpoint for the partial epoch) |
| `profile_warmup` | `5` | leading steps discarded |
| `profile_steps` | `20` | measured steps |
| `profile_trace` | `True` | emit the Chrome trace + timeline |
| `profile_open` | `True` | auto-open `timeline.html` in a browser |

When gradient accumulation is enabled (`nbs > batch`), the profiler rounds the
warmup and measured windows up to accumulation boundaries so the captured window
contains complete optimizer steps.

### Gotcha: augmentation regime

The profiler measures **epoch 0**. If you use a tiny `epochs` with the default
`no_aug_epochs`, mosaic/mixup can be disabled for the window (the no-aug schedule
kicks in immediately), so you would profile a *lighter* dataloader than you
train with. To profile the real augmented pipeline, set `no_aug_epochs=0` — or
use `libreyolo profile run`, which sets `no_aug_epochs=0` and runs enough
epochs to fill the measurement window automatically.

## The CLI — `libreyolo profile`

The profiler is also a toolbox of focused commands so an agent (or you) can
inspect the trace at any abstraction level. Each command writes results to
stdout and supports `--json`.

```
libreyolo profile run     coco128 --weights LibreYOLO9t.pt --size t [--repeat 3 --warmup 5 --steps 20]
libreyolo profile infer   img_or_dir --weights LibreYOLO9t.pt --size t [--batch 1 --half --runs 100]
libreyolo profile summary <profile.json>          # util, verdict, host overhead, kernel mix, top kernels
libreyolo profile get     <profile.json> <field>  # ONE metric (img_per_s, forward_gpu_ms, host_overhead_ms, ...)
libreyolo profile phases  <profile.json>          # per-phase gpu/wall ms + kernel & op counts
libreyolo profile kernels <profile.json> [--phase forward --category gemm --grep bn --tensorcore --sort time --top N]
libreyolo profile ops     <profile.json> [--phase backward --top N]   # aten/autograd ops by CPU time
libreyolo profile what-if <profile.json> [--remove-category layout | --remove-launches 3300]   # project a fix
libreyolo profile compare <before.json> <after.json>   # did it help? img/s + ms/image + significance
```

Every command takes either the **self-contained `profile.json`** (recommended — copy
it freely) or a raw `profile_trace.json`. `get <file>` with no field lists every
metric. `run` follows normal training defaults, including AMP on supported CUDA
devices; pass `--no-amp` to force fp32. For families whose training is gated as
experimental (e.g. ec), pass **`--allow-experimental`**: a profile run trains a
few steps and writes no checkpoint, so the unvalidated-convergence caveat
behind the gate does not apply. Use **`run --repeat N`** for mean±stdev
img/s — a single run *lies* when the step is launch-bound (the bottleneck itself
makes the step time noisy). Repeated runs emit an aggregate `profile_repeat.json`
for `compare`, plus per-trial `prof_*/profile.json` files for drill-down.

### Agent loop

```
libreyolo profile run coco128 --repeat 3                 -> img/s 50 +/- 1.4 | host / launch
libreyolo profile get  profile.json gpu_util             -> 0.29        # GPU 71% idle
libreyolo profile phases profile.json                    -> forward 42/120, unphased 6gpu/11k ops
libreyolo profile ops  profile.json --phase unphased --top 5  -> aten::lerp_ (EMA) ...
libreyolo profile what-if profile.json --remove-launches 3300  -> img/s 50 -> ~69 (+38%)   # worth it
#   ...make the change, re-run with --repeat, then:
libreyolo profile compare before.json after.json         -> img/s +28%  [significant]
```

Train → read → drill → **estimate** (`what-if`) → change → **prove it** (`compare`,
with significance), until images/sec is maxed. The two things that make this safe
for an autonomous agent: `--repeat` (so a noisy single run can't lie) and
`what-if` (so it triages before rewriting code).

## Inference profiling — `libreyolo profile infer`

The same tool also profiles the **predict path**, for deployment/latency tuning.
It drives the model's `preprocess -> forward -> postprocess` stages directly (no
edit to inference — zero overhead on normal predict) over a fixed source, and
emits the **same `profile.json`**, so every lens above (`summary`, `get`,
`kernels`, `ops`, `phases`, `compare`) works on inference profiles unchanged.

```
libreyolo profile infer path/to/img.jpg --weights LibreYOLO9t.pt --size t --batch 1
```

Source defaults to a bundled sample image (so it runs standalone). It reports
**latency p50/p90/p99**, **throughput** (img/s at `--batch`), and a **stage
split** — preprocess / forward / postprocess(**NMS**):

```
latency  p50 4.9 ms | p90 5.4 ms | p99 5.6 ms  |  204 img/s
stages   preprocess 1.2 ms | forward 3.3 ms | postprocess/NMS 0.4 ms
>> COMPUTE — forward dominates the step (3.3 ms); GPU ~86% busy
```

The **inference verdict** is one of:

| verdict | meaning | typical levers |
|---|---|---|
| `nms / postprocess` | NMS dominates the step | lower `--max-det`, higher `--conf`, fewer `classes`, or an end-to-end (NMS-free) model |
| `preprocess` | CPU resize/letterbox dominates | larger `--batch`, faster decode, GPU preprocessing |
| `compute` | forward dominates (GPU saturated, or CPU) | larger batch, `--half`, a smaller model, or export (ONNX/TensorRT) |
| `host / launch` | forward dominates but the GPU is underused (CUDA) | larger `--batch`, or a bigger model to fill the GPU |

Knobs that change the measured work: `--batch`, `--imgsz`, `--half` (fp16
forward, CUDA), `--conf`/`--iou`/`--max-det` (how much NMS runs), and
`--runs`/`--warmup` (window size). `--repeat N` gives mean±stdev throughput for a
significant `compare`, aggregated into `infer_repeat.json`.
