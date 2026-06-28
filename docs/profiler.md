# Training profiler

A built-in profiler that answers one question fast: **where does each training
step's time go, and is the GPU actually busy?** It runs from a single flag,
prints a diagnosis, opens a self-contained timeline, and exposes a CLI an agent
can drive to optimise a training loop.

## Quick start

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t.pt", size="t")
model.train(data="coco1000", profile=True)
```

`profile=True` profiles a short window of real training steps (a few warmup
steps, then a measured window), prints a report, writes its artifacts, and
**stops early** — so a profile run takes seconds, not a full training.

It is **zero overhead when off** (the default); the hot loop's hooks short
-circuit to a no-op. Profiling is ignored under distributed (DDP) training.

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

GPU utilisation is measured honestly: kernel busy-time over the *real* (unsynced)
step time — not wall-clock hand-waving.

### Artifacts (written to the run directory)

| file | what |
|---|---|
| `timeline.html` | self-contained CPU/GPU timeline (auto-opens) + analysis panel — no external viewer |
| `profile_trace.json` | the raw `torch.profiler` Chrome trace (also loads in Perfetto/Nsight) |
| `profile_summary.json` | the computed metrics + verdict |

### Config knobs

| key | default | meaning |
|---|---|---|
| `profile` | `False` | enable the profiler |
| `profile_warmup` | `5` | leading steps discarded |
| `profile_steps` | `20` | measured steps |
| `profile_trace` | `True` | emit the Chrome trace + timeline |
| `profile_open` | `True` | auto-open `timeline.html` in a browser |

### Gotcha: augmentation regime

The profiler measures **epoch 0**. If you use a tiny `epochs` with the default
`no_aug_epochs`, mosaic/mixup can be disabled for the window (the no-aug schedule
kicks in immediately), so you would profile a *lighter* dataloader than you
train with. To profile the real augmented pipeline, set `no_aug_epochs=0` — or
use `libreyolo profile run`, which does.

## The CLI — `libreyolo profile`

The profiler is also a toolbox of focused commands so an agent (or you) can
inspect the trace at any abstraction level. Each command writes results to
stdout and supports `--json`.

```
libreyolo profile run     --data coco1000 --weights LibreYOLO9t.pt --size t   # train+profile -> trace
libreyolo profile summary <trace>                 # util, verdict, kernel mix, top kernels
libreyolo profile get     <trace> <field>         # ONE metric (img_per_s, forward_gpu_ms, ...)
libreyolo profile phases  <trace>                 # per-phase gpu/wall ms + kernel & op counts
libreyolo profile kernels <trace> [--phase forward --category gemm --grep bn --tensorcore --sort time --top N]
libreyolo profile ops     <trace> [--phase backward --top N]   # aten/autograd ops by CPU time
libreyolo profile compare <before> <after>        # did the change help? (img/s, util, mix deltas)
```

`get <trace>` with no field lists every available metric.

### Agent loop

```
libreyolo profile get trace.json bound            -> host / launch
libreyolo profile get trace.json gpu_util         -> 0.29
libreyolo profile phases trace.json               -> forward 42/120, unphased 6gpu/11k ops
libreyolo profile ops trace.json --phase unphased --top 5   -> aten::lerp_ (EMA), .item ...
#   ...change config/code, re-run, then:
libreyolo profile compare before.json after.json  -> img/s +40%, gpu_util 0.29 -> 0.51
```

Train → read → drill → change → **prove it helped**, until images/sec is maxed.
