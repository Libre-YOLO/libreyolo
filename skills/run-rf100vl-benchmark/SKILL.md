---
name: run-rf100vl-benchmark
description: >-
  Run the RF100-VL (Roboflow 100-VL) detection benchmark on LibreYOLO models
  end to end: dataset download with version locking, per-dataset fine-tuning,
  protocol-conformant pycocotools evaluation at maxDets 500, artifact
  publishing, and multi-GPU execution on vast.ai (the default compute). Use
  when someone wants to "run RF100-VL", "benchmark on Roboflow 100",
  reproduce or extend published RF100-VL numbers, add a model family to the
  campaign, asks why campaign GPUs look underutilized or the ETA looks wrong,
  or needs to decide stop-vs-destroy / packing depth. Training and eval run
  in the vision-analysis-benchmark harness; this skill holds the protocol,
  locked decisions, compute playbook, and the operational lessons that
  change money and wall-clock.
---

# Run the RF100-VL benchmark

RF100-VL: 100 real-world detection datasets from Roboflow Universe (164,149
images, 564 classes, 7 domains), paper arXiv 2505.20612 (NeurIPS 2025
Datasets and Benchmarks). Each dataset ships fixed `train`/`valid`/`test`
splits in COCO JSON. There is no official runner; the harness referenced
below is the runner.

Exact install/launch commands: harness
`docs/rf100vl-operator-runbook.md`. Vast rental mechanics (2FA/TOTP, launch,
destroy): `skills/launch-serverless-gpu-job` (Vast section) or the personal
`vast-launch` skill. This skill is the protocol plus the operational
knowledge those two do not cover.

## Protocol (locked decisions)

| Decision | Value | Authority |
|---|---|---|
| Scoring | pycocotools, `maxDets=500`, per-dataset `test` split | paper |
| Headline metric | AP 0.50:0.95, unweighted mean over the 100 datasets | paper |
| Per-domain means | published alongside (7 domains) | our addition |
| Split discipline | train on `train`, select on `valid`, report on `test` | Roboflow reference code |
| Selection | validate every epoch, EMA weights, keep best AP50:95 | Roboflow reference code |
| Epoch budget | fixed 100, early stopping DISABLED (`patience: 0`) | Roboflow reference code; the harness enforces it |
| Effective batch | 16 (physical batch x gradient accumulation) | Roboflow reference code |
| Precision | fp32; bf16 only via explicit `amp_dtype`; never fp16 autocast | LibreYOLO policy |
| Eval thresholds | conf 0.001; NMS IoU 0.65 for NMS families, identical at selection and final eval; DETR families are NMS-free top-k | LibreYOLO policy |
| Seed | 0 | LibreYOLO policy |
| Recipes | one pinned JSON per family in the harness (`va_bench/recipes/rf100vl/`), sha recorded in every run, same recipe for all 100 datasets | LibreYOLO policy |
| Execution (YOLO) | `cuda_graph: true` and `cache: "disk"` in the recipe protocol when the installed LibreYOLO supports them (bit-identical; covered by recipe hash) | measured 2026-08 |

Never report toolkit-native trapezoidal mAP; it inflates up to 2.7 AP on
RF100-VL versus pycocotools (paper, App B). LibreYOLO validation is
pycocotools-based already; the 500 cap is the opt-in `eval_max_det` kwarg
(`model.val(data=..., split="test", eval_max_det=500)`). Defaults for normal
users are unchanged (AP at maxDets 100) and test-locked.

Validation frequency is protocol-mandated every epoch (gdino
`val_interval=1`; rt-detr / d-fine / lw-detr call `evaluate()` unconditionally,
lw-detr twice). The knob that exists is validation COST (image cache, hoisted
validator, graphed val forward), not frequency. LibreYOLO's own
`TrainConfig.eval_interval` defaults to 10 — a short smoke never validates
unless you pass `eval_interval=1`.

## Where the work happens

Repo `LibreYOLO/vision-analysis-benchmark` (harness branch `rf100vl-harness`).

```bash
# per-dataset fine-tuning: one worker per lane, subprocess children,
# atomic status files, resume, timeouts, dense-dataset OOM fallback
va-bench rf100vl-train --data-dir ./rf100-vl --weights-root ./rf100vl-weights \
  --gpus 0,1,2,3,4,5,6,7 --jobs-per-gpu 3

# per-dataset test-split eval
va-bench rf100vl --all --data-dir ./rf100-vl --weights-root ./rf100vl-weights

# both plus checks and rendering as ONE resumable command
va-bench rf100vl-campaign --model yolov9t --data-dir ./rf100-vl \
  --weights-root ./rf100vl-weights --gpus 0,1,2,3,4,5,6,7 --jobs-per-gpu 3
```

`--jobs-per-gpu` is the single biggest throughput lever. It is set by VRAM
per lane and CPU headroom, not by GPU compute. Measured (yolov9t ≈ 6.2
GB/lane, yolov9s ≈ 10.8 GB/lane before val-graph static buffers; re-measure
after stack changes). CUDA graphs improve packing because they remove host
launch contention between lanes sharing a GPU.

Supporting verbs: `rf100vl-preflight`, `rf100vl-dash` (pass `--data-dir` or
queued datasets show no image counts and the ETA is size-blind),
`rf100vl-report`.

**Open the dashboard for the operator as soon as training starts; do not
wait to be asked.** A campaign runs for hours on a rented box and the panel
is the only view of it that is not a wall of log text: per-GPU lanes, the
100-dataset grid, per-dataset loss and mAP curves. Serve it on the box and
forward it, then open the local URL in their browser:

```bash
# on the box, in its own tmux session
va-bench rf100vl-dash --state-root <weights-root>/.state --data-dir <data-dir>
# from the operator's machine (keep the tunnel alive for the whole run)
ssh -N -L 8877:127.0.0.1:8877 -p <PORT> root@<HOST>
# then open http://127.0.0.1:8877/
```

Bind the default `127.0.0.1` on the box and reach it through the tunnel;
never bind `0.0.0.0` on a rented host. Re-open the tunnel after any box
restart, and re-check the URL answers rather than assuming: an `ssh -L` that
loses its forward exits silently, and a dead tunnel looks exactly like a
stalled campaign. If a previous tunnel still holds port 8877, a new one fails
with `ExitOnForwardFailure` while the old one keeps working, so verify by
fetching the page, not by watching the ssh process.

- **Dataset, fast path.** Pull [`LibreYOLO/rf100-vl`](https://huggingface.co/datasets/LibreYOLO/rf100-vl):
  100 per-dataset tars + lock files. Prefer `max_workers=32` and stay logged
  in (`HF_TOKEN`) for the authenticated rate limit. Do not cargo-cult
  `huggingface_hub[hf_transfer]` / `HF_HUB_ENABLE_HF_TRANSFER=1` on hub 1.x.
- **Dataset, canonical path.** `--download` wraps the `rf100vl` pip package
  (`ROBOFLOW_API_KEY`); use it to rebuild/verify the HF copy. Licensing is in
  `va_bench/data/rf100vl_licenses.json`.
- Weights: `best.pt` at `<weights_root>/<dataset>/<weight_file>`.
- A capability guard aborts on builds without `eval_max_det`/`amp_dtype`.
  `cuda_graph` / `cache` are reported, not required — missing them runs the
  protocol correctly, just slower.
- Flag lists: harness README / `--help` win over this skill.

## Decisions BEFORE dataset one (one-way doors)

The run signature hashes the recipe. **Any recipe change
(`cuda_graph`, `cache`, epochs, imgsz, …) means a fresh campaign** — banked
checkpoints under the old recipe cannot be resumed. Enabling `cuda_graph`
mid-campaign once orphaned 973 banked epochs.

1. Recipe final? (`va_bench/recipes/rf100vl/*.json`)
2. LibreYOLO commit pinned and *actually* installed?
   `pip install --upgrade` on a git URL **silently no-ops when the version
   string is unchanged** and reports success. Use
   `pip install -q --force-reinstall --no-deps "git+..."`, then prove the
   installed `TrainConfig` has the fields the recipe sets.
3. Vast TOTP seed saved (`~/.config/vastai/vast_totp_seed`)? A 2FA session
   expiring mid-campaign once forced stopping a billing box through an
   unverifiable path.
4. Shakedown done on the current stack? (below)

## Compute: vast.ai (default)

Account setup, 2FA, launch, exec, pull, destroy: follow
`skills/launch-serverless-gpu-job` (Vast section). RF100-VL specifics:

- **Image, tested end to end:**
  `vastai/pytorch:2.11.0-cu128-cuda-12.9-mini-py312-2026-06-15`. Vast's own
  image takes their key-injection path; plain `pytorch/pytorch` sometimes
  leaves sshd rejecting keys. cu128 is required for 5090 (`sm_120`).
  Interpreter: `/venv/main/bin/python`, not bare `python`.
- **Accept or reject in 60 seconds** with harness
  `deploy/vast/accept-box.sh` (GPUs, kernels, matmul, HF, PyPI, disk ≥ 120
  GB free). Destroy duds immediately. **`loading` is essentially unbilled**
  (meter starts at `running`); a wedged 15-minute pull destroyed after
  measured ~$0.02. An older estimate of "~$1 for the host search" overstated
  this by ~10x — the real cost is operator attention. Never nurse a doubtful
  host because destroying it "feels wasteful."
- **Disk allocate ~120 GB without image cache, ~250 GB with
  `cache: "disk"`.** The offer filter "machine has ≥ 300 GB free" and the
  `--disk` you rent are different; disk bills on allocated GB. One campaign
  needs roughly 70 GB (image + pip + 49 GB dataset + weights); disk-cache
  `.npy` sidecars add ~105 GB across 100 datasets.
- Workload is **CPU / host-bound**, priced GPU-centric. Measured
  [pre-cache]: 46 ms GPU vs 507 ms CPU per step; 8 cores/lane still ~94%
  CPU-saturated. `pick_box.py`'s old `MIN_CORES_PER_LANE = 3.0` was far too
  low. Weight core count and single-core clock heavily (a 2.6 GHz EPYC 7K62
  lost to a consumer Ryzen). Size cores for epoch 1 (cache fill + all lanes
  cold), not the steady-state average. Prefer high-clock CPUs even if the
  GPU $/hr is slightly higher.
- Primary target: one 8× RTX 5090 interruptible box with strong CPU; pack
  with `--jobs-per-gpu` after a VRAM/lane re-measure. Fallback: several
  1–4× 5090/4090 boxes.
- Offer filter: verified, reliability > 0.99, **≥ 8 vCPU per lane after
  packing**, machine disk ≥ 300 GB free (so 120–250 allocated fits), ≥ 500
  Mbps down, download < 0.01 USD/GB, host driver CUDA 12.8+, distinct
  egress IP from known-bad NATs. Bid 20–30% above minimum.
- Interruptible: outbid pauses the box; disk persists and bills; destroy
  deletes. Harness resumes at dataset (status files) and epoch (`last.pt`)
  level. Sync weights/results off-box at milestones; always pull before
  destroy.
- Local-first: one dataset, then rf20vl pilot, then rent. When the stack
  changes (new LibreYOLO commit, recipe, packing, image), shake out on ONE
  cheap GPU first (~$1).

### What healthy looks like (do not "fix" this)

- Launch/host-bound: [pre-cache] healthy meant GPUs at **9–35% util and
  ~170 W of 575 W** with everything fine. Low GPU numbers are the signature
  of this workload, not a fault. (Graphed train+val + image cache should
  raise this — re-baseline in the shakedown.) The runbook once claimed
  60–100% util as healthy; that reading burned an expensive detour.
- `nvidia-smi` util is time-with-a-kernel-resident on the CARD, not die
  occupancy per job. With 3 lanes sharing a GPU the row describes the card.
- Datasets are heterogeneous: **92 to 8,791** train images, **0.31 to 12+
  MP**. Epoch times span ~40×. Longest-first scheduling; one dataset sets
  the makespan.
- Epoch 1 costs 1.3–2.1× a steady epoch [pre-cache]; with post-resize cache
  the ratio **widens** (epoch 1 fills the cache). **Every ETA in the first
  hour is garbage** — do not make money decisions off it. A "16.4 h" ETA
  was once an artifact of averaging epoch 1 into a two-epoch mean.

### When something looks slow: profile, do not theorize

`libreyolo profile` answers "where is the time going" in under a minute.
An hour of py-spy (blocked by container caps), `ps` aggregation, and
log-timestamp forensics produced a confidently wrong answer that
`libreyolo profile run` corrected in 52 seconds. The campaign should print
a Profile hint next to the Monitor hint; if it does not, still run:

```bash
libreyolo profile run ...      # one profiled training epoch
libreyolo profile phases ...   # train / validation / save split
libreyolo profile what-if ...  # projected gain from a fix
```

High self-CPU on an op like `aten::max` with near-zero self-CUDA usually
marks a GPU→CPU sync absorbing device wait, not a CPU bottleneck in that
op. Total self-CUDA ≪ total self-CPU ⇒ launch-bound (CUDA graphs).

### Shakedown (before any full campaign)

1. Rent one cheap high-clock-CPU GPU.
2. Install exactly as the campaign would (`--force-reinstall --no-deps`);
   prove recipe fields exist on the installed `TrainConfig`.
3. Run 2–3 representative datasets (tiny ~100 imgs, huge ~8k, large-image
   12 MP class), 10–15 epochs, `eval_interval=1`, plus one full completion.
4. Kill-and-resume between checkpoints (not mid-write).
5. Record VRAM/lane, cores/lane at epoch 1 and steady state, epoch-1 ratio,
   validation share, healthy util band. Size the real box from those numbers.

### Stopping and resuming

- **Ctrl-C mid-checkpoint can corrupt `last.pt`** and poison resume for that
  dataset. Stop gracefully: `tmux send-keys -t bench C-c`, wait for the
  orchestrator to exit, then `vastai stop instance` if keeping the box.
- A **stopped** box bills disk only; restart needs those exact GPUs still
  free. Decide stop-vs-destroy from disk $/day, re-stage cost (~35 min,
  nearly free bandwidth), and whether banked checkpoints are usable
  (recipe change ⇒ they are not).
- Destroy ends spend. Always pull/sync first.

## Decisions to take per campaign

1. Model list and sizes. Flagships deep (yolo9 and rfdetr); other families
   start with their one or two smallest variants.
2. Pilot first: `rf20vl` end to end. Graduate to 100 only if all 20 complete,
   kill-and-resume reproduces within noise, and AP ordering is sane.
3. Budget and waves: price from live offers (`deploy/vast/pick_box.py`),
   publish after flagships land, append families as they finish.
4. Recipe deviations: live in the recipe JSON, hash-recorded, disclosed.
   No silent knobs. Finalize before dataset one.

## Artifacts and publishing

Keep, per model: per-dataset eval JSONs, raw predictions (COCO detections),
per-run `stats.json` (recipe sha, best epoch, seed, dataset version, wall
time), the recipe JSON, the dataset version lock, and the final submission
JSON. Predictions are on by default for publishable runs.

- Upload with `va-bench sync-artifacts` to a LibreYOLO HF dataset repo
  (create the repo by hand; fine-grained write token scoped to that repo).
  Pass `--eval-dir` so predictions are collected. Stock-pycocotools rescore
  of a saved dump reproduced harness AP to four decimals with no harness
  code imported.
- Leaderboard submission: `vision-analysis` via `submit-benchmark-results`
  (see `benchmark-on-visionanalysis`).
- Never hand-edit result JSONs. Regenerate them.

## Traps

- Stock pycocotools `summarize()` breaks with a non-default maxDets list
  (headline AP becomes -1). Never call it with a modified list.
- Package-cleaned data only. Raw Universe exports keep the dummy class and
  score near zero.
- Dataset named `-grccs`: `--datasets=-grccs` (space form is eaten by argparse).
- Never resume after changing physical batch, accumulation, **or recipe
  fields covered by the run signature** (including `cuda_graph` / `cache`).
- Dense datasets can OOM rfdetr; harness falls back to grad-accum and
  restarts that dataset from epoch 0. Expected.
- ec family: AdamW, no mosaic (mosaic triggers a degenerate-box assertion).
- Largest datasets take hours; raise timeouts for slow families, never remove.
- Differences under 0.5 mAP on the 100-dataset mean are noise. Replicate.
- `ROBOFLOW_API_KEY` and vast credentials: env/local config only. Never commit.

## Published numbers to beat (fully supervised, AP 0.50:0.95)

RF-DETR N/S/M/L/XL/2XL: 57.7 / 60.2 / 61.2 / 62.2 / 62.9 / 63.2. LW-DETR
T to X: 57.1 to 62.1. D-FINE N to X: 58.2 to 62.2. YOLO11 N to X: 55.3 to
56.5. YOLO26 N to X: 52.0 to 60.0. Sources: the rf-detr repo README
(develop) and paper v4 tables. No YOLO9-lineage numbers exist anywhere yet.
