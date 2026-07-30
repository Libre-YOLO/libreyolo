---
name: run-rf100vl-benchmark
description: >-
  Run the RF100-VL (Roboflow 100-VL) detection benchmark on LibreYOLO models
  end to end: dataset download with version locking, per-dataset fine-tuning,
  protocol-conformant pycocotools evaluation at maxDets 500, artifact
  publishing, and multi-GPU execution on vast.ai (the default compute). Use
  when someone wants to "run RF100-VL", "benchmark on Roboflow 100",
  reproduce or extend published RF100-VL numbers, or add a model family to
  the campaign. Training and eval run in the vision-analysis-benchmark
  harness; this skill holds the protocol, the locked decisions, and the
  compute playbook.
---

# Run the RF100-VL benchmark

RF100-VL: 100 real-world detection datasets from Roboflow Universe (164,149
images, 564 classes, 7 domains), paper arXiv 2505.20612 (NeurIPS 2025
Datasets and Benchmarks). Each dataset ships fixed `train`/`valid`/`test`
splits in COCO JSON. There is no official runner; the harness referenced
below is the runner.

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

Never report toolkit-native trapezoidal mAP; it inflates up to 2.7 AP on
RF100-VL versus pycocotools (paper, App B). LibreYOLO validation is
pycocotools-based already; the 500 cap is the opt-in `eval_max_det` kwarg
(`model.val(data=..., split="test", eval_max_det=500)`). Defaults for normal
users are unchanged (AP at maxDets 100) and test-locked.

## Where the work happens

Repo `LibreYOLO/vision-analysis-benchmark` (the harness). Two verbs:

```bash
# per-dataset fine-tuning: one worker per GPU, subprocess children,
# atomic status files, resume, timeouts, dense-dataset OOM fallback
va-bench rf100vl-train --data-dir ./rf100-vl --weights-root ./rf100vl-weights --gpus 0,1,2,3,4,5,6,7

# per-dataset test-split eval: cached and resumable, emits one
# va.submission.v1 JSON with a per-dataset rf100vl block
va-bench rf100vl --all --data-dir ./rf100-vl --weights-root ./rf100vl-weights
```

- **Dataset, fast path (use this on rented boxes).** Pull the pre-built copy at
  [`LibreYOLO/rf100-vl`](https://huggingface.co/datasets/LibreYOLO/rf100-vl):
  100 per-dataset tars, already cleaned, plus `versions.json`, `NOTICE`, and
  `licenses.json`. No `ROBOFLOW_API_KEY`, and version-pinned so two boxes
  cannot silently score different data.

  ```bash
  pip install "huggingface_hub[hf_transfer]"
  export HF_HUB_ENABLE_HF_TRANSFER=1        # THIS is the speed lever
  python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('LibreYOLO/rf100-vl', repo_type='dataset', \
                      local_dir='rf100-vl', max_workers=8)"
  cd rf100-vl && for f in *.tar; do tar xf "$f" && rm "$f"; done
  ```

  Be precise about what makes it fast: `HF_HUB_ENABLE_HF_TRANSFER=1` (the Rust
  downloader) plus few large files instead of 164k small ones. An HF token is
  NOT required for a public dataset and does not raise throughput, but staying
  logged in (`huggingface-cli login`, or `HF_TOKEN` in the env) gets the higher
  authenticated rate limit, which is worth having when several boxes pull at
  once. Measured: Roboflow serves ~2.2 MB/s from a home line and ~16 MB/s from
  a datacenter, so the canonical path costs 45 min to 5.6 h; the tars are minutes.

- **Dataset, canonical path.** `--download` wraps the `rf100vl` pip package and
  needs a free `ROBOFLOW_API_KEY`; about 40 GB. Use it to REBUILD the HF copy,
  to verify it, or if the HF copy is unavailable. The harness writes a version
  lock and replays recorded versions; it never re-resolves latest.
  We host the HF copy; it is not a hedge or a stopgap. Licensing is recorded in
  `va_bench/data/rf100vl_licenses.json`, which deliberately keeps BOTH upstream
  statements rather than resolving them silently: the benchmark repository
  claims Apache-2.0 as a blanket licence over the datasets, while all 100
  Universe projects individually report MIT (checked per project and
  cross-validated against the `License:` line inside downloaded exports). Both
  permit redistribution and derivatives with notice retention, so the
  disagreement changes nothing operationally; we rely on the per-project field
  and preserve every dataset's own README.dataset.txt inside its archive.
- Weights contract: training places `best.pt` at
  `<weights_root>/<dataset>/<weight_file>`; eval resolves exactly that path.
- A capability guard aborts training on any libreyolo build without
  `eval_max_det`/`amp_dtype` support, so a wrong install cannot silently
  produce off-protocol numbers.
- Exact flags and current behavior: the harness README (RF100-VL section)
  and `--help` are authoritative; do not trust this skill for flag lists.

## Decisions to take per campaign

1. Model list and sizes. Flagships deep (yolo9 and rfdetr, all or most
   sizes); every other family starts with its one or two smallest variants.
2. Pilot first: run the `rf20vl` subset end to end. A family graduates to
   the full 100 only if all 20 datasets complete, a kill-and-resume drill
   reproduces the uninterrupted result within noise, and its AP ordering is
   sane.
3. Budget and waves: price the run from current marketplace offers (below),
   publish after the flagships land, append families as they finish.
4. Recipe deviations: any change lives in the recipe JSON, is hash-recorded
   in every run, and is disclosed with the results. No silent knobs.

## Artifacts and publishing

Keep, per model: per-dataset eval JSONs, per-dataset raw predictions (COCO
detections), per-run `stats.json` (recipe sha, best epoch, seed, dataset
version, wall time), the recipe JSON, the dataset version lock, and the
final submission JSON.

The eval verb writes predictions itself (`<fingerprint>.predictions.json.gz`
beside each per-dataset result, path recorded in the submission). It is on by
default; do not turn it off for a run you intend to publish, because it is the
only thing that makes the claim below true.

- Upload the per-model artifact folder to a Hugging Face dataset repo under
  the LibreYOLO org (one folder per model). Anyone can then rescore from the
  JSONs with pycocotools, no GPU needed; that is the reproducibility story.
  Verified end to end: a stock-pycocotools rescore of a saved dump, importing
  no harness code, reproduced the harness AP to four decimals.
- The leaderboard submission goes to the `vision-analysis` repo through its
  `submit-benchmark-results` flow (validate, rebuild, PR, deploy). See the
  `benchmark-on-visionanalysis` skill for the handoff.
- Never hand-edit result JSONs. Regenerate them.

## Compute: vast.ai (default)

Account setup, 2FA, launch, exec, tail, pull, guard, and destroy discipline:
follow `skills/launch-serverless-gpu-job` (Vast section). This section adds
only the RF100-VL specifics.

- Workload shape: independent single-GPU jobs, batch 16 at 640 px, under 12
  GB VRAM for most families. No interconnect requirement, so interruptible
  consumer boxes dominate datacenter cards on price per GPU-hour.
- Primary target: one 8x RTX 5090 interruptible box. Fallback: several 1-4x
  RTX 5090 or RTX 4090 boxes; the orchestrator takes any GPU count and
  multi-box runs split the dataset list.
- Offer filter: verified host, reliability above 0.99, at least 8 vCPU per
  GPU, at least 300 GB disk, at least 500 Mbps down, download cost under
  0.01 USD per GB, host driver CUDA capability 12.8 or newer. Bid 20 to 30
  percent above the current minimum to reduce outbid churn.
- Prices move within hours: re-check offers immediately before every wave
  and archive the query plus results with the run records.
- Interruptible semantics: being outbid pauses the box; its disk persists
  (and bills) while the instance exists; destroy deletes it. The harness
  resumes at dataset level (status files) and epoch level (`last.pt`). Sync
  `weights_root` and results off-box at milestones (HF), and always `pull`
  before `destroy`.
- Local-first rule: the whole flow must pass on a local GPU (one dataset,
  then the rf20vl pilot) before renting anything. Credits are for the
  campaign, not for debugging.
- Rough sizing: 0.5 to 1 consumer-GPU-hour per dataset for YOLO-family
  models, 1 to 2.5 for DETR-family; a full 100-dataset pass per family lands
  roughly between 30 and 200 GPU-hours depending on family.

## Traps

- Stock pycocotools `summarize()` breaks with a non-default maxDets list
  (headline AP becomes -1). LibreYOLO and the harness compute the stats
  directly; never call stock summarize with a modified list.
- Use package-cleaned data only, from either source above. The `rf100vl` package
  rewrites category numbering on download (dummy class 0 removed, ids shifted to
  0-based contiguous, annotation ids from 1), and the HF copy was produced by
  running exactly that code, so both are equivalent and version-pinned. **Never
  a raw export from the Universe website**: those keep the dummy class and the
  original numbering, and scoring them against the benchmark ground truth gives
  close to zero mAP.
- One dataset is literally named `-grccs`: write `--datasets=-grccs` (the
  space form is eaten by argparse).
- Never resume a checkpoint after changing physical batch or accumulation.
  The harness refuses via run signatures; do not override it.
- Dense datasets can OOM rfdetr; the harness picks a grad-accum fallback at
  dataset start and restarts from epoch 0 on a mid-run OOM. Expected.
- The ec family trains with its AdamW no-mosaic recipe (mosaic triggers a
  degenerate-box assertion).
- The largest datasets take hours per run. Per-dataset timeouts plus the
  rerun list handle the tail; raise the timeout for slow families, never
  remove it.
- Differences under 0.5 mAP on the 100-dataset mean are noise. Replicate
  before claiming a win.
- `ROBOFLOW_API_KEY` and vast credentials live in env or local config only.
  Never commit keys; upstream reference repos did, and it cost them.

## Published numbers to beat (fully supervised, AP 0.50:0.95)

RF-DETR N/S/M/L/XL/2XL: 57.7 / 60.2 / 61.2 / 62.2 / 62.9 / 63.2. LW-DETR
T to X: 57.1 to 62.1. D-FINE N to X: 58.2 to 62.2. YOLO11 N to X: 55.3 to
56.5. YOLO26 N to X: 52.0 to 60.0. Sources: the rf-detr repo README
(develop) and paper v4 tables. No YOLO9-lineage numbers exist anywhere yet.
