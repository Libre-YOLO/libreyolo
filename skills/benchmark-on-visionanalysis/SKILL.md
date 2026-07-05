---
name: benchmark-on-visionanalysis
description: >-
  Signpost for benchmarking LibreYOLO models for visionanalysis.org. Use when
  someone wants to "benchmark for visionanalysis", produce a submission for the
  site, measure a model on COCO for publication, or add a hardware/runtime row
  to the site. The actual work lives in two OTHER repos; this skill orients you
  and hands off. It does not run benchmarks itself.
---

# Benchmark a LibreYOLO model for visionanalysis.org

This is a **pointer skill**. The benchmark and publish logic live in two separate
repos, each with its own authoritative skill. Do not run benchmarks from inside
`libreyolo`; switch to the right repo and follow its skill.

## The two repos

| Repo | Role | Authoritative skill (open on GitHub) |
|---|---|---|
| `vision-analysis-benchmark` (the harness) | Runs models, emits `va.submission.v1` JSON | [`generate-benchmark-results`](https://github.com/LibreYOLO/vision-analysis-benchmark/blob/main/skills/generate-benchmark-results/SKILL.md) |
| `vision-analysis` (the website) | Validates JSON, rebuilds the dataset, deploys | [`submit-benchmark-results`](https://github.com/LibreYOLO/vision-analysis/blob/main/skills/submit-benchmark-results/SKILL.md) |

Direct links to the authoritative skills (read these — this signpost only orients):
- Harness: https://github.com/LibreYOLO/vision-analysis-benchmark/blob/main/skills/generate-benchmark-results/SKILL.md
- Website: https://github.com/LibreYOLO/vision-analysis/blob/main/skills/submit-benchmark-results/SKILL.md

Local checkouts on this machine:
`C:\Users\Usuario\Documents\GitHub\vision-analysis-benchmark` and
`C:\Users\Usuario\Documents\GitHub\vision-analysis`.
GitHub: `LibreYOLO/vision-analysis-benchmark`, `LibreYOLO/vision-analysis`.

## What to know before you start

The dataset, layout gotchas, protocol config, supported runtimes, and the
`libreyolo_commit` rule all live in the harness skill's **"Dataset & protocol"**
section (`generate-benchmark-results`). Read that there; this signpost does not
copy it (so it cannot drift). One thing worth knowing up front: the canonical
eval set is the HF dataset `LibreYOLO/coco-val2017-mini500`, not full COCO.

## Flow

1. Go to `vision-analysis-benchmark`, follow `generate-benchmark-results`. This emits
   one `va.submission.v1` JSON per (model x runtime x hardware) run.
2. Hand the emitted JSON(s) to `vision-analysis`, follow `submit-benchmark-results`
   to validate, rebuild `generated/verified-results.v1.json`, and open the PR / deploy.

The two skills above are authoritative. If anything here disagrees with them, they
win - update this signpost rather than diverging.
