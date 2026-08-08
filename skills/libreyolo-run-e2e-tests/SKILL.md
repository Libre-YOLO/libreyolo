---
name: libreyolo-run-e2e-tests
description: >-
  Launch LibreYOLO's end-to-end (e2e) test suite the right way — the heavy GPU
  tests under tests/e2e/ that load real weights, export to ONNX/TensorRT/OpenVINO/
  ncnn/TorchScript/CoreML, train on real datasets, and check inference parity.
  Use whenever someone wants to run e2e tests, the nightly suite, a single e2e
  file, a model family's e2e coverage, an export-backend check, or an RF1/RF5
  training test — or is confused about why e2e tests "don't run" or all skip.
  Covers the Makefile targets, the direct-pytest fallback (Windows / no make+uv),
  the marker taxonomy, weight/dataset provisioning, and how to read the results.
  This is for *running* the e2e suite, not for writing new e2e tests.
---

# Run LibreYOLO e2e tests

The e2e suite lives in `tests/e2e/`. Unlike `tests/unit/` (fast, hermetic, no
weights), e2e tests **load real model weights, run real inference/export/training
on a GPU, and hit the network** to fetch weights and datasets. They are slow and
resource-heavy. Treat them as an integration gate, not a quick check.

## The two rules that trip everyone up

1. **The default marker hides every e2e test.** `pyproject.toml` sets
   `addopts = "-m unit"`. Run pytest against an e2e file with no `-m` and you get
   `no tests collected (N deselected)` — it looks broken but it's just the default
   filter. You **must** pass `-m` on the command line (it overrides the ini
   default; last `-m` wins). The canonical expression is `"e2e and not rf5"`.

2. **One file per process.** After a few CUDA-heavy export/training tests the CUDA
   driver state in the pytest process gets corrupted and later `fork()`/`spawn()`
   segfaults (exit 139). The suite is therefore run **one file at a time, each in
   a fresh pytest process.** The `Makefile` does this for you; if you invoke
   pytest directly, loop over files yourself — never pass the whole `tests/e2e/`
   directory in a single pytest run.

## Preferred path — Makefile targets (Linux/macOS/CI, needs `make` + `uv`)

```bash
make test_e2e                                  # all e2e files, each own process, "e2e and not rf5"
make test_e2e FROM=test_rf1_training.py        # resume from a file (skip earlier ones)
make test_e2e MARKERS='e2e and not extended_backend'   # custom marker expr
make test_e2e MARKER='onnx'                    # MARKER= is an alias for MARKERS=

make test_nightly            # general + flagship nightly (what CI runs)
make test_general_nightly    # smallest-inference case for every public family
make test_flagship_nightly   # heavier YOLO9/RF-DETR: val, video, tracking, CLI, RF1
make test_rf5                # RF5 training benchmark (needs ROBOFLOW_API_KEY)
make print_nightly_suite     # print the versioned nightly contract
```

`make test_e2e` stops at the first failing file and prints
`FAILED: <name> (exit N)`; use `FROM=<that file>` to resume after a fix. Exit
code `5` from a file means "no tests selected" and is counted as **skipped**, not
a failure. The nightly targets set `LIBREYOLO_FAIL_ON_NIGHTLY_SKIP=1`, which
turns a *skipped* nightly-marked test into a **failure** — the gated nightly must
actually execute, never silently skip (e.g. because a weight didn't download).

## Fallback path — direct pytest (Windows, or no make/uv)

Use the repo venv python and set `PYTHONPATH` so tests import local sources.
Preserve the one-file-per-process rule with a loop:

```bash
# from the repo root
PY=.venv/Scripts/python.exe      # Windows; use .venv/bin/python on Linux/macOS
for f in $(find tests/e2e -name 'test_*.py' | sort); do
  echo "== $f =="
  PYTHONPATH=. "$PY" -m pytest "$f" -m "e2e and not rf5" -v -p no:cacheprovider
  rc=$?
  if [ $rc -eq 5 ]; then echo "(no tests selected — skipped)";
  elif [ $rc -ne 0 ]; then echo "FAILED: $f (exit $rc)"; break; fi
done
```

Run a **single** file (fast when you only care about one area):

```bash
PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/e2e/test_onnx.py \
  -m "e2e and not rf5" -v -p no:cacheprovider
```

Narrow to one test with `-k`, e.g. `-k "yolo9 and t"`.

## The groups (markers) — pick what you actually need

Every file carries `pytest.mark.e2e` plus more specific markers. Compose with
`and`/`or`/`not`. The main axes:

| Group | Marker(s) | Files |
|---|---|---|
| **Export backends** | `export_backend` + `supported_backend`/`extended_backend` + backend | `test_onnx` (onnx, *supported*), `test_torchscript`, `test_openvino`, `test_ncnn`, `test_tensorrt` (trt), `test_coreml_roundtrip` (macOS), `test_yolonas_pose_export` |
| **Inference / parity** | `general_nightly`, plus family/task | `test_deterministic_inference`, `test_openvocab_inference`, `test_rfdetr_keypoint_parity`, `test_weight_requirements` |
| **Training** | `rf1`, `rf5`, `slow`, `flagship_nightly` | `test_rf1_training` (rf1), `test_rf5_training` (rf5), `test_training_regression`, `test_rfdetr_lora`, `test_rfdetr_seg_training` |
| **Task / family tiers** | `fomo`, `l2cs`, `sam`, `vlm`, `openvocab`, `yolonas` | `test_fomo`, `test_l2cs_gaze`, `test_sam_smoke`, `test_sam2_smoke`, `test_mobilesam_smoke`, `test_lfm2_vlm_smoke`, `test_yolonas`, `test_sam3dbody_contract` |
| **Pipelines** | `flagship_nightly`, family | `test_video`, `test_tracking` (CUDA-only), `test_val_coco128`, `test_doctor_coco8`, `cli/test_cli` |

Common selections:

```bash
# Only the ONNX backend, across all families
make test_e2e MARKERS='onnx'
# All export backends except extended-runtime coverage (ONNX only, the release gate)
make test_e2e MARKERS='export_backend and not extended_backend'
# One model family end to end
make test_e2e MARKERS='yolo9'
# Everything except training and the slowest export backends
make test_e2e MARKERS='e2e and not rf1 and not rf5 and not tensorrt and not ncnn'
```

Full marker list is in `pyproject.toml` under `[tool.pytest.ini_options] markers`.
Family markers: `yolox yolo9 yolo9_e2e yolonas rfdetr dfine deim deimv2 ec rtdetr
rtdetrv2 rtdetrv4 picodet rtmdet l2cs fomo`. Tier markers: `vlm sam openvocab clip`.

## Weights, datasets & why tests skip

Skips are normal and are **not** failures — decide whether each one is expected:

- **Model weights** come from the catalog in `tests/e2e/conftest.py` (`MODEL_CATALOG`).
  `require_test_weights()` skips a case only when the weight is both missing
  locally **and** has no public auto-download route (LibreYOLO HF mirror, or
  Deci's CDN for YOLO-NAS). Most weights auto-download to the HF cache on first
  run — the first e2e run is slow and needs network.
- **Bare filenames** (`LibreYOLO9t.pt`) resolve via `weights/` then auto-download;
  path-prefixed entries (`weights/…`, `downloads/yolonas/…`) must exist locally or
  have a route.
- **Datasets**: `test_rf1_training` / `test_training_regression` pull the *marbles*
  dataset from HF (`LibreYOLO/marbles`, cached under `~/.cache/libreyolo/marbles`).
  `test_rf5_training` needs `ROBOFLOW_API_KEY` for RF100 and is excluded from the
  default (`not rf5`). `test_tracking`/`test_doctor_coco8`/`test_sam_smoke`/
  `test_lfm2_vlm_smoke`/`test_openvocab_inference` fetch videos/datasets/weights
  over the network (`network` + `external_data` markers).
- **Non-redistributable weights** (L2CS/Gaze360) have no plain-HTTP route, so
  `test_l2cs_gaze` skips in gated nightly by design — it runs only in the
  non-gated per-family suite.

## Hardware / dependency gating

`tests/e2e/conftest.py` provides skip guards — a case skips cleanly when the
capability is absent:

- `requires_cuda` — no GPU (this machine has one; check with
  `python -c "import torch; print(torch.cuda.is_available())"`).
- `requires_tensorrt` — TensorRT not installed (also needs CUDA).
- `requires_openvino` / `requires_ncnn` — `pip install libreyolo[…]` extras.
- `requires_rfdetr` — native RF-DETR extra.
- CoreML tests skip off macOS.

If a whole backend's file reports all-skipped, the extra almost certainly isn't
installed — that's environment, not a code failure.

## Reading the outcome

- Per file, pytest exit `0` = pass, `5` = nothing selected (skip — verify it was
  *meant* to skip), anything else = failure.
- `make test_e2e` aggregates: `all done: P passed, S skipped, F failed`, and stops
  early on the first hard failure.
- For a nightly-parity view, the header line printed under a nightly marker is the
  versioned contract from `tests/e2e/nightly_contract.py` — quote its version when
  reporting a nightly result.

## How CI runs it (for reference)

`.github/workflows/e2e-nightly-dev.yml` (and `-release`/`-pypi`) run on a schedule,
resolve the current `dev` SHA, skip if that SHA was already tested, then execute
on a **Modal GPU** container via
`uvx modal run tools/ci/modal_nightly.py --ref <sha> --target test_nightly`, which
just runs `make test_nightly` remotely and parses a `MODAL_NIGHTLY_RESULT` JSON
line for pass/fail + runtime + estimated GPU cost. These workflows deliberately
have **no `pull_request` trigger** (they hold Modal secrets); e2e never runs on
fork PRs. The PR gate is unit-only (`make test_pr_gate`).
