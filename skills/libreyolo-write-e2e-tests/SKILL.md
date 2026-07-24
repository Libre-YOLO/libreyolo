---
name: libreyolo-write-e2e-tests
description: >-
  Add or extend LibreYOLO e2e test coverage the right way: a new model family's
  GPU coverage, a new export backend file, a nightly case, or a training e2e.
  Use when someone asks to "add e2e tests for X", wire a new family into the
  nightly, cover a new export format end to end, or when a PR review flags
  missing e2e coverage. Covers the marker taxonomy to declare, the
  MODEL_CATALOG and GENERAL_NIGHTLY_INFERENCE_MODELS rows, the versioned
  nightly contract (when to bump it), weight provisioning via
  require_test_weights, and the one-file-per-process constraint that shapes
  how files are laid out. For *running* the suite use libreyolo-run-e2e-tests.
---

# Write LibreYOLO e2e tests

e2e tests live in `tests/e2e/`, load real weights, and run on a GPU. They are
executed one file per pytest process (CUDA state corrupts across heavy files),
selected purely by markers, and partially promoted into a versioned nightly.
Writing one is mostly bookkeeping; this skill is the bookkeeping.

## Where a new test goes

- **New model family**: inference/val cases usually extend existing
  parametrized files via the catalog rows (below) rather than a new file. A
  dedicated `test_<family>*.py` file is for family-specific behavior
  (e.g. `test_yolonas.py`, `test_l2cs_gaze.py`, `test_openvocab_inference.py`).
- **New export backend**: its own `test_<backend>.py`, modeled on
  `test_onnx.py` (supported) or `test_ncnn.py` (experimental).
- **Training behavior**: `test_rf1_training.py` (per-family train+reload on
  the marbles dataset) or `test_training_regression.py`.
- **Smoke for a new optional tier**: pattern of `test_sam_smoke.py`,
  `test_sam2_smoke.py`, `test_mobilesam_smoke.py`, `test_lfm2_vlm_smoke.py`.

Because the runner executes one **file** per process, a file is the isolation
unit: don't mix a heavy training case into an inference file, and don't create
a file whose tests depend on state from another file.

## Markers: declare all the axes

Every e2e test must carry `pytest.mark.e2e` plus the axes that apply, or the
marker-driven selections (and the nightly) will never pick it up, and
`addopts = "-m unit"` will hide it from plain pytest runs:

- family marker (`yolo9`, `rfdetr`, `picodet`, ...) - if the family is new,
  **add the marker to `[tool.pytest.ini_options] markers` in `pyproject.toml`**;
  unknown markers are a lint/CI failure.
- tier marker for non-factory tiers: `vlm`, `sam`, `openvocab`, `clip`.
- backend markers for export tests: `export_backend` + `supported_backend`
  or `experimental_backend` + the backend name (`onnx`, `tensorrt`, ...).
- `network` / `external_data` when it downloads or needs staged files.
- nightly promotion markers: `general_nightly` / `flagship_nightly`
  (see below; do not sprinkle these casually).
- `rf1` / `rf5` / `slow` for training-cost tests.

## The catalog rows

`tests/e2e/conftest.py` holds two lists; know which one you are editing:

- `MODEL_CATALOG`: `(family, size, weight_ref)` rows that feed validation and
  training parametrization. `weight_ref` is either a bare canonical filename
  (`LibreYOLO9t.pt`, resolves via `weights/` then auto-download) or a local
  path (`downloads/yolonas/...`, must exist or have a route).
- `GENERAL_NIGHTLY_INFERENCE_MODELS`: the one-smallest-case-per-family list
  the general nightly runs. A new public detector family with an
  auto-download route **must** add its smallest size here, or the nightly
  silently stops covering the library's newest family.

Weight gating goes through `require_test_weights()` in the conftest: it skips
only when the weight is missing locally **and** has no public auto-download
route. Never hand-roll `os.path.exists` skips; they hide provisioning bugs.

## The nightly contract is versioned

`tests/e2e/nightly_contract.py` pins `NIGHTLY_E2E_SUITE_VERSION` (currently
2.1) and a one-line contract of what green means. If your change adds or
removes nightly coverage, changes thresholds, or changes runtime materially:

- bump **minor** for added coverage / threshold / runtime changes,
- bump **major** when a green run makes a materially different promise,
- update the contract string so it stays true (it is printed in every run
  header and quoted in reports),
- keep `docs/testing.md` in sync (it states the current counts and contract).

A nightly-marked test that **skips** is a **failure** under
`LIBREYOLO_FAIL_ON_NIGHTLY_SKIP=1` (how CI runs it). So only promote a case to
`general_nightly`/`flagship_nightly` if its weight has a public route and it
cannot skip for environmental reasons. Non-redistributable weights (the
L2CS/Gaze360 precedent) get a **non-gated per-family file** that skips cleanly
when the weight is not staged, and stay out of the gated nightly.

## Skeleton for a family inference case

```python
import pytest
from tests.e2e.conftest import require_test_weights

pytestmark = [pytest.mark.e2e, pytest.mark.<family>]

@pytest.mark.general_nightly            # only if promoted; see contract rules
def test_<family>_smallest_inference(sample_image):
    weight = require_test_weights("Libre<Family><smallest>.pt")
    from libreyolo import LibreYOLO
    model = LibreYOLO(weight)
    results = model(sample_image, conf=0.25)
    r = results[0] if isinstance(results, list) else results
    assert r.boxes is not None and len(r) > 0
```

Use the conftest fixtures (`sample_image`, `cuda_device`, `temp_export_dir`)
and the `run_in_subprocess` helper function instead of reinventing them;
`run_in_subprocess` exists precisely because some flows must not share the
pytest process's CUDA state.

## Prove it runs before shipping

The default marker filter means an unmarked file silently collects zero tests.
Before pushing:

```bash
# collection proves markers are right (no GPU needed)
PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/e2e/test_<new>.py \
  --collect-only -q -m "e2e and not rf5"
# then actually run the file once on the local GPU
PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/e2e/test_<new>.py \
  -m "e2e and not rf5" -v -p no:cacheprovider
```

If promoting to nightly, also collect with `-m general_nightly` (or
`flagship_nightly`) and confirm the new case appears, then run once with
`LIBREYOLO_FAIL_ON_NIGHTLY_SKIP=1` to prove it executes rather than skips.

## Related

- `skills/libreyolo-run-e2e-tests/`: running the suite (markers, resume, CI).
- `docs/testing.md`: the tier contract and nightly counts to keep in sync.
- `skills/libreyolo-port-model/`: commit 9 of a port is the e2e catalog row.
