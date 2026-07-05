---
name: libreyolo-run-unit-tests
description: >-
  Run and write LibreYOLO's unit tests and the PR gate: the fast, hermetic,
  CPU-only suite under tests/unit/ that gates every push and PR to dev. Use
  whenever someone wants to run unit tests, "run the tests before pushing",
  check whether a change breaks the PR gate, run one test file or one area
  (CLI, a model family, augmentations), or add a new unit test that must stay
  inside the PR-gate contract. Covers the marker taxonomy (unit /
  external_data / network), the hermeticity rules and the HTTP blocker, the
  Windows no-make fallback, golden fixtures, and how to keep a new test
  PR-gate-safe. For the heavy GPU suite use libreyolo-run-e2e-tests instead.
---

# Run LibreYOLO unit tests (the PR gate)

`tests/unit/` is the merge-blocking suite: fast, CPU-only, no network, no real
weights. It runs on every push and PR to `dev` on Linux, macOS, and Windows
(`.github/workflows/unit-tests.yml`, Python 3.10). If this suite is green and
install smoke is green, the PR gate is green; e2e never runs on PRs.

## The canonical commands

```bash
make test_pr_gate          # exactly what CI runs

# Direct equivalent (what the cross-platform workflow executes):
LIBREYOLO_PR_GATE=1 uv run --no-sync pytest tests/unit -m "unit and not external_data and not network"
```

On this Windows box `make` and `uv` are not on the PowerShell PATH; use the
Bash tool with the repo venv:

```bash
# from the repo root (worktrees reuse the main checkout's .venv)
LIBREYOLO_PR_GATE=1 PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/unit \
  -m "unit and not external_data and not network" -q
```

Scoped runs while iterating (drop `LIBREYOLO_PR_GATE` for speed if you are not
checking hermeticity):

```bash
PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/unit/cli -q                 # one area
PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/unit/test_tasks.py -q      # one file
PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/unit -k "yolo9 and load" -q
```

Before pushing any change that touches `libreyolo/`, run at least the unit
files for the touched area, then the full gate if the change is shared code
(`models/base`, `training/`, `validation/`, `cli/`, `data/`).

## How the markers work (the traps)

- `pyproject.toml` sets `addopts = "-m unit"`. Every pytest invocation
  defaults to unit-marked tests only. This is why running an e2e file "does
  nothing" without `-m`, and why a unit test missing its marker silently
  never runs. The last `-m` on the command line wins.
- Every unit file declares `pytestmark = pytest.mark.unit` at module top.
  A new test file without it is invisible to CI. Check this first when a new
  test "passes locally but CI never ran it".
- `external_data` marks tests that need staged local weights or datasets
  (e.g. the `*_parity.py` tests that compare against real upstream
  checkpoints). They are excluded from the PR gate and run only where the
  files are staged. Marking is per-test or per-module; the test should also
  skip cleanly when its file is absent.
- `network` marks tests that intentionally reach non-local hosts. Also
  excluded from the gate.

## The hermeticity contract (PR Gate v1.0, docs/testing.md)

With `LIBREYOLO_PR_GATE=1`, an autouse fixture in `tests/conftest.py` patches
HTTP entry points and **fails** any test that touches a non-localhost URL
(localhost stays allowed so DDP-on-localhost unit tests work). The contract:

- No downloads: no HF, no GitHub releases, no CDN, no datasets, no weights.
- No GPU, no CUDA, no vendor export runtimes required.
- Needs external bytes anyway? Use a local fixture or mock; if the real
  artifact is essential, mark `external_data`/`network` and accept it leaves
  the gate, or move the coverage to e2e/nightly.

New unit tests are in the PR gate **by default**. That is the point: prefer
writing the test so it stays in.

## Golden fixtures and parity tests

Two patterns to know before touching numeric code:

- **Golden fixtures** pin exact numeric behavior without external data:
  `tests/unit/fixtures/augment_golden/` (augmentation parity),
  `tests/unit/data/ocsort_parity_golden.json` (tracker parity). If a
  deliberate behavior change breaks a golden test, regenerate the fixture
  with the generator script referenced in the test file header and say so in
  the PR; never hand-edit golden values.
- **Parity-vs-upstream tests** (`test_*_parity.py`) are `external_data`:
  they load a real converted checkpoint and assert exact or near-exact output
  agreement. They do not run in the gate; run them manually when touching a
  ported family whose weights are staged under `weights/`.

## Reading failures

- `N deselected, 0 selected` on a file you expected to run: marker problem
  (missing `pytestmark` or your `-m` excluded it), not a collection bug.
- A gate failure that mentions "External HTTP is blocked in the LibreYOLO PR
  gate": the code under test tries to download; fix the test to use a
  fixture, do not mark it `network` just to make CI pass.
- Windows-only failures are real: CI runs Windows, so path handling
  (`Path` vs string, case, separators) and `spawn` multiprocessing must work.
- The suite must pass on Python 3.10 (CI's floor) even if the local venv is
  newer; avoid 3.11+ syntax in tests and library code.

## Writing a new unit test: checklist

1. `pytestmark = pytest.mark.unit` at module top.
2. No network, no real weights: build tiny models from config, use
   `tmp_path`, synthesize images with numpy.
3. If it genuinely needs a staged artifact: `@pytest.mark.external_data`
   plus a clean skip when the artifact is missing.
4. Fast: the whole gate is thousands of tests; keep each under ~1s.
5. Run it under `LIBREYOLO_PR_GATE=1` once before pushing so the HTTP
   blocker vets it the way CI will.

## Related

- `docs/testing.md`: the full test-tier contract (unit / smoke / e2e / QA).
- `skills/libreyolo-run-e2e-tests/`: the GPU suite this skill is not.
- `skills/merge-to-dev/`: run the touched-area unit tests before pushing.
