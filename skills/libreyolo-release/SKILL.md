---
name: libreyolo-release
description: >-
  Cut a LibreYOLO version release end to end: evidence-backed changelog built
  by fan-out agents over the release..dev diff, a gate scoreboard (unit CI,
  GPU e2e on Modal reusing the nightly, weight autodownload probe, wheel
  smoke, notices sync, docs drift, breaking-change sweep), the branch/tag/PyPI
  mechanics, post-publish verification, and the handoff to the marketing
  repo's announcement pipeline. Use whenever the user says "release",
  "cut vX.Y.Z", "ship a version", "prepare the changelog for the release",
  or "is dev ready to release?".
---

# LibreYOLO release

Turn "let's release" into a published PyPI version with a real changelog,
proven gates, and marketing assets queued, without re-deriving the process
each time. The engineering source of truth is `RELEASING.md`; this skill
wraps it with everything that document assumes you already know.

**Golden rule: nothing goes in the changelog without evidence (a commit SHA
or PR number), and nothing ships without a green scoreboard.** The changelog
is built before the gates run so that gate failures can be traced back to the
change that caused them.

## The lay of the land (read once, saves an hour)

- Branches are `dev` and `release`. There is **no `main`**.
- `pyproject.toml` `version` is the source of truth. `dev` carries
  `X.Y.0.dev0`; `release` carries the clean `X.Y.0`. The bump commit lives on
  the release side.
- CI gotcha: `unit-tests.yml` and `install-smoke.yml` trigger only on
  push/PR to **`dev`**. A `dev -> release` PR shows **no checks**; the real
  gate is CI on the dev push. Do not read the empty checks list as green.
- `publish.yml` fires on `v*` tags, rejects tags not reachable from
  `release`, and ends in a manual GitHub-Environment approval ("Publish to
  PyPI"). Trusted Publishing / OIDC; there is no PyPI token anywhere.
- GPU e2e lives on Modal. `tools/ci/modal_nightly.py` takes `--ref <sha>`,
  so the same harness that runs the nightly can test the exact release
  candidate commit.
- The announcement side (GitHub release body polish, LinkedIn carousel,
  Reddit GIF post) is the **marketing repo's** `new-version-release` skill
  (`../marketing/skills/new-version-release/`). This skill produces the
  facts file and changelog it consumes; do not rebuild its pipelines here.

## Phase 0: Preflight and scoreboard

Create a working directory **outside the repo tree** (scratchpad or
`~/release-<version>/`) and start `SCOREBOARD.md` in it. Every phase below
appends one line: `| gate | PASS/FAIL/SKIPPED | evidence link or note |`.
The scoreboard is the deliverable you show the user before asking for the
go/no-go; a release with SKIPPED rows needs their explicit sign-off per row.

Preflight checks (all read-only, run in parallel):

```bash
git fetch upstream --tags
# 1. What is shipping and from where
git log -1 upstream/dev --oneline
grep '^version' pyproject.toml                       # on dev: expect X.Y.0.dev0
# 2. Last released version = base ref for everything
git tag --sort=-creatordate | head -5
# 3. Hotfixes stranded on release that dev never got (MUST be merged back first)
git log upstream/dev..upstream/release --oneline
# 4. CI on dev HEAD is actually green
gh run list -R LibreYOLO/libreyolo --branch dev --limit 10
# 5. Anything already labeled/queued as release-blocking
gh pr list -R LibreYOLO/libreyolo --base dev --state open --limit 30
```

Confirm with the user: version label (`vX.Y.Z`), base tag, and which gates
to run (default: all except RF5). Then proceed without further check-ins
until the go/no-go.

## Phase 1: Changelog by fan-out (facts first, prose second)

Range is `vLAST..upstream/dev`. Fan out parallel read-only agents, one lens
each. Every agent returns **numbered fact entries with evidence**: one-line
claim + commit SHA or PR number + file path. No pointer, no fact. Uncertain
items get marked uncertain, never dropped.

Lenses:

1. **New models and tasks**: model families, task types, weight variants
   added. Check `libreyolo/models/`, registry/config files, `WEIGHT_VARIANTS`,
   `libreyolo/config/datasets/`, docs pages added.
2. **Features and API surface**: new CLI commands/flags, new public API
   (exports in `libreyolo/__init__.py`), export formats, new skills shipped
   in `skills/`.
3. **Bug fixes**: every fix, including obscure ones. `git log --grep` for
   fix/bug plus a pass over merged PRs
   (`gh pr list -R LibreYOLO/libreyolo --state merged --search "merged:>=<base-date>"`).
4. **Breaking changes and deprecations**: changed signatures, renamed or
   removed public names, changed defaults, CLI arg changes. Diff
   `libreyolo/__init__.py`, `libreyolo/cli/`, and anything in
   `docs/*_schema.md` between the two refs. This lens decides whether the
   version is actually a minor or needs louder warnings.
5. **Training / internals / performance**: trainer, losses, augmentations,
   postprocess moves, refactors big enough to mention.
6. **Tests, CI, docs, packaging, licensing**: new test files/counts,
   workflow changes, dependency changes, NOTICE / THIRD_PARTY_NOTICES
   entries added or changed.
7. **People and numbers**: `git shortlog -sn vLAST..upstream/dev`, commit
   count, files touched, closed issues, PR count.

Merge into `facts.md` (numbered `F1..Fn`, grouped by lens), then write
`changelog.md` from it:

```markdown
## LibreYOLO vX.Y.Z

One-paragraph summary, leading with the strongest item.

### New models          (only if any; each with sizes and license posture)
### Features
### Improvements
### Bug fixes           (one line each, plain language)
### Breaking changes    (only if any; what breaks and what to do instead)
### Contributors        (everyone, alphabetized, with what they did)
### Stats               (N commits, N files, +N/-N lines, N new tests)

`pip install --upgrade libreyolo`
```

Tone: factual, concrete, numbers over adjectives, zero hype. No em dashes.
Every line traces to a fact id. Show the changelog to the user **now**;
they curate (a fix can advertise the bug), you never silently drop items.

## Phase 2: Gates

Run these in parallel where possible; append each result to the scoreboard.

### Gate A: unit CI on the exact candidate SHA

Already covered by preflight check 4 if dev HEAD is the candidate. If the
latest dev run is stale or red, stop here.

### Gate B: GPU e2e on Modal (reuse the nightly, don't reinvent it)

The nightly harness is the canonical GPU gate. Two ways to run it against
the candidate; prefer the first (no local credentials needed):

```bash
# 1. Dispatch the existing workflow; force bypasses the tested-SHA cache
gh workflow run e2e-nightly-dev.yml -R LibreYOLO/libreyolo -f force=true
gh run watch -R LibreYOLO/libreyolo <run-id>   # ~up to 3h; check step summary

# 2. Or run the Modal app directly (needs MODAL_TOKEN_ID/SECRET in env)
uvx modal run tools/ci/modal_nightly.py --ref <candidate-sha> --target test_nightly
```

Both print a `MODAL_NIGHTLY_RESULT {json}` line with `status`, runtime, and
estimated GPU cost; record status **and cost** on the scoreboard. If the
last successful nightly already ran on the candidate SHA (check the run's
step summary for the ref), count it and skip the re-run; that is the whole
point of reusing the nightly.

For gates beyond the nightly contract (RF5 training benchmark, a heavier
suite, a specific GPU), use the `launch-serverless-gpu-job` skill
(Vast / Modal / Beam) instead of hand-rolling anything.

### Gate C: risk-targeted e2e subset (local GPU, optional but cheap)

Map Phase-1 facts to test files: any model family touched in the range gets
its `tests/e2e/test_<family>*.py` run locally via the
`libreyolo-run-e2e-tests` skill (one file per process, `-m "e2e and not
rf5"`). This catches family-specific regressions the general nightly case
may not exercise. Skip if no local GPU; note it on the scoreboard.

### Gate D: weight autodownload probe

Every weight variant that is new or renamed in the range must actually
resolve. For each, HEAD-request its Hugging Face URL (or instantiate the
model with autodownload in a temp cache) and record HTTP 200. A release
that advertises a model whose weights 404 is the most embarrassing failure
this project can have; this gate exists because of that.

### Gate E: wheel build + fresh-venv smoke

```bash
python -m build              # sdist + wheel; MANIFEST.in must keep weights out
python -m venv /tmp/relsmoke && /tmp/relsmoke/bin/pip install dist/*.whl
/tmp/relsmoke/bin/python -c "import libreyolo; print(libreyolo.__version__)"
/tmp/relsmoke/bin/libreyolo --help
```

Check the sdist/wheel size is sane (no accidentally bundled weights or
datasets) and the version string matches expectation.

### Gate F: notices and license sync

If Phase-1 lens 1 found new model families or ported code, verify `NOTICE`,
`THIRD_PARTY_NOTICES.txt`, and `weights/LICENSE_NOTICE.txt` gained the
matching entries in the same range. A new family with no notices diff is a
red flag; surface it as a licensing decision for the user, never patch it
silently.

### Gate G: docs drift

Every headline changelog item should be reachable in docs: repo `docs/`
and, where relevant, the website repo. List headline facts with no doc
mention; the user decides whether that blocks or ships as follow-up.

## Phase 3: Go/no-go, then cut it

Present the scoreboard + curated changelog. On explicit "go":

1. **Merge `release` -> `dev` first** (recovers release-only hotfixes;
   preflight check 3 told you if there are any). Resolve conflicts on dev;
   the version line auto-merges to the clean release value, so **manually
   set dev back to the next `X.Y.0.dev0`**.
2. **Bump + PR.** Branch off merged dev, set `pyproject.toml` to clean
   `X.Y.0`, push, open PR with base `release`. Remember: this PR shows no
   CI checks by design. Merge with a **merge commit, never squash** (squash
   collapses the whole cycle's history).
3. **GitHub release.** Draft it with the changelog so the publish moment is
   one click:

   ```bash
   gh release create vX.Y.Z -R LibreYOLO/libreyolo --target release \
     --title "LibreYOLO vX.Y.Z" --notes-file changelog.md --draft
   ```

   Publishing the release creates the tag, which fires `publish.yml`.
4. **Approve "Publish to PyPI"** in the Actions run. This is the only
   human-required click; do not try to automate it.

## Phase 4: Post-publish verification (do not skip)

```bash
# PyPI index catches up within minutes; poll, don't assume
pip index versions libreyolo
python -m venv /tmp/pypismoke && /tmp/pypismoke/bin/pip install libreyolo==X.Y.Z
/tmp/pypismoke/bin/python -c "import libreyolo; assert libreyolo.__version__ == 'X.Y.Z'"
# GPU e2e against the released branch (manual dispatch only)
gh workflow run e2e-nightly-release.yml -R LibreYOLO/libreyolo
```

Append results to the scoreboard. The release is "done" when PyPI installs
clean and the release nightly is green (or dispatched with the user's OK to
not wait).

## Phase 5: Marketing handoff

Hand `facts.md` + `changelog.md` to the marketing repo's
`new-version-release` skill (checkout at `../marketing`). It owns curation
for the announcement, the GitHub release body polish, the LinkedIn carousel
pipeline (generate -> audit -> simulator -> final review -> queue), and the
Reddit GIF post pack. Your facts file matches its expected format (numbered
facts with evidence), so it can skip its own fact-finding phase and start
at curation. Never post to social platforms yourself; assets are queued for
the user to post by hand.

## Anti-patterns

- Writing the changelog from memory or from PR titles alone; titles lie,
  diffs don't.
- Treating the empty checks list on the `dev -> release` PR as green.
- Re-running a 3-hour Modal nightly when the last green run already tested
  the candidate SHA.
- Squash-merging the release PR.
- Bumping the version on dev instead of on the release-PR branch.
- Skipping Phase 4 because "publish.yml was green" (green publish and a
  broken install have coexisted before in other projects; verify the
  artifact users actually get).
- Fixing a missing license notice yourself instead of surfacing it.

## Related

- `RELEASING.md`: the minimal engineering process this skill wraps.
- `skills/libreyolo-run-e2e-tests/`: how to run e2e correctly (markers,
  one-file-per-process rule).
- `skills/launch-serverless-gpu-job/`: rented/serverless GPUs for gates
  beyond the nightly contract.
- `../marketing/skills/new-version-release/`: the announcement pipeline
  (curation, GitHub release body, LinkedIn carousel, Reddit GIF).
- `.github/workflows/`: `e2e-nightly-dev.yml`, `e2e-nightly-release.yml`,
  `publish.yml`, `unit-tests.yml`, `install-smoke.yml`.
