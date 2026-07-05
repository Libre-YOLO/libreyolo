---
name: libreyolo-triage-issue
description: >-
  Triage a LibreYOLO GitHub issue or user bug report: reproduce it, classify
  it, find the owning code, and hand the user a decision-ready assessment.
  Use when someone says "look at issue #N", "a user reports X", "triage the
  new issues", or pastes a bug report / stack trace from a LibreYOLO user.
  Covers the reproduce-first discipline, the version gotcha (released vs
  dev), the classification taxonomy, multi-part report handling, and the
  reply the human can post. Agents never open issues, close issues, or post
  comments; triage output goes to the user.
---

# Triage a LibreYOLO issue

Triage means the human can decide in one read: real or not, what kind, how
bad, where it lives, and what the fix path is. It does not mean fixing it
(that is a separate, explicitly-ordered step; when describing a problem, the
deliverable is the assessment).

## Ground rules

- Agents must not open issues, post issue comments, or close issues
  (AGENTS.md). Draft the reply; the human posts it.
- Reproduce before classifying. A report you have not reproduced (or
  concretely failed to reproduce) is an anecdote, not a bug.
- Multi-part reports get triaged **per claim**, never as a blob. Users often
  bundle one real bug, one already-fixed bug, one environment problem, and
  one misunderstanding in a single issue; each part gets its own verdict.
  (A four-part training-quality report once decomposed into two new bugs,
  one fixed-on-dev-only, and one expected behavior.)

## Step 1: pin the version (the recurring gotcha)

Users run the released PyPI package; you develop on `dev`. A bug can be
real on the release and already fixed on dev, or introduced on dev and
absent from the release. Establish both sides before reproducing:

- Ask/read the report for `libreyolo version` (or infer from the traceback's
  installed paths).
- Check whether the suspect code differs between `upstream/release` and
  `upstream/dev` (`git log upstream/release..upstream/dev -- <file>`).
- Reproduce against the user's version first (a scratch venv with the PyPI
  release), then against dev. The four verdict combinations mean different
  replies ("fixed in the next release" vs "confirmed, fixing" vs "cannot
  reproduce, need more info" vs "dev regression, not user-facing yet").

## Step 2: reproduce minimally

Shrink to the smallest command that shows the behavior, preferring bundled
assets (`SAMPLE_IMAGE`, `coco8.yaml`, tiny synthetic data) so the repro runs
anywhere. Record the exact command and output; it becomes the regression
test later and the evidence in the reply. If reproduction needs the user's
data/weights and they are not shareable, say precisely what minimal artifact
you need; that request is the draft reply.

For environment-shaped reports (imports, CUDA, missing extras): run the
user's scenario through `libreyolo checks` first; a large fraction of
"bugs" are a missing optional extra, and the reply is the `pip install
libreyolo[<extra>]` line plus, where fair, a finding that the error message
should have said so (which is itself a small real issue).

## Step 3: classify

One primary label per claim:

- **Correctness bug**: wrong output/metric/behavior. Note blast radius:
  one family or shared code; silently-wrong (bad) or loud-crash (less bad).
- **Regression**: worked in version X, broken in Y. Find the breaking commit
  (`git bisect` or targeted `git log` over the suspect file); regressions
  outrank equal-severity bugs because they break existing users.
- **Environment/install**: not a code defect; answer + consider an error-UX
  improvement.
- **Docs gap / expectation mismatch**: code behaves as designed but the
  design surprised a reasonable user. The fix is docs or an API discussion,
  answer honestly rather than defending the surprise.
- **Feature request**: route to the maintainer's roadmap judgement; do not
  triage-accept scope.
- **Upstream**: torch/transformers/exporter dependency behavior. Verify
  against a plain-upstream repro before claiming it, and note the pinned
  version that fixes it if known.

Severity, honestly assessed: does it corrupt results silently (worst class:
model produces plausible-but-wrong numbers), block a mainline workflow
(flagship-family predict/train/val/export), or annoy an edge case?

## Step 4: locate the owner

Map to the owning module: family code (`libreyolo/models/<family>/`),
shared trainer/validator/data/CLI, or export backend. Name the exact
function where the behavior originates, verified by reading it, not by
pattern-matching the stack trace's top frame (the true cause is often
upstream of the crash site). Check for siblings: a bug in one family's copy
of a pattern often exists in the families that share the pattern; a
one-line sibling sweep massively raises the triage's value.

## Step 5: deliver

Report per claim: verdict (confirmed / not-reproducible / by-design /
fixed-on-dev), version matrix, minimal repro, owning code (`file:line`),
severity, suggested fix path (one line, plus branch name convention
`<issue-number>-<short-slug>` if work proceeds), and a **draft reply** the
human can paste. Keep the draft factual and warm; users who file good
reports are contributors in the making. Do not start the fix unless the
user orders it (then: fix on a branch and hand to `skills/merge-to-dev`).

## Related

- `skills/libreyolo-run-unit-tests/`: turn the repro into a gate test.
- `skills/libreyolo-verify-training/`: triage for "training is bad" reports.
- `skills/merge-to-dev/`: the landing dance once a fix is ordered.
