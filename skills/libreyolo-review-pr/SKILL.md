---
name: libreyolo-review-pr
description: >-
  Review a LibreYOLO pull request the way this repo expects: contract-first
  (REVIEW.md axioms + /docs schemas), evidence-based, and verified live in a
  worktree rather than by reading the diff alone. Use whenever the user asks
  to review a PR, assess an external contribution, second-opinion a branch,
  or "deep review" something before merge. Covers the reading order, the
  worktree + live-verification method, the finding taxonomy and severity
  bar, and the delivery rules (findings go to the user; agents never post PR
  comments or reviews themselves).
---

# Review a LibreYOLO PR

The repo has explicit review doctrine: `REVIEW.md` (axioms + focus list) and
`AGENTS.md` (what agents may and may not do). This skill turns them into a
working procedure. The one-line version: **check the PR against the
contracts, then run it; deliver findings to the user, never to GitHub.**

## Ground rules (AGENTS.md, verbatim intent)

- Agents do not submit reviews, do not post PR comments, do not approve or
  request changes. Findings go in your message to the user; they decide what
  lands on the thread.
- Read `REVIEW.md` before reviewing. Read the `/docs` contract files touched
  by the PR (`checkpoint_schema.md`, `nomenclature.md`, `dataset_schema.md`,
  `testing.md`, relevant `adr/`). A PR that conflicts with a contract gets
  flagged with concrete file evidence, even if the code is good.
- Scope discipline cuts both ways: flag unrelated changes bundled in, and
  do not demand out-of-scope improvements as blockers.

## Reading order (before any opinion)

1. The linked issue: what problem was agreed? (CONTRIBUTING requires one for
   non-trivial PRs; its absence on a large PR is itself a finding.)
2. The PR description vs the diff stat: does the description cover every
   meaningful behavior change? Omissions are a REVIEW.md focus item.
3. `REVIEW.md` axioms most likely violated by this PR's shape. Recurring
   high-yield ones: metadata is the loading source of truth (no filename
   heuristics), cross-family/cross-task loads must fail, explicit user
   kwargs beat defaults, DDP fixes must not regress single-GPU, original
   canvas coordinates are canonical, no silently-ignored options, no
   heavyweight tests in the unit suite, license compatibility.
4. The diff itself, shared-code files first (`models/base/`, `training/`,
   `validation/`, `cli/`, `data/`): blast radius before details.

## Verify live, not by eyeball (the part most reviews skip)

Check the PR out into a worktree and prove the claims:

```bash
git fetch upstream pull/<n>/head:pr-<n>
git worktree add .claude/worktrees/review-pr-<n> pr-<n>
cd .claude/worktrees/review-pr-<n>
```

Then, scaled to the PR's risk:

- **Always**: run the unit tests for the touched areas
  (`PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/unit/<area> -q`),
  plus the full PR gate if shared code moved (`libreyolo-run-unit-tests`).
- **Model/inference PRs**: load the model and run a real predict on
  `SAMPLE_IMAGE`; check the claimed outputs exist on `Results`.
- **Training PRs**: at minimum the rung-0 overfit check from
  `libreyolo-verify-training`; a trainer claim without a run behind it is
  unverified, say so.
- **Metric/validation PRs**: run `val` on a small set before and after;
  numbers that move must be explained by the PR, not discovered by users.
- **New-weights PRs**: resolve the weight (autodownload or staged) and
  confirm checkpoint metadata against `docs/checkpoint_schema.md`.
- **Ported-code PRs**: run the license/provenance checklist from
  `libreyolo-license-audit`; a licensing doubt is a blocking finding.

The point of live verification is asymmetry: a diff can look perfect while
the feature does not work (a validator default that buries a model's real
F1, a train arg that is parsed and ignored). Every past deep review that
found the big bug found it by running the code, not reading it.

## Finding taxonomy and severity bar

Report findings ranked, each with file:line evidence and, for bugs, the
concrete failure scenario (inputs, then wrong output). Severity language:

- **Blocking**: correctness bugs, contract violations (docs schemas,
  REVIEW.md axioms), licensing, silent behavior changes to existing users,
  API that accepts-and-ignores options.
- **Should-fix**: missing tests for the changed behavior, docs drift for a
  contract file, error messages that will generate support issues.
- **Note**: style, naming, non-blocking simplifications. Keep these few;
  the repo values small focused reviews over exhaustive nit lists.

Do not restyle the contributor's code, and judge by the repo's actual
conventions, not personal preference. When a Greptile bot review exists,
read it and fold it in: agree, rebut with evidence, or mark as judgement
call; do not duplicate or blindly endorse it (the babysitting loop itself
belongs to `skills/merge-to-dev`).

## External-contributor PRs

You cannot push fixes to a fork branch, and posting review comments needs an
explicit human ask. So the deliverable is a message the user can act on:
findings ordered by severity, with copy-pasteable suggestions where cheap.
Be respectful of the contribution in tone; the summary you write may be
pasted verbatim.

## Deliver

End with: verdict (mergeable / mergeable-after-fixes / needs-rework), the
ranked findings, what you verified live (commands + outcomes) vs only read,
and any contract files the PR must update before merge. Then clean up the
review worktree (`git worktree remove ...`) unless iterating.

## Related

- `REVIEW.md`: the axiom list this skill applies.
- `skills/merge-to-dev/`: landing your own work + Greptile babysitting.
- `skills/libreyolo-verify-training/`, `skills/libreyolo-run-unit-tests/`,
  `skills/libreyolo-license-audit/`: the verification depth per PR type.
