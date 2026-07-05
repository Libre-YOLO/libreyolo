---
name: merge-to-dev
description: >-
  The whole dance for landing code on LibreYOLO's dev branch: branch, commit,
  push to upstream, hand the user a one-click PR link (or open the PR),
  then babysit the Greptile bot review until it is happy. Use whenever the
  user says "put this on dev", "push this to dev", "merge this to dev",
  "ship this", "open a PR for this", or hands over finished work on
  LibreYOLO/libreyolo. The user should never have to ask for the PR link;
  producing it (and handling Greptile) IS the task.
---

# Merge code to dev

There is exactly one way code lands on `dev` in `LibreYOLO/libreyolo`:
**branch -> commit -> push to upstream -> PR with base `dev`**. Never push
to `dev` directly, even though the account has admin. When the user says
"put this on dev", run the entire dance below and end your turn with the
PR link and the Greptile verdict, not with a question.

## Environment gotchas (read first, they bite every session)

- The repo has **no `main`**; branches are `dev` and `release`. PRs base on
  `dev` unless the user says otherwise.
- The `origin` remote is dead. Push to **`upstream`** (LibreYOLO/libreyolo);
  the account (EHxuban11) has admin there.
- On this Windows box, `git` and `uv` are not on the PowerShell PATH; use
  the **Bash** tool for git work.
- The main checkout's working tree is usually dirty with unrelated
  experiments. Commit **only the files that belong to this change**; never
  `git add -A` in the main checkout. If the change is tangled with
  unrelated edits, stage file by file (or hunks with `git add -p`).

## The dance

### 1. Branch

Branch off up-to-date dev, named `<issue-number>-<short-slug>` when there
is an issue (repo convention, e.g. `477-add-deblurring`), otherwise a short
descriptive slug:

```bash
git fetch upstream
git switch -c 512-fix-thing upstream/dev
```

If the work already sits on a correctly-named branch, reuse it. If the work
sits on the **wrong** branch (someone committed on top of an unrelated
feature branch), move it: branch from `upstream/dev` and cherry-pick or
re-stage just the relevant files. Do not open a PR whose diff drags in an
unrelated feature.

### 2. Commit

Small, plain, imperative subject lines matching repo history ("Fix X",
"Add Y"). Run the relevant unit tests before pushing when the change
touches `libreyolo/`:

```bash
PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/unit/<touched-area> -q
```

Skills/docs-only changes have no tests to run; say so and move on.

### 3. Push and produce the PR

```bash
git push -u upstream <branch>
```

Then either open the PR directly (default when the change is
self-explanatory):

```bash
gh pr create -R LibreYOLO/libreyolo --base dev --head <branch> \
  --title "<title>" --body "<what and why, issue ref like 'Closes #512'>"
```

or, if the user likes to write the description themselves, give the
one-click compare URL:

```
https://github.com/LibreYOLO/libreyolo/compare/dev...<branch>?expand=1
```

**Always deliver one of these two without being asked.** "Pushed the
branch" is not a finished turn; the link is the deliverable. Note: CI
(`unit-tests.yml`, `install-smoke.yml`) runs on PRs to `dev`, so the PR is
also what buys you the CI signal.

### 4. Babysit Greptile

When the PR author is one of the repo admins, the Greptile bot reviews the
PR automatically a few minutes after it opens (and again after each push).
Its reviews are usually good; treat them as a real reviewer, not noise.

Loop until happy:

1. Wait ~2-3 minutes after opening/pushing, then read everything:

   ```bash
   gh pr view <n> -R LibreYOLO/libreyolo --json reviews,comments
   gh api repos/LibreYOLO/libreyolo/pulls/<n>/comments   # inline comments
   ```

   If nothing from Greptile yet, poll every couple of minutes (up to ~10);
   don't declare victory on an empty review list.
2. For each Greptile finding, judge it on the merits:
   - **Right** (real bug, real improvement): fix the code, commit, push.
     The push triggers a re-review; go back to step 1.
   - **Wrong or not applicable**: don't change the code to appease the bot.
     Reply on the thread with a one-line factual reason so the resolution
     is recorded, and tell the user in the summary.
   - **Judgement call** (style, scope): lean toward fixing cheap ones,
     surface expensive ones to the user.
3. Done when the latest Greptile review has no unaddressed findings and its
   summary reads as approving (it scores confidence like "5/5, safe to
   merge"). Also confirm CI checks are green: `gh pr checks <n>`.

### 5. Report

End the turn with: PR URL, one-line summary of the change, CI status,
Greptile verdict (score + how many findings were fixed vs rebutted), and
whether it is ready to merge. **Do not merge the PR yourself** unless the
user explicitly says to merge; merging to dev is their click.

## Common variants

- **"This is for release, not dev"**: same dance with `--base release`,
  but that only happens during a release cut or hotfix; confirm first.
  Remember release PRs from dev show no CI checks (workflows trigger on
  dev only).
- **Work in a worktree**: same flow; push from the worktree. The branch is
  what matters, not which checkout it sits in.
- **User says "ship it" on an already-open PR**: skip to step 4; the job
  is Greptile + CI + report.
- **Fork PRs / external contributors**: Greptile still reviews, but you
  cannot push to their branch; findings become review comments instead of
  commits.

## Anti-patterns

- Pushing to `dev` or `release` directly. Never, admin or not.
- Ending the turn with "want me to open a PR?". Open it or hand the link.
- `git add -A` in the dirty main checkout.
- Blindly applying every Greptile comment. It's usually right, not always
  right; a wrong "fix" that lands because a bot suggested it is still your
  bug.
- Marking the dance done before Greptile's re-review of your latest push.
- Merging without being told.
