---
name: merge-to-dev
description: >-
  The whole dance for landing code on LibreYOLO's dev branch: branch, commit,
  push to upstream, then either open the PR against dev or hand the user a
  one-click compare URL that pre-fills the PR title and description (including
  the required Code provenance section), and once the PR exists, babysit the
  Greptile bot review until it is happy. Agents may open the PR but never
  approve or merge it. Use whenever the user says "put this on dev", "push this to dev",
  "merge this to dev", "ship this", "open a PR for this", or hands over
  finished work on LibreYOLO/libreyolo. The user should never have to ask for
  the PR link; producing it is the task.
---

# Merge code to dev

There is exactly one way code lands on `dev` in `LibreYOLO/libreyolo`:
**branch -> commit -> push to upstream -> a PR with base `dev`**. Never push to
`dev` directly, even though the account has admin.

Per `AGENTS.md`, an agent **may open that PR**, but opening it is where the
agent's authority stops: never approve, never merge, never dismiss a review
finding. Two valid endings, and the choice is the user's:

- **Open the PR** (`gh pr create --base dev`) when the user asked for a PR or
  the work is finished and self-contained. Say in the description that an agent
  opened it, and what was verified.
- **Hand over a one-click compare URL** that pre-fills the title and description
  (with the required Code provenance section) when the work is exploratory, or
  the user wants the description in their own words.

Either way the description carries an accurate `## Code provenance` section, and
you end your turn with the PR or compare link, not with a question.

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

First resolve the remote that points at `LibreYOLO/libreyolo` and use it
everywhere below (do not hardcode `upstream`; a fresh clone may only have
`origin`). This same `$REMOTE` is reused at the push step:

```bash
REMOTE=$(git remote -v | awk 'tolower($0) ~ /github.com[:\/]libreyolo\/libreyolo/ {print $1; exit}')
[ -n "$REMOTE" ] || { echo "no remote points at LibreYOLO/libreyolo; add one first"; exit 1; }
```

Branch off up-to-date dev, named `<issue-number>-<short-slug>` when there
is an issue (repo convention, e.g. `477-add-deblurring`), otherwise a short
descriptive slug:

```bash
git fetch "$REMOTE"
git switch -c 512-fix-thing "$REMOTE/dev"
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

### 3. Pre-review with the Greptile CLI (skip if not installed)

If the `greptile` CLI is available (`command -v greptile`), run a local
review after committing and judge its findings the same way as the bot
review in step 5: fix what's right, rebut what's wrong. Not installed?
Skip this step; the bot still reviews the PR once it opens.

```bash
greptile review --agent -b dev   # -b release for release PRs
```

Reviews run server-side and can take 10+ minutes, so run it in the
background rather than foreground. The exit code is 0 even when there are
findings; read the output.

### 4. Push and hand over the one-click PR link

Push through the same `$REMOTE` resolved in step 1 (normally `upstream`
here; `origin` is dead):

```bash
git push -u "$REMOTE" <branch>
```

The compare URL below is a github.com URL, so it is unaffected by which
remote name you pushed through.

Then build the handoff URL with the **title and description pre-filled** and
hand it over. A bare `compare/...?expand=1` link is not enough: for a
single-commit branch GitHub fills the body from the commit message, and the
required provenance section comes up blank. GitHub honours `title` and `body`
query params on the compare page, so pre-fill them yourself and the human
lands on a PR form already filled in.

Write the body the way `.github/pull_request_template.md` asks: a description
plus, **required, a `## Code provenance` section**. The `provenance-check` CI
gate fails the PR if that section is missing or empty, so it is the one part you
must always include.

**Keep it short and factual. Bullets, not prose.** Per `AGENTS.md`: what
changed, why, what to check, what was not verified. No process narration, no
adjectives, no selling. Ten lines beats fifty. Shape:

```markdown
What: <one line>
Why: <one line>

- <change 1>
- <change 2>

Check: <what the reviewer should look at>
Not verified: <what you did not test, or "nothing outstanding">
Opened by an agent.

## Code provenance
<one accurate line, see below>
```

For the provenance section itself, short factual prose, no checkboxes and no
tables:

- First-party only: a `## Code provenance` heading followed by one line, e.g.
  "Original code written for this PR; bug fixes to LibreYOLO's own first-party
  code, no third-party code ported, adapted, or introduced; no
  GPL/AGPL/LGPL/non-commercial/unknown-license material involved."
- Ported or adapted code: name the upstream repository, the commit or version,
  and its license (permissive only: MIT, Apache-2.0, BSD, or similar), and
  update the notice files.

Write the description to a scratch `body.md` (so multi-line content survives),
then URL-encode title and body and print the link (same mechanism the
`libreyolo-report-issue` skill uses for issues):

```bash
python -c "import urllib.parse,sys; print('https://github.com/LibreYOLO/libreyolo/compare/dev...'+sys.argv[1]+'?expand=1&title='+urllib.parse.quote(sys.argv[2])+'&body='+urllib.parse.quote(sys.argv[3]))" <branch> "<title>" "$(cat body.md)"
```

Hand the user that single pre-filled link and stop.

**Or open it yourself.** `AGENTS.md` allows an agent to open the PR
(`gh pr create --base dev --title ... --body ...`), but never to approve, merge,
or dismiss a review finding. Prefer opening it when the user asked for a PR;
prefer the pre-filled link when the work is exploratory or the user wants the
prose in their own words. Either way the `## Code provenance` section is the
part you own: never leave it blank or the CI gate fails, and never let the
description imply a review that has not happened.

**Deliver the link without being asked.** "Pushed the branch" is not a
finished turn; the PR or the compare link is the deliverable. CI
(`unit-tests.yml`, `install-smoke.yml`) runs once the PR to `dev` exists.

### 5. Babysit Greptile (after the PR is open)

This step needs a PR to exist, so it starts once you have opened it or the
human has (or tells you "it's open" / "ship it"). Do not sit polling
for a PR number before then. When the PR author is one of the repo admins,
the Greptile bot reviews the PR automatically a few minutes after it opens
(and again after each push).
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
     Put a one-line factual rebuttal in your summary to the user, and let
     **them** reply on the thread if they want it recorded. `AGENTS.md`:
     "Agents must not post ... PR comments unless a human explicitly asks."
     Do not reply on the Greptile thread yourself.
   - **Judgement call** (style, scope): lean toward fixing cheap ones,
     surface expensive ones to the user.
3. Done when the latest Greptile review has no unaddressed findings and its
   summary reads as approving (it scores confidence like "5/5, safe to
   merge"). Also confirm CI checks with `gh pr checks <n>`, and read the
   result rather than just counting green. An **empty** or all-skipped check list
   is not "green": a PR based on `release` triggers no CI at all (workflows
   fire on `dev` only), so "no checks" there means "untested", not "passed".
   Report that honestly rather than as a pass.

Note on how to read Greptile's output: its substantive review is posted as
a PR summary comment (fetch `gh api repos/LibreYOLO/libreyolo/issues/<n>/comments`),
with per-finding items and a confidence score, plus sometimes inline
comments. The GitHub "review" object body is often empty, so an empty
review body does not mean it found nothing; read the summary comment. When
re-reviewing, findings anchored to a superseded commit SHA are already
addressed; only findings on the latest commit are live.

### 6. Report

End the turn with: PR URL, one-line summary of the change, CI status
(honest about untested release PRs), Greptile verdict (score + how many
findings were fixed vs rebutted), and whether it is ready to merge. **Do
not merge the PR yourself** unless the user explicitly says to merge;
merging to dev is their click.

## Common variants

- **"This is for release, not dev"**: same dance, but the compare URL bases
  on `release` (`compare/release...<branch>?expand=1`); that only happens
  during a release cut or hotfix, so confirm first. Release PRs trigger no
  CI (workflows fire on `dev` only), so `gh pr checks` returns an empty or
  all-skipped list. Report that as "no CI ran (untested)", never as green,
  and do not run the Greptile-done "checks are green" step against it.
- **Work in a worktree**: same flow; push from the worktree. The branch is
  what matters, not which checkout it sits in.
- **User says "ship it" on an already-open PR**: skip to step 5; the job
  is Greptile + CI + report.
- **Fork PRs / external contributors**: Greptile still reviews, but you
  cannot push to their branch. Do **not** post the findings as PR comments
  to route them (that needs the human's explicit ask, per `AGENTS.md`);
  relay them to the user in your summary and let the human decide what to
  post or request from the contributor.

## Anti-patterns

- Pushing to `dev` or `release` directly. Never, admin or not.
- Approving or merging your own PR, or dismissing a Greptile finding to make a
  check go green. Opening the PR is allowed; deciding it is good is not.
- Handing a bare `?expand=1` link with no `&body=`. It leaves the required
  `## Code provenance` section blank, so the `provenance-check` CI gate fails.
- Padding the description with checkbox lists or a provenance table. The
  template is free-form prose; only a filled `## Code provenance` section is
  required. Keep it simple.
- Ending the turn with "want me to open a PR?". Hand the link.
- Replying on the Greptile thread to rebut a finding. Put the rebuttal in
  your summary; the human comments if they want to.
- Passing `-c "<comment>"` to `gh pr close`/`gh pr reopen`. That posts a PR
  comment, which needs the human's explicit ask. Close/reopen bare.
- `git add -A` in the dirty main checkout.
- Blindly applying every Greptile comment. It's usually right, not always
  right; a wrong "fix" that lands because a bot suggested it is still your
  bug.
- Marking the dance done before Greptile's re-review of your latest push.
- Merging without being told.
