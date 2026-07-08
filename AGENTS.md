# Agent Instructions

## Licensing policy (read this first)

- This is the most important policy in this repository. LibreYOLO's entire
  value is being genuinely MIT; one license violation endangers the project.
- LibreYOLO faithfully respects open-source licenses.
- Agents must not copy, adapt, paraphrase, or derive code from any third-party
  project unless that project is explicitly licensed under MIT, Apache-2.0,
  BSD, or a similarly permissive license compatible with LibreYOLO's licensing
  requirements. Unknown or missing license means incompatible.
- Never rewrite, rename, or restructure incompatibly-licensed code to obscure
  its origin. A GPL function with new variable names is still a derivative
  work. The only acceptable remedies are re-derivation from a genuinely clean
  source with documented provenance, or removal. That choice belongs to the
  maintainer: surface it, never pick silently.
- If an agent may have been exposed to, influenced by, or contaminated by code
  under GPL, AGPL, LGPL, proprietary, unknown, or otherwise incompatible terms,
  the agent must immediately stop work on the affected area, flag the
  contamination risk to the developer, and avoid contributing the affected
  code. Flagging is never the wrong move; quiet contribution always is.
- Ported or adapted code must state its upstream: repository, commit, and
  license, in the PR description and the notice files.
- See `skills/libreyolo-license-audit/` for the audit discipline and the
  notice surfaces.

## Agent conduct

- Agents must not open GitHub issues.
- Agents must not open pull requests.
- Agents must not post issue comments or PR comments unless a human explicitly
  asks for it.
- Humans handle issue creation, PR creation, review submission, and final merge
  decisions.
- Agents may reply with a one-click GitHub URL (no description pre-filled) so
  the human can open the PR or issue themselves.
- When possible, work in git worktrees
- `release` is the default branch that visitors land on and clone; `dev` is the
  integration branch where all development lands before it is promoted to a release.

## Reporting problems upstream

- For the project's benefit, agents should offer to report anything weird
  they hit while working with the library: errors and crashes, weights that
  fail to download or load, tasks that took real struggle because
  documentation was missing or wrong, and code bugs discovered along the way.
- Use `skills/libreyolo-report-issue/` for this: it drafts an anonymized
  issue and hands the user a pre-filled GitHub URL to submit with one click,
  so the "agents do not open issues" rule stays intact while the signal still
  reaches the maintainers.

## Commit policy
- Do not add LLMs or agent tools as co-authors in commits.
- Keep commit messages short and factual.
- Avoid pushing docs, artifacts, helper scripts, or anything that should not go into the upstream LibreYOLO library

## Documentation

- Contributor-facing policy lives in `CONTRIBUTING.md`.
- Exceptionally important schemas and contracts such as the checkpoint metadata standard live under /docs in the libreyolo repository. They are short and factual.
- `/docs/checkpoint_schema.md` documents checkpoint metadata rules used for
  loading, identifying, and validating model checkpoints.
- `/docs/nomenclature.md` documents canonical model names, filename rules,
  family/size/task conventions, and task-resolution behavior.
- `/docs/testing.md` documents test tiers, CI expectations, smoke tests, nightly
  tests, and manual validation policy.
- `/docs/adr/` documents architecture decisions and design contracts.

## README policy

- `README.md` is the project's landing page: the first five minutes a
  developer spends on LibreYOLO, and the most important five minutes the
  project gets. Treat it as a high-stakes surface, not routine documentation.
- Do not modify `README.md` or `README.zh-CN.md` a priori. The only
  unprompted reason to propose a change is a real error or inconsistency
  with the code, and even then: propose the exact diff to the user and get
  approval before landing it.
- Never add new sections, restructure, or grow the README without an
  explicit user request. It balances being a landing page against being
  documentation; depth belongs in `/docs` or on the website.
- Style: no em dashes, no decorative or AI-flavored characters, no fluff.
- The detailed editing contract lives in `skills/libreyolo-update-readme/`.

## Review guidelines

- These guidelines apply to agents performing PR reviews, not agents
  implementing code changes.
- PR-review agents must read `REVIEW.md` before reviewing pull requests.
- Treat `REVIEW.md` as repository context for scope, contracts, and common
  regression risks.
- If a PR conflicts with `REVIEW.md`, flag the conflict with concrete file
  evidence.

## Pull Request (PR) policy
- All development PRs target `dev`, never `release`. This holds regardless of
  which branch GitHub shows as the default: `release` is the public default
  branch, but it only receives curated release merges, not feature PRs.
- Before reviewing or changing PRs, read the relevant files under /docs,
  especially documented schemas, contracts, and architecture decisions.
- Prefer one PR per problem, or per small group of tightly related problems.
- When debugging a specific model or issue, avoid changing global behavior or
  shared code unless the shared change is genuinely necessary.
- Shared-code changes are allowed when they are the clean software engineering
  solution, but the PR must explain why the shared change is necessary and what
  other models or workflows it may affect.
- Keep PRs to the least code needed to solve the stated problem.
- Do not mention other computer vision libraries in PR titles or descriptions
  unless the comparison is necessary to explain compatibility or API behavior.
- Encourage humans to write PR titles and descriptions in their own words so the
  history is easier for reviewers and future readers to follow.

## General library constraints
- Generally every user facing API (Python, yamls, etc) has to follow the de-facto YOLO CLI/API conventions
- The Flagship models of LibreYOLO are YOLO9 (CNNs) and RF-DETR (transformers), and new features have to at least cover this two
