---
name: libreyolo-update-readme
description: >-
  Edit LibreYOLO's README.md (and its README.zh-CN.md mirror) safely. Use
  whenever someone asks to update, fix, extend, or review the README, when a
  README claim contradicts the code ("the readme says X but..."), when a
  release or new feature tempts a README mention, or when reviewing a PR that
  touches README files. The README is the project's landing page and is
  deliberately protected: default to NOT changing it, propose before editing,
  and never add sections or restructure without the user's explicit OK.
---

# Update the LibreYOLO README

The README is the first five minutes a developer spends on LibreYOLO, and
those are the highest-stakes five minutes the project has. It is a landing
page first and documentation second. Every edit is therefore opt-in, small,
and verified; "I improved the README while I was at it" is a bug, not a
favor.

## The prime directive: do not touch it a priori

- The only unprompted reason to propose a README change is a **real error or
  inconsistency**: a command that no longer runs, a weight or class name that
  does not exist, a compatibility-table cell that contradicts
  `SUPPORTED_TASKS` or the callable API, a dead link.
- Even then, **propose first**: show the user the exact diff and the evidence
  (what you ran, what the code says), and land it only after their approval.
- A new feature landing in the library is NOT by itself a reason to grow the
  README. Ask whether it belongs there; the answer is usually "no, it goes
  in /docs or on the website".
- Never bundle README edits into an unrelated PR. One PR, one problem.

## Landing page vs documentation (the balance to hold)

The README sells the project honestly and gets a developer to a working
`predict` in minutes. It is not the manual:

- Prefer linking out (website docs, /docs contracts) over explaining in
  place. Depth lives elsewhere; the README carries the shortest true version.
- Every added line must earn its place against the whole page's scan-ability.
  If a section needs scrolling to skim, it is too long for a landing page.
- **No new sections, no reordering, no renaming of sections without the
  user's explicit approval.** Read the current section set from the file at
  edit time; do not assume it from memory.

## Style contract (hard rules)

- **No em dashes.** Use a hyphen, comma, colon, or a new sentence.
- **No AI-flavored characters or decoration**: no smart quotes, arrows,
  sparkles, decorative emoji, box-drawing, or invisible Unicode. Keep
  README.md ASCII apart from what legitimately exists (names, badges);
  README.zh-CN.md is CJK by nature, the rule there is "no decoration", not
  "no non-ASCII".
- **No fluff.** No hype adjectives, no "blazingly", no superlative without a
  verifiable number next to it. Plain, factual, concrete.
- Match the existing voice and formatting; the README should read as one
  hand's writing after your edit.
- Repo-wide rule applies here with extra force: no third-party CV library
  names unless the comparison is technically necessary (AGENTS.md).

## Truth rules (verify, never trust memory)

- Every command must run as pasted. Actually run it before proposing it.
- Model classes and weight filenames come from `docs/nomenclature.md` and
  `libreyolo models`, never from memory.
- Compatibility claims must match the callable API. Mark training as supported
  when an ordinary `train()` call reaches the trainer; keep RF1 evidence and
  known limits in the detailed documentation.
- Mind the version gap: the GitHub README is read by PyPI users of the
  released package, but it renders from `dev`. If a claim is true on dev and
  false on the released version, say so to the user and let them decide
  timing (usually: the README change rides the release, not the feature PR).
- Raw links into the repo use `/release/` (or `/dev/` deliberately), never
  `/main/`; there is no `main` branch.

## The zh-CN mirror

`README.zh-CN.md` mirrors README.md. Any approved README.md content change
either updates the mirror in the same PR or explicitly tells the user the
mirror now lags; silent drift between the two is a bug you created.

## Process

1. Gather evidence (run the command, read the code, check the link).
2. Show the user: what is wrong, the proposed diff, the mirror plan.
3. On approval, land it via the normal dance (`skills/merge-to-dev/`),
   README-only, smallest possible diff.
4. After merge, re-read the rendered page on GitHub once; rendering bugs
   (tables, badges) only show up rendered.

## Related

- `AGENTS.md` "README policy": the repo-level rule this skill implements.
- `skills/libreyolo-update-website-docs/`: where deep user docs actually go.
- `skills/libreyolo-api-conventions/` and `docs/nomenclature.md`: the naming
  facts README snippets must agree with.
- `skills/merge-to-dev/`: how the approved edit lands.
