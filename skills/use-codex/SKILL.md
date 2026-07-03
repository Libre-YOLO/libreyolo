---
name: use-codex
description: >-
  Delegate work to OpenAI's Codex from inside Claude Code — a generic task
  (implement a feature, investigate a bug, refactor) or a code review as a
  cross-model second opinion. Trigger ONLY when the user explicitly names
  Codex — e.g. "use codex to implement this", "have codex fix this", "delegate
  this to codex", "review this with codex", "codex adversarial review", "set up
  codex". Do NOT trigger on a plain "implement…" / "fix…" / "review…" with no
  mention of codex — those are Claude's own job (reviews go to the built-in
  /code-review). Also covers installing/setting up Codex.
---

# Use Codex

Codex is a separate agent with its own model and reasoning. You orchestrate it
and own the final integration: **Codex drafts → you review and reconcile → user
approves.** Never auto-apply its output. It starts cold on this repo — always
include the house rules (bottom) in the brief.

## Setup

Requires the `codex` CLI: `npm install -g @openai/codex`, then the user runs
`codex login` (interactive, ChatGPT/OpenAI account). On Windows, npm globals
often aren't on the shell PATH — if `codex` isn't found but node is installed,
prepend them first:

```powershell
$env:Path = "C:\Program Files\nodejs;$env:APPDATA\npm;$env:Path"
```

## Delegate a generic task

```
codex exec --sandbox workspace-write -a never "<task, acceptance criteria, constraints>"
```

- Codex edits the working tree directly. Show the user its plan and the
  resulting `git diff` before integrating.
- Analysis only, no edits: use `--sandbox read-only`.
- Model / effort per run: `-m <model>` and
  `-c model_reasoning_effort=<minimal|low|medium|high>`. Deep pass:
  `codex exec -m gpt-5-codex -c model_reasoning_effort=high …`.
  Persistent defaults live in `~/.codex/config.toml` (or `-p <profile>`).

## Delegate a code review

If the official plugin is installed, tell the user to run `/codex:review`
(`--base <ref>`, `--wait`/`--background`) or `/codex:adversarial-review`
(challenges the design, good before big merges) — you cannot type slash
commands yourself. Otherwise run it directly:

```
codex exec review --base dev
```

Feature branches review against `dev`. Require `file:line` citations; priority
correctness > security > spec > performance > maintainability > tests; no style
nits. Render verdict + findings for the user; do not apply fixes unasked.

## Other routes

- **MCP server** (autonomous multi-turn hand-off): user registers once with
  `claude mcp add codex -- codex mcp-server`; you then get `mcp__codex__codex`
  / `mcp__codex__codex-reply` tools to drive Codex sessions yourself.
- **Official plugin** (`codex-plugin-cc`) — user installs with:
  `/plugin marketplace add openai/codex-plugin-cc`, then
  `/plugin install codex@openai-codex`, `/reload-plugins`, `/codex:setup`.
  Beyond reviews it adds `/codex:rescue` (delegate a fix), `/codex:transfer`
  (persistent Codex thread), `/codex:status|result|cancel` (background jobs).
  Optional `--enable-review-gate` blocks Claude until Codex signs off — off by
  default, suggest only if asked. Plugin commands take model/effort from
  `~/.codex/config.toml`, not flags.

## House rules (include in every brief)

- Follow the repo's licensing policy: no third-party CV library names in
  committed text; respect clean-room boundaries.
- Branch flow is dev → release; feature branches base on `dev`.
- No new dependencies without surfacing them; match existing style; leave
  tests runnable.
