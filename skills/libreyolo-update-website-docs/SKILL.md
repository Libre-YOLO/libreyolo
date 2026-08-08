---
name: libreyolo-update-website-docs
description: >-
  Signpost for updating www.libreyolo.com (docs pages, feature announcements,
  SEO articles) when something ships in the library. Use when a change needs
  user-facing docs beyond the repo ("document this on the website", "add a
  docs page for X", "the site still says Y"), when the release process flags
  docs drift (Gate G), or when someone wants an article written or the site
  deployed. The website is a SEPARATE repo with its own skills; this skill
  orients you, states when a website update is required, and hands off.
---

# Update the LibreYOLO website (signpost)

The website is **not** in this repo. Deep guides live with the code they
describe; this signpost only tells you where to go and when you must.

## The repos

| Repo | Role | Authoritative skills there |
|---|---|---|
| `LibreYOLO/libreyolo-website` (local: `C:\Users\Usuario\Documents\GitHub\libreyolo-website`) | The Next.js site behind `https://www.libreyolo.com` (docs pages, articles content) | `put-website-in-prod` (the only supported deploy path) |
| `marketing` (local: `C:\Users\Usuario\Documents\GitHub\marketing`) | Content pipelines | `website-article-writer`, `website-article-publisher`, `new-version-release` |

Those skills are authoritative for layout, conventions, and deploy
mechanics. If anything here disagrees with them, they win; update this
signpost rather than diverging.

## Two facts worth carrying in (they bite outsiders)

1. **Deploys are manual.** Pushing to the website repo does NOT deploy.
   Production goes live only via the `put-website-in-prod` skill (global
   `vercel` CLI, `vercel --prod`). "I merged the docs change" is not "the
   docs are live".
2. **Public raw links into the library repo must use `/release/`, never
   `/main/`.** The library has no `main` branch; a `/main/` raw URL 404s.
   Use `/dev/` only for deliberately-unstable references.

## When a library change REQUIRES a website update

- A new model family, task, or CLI command that users are meant to find
  (the repo `docs/` are contributor contracts; user docs live on the site).
- Changed user-facing behavior the site currently documents (check before
  shipping: search the website repo for the old name/flag).
- A release: the release process (`skills/libreyolo-release/` Gate G) lists
  headline changelog items with no doc mention; each needs a site update or
  the user's explicit "ship without docs" per item.
- For features without full validation, state the exact completed checks and
  known limits without changing whether the implemented API is shown as available.

When a feature ships without its docs, say so in the PR/release handoff
rather than letting it be discovered.

## Flow

1. Land and verify the library change first (docs describing unmerged
   behavior is drift in the other direction).
2. Switch to the website checkout; follow its conventions (articles have a
   bilingual `.md` + `.zh.md` convention and FAQ frontmatter; read the
   marketing repo's `website-article-writer` before writing any article).
3. Deploy with `put-website-in-prod` from the website repo. Verify the live
   page renders after deploy; do not report done on a successful build
   alone.

## Related

- `skills/libreyolo-release/` Gate G: the docs-drift gate this feeds.
- `skills/benchmark-on-visionanalysis/`: the analogous signpost for the
  benchmark site (visionanalysis.org), which is a third, different repo.
