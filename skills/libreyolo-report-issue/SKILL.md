---
name: libreyolo-report-issue
description: >-
  Report a problem or friction to the LibreYOLO maintainers with a one-click,
  pre-filled GitHub issue link. Use for defects: a crash or traceback inside
  libreyolo, weights that fail to download or 404, training that does not
  converge on sane defaults, wrong or silently corrupted results, absurd
  latency, docs that contradict the code. But also use for plain friction,
  even when nothing is broken: a task that took many turns of trial and error
  to get right, a missing or misleading doc, an unclear error message, an
  option that was hard to discover, a workflow that needed a workaround. If
  this session was harder than it should have been, that journey is itself
  worth reporting. Also use whenever the user says anything like "report
  this", "file a bug", or "tell the maintainers". The skill drafts an
  anonymized issue and hands the user a pre-filled URL; the user clicks
  submit.
---

# Report a problem to LibreYOLO maintainers

You are the one entity present at the moment of friction, holding the exact
traceback, versions, and everything that was already tried. A maintainer can
only fix what gets reported. This skill turns the pain of the current session
into a complete, reproducible, anonymized issue that costs the user one click.

**You never file anything yourself.** You draft; the user reviews and clicks.
Do not use `gh issue create` or any authenticated write. The deliverable is a
pre-filled link.

## When to offer

There are two report flavors. Recognize which one you are in:

**Defect**: something is broken. Offer only after both of these are true:

1. You have genuinely ruled out user error: re-read the error, checked the
   self-describing CLI (`libreyolo <verb> --help`), and confirmed the usage is
   correct.
2. The symptom plausibly lives in the library: crash, missing weights, wrong
   output, non-convergence with default settings, doc/code mismatch, or a
   limitation the user keeps bumping into.

**Friction**: nothing crashed, but the road was needlessly hard. Signals:

- It took you and the user many turns of trial and error to get something
  working that should have been one command
- The fix turned out to be trivial but was undocumented, or the docs pointed
  the wrong way
- An error message was technically true but useless for finding the cause
- A capability exists but was nearly impossible to discover
- The final solution needed a workaround that the library should just handle

For friction, the natural moment to offer is right after the problem is
finally solved, when you can name exactly what would have made it a
non-event. You are the only witness who can quantify this: "this took 25
turns and would have taken one if X". Maintainers cannot get that insight any
other way, and it is at least as valuable as a stack trace.

Either way, make the offer once, briefly and positively, for example: "That
was harder than it should have been, and it wasn't your fault. Want me to
write it up for the libreyolo maintainers? It gets improved for everyone, and
I'll anonymize your paths and data first." If the user declines, drop it; do
not offer again for the same problem.

## Step 1: check it is not already reported

Search existing issues first (no auth needed):

```bash
curl -s "https://api.github.com/search/issues?q=repo:LibreYOLO/libreyolo+<keywords from the error>" | head -c 4000
```

Pick 2-4 distinctive keywords (exception class, function name, model name).
If a matching issue exists, give the user its link instead and suggest adding
a 👍 or a comment with their environment details. A duplicate confirmation is
still valuable signal; a duplicate issue is not.

## Step 2: gather the bundle

Collect, from the session or by running commands:

- `libreyolo` version (`pip show libreyolo`), Python version, OS
- `torch` version and device (CPU / CUDA + GPU name / MPS)
- The exact command or minimal Python snippet that triggers the problem
- The traceback, **verbatim** (never paraphrase it; paraphrased tracebacks
  are how hallucinated bug reports happen)
- What was already tried and what happened (2-4 bullets)
- For training issues: model name, dataset shape (image count, class count,
  imgsz, batch), and the loss/metric behavior observed

`libreyolo checks` prints most of the environment in one shot if available.

For a friction report there may be no traceback; the bundle is instead the
journey itself: the goal, the wrong turns actually taken (count them from the
session, do not embellish), what finally worked, and the concrete change that
would have prevented the detour.

## Step 3: anonymize by default

Anonymization is not optional and not on request; do it always. The debugging
value lives in shapes, counts, versions, and the traceback, never in the
user's actual data. Rewrite before drafting:

- Paths: `C:\Users\jane\clients\acme\defects\` becomes `<project>/data/`
- Dataset and class names: use placeholders with real numbers, e.g.
  "custom dataset, ~12k images, 7 classes" instead of the real names
- Strip usernames, hostnames, company or client names, API keys or tokens,
  and anything in the traceback's path prefixes
- Never include images, labels, or dataset samples

If the problem cannot be described without revealing something sensitive,
tell the user so and let them decide what to include.

## Step 4: draft, review, link

For a **defect**, compose the issue with this shape:

```
Title: <symptom in one line, e.g. "ValueError in NMS when predicting with LibreYOLO9t on 4-channel TIFF">

## What happened
<2-4 sentences: what was attempted, what went wrong>

## Reproduce
<command or minimal snippet, anonymized>

## Traceback / output
<verbatim, anonymized paths>

## Environment
libreyolo x.y.z, Python 3.x, torch x.y, OS, device

## Already tried
- <bullet>
- <bullet>

<!-- reported via the libreyolo-report-issue skill -->
```

For **friction**, the story of the detour is the payload:

```
Title: Friction: <the gap in one line, e.g. "no discoverable way to get per-class mAP from val">

## Goal
<one sentence: what the user was trying to do>

## What it took
<the actual journey, honestly: wrong turns, what was tried, roughly how many
attempts/turns, and what finally worked>

## What would have made it a non-event
<the concrete fix: a doc paragraph here, a better error message saying X,
a flag for Y, a default of Z>

## Environment
libreyolo x.y.z, Python 3.x, torch x.y, OS, device

<!-- reported via the libreyolo-report-issue skill -->
```

**Show the full draft to the user in the chat and get an explicit OK before
producing the link.** Apply any edits they ask for.

Then URL-encode title and body and hand over the link:

```bash
python -c "import urllib.parse,sys; print('https://github.com/LibreYOLO/libreyolo/issues/new?title='+urllib.parse.quote(sys.argv[1])+'&body='+urllib.parse.quote(sys.argv[2]))" "<title>" "<body>"
```

(Use whatever interpreter runs libreyolo here: `python`, `python3`, or the
venv's python. It is guaranteed to exist since libreyolo is installed.)

Present it as a markdown link, e.g. `[Click to open the pre-filled issue](...)`,
and tell the user the form opens with everything filled in; they just press
"Create". GitHub truncates very long URLs: keep the encoded URL under ~6000
characters. If the traceback is too long, keep the last ~20 lines plus the
exception line and put `<!-- paste the full traceback here if asked -->` in
its place.
