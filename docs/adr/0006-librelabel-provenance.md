# 0006 — LibreLabel provenance & clean-room policy

Status: accepted
Date: 2026-06-15

## Context

LibreLabel (`libreyolo label`) is a browser-based bounding-box annotator shipped
inside the MIT-licensed `libreyolo` package. The annotation-tool field is crowded
with GPL/AGPL projects (labelme, AnyLabeling, X-AnyLabeling, makesense.ai) whose
source must never contaminate an MIT codebase. This ADR records how LibreLabel was
built and what it does (and does not) derive from, so the provenance is auditable.

## Decision

LibreLabel is a **clean-room, original implementation**. It copies, adapts, and
links **no** code from any third-party annotation tool — neither GPL/AGPL ones nor
permissively-licensed ones (CVAT, Label Studio, labelImg).

Concretely:

1. **Format source of truth is LibreYOLO's own code.** The image↔label mapping and
   `data.yaml` resolution come exclusively from `libreyolo/data/` (`img2label_paths`,
   `load_data_config`); parse/serialize lives in our own `libreyolo/label/labelio.py`.
   No format detail was taken from an external annotator.

2. **The server pattern mirrors LibreYOLO's own `libreyolo/ui` module** (in-house,
   MIT): a stdlib `http.server.ThreadingHTTPServer` serving one embedded HTML page.

3. **No third-party labelling code and no vendored JavaScript.** The canvas is
   hand-written vanilla Canvas 2D — no Konva/Fabric or any JS library was used. The
   inline SVG icons are simple geometric paths authored by hand. Result: **zero new
   runtime dependencies** beyond what `libreyolo` already ships.

4. **GPL/AGPL projects were studied by documentation only**, never by reading or
   reproducing their source. Industry-standard interaction idioms that LibreLabel
   shares with the field — drag-to-draw boxes, number-key class assignment, dashed
   "ghost" model suggestions with accept/reject review, a command-palette class
   search, a dataset-health distribution panel — are unprotectable conventions, not
   copyrightable expression.

5. **AI auto-label rides LibreYOLO's own predict path** (`AssistEngine` reuses the
   `ui` server's lazy-model pattern over `LibreYOLO(weight).predict`). Default
   weights are the user's own YOLO9 (MIT) / RF-DETR (Apache-2.0) checkpoints; nothing
   is downloaded by default. No external/cloud model service is involved.

## Consequences

- **`THIRD_PARTY_NOTICES` needs no new entry** for LibreLabel: it carries no
  third-party code. (Were a permissive JS library ever vendored, e.g. Konva (MIT),
  it would be added there; none is today.)
- The GPL "derivative work / based on the Program" clause never attaches, because
  nothing GPL/AGPL is copied, linked, or adapted.
- **Stop-rule:** if GPL/AGPL/proprietary source is ever pasted in, stop, mark
  `CONTAMINATION RISK:` with the source, and re-implement the affected component from
  the specification by someone who has not seen that source.

## Addendum — data-quality + AI superpowers (2026-06-15)

The following features were added; all are **clean-room original**, built from a
first-principles probe of LibreYOLO's *own* internals (model forward/train APIs),
not from any third-party annotation or active-learning tool:

- **Label-Error Radar** (`radar.py`) — audits already-accepted labels by running the
  user's own detector (via `AssistEngine`) and surfacing disagreements. The
  match is textbook greedy IoU; the phantom/miss/class-slip taxonomy is our own. It
  **writes nothing** — findings are parked in memory; the human fixes by hand.
- **Geometry linter** (`quality.py`) — pure-Python thresholds (tiny-at-imgsz, sliver,
  full-frame) over the normalized label contract. No external code.
- **Leakage/duplicate fixer** (`DatasetSession.resolve_duplicates`) — reuses the dHash
  already in `insights()`; collapses a group to one survivor and **moves** the rest to
  a reversible `.librelabel_quarantine/` (never a blind delete). Refuses `.txt`-manifest
  splits. Original design.
- **Magnetic edges + Tighten** and **Loupe** — entirely client-side, hand-written
  Canvas 2D: a Sobel-style luminance-gradient edge snap and a pixel magnifier. No JS
  library (no OpenCV.js, no Konva), no vendored code; `getImageData` runs on the
  same-origin image only.
- **Boost** (`boost.py`) — a frozen-backbone, head-only fine-tune that rides LibreYOLO's
  **own public `model.train(freeze=[...])`** API (in-house, MIT). It snapshots accepted
  labels into a throwaway temp dataset and trains into a temp dir — the source labels
  are never opened for writing. No third-party training/active-learning code.
- **Embedding map** (`embed.py`) — global-average-pools LibreYOLO9's own neck features
  (`x8/x16/x32`) and projects to 2-D with a **NumPy-only PCA** (`np.linalg.svd`). No
  scikit-learn, no UMAP/t-SNE library, no new dependency.

- **Project home + registry** (`projects.py`, `server.py` runtime session-switching) — a
  home screen listing datasets you've opened, stored in `~/.librelabel/projects.json`
  (paths + cached counts only; never copies/moves/deletes a dataset). Original; stdlib
  JSON, a `threading.Lock` + unique temp file for atomic writes, no database.
- **Teammate share** (`--share` / `_lan_ip` / `GET /api/server`) — binds the *existing*
  stdlib server on the LAN so teammates label the same dataset, surfaced as a copyable
  URL. A per-project `epoch` guard rejects a stale save so it can never land in the wrong
  dataset. No account, no cloud, no new dependency.
- **Carry-forward** (copy the previous image's labels) and **the boosted model as a
  selectable option** are thin reuses of the existing human-save path and the assist
  model cache respectively — no new third-party logic.

Net: still **zero new runtime dependencies** (numpy/torch/opencv/PIL already ship with
`libreyolo`), and still no third-party labelling, active-learning, embedding, project-
management, or collaboration code — all original, built on LibreYOLO's own surfaces.
