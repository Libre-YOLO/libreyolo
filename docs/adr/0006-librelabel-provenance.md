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
