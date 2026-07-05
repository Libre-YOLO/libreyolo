---
name: libreyolo-license-audit
description: >-
  Audit and maintain LibreYOLO's licensing and provenance surfaces, and handle
  contamination correctly. Use when adding ported code or weights (which
  notice files must change), when reviewing a PR for license risk, when
  someone asks "can we use/host/ship X?", when a family's provenance is
  questioned, or when GPL/AGPL/NC-licensed material may have touched the
  work. Covers the four notice surfaces and when each changes, the
  code-vs-weights license distinction, the decision table for common
  licenses, the contamination protocol, and the hard rules: no clean-room
  laundering, surface decisions to the maintainer instead of quietly
  "fixing" them.
---

# LibreYOLO license and provenance audit

LibreYOLO's wedge is being genuinely MIT. That only holds if every ported
file and every hosted weight has clean, documented provenance. This skill is
the audit discipline; the policy source of truth is the "Licensing policy"
section of `AGENTS.md`.

## The hard rules (non-negotiable)

1. **Only MIT / Apache-2.0 (and compatibly-licensed) upstreams** may be
   copied, adapted, paraphrased, or derived from. GPL, AGPL, LGPL,
   proprietary, unknown, or no-license upstreams may not, in any form.
2. **No clean-room laundering.** Never rewrite, rename, or restructure
   incompatibly-licensed code to obscure where it came from. A GPL function
   with new variable names is still a derivative. If provenance is tainted,
   the options are: re-derive from a genuinely clean source (paper, spec,
   compatibly-licensed implementation) with the new provenance documented,
   or drop the feature. Which option applies is the **maintainer's
   decision**; surface it, do not pick silently.
3. **Contamination protocol.** If you may have been exposed to or influenced
   by incompatible code while working (read an AGPL repo "for reference",
   pasted a snippet of unknown origin), stop, flag it explicitly to the
   maintainer, and do not contribute the affected code. Flagging is never
   the wrong move; quiet contribution is.
4. **License compatibility is non-negotiable** (REVIEW.md axiom). In PR
   review, a licensing doubt is a blocking finding, not a nit.
5. Keep third-party CV library names out of committed text (code, docs, PR
   titles/descriptions) unless a comparison is technically necessary.

## Code license vs weights license (audit them separately)

They routinely differ and each can independently block:

- **Code**: the architecture/implementation you port. Must be MIT/Apache-2.0
  (or public domain, e.g. the Darknet lineage).
- **Weights**: learned parameters. Carry their own license, often inherited
  from *training data terms* (Gaze360 makes L2CS weights non-redistributable;
  some upstream checkpoints are CC-BY-NC even where the code is Apache-2.0;
  VisDrone training data makes a fine-tune CC-BY-NC-SA). Apache code +
  NC weights means: ship the code, do not host the weights, or host behind
  an explicit variant notice (the `-visdrone` preview pattern).
- **Datasets**: gate before hosting or wiring auto-download; that gate lives
  in `skills/libreyolo-upload-hf-dataset`.

Quick decision table (source license, what it means here):

| Upstream license | Port code? | Host weights? |
|---|---|---|
| MIT / Apache-2.0 / BSD / public domain | yes, with attribution | yes, with LICENSE+NOTICE |
| CC-BY (weights/data) | n/a | yes, attribute |
| CC-BY-NC / research-only | **no** | **no**; surface to maintainer |
| GPL / AGPL / LGPL | **no** | **no** (weights from AGPL *code* are a maintainer call; ask) |
| Custom (DINOv3-style, Deci, model-specific) | maintainer call | maintainer call; document verbatim |
| Unknown / no license | treat as all-rights-reserved: **no** | **no** |

## The four notice surfaces (what changes when)

| Surface | Covers | Changes when |
|---|---|---|
| `NOTICE` (root) | Bundled third-party **source code under non-MIT licenses** kept verbatim in-tree (e.g. the DINOv3 backbone under its custom license) | You vendor non-MIT-licensed files |
| `THIRD_PARTY_NOTICES.txt` | Every project LibreYOLO **derives or ports code from**, with license text, copyright, and the LibreYOLO module that uses it | Any new port, adaptation, or derived module |
| `weights/LICENSE_NOTICE.txt` | Per-family summary of **weight** licenses and upstreams (no weights ship in-tree; this documents the HF-hosted ones) | New family weights, new variant, license change |
| `libreyolo/models/<family>/NOTICE` | Per-family attribution shipped next to the ported code (timm-derived classifiers, nafnet, pidnet, eomt, mobilesam, clip have them) | New ported family; follow the siblings' format |

Plus per-HF-repo `LICENSE` + `NOTICE` files, owned by
`skills/libreyolo-upload-hf-model` / `-dataset`.

**The audit invariant: a new ported family with no notices diff is a red
flag.** The release process (Gate F in `skills/libreyolo-release`) checks
exactly this; catch it at PR time instead. Conversely, never patch a missing
notice silently during a release: surface it, because the missing notice may
mean the provenance was never checked at all.

## Auditing a family's provenance (the checklist)

For "is family X clean?" or PR review of a new port:

1. **Identify the true upstream.** The repo the code was actually derived
   from, not the paper's official repo if those differ. Check the family's
   NOTICE, `THIRD_PARTY_NOTICES.txt` entry, and the conversion script header
   (`weights/convert_<family>_weights.py`).
2. **Read that upstream's LICENSE at the pinned commit.** Licenses change
   over a repo's history; the pin matters. Watch for repos that are
   "Apache-2.0" at root with GPL files vendored inside (it happens; the
   RTMDet assigner incident started exactly there).
3. **Check sub-components separately.** Backbones, assigners, tokenizers,
   loss functions often have their own upstreams (a Swin backbone inside a
   detector, a BERT tower inside an open-vocab model). Each needs its own
   provenance line.
4. **Check the weights' terms**, including training-data conditions
   (Gaze360, ImageNet terms, NC dataset fine-tunes).
5. **Confirm all four surfaces** (table above) carry matching entries.
6. **Verify attribution claims are true**: "state-dict key remapping only,
   learned parameters unchanged" must match what the converter actually
   does.

Past incidents worth knowing (they shaped these rules): a GPL-derived
assigner was replaced by a re-port from an Apache upstream with notices
fixed; a head with unclear provenance was re-derived from an MIT source with
bitwise parity as proof; NC-licensed depth weights were flagged for hosting
against policy and escalated rather than hidden. The pattern in all three:
detect, surface, re-derive from clean source or escalate. Never quietly
rename.

## When asked "can we add model X?"

Answer with the split verdict: code license (portable or not), weights
license (hostable or not), dataset story (trainable/finetunable or not), and
any custom-license sub-components. "Yes to code, no to hosting b/l weights,
s is Apache" is a normal, useful answer. Put the verdict in the issue before
any porting starts; it is the cheapest point to stop.

## Related

- `AGENTS.md` "Licensing policy": the policy this skill operationalizes.
- `skills/libreyolo-port-model/` section 1: the pre-port license check.
- `skills/libreyolo-upload-hf-model/`, `skills/libreyolo-upload-hf-dataset/`:
  per-repo LICENSE/NOTICE contracts and the dataset redistribution gate.
- `skills/libreyolo-release/` Gate F: the release-time notices sync check.
