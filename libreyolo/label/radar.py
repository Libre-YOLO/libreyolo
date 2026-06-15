"""Label-Error Radar for LibreLabel: audit already-accepted labels with the model.

The labels a human has *already accepted* are the most valuable signal in the
project -- and almost no tool re-examines them. Radar runs the user's own
detector over every image that already has box labels and surfaces the
*confident disagreements* between the model and the human:

* ``phantom``  -- a labelled box the model sees nothing in (a possible mislabel)
* ``miss``     -- an obvious object the model is sure of, with no box
* ``class``    -- a matched box whose class the model strongly disputes (a slip)

Trust contract (same as :mod:`libreyolo.label.assist`): Radar writes **nothing**.
It produces findings parked in memory; the human jumps to each image and fixes it
by hand with the normal tools. A machine disagreement is a *prompt to look*,
never an edit silently written to ``labels/*.txt``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional


def _xywh_to_xyxy(b):
    cx, cy, w, h = b
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b) -> float:
    """IoU of two normalised ``(cx, cy, w, h)`` boxes."""
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _ann_to_box(a) -> Optional[tuple]:
    """A box annotation -> ``(cls, cx, cy, w, h)``; a polygon -> its bbox."""
    if a.get("type") == "poly":
        pts = a.get("points") or []
        xs, ys = pts[0::2], pts[1::2]
        if not xs or not ys:
            return None
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return (int(a["cls"]), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    try:
        return (int(a["cls"]), float(a["cx"]), float(a["cy"]), float(a["w"]), float(a["h"]))
    except (KeyError, TypeError, ValueError):
        return None


def classify(labels, preds, *, iou_thr: float = 0.45, miss_conf: float = 0.55,
             phantom_weight: float = 0.5):
    """Greedy IoU-match ``labels`` <-> ``preds`` and return ``(findings, score)``.

    ``labels``: list of ``(cls, cx, cy, w, h)`` (the human's accepted boxes).
    ``preds``: list of suggestion dicts (``cls`` dataset idx or ``None``, ``cx``,
    ``cy``, ``w``, ``h``, ``conf``, ``name``, ``mapped``) from the detector.

    Only *mapped* predictions (a class the dataset actually has) participate, so
    a detector that knows 80 classes doesn't cry "miss" over a class the user
    never intends to label. Higher ``score`` == more/worse disagreement.
    """
    pairs = []
    for li, lb in enumerate(labels):
        for pi, pr in enumerate(preds):
            if not pr.get("mapped"):
                continue
            ov = iou(lb[1:5], (pr["cx"], pr["cy"], pr["w"], pr["h"]))
            if ov >= iou_thr:
                pairs.append((ov, li, pi))
    pairs.sort(key=lambda t: -t[0])

    lab_used: set = set()
    pred_used: set = set()
    findings: List[dict] = []
    score = 0.0
    for ov, li, pi in pairs:
        if li in lab_used or pi in pred_used:
            continue
        lab_used.add(li)
        pred_used.add(pi)
        lb, pr = labels[li], preds[pi]
        # Radar surfaces *confident* disagreements: only a class slip the model is
        # at least as sure about as a "miss" counts, so weak overlapping guesses
        # don't flood the deck with low-signal class findings.
        conf = float(pr.get("conf", 0.0))
        if int(pr["cls"]) != int(lb[0]) and conf >= miss_conf:
            findings.append({
                "type": "class", "iou": round(ov, 3),
                "label_cls": int(lb[0]), "pred_cls": int(pr["cls"]),
                "pred_name": pr.get("name"), "conf": conf,
                "box": [lb[1], lb[2], lb[3], lb[4]],
            })
            score += conf

    for li, lb in enumerate(labels):
        if li in lab_used:
            continue
        findings.append({"type": "phantom", "label_cls": int(lb[0]),
                         "box": [lb[1], lb[2], lb[3], lb[4]]})
        score += phantom_weight

    for pi, pr in enumerate(preds):
        if pi in pred_used or not pr.get("mapped"):
            continue
        c = float(pr.get("conf", 0.0))
        if c >= miss_conf:
            findings.append({"type": "miss", "pred_cls": int(pr["cls"]),
                             "pred_name": pr.get("name"), "conf": c,
                             "box": [pr["cx"], pr["cy"], pr["w"], pr["h"]]})
            score += c

    return findings, score


def scan_dataset(predict: Callable, session, *, model_name: Optional[str] = None,
                 conf: float = 0.25, iou_thr: float = 0.45, miss_conf: float = 0.55,
                 progress: Optional[Callable[[dict], None]] = None) -> dict:
    """Run the detector over every *labelled* image and collect disagreements.

    ``predict(image_path, names, model_name, conf) -> [suggestion]`` is the
    :class:`~libreyolo.label.assist.AssistEngine` predict callable (already
    name-mapped to the dataset). Returns ``{"deck", "findings", ...}`` worst
    image first. Images with no box/polygon labels are skipped -- Radar can only
    audit labels that exist. Writes nothing.
    """
    names = session.names
    total = len(session)
    deck: List[dict] = []
    findings_map: Dict[int, list] = {}
    scanned = flagged = 0
    deleted = getattr(session, "_deleted", set())
    for idx in range(total):
        if idx in deleted:
            continue   # quarantined/removed image -> nothing on disk to audit
        anns, _editable = session.read_label(idx)
        name = session.image_path(idx).name
        labels = [b for b in (_ann_to_box(a) for a in anns) if b is not None]
        if not labels:
            if progress:
                progress({"type": "progress", "i": idx + 1, "total": total,
                          "id": idx, "name": name, "issues": 0, "skipped": True})
            continue
        try:
            preds = predict(session.image_path(idx), names, model_name, conf)
        except RuntimeError:
            # Systemic: the model can't run at all (missing local weights, assist
            # disabled, opt-in not set). Abort so the caller surfaces the real error
            # instead of finishing with scanned==0 ("no labelled images to audit").
            raise
        except Exception as exc:  # noqa: BLE001 - per-image inference problem
            if progress:
                progress({"type": "progress", "i": idx + 1, "total": total,
                          "id": idx, "name": name, "issues": 0, "error": str(exc)})
            continue
        scanned += 1
        f, score = classify(labels, preds, iou_thr=iou_thr, miss_conf=miss_conf)
        if f:
            findings_map[idx] = f
            flagged += 1
            counts: Dict[str, int] = {}
            for it in f:
                counts[it["type"]] = counts.get(it["type"], 0) + 1
            deck.append({"id": idx, "name": name, "score": round(score, 3),
                         "issues": len(f), "counts": counts})
        if progress:
            progress({"type": "progress", "i": idx + 1, "total": total,
                      "id": idx, "name": name, "issues": len(f)})
    deck.sort(key=lambda d: -d["score"])
    return {"type": "done", "deck": deck, "scanned": scanned, "flagged": flagged,
            "total": total, "findings": findings_map}
