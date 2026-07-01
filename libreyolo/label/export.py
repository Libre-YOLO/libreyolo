"""Export a labelled LibreLabel dataset to YOLO / COCO / Pascal VOC.

Two modes:
- copy (default): leave the working folder untouched and write a self-contained
  dataset into a destination folder, optionally zipped. Images are copied once
  and shared across every requested format, so exporting to all three formats
  does not triplicate the image bytes.
- in place: re-organise the working folder into train/val(/test) subdirs and
  rewrite ``data.yaml`` (YOLO only; it is the live dataset).

An optional seeded train/val/test split is applied either way. Labels read from
the YOLO ``.txt`` files the rest of LibreLabel already wrote; nothing here parses
pixels except to read image dimensions for COCO/VOC.
"""

from __future__ import annotations

import json
import shutil
import xml.sax.saxutils as _sx
import zipfile
from pathlib import Path
from typing import List, Optional

import yaml

from .dataset import _atomic_write_text
from .labelio import parse_annotations


def _assign_splits(n: int, mode: str, val_frac: float, test_frac: float, seed: int) -> List[str]:
    """A per-item split label list ('train'/'val'/'test'), seeded and reproducible."""
    if mode == "none" or n <= 1:
        return ["train"] * n
    import random
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    vf = max(0.0, float(val_frac))
    tf = max(0.0, float(test_frac)) if mode == "trainvaltest" else 0.0
    n_test = int(round(n * tf)) if tf else 0
    n_val = int(round(n * vf)) if vf else 0
    n_val = min(n_val, max(0, n - 1 - n_test))   # always keep >= 1 train image
    out = ["train"] * n
    for j in idx[:n_test]:
        out[j] = "test"
    for j in idx[n_test:n_test + n_val]:
        out[j] = "val"
    return out


def _xyxy_px(cx, cy, w, h, W, H):
    return (max(0.0, (cx - w / 2) * W), max(0.0, (cy - h / 2) * H),
            min(float(W), (cx + w / 2) * W), min(float(H), (cy + h / 2) * H))


def _img_size(p: Path):
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size                      # (W, H)
    except Exception:  # noqa: BLE001 - unreadable image -> skip dims-dependent formats
        return (0, 0)


def _write_yaml(base: Path, splits, names, nc, task) -> str:
    cfg = {"path": base.resolve().as_posix(), "train": "images/train"}
    if any(s == "val" for s in splits):
        cfg["val"] = "images/val"
    if any(s == "test" for s in splits):
        cfg["test"] = "images/test"
    cfg["names"] = list(names)
    cfg["nc"] = int(nc)
    t = str(task or "").strip().lower()
    if t and t != "detect":
        cfg["task"] = t
    text = ("# LibreLabel export -- YOLO dataset.\n"
            "# Train with: libreyolo train data=data.yaml\n\n"
            + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    out = base / "data.yaml"
    _atomic_write_text(out, text)
    return str(out)


def _coco_init(names):
    return {"images": [], "annotations": [],
            "categories": [{"id": i, "name": n} for i, n in enumerate(names)]}


def _coco_add(c, state, fname, W, H, anns):
    state["img"] += 1
    img_id = state["img"]
    c["images"].append({"id": img_id, "file_name": fname, "width": W, "height": H})
    for a in anns:
        state["ann"] += 1
        if a["type"] == "box":
            x1, y1, x2, y2 = _xyxy_px(a["cx"], a["cy"], a["w"], a["h"], W, H)
            seg = []
        else:
            pts = a.get("points") or []
            xs = [pts[i] * W for i in range(0, len(pts) - 1, 2)]
            ys = [pts[i] * H for i in range(1, len(pts), 2)]
            if not xs:
                continue
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            seg = [[v for xy in zip(xs, ys) for v in xy]]
        bw, bh = x2 - x1, y2 - y1
        c["annotations"].append({
            "id": state["ann"], "image_id": img_id, "category_id": int(a["cls"]),
            "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
            "area": round(bw * bh, 2), "iscrowd": 0, "segmentation": seg})


def _voc_write(out_dir: Path, fname, W, H, anns, names):
    objs = []
    for a in anns:
        if a["type"] == "box":
            x1, y1, x2, y2 = _xyxy_px(a["cx"], a["cy"], a["w"], a["h"], W, H)
        else:
            pts = a.get("points") or []
            xs = [pts[i] * W for i in range(0, len(pts) - 1, 2)]
            ys = [pts[i] * H for i in range(1, len(pts), 2)]
            if not xs:
                continue
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        cls = names[a["cls"]] if 0 <= a["cls"] < len(names) else str(a["cls"])
        objs.append(
            "  <object><name>%s</name><pose>Unspecified</pose><truncated>0</truncated>"
            "<difficult>0</difficult><bndbox><xmin>%d</xmin><ymin>%d</ymin>"
            "<xmax>%d</xmax><ymax>%d</ymax></bndbox></object>" % (
                _sx.escape(cls), int(round(x1)), int(round(y1)),
                int(round(x2)), int(round(y2))))
    xml = ('<annotation><folder>images</folder><filename>%s</filename>'
           '<size><width>%d</width><height>%d</height><depth>3</depth></size>\n%s\n</annotation>'
           % (_sx.escape(fname), W, H, "\n".join(objs)))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / (Path(fname).stem + ".xml")).write_text(xml, encoding="utf-8")


def _zip_dir(src: Path, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(src.parent))


def _uniquifier():
    """Per-split unique basenames: nested source dirs can hold same-named images
    (``a/0001.jpg`` + ``b/0001.jpg``), and flattening into ``images/<split>/``
    would otherwise silently overwrite one with the other."""
    used = {sp: set() for sp in ("train", "val", "test")}

    def unique(sp: str, name: str) -> str:
        stem, ext = Path(name).stem, Path(name).suffix
        cand, n = name, 1
        while cand.lower() in used[sp]:
            n += 1
            cand = f"{stem}_{n}{ext}"
        used[sp].add(cand.lower())
        return cand

    return unique


def export_dataset(session, *, dst: Optional[str] = None, formats=("yolo",),
                   split: str = "trainval", val_frac: float = 0.2,
                   test_frac: float = 0.0, seed: int = 1234,
                   in_place: bool = False, make_zip: bool = False) -> dict:
    """Export the open ``session``. Returns a dict describing what was written."""
    items = [(ip, lp) for i, (ip, lp, _s) in enumerate(session._items)
             if i not in getattr(session, "_deleted", set())]
    if not items:
        raise ValueError("No images to export.")
    names = list(session.names or [])
    nc = int(session.nc or len(names))
    task = getattr(session, "_task", None) or "detect"
    splits = _assign_splits(len(items), split, val_frac, test_frac, seed)

    if in_place:
        base = Path(session.root or Path(session.yaml_file).parent).resolve()
        unique = _uniquifier()
        for (ip, lp), sp in zip(items, splits):
            ip = Path(ip)
            if not ip.exists():
                continue
            dest_img = base / "images" / sp / ip.name
            if ip.resolve() == dest_img.resolve():
                unique(sp, ip.name)          # already in place: just reserve its name
                dest_name = ip.name
            else:
                dest_name = unique(sp, ip.name)
                dest_img = base / "images" / sp / dest_name
                dest_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ip), str(dest_img))
            lp = Path(lp)
            if lp.exists():
                dest_lbl = base / "labels" / sp / (Path(dest_name).stem + ".txt")
                dest_lbl.parent.mkdir(parents=True, exist_ok=True)
                if lp.resolve() != dest_lbl.resolve():
                    shutil.move(str(lp), str(dest_lbl))
        yaml_path = _write_yaml(base, splits, names, nc, task)
        return {"in_place": True, "out": str(base), "yaml": yaml_path,
                "counts": {s: splits.count(s) for s in ("train", "val", "test")}}

    if not dst:
        raise ValueError("A destination folder is required for a copy export.")
    dstp = Path(dst)
    if dstp.is_dir() and any(dstp.iterdir()):
        # A previous export's leftovers would be rescanned by the generated
        # data.yaml (stale images/labels silently joining the new dataset).
        raise ValueError("The destination folder is not empty - pick a new or empty "
                         "folder so stale files can't mix into the export.")
    dstp.mkdir(parents=True, exist_ok=True)
    fmts = {f.lower() for f in formats} or {"yolo"}
    need_dims = bool(fmts & {"coco", "voc"})
    coco = {sp: _coco_init(names) for sp in ("train", "val", "test")}
    cstate = {sp: {"img": 0, "ann": 0} for sp in ("train", "val", "test")}

    unique = _uniquifier()
    for (ip, lp), sp in zip(items, splits):
        ip = Path(ip)
        if not ip.exists():
            continue
        img_out = dstp / "images" / sp / unique(sp, ip.name)
        img_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(ip), str(img_out))
        txt = Path(lp).read_text(encoding="utf-8") if Path(lp).exists() else ""
        anns = parse_annotations(txt)
        if "yolo" in fmts:
            lo = dstp / "labels" / sp / (img_out.stem + ".txt")
            lo.parent.mkdir(parents=True, exist_ok=True)
            lo.write_text(txt, encoding="utf-8")
        if need_dims:
            W, H = _img_size(ip)
            if "coco" in fmts:
                _coco_add(coco[sp], cstate[sp], img_out.name, W, H, anns)
            if "voc" in fmts:
                _voc_write(dstp / "voc" / sp / "Annotations", img_out.name, W, H, anns, names)

    if "yolo" in fmts:
        _write_yaml(dstp, splits, names, nc, task)
    if "coco" in fmts:
        ann_dir = dstp / "annotations"
        ann_dir.mkdir(exist_ok=True)
        for sp, c in coco.items():
            if c["images"]:
                _atomic_write_text(ann_dir / f"instances_{sp}.json", json.dumps(c))

    res = {"in_place": False, "out": str(dstp),
           "counts": {s: splits.count(s) for s in ("train", "val", "test")},
           "formats": sorted(fmts)}
    if make_zip:
        zp = str(dstp.resolve()) + ".zip"
        _zip_dir(dstp, zp)
        res["zip"] = zp
    return res
