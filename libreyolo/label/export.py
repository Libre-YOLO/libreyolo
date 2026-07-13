"""Export a labelled LibreLabel dataset to YOLO / COCO / Pascal VOC.

Two modes:
- copy (default): leave the working folder untouched and write a self-contained
  dataset into a destination folder, optionally zipped. Images are copied once
  and shared across every requested format, so exporting to all three formats
  does not triplicate the image bytes.
- in place: re-organise the working folder into train/val(/test) subdirs and
  rewrite the opened dataset YAML (YOLO only; it is the live dataset).

An optional seeded train/val/test split is applied either way. Labels read from
the YOLO ``.txt`` files the rest of LibreLabel already wrote; nothing here parses
pixels except to read image dimensions for COCO/VOC.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import xml.sax.saxutils as _sx
import zipfile
from pathlib import Path
from typing import List, Optional

import yaml

from .dataset import IMG_EXTS, _atomic_write_text, _path_identity, _publish_no_clobber
from .labelio import (
    has_degenerate_polygon,
    has_out_of_bounds_coords,
    has_out_of_range_rows,
    has_unsupported_rows,
    has_zero_area_box,
    parse_annotations,
    sanitize_annotations,
)


_FORMATS = {"yolo", "coco", "voc"}
_TASK_FORMATS = {
    "detect": _FORMATS,
    "segment": {"yolo", "coco"},
    "obb": {"yolo"},
}


def _assign_splits(n: int, mode: str, val_frac: float, test_frac: float, seed: int) -> List[str]:
    """A per-item split label list ('train'/'val'/'test'), seeded and reproducible."""
    if mode not in {"none", "trainval", "trainvaltest"}:
        raise ValueError("split must be 'none', 'trainval', or 'trainvaltest'")
    try:
        vf = float(val_frac)
        tf = float(test_frac)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("split fractions must be finite numbers in [0, 1]") from exc
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (vf, tf)):
        raise ValueError("split fractions must be finite numbers in [0, 1]")
    if mode == "none" or n <= 1:
        return ["train"] * n
    import random
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    tf = tf if mode == "trainvaltest" else 0.0
    n_test = min(int(round(n * tf)), n - 1) if tf else 0
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


def _write_yaml(
    base: Path,
    splits,
    names,
    nc,
    task,
    *,
    dataset_root: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> str:
    cfg = {"path": (dataset_root or base).resolve().as_posix(), "train": "images/train"}
    if any(s == "val" for s in splits):
        cfg["val"] = "images/val"
    if any(s == "test" for s in splits):
        cfg["test"] = "images/test"
    cfg["names"] = list(names)
    cfg["nc"] = int(nc)
    t = str(task or "").strip().lower()
    if t:
        cfg["task"] = t
    out = Path(output_path) if output_path is not None else base / "data.yaml"
    text = ("# LibreLabel export -- YOLO dataset.\n"
            f"# Train with: libreyolo train data={out.name}\n\n"
            + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    _atomic_write_text(out, text)
    return str(out)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Restore an exact byte snapshot without platform newline translation."""
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = 0o644
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _coco_init(names):
    return {"images": [], "annotations": [],
            "categories": [{"id": i, "name": n} for i, n in enumerate(names)]}


def _coco_add(c, state, fname, W, H, anns, task="detect"):
    state["img"] += 1
    img_id = state["img"]
    c["images"].append({"id": img_id, "file_name": fname, "width": W, "height": H})
    for a in anns:
        state["ann"] += 1
        if a["type"] == "box":
            x1, y1, x2, y2 = _xyxy_px(a["cx"], a["cy"], a["w"], a["h"], W, H)
            # A five-field row in a segment dataset means a rectangular mask, not
            # merely a detection box. Preserve that mask in COCO instead of emitting
            # an empty segmentation payload.
            seg = [[x1, y1, x2, y1, x2, y2, x1, y2]] if task == "segment" else []
        else:
            pts = a.get("points") or []
            xs = [pts[i] * W for i in range(0, len(pts) - 1, 2)]
            ys = [pts[i] * H for i in range(1, len(pts), 2)]
            if not xs:
                continue
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            seg = [[v for xy in zip(xs, ys) for v in xy]]
        bw, bh = x2 - x1, y2 - y1
        if seg:
            ring = seg[0]
            area = abs(
                sum(
                    ring[i] * ring[(i + 3) % len(ring)]
                    - ring[(i + 2) % len(ring)] * ring[(i + 1) % len(ring)]
                    for i in range(0, len(ring), 2)
                )
            ) / 2.0
        else:
            area = bw * bh
        c["annotations"].append({
            "id": state["ann"], "image_id": img_id, "category_id": int(a["cls"]),
            "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
            "area": round(area, 2), "iscrowd": 0, "segmentation": seg})


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


def _zip_dir(src: Path, zip_path: str, *, archive_root: Optional[str] = None):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*"):
            if p.is_file():
                root = Path(archive_root or src.name)
                z.write(p, root / p.relative_to(src))


def _uniquifier():
    """Per-split unique *stems*, including collisions across image suffixes.

    ``a/foo.jpg`` and ``b/foo.png`` are distinct image filenames but both derive
    ``foo.txt`` (and ``foo.xml``), so reserving full names is insufficient.
    """
    used = {sp: set() for sp in ("train", "val", "test")}

    def unique(sp: str, name: str) -> str:
        stem, ext = Path(name).stem, Path(name).suffix
        cand, n = name, 1
        while Path(cand).stem.casefold() in used[sp]:
            n += 1
            cand = f"{stem}_{n}{ext}"
        used[sp].add(Path(cand).stem.casefold())
        return cand

    return unique


def _validate_request(task: str, formats, in_place: bool) -> set[str]:
    if isinstance(formats, (str, bytes)):
        formats = (formats,)
    fmts = {str(value).strip().lower() for value in formats if str(value).strip()} or {"yolo"}
    unknown = fmts - _FORMATS
    if unknown:
        raise ValueError(f"Unsupported export format(s): {', '.join(sorted(unknown))}")
    if in_place and fmts != {"yolo"}:
        raise ValueError("In-place export supports YOLO only; COCO/VOC require a copy export.")
    allowed = _TASK_FORMATS.get(task)
    if allowed is None:
        raise ValueError(
            f"LibreLabel cannot safely export task {task!r}: this exporter only preserves "
            "detection boxes, segmentation polygons, and OBB corners."
        )
    lossy = fmts - allowed
    if lossy:
        raise ValueError(
            f"Exporting task {task!r} as {', '.join(sorted(lossy))} would discard "
            "task-specific geometry; choose a lossless format."
        )
    return fmts


def _validated_label_text(path: Path, nc: int, task: str) -> tuple[bytes, List[dict]]:
    if not path.exists():
        return b"", []
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    invalid = (
        has_unsupported_rows(text)
        or has_out_of_range_rows(text, nc)
        or has_out_of_bounds_coords(text)
        or has_degenerate_polygon(text)
        or has_zero_area_box(text)
    )
    if invalid:
        raise ValueError(f"Cannot export invalid label file without data loss: {path}")
    annotations = parse_annotations(text)
    try:
        sanitize_annotations(annotations, nc, task=task)
    except ValueError as exc:
        raise ValueError(f"Cannot export invalid label file {path}: {exc}") from exc
    return raw, annotations


def _normpath(path: Path) -> str:
    return _path_identity(path)


def _flatten_split_sources(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_split_sources(item)
    elif value is not None:
        yield value


def _recursive_split_directories(session) -> List[Path]:
    """Return configured directory splits that recursively discover images."""
    base = Path(session.root or Path(session.yaml_file).parent)
    directories = []
    for value in getattr(session, "_split_sources", {}).values():
        for source in _flatten_split_sources(value):
            path = Path(str(source)).expanduser()
            if not path.is_absolute():
                path = base / path
            try:
                resolved = path.resolve(strict=False)
            except (OSError, RuntimeError):
                resolved = Path(os.path.abspath(str(path)))
            if resolved.is_dir():
                directories.append(resolved)
    return directories


def _reject_destination_inside_source(session, destination: Path) -> None:
    try:
        resolved = destination.expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        resolved = Path(os.path.abspath(str(destination)))
    for source_dir in _recursive_split_directories(session):
        try:
            resolved.relative_to(source_dir)
        except ValueError:
            continue
        raise ValueError(
            f"Export destination {resolved} is inside recursive source split "
            f"{source_dir}; choose a folder outside the source dataset."
        )


def _rollback_moves(moves: List[dict]) -> None:
    """Return every staged/finalized move to its original path in two phases."""
    for move in moves:
        if move["location"] == "dest" and move["dest"].exists():
            move["stage"].parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(move["dest"]), str(move["stage"]))
            move["location"] = "stage"
    for move in moves:
        if move["location"] == "stage" and move["stage"].exists():
            move["orig"].parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(move["stage"]), str(move["orig"]))
            move["location"] = "orig"


def _mkdir_parents_recorded(path: Path, created: List[Path]) -> None:
    """Create missing parents and record only directories made by this call."""
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        else:
            created.append(directory)


def _in_place_export(
    base: Path, items, splits, names, nc, task, *, yaml_path: Path, validator=None
) -> tuple[str, object | None]:
    """Reorganize through an external staging dir and roll back on any failure."""
    unique = _uniquifier()
    plans = []
    for (ip, lp), split_name in zip(items, splits, strict=True):
        ip, lp = Path(ip), Path(lp)
        if not ip.is_file():
            raise FileNotFoundError(f"Image disappeared before export: {ip}")
        dest_name = unique(split_name, ip.name)
        plans.append(
            (
                ip,
                base / "images" / split_name / dest_name,
                lp if lp.exists() else None,
                base / "labels" / split_name / f"{Path(dest_name).stem}.txt",
            )
        )

    source_paths = {
        _normpath(path)
        for ip, _di, lp, _dl in plans
        for path in (ip, lp)
        if path is not None
    }
    # The regenerated YAML recursively scans these canonical split directories.
    # Any supported image or label not owned by this transaction would silently
    # join the exported dataset, even when it does not collide with a planned name.
    for split_name in ("train", "val", "test"):
        image_dir = base / "images" / split_name
        if image_dir.is_dir():
            for existing in image_dir.rglob("*"):
                if (
                    existing.is_file()
                    and existing.suffix.lower() in IMG_EXTS
                    and _normpath(existing) not in source_paths
                ):
                    raise FileExistsError(
                        "In-place export target contains an unrelated image: "
                        f"{existing}"
                    )
        label_dir = base / "labels" / split_name
        if label_dir.is_dir():
            for existing in label_dir.rglob("*.txt"):
                if existing.is_file() and _normpath(existing) not in source_paths:
                    raise FileExistsError(
                        "In-place export target contains an unrelated label: "
                        f"{existing}"
                    )
    for _ip, dest_img, lp, dest_lbl in plans:
        for dest in (dest_img, dest_lbl if lp is not None else None):
            if dest is not None and dest.exists() and _normpath(dest) not in source_paths:
                raise FileExistsError(f"In-place export destination already exists: {dest}")
        if dest_img.parent.is_dir():
            for existing in dest_img.parent.iterdir():
                if (
                    existing.is_file()
                    and existing.suffix.lower() in IMG_EXTS
                    and existing.stem.casefold() == dest_img.stem.casefold()
                    and _normpath(existing) not in source_paths
                ):
                    raise FileExistsError(
                        "In-place export image stem collides with an existing file: "
                        f"{existing}"
                    )
        if lp is not None and dest_lbl.parent.is_dir():
            for existing in dest_lbl.parent.iterdir():
                if (
                    existing.is_file()
                    and existing.suffix.lower() == ".txt"
                    and existing.stem.casefold() == dest_lbl.stem.casefold()
                    and _normpath(existing) not in source_paths
                ):
                    raise FileExistsError(
                        "In-place export label stem collides with an existing file: "
                        f"{existing}"
                    )

    stage = Path(tempfile.mkdtemp(prefix=f".{base.name}-librelabel-export-", dir=str(base.parent)))
    yaml_path = Path(yaml_path).resolve()
    original_yaml = yaml_path.read_bytes() if yaml_path.exists() else None
    moves: List[dict] = []
    created_dirs: List[Path] = []
    committed = False
    try:
        for index, (ip, dest_img, lp, dest_lbl) in enumerate(plans):
            pairs = [(ip, dest_img)]
            if lp is not None:
                pairs.append((lp, dest_lbl))
            for subindex, (orig, dest) in enumerate(pairs):
                staged = stage / f"{index}-{subindex}{orig.suffix}"
                shutil.move(str(orig), str(staged))
                moves.append(
                    {"orig": orig, "stage": staged, "dest": dest, "location": "stage"}
                )
        for move in moves:
            _mkdir_parents_recorded(move["dest"].parent, created_dirs)
            shutil.move(str(move["stage"]), str(move["dest"]))
            move["location"] = "dest"
        written_yaml = _write_yaml(
            base, splits, names, nc, task, output_path=yaml_path
        )
        reopened = validator(written_yaml) if validator is not None else None
        committed = True
        return written_yaml, reopened
    except BaseException:
        if any(move["location"] != "orig" for move in moves):
            _rollback_moves(moves)
        if original_yaml is None:
            try:
                yaml_path.unlink()
            except FileNotFoundError:
                pass
        else:
            _atomic_write_bytes(yaml_path, original_yaml)
        raise
    finally:
        if not committed:
            for directory in reversed(created_dirs):
                try:
                    directory.rmdir()
                except OSError:
                    pass
        shutil.rmtree(stage, ignore_errors=True)


def export_dataset(session, *, dst: Optional[str] = None, formats=("yolo",),
                   split: str = "trainval", val_frac: float = 0.2,
                   test_frac: float = 0.0, seed: int = 1234,
                   in_place: bool = False, make_zip: bool = False,
                   _in_place_validator=None) -> dict:
    """Export the open ``session``. Returns a dict describing what was written."""
    if getattr(session, "_label_clash", False):
        raise ValueError(
            "Two source images share one derived label file (for example foo.jpg and "
            "foo.png). Rename one image before exporting so neither image or label is lost."
        )
    items = [(ip, lp) for i, (ip, lp, _s) in enumerate(session._items)
             if i not in getattr(session, "_deleted", set())]
    if not items:
        raise ValueError("No images to export.")
    names = list(session.names or [])
    nc = int(session.nc or len(names))
    task = getattr(session, "_task", None) or "detect"
    task_resolved = bool(
        getattr(
            session,
            "_task_declared_or_inferred",
            bool(getattr(session, "_task", None)),
        )
    )
    if getattr(session, "_lossy_export", False):
        raise ValueError(
            "This project uses pose, mask, depth, classification, or other task-specific "
            "labels that LibreLabel's box/polygon exporter cannot preserve."
        )
    fmts = _validate_request(task, formats, in_place)
    if not task_resolved and fmts != {"yolo"}:
        raise ValueError(
            "This dataset does not declare a task and its labels do not resolve "
            "one unambiguously. Only a lossless YOLO copy or in-place reorganization "
            "is safe until task: detect, segment, or obb is declared; converting it "
            "could omit default task-specific targets."
        )
    yaml_task = task if task_resolved else ""
    if in_place and make_zip:
        raise ValueError("Create a copy export before making a zip; in-place export cannot be zipped safely.")
    splits = _assign_splits(len(items), split, val_frac, test_frac, seed)
    payloads = []
    for ip, lp in items:
        if not Path(ip).is_file():
            raise FileNotFoundError(f"Image disappeared before export: {ip}")
        validation_task = task if task_resolved else ""
        payloads.append(_validated_label_text(Path(lp), nc, validation_task))

    if in_place and getattr(session, "linked", False):
        raise ValueError("Linked projects never modify the source folder - "
                         "use a copy export instead of re-splitting in place.")
    if in_place:
        base = Path(session.root or Path(session.yaml_file).parent).resolve()
        yaml_path, reopened = _in_place_export(
            base,
            items,
            splits,
            names,
            nc,
            yaml_task,
            yaml_path=Path(session.yaml_file),
            validator=_in_place_validator,
        )
        result = {"in_place": True, "out": str(base), "yaml": yaml_path,
                  "counts": {s: splits.count(s) for s in ("train", "val", "test")}}
        if reopened is not None:
            result["_reopened_session"] = reopened
        return result

    if not dst:
        raise ValueError("A destination folder is required for a copy export.")
    dstp = Path(dst)
    _reject_destination_inside_source(session, dstp)
    if dstp.exists() and not dstp.is_dir():
        raise ValueError("The export destination exists and is not a folder.")
    if dstp.is_dir() and any(dstp.iterdir()):
        # A previous export's leftovers would be rescanned by the generated
        # data.yaml (stale images/labels silently joining the new dataset).
        raise ValueError("The destination folder is not empty - pick a new or empty "
                         "folder so stale files can't mix into the export.")
    dstp.parent.mkdir(parents=True, exist_ok=True)
    had_empty_destination = dstp.is_dir()
    zip_target = Path(str(dstp.absolute()) + ".zip") if make_zip else None
    if zip_target is not None and zip_target.exists():
        raise ValueError(f"The export zip already exists: {zip_target}")
    stage = Path(
        tempfile.mkdtemp(prefix=f".{dstp.name or 'export'}-librelabel-", dir=str(dstp.parent))
    )
    zip_stage: Optional[Path] = None
    need_dims = bool(fmts & {"coco", "voc"})
    coco = {sp: _coco_init(names) for sp in ("train", "val", "test")}
    cstate = {sp: {"img": 0, "ann": 0} for sp in ("train", "val", "test")}

    committed = False
    try:
        unique = _uniquifier()
        for (ip, _lp), sp, (raw, anns) in zip(items, splits, payloads, strict=True):
            ip = Path(ip)
            img_out = stage / "images" / sp / unique(sp, ip.name)
            img_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(ip), str(img_out))
            if "yolo" in fmts:
                lo = stage / "labels" / sp / (img_out.stem + ".txt")
                lo.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_bytes(lo, raw)
            if need_dims:
                W, H = _img_size(ip)
                if W <= 0 or H <= 0:
                    raise ValueError(f"Cannot read image dimensions for export: {ip}")
                if "coco" in fmts:
                    _coco_add(coco[sp], cstate[sp], img_out.name, W, H, anns, task)
                if "voc" in fmts:
                    _voc_write(stage / "voc" / sp / "Annotations", img_out.name, W, H, anns, names)

        if "yolo" in fmts:
            _write_yaml(
                stage, splits, names, nc, yaml_task, dataset_root=dstp.absolute()
            )
        if "coco" in fmts:
            ann_dir = stage / "annotations"
            ann_dir.mkdir(exist_ok=True)
            for sp, c in coco.items():
                if c["images"]:
                    _atomic_write_text(ann_dir / f"instances_{sp}.json", json.dumps(c))
        if make_zip:
            fd, zip_name = tempfile.mkstemp(
                prefix=f".{dstp.name or 'export'}-", suffix=".zip.tmp", dir=str(dstp.parent)
            )
            os.close(fd)
            zip_stage = Path(zip_name)
            _zip_dir(stage, str(zip_stage), archive_root=dstp.name)
            os.chmod(zip_stage, 0o644)

        if had_empty_destination:
            dstp.rmdir()
        os.replace(stage, dstp)
        committed = True
        if zip_stage is not None and zip_target is not None:
            # Publish without replacing a zip another exporter created after the
            # preflight check, without requiring hard-link filesystem support.
            _publish_no_clobber(zip_stage, zip_target)
            zip_stage = None
    except BaseException:
        if committed and dstp.exists():
            os.replace(dstp, stage)
            committed = False
        if had_empty_destination and not dstp.exists():
            dstp.mkdir(parents=True, exist_ok=True)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if zip_stage is not None:
            try:
                zip_stage.unlink()
            except OSError:
                pass

    res = {"in_place": False, "out": str(dstp),
           "counts": {s: splits.count(s) for s in ("train", "val", "test")},
           "formats": sorted(fmts)}
    if zip_target is not None:
        res["zip"] = str(zip_target)
    return res
