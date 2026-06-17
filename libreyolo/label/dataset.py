"""Dataset session for LibreLabel: enumerate images and round-trip YOLO labels.

Thin wrapper over LibreYOLO's own ``load_data_config`` / ``img2label_paths`` so
that the image<->label mapping and ``data.yaml`` resolution are *identical* to
what training uses -- LibreLabel writes labels exactly where the trainer reads
them. No database; the filesystem dataset is the store.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from libreyolo.data.utils import get_img_files, img2label_paths, load_data_config

from .labelio import (
    format_annotations,
    has_degenerate_polygon,
    has_obb_shaped_rows,
    has_out_of_bounds_coords,
    has_out_of_range_rows,
    has_unsupported_rows,
    has_zero_area_box,
    parse_annotations,
    sanitize_annotations,
)


def _names_to_list(names) -> List[str]:
    """Normalise ``names`` (dict ``{0: cat}`` or list) to an ordered list."""
    if names is None:
        return []
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return [str(n) for n in names]


def _resolve_data_arg(data: str) -> str:
    """Accept a ``data.yaml`` path, or a directory that contains one."""
    p = Path(data)
    if p.is_dir():
        for cand in ("data.yaml", "dataset.yaml"):
            if (p / cand).exists():
                return str(p / cand)
        yamls = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
        if yamls:
            return str(yamls[0])
        raise FileNotFoundError(f"No dataset YAML found in directory: {p}")
    return data


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + ``os.replace`` so a label is never half-written."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def folder_yaml(folder: str) -> Optional[str]:
    """Return an existing dataset YAML inside ``folder`` (``data.yaml`` /
    ``dataset.yaml``, else the first ``*.yaml`` / ``*.yml``), or ``None``."""
    p = Path(folder)
    if not p.is_dir():
        return None
    for cand in ("data.yaml", "dataset.yaml"):
        if (p / cand).exists():
            return str(p / cand)
    ys = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
    return str(ys[0]) if ys else None


def count_images(folder: str) -> int:
    """How many supported images live under ``folder`` (recursive); 0 if none."""
    p = Path(folder)
    if not p.is_dir():
        return 0
    try:
        return len(get_img_files(p))
    except (FileNotFoundError, ValueError):
        return 0


def scaffold_data_yaml(folder: str, names: Optional[List[str]] = None,
                       task: Optional[str] = None) -> str:
    """Write a minimal LibreYOLO ``data.yaml`` for a bare folder of images.

    The folder of images *is* the dataset: the YAML is written beside the images
    (the exact layout ``libreyolo train`` reads) with a single ``train`` split
    pointing at the folder, so labels round-trip alongside the images and flow
    straight into training -- no export, no copy, nothing moved. The recursive
    scan means this works for a flat folder *and* an ``images/`` sub-tree (where
    the ``images``->``labels`` convention puts labels in a parallel ``labels/``).

    Returns the path to the written YAML. Raises ``FileNotFoundError`` if the
    folder is missing or holds no supported images, and ``FileExistsError`` if a
    dataset YAML is already there (open that instead of overwriting it).
    """
    p = Path(folder)
    if not p.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    existing = folder_yaml(folder)
    if existing:
        raise FileExistsError(existing)
    if not get_img_files(p):
        raise FileNotFoundError(f"No images found in {folder}")
    classes = [str(n).strip() for n in (names or []) if str(n).strip()]
    cfg = {
        "path": p.resolve().as_posix(),   # forward slashes: unambiguous in YAML on every OS
        "train": ".",
        "names": classes,
        "nc": len(classes),
    }
    t = str(task or "").strip().lower()
    if t and t != "detect":               # detect is the default; only stamp non-default tasks
        cfg["task"] = t
    text = (
        "# LibreLabel project -- created from a folder of images.\n"
        "# Labels are written next to the images, where `libreyolo train` reads them.\n"
        "# Add a `val:` split (a held-out set) before training for real.\n\n"
        + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    )
    out = p / "data.yaml"
    _atomic_write_text(out, text)
    return str(out)


def update_class_names(yaml_file: str, names: List[str]) -> None:
    """Rewrite a dataset YAML's ``names`` / ``nc`` in place, preserving every
    other key. Callers must only rename or append (never delete or reorder) so
    existing label class ids keep their meaning."""
    p = Path(yaml_file)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg["names"] = list(names)
    cfg["nc"] = len(names)
    _atomic_write_text(p, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


class DatasetSession:
    """An open dataset: ordered images across train/val/test + label R/W."""

    def __init__(self, data: str):
        resolved = _resolve_data_arg(str(data))
        cfg = load_data_config(resolved, autodownload=False)
        self.yaml_file = cfg.get("yaml_file", resolved)
        self.root = cfg.get("path") or cfg.get("root") or ""
        self.names = _names_to_list(cfg.get("names"))
        nc = cfg.get("nc")
        self.nc = int(nc) if nc else len(self.names)

        self._items: List[Tuple[Path, Path, str]] = []
        seen: set = set()
        self._path_splits: dict = {}   # normalized label path -> {splits it appears in}
        for split in ("train", "val", "test"):
            imgs = cfg.get(f"{split}_img_files") or []
            labels = cfg.get(f"{split}_label_files") or img2label_paths(
                [Path(i) for i in imgs]
            )
            for ip, lp in zip(imgs, labels, strict=True):
                # A yaml may reuse a folder across splits; expose each label file
                # once so a single image can't be saved twice under two ids -- but
                # remember every split it was in, for exact-overlap leakage detection.
                key = os.path.normcase(os.path.normpath(str(lp)))
                self._path_splits.setdefault(key, set()).add(split)
                if key in seen:
                    continue
                seen.add(key)
                self._items.append((Path(ip), Path(lp), split))

        # Raw split sources (resolved paths/lists) so the duplicate fixer can
        # refuse .txt-manifest splits, where deleting a file leaves a dangling row.
        self._split_sources = {
            s: cfg.get(s) for s in ("train", "val", "test") if cfg.get(s)
        }
        self.writable, self.reason = self._check_writable()
        # Pose (kpt_shape), semantic-seg (masks_dir) and depth datasets store dense
        # labels LibreLabel can't edit; writing YOLO boxes would pollute them. The
        # `task:` key alone is enough to know -- a depth yaml may omit depths_dir
        # (the loader defaults it), a classify yaml uses no .txt labels at all, and
        # an OBB yaml's 9-field rows are oriented rectangles we'd corrupt if saved as
        # arbitrary polygons -- so treat those tasks as view-only on the task key too.
        task = str(cfg.get("task") or "").strip().lower()
        self._task = task   # used to disambiguate 4-point (OBB-vs-polygon) rows on read/write
        dense = (cfg.get("kpt_shape") or cfg.get("masks_dir")
                 or cfg.get("depths_dir") or cfg.get("depth")
                 or task in ("depth", "classify", "pose", "obb"))
        if self.writable and dense:
            self.writable = False
            self.reason = ("Keypoint / OBB / mask / depth / classify dataset: view-only in "
                           "LibreLabel — it edits boxes and polygons, and saving would drop "
                           "or corrupt the dense / task-specific labels.")
        self._deleted: set = set()  # ids of duplicates removed this session (tombstones)

    # -- safety ------------------------------------------------------------
    def _check_writable(self) -> Tuple[bool, str]:
        """Guard against the greedy ``images``->``labels`` substring swap.

        ``img2label_paths`` replaces *every* ``images`` path segment, so a root
        that itself contains ``images`` (e.g. ``my/images/proj/images/train``)
        derives a wrong label path and would silently corrupt the dataset.
        Detect the ambiguity up front and make the session read-only.
        """
        root = None
        if self.root:
            try:
                root = Path(self.root).resolve()
            except Exception:  # noqa: BLE001
                root = None
        for ip, lp, _ in self._items:
            # img2label_paths rewrites every "<sep>images" prefix, so a component that
            # *starts with* "images" (e.g. "images_2026" -> "labels_2026") mis-derives
            # the label path. A component that merely *contains* "images" but doesn't
            # start with it (e.g. "my_images") is NOT rewritten -> still writable.
            risky = [p for p in ip.parts if p.startswith("images")]
            if len(risky) > 1 or (len(risky) == 1 and risky[0] != "images"):
                return (
                    False,
                    "Ambiguous dataset layout: a path segment contains 'images' in a "
                    "way that makes the label path ambiguous, so saving could write "
                    "outside the dataset. Rename the ancestor (e.g. to 'imgs/') and reopen.",
                )
            if lp == ip:
                return False, f"Could not derive a label path for {ip}."
            # A single 'images' segment is fine *inside* the dataset (the conventional
            # images/->labels/ sibling layout), but if it sits ABOVE the root -- e.g. a
            # flat folder /home/me/images/cats opened as `train: .` -- the rewrite still
            # fires and sends labels OUTSIDE the dataset. Require the label path to stay
            # within the root for any image that is itself under the root.
            if root is not None:
                try:
                    ip.resolve().relative_to(root)
                except ValueError:
                    continue   # image not under the dataset root (unusual) -> don't second-guess
                try:
                    lp.resolve().relative_to(root)
                except ValueError:
                    return (
                        False,
                        "Saving would write labels outside the dataset folder: an ancestor "
                        "path segment named 'images' gets rewritten to 'labels'. Move the "
                        "images into an 'images/' subfolder (or rename the ancestor) and reopen.",
                    )
        return True, ""

    # -- queries -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def meta(self) -> dict:
        return {
            "root": str(self.root),
            "yaml": str(self.yaml_file),
            "names": self.names,
            "nc": self.nc,
            "count": len(self._items),
            "writable": self.writable,
            "reason": self.reason,
            "task": self._task or "detect",
            "has_val": any(s in ("val", "test") for _, _, s in self._items),
        }

    def _status(self, lp: Path) -> str:
        if not lp.exists():
            return "unlabeled"
        try:
            return "labeled" if lp.stat().st_size > 0 else "empty"
        except OSError:
            return "unlabeled"

    def list_images(self) -> List[dict]:
        rows = []
        for i, (ip, lp, split) in enumerate(self._items):
            status = "deleted" if i in self._deleted else self._status(lp)
            rows.append({"id": i, "name": ip.name, "split": split, "status": status})
        return rows

    def stats(self) -> dict:
        """Aggregate the on-disk (accepted) labels into a dataset-health summary."""
        from collections import Counter

        counts: Counter = Counter()
        labeled = empty = total_boxes = 0
        for i, (_ip, lp, _split) in enumerate(self._items):
            if i in self._deleted or not lp.exists():
                continue
            try:
                text = lp.read_text(encoding="utf-8")
            except OSError:
                continue
            anns = parse_annotations(text)
            if anns:
                labeled += 1
                total_boxes += len(anns)
                for a in anns:
                    counts[a["cls"]] += 1
            else:
                empty += 1
        n = len(self.names)
        top = [
            [self.names[c] if 0 <= c < n else str(c), cnt]
            for c, cnt in counts.most_common(12)
        ]
        live = len(self._items) - len(self._deleted)
        return {
            "total": live,
            "labeled": labeled,
            "empty": empty,
            "unlabeled": live - labeled - empty,
            "boxes": total_boxes,
            "classes": top,
        }

    def insights(self) -> dict:
        """Dataset intelligence: dimension stats + perceptual-hash duplicates.

        Decodes each image once (downscaled) to compute a dHash and read its
        size. Cached for the session. Surfaces the data-quality issues that
        matter most for YOLO training: duplicate images and, especially,
        train/val *leakage* (the same image in two splits).
        """
        if getattr(self, "_insights_cache", None) is not None:
            return self._insights_cache

        from collections import Counter

        from PIL import Image

        dims: list = []          # (w, h, idx, split)
        hashes: dict = {}        # dhash -> [idx, ...]
        failed = 0
        for i, (ip, _lp, split) in enumerate(self._items):
            if i in self._deleted:
                continue
            try:
                with Image.open(ip) as im:
                    w, h = im.size
                    g = list(im.convert("L").resize((9, 8)).getdata())
            except Exception:  # noqa: BLE001
                failed += 1
                continue
            dims.append((w, h, i, split))
            bits = 0
            for row in range(8):
                base = row * 9
                for col in range(8):
                    bits = (bits << 1) | (1 if g[base + col] > g[base + col + 1] else 0)
            hashes.setdefault(bits, []).append(i)

        def _stat(vals):
            if not vals:
                return {"min": 0, "max": 0, "mean": 0, "median": 0}
            s = sorted(vals)
            return {
                "min": s[0], "max": s[-1],
                "mean": round(sum(s) / len(s)),
                "median": s[len(s) // 2],
            }

        ws = [d[0] for d in dims]
        hs = [d[1] for d in dims]
        mp = [round(w * h / 1e6, 2) for w, h, _i, _s in dims]
        res_top = Counter((w, h) for w, h, _i, _s in dims).most_common(6)
        name = lambda i: self._items[i][0].name  # noqa: E731
        split_of = lambda i: self._items[i][2]    # noqa: E731

        dup_groups = []
        leak_groups = []
        for ids in hashes.values():
            if len(ids) < 2:
                continue
            grp = {"ids": ids, "names": [name(i) for i in ids],
                   "splits": sorted({split_of(i) for i in ids})}
            dup_groups.append(grp)
            if len(grp["splits"]) > 1:
                leak_groups.append(grp)
        dup_groups.sort(key=lambda g: -len(g["ids"]))
        # Exact same label-path listed in >1 split (deduped out of _items, so the
        # dHash pass above can't see it) -- still real train/val leakage.
        kidx = {os.path.normcase(os.path.normpath(str(self._items[i][1]))): i
                for i in range(len(self._items)) if i not in self._deleted}
        for key, splits in self._path_splits.items():
            if len(splits) > 1 and key in kidx:
                i = kidx[key]
                leak_groups.append({"ids": [i], "names": [name(i)],
                                    "splits": sorted(splits), "exact": True})

        self._insights_cache = {
            "count": len(self._items) - len(self._deleted),
            "measured": len(dims),
            "failed": failed,
            "width": _stat(ws),
            "height": _stat(hs),
            "megapixels": _stat(mp) if mp else {"min": 0, "max": 0, "mean": 0, "median": 0},
            "top_resolutions": [[w, h, c] for (w, h), c in res_top],
            "duplicate_groups": dup_groups[:50],
            "duplicate_image_count": sum(len(g["ids"]) for g in dup_groups),
            "leakage_groups": leak_groups[:50],
        }
        return self._insights_cache

    def quality(self, imgsz: int = 640) -> dict:
        """Geometry-lint accepted labels: tiny / sliver / full-frame boxes.

        Surfaces annotations a detector physically can't learn from at ``imgsz``
        (a few-pixel box), plus absurd aspect ratios and whole-frame boxes that
        are almost always slips. Reports only -- never edits a label.
        """
        from .quality import lint_annotations

        flagged: List[dict] = []
        counts = {"tiny": 0, "sliver": 0, "fullframe": 0}
        total_issues = 0
        for i, (ip, lp, _split) in enumerate(self._items):
            if i in self._deleted or not lp.exists():
                continue
            anns, editable = self.read_label(i)
            if not editable:
                continue   # view-only/dense (e.g. OBB) labels: don't lint a partial polygon view
            if not anns:
                continue
            issues = lint_annotations(anns, imgsz=imgsz)
            if issues:
                total_issues += len(issues)
                for it in issues:
                    counts[it["type"]] = counts.get(it["type"], 0) + 1
                flagged.append({"id": i, "name": ip.name, "count": len(issues),
                                "issues": issues})
        flagged.sort(key=lambda d: -d["count"])
        return {"imgsz": imgsz, "issues": total_issues, "counts": counts,
                "flagged": flagged[:100]}

    def resolve_duplicates(self, ids: List[int], *, purge: bool = False) -> dict:
        """Collapse a duplicate/leakage group to one survivor (reversible default).

        Keeps exactly one copy -- preferring the ``train`` copy so train/val
        leakage is eliminated -- and MOVES the rest (image + its label, together)
        into ``<root>/.librelabel_quarantine/`` so a probabilistic perceptual-hash
        match is never destructive. ``purge=True`` hard-deletes instead. Removed
        ids are tombstoned so open image ids stay stable for the UI. No-op +
        raises when the session is read-only; refuses ``.txt``-manifest splits
        (deleting a file there would leave a dangling manifest line).
        """
        import shutil

        if not self.writable:
            raise RuntimeError(self.reason)
        valid = [i for i in dict.fromkeys(ids)
                 if 0 <= i < len(self._items) and i not in self._deleted]
        if len(valid) < 2:
            return {"removed": [], "kept": valid[0] if valid else None,
                    "quarantine": None}
        # Prefer a survivor that actually has labels (then train split, then lowest
        # id): keeping an unlabelled train copy while quarantining the only labelled
        # copy would silently turn a labelled image unlabelled after Fix.
        def _labelled(i):
            lp = self._items[i][1]
            try:
                return lp.exists() and lp.stat().st_size > 0
            except OSError:
                return False

        rank = {"train": 0, "val": 1, "test": 2}
        keep = min(valid, key=lambda i: (0 if _labelled(i) else 1,
                                         rank.get(self._items[i][2], 3), i))
        redundant = [i for i in valid if i != keep]
        # A split defined by an explicit file list (a .txt manifest OR an inline YAML
        # list of image files) still references the file we'd move/delete, leaving a
        # dangling entry on the next load -- refuse so the user fixes the list first.
        _listed = (".txt", ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
        for i in redundant:
            src = self._split_sources.get(self._items[i][2])
            srcs = src if isinstance(src, list) else [src]
            if any(str(x).lower().endswith(_listed) for x in srcs):
                raise RuntimeError(
                    "This split is defined by an explicit file list (a .txt manifest or "
                    "an inline YAML image list); update the list before pruning so no "
                    "dangling references are left.")
        qbase = Path(self.root) if self.root else Path(self.yaml_file).parent
        qroot = qbase / ".librelabel_quarantine"
        # A broad/recursive split (e.g. ``train: .``) rglob-scans the whole tree, so a
        # quarantine dir INSIDE it would be rediscovered on the next load/train and the
        # cleanup wouldn't stick. Refuse pruning such a split rather than silently
        # un-quarantining. (Purge has no quarantine dir, so it's unaffected.)
        if not purge:
            try:
                qres = qroot.resolve()
            except OSError:
                qres = qroot
            # Check EVERY split, not just the redundant ids' own: a broad split
            # elsewhere (e.g. `test: .`) would still rglob the quarantine dir back in.
            for src in self._split_sources.values():
                for s in (src if isinstance(src, list) else [src]):
                    if not s:
                        continue
                    try:
                        d = Path(s).resolve()
                    except OSError:
                        continue
                    if d.is_dir() and (qres == d or d in qres.parents):
                        raise RuntimeError(
                            "A split is a recursive directory that would re-scan the "
                            "quarantine folder; prune with purge, or point the split at a "
                            "narrower images/ subdirectory.")
        removed: List[dict] = []
        for i in redundant:
            ip, lp, split = self._items[i]
            try:
                if purge:
                    # Delete the IMAGE first: if that fails (lock/permission) the
                    # OSError below skips the tombstone with the labelled pair fully
                    # intact -- we never delete a label whose image survives (which
                    # would silently turn a labelled image unlabelled). Once the image
                    # is gone, the label cleanup is best-effort (an orphaned label is
                    # ignored by the loader, which iterates images).
                    if ip.exists():
                        ip.unlink()
                    try:
                        if lp.exists():
                            lp.unlink()
                    except OSError:
                        pass
                else:
                    dst_img = qroot / "images" / split / f"{i}_{ip.name}"   # id prefix: never collide
                    dst_lbl = qroot / "labels" / split / f"{i}_{lp.name}"
                    dst_img.parent.mkdir(parents=True, exist_ok=True)
                    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                    moved_img = False
                    if ip.exists():
                        shutil.move(str(ip), str(dst_img))
                        moved_img = True
                    try:
                        if lp.exists():
                            shutil.move(str(lp), str(dst_lbl))
                    except OSError:
                        if moved_img:  # roll back so the item stays consistent
                            shutil.move(str(dst_img), str(ip))
                        raise
            except OSError:
                continue
            self._deleted.add(i)
            removed.append({"id": i, "name": ip.name, "split": split})
        self._insights_cache = None  # dimensions / dup groups changed
        return {"removed": removed, "kept": keep,
                "kept_name": self._items[keep][0].name,
                "mode": "purge" if purge else "quarantine",
                "quarantine": None if purge else str(qroot)}

    def _check_index(self, idx: int) -> None:
        if not (0 <= idx < len(self._items)):
            raise IndexError(f"image id out of range: {idx}")

    def image_path(self, idx: int) -> Path:
        self._check_index(idx)
        return self._items[idx][0]

    def has_label_file(self, idx: int) -> bool:
        """Whether a label ``.txt`` exists on disk (an empty file = reviewed background)."""
        self._check_index(idx)
        return self._items[idx][1].exists()

    def read_label(self, idx: int) -> Tuple[List[dict], bool]:
        """Return ``(annotations, editable)`` — mixed box/polygon annotations.

        ``editable`` is ``False`` for files holding keypoint/pose or malformed rows
        (which we don't parse), so a save never silently drops those fields.
        """
        self._check_index(idx)
        lp = self._items[idx][1]
        if not lp.exists():
            return [], self.writable   # a read-only session stays inert even for unlabeled images
        text = lp.read_text(encoding="utf-8")
        # A file is editable only if the whole dataset is writable (a dense/pose/OBB
        # dataset's box-shaped rows are a partial view we must never round-trip) AND
        # the file has no rows a save would silently alter: keypoint/malformed rows,
        # an out-of-[0,nc) class, or out-of-[0,1] coordinates the sanitizers clamp.
        editable = self.writable and not (
            has_unsupported_rows(text)
            or has_out_of_range_rows(text, self.nc)
            or has_out_of_bounds_coords(text)
            or has_degenerate_polygon(text)
            or has_zero_area_box(text)
            # 4-point rows are OBB-or-polygon-ambiguous unless the dataset says segment
            or (self._task != "segment" and has_obb_shaped_rows(text)))
        return parse_annotations(text), editable

    def label_rev(self, idx: int) -> int:
        """A monotonic revision token for the label file (its mtime in ns, 0 if
        absent), so a save can detect that another teammate rewrote it meanwhile."""
        self._check_index(idx)
        lp = self._items[idx][1]
        try:
            return lp.stat().st_mtime_ns if lp.exists() else 0
        except OSError:
            return 0

    # -- mutation ----------------------------------------------------------
    def write_label(self, idx: int, annotations: List[dict], expected_rev: Optional[int] = None) -> int:
        """Write annotations (boxes and/or polygons) atomically. Returns count.

        ``expected_rev`` (a :meth:`label_rev` token) enables optimistic concurrency:
        if the file was rewritten since the caller loaded it (another teammate saved),
        the write is refused so collaborative edits don't clobber each other.
        """
        self._check_index(idx)
        if idx in self._deleted:
            # Tombstoned by duplicate/leakage cleanup: a stale client must not be
            # able to recreate a label file for a removed image.
            raise RuntimeError("This image was removed during duplicate cleanup; it is no longer editable.")
        if not self.writable:
            raise RuntimeError(self.reason)
        lp = self._items[idx][1]
        if expected_rev is not None and self.label_rev(idx) != expected_rev:
            raise RuntimeError("This image was changed by someone else since you opened it; "
                               "reload it before saving so their labels aren't overwritten.")
        if lp.exists():
            existing = lp.read_text(encoding="utf-8")
            if has_unsupported_rows(existing):
                raise RuntimeError("This label file has keypoint/unsupported rows; it is read-only.")
            if has_out_of_range_rows(existing, self.nc):
                raise RuntimeError("This label file has class ids outside the dataset's nc; it is read-only.")
            if has_out_of_bounds_coords(existing):
                raise RuntimeError("This label file has coordinates outside [0, 1]; it is read-only.")
            if has_degenerate_polygon(existing):
                raise RuntimeError("This label file has a zero-area (collinear/collapsed) polygon; it is read-only.")
            if has_zero_area_box(existing):
                raise RuntimeError("This label file has a zero-width/height box; it is read-only.")
            if self._task != "segment" and has_obb_shaped_rows(existing):
                raise RuntimeError("This label file has 4-point (OBB/quad) rows; without task: segment "
                                   "they're treated as oriented boxes and kept read-only.")
        clean = sanitize_annotations(annotations, self.nc)
        lp.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(lp, format_annotations(clean))
        return len(clean)
