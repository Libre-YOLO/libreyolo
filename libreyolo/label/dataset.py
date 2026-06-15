"""Dataset session for LibreLabel: enumerate images and round-trip YOLO labels.

Thin wrapper over LibreYOLO's own ``load_data_config`` / ``img2label_paths`` so
that the image<->label mapping and ``data.yaml`` resolution are *identical* to
what training uses -- LibreLabel writes labels exactly where the trainer reads
them. No database; the filesystem dataset is the store.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from libreyolo.data.utils import img2label_paths, load_data_config

from .labelio import (
    format_annotations,
    has_out_of_range_rows,
    has_unsupported_rows,
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
            try:
                anns = parse_annotations(lp.read_text(encoding="utf-8"))
            except OSError:
                continue
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
        removed: List[dict] = []
        for i in redundant:
            ip, lp, split = self._items[i]
            try:
                if purge:
                    # Remove the label first: the worst partial state is then an
                    # unlabeled (still-present) image -- recoverable -- never a
                    # label orphaned from a deleted image. If the image unlink
                    # fails we skip the tombstone so the surviving image stays live.
                    if lp.exists():
                        lp.unlink()
                    if ip.exists():
                        ip.unlink()
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
            return [], True
        text = lp.read_text(encoding="utf-8")
        # A file with keypoint/pose or malformed rows -- or an integer class outside
        # the dataset's nc -- stays read-only, so a save can't drop the fields we
        # don't parse or sanitize an out-of-range class away.
        editable = not (has_unsupported_rows(text) or has_out_of_range_rows(text, self.nc))
        return parse_annotations(text), editable

    # -- mutation ----------------------------------------------------------
    def write_label(self, idx: int, annotations: List[dict]) -> int:
        """Write annotations (boxes and/or polygons) atomically. Returns count."""
        self._check_index(idx)
        if idx in self._deleted:
            # Tombstoned by duplicate/leakage cleanup: a stale client must not be
            # able to recreate a label file for a removed image.
            raise RuntimeError("This image was removed during duplicate cleanup; it is no longer editable.")
        if not self.writable:
            raise RuntimeError(self.reason)
        lp = self._items[idx][1]
        if lp.exists():
            existing = lp.read_text(encoding="utf-8")
            if has_unsupported_rows(existing):
                raise RuntimeError("This label file has keypoint/unsupported rows; it is read-only.")
            if has_out_of_range_rows(existing, self.nc):
                raise RuntimeError("This label file has class ids outside the dataset's nc; it is read-only.")
        clean = sanitize_annotations(annotations, self.nc)
        lp.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(lp, format_annotations(clean))
        return len(clean)
