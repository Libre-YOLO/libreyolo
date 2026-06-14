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

from .labelio import format_annotations, parse_annotations, sanitize_annotations


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
        for split in ("train", "val", "test"):
            imgs = cfg.get(f"{split}_img_files") or []
            labels = cfg.get(f"{split}_label_files") or img2label_paths(
                [Path(i) for i in imgs]
            )
            for ip, lp in zip(imgs, labels):
                # A yaml may reuse a folder across splits; expose each label file
                # once so a single image can't be saved twice under two ids.
                key = os.path.normcase(os.path.normpath(str(lp)))
                if key in seen:
                    continue
                seen.add(key)
                self._items.append((Path(ip), Path(lp), split))

        self.writable, self.reason = self._check_writable()

    # -- safety ------------------------------------------------------------
    def _check_writable(self) -> Tuple[bool, str]:
        """Guard against the greedy ``images``->``labels`` substring swap.

        ``img2label_paths`` replaces *every* ``images`` path segment, so a root
        that itself contains ``images`` (e.g. ``my/images/proj/images/train``)
        derives a wrong label path and would silently corrupt the dataset.
        Detect the ambiguity up front and make the session read-only.
        """
        for ip, lp, _ in self._items:
            if sum(1 for part in ip.parts if part == "images") > 1:
                return (
                    False,
                    "Ambiguous dataset layout: a path contains the segment "
                    "'images' more than once, so label paths cannot be derived "
                    "safely. Rename the ancestor folder (e.g. to 'imgs/') and "
                    "reopen.",
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
            rows.append(
                {"id": i, "name": ip.name, "split": split, "status": self._status(lp)}
            )
        return rows

    def stats(self) -> dict:
        """Aggregate the on-disk (accepted) labels into a dataset-health summary."""
        from collections import Counter

        counts: Counter = Counter()
        labeled = empty = total_boxes = 0
        for _ip, lp, _split in self._items:
            if not lp.exists():
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
        return {
            "total": len(self._items),
            "labeled": labeled,
            "empty": empty,
            "unlabeled": len(self._items) - labeled - empty,
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

        self._insights_cache = {
            "count": len(self._items),
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

    def _check_index(self, idx: int) -> None:
        if not (0 <= idx < len(self._items)):
            raise IndexError(f"image id out of range: {idx}")

    def image_path(self, idx: int) -> Path:
        self._check_index(idx)
        return self._items[idx][0]

    def read_label(self, idx: int) -> Tuple[List[dict], bool]:
        """Return ``(annotations, editable)`` — mixed box/polygon annotations.

        Editable is always True now: boxes and polygons both round-trip.
        """
        self._check_index(idx)
        lp = self._items[idx][1]
        if not lp.exists():
            return [], True
        return parse_annotations(lp.read_text(encoding="utf-8")), True

    # -- mutation ----------------------------------------------------------
    def write_label(self, idx: int, annotations: List[dict]) -> int:
        """Write annotations (boxes and/or polygons) atomically. Returns count."""
        self._check_index(idx)
        if not self.writable:
            raise RuntimeError(self.reason)
        lp = self._items[idx][1]
        clean = sanitize_annotations(annotations, self.nc)
        lp.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(lp, format_annotations(clean))
        return len(clean)
