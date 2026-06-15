"""Model-assisted auto-labelling for LibreLabel (the AI wedge).

Runs the user's *own* in-package detector (YOLO9 / RF-DETR, etc.) over images and
**suggests** boxes the human reviews and accepts -- fully local, offline, MIT, no
cloud and no account. Reuses the exact predict path ``libreyolo ui`` already uses
(lazy cached model + a lock); adds **no** new runtime dependency.

Trust contract (load-bearing): nothing in this module ever writes a label file.
Suggestions are produced and parked in an in-memory ``pending`` map; the *only*
way a box reaches ``labels/*.txt`` is an explicit human accept through
``POST /api/label/<id>``. A 0.25-confidence machine guess must never become
ground truth indistinguishable from a hand-verified box.

The detector predicts in *its* class space (e.g. COCO-80); the dataset has its
own ``data.yaml`` names. Suggestions are mapped to the dataset by **class name**
(normalised + a small synonym table) with a clear ``mapped`` flag, so unmatched
detections are shown for review (never silently dropped) rather than written.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolo9-t"

# Common cross-dataset aliases (PASCAL/COCO spellings, etc.). Resolved on both
# the model side and the dataset side so direction doesn't matter.
_SYNONYMS = {
    "people": "person",
    "pedestrian": "person",
    "tvmonitor": "tv",
    "television": "tv",
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "plane": "airplane",
    "sofa": "couch",
    "pottedplant": "potted_plant",
    "diningtable": "dining_table",
    "cellphone": "cell_phone",
    "mobilephone": "cell_phone",
}


def _norm(s) -> str:
    return re.sub(r"[\s\-]+", "_", str(s).strip().lower())


def _canon(s) -> str:
    n = _norm(s)
    return _SYNONYMS.get(n, n)


def build_class_map(ds_names: List[str]) -> Callable[[str], Optional[int]]:
    """Return ``resolve(model_class_name) -> dataset_index | None`` (by name)."""
    table: Dict[str, int] = {}
    for i, n in enumerate(ds_names):
        table.setdefault(_norm(n), i)
        table.setdefault(_canon(n), i)

    def resolve(model_class_name: str) -> Optional[int]:
        idx = table.get(_norm(model_class_name))
        return idx if idx is not None else table.get(_canon(model_class_name))

    return resolve


class AssistEngine:
    """Lazy, thread-safe wrapper around the LibreYOLO predict path (no writes)."""

    def __init__(self, device: str = "auto", default_model: str = DEFAULT_MODEL, enabled: bool = True):
        self.device = device
        self.default_model = default_model
        self.enabled = enabled
        self._models: dict = {}
        self._la = None                # LocateAnything (open-vocab), lazy
        self._lock = threading.Lock()  # serialize inference (models not thread-safe)
        self.pending: Dict[int, List[dict]] = {}  # idx -> suggestions (never written)
        self._pending_lock = threading.Lock()

    # -- availability ------------------------------------------------------
    def model_names(self) -> List[str]:
        if not self.enabled:
            return []
        try:
            from libreyolo.cli.config import get_all_cli_names

            return sorted(get_all_cli_names())
        except Exception:  # noqa: BLE001
            logger.exception("could not list assist models")
            return []

    def status(self) -> dict:
        names = self.model_names()
        default = (
            self.default_model
            if self.default_model in names
            else (names[0] if names else None)
        )
        return {"available": bool(names) and self.enabled, "models": names,
                "default": default, "locate": self.locate_available()}

    # -- pending suggestions (in-memory only) ------------------------------
    def set_pending(self, idx: int, suggestions: List[dict]) -> None:
        with self._pending_lock:
            if suggestions:
                self.pending[idx] = suggestions
            else:
                self.pending.pop(idx, None)

    def get_pending(self, idx: int) -> List[dict]:
        with self._pending_lock:
            return list(self.pending.get(idx, []))

    def clear_pending(self) -> None:
        with self._pending_lock:
            self.pending.clear()

    # -- inference ---------------------------------------------------------
    def _get_model(self, name: str):
        model = self._models.get(name)
        if model is None:
            from libreyolo import LibreYOLO
            from libreyolo.cli.config import resolve_model_name

            weight = resolve_model_name(name)
            logger.info("LibreLabel assist loading model %s (%s)", name, weight)
            model = LibreYOLO(weight, device=self.device)
            self._models[name] = model
        return model

    def predict_image(
        self,
        image_path: Path,
        names: List[str],
        model_name: Optional[str] = None,
        conf: float = 0.25,
        map_all_to_zero: bool = False,
    ) -> List[dict]:
        """Suggested boxes for one image.

        Each: ``{cls, name, cx, cy, w, h, conf, mapped}`` where ``cls`` is the
        dataset class index (or ``None`` when the model's class name isn't in the
        dataset) and coordinates are normalised ``[0, 1]``. Unmapped suggestions
        are returned (not dropped) so the UI can show them for review.
        """
        model_name = model_name or self.default_model
        single_class = len(names) == 1
        with self._lock:
            model = self._get_model(model_name)
            result = model(str(image_path), conf=conf)
        return self._extract(result, image_path, names,
                             single_class=single_class, map_all_to_zero=map_all_to_zero)

    def _extract(self, result, image_path, names, *, single_class=False, map_all_to_zero=False):
        """Turn a detector/VLM Results into normalised, name-mapped suggestions."""
        resolve = build_class_map(names)
        r = result[0] if isinstance(result, list) else result
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        bn = boxes.numpy()
        try:
            xywhn = bn.xywhn
        except Exception:  # noqa: BLE001 - orig_shape missing; normalise from xyxy
            from PIL import Image

            with Image.open(image_path) as im:
                iw, ih = im.size
            xywhn = [
                [((x1 + x2) / 2) / iw, ((y1 + y2) / 2) / ih, (x2 - x1) / iw, (y2 - y1) / ih]
                for x1, y1, x2, y2 in bn.xyxy
            ]
        cls_arr, conf_arr = bn.cls, bn.conf
        model_names = getattr(r, "names", {}) or {}
        out: List[dict] = []
        for (cx, cy, w, h), c, p in zip(xywhn, cls_arr, conf_arr):
            cid = int(c)
            mname = str(model_names.get(cid, cid))
            didx = 0 if (map_all_to_zero and single_class) else resolve(mname)
            out.append({"cls": didx, "name": mname, "cx": float(cx), "cy": float(cy),
                        "w": float(w), "h": float(h), "conf": float(p), "mapped": didx is not None})
        return out

    # -- LocateAnything (open-vocab text-prompt detection) -----------------
    def locate_available(self) -> bool:
        # Surface the option whenever transformers is present; the actual 3B model
        # + its VLM deps download/load on use and error with a clear install hint.
        if not self.enabled:
            return False
        try:
            import importlib.util

            return importlib.util.find_spec("transformers") is not None
        except Exception:  # noqa: BLE001
            return False

    def _get_la(self):
        if self._la is None:
            from libreyolo import LibreVLM

            logger.info("LibreLabel loading LocateAnything (NVIDIA, non-commercial)")
            self._la = LibreVLM("locate-anything", device=self.device)
        return self._la

    def predict_locate(self, image_path: Path, names: List[str], classes: List[str]) -> List[dict]:
        """Open-vocabulary detection via LocateAnything. ``classes`` = label strings."""
        clean = [c.strip() for c in classes if c and c.strip()]
        if not clean:
            return []
        with self._lock:
            m = self._get_la()
            m.set_classes(clean)
            result = m.predict(str(image_path))
        return self._extract(result, image_path, names)

    def autolabel_dataset(
        self,
        session,
        model_name: Optional[str] = None,
        conf: float = 0.25,
        only_unlabeled: bool = True,
        progress: Optional[Callable[[dict], None]] = None,
        engine: str = "yolo",
        classes: Optional[List[str]] = None,
    ) -> dict:
        """Predict over every (unlabeled) image and PARK suggestions in ``pending``.

        Writes nothing to disk. Skips images that already have labels or hold
        polygon/OBB rows (not box-editable). Calls ``progress(event)`` per image.
        """
        from collections import Counter

        names = session.names
        total = len(session)
        suggested_images = 0
        suggested_boxes = 0
        skipped = 0
        cls_counts: Counter = Counter()
        for idx in range(total):
            existing, editable = session.read_label(idx)
            name = session.image_path(idx).name
            if only_unlabeled and (existing or not editable):
                skipped += 1
                if progress:
                    progress({"type": "progress", "i": idx + 1, "total": total,
                              "id": idx, "name": name, "count": 0, "skipped": True})
                continue
            try:
                if engine == "locate":
                    sugg = self.predict_locate(session.image_path(idx), names, classes or [])
                else:
                    sugg = self.predict_image(session.image_path(idx), names, model_name, conf)
            except Exception as exc:  # noqa: BLE001
                logger.exception("auto-label failed on image %d", idx)
                if progress:
                    progress({"type": "progress", "i": idx + 1, "total": total,
                              "id": idx, "name": name, "count": 0, "error": str(exc)})
                continue
            self.set_pending(idx, sugg)
            for s in sugg:
                if s.get("mapped") and s.get("name"):
                    cls_counts[str(s["name"])] += 1
            if sugg:
                suggested_images += 1
                suggested_boxes += len(sugg)
            if progress:
                progress({"type": "progress", "i": idx + 1, "total": total,
                          "id": idx, "name": name, "count": len(sugg)})
        return {"type": "done", "suggested": suggested_images, "boxes": suggested_boxes,
                "skipped": skipped, "total": total, "classes": cls_counts.most_common(8)}
