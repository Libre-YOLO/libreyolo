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


def weight_is_local(weight: str) -> bool:
    """Whether a model weight is already on disk (no download needed).

    The offline contract for *every* model-backed LibreLabel action (assist
    prelabel/auto-label, the embedding Map, Boost): a "fully local" labeller must
    not silently pull a checkpoint over the network. A passthrough path
    (``best.pt`` / an absolute path) is checked directly; a known weight filename
    resolves under the ``weights/`` dir, the same convention as the rest of
    LibreYOLO.
    """
    from pathlib import Path

    p = Path(weight)
    return p.is_file() or (Path("weights") / p.name).is_file()


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
    # Task aliases that don't return Results.boxes -> useless as box auto-labelers.
    _NON_BOX = ("-cls", "-sem", "-depth", "l2cs")

    def model_names(self) -> List[str]:
        if not self.enabled:
            return []
        try:
            from libreyolo.cli.config import get_all_cli_names

            names = get_all_cli_names()
            return sorted(n for n in names if not any(s in n for s in self._NON_BOX))
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

    def has_model(self, name: str) -> bool:
        """Whether a model is loaded under ``name`` (e.g. the boosted model), locked."""
        with self._lock:
            return name in self._models

    def _require_enabled(self) -> None:
        """Hard gate: ``--no-assist`` must block *inference*, not just the listing.

        ``model_names()``/``locate_available()`` already honor ``enabled``; this
        ensures ``predict_*``/``autolabel_*`` can't load or run a model when the
        engine was constructed disabled, closing the bypass.
        """
        if not self.enabled:
            raise RuntimeError("AI assist is disabled (--no-assist).")

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
            # Offline by contract: a "fully local" labeller must not silently pull a
            # checkpoint over the network on first prelabel/auto-label. Refuse with a
            # clear hint instead of letting the factory auto-download.
            if not weight_is_local(weight):
                raise RuntimeError(
                    f"Assist model '{name}' ({weight}) isn't present locally. LibreLabel "
                    f"won't download weights automatically -- fetch it once (e.g. "
                    f"`libreyolo predict model={name} source=...`), then retry.")
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
        self._require_enabled()
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
        for (cx, cy, w, h), c, p in zip(xywhn, cls_arr, conf_arr, strict=True):
            cid = int(c)
            mname = str(model_names.get(cid, cid))
            didx = 0 if (map_all_to_zero and single_class) else resolve(mname)
            out.append({"cls": didx, "name": mname, "cx": float(cx), "cy": float(cy),
                        "w": float(w), "h": float(h), "conf": float(p), "mapped": didx is not None})
        return out

    # -- LocateAnything (open-vocab text-prompt detection) -----------------
    def locate_available(self) -> bool:
        # LocateAnything is a *non-commercial* (research-only) model, so unlike the
        # MIT/Apache assist path it is OFF by default and never advertised unless the
        # user explicitly opts in via LIBRELABEL_ENABLE_LOCATE=1 -- an informed,
        # deliberate acknowledgement of its license. Even then it needs the VLM extra.
        import os

        if not self.enabled:
            return False
        # Require an explicit affirmative value: a bare truthiness test would treat
        # LIBRELABEL_ENABLE_LOCATE=0 / "false" as enabled (non-empty strings are
        # truthy), accidentally exposing the non-commercial model.
        if os.environ.get("LIBRELABEL_ENABLE_LOCATE", "").strip().lower() not in (
                "1", "true", "yes", "on"):
            return False
        try:
            import importlib.util

            # LocateAnything needs the VLM extra (its remote-code deps), not just
            # transformers; decord is the distinctive one. Gate on it so the option
            # isn't advertised in plain `label`/`sam` installs where it fails to load.
            return all(importlib.util.find_spec(m) is not None
                       for m in ("transformers", "decord"))
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
        self._require_enabled()
        # Fail closed: the hidden engine=locate path must honour the opt-in too, not
        # just the /api/assist/status advertising.
        if not self.locate_available():
            raise RuntimeError(
                "LocateAnything is a non-commercial model and is opt-in only: set "
                "LIBRELABEL_ENABLE_LOCATE=1 (and install the VLM extra) to use it.")
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
        current: Optional[Callable[[], bool]] = None,
        store: Optional[Callable[[int, list], bool]] = None,
    ) -> dict:
        """Predict over every (unlabeled) image and PARK suggestions in ``pending``.

        Writes nothing to disk. Skips images that already have labels or hold
        polygon/OBB rows (not box-editable). Calls ``progress(event)`` per image.
        ``store(idx, sugg) -> bool`` (when given) publishes suggestions atomically
        with the project switch; a ``False`` return means the project changed and
        the run stops.
        """
        from collections import Counter

        self._require_enabled()
        # Pre-flight the model once so a systemic load failure (missing weights, the
        # locate opt-in, a disabled engine) aborts the whole run with an actionable
        # error -- not swallowed per-image into a misleading "0 boxes" finish.
        if engine == "locate":
            if not self.locate_available():
                raise RuntimeError(
                    "LocateAnything is opt-in only: set LIBRELABEL_ENABLE_LOCATE=1 "
                    "(and install the VLM extra) to use it.")
        else:
            with self._lock:
                self._get_model(model_name or self.default_model)
        names = session.names
        total = len(session)
        suggested_images = 0
        suggested_boxes = 0
        skipped = 0
        cls_counts: Counter = Counter()
        deleted = getattr(session, "_deleted", set())
        for idx in range(total):
            if current is not None and not current():
                break   # project switched away -> stop populating pending for a stale session
            if idx in deleted:
                continue   # quarantined/removed image: no file on disk to label
            _existing, editable = session.read_label(idx)
            name = session.image_path(idx).name
            # Read-only files (pose/keypoint/unsupported rows) are view-only and must
            # never collect suggestions, regardless of only_unlabeled. Beyond that, a
            # label file that exists (even empty = reviewed background) is "done".
            if not editable or (only_unlabeled and session.has_label_file(idx)):
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
            if current is not None and not current():
                break   # switched projects *during* the slow predict: don't write stale
                        # suggestions into the pending map open_project() just cleared
            if store is not None:
                if not store(idx, sugg):
                    break   # atomic store refused -> project switched, stop the run
            else:
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
