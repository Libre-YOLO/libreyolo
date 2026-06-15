"""Boost: train-in-the-loop on the labels you've already accepted.

After you've accepted enough boxes, LibreLabel can quietly fine-tune the *head*
of your own (tiny) detector on those labels -- backbone and neck frozen -- and
tell you whether it helped: "model agrees 88% (was 61%)". That delta is the
whole point. It is the difference between *help me make labels* and *the tool is
learning from me*.

Honesty (this is ``REAL_MINIMAL``, not "real training"): a frozen-backbone,
1-2 epoch CPU fine-tune of a few dozen images is a *probe* -- it nudges the head
toward your corrections and measures the agreement shift. Budget a couple of
minutes on a laptop CPU. It runs on a background thread; the UI shows a chip.

Hard invariant (the trust contract): this writes **nothing** to the dataset's
label files. It snapshots the accepted labels into a throwaway temp dataset
(symlinked images + freshly written ``.txt``), trains into a temp dir, and
deletes the whole temp tree when done. Your source labels are never opened for
writing.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

MIN_IMAGES = 8  # below this a head nudge is noise; refuse with a clear message


class BoostEngine:
    """Lazy, single-flight background fine-tune of the detector head + a metric."""

    def __init__(self, session, model_name: str, enabled: bool = True, engine=None):
        self.session = session
        self.model_name = model_name
        self.enabled = enabled
        self.engine = engine          # AssistEngine: where the boosted model is registered
        self.boosted_model = None     # in-memory boosted model (usable after a run)
        self._lock = threading.Lock()
        self._thread = None
        self.state: dict = {"state": "idle"}

    # -- public surface ----------------------------------------------------
    def status(self) -> dict:
        return dict(self.state)

    def _running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _set(self, **kw) -> None:
        # Atomic reference swap (not in-place mutation) so a concurrent status()
        # read never sees a half-updated dict.
        self.state = {**self.state, **kw}

    def on_project_switch(self, session) -> None:
        """Rebind to a new project. A still-running boost keeps its state and its
        own (thread-local) temp dir -- we never stomp a live worker to 'idle'."""
        self.session = session
        self.boosted_model = None
        if not self._running():
            self.state = {"state": "idle"}

    def start(self, *, epochs: int = 2, imgsz: int = 512, batch: int = 4) -> dict:
        with self._lock:
            session = self.session   # one snapshot for the whole start: a concurrent
            if session is None:      # on_project_switch() can't mix images from one
                return {"started": False, "reason": "No project open."}  # project with names from another
            if self.state.get("state") == "running" or self._running():
                return {"started": False, "reason": "A boost is already running."}
            if not self.enabled:
                return {"started": False, "reason": "Assist is disabled."}
            epochs = max(1, min(int(epochs), 10))    # bound local CPU/temp-disk use
            imgsz = max(64, min(int(imgsz), 1280))
            batch = max(1, min(int(batch), 64))
            images, labels, accepted_cls = self._gather(session)
            if len(images) < MIN_IMAGES:
                msg = (f"Boost needs at least {MIN_IMAGES} labelled images "
                       f"(you have {len(images)}). Accept a few more, then try again.")
                self.state = {"state": "error", "error": msg}
                return {"started": False, "reason": msg}
            names = list(session.names)   # frozen now: a mid-boost switch can't skew it
            started_session = session     # only publish the result if still on this project
            self.state = {"state": "running", "phase": "starting", "n": len(images)}
            self._thread = threading.Thread(
                target=self._run, name="librelabel-boost", daemon=True,
                args=(images, labels, accepted_cls, names, started_session, epochs, imgsz, batch))
            self._thread.start()
            return {"started": True, "n": len(images)}

    # -- gather accepted labels (read-only) --------------------------------
    def _gather(self, session):
        from collections import Counter

        from .radar import _ann_to_box

        images: List[str] = []
        labels: Dict[str, list] = {}
        accepted_cls: Dict[str, int] = {}
        deleted = getattr(session, "_deleted", set())
        for idx in range(len(session)):
            if idx in deleted:
                continue
            anns, editable = session.read_label(idx)
            if not editable:
                continue   # keypoint/OBB/out-of-range file: parsed boxes are only a
                           # partial view -> training on the subset skews the metric
            boxes = [b for b in (_ann_to_box(a) for a in anns) if b is not None]
            if not boxes:
                continue
            p = str(session.image_path(idx))
            images.append(p)
            labels[p] = [(int(c), cx, cy, w, h) for (c, cx, cy, w, h) in boxes]
            # the metric is class-top-1; the image's dominant class is the target
            accepted_cls[p] = Counter(int(b[0]) for b in boxes).most_common(1)[0][0]
        return images, labels, accepted_cls

    # -- throwaway snapshot dataset (writes only to temp) ------------------
    def _snapshot(self, root: str, images, labels, class_names) -> str:
        import json
        import shutil

        base = Path(root) / "boost_ds"
        (base / "images" / "train").mkdir(parents=True, exist_ok=True)
        (base / "labels" / "train").mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            src = Path(img)
            stem = f"{i}_{src.stem}"  # prefix dodges same-name collisions across dirs
            link = base / "images" / "train" / f"{stem}{src.suffix}"
            try:
                if not link.exists():
                    link.symlink_to(src.resolve())
            except (OSError, NotImplementedError):  # Windows without symlink priv
                shutil.copy2(img, link)
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                     for (c, cx, cy, w, h) in labels.get(img, [])]
            (base / "labels" / "train" / f"{stem}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        # JSON-encode each scalar: a JSON string is valid YAML and survives class
        # names / paths containing YAML-significant chars (':', '#', quotes, ...),
        # so boost trains on the same schema the source dataset declares.
        names = "\n".join(f"  {i}: {json.dumps(str(n))}" for i, n in enumerate(class_names))
        data_yaml = base / "data.yaml"
        data_yaml.write_text(
            f"path: {json.dumps(base.as_posix())}\ntrain: images/train\nval: images/train\n"
            f"nc: {len(class_names)}\nnames:\n{names}\n", encoding="utf-8")
        return str(data_yaml)

    # -- agreement metric --------------------------------------------------
    def _agreement(self, model, images, accepted_cls, names) -> float:
        """Fraction of images whose top detection's class matches the label.

        Compared in the **dataset's** class space: the base model predicts in its
        own space (e.g. COCO-80) while the boosted model predicts in the dataset's
        space, so we resolve the predicted class *name* to a dataset index via the
        same name map ``assist``/``radar`` use. This keeps base vs boosted on the
        same axis -- without it the headline delta would be meaningless.
        """
        import numpy as np

        from .assist import build_class_map

        resolve = build_class_map(names)
        hits = total = 0
        for img in images:
            want = accepted_cls.get(img)
            if want is None:
                continue
            total += 1
            try:
                r = model(img)
            except Exception:  # noqa: BLE001
                continue
            r = r[0] if isinstance(r, list) else r
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            bn = boxes.numpy()
            top_idx = int(bn.cls[int(np.argmax(bn.conf))])
            rnames = getattr(r, "names", {}) or {}
            didx = resolve(str(rnames.get(top_idx, top_idx)))
            hits += int(didx == want)
        return hits / total if total else 0.0

    # -- background worker -------------------------------------------------
    def _run(self, images, labels, accepted_cls, names, started_session, epochs, imgsz, batch):
        import shutil
        import tempfile

        tmp = None    # thread-local: each worker owns + cleans only its own temp tree
        torch_mod = None
        prev_threads = None
        try:
            import torch

            torch_mod = torch
            try:
                prev_threads = torch.get_num_threads()
                torch.set_num_threads(max(1, (prev_threads or 4) - 1))
            except Exception:  # noqa: BLE001
                prev_threads = None  # don't try to restore a value we never read
            from libreyolo import LibreYOLO
            from libreyolo.cli.config import resolve_model_name

            from .assist import weight_is_local

            weight = resolve_model_name(self.model_name)
            # Offline contract: Boost must not silently download the base checkpoint
            # before measuring/training -- fail closed with the same hint as assist.
            if not weight_is_local(weight):
                raise RuntimeError(
                    f"Boost base model ({weight}) isn't present locally. LibreLabel won't "
                    f"download weights automatically -- fetch it once (e.g. "
                    f"`libreyolo predict model={self.model_name} source=...`), then try Boost again.")

            self._set(phase="measuring your model's current agreement")
            base = LibreYOLO(weight, device="cpu")
            base_agree = self._agreement(base, images, accepted_cls, names)
            self._set(base_agreement=round(base_agree, 3))

            self._set(phase="snapshotting your accepted labels")
            tmp = tempfile.mkdtemp(prefix="librelabel_boost_")
            data_yaml = self._snapshot(tmp, images, labels, names)

            self._set(phase="training the head on your labels (a few minutes)")
            model = LibreYOLO(weight, device="cpu")
            results = model.train(
                data=data_yaml, epochs=epochs, batch=batch, imgsz=imgsz,
                device="cpu", amp=False, workers=0, freeze=["backbone", "neck"],
                lr0=0.02, patience=0, project=str(Path(tmp) / "runs"),
                name="boost", exist_ok=True, seed=0)
            ckpt = results.get("best_checkpoint") or results.get("last_checkpoint")
            if not ckpt:
                raise RuntimeError("training produced no checkpoint")

            self._set(phase="measuring the boosted model")
            boosted = LibreYOLO(ckpt, device="cpu")
            boost_agree = self._agreement(boosted, images, accepted_cls, names)

            # Keep the boosted model usable: its weights are already in memory, so
            # the temp checkpoint can be deleted. Register it on the assist engine
            # under "boosted" so prelabel/auto-label/Radar can use it (the flywheel
            # actually turning -- not just a number).
            usable = False
            if self.engine is not None:
                try:
                    with self.engine._lock:
                        if self.session is started_session:   # discard if switched away mid-train
                            self.engine._models["boosted"] = boosted
                            self.boosted_model = boosted
                            usable = True
                except Exception:  # noqa: BLE001
                    logger.exception("could not register the boosted model")

            self.state = {"state": "done", "n": len(images),
                          "base_agreement": round(base_agree, 3),
                          "boosted_agreement": round(boost_agree, 3),
                          "delta": round(boost_agree - base_agree, 3),
                          "epochs": epochs, "usable": usable}
        except Exception as exc:  # noqa: BLE001 - surface as a chip, never crash the server
            logger.exception("boost failed")
            self.state = {"state": "error", "error": str(exc)}
        finally:
            # Restore the process-wide thread count: set_num_threads() is global,
            # so without this each boost run permanently shrinks the pool for later
            # inference/training until it bottoms out at one thread.
            if torch_mod is not None and prev_threads is not None:
                try:
                    torch_mod.set_num_threads(prev_threads)
                except Exception:  # noqa: BLE001
                    logger.exception("could not restore torch thread count")
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
