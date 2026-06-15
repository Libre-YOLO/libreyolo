"""Interactive segmentation for LibreLabel via the in-package LibreSAM tier.

Wraps ``libreyolo.models.sam.LibreSAM`` (SAM-1, Apache-2.0) and its encode-once /
prompt-many surface: ``set_image()`` caches the image embedding so repeated
clicks and +/- point refinements on the same image are near-instant. A click
(or +/- points, or a box) -> mask -> simplified polygon the human accepts/edits.
Fully local/offline; nothing here writes a label.

LibreSAM is the preferred segmentation engine (vs a raw detector) because it is
promptable and class-agnostic — exactly the click-to-mask interaction labellers
want. Optional + gated: absent the ``[sam]`` extra it simply stays hidden.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SAM_SIZE = "base"


class SamEngine:
    """Lazy, thread-safe LibreSAM wrapper: point/box/multi-point -> polygon."""

    def __init__(self, enabled: bool = True, device: str = "auto"):
        self.enabled = enabled
        s = str(device).strip()
        self._device = ("cuda:" + s) if s.isdigit() else s   # `device=0` -> cuda:0, like the detector path
        self._model = None
        self._cur: Optional[str] = None       # path currently encoded via set_image
        self._dims: dict = {}                  # path -> (w, h)
        self._lock = threading.Lock()
        self._checked: Optional[bool] = None

    def available(self) -> bool:
        if not self.enabled:
            return False
        if self._checked is None:
            self._checked = self._sam_cached()
        return self._checked

    @staticmethod
    def _sam_cached() -> bool:
        """True only if transformers is installed AND the LibreSAM *base* snapshot
        is already on disk -- so we never advertise Smart-segment when using it
        would download.

        ``LibreSAMModel._ensure_weights`` stores the snapshot under
        ``weights/LibreSAM<size>/`` (NOT the HF hub cache) and marks completion
        with ``.libreyolo_snapshot_complete``. We mirror that exact, size-specific
        check here (cheaply, without importing the torch-heavy model module) so a
        *different* SAM variant cached elsewhere can't make us claim availability.
        """
        import importlib.util
        from pathlib import Path

        if importlib.util.find_spec("transformers") is None:
            return False
        # Kept in lockstep with libreyolo.models.sam.{model.FILENAME_PREFIX,
        # base._SNAPSHOT_COMPLETE_MARKER}; cheap on purpose (runs on status()).
        local_dir = Path("weights") / ("LibreSAM" + SAM_SIZE)
        if not (local_dir / ".libreyolo_snapshot_complete").exists():
            return False
        if not (local_dir / "config.json").exists():
            return False
        try:
            return any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))
        except OSError:
            return False

    def _ensure(self):
        if self._model is None:
            from libreyolo import LibreSAM

            # Offline-only by contract: LibreLabel never triggers a multi-hundred-MB
            # download on demand. Rather than mutate the process-wide HF_HUB_OFFLINE
            # env (unsafe in a threaded server), we refuse up front when the base
            # snapshot isn't already on disk; with a complete local snapshot,
            # LibreSAM loads straight from the weights/ dir and never hits the network.
            if not self._sam_cached():
                raise RuntimeError(
                    "SAM weights are not present locally. LibreLabel never downloads "
                    "them on demand -- run the LibreSAM model once (e.g. via the "
                    "`sam` command) to fetch them, then reopen Smart-segment.")
            logger.info("LibreLabel loading LibreSAM(%s) on %s", SAM_SIZE, self._device)
            self._model = LibreSAM(SAM_SIZE, device=self._device)
        return self._model

    def _size(self, image_path: Path):
        key = str(image_path)
        wh = self._dims.get(key)
        if wh is None:
            from PIL import Image

            with Image.open(image_path) as im:
                wh = im.size
            self._dims[key] = wh
        return wh

    def segment(
        self,
        image_path: Path,
        point: Optional[tuple] = None,
        points: Optional[list] = None,
        labels: Optional[list] = None,
        box: Optional[List[float]] = None,
        simplify: float = 0.005,
    ) -> Optional[List[float]]:
        """Prompt LibreSAM and return a normalised polygon ``[x1,y1,...]`` or None.

        Prompts (normalised [0,1]): a single ``point=(x,y)``, multiple
        ``points=[(x,y),...]`` with optional ``labels`` (1 positive / 0 negative,
        all refining one mask), or a ``box=[x1,y1,x2,y2]``.
        """
        w_img, h_img = self._size(image_path)
        with self._lock:
            model = self._ensure()
            key = str(image_path)
            if self._cur != key:                  # encode once per image
                model.set_image(key)
                self._cur = key
            if box is not None:
                x1, y1, x2, y2 = box
                r = model.predict(bboxes=[float(x1) * w_img, float(y1) * h_img,
                                          float(x2) * w_img, float(y2) * h_img])
            else:
                if points is None and point is None:
                    raise ValueError("a point, points, or box prompt is required")
                pts = points if points is not None else [point]
                px = [[float(p[0]) * w_img, float(p[1]) * h_img] for p in pts]
                lbls = list(labels) if labels is not None else [1] * len(px)
                r = model.predict(points=[px], labels=[lbls])

        masks = getattr(r, "masks", None)
        polys_px = getattr(masks, "xy", None) if masks is not None else None
        if not polys_px:
            return None

        import cv2
        import numpy as np

        cnt = np.asarray(polys_px[0], dtype=np.float32).reshape(-1, 1, 2)
        if cnt.shape[0] < 3:
            return None
        eps = simplify * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if approx.shape[0] < 3:
            return None
        out: List[float] = []
        for vx, vy in approx:
            out.append(min(max(float(vx) / w_img, 0.0), 1.0))
            out.append(min(max(float(vy) / h_img, 0.0), 1.0))
        return out
