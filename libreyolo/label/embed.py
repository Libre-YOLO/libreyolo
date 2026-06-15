"""Per-image embeddings for LibreLabel: the dataset as a map you can see.

A single CPU forward pass through the user's own (tiny) detector yields a
fixed-length, L2-normalised feature vector per image. Project those to 2-D with
a dependency-free PCA and the whole dataset becomes a scatter you can read:
tight clusters are near-duplicates, lonely points are the rare/hard cases. No
training, no sklearn, no new dependency -- numpy plus the model you already ship.

Like the rest of the assist layer this only *reads* pixels; it writes nothing.
The extraction recipe is grounded in YOLO9's eval-mode forward, which already
returns the ``x8``/``x16``/``x32`` neck maps -- we global-average-pool and
concat them. For other families we fall back to a forward hook on the deepest
backbone module.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "yolo9-t"  # cheapest; resolve_model_name -> LibreYOLO9t.pt
_PREFERRED = ["backbone_spp", "neck_elan_down2", "backbone_dark5"]


class EmbedEngine:
    """Lazy, thread-safe per-image embeddings + a numpy PCA-to-2D projection."""

    def __init__(self, device: str = "auto", model_name: str = DEFAULT_EMBED_MODEL,
                 enabled: bool = True):
        self.device = device
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._lock = threading.Lock()
        self._cache: dict = {}  # str(path) -> np.ndarray (1-D, L2-normalised)

    def available(self) -> bool:
        return bool(self.enabled)

    # -- model -------------------------------------------------------------
    def _ensure(self):
        if self._model is None:
            from libreyolo import LibreYOLO
            from libreyolo.cli.config import resolve_model_name

            weight = resolve_model_name(self.model_name)
            logger.info("LibreLabel embeddings loading %s (%s)", self.model_name, weight)
            self._model = LibreYOLO(weight, device=self.device)
        return self._model

    def _hooked(self, model, tensor):
        """Robust fallback for non-YOLO9 families: GAP the deepest backbone map."""
        import torch

        layers = model._get_available_layers()
        if not layers:
            raise RuntimeError("model exposes no backbone layers to embed")
        name = next((n for n in _PREFERRED if n in layers), sorted(layers)[-1])
        captured: dict = {}

        def hook(_m, _i, output):
            captured["f"] = output[0] if isinstance(output, (tuple, list)) else output

        h = layers[name].register_forward_hook(hook)
        try:
            model._forward(tensor.to(model.device))
        finally:
            h.remove()
        if "f" not in captured:
            raise RuntimeError(f"embedding hook on '{name}' produced no feature map")
        return torch.nn.functional.adaptive_avg_pool2d(captured["f"], 1).flatten(1)

    # -- embedding ---------------------------------------------------------
    def embed_one(self, image_path):
        """One image -> a 1-D float32 numpy vector (L2-normalised, cached)."""
        key = str(image_path)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        import torch

        with self._lock:
            # Re-check under the lock: a concurrent request for the same image
            # may have filled the cache while we waited, so we don't redo the
            # expensive forward pass.
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            model = self._ensure()
            with torch.inference_mode():
                tensor = model._preprocess(key)[0]
                out = model.model(tensor.to(model.device))
                if isinstance(out, dict) and "x32" in out:   # YOLO9 primary path
                    feats = [out["x8"]["features"], out["x16"]["features"],
                             out["x32"]["features"]]
                    vec = torch.cat([f.mean(dim=(2, 3)) for f in feats], dim=1)
                else:                                         # any-family fallback
                    vec = self._hooked(model, tensor)
                vec = torch.nn.functional.normalize(vec, dim=1)
                arr = vec.squeeze(0).cpu().numpy().astype("float32")
            self._cache[key] = arr   # publish under the lock so the re-check above sees it
        return arr

    def embed_dataset(self, session,
                      progress: Optional[Callable[[dict], None]] = None) -> Tuple[List[int], object]:
        """Embed every live image; return ``(ids, features)`` (an ``(N, D)`` array)."""
        import numpy as np

        ids: List[int] = []
        rows: List = []
        total = len(session)
        deleted = getattr(session, "_deleted", set())
        for idx in range(total):
            name = session.image_path(idx).name
            if idx in deleted:
                continue
            try:
                v = self.embed_one(session.image_path(idx))
            except Exception as exc:  # noqa: BLE001 - unreadable image / model issue
                if progress:
                    progress({"type": "progress", "i": idx + 1, "total": total,
                              "id": idx, "name": name, "error": str(exc)})
                continue
            ids.append(idx)
            rows.append(v)
            if progress:
                progress({"type": "progress", "i": idx + 1, "total": total,
                          "id": idx, "name": name})
        feats = np.vstack(rows) if rows else np.empty((0, 0), dtype="float32")
        return ids, feats

    @staticmethod
    def pca_2d(features):
        """``(N, D)`` features -> ``(N, 2)`` projection. NumPy-only, no sklearn."""
        import numpy as np

        X = np.asarray(features, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)
        n = X.shape[0]
        if n == 1:
            return np.zeros((1, 2), dtype=np.float32)
        Xc = X - X.mean(axis=0, keepdims=True)
        _U, _S, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:2]
        if comps.shape[0] < 2:  # D == 1 degenerate
            comps = np.vstack([comps, np.zeros((2 - comps.shape[0], X.shape[1]))])
        return (Xc @ comps.T).astype(np.float32)

    def scatter(self, session,
                progress: Optional[Callable[[dict], None]] = None) -> List[dict]:
        """Compute the 2-D embedding map: ``[{"id", "x", "y"}, ...]`` for live images."""
        ids, feats = self.embed_dataset(session, progress=progress)
        pts = self.pca_2d(feats)
        return [{"id": ids[k], "x": float(pts[k][0]), "y": float(pts[k][1])}
                for k in range(len(ids))]
