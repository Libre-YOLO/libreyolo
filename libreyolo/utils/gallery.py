"""Named reference galleries for the generic ``embed`` task.

A :class:`Gallery` stores L2-normalized reference vectors and matches query
rows with a dense cosine-similarity matrix. Multiple references for one name
remain separate; the name's score is the maximum cosine across its references.
Below-threshold queries remain unknown instead of being assigned the nearest
name.

Galleries are bound to the model that produced them through the embedding
dimension and a weights fingerprint. Matching with another model raises
instead of silently comparing incompatible vector spaces.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

_FINGERPRINT_CHUNK = 1024 * 1024
_GALLERY_FORMAT = "libreyolo-gallery-v1"
_LEGACY_FORMATS = frozenset({"libreyolo-face-gallery-v1"})


def model_file_fingerprint(model_path: str | Path) -> str:
    """Return a stable short SHA-256 fingerprint of a weights file."""
    h = hashlib.sha256()
    with open(model_path, "rb") as file:
        while True:
            chunk = file.read(_FINGERPRINT_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-10:
        raise ValueError("Cannot enroll an all-zero embedding.")
    return vector / norm


def _coalesce_model(model: Any = None, embedder: Any = None) -> Any:
    if model is not None and embedder is not None and model is not embedder:
        raise ValueError("Pass either model= or embedder=, not two different models.")
    return model if model is not None else embedder


def _iter_results(prediction: Any) -> Iterable[Any]:
    if prediction is None:
        return
    if hasattr(prediction, "embeddings"):
        yield prediction
        return
    if isinstance(prediction, (str, bytes, np.ndarray)):
        raise TypeError(
            "Embedding prediction returned an unsupported value instead of "
            "Results or a sequence of Results."
        )
    try:
        iterator = iter(prediction)
    except TypeError as exc:
        raise TypeError(
            "Embedding prediction returned an unsupported value instead of "
            "Results or a sequence of Results."
        ) from exc
    for item in iterator:
        yield from _iter_results(item)


class Gallery:
    """Named reference vectors for identification and retrieval.

    Args:
        model: Optional embed-capable model used by :meth:`enroll` and to bind
            this gallery to its weights. ``embedder=`` remains accepted for
            compatibility with :class:`FaceGallery`.
        embedder: Compatibility alias for ``model``.
    """

    def __init__(self, model: Any = None, *, embedder: Any = None):
        self._names: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._dim: Optional[int] = None
        self._model_fingerprint: Optional[str] = None
        self.embedder = _coalesce_model(model, embedder)
        if self.embedder is not None:
            self._model_fingerprint = _embedder_fingerprint(self.embedder)

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------
    def enroll(
        self,
        name: str,
        sources,
        *,
        model: Any = None,
        embedder: Any = None,
    ) -> int:
        """Embed one or more sources and store every returned row.

        Whole-image embedders contribute one row per source. Region embedders
        contribute every region row returned for a source. The return value is
        the number of reference rows added.
        """
        selected = _coalesce_model(model, embedder) or self.embedder
        if selected is None:
            raise ValueError(
                "Gallery.enroll needs a model: construct Gallery(model), pass "
                "model=, or use enroll_embedding() for precomputed vectors."
            )
        self._bind_or_check_model(selected)

        source_items = (
            list(sources) if isinstance(sources, (list, tuple)) else [sources]
        )
        added = 0
        for source in source_items:
            predict = getattr(selected, "predict", None)
            prediction = predict(source) if callable(predict) else selected(source)
            source_rows = 0
            for result in _iter_results(prediction):
                embeddings = getattr(result, "embeddings", None)
                if embeddings is None or len(embeddings) == 0:
                    continue
                data = np.asarray(_to_numpy(embeddings.data), dtype=np.float32)
                if data.ndim == 1:
                    data = data[None, :]
                if data.ndim != 2:
                    raise ValueError(
                        "Embedding predictions must have shape (N, D); "
                        f"got {tuple(data.shape)}."
                    )
                for vector in data:
                    self.enroll_embedding(name, vector)
                    source_rows += 1
            if source_rows == 0:
                raise ValueError(
                    f"No embeddings found in enrollment source: {source!r}"
                )
            added += source_rows
        return added

    def enroll_embedding(self, name: str, vector) -> None:
        """Enroll one named reference from a precomputed embedding."""
        vec = _unit(_to_numpy(vector))
        if self._dim is None:
            self._dim = int(vec.shape[0])
        elif int(vec.shape[0]) != self._dim:
            raise ValueError(
                f"Embedding dim mismatch: gallery holds {self._dim}-d vectors, "
                f"got {int(vec.shape[0])}-d."
            )
        self._names.append(str(name))
        self._vectors.append(vec)

    def remove(self, name: str) -> int:
        """Remove every reference with ``name`` and return the number removed."""
        keep = [(n, v) for n, v in zip(self._names, self._vectors) if n != name]
        removed = len(self._names) - len(keep)
        self._names = [n for n, _ in keep]
        self._vectors = [v for _, v in keep]
        return removed

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def identities(self) -> List[str]:
        """Unique enrolled names in first-enrolled order."""
        seen: Dict[str, None] = {}
        for name in self._names:
            seen.setdefault(name)
        return list(seen)

    @property
    def dim(self) -> Optional[int]:
        """Embedding dimension, or ``None`` before the first enrollment."""
        return self._dim

    def __len__(self) -> int:
        return len(self.identities)

    def __contains__(self, name: str) -> bool:
        return name in self._names

    def __repr__(self) -> str:
        return (
            f"Gallery(identities={len(self)}, references={len(self._names)}, "
            f"dim={self._dim})"
        )

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def match(
        self,
        embeddings,
        *,
        top_k: int = 1,
        threshold: float = 0.4,
        model: Any = None,
    ) -> List[List[Tuple[str, float]]]:
        """Match query rows against the gallery using max-cosine per name."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")
        if model is not None:
            self._bind_or_check_model(model)
        data = embeddings.data if hasattr(embeddings, "data") else embeddings
        queries = np.asarray(_to_numpy(data), dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        if queries.ndim != 2:
            raise ValueError(
                f"Query embeddings must have shape (N, D); got {tuple(queries.shape)}."
            )
        if self._dim is not None and queries.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dim mismatch: gallery holds {self._dim}-d vectors, "
                f"queries are {queries.shape[1]}-d. Was this gallery built "
                "with a different embedding model?"
            )
        if not self._names or queries.shape[0] == 0:
            return [[] for _ in range(queries.shape[0])]

        refs = np.stack(self._vectors)
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.clip(norms, 1e-10, None)
        similarities = queries @ refs.T

        reference_names = np.asarray(self._names)
        output: List[List[Tuple[str, float]]] = []
        for row in similarities:
            best: Dict[str, float] = {}
            for name, score in zip(reference_names, row):
                value = float(score)
                if value > best.get(name, -2.0):
                    best[name] = value
            ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
            output.append(
                [(name, score) for name, score in ranked[:top_k] if score >= threshold]
            )
        return output

    def identify(
        self,
        embeddings,
        *,
        threshold: float = 0.4,
        model: Any = None,
    ):
        """Return row-aligned names/scores while preserving unknown outcomes."""
        from .results import Identities

        if len(self) == 0:
            raise ValueError(
                "Cannot identify against an empty FaceGallery/Gallery; enroll "
                "at least one reference first."
            )
        matches = self.match(
            embeddings,
            top_k=1,
            threshold=-1.0,
            model=model,
        )
        names: List[Optional[str]] = []
        scores: List[float] = []
        for match in matches:
            best_name, best_score = match[0] if match else (None, float("nan"))
            names.append(best_name if best_score >= threshold else None)
            scores.append(best_score)
        return Identities(names, np.asarray(scores, dtype=np.float32))

    def _bind_or_check_model(self, model: Any) -> None:
        fingerprint = _embedder_fingerprint(model)
        if self._model_fingerprint is None:
            self._model_fingerprint = fingerprint
            return
        if fingerprint is None:
            raise ValueError(
                "This gallery is bound to a weights fingerprint, but the supplied "
                "model has no fingerprintable weights."
            )
        if fingerprint != self._model_fingerprint:
            raise ValueError(
                "This gallery was built with a different embedding model "
                f"(gallery fingerprint {self._model_fingerprint}, model "
                f"fingerprint {fingerprint}). Re-enroll with the current model "
                "or load the matching gallery."
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> str:
        """Save vectors, names, dimension, and model fingerprint to ``.npz``."""
        path = Path(path)
        if not self._names:
            raise ValueError("Cannot save an empty gallery.")
        metadata = {
            "format": _GALLERY_FORMAT,
            "dim": self._dim,
            "model_fingerprint": self._model_fingerprint,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=np.stack(self._vectors),
            names=np.asarray(self._names),
            meta=np.asarray(json.dumps(metadata)),
        )
        return str(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        model: Any = None,
        embedder: Any = None,
    ) -> "Gallery":
        """Load a generic or legacy face-gallery archive."""
        with np.load(Path(path), allow_pickle=False) as archive:
            try:
                metadata = json.loads(str(archive["meta"]))
                vectors = archive["vectors"]
                names = archive["names"]
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Not a LibreYOLO gallery: {path}") from exc

        format_name = metadata.get("format")
        if format_name != _GALLERY_FORMAT and format_name not in _LEGACY_FORMATS:
            raise ValueError(f"Not a LibreYOLO gallery: {path}")
        if vectors.ndim != 2 or len(names) != vectors.shape[0]:
            raise ValueError(f"Corrupt LibreYOLO gallery payload: {path}")

        gallery = cls()
        gallery._model_fingerprint = metadata.get("model_fingerprint")
        for name, vector in zip(names.tolist(), vectors):
            gallery.enroll_embedding(str(name), vector)
        saved_dim = metadata.get("dim")
        if saved_dim is not None and int(saved_dim) != gallery.dim:
            raise ValueError(
                f"Corrupt LibreYOLO gallery dimension: metadata says {saved_dim}, "
                f"vectors are {gallery.dim}-d."
            )

        selected = _coalesce_model(model, embedder)
        if selected is not None:
            gallery.embedder = selected
            gallery._bind_or_check_model(selected)
        return gallery


# Permanent source compatibility for the original face-specific API.
FaceGallery = Gallery


def _to_numpy(data):
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _state_dict_fingerprint(model: Any) -> Optional[str]:
    module = getattr(model, "model", model)
    state_dict = getattr(module, "state_dict", None)
    if not callable(state_dict):
        return None
    try:
        state = state_dict()
    except Exception:
        return None
    if not isinstance(state, dict) or not state:
        return None

    h = hashlib.sha256()
    hashed = 0
    for name in sorted(state):
        value = state[name]
        if not hasattr(value, "detach"):
            continue
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        if tensor.numel():
            byte_view = tensor.reshape(-1).view(torch.uint8)
            h.update(byte_view.numpy().tobytes())
        hashed += 1
    return h.hexdigest()[:16] if hashed else None


def _embedder_fingerprint(model: Any) -> Optional[str]:
    """Best-effort weights fingerprint, cached on the model wrapper."""
    cached = getattr(model, "_weights_fingerprint", None)
    if cached is not None:
        return str(cached)
    model_path = getattr(model, "model_path", None)
    if model_path and Path(model_path).is_file():
        fingerprint = model_file_fingerprint(model_path)
    else:
        fingerprint = _state_dict_fingerprint(model)
    if fingerprint is not None:
        try:
            model._weights_fingerprint = fingerprint
        except (AttributeError, TypeError):
            pass
    return fingerprint


__all__ = ["Gallery", "FaceGallery", "model_file_fingerprint"]
