"""Identity gallery for 1:N face identification (``embed`` task).

A :class:`FaceGallery` holds named reference embeddings. Identification is a
cosine match of query embeddings against every reference: per-reference
storage with max-cosine scoring (enrolling K images of one person keeps K
vectors; a person's score is the best of their references). Below-threshold
queries resolve to *unknown* (``None``), never to the nearest wrong person.

Galleries are bound to the embedder that produced them: ``save()`` records
the embedding dimension and a fingerprint of the model file, and matching
with a different model raises instead of silently comparing incompatible
vector spaces.

Scale note: matching is a single dense matmul, comfortably microseconds for
thousands of identities. Larger deployments should export raw embeddings
(``results.embeddings``) to a dedicated vector store instead.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_FINGERPRINT_CHUNK = 1024 * 1024


def model_file_fingerprint(model_path: str | Path) -> str:
    """Stable fingerprint of an ONNX file or complete Core ML package."""

    path = Path(model_path)
    if path.is_symlink():
        raise ValueError(
            f"Face model artifacts must not be symbolic links: {path}"
        )
    if path.is_dir():
        if path.suffix.lower() != ".mlpackage":
            raise ValueError(
                "Face model directories must be Core ML .mlpackage "
                f"artifacts, got {path}."
            )
        return _directory_fingerprint(path)
    if path.suffix.lower() == ".onnx":
        from ...export.coreml_facerec import (
            facerec_onnx_source_manifest,
        )

        digest, _ = facerec_onnx_source_manifest(path)
        return digest[:16]
    if not path.is_file():
        raise FileNotFoundError(f"Face model artifact does not exist: {path}")

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_FINGERPRINT_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def _directory_fingerprint(root: Path) -> str:
    files = []
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise ValueError(
                "Face Core ML packages must not contain symbolic links: "
                f"{entry}"
            )
        if entry.is_file():
            files.append(entry)
        elif not entry.is_dir():
            raise ValueError(
                f"Face Core ML package contains a special entry: {entry}"
            )
    if not files:
        raise ValueError(f"Face Core ML package is empty: {root}")

    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(_FINGERPRINT_CHUNK):
                digest.update(chunk)
    return digest.hexdigest()[:16]


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-10:
        raise ValueError("Cannot enroll an all-zero embedding.")
    return vec / norm


class FaceGallery:
    """Named reference embeddings for face identification.

    Args:
        embedder: Optional ``LibreFaceEmbedder`` used by :meth:`enroll` to
            embed reference images and to stamp the gallery's model
            fingerprint. Precomputed vectors can always be added with
            :meth:`enroll_embedding`, embedder or not.
    """

    def __init__(self, embedder: Any = None):
        self._names: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._dim: Optional[int] = None
        self._model_fingerprint: Optional[str] = None
        self.embedder = embedder
        if embedder is not None:
            self._model_fingerprint = _embedder_fingerprint(embedder)

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------
    def enroll(self, name: str, sources, *, embedder: Any = None) -> int:
        """Enroll one identity from one or more reference images.

        ``sources`` is an image or a sequence of images (paths, URLs, arrays,
        PIL images). Each image contributes the embedding of its most
        prominent face. Returns the number of references added.
        """
        model = embedder or self.embedder
        if model is None:
            raise ValueError(
                "FaceGallery.enroll needs an embedder: construct the gallery "
                "with FaceGallery(embedder=model) or pass embedder=model. For "
                "precomputed vectors use enroll_embedding()."
            )
        # A gallery is only meaningful within one embedding space, so an
        # enrolment through a different model has to fail here rather than
        # append a vector that silently will not compare.
        if self._model_fingerprint is None:
            self._model_fingerprint = _embedder_fingerprint(model)
        else:
            self._check_model(model)

        if isinstance(sources, (str, Path)) or not isinstance(sources, Sequence):
            sources = [sources]

        added = 0
        for source in sources:
            result = model(source)
            emb = result.embeddings
            if emb is None or len(emb) == 0:
                raise ValueError(f"No face found in enrollment image: {source!r}")
            data = np.asarray(_to_numpy(emb.data), dtype=np.float32)
            idx = 0
            if result.boxes is not None and len(result.boxes) == len(data):
                conf = np.asarray(_to_numpy(result.boxes.conf), dtype=np.float32)
                if conf.size:
                    idx = int(np.argmax(conf))
            self.enroll_embedding(name, data[idx])
            added += 1
        return added

    def enroll_embedding(self, name: str, vector) -> None:
        """Enroll one identity reference from a precomputed embedding."""
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
        """Remove every reference of an identity. Returns how many."""
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
        """Unique enrolled identity names, in first-enrolled order."""
        seen: Dict[str, None] = {}
        for n in self._names:
            seen.setdefault(n)
        return list(seen)

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    def __len__(self) -> int:
        return len(self.identities)

    def __contains__(self, name: str) -> bool:
        return name in self._names

    def __repr__(self) -> str:
        return (
            f"FaceGallery(identities={len(self)}, references={len(self._names)}, "
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
        """Match query embeddings against the gallery.

        Accepts an ``Embeddings`` payload or an ``(N, D)`` array. Returns, per
        query row, up to ``top_k`` ``(name, score)`` pairs with score >=
        ``threshold``, best first. Scores are max-cosine over each identity's
        references.
        """
        if model is not None:
            self._check_model(model)
        data = embeddings.data if hasattr(embeddings, "data") else embeddings
        queries = np.asarray(_to_numpy(data), dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        if not self._names or queries.size == 0:
            return [[] for _ in range(queries.shape[0])]
        if self._dim is not None and queries.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dim mismatch: gallery holds {self._dim}-d vectors, "
                f"queries are {queries.shape[1]}-d. Was this gallery built "
                f"with a different embedding model?"
            )

        refs = np.stack(self._vectors)  # (R, D), unit rows
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.clip(norms, 1e-10, None)
        sims = queries @ refs.T  # (N, R)

        names = np.asarray(self._names)
        out: List[List[Tuple[str, float]]] = []
        for row in sims:
            best: Dict[str, float] = {}
            for name, s in zip(names, row):
                s = float(s)
                if s > best.get(name, -2.0):
                    best[name] = s
            ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
            out.append([(n, s) for n, s in ranked[:top_k] if s >= threshold])
        return out

    def _check_model(self, model: Any) -> None:
        fp = _embedder_fingerprint(model)
        if (
            fp is not None
            and self._model_fingerprint is not None
            and fp != self._model_fingerprint
        ):
            raise ValueError(
                "This gallery was built with a different embedding model "
                f"(gallery fingerprint {self._model_fingerprint}, model "
                f"fingerprint {fp}). Re-enroll with the current model or load "
                "the matching gallery."
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> str:
        """Save to an ``.npz`` archive (vectors + names + metadata)."""
        path = Path(path)
        if not self._names:
            raise ValueError("Cannot save an empty gallery.")
        meta = {
            "format": "libreyolo-face-gallery-v1",
            "dim": self._dim,
            "model_fingerprint": self._model_fingerprint,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=np.stack(self._vectors),
            names=np.asarray(self._names),
            meta=np.asarray(json.dumps(meta)),
        )
        return str(path)

    @classmethod
    def load(cls, path: str | Path, *, embedder: Any = None) -> "FaceGallery":
        """Load a gallery previously written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            try:
                meta = json.loads(str(archive["meta"]))
                vectors = archive["vectors"]
                names = archive["names"]
            except KeyError as e:
                raise ValueError(f"Not a LibreYOLO face gallery: {path}") from e
        if meta.get("format") != "libreyolo-face-gallery-v1":
            raise ValueError(f"Not a LibreYOLO face gallery: {path}")

        gallery = cls()
        gallery._model_fingerprint = meta.get("model_fingerprint")
        for name, vec in zip(names.tolist(), vectors):
            gallery.enroll_embedding(str(name), vec)
        if embedder is not None:
            gallery.embedder = embedder
            gallery._check_model(embedder)
        return gallery


def _to_numpy(data):
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _embedder_fingerprint(model: Any) -> Optional[str]:
    """Best-effort fingerprint of an embedder's weight file (cached on it)."""
    cached = getattr(model, "_weights_fingerprint", None)
    if cached is not None:
        return cached
    model_path = getattr(model, "model_path", None)
    if not model_path or not Path(model_path).exists():
        return None
    fp = model_file_fingerprint(model_path)
    try:
        model._weights_fingerprint = fp
    except AttributeError:
        pass
    return fp
