"""Named weights and auto-download for the face-embedding (``embed``) task.

The family ships two ONNX artifacts on the LibreYOLO Hugging Face org:

- ``librefacerec-l.onnx``   — iResNet100 recognition head, 512-d embeddings
  (mirrored single-file from an Apache-2.0 upstream release).
- ``librefacerec-det.onnx`` — lightweight face detector with 5 landmarks
  (MIT-licensed artifact from the OpenCV zoo), used as the default detector.

Any other ArcFace-convention ONNX (aligned 112x112 in, ``(N, D)`` out) can be
used by passing its file path directly (bring-your-own-weights).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_HF_BASE = "https://huggingface.co/LibreYOLO"
_HASH_CHUNK = 1024 * 1024

FACEREC_OFFICIAL_EMBEDDER = {
    "filename": "librefacerec-l.onnx",
    "repo": "LibreYOLO/librefacerec-l",
    "revision": "e8b0e91bf2931579177b9821171d35a759579df6",
    "size_bytes": 260_694_151,
    "sha256": "a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60",
    "upstream": "fal/AuraFace-v1",
    "upstream_revision": "af6d057c9b0ec4071d4c49c80e3539258798b609",
    "license": "Apache-2.0",
}
FACEREC_OFFICIAL_EMBEDDER["url"] = (
    f"{_HF_BASE}/librefacerec-l/resolve/"
    f"{FACEREC_OFFICIAL_EMBEDDER['revision']}/"
    f"{FACEREC_OFFICIAL_EMBEDDER['filename']}"
)

#: Canonical downloadable artifacts: filename -> HF resolve URL.
FACEREC_WEIGHT_URLS = {
    "librefacerec-l.onnx": str(FACEREC_OFFICIAL_EMBEDDER["url"]),
    "librefacerec-det.onnx": f"{_HF_BASE}/librefacerec-det/resolve/main/librefacerec-det.onnx",
}

#: Embedder sizes the factory accepts as ``librefacerec-<size>``.
FACEREC_SIZES = ("l",)


def is_facerec_weight_name(model_path: str) -> bool:
    """True for ``librefacerec-*`` names/paths (with or without ``.onnx``)."""
    return Path(model_path).name.lower().startswith("librefacerec-")


def verify_facerec_weight_file(
    local_path: str | Path,
    source_url: str | None = None,
) -> None:
    """Verify reserved official embedder bytes before they are loaded."""

    path = Path(local_path)
    source_name = Path(str(source_url or "")).name.lower()
    official_name = str(FACEREC_OFFICIAL_EMBEDDER["filename"])
    if path.name.removesuffix(".part").lower() != official_name and (
        source_name != official_name
    ):
        return
    actual_size = path.stat().st_size
    expected_size = int(FACEREC_OFFICIAL_EMBEDDER["size_bytes"])
    if actual_size != expected_size:
        raise ValueError(
            f"{official_name} has the wrong byte length: expected "
            f"{expected_size}, got {actual_size}."
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    actual_hash = digest.hexdigest()
    expected_hash = str(FACEREC_OFFICIAL_EMBEDDER["sha256"])
    if actual_hash != expected_hash:
        raise ValueError(
            f"{official_name} failed SHA-256 verification: expected "
            f"{expected_hash}, got {actual_hash}."
        )


def resolve_facerec_weight(model_path: str) -> str:
    """Resolve a ``librefacerec-*`` name to a local path, downloading if needed.

    Bare names resolve into the standard ``weights/`` directory. Existing
    file paths are returned unchanged.
    """
    path = Path(model_path)
    name = path.name.lower()
    if not name.endswith(".onnx"):
        name += ".onnx"

    if path.exists():
        verify_facerec_weight_file(path)
        return str(path)

    if name not in FACEREC_WEIGHT_URLS:
        known = ", ".join(sorted(FACEREC_WEIGHT_URLS))
        raise FileNotFoundError(
            f"Unknown face-embedding weight '{path.name}'. Known downloadable "
            f"names: {known}. For third-party recognition models, pass the "
            f"path to a local ArcFace-convention ONNX file instead."
        )

    # Bare name (or weights/-prefixed name from the factory resolver).
    dest = path if path.parent != Path(".") else Path("weights") / name
    dest = dest.with_name(name)
    if not dest.exists():
        from ...utils.download import download_url_to_path

        download_url_to_path(
            FACEREC_WEIGHT_URLS[name],
            dest,
            verify=verify_facerec_weight_file,
        )
    verify_facerec_weight_file(dest)
    return str(dest)


def default_face_detector():
    """Build the default face detector, downloading its weights if needed."""
    from .model import OpenCVFaceDetector

    det_path = resolve_facerec_weight("librefacerec-det.onnx")
    return OpenCVFaceDetector(det_path)
