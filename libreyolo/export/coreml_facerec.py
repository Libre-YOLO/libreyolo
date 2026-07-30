"""Opaque-ONNX to Core ML export for the ``facerec/embed`` component.

LibreFaceEmbedder consumes a recognition head as an opaque ONNX graph.  This
module keeps that provenance boundary intact: it asks the Apache-2.0
``onnx2torch`` converter for an equivalent PyTorch graph, checks the converted
graph numerically against the already-loaded ONNX Runtime session, and then
hands the checked TorchScript graph to Apple's Core ML converter.

No third-party recognition architecture source is copied or reconstructed in
LibreYOLO.  The resulting package accepts the exact model-ready aligned-face
tensor produced by ``PreprocCfg`` and emits one raw embedding.  Alignment and
L2 normalization remain host operations, matching the native ONNX path.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..models.facerec.model import LibreFaceEmbedder
    from ..models.facerec.preprocess import PreprocCfg

FACEREC_COREML_CONTRACT = "aligned_face_embedding_component_v1"
FACEREC_COREML_INPUT_NAME = "aligned_face"
FACEREC_COREML_OUTPUT_NAME = "embedding"
FACEREC_COREML_PREPROCESS_KEY = "facerec_preprocess_json"
FACEREC_COREML_PREPROCESS_HASH_KEY = "facerec_preprocess_sha256"
FACEREC_COREML_SOURCE_HASH_KEY = "facerec_source_manifest_sha256"
FACEREC_COREML_SOURCE_MANIFEST_KEY = "facerec_source_manifest_json"
FACEREC_COREML_ARTIFACT_SCOPE = "host_aligned_face_embedding_component"
FACEREC_COREML_GEOMETRY = "host_aligned_face"
FACEREC_COREML_REQUIRED_COMPUTE_UNITS = "cpu_only"

_FACEREC_SOURCE_MANIFEST_DOMAIN = (
    b"libreyolo.facerec.onnx-source-manifest.v1"
)
_ONNX_EXTERNAL_DATA_KEYS = frozenset(
    {"location", "offset", "length", "checksum", "basepath"}
)
_ONNX2TORCH_VERSION = "1.5.15"
_ONNX_TENSOR_FLOAT = 1
_SHA256_HEX_LENGTH = 64
_OFFICIAL_PREPROCESS = {
    "size": 112,
    "color_order": "RGB",
    "mean": 127.5,
    "scale": 1.0 / 127.5,
    "layout": "NCHW",
}


class _FaceEmbeddingOutput(nn.Module):
    """Pin an opaque converted graph to one FP32 embedding tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, aligned_face: torch.Tensor) -> torch.Tensor:
        output = self.model(aligned_face)
        if isinstance(output, (tuple, list)):
            if len(output) != 1:
                raise RuntimeError(
                    "Face-embedding ONNX conversion returned multiple outputs."
                )
            output = output[0]
        if not torch.is_tensor(output):
            raise RuntimeError(
                "Face-embedding ONNX conversion must return one tensor."
            )
        return output.float()


def _preprocess_payload(cfg: "PreprocCfg") -> dict[str, Any]:
    size = int(cfg.size)
    color = str(cfg.color_order).strip().upper()
    layout = str(cfg.layout).strip().upper()
    mean = float(cfg.mean)
    scale = float(cfg.scale)
    if size <= 0:
        raise ValueError("Face-embedding preprocessing size must be positive.")
    if color not in {"RGB", "BGR"}:
        raise ValueError(
            "Face-embedding preprocessing color_order must be RGB or BGR."
        )
    if layout not in {"NCHW", "NHWC"}:
        raise ValueError(
            "Face-embedding preprocessing layout must be NCHW or NHWC."
        )
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Face-embedding preprocessing mean/scale must be finite and "
            "scale must be positive."
        )
    return {
        "size": size,
        "color_order": color,
        "mean": mean,
        "scale": scale,
        "layout": layout,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def facerec_coreml_preprocess_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(payload)).encode("utf-8")).hexdigest()


def _strict_positive_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Face Core ML metadata {key!r} must be an integer.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise ValueError(f"Face Core ML metadata {key!r} must be an integer.")
    if parsed <= 0:
        raise ValueError(f"Face Core ML metadata {key!r} must be positive.")
    return parsed


def _strict_false(value: Any, *, key: str) -> bool:
    if value is False or (
        isinstance(value, str) and value in {"False", "false"}
    ):
        return False
    raise ValueError(f"Face Core ML metadata {key!r} must be false.")


def _metadata_json(
    metadata: Mapping[str, Any],
    key: str,
) -> Any:
    value = metadata.get(key)
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Face Core ML metadata {key!r} must be valid JSON."
        ) from exc


def _same_json_contract(actual: Any, expected: Any) -> bool:
    """Compare JSON-like values without bool/int or int/float coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_json_contract(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json_contract(left, right)
            for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _official_embedder_spec() -> dict[str, Any]:
    from ..models.facerec.weights import FACEREC_OFFICIAL_EMBEDDER

    required = {
        "filename",
        "repo",
        "revision",
        "url",
        "size_bytes",
        "sha256",
        "upstream",
        "upstream_revision",
        "license",
    }
    spec = dict(FACEREC_OFFICIAL_EMBEDDER)
    if set(spec) != required:
        raise RuntimeError(
            "The official face-embedding source specification changed: "
            f"expected {sorted(required)}, got {sorted(spec)}."
        )
    return spec


def _official_provenance_metadata() -> dict[str, Any]:
    spec = _official_embedder_spec()
    return {
        "facerec_source_filename": spec["filename"],
        "facerec_source_repo": spec["repo"],
        "facerec_source_revision": spec["revision"],
        "facerec_source_url": spec["url"],
        "facerec_source_size_bytes": spec["size_bytes"],
        "facerec_source_sha256": spec["sha256"],
        "facerec_source_upstream": spec["upstream"],
        "facerec_source_upstream_revision": spec["upstream_revision"],
        "facerec_source_license": spec["license"],
        # Retain the established mirror labels while binding them to the same
        # shared official source specification.
        "facerec_source_mirror": spec["repo"],
        "facerec_source_mirror_revision": spec["revision"],
    }


def _require_official_preprocess(preprocess: Mapping[str, Any]) -> None:
    if not _same_json_contract(dict(preprocess), _OFFICIAL_PREPROCESS):
        raise ValueError(
            "Official size-l face-recognition weights require the exact "
            "112x112 ArcFace RGB preprocessing contract."
        )


def validate_facerec_coreml_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete host-visible aligned-face component contract."""
    expected = {
        "libreyolo_producer": "libreyolo",
        "artifact_format": "coreml",
        "coreml_io_schema_version": "2",
        "model_family": "facerec",
        "task": "embed",
        "facerec_contract": FACEREC_COREML_CONTRACT,
        "artifact_scope": FACEREC_COREML_ARTIFACT_SCOPE,
    }
    for key, value in expected.items():
        actual = str(metadata.get(key, "")).strip()
        if actual != value:
            raise ValueError(
                f"Face Core ML metadata {key!r} must be {value!r}, "
                f"got {actual!r}."
            )

    from ..utils.serialization import SCHEMA_VERSION

    if str(metadata.get("schema_version", "")).strip() != SCHEMA_VERSION:
        raise ValueError(
            f"Face Core ML schema_version must be {SCHEMA_VERSION!r}."
        )

    if _metadata_json(metadata, "supported_tasks") != ["embed"]:
        raise ValueError(
            "Face Core ML supported_tasks must be exactly ['embed']."
        )
    if str(metadata.get("default_task", "")).strip() != "embed":
        raise ValueError("Face Core ML default_task must be 'embed'.")
    if _metadata_json(metadata, "names") != {"0": "face"}:
        raise ValueError(
            "Face Core ML class-name metadata must be exactly {'0': 'face'}."
        )
    if _strict_positive_int(metadata.get("nc"), key="nc") != 1:
        raise ValueError("Face Core ML metadata nc must be 1.")
    if _strict_positive_int(metadata.get("nb_classes"), key="nb_classes") != 1:
        raise ValueError("Face Core ML metadata nb_classes must be 1.")

    raw = metadata.get(FACEREC_COREML_PREPROCESS_KEY)
    try:
        payload = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Face Core ML preprocessing metadata must be valid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Face Core ML preprocessing metadata must be an object.")

    required = {"size", "color_order", "mean", "scale", "layout"}
    if set(payload) != required:
        raise ValueError(
            "Face Core ML preprocessing metadata fields changed: expected "
            f"{sorted(required)}, got {sorted(payload)}."
        )
    if isinstance(payload["size"], bool) or not isinstance(payload["size"], int):
        raise ValueError(
            "Face Core ML preprocessing size must be a JSON integer."
        )
    size = int(payload["size"])
    if size <= 0:
        raise ValueError("Face Core ML preprocessing size must be positive.")
    if not isinstance(payload["color_order"], str) or not isinstance(
        payload["layout"], str
    ):
        raise ValueError(
            "Face Core ML preprocessing color/layout must be JSON strings."
        )
    color = payload["color_order"].strip().upper()
    layout = payload["layout"].strip().upper()
    if any(
        isinstance(payload[key], bool)
        or not isinstance(payload[key], (int, float))
        for key in ("mean", "scale")
    ):
        raise ValueError(
            "Face Core ML preprocessing mean/scale must be JSON numbers."
        )
    mean = float(payload["mean"])
    scale = float(payload["scale"])
    normalized = {
        "size": size,
        "color_order": color,
        "mean": mean,
        "scale": scale,
        "layout": layout,
    }
    if not _same_json_contract(payload, normalized):
        raise ValueError(
            "Face Core ML preprocessing JSON must use its canonical field "
            "types and normalized color/layout values."
        )
    # Reuse the same invariants as export without importing the face model.
    if color not in {"RGB", "BGR"} or layout not in {"NCHW", "NHWC"}:
        raise ValueError("Face Core ML preprocessing color/layout is invalid.")
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Face Core ML preprocessing mean/scale is invalid.")
    declared_hash = str(metadata.get(FACEREC_COREML_PREPROCESS_HASH_KEY, ""))
    actual_hash = facerec_coreml_preprocess_hash(normalized)
    if declared_hash != actual_hash:
        raise ValueError(
            "Face Core ML preprocessing hash does not match its serialized "
            "contract."
        )

    for key in ("imgsz", "imgsz_h", "imgsz_w"):
        declared_size = _strict_positive_int(metadata.get(key), key=key)
        if declared_size != size:
            raise ValueError(
                f"Face Core ML metadata {key!r} must match preprocessing "
                f"size {size}, got {declared_size}."
            )
    precision = str(metadata.get("precision", "")).strip().lower()
    if precision != "fp32":
        raise ValueError(
            "Face Core ML precision must be exactly 'fp32'."
        )
    required_compute_units = str(
        metadata.get("coreml_required_compute_units", "")
    ).strip().lower()
    if required_compute_units != FACEREC_COREML_REQUIRED_COMPUTE_UNITS:
        raise ValueError(
            "Face Core ML required compute units must be exactly "
            f"{FACEREC_COREML_REQUIRED_COMPUTE_UNITS!r}."
        )
    _strict_false(metadata.get("dynamic"), key="dynamic")

    source_hash = str(metadata.get(FACEREC_COREML_SOURCE_HASH_KEY, ""))
    if not _is_lower_sha256(source_hash):
        raise ValueError(
            "Face Core ML source manifest SHA-256 must be 64 lowercase hex "
            "characters."
        )
    manifest = _metadata_json(metadata, FACEREC_COREML_SOURCE_MANIFEST_KEY)
    source_entries = _validate_source_manifest_entries(manifest)
    actual_source_hash = _source_manifest_hash(source_entries)
    if source_hash != actual_source_hash:
        raise ValueError(
            "Face Core ML source manifest hash does not match its canonical "
            "length-framed manifest."
        )

    embedding_dim = _strict_positive_int(
        metadata.get("facerec_embedding_dim"),
        key="facerec_embedding_dim",
    )
    try:
        conversion_error = float(
            metadata.get("facerec_onnx_to_torch_max_abs_error")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Face Core ML conversion error metadata must be numeric."
        ) from exc
    if not math.isfinite(conversion_error) or conversion_error < 0:
        raise ValueError(
            "Face Core ML conversion error metadata must be finite and "
            "non-negative."
        )

    expected_io = {
        "input": {
            "name": FACEREC_COREML_INPUT_NAME,
            "kind": "tensor",
            "layout": layout,
            "color": color.lower(),
            "range": "standardized",
            "mean": [mean / 255.0] * 3,
            "std": [1.0 / (scale * 255.0)] * 3,
            "geometry": FACEREC_COREML_GEOMETRY,
            "interpolation": "bilinear",
            "resize_backend": "opencv",
            "pad_value": 0,
            "shape_mode": "fixed",
        },
        "validation": {
            "color": color.lower(),
            "range": "standardized",
            "mean": [mean / 255.0] * 3,
            "std": [1.0 / (scale * 255.0)] * 3,
        },
        "outputs": [
            {
                "name": FACEREC_COREML_OUTPUT_NAME,
                "role": "embedding",
                "encoding": "raw_identity_embedding",
                "rank": 2,
                "dtype": "float32",
                "shape": [1, embedding_dim],
            }
        ],
    }
    declared_io = _metadata_json(metadata, "coreml_io")
    if not _same_json_contract(declared_io, expected_io):
        raise ValueError(
            "Face Core ML IO metadata disagrees with the aligned-face "
            "component contract."
        )
    if _metadata_json(metadata, "coreml_output_names") != [
        FACEREC_COREML_OUTPUT_NAME
    ]:
        raise ValueError(
            "Face Core ML output-name metadata was modified."
        )

    size_name = str(metadata.get("size", "")).strip().lower()
    model_size = str(metadata.get("model_size", "")).strip().lower()
    if size_name != model_size or size_name not in {"l", "custom"}:
        raise ValueError(
            "Face Core ML size/model_size must agree and be 'l' or 'custom'."
        )
    provenance = _official_provenance_metadata()
    if size_name == "l":
        _require_official_preprocess(normalized)
        spec = _official_embedder_spec()
        expected_entry = {
            "path": str(spec["filename"]),
            "kind": "onnx",
            "bytes": int(spec["size_bytes"]),
            "sha256": str(spec["sha256"]),
        }
        if not _same_json_contract(source_entries, [expected_entry]):
            raise ValueError(
                "Official size-l Face Core ML metadata must bind the exact "
                "single-file pinned ONNX artifact."
            )
        for key, expected_value in provenance.items():
            if str(metadata.get(key, "")).strip() != str(expected_value):
                raise ValueError(
                    "Official size-l Face Core ML provenance field "
                    f"{key!r} was modified."
                )
    elif any(str(metadata.get(key, "")).strip() for key in provenance):
        raise ValueError(
            "Custom Face Core ML artifacts must not claim official size-l "
            "provenance."
        )

    return {
        "preprocess": normalized,
        "embedding_dim": embedding_dim,
        "source_manifest_sha256": source_hash,
        "source_manifest": source_entries,
        "precision": precision,
        "required_compute_units": required_compute_units,
        "size": size_name,
    }


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _framed(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big", signed=False) + value


def _source_manifest_hash(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(_framed(_FACEREC_SOURCE_MANIFEST_DOMAIN))
    digest.update(len(entries).to_bytes(8, "big", signed=False))
    for entry in entries:
        digest.update(_framed(str(entry["path"]).encode("utf-8")))
        digest.update(_framed(str(entry["kind"]).encode("ascii")))
        digest.update(int(entry["bytes"]).to_bytes(8, "big", signed=False))
        digest.update(_framed(bytes.fromhex(str(entry["sha256"]))))
    return digest.hexdigest()


def _canonical_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty path string.")
    if "\\" in value or ":" in value:
        raise ValueError(f"{label} must use a portable relative POSIX path.")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or value.startswith(("/", "\\"))
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
        or posix.as_posix() != value
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path.")
    return value


def _validate_source_manifest_entries(
    value: Any,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            "Face Core ML source manifest must be a non-empty JSON array."
        )
    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Face Core ML source manifest entry {index} must be an object."
            )
        required = {"path", "kind", "bytes", "sha256"}
        if set(entry) != required:
            raise ValueError(
                "Face Core ML source manifest fields changed: expected "
                f"{sorted(required)}, got {sorted(entry)}."
            )
        path = _canonical_relative_path(
            entry["path"],
            label=f"Face Core ML source manifest entry {index} path",
        )
        folded = path.casefold()
        if folded in seen_paths:
            raise ValueError(
                "Face Core ML source manifest paths must be unique "
                "case-insensitively."
            )
        seen_paths.add(folded)
        kind = entry["kind"]
        if kind not in {"onnx", "external_data"}:
            raise ValueError(
                "Face Core ML source manifest kind must be 'onnx' or "
                "'external_data'."
            )
        byte_count = entry["bytes"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int):
            raise ValueError(
                "Face Core ML source manifest byte counts must be JSON integers."
            )
        if byte_count < 0 or (kind == "onnx" and byte_count == 0):
            raise ValueError(
                "Face Core ML source manifest byte counts are invalid."
            )
        sha256 = entry["sha256"]
        if not _is_lower_sha256(sha256):
            raise ValueError(
                "Face Core ML source manifest hashes must be 64 lowercase "
                "hex characters."
            )
        normalized.append(
            {
                "path": path,
                "kind": kind,
                "bytes": byte_count,
                "sha256": sha256,
            }
        )
    if normalized != sorted(normalized, key=lambda item: item["path"]):
        raise ValueError(
            "Face Core ML source manifest entries must be sorted by path."
        )
    onnx_entries = [entry for entry in normalized if entry["kind"] == "onnx"]
    if len(onnx_entries) != 1 or "/" in onnx_entries[0]["path"]:
        raise ValueError(
            "Face Core ML source manifest must contain exactly one root ONNX "
            "protobuf entry."
        )
    return normalized


def _iter_protobuf_tensors(message: Any) -> Iterator[Any]:
    """Yield TensorProto messages through every protobuf message field."""
    descriptor = getattr(message, "DESCRIPTOR", None)
    if descriptor is None:
        return
    if str(getattr(descriptor, "full_name", "")) == "onnx.TensorProto":
        yield message
        return
    for field, value in message.ListFields():
        if getattr(field, "message_type", None) is None:
            continue
        repeated_marker = getattr(field, "is_repeated", None)
        is_repeated = (
            bool(repeated_marker)
            if repeated_marker is not None
            else int(getattr(field, "label", 0))
            == int(getattr(field, "LABEL_REPEATED", 3))
        )
        if is_repeated:
            for item in value:
                yield from _iter_protobuf_tensors(item)
        else:
            yield from _iter_protobuf_tensors(value)


def _external_tensor_locations(model_proto: Any) -> list[str]:
    locations: set[str] = set()
    for tensor in _iter_protobuf_tensors(model_proto):
        pairs = [
            (str(item.key), str(item.value))
            for item in getattr(tensor, "external_data", ())
        ]
        data_location = int(getattr(tensor, "data_location", 0) or 0)
        if not pairs and data_location != 1:
            continue
        if data_location != 1:
            raise ValueError(
                "Face-embedding ONNX tensors with external_data must declare "
                "data_location=EXTERNAL."
            )
        if bytes(getattr(tensor, "raw_data", b"")):
            raise ValueError(
                "Face-embedding ONNX tensors must not mix inline raw_data "
                "with external_data."
            )
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "Face-embedding ONNX external_data keys must be unique."
            )
        unknown = set(keys) - _ONNX_EXTERNAL_DATA_KEYS
        if unknown:
            raise ValueError(
                "Face-embedding ONNX external_data contains unsupported keys: "
                + ", ".join(sorted(unknown))
            )
        values = dict(pairs)
        location = values.get("location", "")
        _canonical_relative_path(
            location,
            label="Face-embedding ONNX external_data location",
        )
        if values.get("basepath", ""):
            raise ValueError(
                "Face-embedding ONNX external_data must not override basepath."
            )
        for key in ("offset", "length"):
            raw = values.get(key)
            if raw is None:
                continue
            if (
                not raw.isdigit()
                or raw != str(int(raw))
                or (key == "length" and int(raw) <= 0)
            ):
                qualifier = "positive" if key == "length" else "non-negative"
                raise ValueError(
                    "Face-embedding ONNX external_data "
                    f"{key} must be a canonical {qualifier} decimal integer."
                )
        locations.add(location)
    return sorted(locations)


def _safe_external_path(root: Path, location: str) -> Path:
    relative = PurePosixPath(
        _canonical_relative_path(
            location,
            label="Face-embedding ONNX external_data location",
        )
    )
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        is_junction = getattr(candidate, "is_junction", None)
        if candidate.is_symlink() or (
            callable(is_junction) and is_junction()
        ):
            raise ValueError(
                "Face-embedding ONNX external data must not use symlinks; "
                f"rejected location {location!r}."
            )
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Face-embedding ONNX external data is missing: {candidate}"
        ) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Face-embedding ONNX external data must remain beside the model; "
            f"rejected location {location!r}."
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Face-embedding ONNX external data is not a file: {resolved}"
        )
    return resolved


def _hash_stable_file(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(
            f"Face-embedding ONNX source changed while hashing: {path}"
        )
    return int(after.st_size), digest.hexdigest()


def _build_source_entries(
    onnx_path: Path,
    external_locations: list[str],
) -> list[dict[str, Any]]:
    root = onnx_path.parent.resolve(strict=True)
    source_resolved = onnx_path.resolve(strict=True)
    source_bytes, source_hash = _hash_stable_file(source_resolved)
    entries = [
        {
            "path": onnx_path.name,
            "kind": "onnx",
            "bytes": source_bytes,
            "sha256": source_hash,
        }
    ]
    for location in external_locations:
        external_path = _safe_external_path(root, location)
        if external_path == source_resolved:
            raise ValueError(
                "Face-embedding ONNX external_data must not point back to "
                "the ONNX protobuf."
            )
        byte_count, sha256 = _hash_stable_file(external_path)
        entries.append(
            {
                "path": location,
                "kind": "external_data",
                "bytes": byte_count,
                "sha256": sha256,
            }
        )
    entries.sort(key=lambda item: item["path"])
    return _validate_source_manifest_entries(entries)


def _inspect_onnx_source(
    onnx_module: Any,
    path: str | Path,
) -> tuple[Any, str, list[dict[str, Any]], list[str]]:
    onnx_path = Path(path)
    if onnx_path.suffix.lower() != ".onnx" or not onnx_path.is_file():
        raise ValueError(
            "Face Core ML export requires a readable ONNX source file, got "
            f"{onnx_path}."
        )
    model_proto = onnx_module.load(
        str(onnx_path),
        load_external_data=False,
    )
    locations = _external_tensor_locations(model_proto)
    entries = _build_source_entries(onnx_path, locations)
    return model_proto, _source_manifest_hash(entries), entries, locations


def facerec_onnx_source_manifest(
    path: str | Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Safely fingerprint an ONNX protobuf and all of its external data."""
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "Face ONNX source fingerprinting requires ONNX. Install with: "
            "pip install 'libreyolo[onnx]'"
        ) from exc
    _, digest, entries, _ = _inspect_onnx_source(onnx, path)
    return digest, entries


def _validate_reserved_official_source(
    onnx_path: Path,
    entries: list[dict[str, Any]],
) -> bool:
    spec = _official_embedder_spec()
    official_filename = str(spec["filename"])
    if onnx_path.name.casefold() != official_filename.casefold():
        return False
    if onnx_path.name != official_filename:
        raise ValueError(
            "The reserved official face-recognition filename is "
            f"case-sensitive and must be {official_filename!r}."
        )
    expected = [
        {
            "path": official_filename,
            "kind": "onnx",
            "bytes": int(spec["size_bytes"]),
            "sha256": str(spec["sha256"]),
        }
    ]
    if not _same_json_contract(entries, expected):
        raise ValueError(
            "The reserved official face-recognition filename must contain "
            "exactly the pinned single-file ONNX artifact. Rename custom "
            "weights or restore the official checkpoint."
        )
    return True


def _require_exact_onnx2torch():
    try:
        version = importlib.metadata.version("onnx2torch")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(
            "Face Core ML export requires onnx2torch==1.5.15. Install with: "
            "pip install 'onnx2torch==1.5.15'"
        ) from exc
    if version != _ONNX2TORCH_VERSION:
        raise RuntimeError(
            "Face Core ML export requires exactly onnx2torch=="
            f"{_ONNX2TORCH_VERSION}; found {version!r}."
        )
    try:
        from onnx2torch import convert as convert_onnx_to_torch
    except ImportError as exc:
        raise ImportError(
            "The installed onnx2torch==1.5.15 distribution could not be "
            "imported."
        ) from exc
    return convert_onnx_to_torch


def _fixed_onnx_shape(
    value_info: Any,
    *,
    label: str,
    bind_dynamic_batch: bool = False,
) -> tuple[int, ...]:
    tensor_type = value_info.type.tensor_type
    element_type = int(getattr(tensor_type, "elem_type", 0) or 0)
    if element_type != _ONNX_TENSOR_FLOAT:
        raise ValueError(
            "Face Core ML export requires FLOAT ONNX tensors; "
            f"{label} element type is {element_type}."
        )
    dimensions = []
    for index, dimension in enumerate(tensor_type.shape.dim):
        value = int(getattr(dimension, "dim_value", 0) or 0)
        if value <= 0:
            if bind_dynamic_batch and index == 0:
                # The public face API evaluates aligned crops one at a time.
                # Official and bring-your-own ONNX graphs may therefore expose
                # a symbolic batch axis even though the deployment ABI is
                # intentionally fixed to batch 1.
                value = 1
            else:
                raise NotImplementedError(
                    "Face Core ML export requires fixed ONNX dimensions "
                    "outside the batch axis; "
                    f"{label} axis {index} is dynamic."
                )
        dimensions.append(value)
    if not dimensions:
        raise ValueError(f"Face-embedding ONNX {label} has no tensor shape.")
    return tuple(dimensions)


def _validate_onnx_io(
    model_proto: Any,
    *,
    preprocess: Mapping[str, Any],
) -> tuple[tuple[int, ...], int]:
    initializer_names = {item.name for item in model_proto.graph.initializer}
    inputs = [
        item for item in model_proto.graph.input if item.name not in initializer_names
    ]
    outputs = list(model_proto.graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise NotImplementedError(
            "Face Core ML export requires exactly one ONNX input and one "
            f"output; found {len(inputs)} input(s), {len(outputs)} output(s)."
        )
    input_shape = _fixed_onnx_shape(
        inputs[0],
        label="input",
        bind_dynamic_batch=True,
    )
    output_shape = _fixed_onnx_shape(
        outputs[0],
        label="output",
        bind_dynamic_batch=True,
    )
    size = int(preprocess["size"])
    expected_input = (
        (1, 3, size, size)
        if preprocess["layout"] == "NCHW"
        else (1, size, size, 3)
    )
    if input_shape != expected_input:
        raise ValueError(
            "Face-embedding ONNX input shape disagrees with preprocessing: "
            f"expected {expected_input}, got {input_shape}."
        )
    if len(output_shape) != 2 or output_shape[0] != 1 or output_shape[1] <= 0:
        raise ValueError(
            "Face-embedding ONNX output must have fixed shape [1, D], got "
            f"{output_shape}."
        )
    return input_shape, int(output_shape[1])


def _face_probe(cfg: "PreprocCfg", *, invert: bool) -> np.ndarray:
    from ..models.facerec.preprocess import preprocess_aligned

    size = int(cfg.size)
    y = np.arange(size, dtype=np.uint16)[:, None]
    x = np.arange(size, dtype=np.uint16)[None, :]
    image = np.stack(
        [
            np.broadcast_to((3 * x + 2 * y) % 256, (size, size)),
            np.broadcast_to((x + 5 * y + 37) % 256, (size, size)),
            np.broadcast_to((7 * x + y + 91) % 256, (size, size)),
        ],
        axis=-1,
    ).astype(np.uint8)
    if invert:
        image = 255 - image
    return preprocess_aligned(image, cfg)


def _checked_trace(
    model: "LibreFaceEmbedder",
    graph: nn.Module,
) -> tuple[torch.jit.ScriptModule, int, float]:
    probes = [
        _face_probe(model.cfg, invert=False),
        _face_probe(model.cfg, invert=True),
    ]
    adapter = _FaceEmbeddingOutput(graph).cpu().eval()
    eager: list[torch.Tensor] = []
    references: list[np.ndarray] = []
    maximum_error = 0.0
    with torch.inference_mode():
        for probe in probes:
            tensor = torch.from_numpy(np.ascontiguousarray(probe)).float()
            value = adapter(tensor)
            if value.ndim != 2 or value.shape[0] != 1:
                raise RuntimeError(
                    "Converted face-embedding graph must emit shape [1, D], "
                    f"got {tuple(value.shape)}."
                )
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(
                    "Converted face-embedding graph emitted NaN or infinity."
                )
            reference = np.asarray(
                model.session.run(None, {model.input_name: probe})[0],
                dtype=np.float32,
            )
            if reference.shape != tuple(value.shape):
                raise RuntimeError(
                    "ONNX and converted face-embedding output shapes differ: "
                    f"{reference.shape} != {tuple(value.shape)}."
                )
            converted = value.detach().cpu().numpy()
            error = float(np.max(np.abs(converted - reference)))
            maximum_error = max(maximum_error, error)
            np.testing.assert_allclose(
                converted,
                reference,
                rtol=1e-3,
                atol=1e-4,
                err_msg=(
                    "Opaque ONNX-to-PyTorch face conversion failed numeric parity."
                ),
            )
            eager.append(value.detach().clone())
            references.append(reference)

    reference_sensitivity = float(np.linalg.norm(references[0] - references[1]))
    reference_scale = max(
        float(np.linalg.norm(references[0])),
        float(np.linalg.norm(references[1])),
        1.0,
    )
    if reference_sensitivity <= max(
        100.0 * maximum_error,
        1e-6 * reference_scale,
    ):
        raise RuntimeError(
            "Face-embedding parity probes are not meaningfully input-sensitive; "
            "refusing a false-positive conversion."
        )

    first = torch.from_numpy(np.ascontiguousarray(probes[0])).float()
    second = torch.from_numpy(np.ascontiguousarray(probes[1])).float()
    traced = torch.jit.trace(
        adapter,
        first,
        check_trace=True,
        check_inputs=[(second,)],
        strict=True,
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            traced(first),
            eager[0],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            traced(second),
            eager[1],
            rtol=0.0,
            atol=0.0,
        )
    return traced, int(eager[0].shape[1]), maximum_error


def _resolve_options(
    model: "LibreFaceEmbedder",
    kwargs: Mapping[str, Any],
) -> tuple[str, str, str]:
    from .coreml import _normalize_mlpackage_path

    options = dict(kwargs)
    output_path = options.pop("output_path", None)
    output_alias = options.pop("output", None)
    normalized_output = (
        _normalize_mlpackage_path(output_path)
        if output_path not in (None, "")
        else None
    )
    normalized_alias = (
        _normalize_mlpackage_path(output_alias)
        if output_alias not in (None, "")
        else None
    )
    if (
        normalized_output is not None
        and normalized_alias is not None
        and normalized_output != normalized_alias
    ):
        raise ValueError("Pass only one Core ML destination: output_path= or output=.")
    source = Path(model.model_path)
    destination = _normalize_mlpackage_path(
        normalized_output
        or normalized_alias
        or Path("weights") / source.with_suffix(".mlpackage").name
    )

    def option_bool(name: str, default: bool) -> bool:
        value = options.pop(name, default)
        if not isinstance(value, bool):
            raise TypeError(
                f"Face Core ML export option {name!r} must be a boolean."
            )
        return value

    half = option_bool("half", False)
    int8 = option_bool("int8", False)
    dynamic = option_bool("dynamic", False)
    batch_value = options.pop("batch", 1)
    if isinstance(batch_value, bool) or not isinstance(batch_value, int):
        raise TypeError("Face Core ML export option 'batch' must be an integer.")
    batch = int(batch_value)
    nms = option_bool("nms", False)
    imgsz = options.pop("imgsz", None)
    device = options.pop("device", None)
    compute_units = str(
        options.pop(
            "compute_units",
            "cpu_only",
        )
    ).strip().lower()
    if options:
        raise TypeError(
            "Unsupported or irrelevant face Core ML export options: "
            + ", ".join(sorted(options))
        )
    if dynamic:
        raise NotImplementedError(
            "Face Core ML export uses fixed aligned-face tensors; "
            "dynamic=True is not supported."
        )
    if batch != 1:
        raise ValueError(f"Face Core ML export requires batch=1; got {batch}.")
    if int8:
        raise NotImplementedError("Face Core ML export does not support int8.")
    if half:
        raise NotImplementedError(
            "Face Core ML export is FP32-only. Fresh Apple M4 validation "
            "measured 1.99e-2 relative raw-embedding error with FP16 versus "
            "5.71e-6 with FP32; pass half=False."
        )
    if nms:
        raise NotImplementedError("NMS is not applicable to face embeddings.")
    if device not in (None, "", "auto", "cpu", torch.device("cpu")):
        raise NotImplementedError(
            "Face Core ML conversion runs on CPU; pass device='cpu', "
            "device='auto', or omit device."
        )
    if imgsz is not None:
        if isinstance(imgsz, bool):
            raise TypeError("Face Core ML imgsz must be an integer or (h, w).")
        if isinstance(imgsz, (tuple, list)):
            if len(imgsz) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in imgsz
            ):
                raise TypeError(
                    "Face Core ML imgsz must be an integer or two integers."
                )
            requested = (int(imgsz[0]), int(imgsz[1]))
        elif isinstance(imgsz, int):
            requested = (imgsz, imgsz)
        else:
            raise TypeError("Face Core ML imgsz must be an integer or (h, w).")
        if requested != (int(model.cfg.size), int(model.cfg.size)):
            raise NotImplementedError(
                "Face Core ML export must preserve the recognition head's "
                f"fixed {model.cfg.size}x{model.cfg.size} input; got {requested}."
            )
    valid_compute_units = {
        "validated",
        "all",
        "cpu_and_gpu",
        "cpu_and_ne",
        "cpu_only",
    }
    if compute_units not in valid_compute_units:
        raise ValueError(
            f"Invalid Core ML compute_units {compute_units!r}; expected one of "
            f"{sorted(valid_compute_units)}."
        )
    if compute_units not in {
        "validated",
        FACEREC_COREML_REQUIRED_COMPUTE_UNITS,
    }:
        raise NotImplementedError(
            "Face Core ML export is validated only with "
            "compute_units='cpu_only'. Other planners have not passed the "
            "raw-embedding hardware parity gate."
        )
    return (
        destination,
        "fp32",
        compute_units,
    )


def _apply_coreml_execution_profile(
    metadata: Mapping[str, Any],
    *,
    size: str,
    canvas: int,
    precision: str,
    compute_units: str,
    embedding_dim: int,
) -> tuple[dict[str, Any], str, Any]:
    """Resolve FaceRec source identity while deferring final protobuf ABI."""
    from .coreml_identity import (
        COREML_PROFILE_SOURCE_KIND_KEY,
        COREML_PROFILE_SOURCE_SHA256_KEY,
    )
    from .coreml_profiles import (
        resolve_coreml_export_compute_units,
    )

    identified = dict(metadata)
    identified.setdefault(
        COREML_PROFILE_SOURCE_KIND_KEY,
        "facerec-onnx-source-manifest-v1",
    )
    identified.setdefault(
        COREML_PROFILE_SOURCE_SHA256_KEY,
        identified.get(FACEREC_COREML_SOURCE_HASH_KEY),
    )
    resolved_compute_units, profile = resolve_coreml_export_compute_units(
        compute_units,
        family="facerec",
        task="embed",
        size=size,
        canvas=canvas,
        precision=precision,
        nms=False,
        class_count=1,
        embedding_dim=embedding_dim,
        source_kind=identified.get(COREML_PROFILE_SOURCE_KIND_KEY),
        source_sha256=identified.get(COREML_PROFILE_SOURCE_SHA256_KEY),
    )
    return identified, resolved_compute_units, profile


def export_facerec_coreml(
    model: "LibreFaceEmbedder",
    kwargs: Mapping[str, Any],
) -> str:
    """Export one loaded opaque face-recognition ONNX as a strict package."""
    output_path, precision, compute_units = _resolve_options(model, kwargs)
    onnx_path = Path(model.model_path)
    if onnx_path.suffix.lower() != ".onnx" or not onnx_path.is_file():
        raise ValueError(
            "Face Core ML export requires the loaded source model to be an "
            f"ONNX file, got {onnx_path}."
        )

    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "Face Core ML export requires ONNX. Install with: "
            "pip install 'libreyolo[onnx]'"
        ) from exc

    model_proto, source_hash, source_entries, external_locations = (
        _inspect_onnx_source(onnx, onnx_path)
    )
    official_source = _validate_reserved_official_source(
        onnx_path,
        source_entries,
    )
    # External locations were recursively enumerated, contained, and hashed
    # before this API is allowed to open any external tensor bytes.
    onnx.load_external_data_for_model(
        model_proto,
        str(onnx_path.parent.resolve(strict=True)),
    )
    remaining_external = _external_tensor_locations(model_proto)
    if remaining_external:
        raise RuntimeError(
            "This ONNX version did not hydrate every recursively discovered "
            "external tensor; refusing partial conversion."
        )
    rehashed_entries = _build_source_entries(onnx_path, external_locations)
    if not _same_json_contract(rehashed_entries, source_entries):
        raise RuntimeError(
            "Face-embedding ONNX source changed between verification and "
            "external-data loading."
        )
    onnx.checker.check_model(model_proto)

    preprocess = _preprocess_payload(model.cfg)
    if official_source:
        _require_official_preprocess(preprocess)
    input_shape, declared_dim = _validate_onnx_io(
        model_proto,
        preprocess=preprocess,
    )
    size = "l" if official_source else "custom"
    requested_compute_units = compute_units
    profile_identity, compute_units, execution_profile = (
        _apply_coreml_execution_profile(
            {FACEREC_COREML_SOURCE_HASH_KEY: source_hash},
            size=size,
            canvas=int(preprocess["size"]),
            precision=precision,
            compute_units=requested_compute_units,
            embedding_dim=declared_dim,
        )
    )

    try:
        import coremltools as ct
    except ImportError as exc:
        raise ImportError(
            "Face Core ML export requires coremltools. Install with: "
            "pip install 'libreyolo[coreml]'"
        ) from exc
    from .coreml import _coreml_profile_for_toolchain

    execution_profile = _coreml_profile_for_toolchain(
        execution_profile,
        requested_compute_units=requested_compute_units,
        coremltools=ct,
    )
    convert_onnx_to_torch = _require_exact_onnx2torch()

    # Passing ModelProto avoids onnx2torch's temporary-file path on Windows and
    # ensures verified external data was resolved before graph conversion.
    converted = convert_onnx_to_torch(model_proto).cpu().eval()
    traced, embedding_dim, conversion_error = _checked_trace(model, converted)
    if embedding_dim != declared_dim:
        raise RuntimeError(
            "Converted face-embedding dimension disagrees with ONNX metadata: "
            f"{embedding_dim} != {declared_dim}."
        )

    from .. import __version__
    from ..backends.coreml_facerec import validate_facerec_coreml_spec
    from ..utils.serialization import SCHEMA_VERSION
    from .coreml import (
        _replace_user_defined_metadata,
        _save_mlpackage_atomic,
        _stringify_metadata,
    )
    from .coreml_identity import (
        COREML_PROFILE_SOURCE_KIND_KEY,
        COREML_PROFILE_SOURCE_SHA256_KEY,
        bind_coreml_deployment_abi,
        validate_coreml_deployment_abi,
    )
    from .coreml_profiles import (
        finalize_coreml_execution_profile_metadata,
        validate_coreml_execution_profile_metadata,
    )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "libreyolo_version": __version__,
        "libreyolo_producer": "libreyolo",
        "artifact_format": "coreml",
        "coreml_io_schema_version": "2",
        "model_family": "facerec",
        "artifact_scope": FACEREC_COREML_ARTIFACT_SCOPE,
        "size": size,
        "model_size": size,
        "task": "embed",
        "supported_tasks": ["embed"],
        "default_task": "embed",
        "names": {"0": "face"},
        "nc": 1,
        "nb_classes": 1,
        "imgsz": int(preprocess["size"]),
        "imgsz_h": int(preprocess["size"]),
        "imgsz_w": int(preprocess["size"]),
        "precision": precision,
        "coreml_required_compute_units": (
            FACEREC_COREML_REQUIRED_COMPUTE_UNITS
        ),
        "dynamic": False,
        "facerec_contract": FACEREC_COREML_CONTRACT,
        FACEREC_COREML_PREPROCESS_KEY: _canonical_json(preprocess),
        FACEREC_COREML_PREPROCESS_HASH_KEY: facerec_coreml_preprocess_hash(
            preprocess
        ),
        FACEREC_COREML_SOURCE_HASH_KEY: source_hash,
        FACEREC_COREML_SOURCE_MANIFEST_KEY: _canonical_json(source_entries),
        "facerec_embedding_dim": embedding_dim,
        "facerec_onnx_to_torch_max_abs_error": conversion_error,
        "coreml_output_names": [FACEREC_COREML_OUTPUT_NAME],
        "coreml_io": {
            "input": {
                "name": FACEREC_COREML_INPUT_NAME,
                "kind": "tensor",
                "layout": preprocess["layout"],
                "color": preprocess["color_order"].lower(),
                "range": "standardized",
                "mean": [preprocess["mean"] / 255.0] * 3,
                "std": [1.0 / (preprocess["scale"] * 255.0)] * 3,
                "geometry": FACEREC_COREML_GEOMETRY,
                "interpolation": "bilinear",
                "resize_backend": "opencv",
                "pad_value": 0,
                "shape_mode": "fixed",
            },
            "validation": {
                "color": preprocess["color_order"].lower(),
                "range": "standardized",
                "mean": [preprocess["mean"] / 255.0] * 3,
                "std": [1.0 / (preprocess["scale"] * 255.0)] * 3,
            },
            "outputs": [
                {
                    "name": FACEREC_COREML_OUTPUT_NAME,
                    "role": "embedding",
                    "encoding": "raw_identity_embedding",
                    "rank": 2,
                    "dtype": "float32",
                    "shape": [1, embedding_dim],
                }
            ],
        },
    }
    if official_source:
        metadata.update(_official_provenance_metadata())
    metadata.update(
        {
            COREML_PROFILE_SOURCE_KIND_KEY: (
                profile_identity[COREML_PROFILE_SOURCE_KIND_KEY]
            ),
            COREML_PROFILE_SOURCE_SHA256_KEY: profile_identity[
                COREML_PROFILE_SOURCE_SHA256_KEY
            ],
        }
    )
    validate_facerec_coreml_metadata(metadata)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name=FACEREC_COREML_INPUT_NAME,
                shape=input_shape,
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(
                name=FACEREC_COREML_OUTPUT_NAME,
                dtype=np.float32,
            )
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS15,
        compute_units={
            "all": ct.ComputeUnit.ALL,
            "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
            "cpu_only": ct.ComputeUnit.CPU_ONLY,
        }[compute_units],
    )
    spec = mlmodel.get_spec()
    input_names = [str(item.name) for item in spec.description.input]
    output_names = [str(item.name) for item in spec.description.output]
    if input_names != [FACEREC_COREML_INPUT_NAME] or output_names != [
        FACEREC_COREML_OUTPUT_NAME
    ]:
        raise RuntimeError(
            "Core ML converter changed the face component ABI: "
            f"inputs={input_names}, outputs={output_names}."
        )
    metadata = bind_coreml_deployment_abi(metadata, spec)
    metadata, execution_profile = (
        finalize_coreml_execution_profile_metadata(
            metadata,
            execution_profile,
            requested_compute_units=requested_compute_units,
            conversion_compute_units=compute_units,
            deployment_abi_sha256=metadata[
                "coreml_profile_abi_sha256"
            ],
        )
    )
    validate_facerec_coreml_metadata(metadata)
    serialized_metadata = _stringify_metadata(metadata)
    validate_facerec_coreml_metadata(serialized_metadata)
    validate_coreml_execution_profile_metadata(serialized_metadata)
    validate_facerec_coreml_spec(spec, serialized_metadata)

    _replace_user_defined_metadata(mlmodel, serialized_metadata)

    def validate_candidate(candidate: Path) -> None:
        staged_spec = ct.utils.load_spec(str(candidate))
        staged_metadata_container = getattr(
            getattr(staged_spec, "description", None),
            "metadata",
            None,
        )
        staged_metadata = dict(
            getattr(staged_metadata_container, "userDefined", None) or {}
        )
        if staged_metadata != serialized_metadata:
            raise RuntimeError(
                "Staged Face Core ML metadata differs from the validated "
                "pre-save contract."
            )
        validate_facerec_coreml_metadata(staged_metadata)
        validate_facerec_coreml_spec(staged_spec, staged_metadata)
        validate_coreml_deployment_abi(staged_spec, staged_metadata)
        validate_coreml_execution_profile_metadata(staged_metadata)

    _save_mlpackage_atomic(
        mlmodel,
        output_path,
        validate_candidate=validate_candidate,
    )
    return str(output_path)


__all__ = [
    "FACEREC_COREML_ARTIFACT_SCOPE",
    "FACEREC_COREML_CONTRACT",
    "FACEREC_COREML_GEOMETRY",
    "FACEREC_COREML_INPUT_NAME",
    "FACEREC_COREML_OUTPUT_NAME",
    "FACEREC_COREML_REQUIRED_COMPUTE_UNITS",
    "FACEREC_COREML_SOURCE_MANIFEST_KEY",
    "export_facerec_coreml",
    "facerec_onnx_source_manifest",
    "facerec_coreml_preprocess_hash",
    "validate_facerec_coreml_metadata",
]
