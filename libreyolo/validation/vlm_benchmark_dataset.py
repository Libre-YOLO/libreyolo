"""Build and verify the local-only dataset for the VLM confidence gate.

This module is intentionally internal.  It accepts an already-provisioned
COCO 2017 validation tree, selects only the pinned CC-BY-2.0 image cohort, and
writes metadata artifacts.  It has no download path and never copies, links,
or rewrites an image.  The production contract binds the selected local bytes
to an aggregate independently derived from the pinned official image archive.
A successful verification is still not a legal, attribution, privacy, or
benchmark-suitability approval. The local source tree is assumed to be trusted
and quiescent while its paths are checked, hashed, and decoded.

The implementation is original and follows only the public COCO JSON data
contract; no third-party implementation source was used.

The source digest is over canonical JSON, with the four COCO record arrays
sorted by integer ``id``.  This preserves a complete content pin while making
the result independent of harmless source-object and source-array ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError

_SCHEMA = "libreyolo.vlm-benchmark-dataset.v1"
_STATUS_SCHEMA = "libreyolo.vlm-benchmark-dataset-status.v1"
_REVIEW_SCHEMA = "libreyolo.vlm-benchmark-dataset-review.v1"
_PROMOTION_ROLE = "zero_shot_confidence_promotion"
_REVIEW_STATUS = "approved"
_SOURCE_CANONICALIZATION = "coco-arrays-by-id-json-v1"
_SELECTION_ALGORITHM = "category-coverage-then-sha256-rank-v1"
_SELECTION_SALT = "libreyolo:coco-val2017:license-4:v1"
_SOURCE_FILE_NAME = "instances_val2017.json"
_MANIFEST_NAME = "manifest.json"
_ATTRIBUTION_PATH = "ATTRIBUTION.jsonl"
_ANNOTATION_NOTICE_PATH = "ANNOTATION_NOTICE.txt"
_HOLDOUT_COUNT = 100
_SELECTED_COUNT = 500
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_REVIEW_ATTESTATION_BYTES = 64 * 1024
_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 10_000_000
_MAX_SAFE_INTEGER = (1 << 53) - 1
_UTC_RFC3339 = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_JSON_ARRAY_FIELDS = ("licenses", "images", "annotations", "categories")
_SOURCE_FIELDS = {"info", *_JSON_ARRAY_FIELDS}
_LICENSE_ID = 4
_LICENSE_SPDX = "CC-BY-2.0"
_LICENSE_NAME = "Attribution License"
_LICENSE_URL = "http://creativecommons.org/licenses/by/2.0/"
_SOURCE_HOMEPAGE = "http://cocodataset.org"
_SELECTED_IMAGE_CANONICALIZATION = "selected-image-id-name-size-sha256-json-v1"
_ANNOTATION_LICENSE_SPDX = "CC-BY-4.0"
_ANNOTATION_LICENSE_NAME = "Creative Commons Attribution 4.0 International"
_ANNOTATION_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode"
_ANNOTATION_RIGHTS_HOLDER = "COCO Consortium"
_ANNOTATION_TERMS_URL = "https://cocodataset.org/#termsofuse"
_ANNOTATION_ARCHIVE_URL = (
    "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
_ANNOTATION_TERMS_SOURCE = (
    "https://github.com/cocodataset/cocodataset.github.io/blob/"
    "aaa6a5a0cc24bf1350247169cc512edd7ddf28b9/dataset/termsofuse.htm"
)


@dataclass(frozen=True)
class _SourceContract:
    canonical_sha256: str
    canonical_size_bytes: int
    image_count: int
    annotation_count: int
    category_count: int
    license_image_count: int
    eligible_image_count: int
    available_category_count: int
    unavailable_category_ids: tuple[int, ...]
    coverage_seed_image_count: int
    image_archive_url: str | None = None
    image_archive_size_bytes: int | None = None
    image_archive_sha256: str | None = None
    image_archive_etag: str | None = None
    image_archive_last_modified: str | None = None
    selected_image_identity_sha256: str | None = None
    selected_image_identity_size_bytes: int | None = None
    selected_image_bytes_total: int | None = None
    partition_unrepresented_category_ids: tuple[tuple[str, tuple[int, ...]], ...] = ()


_SOURCE_CONTRACT = _SourceContract(
    canonical_sha256="9804c52e59ccf08af59ca2dbeb5d1529cd49fce554954523b2c79d93242488f4",
    canonical_size_bytes=19_689_206,
    image_count=5_000,
    annotation_count=36_781,
    category_count=80,
    license_image_count=857,
    eligible_image_count=790,
    available_category_count=79,
    unavailable_category_ids=(89,),
    coverage_seed_image_count=43,
    image_archive_url="https://images.cocodataset.org/zips/val2017.zip",
    image_archive_size_bytes=815_585_330,
    image_archive_sha256="4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
    image_archive_etag='"d366be60d3dc737327160d62453e3973-98"',
    image_archive_last_modified="Wed, 11 Jul 2018 05:08:47 GMT",
    selected_image_identity_sha256=(
        "73e35dbb1ce5058953bccbc99ab15db46474f36cc160046cbcac71350662d29c"
    ),
    selected_image_identity_size_bytes=73_312,
    selected_image_bytes_total=81_833_238,
    partition_unrepresented_category_ids=(
        ("holdout100", (89,)),
        ("train400", (21, 38, 87, 89)),
        ("promotion500", (89,)),
    ),
)

_ARTIFACT_PATHS = {
    "holdout100": "annotations/instances_val2017_holdout100.json",
    "train400": "annotations/instances_val2017_train400.json",
    "promotion500": "annotations/instances_val2017_promotion500.json",
}


class BenchmarkDatasetError(ValueError):
    """A source dataset, manifest, or metadata artifact is invalid."""


class BenchmarkDatasetOutputExistsError(FileExistsError):
    """The immutable dataset-metadata destination is already occupied."""


@dataclass(frozen=True)
class BenchmarkDatasetArtifacts:
    """The immutable files produced by a successful metadata-only build."""

    output_dir: Path
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class VerifiedBenchmarkDataset:
    """Identity returned after reconstructing and checking a local bundle."""

    output_dir: Path
    manifest_path: Path
    manifest_sha256: str
    selected_image_count: int


@dataclass(frozen=True)
class VerifiedBenchmarkRunInputs:
    """Strictly verified local inputs for one fixed benchmark partition."""

    manifest_path: Path
    manifest_sha256: str
    source_annotations: Path
    source_canonical_sha256: str
    source_file_sha256: str
    source_file_size_bytes: int
    images_dir: Path
    selected_image_identity_sha256: str
    partition_name: str
    partition_role: str
    partition_start: int
    partition_stop: int
    annotation_path: Path
    annotation_sha256: str
    annotation_size_bytes: int
    class_names: tuple[str, ...]
    expected_images: tuple[Mapping[str, Any], ...]
    expected_categories: tuple[Mapping[str, Any], ...]
    expected_annotations: tuple[Mapping[str, Any], ...]
    review_attestation_path: Path
    review_attestation_sha256: str
    review_attestation: Mapping[str, Any]


@dataclass(frozen=True)
class _SourceData:
    root: dict[str, Any]
    file_sha256: str
    file_size_bytes: int
    canonical_sha256: str
    canonical_size_bytes: int
    license: dict[str, Any]
    images: dict[int, dict[str, Any]]
    annotations: dict[int, dict[str, Any]]
    annotations_by_image: dict[int, tuple[dict[str, Any], ...]]
    categories: dict[int, dict[str, Any]]
    eligible_image_ids: frozenset[int]
    available_category_ids: tuple[int, ...]
    unavailable_category_ids: tuple[int, ...]


@dataclass(frozen=True)
class _DerivedBundle:
    manifest: dict[str, Any]
    files: dict[str, bytes]
    source_file_sha256: str
    source_file_size_bytes: int


@dataclass(frozen=True)
class _VerifiedBundleEvidence:
    manifest_path: Path
    bundle_root: Path
    source_annotations: Path
    images_dir: Path
    manifest_sha256: str
    derived: _DerivedBundle


def _path_exists(path: Path) -> bool:
    """Return true for ordinary paths and broken symlinks."""

    return os.path.lexists(os.fspath(path))


def _path_argument(value: Any, label: str) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{label} must be a filesystem path")
    return Path(value).expanduser()


def _required_file(value: Any, label: str) -> Path:
    path = _path_argument(value, label)
    if path.is_symlink():
        raise BenchmarkDatasetError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise BenchmarkDatasetError(f"{label} is not an existing file: {path}") from exc
    if not resolved.is_file():
        raise BenchmarkDatasetError(f"{label} is not an existing file: {path}")
    return resolved


def _required_directory(value: Any, label: str) -> Path:
    path = _path_argument(value, label)
    if path.is_symlink():
        raise BenchmarkDatasetError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise BenchmarkDatasetError(
            f"{label} is not an existing directory: {path}"
        ) from exc
    if not resolved.is_dir():
        raise BenchmarkDatasetError(f"{label} is not an existing directory: {path}")
    return resolved


def _assert_separate_bundle_root(bundle_root: Path, images_dir: Path) -> None:
    bundle_root = bundle_root.resolve(strict=False)
    if (
        bundle_root == images_dir
        or images_dir in bundle_root.parents
        or bundle_root in images_dir.parents
    ):
        raise BenchmarkDatasetError(
            "metadata output and source image directory must be disjoint"
        )


def _read_bounded(path: Path, *, max_bytes: int, label: str) -> bytes:
    before = path.stat()
    if before.st_size > max_bytes:
        raise BenchmarkDatasetError(
            f"{label} exceeds the {max_bytes}-byte safety limit"
        )
    with path.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    after = path.stat()
    if len(payload) > max_bytes:
        raise BenchmarkDatasetError(
            f"{label} exceeds the {max_bytes}-byte safety limit"
        )
    if (
        before.st_size != len(payload)
        or after.st_size != len(payload)
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise BenchmarkDatasetError(f"{label} changed while it was being read")
    return payload


def _parse_int(value: str) -> int:
    number = int(value)
    if abs(number) > _MAX_SAFE_INTEGER:
        raise BenchmarkDatasetError("JSON integer exceeds the exact safe range")
    return number


def _parse_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise BenchmarkDatasetError("JSON number must be finite")
    return number


def _reject_constant(value: str) -> None:
    raise BenchmarkDatasetError(f"JSON constant {value!r} is not permitted")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkDatasetError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _validate_json_tree(value: Any, label: str) -> None:
    stack = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise BenchmarkDatasetError(
                f"{label} exceeds the {_MAX_JSON_NODES}-node safety limit"
            )
        if depth > _MAX_JSON_DEPTH:
            raise BenchmarkDatasetError(
                f"{label} exceeds the {_MAX_JSON_DEPTH}-level nesting limit"
            )
        if isinstance(current, Mapping):
            if any(not isinstance(key, str) for key in current):
                raise BenchmarkDatasetError(f"{label} object keys must be strings")
            stack.extend((nested, depth + 1) for nested in current.values())
        elif isinstance(current, list):
            stack.extend((nested, depth + 1) for nested in current)
        elif current is None or isinstance(current, (str, bool)):
            continue
        elif type(current) is int:
            if abs(current) > _MAX_SAFE_INTEGER:
                raise BenchmarkDatasetError(
                    f"{label} integer exceeds the exact safe range"
                )
        elif isinstance(current, float):
            if not math.isfinite(current):
                raise BenchmarkDatasetError(f"{label} numbers must be finite")
        else:
            raise BenchmarkDatasetError(
                f"{label} contains unsupported {type(current).__name__} data"
            )


def _decode_json(payload: bytes, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BenchmarkDatasetError(f"{label} must be UTF-8 JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_parse_int,
            parse_float=_parse_float,
            parse_constant=_reject_constant,
        )
    except BenchmarkDatasetError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise BenchmarkDatasetError(f"{label} is not valid bounded JSON") from exc
    _validate_json_tree(value, label)
    return value


def _load_json(path: Path, label: str) -> Any:
    return _decode_json(
        _read_bounded(path, max_bytes=_MAX_JSON_BYTES, label=label), label
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_file_bytes(value: Any) -> bytes:
    return _canonical_json(value) + b"\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_identity(path: Path) -> tuple[str, int]:
    before = path.stat()
    if before.st_size > _MAX_IMAGE_BYTES:
        raise BenchmarkDatasetError(
            f"selected image exceeds the {_MAX_IMAGE_BYTES}-byte safety limit: {path}"
        )
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            if size > _MAX_IMAGE_BYTES:
                raise BenchmarkDatasetError(
                    "selected image exceeds the "
                    f"{_MAX_IMAGE_BYTES}-byte safety limit: {path}"
                )
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_size != size
        or after.st_size != size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise BenchmarkDatasetError(f"selected image changed while hashed: {path}")
    return digest.hexdigest(), size


def _exact_fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkDatasetError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unsupported " + ", ".join(extra))
        raise BenchmarkDatasetError(f"{label} has {'; '.join(detail)}")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise BenchmarkDatasetError(f"{label} must be a JSON array")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise BenchmarkDatasetError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise BenchmarkDatasetError(f"{label} must be >= {minimum}")
    return value


def _real(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkDatasetError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise BenchmarkDatasetError(f"{label} must be finite")
    return number


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkDatasetError(f"{label} must be a non-empty string")
    return value


def _utc_rfc3339(value: Any, label: str) -> str:
    rendered = _nonempty_string(value, label)
    if _UTC_RFC3339.fullmatch(rendered) is None:
        raise BenchmarkDatasetError(
            f"{label} must be a strict UTC RFC3339 timestamp ending in 'Z'"
        )
    try:
        parsed = datetime.fromisoformat(rendered[:-1] + "+00:00")
    except ValueError as exc:
        raise BenchmarkDatasetError(
            f"{label} must be a valid UTC RFC3339 timestamp"
        ) from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise BenchmarkDatasetError(f"{label} must use the UTC timezone")
    return rendered


def _http_url(value: Any, label: str) -> str:
    rendered = _nonempty_string(value, label)
    parts = urlsplit(rendered)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.username is not None
        or parts.password is not None
    ):
        raise BenchmarkDatasetError(f"{label} must be an ordinary HTTP(S) URL")
    return rendered


def _indexed_records(value: Any, label: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(_sequence(value, label)):
        if not isinstance(raw, Mapping):
            raise BenchmarkDatasetError(f"{label}[{index}] must be a JSON object")
        record = dict(raw)
        record_id = _integer(record.get("id"), f"{label}[{index}].id", minimum=0)
        if record_id in records:
            raise BenchmarkDatasetError(f"{label} contains duplicate id {record_id}")
        records[record_id] = record
    return records


def _normalized_source_bytes(root: Mapping[str, Any]) -> bytes:
    """Return the source-order-independent canonical COCO JSON bytes."""

    normalized = dict(root)
    for field in _JSON_ARRAY_FIELDS:
        rows = _sequence(normalized.get(field), f"source.{field}")
        normalized[field] = sorted(
            rows,
            key=lambda row: _integer(
                row.get("id") if isinstance(row, Mapping) else None,
                f"source.{field}.id",
                minimum=0,
            ),
        )
    return _canonical_json(normalized)


def _load_source(source_annotations: Path) -> _SourceData:
    if source_annotations.name != _SOURCE_FILE_NAME:
        raise BenchmarkDatasetError(
            f"source annotation filename must be {_SOURCE_FILE_NAME!r}"
        )
    source_payload = _read_bounded(
        source_annotations,
        max_bytes=_MAX_JSON_BYTES,
        label="source annotations",
    )
    raw_root = _decode_json(source_payload, "source annotations")
    root = _exact_fields(raw_root, _SOURCE_FIELDS, "source")
    info = root["info"]
    if not isinstance(info, Mapping):
        raise BenchmarkDatasetError("source.info must be a JSON object")
    if (
        info.get("description") != "COCO 2017 Dataset"
        or info.get("version") != "1.0"
        or info.get("year") != 2017
        or info.get("url") != _SOURCE_HOMEPAGE
    ):
        raise BenchmarkDatasetError("source.info does not identify COCO 2017")

    licenses = _indexed_records(root["licenses"], "source.licenses")
    license_record = licenses.get(_LICENSE_ID)
    if license_record is None:
        raise BenchmarkDatasetError("source license id 4 is missing")
    if set(license_record) != {"id", "name", "url"}:
        raise BenchmarkDatasetError("source license id 4 has unsupported fields")
    if license_record["name"] != _LICENSE_NAME or license_record["url"] != _LICENSE_URL:
        raise BenchmarkDatasetError(
            "source license id 4 is not the pinned CC-BY-2.0 license"
        )

    images = _indexed_records(root["images"], "source.images")
    categories = _indexed_records(root["categories"], "source.categories")
    annotations = _indexed_records(root["annotations"], "source.annotations")
    if len(images) != _SOURCE_CONTRACT.image_count:
        raise BenchmarkDatasetError("source image count does not match the pin")
    if len(categories) != _SOURCE_CONTRACT.category_count:
        raise BenchmarkDatasetError("source category count does not match the pin")
    if len(annotations) != _SOURCE_CONTRACT.annotation_count:
        raise BenchmarkDatasetError("source annotation count does not match the pin")

    for category_id, category in categories.items():
        _nonempty_string(category.get("name"), f"category {category_id} name")

    for image_id, image in images.items():
        filename = _nonempty_string(
            image.get("file_name"), f"image {image_id} filename"
        )
        if (
            filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or Path(filename).name != filename
            or Path(filename).suffix.lower() not in {".jpg", ".jpeg"}
        ):
            raise BenchmarkDatasetError(
                f"image {image_id} file_name must be a local JPEG basename"
            )
        _integer(image.get("license"), f"image {image_id} license", minimum=0)
        _integer(image.get("width"), f"image {image_id} width", minimum=1)
        _integer(image.get("height"), f"image {image_id} height", minimum=1)
        _http_url(image.get("flickr_url"), f"image {image_id} flickr_url")
        _http_url(image.get("coco_url"), f"image {image_id} coco_url")

    by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
    blocked_image_ids: set[int] = set()
    for annotation_id, annotation in annotations.items():
        image_id = _integer(
            annotation.get("image_id"),
            f"annotation {annotation_id} image_id",
            minimum=0,
        )
        category_id = _integer(
            annotation.get("category_id"),
            f"annotation {annotation_id} category_id",
            minimum=0,
        )
        if image_id not in images:
            raise BenchmarkDatasetError(
                f"annotation {annotation_id} references unknown image {image_id}"
            )
        if category_id not in categories:
            raise BenchmarkDatasetError(
                f"annotation {annotation_id} references unknown category {category_id}"
            )
        bbox = _sequence(annotation.get("bbox"), f"annotation {annotation_id} bbox")
        if len(bbox) != 4:
            raise BenchmarkDatasetError(
                f"annotation {annotation_id} bbox must contain four values"
            )
        x, y, width, height = (
            _real(value, f"annotation {annotation_id} bbox") for value in bbox
        )
        area = _real(annotation.get("area"), f"annotation {annotation_id} area")
        if width <= 0.0 or height <= 0.0 or area <= 0.0:
            raise BenchmarkDatasetError(
                f"annotation {annotation_id} must have positive area and extent"
            )
        image = images[image_id]
        if (
            x >= int(image["width"])
            or y >= int(image["height"])
            or x + width <= 0.0
            or y + height <= 0.0
        ):
            raise BenchmarkDatasetError(
                f"annotation {annotation_id} does not intersect its image"
            )
        iscrowd = _integer(
            annotation.get("iscrowd"), f"annotation {annotation_id} iscrowd"
        )
        ignore = _integer(
            annotation.get("ignore", 0), f"annotation {annotation_id} ignore"
        )
        if iscrowd not in {0, 1} or ignore not in {0, 1}:
            raise BenchmarkDatasetError(
                f"annotation {annotation_id} crowd/ignore flags must be 0 or 1"
            )
        if iscrowd or ignore:
            blocked_image_ids.add(image_id)
        by_image[image_id].append(annotation)

    canonical = _normalized_source_bytes(root)
    canonical_digest = _sha256(canonical)
    if canonical_digest != _SOURCE_CONTRACT.canonical_sha256:
        raise BenchmarkDatasetError(
            "source annotations do not match the pinned canonical SHA256"
        )
    if len(canonical) != _SOURCE_CONTRACT.canonical_size_bytes:
        raise BenchmarkDatasetError(
            "source annotations do not match the pinned canonical byte size"
        )

    license_image_ids = {
        image_id
        for image_id, image in images.items()
        if _integer(image["license"], f"image {image_id} license") == _LICENSE_ID
    }
    if len(license_image_ids) != _SOURCE_CONTRACT.license_image_count:
        raise BenchmarkDatasetError("license-id-4 image count does not match the pin")
    eligible = license_image_ids - blocked_image_ids
    if len(eligible) != _SOURCE_CONTRACT.eligible_image_count:
        raise BenchmarkDatasetError("eligible image count does not match the pin")
    available_categories = tuple(
        sorted(
            {
                int(annotation["category_id"])
                for image_id in eligible
                for annotation in by_image[image_id]
            }
        )
    )
    unavailable_categories = tuple(sorted(set(categories) - set(available_categories)))
    if (
        len(available_categories) != _SOURCE_CONTRACT.available_category_count
        or unavailable_categories != _SOURCE_CONTRACT.unavailable_category_ids
    ):
        raise BenchmarkDatasetError("eligible category coverage does not match the pin")

    return _SourceData(
        root=root,
        file_sha256=_sha256(source_payload),
        file_size_bytes=len(source_payload),
        canonical_sha256=canonical_digest,
        canonical_size_bytes=len(canonical),
        license=dict(license_record),
        images=images,
        annotations=annotations,
        annotations_by_image={
            image_id: tuple(sorted(rows, key=lambda row: int(row["id"])))
            for image_id, rows in by_image.items()
        },
        categories=categories,
        eligible_image_ids=frozenset(eligible),
        available_category_ids=available_categories,
        unavailable_category_ids=unavailable_categories,
    )


def _rank_digest(image_id: int) -> str:
    payload = _SELECTION_SALT.encode("utf-8") + b"\0" + str(image_id).encode("ascii")
    return _sha256(payload)


def _select_images(
    source: _SourceData,
) -> tuple[tuple[int, ...], dict[int, str], int]:
    ranks = {image_id: _rank_digest(image_id) for image_id in source.eligible_image_ids}
    categories_by_image = {
        image_id: {
            int(annotation["category_id"])
            for annotation in source.annotations_by_image[image_id]
        }
        for image_id in source.eligible_image_ids
    }
    selected: set[int] = set()
    covered: set[int] = set()
    for category_id in sorted(source.available_category_ids):
        if category_id in covered:
            continue
        candidates = [
            image_id
            for image_id in source.eligible_image_ids
            if category_id in categories_by_image[image_id]
        ]
        if not candidates:
            raise BenchmarkDatasetError(
                f"eligible category {category_id} has no selectable image"
            )
        chosen = min(candidates, key=lambda image_id: (ranks[image_id], image_id))
        selected.add(chosen)
        covered.update(categories_by_image[chosen])

    coverage_seed = sorted(selected, key=lambda image_id: (ranks[image_id], image_id))
    if len(coverage_seed) != _SOURCE_CONTRACT.coverage_seed_image_count:
        raise BenchmarkDatasetError("coverage seed count does not match the pin")
    if len(coverage_seed) > _HOLDOUT_COUNT:
        raise BenchmarkDatasetError(
            "the fixed holdout cannot cover every eligible source category"
        )
    remaining = sorted(
        source.eligible_image_ids - selected,
        key=lambda image_id: (ranks[image_id], image_id),
    )
    ordered = tuple((coverage_seed + remaining)[:_SELECTED_COUNT])
    if len(ordered) != _SELECTED_COUNT or len(set(ordered)) != _SELECTED_COUNT:
        raise BenchmarkDatasetError(
            f"source cannot provide {_SELECTED_COUNT} unique eligible images"
        )
    holdout_categories = {
        int(annotation["category_id"])
        for image_id in ordered[:_HOLDOUT_COUNT]
        for annotation in source.annotations_by_image[image_id]
    }
    if holdout_categories != set(source.available_category_ids):
        raise BenchmarkDatasetError(
            "the fixed holdout does not cover every eligible source category"
        )
    return ordered, ranks, len(coverage_seed)


def _selected_image_path(images_dir: Path, image: Mapping[str, Any]) -> Path:
    filename = str(image["file_name"])
    candidate = images_dir / filename
    if candidate.is_symlink():
        raise BenchmarkDatasetError(
            f"selected image must not be a symlink: {candidate}"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise BenchmarkDatasetError(f"selected image is missing: {candidate}") from exc
    if not resolved.is_file() or resolved.parent != images_dir:
        raise BenchmarkDatasetError(
            f"selected image must be a direct regular file under {images_dir}: {candidate}"
        )
    return resolved


def _inspect_images(
    source: _SourceData,
    selected_ids: Sequence[int],
    ranks: Mapping[int, str],
    images_dir: Path,
) -> list[dict[str, Any]]:
    manifest_images = []
    for rank_index, image_id in enumerate(selected_ids):
        image = source.images[image_id]
        path = _selected_image_path(images_dir, image)
        first_digest, first_size = _file_identity(path)
        expected_size = (int(image["width"]), int(image["height"]))
        if (
            Image.MAX_IMAGE_PIXELS is not None
            and expected_size[0] * expected_size[1] > Image.MAX_IMAGE_PIXELS
        ):
            raise BenchmarkDatasetError(
                f"selected image exceeds the safe pixel limit: {path}"
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as opened:
                    actual_size = tuple(int(value) for value in opened.size)
                    actual_format = opened.format
                    opened.load()
        except (
            OSError,
            UnidentifiedImageError,
            ValueError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise BenchmarkDatasetError(
                f"selected image is not a valid JPEG: {path}"
            ) from exc
        if actual_format != "JPEG":
            raise BenchmarkDatasetError(f"selected image is not JPEG content: {path}")
        if actual_size != expected_size:
            raise BenchmarkDatasetError(
                f"selected image dimensions disagree with source metadata: {path}"
            )
        second_digest, second_size = _file_identity(path)
        if (first_digest, first_size) != (second_digest, second_size):
            raise BenchmarkDatasetError(
                f"selected image changed while inspected: {path}"
            )
        image_annotations = source.annotations_by_image[image_id]
        manifest_images.append(
            {
                "rank_index": rank_index,
                "rank_sha256": ranks[image_id],
                "image_id": image_id,
                "file_name": str(image["file_name"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "size_bytes": first_size,
                "sha256": first_digest,
                "license_id": int(image["license"]),
                "source_url": str(image["flickr_url"]),
                "distribution_url": str(image["coco_url"]),
                "annotation_ids": [int(row["id"]) for row in image_annotations],
                "annotations_sha256": _sha256(_canonical_json(list(image_annotations))),
            }
        )
    return manifest_images


def _selected_image_identity(
    manifest_images: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        {
            "image_id": image["image_id"],
            "file_name": image["file_name"],
            "size_bytes": image["size_bytes"],
            "sha256": image["sha256"],
        }
        for image in manifest_images
    ]
    payload = _canonical_json(rows)
    identity = {
        "canonicalization": _SELECTED_IMAGE_CANONICALIZATION,
        "canonical_size_bytes": len(payload),
        "selected_image_bytes_total": sum(int(row["size_bytes"]) for row in rows),
        "sha256": _sha256(payload),
        "publisher_archive_member_pin_enforced": (
            _SOURCE_CONTRACT.selected_image_identity_sha256 is not None
        ),
    }
    expected = (
        _SOURCE_CONTRACT.selected_image_identity_sha256,
        _SOURCE_CONTRACT.selected_image_identity_size_bytes,
        _SOURCE_CONTRACT.selected_image_bytes_total,
    )
    if any(value is not None for value in expected):
        if any(value is None for value in expected):
            raise BenchmarkDatasetError(
                "selected-image source contract is only partially defined"
            )
        actual = (
            identity["sha256"],
            identity["canonical_size_bytes"],
            identity["selected_image_bytes_total"],
        )
        if actual != expected:
            raise BenchmarkDatasetError(
                "selected image bytes do not match the pinned official archive members"
            )
    return identity


def _filtered_annotations(
    source: _SourceData, image_ids: Sequence[int]
) -> dict[str, Any]:
    order = {image_id: index for index, image_id in enumerate(image_ids)}
    annotations = [
        annotation
        for image_id in image_ids
        for annotation in source.annotations_by_image[image_id]
    ]
    annotations.sort(
        key=lambda row: (
            order[int(row["image_id"])],
            int(row["category_id"]),
            int(row["id"]),
        )
    )
    return {
        "info": source.root["info"],
        "licenses": [source.license],
        "images": [source.images[image_id] for image_id in image_ids],
        "annotations": annotations,
        "categories": [source.categories[key] for key in sorted(source.categories)],
    }


def _attribution_bytes(source: _SourceData, selected_ids: Sequence[int]) -> bytes:
    lines = []
    for image_id in selected_ids:
        image = source.images[image_id]
        row = {
            "image_id": image_id,
            "file_name": str(image["file_name"]),
            "license_id": _LICENSE_ID,
            "license_spdx": _LICENSE_SPDX,
            "license_name": _LICENSE_NAME,
            "license_url": _LICENSE_URL,
            "source_url": str(image["flickr_url"]),
            "distribution_url": str(image["coco_url"]),
            "creator": None,
            "title": None,
            "creator_supplied_by_source": False,
        }
        lines.append(_canonical_json(row))
    return b"\n".join(lines) + b"\n"


def _annotation_notice_bytes() -> bytes:
    return (
        "COCO 2017 annotation subset notice\n\n"
        f"Source: COCO 2017 {_SOURCE_FILE_NAME}\n"
        f"Source archive: {_ANNOTATION_ARCHIVE_URL}\n"
        f"Rights holder and attribution party: {_ANNOTATION_RIGHTS_HOLDER}\n"
        f"License: {_ANNOTATION_LICENSE_NAME} ({_ANNOTATION_LICENSE_SPDX})\n"
        f"License text: {_ANNOTATION_LICENSE_URL}\n"
        f"COCO terms: {_ANNOTATION_TERMS_URL}\n\n"
        "Modification notice: LibreYOLO selected, filtered, reordered, and summarized "
        "records from the source annotation file to create the annotation JSON, "
        "attribution, and manifest artifacts in this bundle. No image bytes are "
        "included. Image copyrights remain with their respective owners and are "
        "governed separately.\n"
    ).encode("utf-8")


def _partition_rows(
    source: _SourceData, partition_ids: Mapping[str, Sequence[int]]
) -> list[dict[str, Any]]:
    definitions = (
        (
            "holdout100",
            0,
            _HOLDOUT_COUNT,
            ["confidence_smoke", "fine_tune_validation"],
        ),
        (
            "train400",
            _HOLDOUT_COUNT,
            _SELECTED_COUNT,
            ["fine_tune_training"],
        ),
        (
            "promotion500",
            0,
            _SELECTED_COUNT,
            ["zero_shot_confidence_promotion"],
        ),
    )
    all_category_ids = set(source.categories)
    rows = []
    for name, start, stop, roles in definitions:
        category_ids = sorted(
            {
                int(annotation["category_id"])
                for image_id in partition_ids[name]
                for annotation in source.annotations_by_image[image_id]
            }
        )
        rows.append(
            {
                "name": name,
                "start": start,
                "stop": stop,
                "roles": roles,
                "annotation_artifact": name,
                "represented_category_count": len(category_ids),
                "represented_category_ids": category_ids,
                "unrepresented_category_ids": sorted(
                    all_category_ids - set(category_ids)
                ),
            }
        )
    expected = dict(_SOURCE_CONTRACT.partition_unrepresented_category_ids)
    if expected:
        actual = {row["name"]: tuple(row["unrepresented_category_ids"]) for row in rows}
        if actual != expected:
            raise BenchmarkDatasetError(
                "partition category coverage does not match the pinned source contract"
            )
    return rows


def _derive_bundle(source_annotations: Path, images_dir: Path) -> _DerivedBundle:
    source = _load_source(source_annotations)
    selected_ids, ranks, coverage_seed_count = _select_images(source)
    manifest_images = _inspect_images(source, selected_ids, ranks, images_dir)
    selected_image_identity = _selected_image_identity(manifest_images)
    partition_ids = {
        "holdout100": selected_ids[:_HOLDOUT_COUNT],
        "train400": selected_ids[_HOLDOUT_COUNT:_SELECTED_COUNT],
        "promotion500": selected_ids[:_SELECTED_COUNT],
    }
    artifact_payloads = {
        _ARTIFACT_PATHS[name]: _json_file_bytes(
            _filtered_annotations(source, image_ids)
        )
        for name, image_ids in partition_ids.items()
    }
    artifact_payloads[_ATTRIBUTION_PATH] = _attribution_bytes(source, selected_ids)
    artifact_payloads[_ANNOTATION_NOTICE_PATH] = _annotation_notice_bytes()
    artifacts = {
        name: {
            "path": path,
            "size_bytes": len(artifact_payloads[path]),
            "sha256": _sha256(artifact_payloads[path]),
        }
        for name, path in _ARTIFACT_PATHS.items()
    }
    artifacts["attribution"] = {
        "path": _ATTRIBUTION_PATH,
        "size_bytes": len(artifact_payloads[_ATTRIBUTION_PATH]),
        "sha256": _sha256(artifact_payloads[_ATTRIBUTION_PATH]),
    }
    artifacts["annotation_notice"] = {
        "path": _ANNOTATION_NOTICE_PATH,
        "size_bytes": len(artifact_payloads[_ANNOTATION_NOTICE_PATH]),
        "sha256": _sha256(artifact_payloads[_ANNOTATION_NOTICE_PATH]),
    }
    manifest = {
        "schema": _SCHEMA,
        "source": {
            "dataset": "COCO 2017",
            "split": "val2017",
            "homepage": _SOURCE_HOMEPAGE,
            "image_bytes_included": False,
            "image_archive": {
                "url": _SOURCE_CONTRACT.image_archive_url,
                "size_bytes": _SOURCE_CONTRACT.image_archive_size_bytes,
                "sha256": _SOURCE_CONTRACT.image_archive_sha256,
                "etag": _SOURCE_CONTRACT.image_archive_etag,
                "last_modified": _SOURCE_CONTRACT.image_archive_last_modified,
                "read_by_builder": False,
            },
            "selected_image_identity": selected_image_identity,
            "annotation": {
                "file_name": _SOURCE_FILE_NAME,
                "canonicalization": _SOURCE_CANONICALIZATION,
                "canonical_size_bytes": source.canonical_size_bytes,
                "sha256": source.canonical_sha256,
                "image_count": len(source.images),
                "annotation_count": len(source.annotations),
                "category_count": len(source.categories),
            },
        },
        "license_gate": {
            "required_image_license_id": _LICENSE_ID,
            "spdx": _LICENSE_SPDX,
            "name": _LICENSE_NAME,
            "url": _LICENSE_URL,
        },
        "annotation_license": {
            "source": f"COCO 2017 {_SOURCE_FILE_NAME}",
            "source_archive_url": _ANNOTATION_ARCHIVE_URL,
            "rights_holder_and_attribution_party": _ANNOTATION_RIGHTS_HOLDER,
            "spdx": _ANNOTATION_LICENSE_SPDX,
            "name": _ANNOTATION_LICENSE_NAME,
            "url": _ANNOTATION_LICENSE_URL,
            "terms_url": _ANNOTATION_TERMS_URL,
            "terms_source": _ANNOTATION_TERMS_SOURCE,
            "artifacts_are_modified_derivatives": True,
            "derived_artifacts": [
                _MANIFEST_NAME,
                _ATTRIBUTION_PATH,
                *_ARTIFACT_PATHS.values(),
            ],
        },
        "selection": {
            "algorithm": _SELECTION_ALGORITHM,
            "salt": _SELECTION_SALT,
            "reject_image_if_any_iscrowd": True,
            "reject_image_if_any_ignore": True,
            "eligible_image_count": len(source.eligible_image_ids),
            "selected_image_count": len(selected_ids),
            "coverage_seed_image_count": coverage_seed_count,
            "available_category_count": len(source.available_category_ids),
            "unavailable_category_ids": list(source.unavailable_category_ids),
        },
        "partitions": _partition_rows(source, partition_ids),
        "artifacts": artifacts,
        "images": manifest_images,
        "manual_review": {
            "status": "required-outside-manifest",
            "checks": [
                "canonical_source",
                "image_attribution_sufficiency",
                "annotation_license_and_redistribution",
                "privacy_and_pii",
                "visual_quality",
                "selection_salt_freeze",
                "benchmark_suitability",
                "publication_upload_authorization",
            ],
        },
    }
    files = dict(artifact_payloads)
    files[_MANIFEST_NAME] = _json_file_bytes(manifest)
    return _DerivedBundle(
        manifest=manifest,
        files=files,
        source_file_sha256=source.file_sha256,
        source_file_size_bytes=source.file_size_bytes,
    )


def _safe_relative(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise BenchmarkDatasetError(f"unsafe metadata artifact path: {relative!r}")
    resolved = (root / path).resolve(strict=False)
    if root != resolved and root not in resolved.parents:
        raise BenchmarkDatasetError(f"metadata artifact escapes output: {relative!r}")
    return resolved


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _staged_directory(destination: Path) -> Iterator[Path]:
    if _path_exists(destination):
        raise BenchmarkDatasetOutputExistsError(
            f"metadata output already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(f".{destination.name}.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise BenchmarkDatasetOutputExistsError(
            f"metadata output is reserved by another process: {destination}"
        ) from exc
    stage: Path | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(f"{os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        stage = Path(
            tempfile.mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.tmp-",
            )
        ).resolve()
        yield stage
        if _path_exists(destination):
            raise BenchmarkDatasetOutputExistsError(
                f"metadata output appeared during build: {destination}"
            )
        stage.rename(destination)
        stage = None
    finally:
        if stage is not None and stage.is_dir():
            shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


def _verify_tree(root: Path, expected_files: set[str]) -> None:
    expected_directories = {
        parent.as_posix()
        for relative in expected_files
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise BenchmarkDatasetError(
                f"metadata bundle must not contain symlinks: {relative}"
            )
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise BenchmarkDatasetError(
                f"metadata bundle contains a non-regular entry: {relative}"
            )
    if actual_files != expected_files or actual_directories != expected_directories:
        raise BenchmarkDatasetError(
            "metadata bundle tree does not exactly match the manifest contract"
        )


def _verify_bundle_evidence(
    manifest: str | os.PathLike[str],
    source_annotations: str | os.PathLike[str],
    images_dir: str | os.PathLike[str],
) -> _VerifiedBundleEvidence:
    manifest_path = _required_file(manifest, "manifest")
    if manifest_path.name != _MANIFEST_NAME:
        raise BenchmarkDatasetError(f"manifest filename must be {_MANIFEST_NAME!r}")
    bundle_root = manifest_path.parent
    source_path = _required_file(source_annotations, "source annotations")
    image_root = _required_directory(images_dir, "source image directory")
    _assert_separate_bundle_root(bundle_root, image_root)
    loaded_manifest = _load_json(manifest_path, "benchmark dataset manifest")
    derived = _derive_bundle(source_path, image_root)
    if loaded_manifest != derived.manifest:
        raise BenchmarkDatasetError(
            "benchmark dataset manifest does not match reconstructed source evidence"
        )
    expected_files = set(derived.files)
    _verify_tree(bundle_root, expected_files)
    manifest_payload: bytes | None = None
    for relative, expected in derived.files.items():
        path = _safe_relative(bundle_root, relative)
        if path.is_symlink() or not path.is_file():
            raise BenchmarkDatasetError(f"metadata artifact is missing: {relative}")
        actual = _read_bounded(
            path,
            max_bytes=max(len(expected), 1),
            label=f"metadata artifact {relative}",
        )
        if actual != expected:
            raise BenchmarkDatasetError(f"metadata artifact was modified: {relative}")
        if relative == _MANIFEST_NAME:
            manifest_payload = actual
    if manifest_payload is None:  # pragma: no cover - fixed internal artifact contract
        raise RuntimeError("derived benchmark bundle omitted its manifest")
    return _VerifiedBundleEvidence(
        manifest_path=manifest_path,
        bundle_root=bundle_root,
        source_annotations=source_path,
        images_dir=image_root,
        manifest_sha256=_sha256(manifest_payload),
        derived=derived,
    )


def verify_benchmark_dataset(
    manifest: str | os.PathLike[str],
    source_annotations: str | os.PathLike[str],
    images_dir: str | os.PathLike[str],
) -> VerifiedBenchmarkDataset:
    """Rebuild all expected metadata and verify an immutable local bundle.

    Paths are operational inputs and are never stored in the manifest.  All
    selected image bytes are hashed and decoded in place, without writes.
    """

    evidence = _verify_bundle_evidence(manifest, source_annotations, images_dir)
    return VerifiedBenchmarkDataset(
        output_dir=evidence.bundle_root,
        manifest_path=evidence.manifest_path,
        manifest_sha256=evidence.manifest_sha256,
        selected_image_count=len(evidence.derived.manifest["images"]),
    )


def _verified_review_attestation(
    path: Path,
    *,
    manifest_sha256: str,
    partition_role: str,
    expected_checks: Sequence[str],
) -> tuple[str, Mapping[str, Any]]:
    payload = _read_bounded(
        path,
        max_bytes=_MAX_REVIEW_ATTESTATION_BYTES,
        label="benchmark review attestation",
    )
    root = _exact_fields(
        _decode_json(payload, "benchmark review attestation"),
        {
            "schema",
            "manifest_sha256",
            "partition_role",
            "status",
            "reviewer",
            "reviewed_at",
            "checks",
        },
        "benchmark review attestation",
    )
    if root["schema"] != _REVIEW_SCHEMA:
        raise BenchmarkDatasetError(
            "benchmark review attestation schema is unsupported"
        )
    claimed_manifest = root["manifest_sha256"]
    if (
        not isinstance(claimed_manifest, str)
        or len(claimed_manifest) != 64
        or any(character not in "0123456789abcdef" for character in claimed_manifest)
    ):
        raise BenchmarkDatasetError(
            "benchmark review attestation manifest_sha256 must be lowercase SHA-256"
        )
    if claimed_manifest != manifest_sha256:
        raise BenchmarkDatasetError(
            "benchmark review attestation does not bind the verified manifest"
        )
    if root["partition_role"] != partition_role:
        raise BenchmarkDatasetError(
            "benchmark review attestation does not approve the required partition role"
        )
    if root["status"] != _REVIEW_STATUS:
        raise BenchmarkDatasetError("benchmark review attestation is not approved")
    reviewer = _nonempty_string(
        root["reviewer"], "benchmark review attestation reviewer"
    ).strip()
    if len(reviewer) > 256:
        raise BenchmarkDatasetError(
            "benchmark review attestation reviewer exceeds 256 characters"
        )
    reviewed_at = _utc_rfc3339(
        root["reviewed_at"], "benchmark review attestation reviewed_at"
    )
    checks = _exact_fields(
        root["checks"],
        set(expected_checks),
        "benchmark review attestation checks",
    )
    for check in expected_checks:
        if type(checks[check]) is not bool or checks[check] is not True:
            raise BenchmarkDatasetError(
                f"benchmark review attestation check {check!r} must be true"
            )
    normalized_checks = MappingProxyType({check: True for check in expected_checks})
    normalized = MappingProxyType(
        {
            "schema": _REVIEW_SCHEMA,
            "manifest_sha256": manifest_sha256,
            "partition_role": partition_role,
            "status": _REVIEW_STATUS,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "checks": normalized_checks,
        }
    )
    return _sha256(payload), normalized


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _immutable_rows(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    rows = _sequence(value, label)
    immutable = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):  # pragma: no cover - derived internally
            raise RuntimeError(f"{label}[{index}] is not an object")
        immutable.append(_freeze_json(row))
    return tuple(immutable)


def verify_benchmark_run_inputs(
    manifest: str | os.PathLike[str],
    source_annotations: str | os.PathLike[str],
    images_dir: str | os.PathLike[str],
    review_attestation: str | os.PathLike[str],
    *,
    required_role: str = _PROMOTION_ROLE,
) -> VerifiedBenchmarkRunInputs:
    """Verify the fixed promotion dataset and its external human-review assertion.

    The attestation records a human declaration; it does not authenticate the
    reviewer's identity. No network access or source-tree mutation is performed.
    """

    if required_role != _PROMOTION_ROLE:
        raise BenchmarkDatasetError(
            f"benchmark partition role must be {_PROMOTION_ROLE!r}"
        )
    attestation_path = _required_file(
        review_attestation, "benchmark review attestation"
    )
    manifest_candidate = _required_file(manifest, "manifest")
    bundle_candidate = manifest_candidate.parent
    if (
        attestation_path == bundle_candidate
        or bundle_candidate in attestation_path.parents
    ):
        raise BenchmarkDatasetError(
            "benchmark review attestation must remain outside the metadata bundle"
        )
    evidence = _verify_bundle_evidence(
        manifest_candidate, source_annotations, images_dir
    )
    derived_manifest = evidence.derived.manifest
    matching_partitions = [
        partition
        for partition in derived_manifest["partitions"]
        if required_role in partition["roles"]
    ]
    if len(matching_partitions) != 1:  # pragma: no cover - derived fixed contract
        raise BenchmarkDatasetError(
            "benchmark manifest must contain exactly one required partition role"
        )
    partition = matching_partitions[0]
    artifact_name = str(partition["annotation_artifact"])
    artifact = derived_manifest["artifacts"][artifact_name]
    relative_annotation_path = str(artifact["path"])
    annotation_path = _safe_relative(evidence.bundle_root, relative_annotation_path)
    annotation_payload = evidence.derived.files[relative_annotation_path]
    annotation_root = _exact_fields(
        _decode_json(annotation_payload, f"benchmark artifact {artifact_name}"),
        set(_SOURCE_FIELDS),
        f"benchmark artifact {artifact_name}",
    )
    expected_categories = _immutable_rows(
        annotation_root["categories"], "benchmark expected categories"
    )
    class_names = tuple(
        _nonempty_string(category["name"], "benchmark category name")
        for category in expected_categories
    )
    manual_review = _exact_fields(
        derived_manifest["manual_review"],
        {"status", "checks"},
        "benchmark manifest manual_review",
    )
    if manual_review["status"] != "required-outside-manifest":
        raise BenchmarkDatasetError(
            "benchmark manifest does not require an external review attestation"
        )
    expected_checks = tuple(
        _nonempty_string(check, "benchmark manifest manual review check")
        for check in _sequence(
            manual_review["checks"], "benchmark manifest manual_review.checks"
        )
    )
    if len(set(expected_checks)) != len(expected_checks):
        raise BenchmarkDatasetError("benchmark manifest review checks are not unique")
    review_sha256, normalized_review = _verified_review_attestation(
        attestation_path,
        manifest_sha256=evidence.manifest_sha256,
        partition_role=required_role,
        expected_checks=expected_checks,
    )
    partition_start = int(partition["start"])
    partition_stop = int(partition["stop"])
    return VerifiedBenchmarkRunInputs(
        manifest_path=evidence.manifest_path,
        manifest_sha256=evidence.manifest_sha256,
        source_annotations=evidence.source_annotations,
        source_canonical_sha256=str(derived_manifest["source"]["annotation"]["sha256"]),
        source_file_sha256=evidence.derived.source_file_sha256,
        source_file_size_bytes=evidence.derived.source_file_size_bytes,
        images_dir=evidence.images_dir,
        selected_image_identity_sha256=str(
            derived_manifest["source"]["selected_image_identity"]["sha256"]
        ),
        partition_name=str(partition["name"]),
        partition_role=required_role,
        partition_start=partition_start,
        partition_stop=partition_stop,
        annotation_path=annotation_path,
        annotation_sha256=str(artifact["sha256"]),
        annotation_size_bytes=int(artifact["size_bytes"]),
        class_names=class_names,
        expected_images=_immutable_rows(
            derived_manifest["images"][partition_start:partition_stop],
            "benchmark expected images",
        ),
        expected_categories=expected_categories,
        expected_annotations=_immutable_rows(
            annotation_root["annotations"], "benchmark expected annotations"
        ),
        review_attestation_path=attestation_path,
        review_attestation_sha256=review_sha256,
        review_attestation=normalized_review,
    )


def build_benchmark_dataset(
    source_annotations: str | os.PathLike[str],
    images_dir: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
) -> BenchmarkDatasetArtifacts:
    """Build and self-verify the pinned metadata-only benchmark bundle."""

    source_path = _required_file(source_annotations, "source annotations")
    image_root = _required_directory(images_dir, "source image directory")
    requested = _path_argument(output_root, "output root")
    if _path_exists(requested):
        raise BenchmarkDatasetOutputExistsError(
            f"metadata output already exists: {requested}"
        )
    destination = requested.resolve(strict=False)
    _assert_separate_bundle_root(destination, image_root)
    derived = _derive_bundle(source_path, image_root)
    with _staged_directory(destination) as stage:
        for relative, payload in derived.files.items():
            _write_bytes(_safe_relative(stage, relative), payload)
        verified = verify_benchmark_dataset(
            stage / _MANIFEST_NAME,
            source_path,
            image_root,
        )
        manifest_sha256 = verified.manifest_sha256
    return BenchmarkDatasetArtifacts(
        output_dir=destination,
        manifest_path=destination / _MANIFEST_NAME,
        manifest_sha256=manifest_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m libreyolo.validation.vlm_benchmark_dataset"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    build = subparsers.add_parser("build", help="build local metadata artifacts")
    build.add_argument("--annotations", required=True, type=Path)
    build.add_argument("--images-dir", required=True, type=Path)
    build.add_argument("--output-root", required=True, type=Path)
    verify = subparsers.add_parser("verify", help="verify local metadata artifacts")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--annotations", required=True, type=Path)
    verify.add_argument("--images-dir", required=True, type=Path)
    return parser


def _status(value: Mapping[str, Any]) -> None:
    payload = _json_file_bytes(value)
    binary = getattr(sys.stdout, "buffer", None)
    if binary is not None:
        binary.write(payload)
    else:
        sys.stdout.write(payload.decode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "build":
            result = build_benchmark_dataset(
                args.annotations,
                args.images_dir,
                args.output_root,
            )
            _status(
                {
                    "schema": _STATUS_SCHEMA,
                    "status": "ok",
                    "mode": "build",
                    "output_root": str(result.output_dir),
                    "manifest": str(result.manifest_path),
                    "manifest_sha256": result.manifest_sha256,
                }
            )
        else:
            result = verify_benchmark_dataset(
                args.manifest,
                args.annotations,
                args.images_dir,
            )
            _status(
                {
                    "schema": _STATUS_SCHEMA,
                    "status": "ok",
                    "mode": "verify",
                    "output_root": str(result.output_dir),
                    "manifest": str(result.manifest_path),
                    "manifest_sha256": result.manifest_sha256,
                    "selected_image_count": result.selected_image_count,
                }
            )
        return 0
    except BenchmarkDatasetOutputExistsError as exc:
        _status(
            {
                "schema": _STATUS_SCHEMA,
                "status": "error",
                "mode": args.mode,
                "kind": "output_exists",
                "message": str(exc),
            }
        )
        return 3
    except (BenchmarkDatasetError, OSError) as exc:
        _status(
            {
                "schema": _STATUS_SCHEMA,
                "status": "error",
                "mode": args.mode,
                "kind": "invalid_dataset",
                "message": str(exc),
            }
        )
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
