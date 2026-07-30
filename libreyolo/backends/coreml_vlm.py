"""Portable bundle and strict host runtime for SmolVLM2 Core ML.

This backend is deliberately separate from the one-shot ``CoreMLBackend``.
Generative VLM inference requires three named functions, a request-local Core
ML state, an append-only cache cursor, tokenizer/processor assets, and a host
generation loop.

Provenance
----------
The model and processor contract is for
``HuggingFaceTB/SmolVLM2-500M-Video-Instruct`` at revision
``7b375e1b73b11138ff12fe22c8f2822d8fe03467`` (Apache-2.0). The host equations
and processor behavior are aligned with Hugging Face Transformers 5.12.1 at
commit ``ddb849abe009d1089e6c691bfc897f27211c663c`` (Apache-2.0). This module
contains an independent LibreYOLO implementation under MIT and imports the
conversion contract from :mod:`libreyolo.export.coreml_vlm`.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from ..export.coreml_vlm import (
    COREML_VLM_CAUSAL_MASK_INPUT,
    COREML_VLM_DECODE_FUNCTION,
    COREML_VLM_EMBED_TOKENS_FUNCTION,
    COREML_VLM_ENCODE_IMAGE_FUNCTION,
    COREML_VLM_FUNCTION_NAMES,
    COREML_VLM_IMAGE_EMBEDDINGS_OUTPUT,
    COREML_VLM_INPUT_IDS_INPUT,
    COREML_VLM_LAST_LOGITS_OUTPUT,
    COREML_VLM_PIXEL_VALUES_INPUT,
    COREML_VLM_POSITION_IDS_INPUT,
    COREML_VLM_TOKEN_EMBEDDINGS_INPUT,
    COREML_VLM_TOKEN_EMBEDDINGS_OUTPUT,
    COREML_VLM_TRANSFORMERS_COMMIT,
    COREML_VLM_TRANSFORMERS_VERSION,
    CoreMLVLMDecodeCursor,
    CoreMLVLMProfile,
    SMOLVLM2_500M_COMPONENT_CONTRACT,
    SMOLVLM2_500M_EOS_TOKEN_ID,
    SMOLVLM2_500M_REPETITION_PENALTY,
    SMOLVLM2_500M_REPO,
    SMOLVLM2_500M_REQUIRED_ASSETS,
    SMOLVLM2_500M_REVISION,
    SMOLVLM2_500M_WEIGHTS_FILENAME,
    SMOLVLM2_500M_WEIGHTS_SHA256,
    SMOLVLM2_500M_WEIGHTS_SIZE,
    _publish_directory_no_replace,
    merge_coreml_vlm_image_embeddings,
    prepare_smolvlm2_500m_coreml_processor_batch,
    preprocess_smolvlm2_500m_coreml_image,
    require_coreml_vlm_toolchain,
    require_coreml_vlm_transformers_toolchain,
    smolvlm2_500m_coreml_metadata,
    smolvlm2_500m_coreml_profile,
    stringify_coreml_vlm_metadata,
    validate_coreml_vlm_metadata,
    validate_coreml_vlm_multifunction_spec,
    validate_smolvlm2_500m_processor_assets,
)
from ..export.coreml_profiles import resolve_coreml_runtime_compute_units
from ..models.vlm.parsing import build_detection_dict, extract_detections
from ..utils.image_loader import ImageInput, ImageLoader


COREML_VLM_BUNDLE_FORMAT = "libreyolo_coreml_vlm_bundle"
COREML_VLM_BUNDLE_SCHEMA_VERSION = 1
COREML_VLM_BUNDLE_SUFFIX = ".coremlvlm"
COREML_VLM_BUNDLE_MANIFEST = "manifest.json"
COREML_VLM_BUNDLE_MODEL_ROOT = "Model.mlpackage"
COREML_VLM_BUNDLE_PROCESSOR_ROOT = "Processor"
COREML_VLM_BUNDLE_APACHE_LICENSE = "LICENSES/Apache-2.0.txt"
COREML_VLM_BUNDLE_NOTICE = "NOTICE.txt"
COREML_VLM_RUNTIME_CONTEXTS = (2048, 4096)

_APACHE_2_CANONICAL_SIZE = 11_357
_APACHE_2_CANONICAL_SHA256 = (
    "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
)

_COREML_VLM_BUNDLE_KEYS = frozenset(
    {
        "bundle_format",
        "bundle_schema_version",
        "component_contract",
        "model_path",
        "processor_path",
        "profile",
        "coreml_contract_sha256",
        "processor",
        "source_weights_included",
        "provenance",
        "licenses",
        "notice",
        "payload_files",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apache_2_license_bytes() -> bytes:
    """Load LibreYOLO's shipped license copy and normalize it for the bundle."""

    candidates = [
        Path(__file__).resolve().parents[2] / "licenses" / "Apache-2.0.txt"
    ]
    try:
        distribution = importlib.metadata.distribution("libreyolo")
        for entry in distribution.files or ():
            normalized = str(entry).replace("\\", "/")
            if normalized.endswith("/licenses/Apache-2.0.txt"):
                candidates.append(Path(distribution.locate_file(entry)))
    except importlib.metadata.PackageNotFoundError:
        pass
    checked: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved in checked or candidate.is_symlink() or not resolved.is_file():
            continue
        checked.add(resolved)
        canonical = resolved.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(canonical).hexdigest()
        if (
            len(canonical) == _APACHE_2_CANONICAL_SIZE
            and hmac.compare_digest(digest, _APACHE_2_CANONICAL_SHA256)
        ):
            return canonical
    raise RuntimeError(
        "LibreYOLO's canonical Apache-2.0 license asset is missing or modified."
    )


def _bundle_notice_bytes() -> bytes:
    return (
        "LibreYOLO Core ML VLM bundle\n"
        "\n"
        "Model and processor:\n"
        f"  {SMOLVLM2_500M_REPO}\n"
        f"  revision {SMOLVLM2_500M_REVISION}\n"
        "  license Apache-2.0\n"
        "\n"
        "Conversion and host-semantics reference:\n"
        "  https://github.com/huggingface/transformers\n"
        f"  commit {COREML_VLM_TRANSFORMERS_COMMIT}\n"
        f"  version {COREML_VLM_TRANSFORMERS_VERSION}\n"
        "  license Apache-2.0\n"
        "\n"
        "The bundle contains converted Core ML weights and exactly the pinned\n"
        "processor/tokenizer assets. It does not contain model.safetensors.\n"
        "See LICENSES/Apache-2.0.txt for the full license terms.\n"
    ).encode("ascii")


def _ensure_contained(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its declared root: {path}.") from exc
    return resolved


def _safe_tree_files(root: Path, *, label: str) -> list[tuple[Path, Path]]:
    """Return sorted regular files without following links or special nodes."""

    if root.is_symlink():
        raise ValueError(f"{label} root must not be a symbolic link: {root}.")
    if not root.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {root}.")
    resolved_root = root.resolve(strict=True)
    files: list[tuple[Path, Path]] = []
    for directory, directory_names, file_names in os.walk(
        resolved_root,
        topdown=True,
        followlinks=False,
    ):
        directory_names.sort()
        file_names.sort()
        current = Path(directory)
        for name in tuple(directory_names):
            child = current / name
            if child.is_symlink():
                raise ValueError(f"{label} contains a symbolic link: {child}.")
            mode = child.stat(follow_symlinks=False).st_mode
            if not stat.S_ISDIR(mode):
                raise ValueError(f"{label} contains a non-directory node: {child}.")
            _ensure_contained(child, resolved_root, label=label)
        for name in file_names:
            child = current / name
            if child.is_symlink():
                raise ValueError(f"{label} contains a symbolic link: {child}.")
            mode = child.stat(follow_symlinks=False).st_mode
            if not stat.S_ISREG(mode):
                raise ValueError(f"{label} contains a non-regular file: {child}.")
            resolved = _ensure_contained(child, resolved_root, label=label)
            files.append((resolved.relative_to(resolved_root), resolved))
    return sorted(files, key=lambda item: item[0].as_posix())


def _validate_bundle_destination(path: Path) -> None:
    if path.suffix.lower() != COREML_VLM_BUNDLE_SUFFIX:
        raise ValueError(
            "Core ML VLM bundle output must end in "
            f"{COREML_VLM_BUNDLE_SUFFIX!r}."
        )
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite Core ML VLM bundle: {path}.")


def _runtime_profile(context_length: Any) -> CoreMLVLMProfile:
    if isinstance(context_length, bool):
        raise ValueError("Core ML VLM runtime context_length must be an integer.")
    try:
        context = int(context_length)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Core ML VLM runtime context_length must be an integer."
        ) from exc
    if context not in COREML_VLM_RUNTIME_CONTEXTS:
        raise ValueError(
            "Core ML VLM host runtime currently permits only the reviewed "
            f"2K/4K contexts {list(COREML_VLM_RUNTIME_CONTEXTS)}; got {context}. "
            "The 8K graph is intentionally rejected until its peak prefill "
            "memory is validated on Apple hardware."
        )
    return smolvlm2_500m_coreml_profile(context)


def _metadata_from_spec(spec: Any) -> dict[str, str]:
    description = getattr(spec, "description", None)
    metadata = getattr(description, "metadata", None)
    values = getattr(metadata, "userDefined", None)
    if values is None:
        return {}
    return {str(key): str(value) for key, value in dict(values).items()}


def _load_package_contract(
    package_path: Path,
    *,
    coremltools_module: Any,
) -> tuple[CoreMLVLMProfile, dict[str, Any], dict[str, str]]:
    """Validate a source package without compiling or invoking it."""

    if package_path.is_symlink():
        raise ValueError("Core ML VLM package must not be a symbolic link.")
    if (
        not package_path.is_dir()
        or package_path.suffix.lower() != ".mlpackage"
    ):
        raise ValueError(
            "Core ML VLM model must be an .mlpackage directory: "
            f"{package_path}."
        )
    package_files = _safe_tree_files(
        package_path,
        label="Core ML VLM package",
    )
    _validate_apple_package_manifest(package_path, package_files)
    spec = coremltools_module.utils.load_spec(str(package_path))
    metadata = _metadata_from_spec(spec)
    validated = validate_coreml_vlm_metadata(metadata)
    profile = _runtime_profile(validated["vlm_profile"]["context_length"])
    if validated["vlm_profile"] != profile.as_dict():
        raise ValueError(
            "Core ML VLM package metadata conflicts with its runtime profile."
        )
    validate_coreml_vlm_multifunction_spec(spec, profile=profile)
    return profile, validated, metadata


def _package_ct(coremltools_module: Any | None = None) -> Any:
    if coremltools_module is not None:
        ct = coremltools_module
    else:
        try:
            import coremltools as ct
        except ImportError as exc:
            raise ImportError(
                "Core ML VLM bundles require coremltools 9.x. Install the "
                "dedicated Core ML VLM dependencies."
            ) from exc
    require_coreml_vlm_toolchain(ct)
    return ct


def _payload_record(path: Path) -> dict[str, Any]:
    return {
        "size_bytes": int(path.stat().st_size),
        "sha256": _file_sha256(path),
    }


def _validate_apple_package_manifest(
    package_root: Path,
    package_files: list[tuple[Path, Path]],
) -> None:
    """Reject package files not covered by Apple's own package manifest."""

    by_name = {
        relative.as_posix(): path for relative, path in package_files
    }
    manifest_path = by_name.get("Manifest.json")
    if manifest_path is None:
        raise ValueError("Core ML VLM package is missing Apple's Manifest.json.")
    manifest = _load_json_object(manifest_path)
    if manifest.get("fileFormatVersion") != "1.0.0":
        raise ValueError("Core ML VLM package has an unknown file format version.")
    entries = manifest.get("itemInfoEntries")
    root_identifier = manifest.get("rootModelIdentifier")
    if (
        not isinstance(entries, dict)
        or not entries
        or not isinstance(root_identifier, str)
        or root_identifier not in entries
    ):
        raise ValueError("Core ML VLM package manifest entries are malformed.")
    covered = {"Manifest.json"}
    for identifier, raw_entry in entries.items():
        if not isinstance(identifier, str) or not isinstance(raw_entry, dict):
            raise ValueError("Core ML VLM package manifest entry is malformed.")
        raw_path = raw_entry.get("path")
        path = _validate_payload_name(raw_path)
        data_name = (PurePosixPath("Data") / path).as_posix()
        data_path = package_root / Path(data_name)
        _ensure_contained(
            data_path,
            package_root,
            label="Core ML package manifest path",
        )
        if data_path.is_file():
            covered.add(data_name)
        elif data_path.is_dir():
            prefix = data_name.rstrip("/") + "/"
            nested = {
                name for name in by_name if name.startswith(prefix)
            }
            if not nested:
                raise ValueError(
                    "Core ML package manifest points to an empty payload "
                    f"directory: {path!r}."
                )
            covered.update(nested)
        else:
            raise ValueError(
                f"Core ML package manifest payload does not exist: {path!r}."
            )
    if set(by_name) != covered:
        extras = sorted(set(by_name) - covered)
        raise ValueError(
            f"Core ML VLM package has unmanifested payload files: {extras}."
        )


def _is_source_weight_record(record: Mapping[str, Any]) -> bool:
    return (
        record.get("size_bytes") == SMOLVLM2_500M_WEIGHTS_SIZE
        and record.get("sha256") == SMOLVLM2_500M_WEIGHTS_SHA256
    )


def _verify_staged_payload(
    root: Path,
    payload: Mapping[str, Mapping[str, Any]],
) -> None:
    files = _safe_tree_files(root, label="staged Core ML VLM bundle")
    by_name = {
        relative.as_posix(): path
        for relative, path in files
        if relative.as_posix() != COREML_VLM_BUNDLE_MANIFEST
    }
    if set(by_name) != set(payload):
        raise RuntimeError("Staged Core ML VLM bundle inventory changed.")
    for name, expected in payload.items():
        actual = _payload_record(by_name[name])
        if actual != expected:
            raise RuntimeError(
                f"Staged Core ML VLM bundle payload {name!r} changed."
            )


def _manifest_without_payload(
    profile: CoreMLVLMProfile,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    notice = _bundle_notice_bytes()
    return {
        "bundle_format": COREML_VLM_BUNDLE_FORMAT,
        "bundle_schema_version": COREML_VLM_BUNDLE_SCHEMA_VERSION,
        "component_contract": SMOLVLM2_500M_COMPONENT_CONTRACT,
        "model_path": COREML_VLM_BUNDLE_MODEL_ROOT,
        "processor_path": COREML_VLM_BUNDLE_PROCESSOR_ROOT,
        "profile": profile.as_dict(),
        "coreml_contract_sha256": metadata["coreml_vlm_contract_sha256"],
        "processor": metadata["processor"],
        "source_weights_included": False,
        "licenses": {
            "Apache-2.0": {
                "path": COREML_VLM_BUNDLE_APACHE_LICENSE,
                "spdx": "Apache-2.0",
                "size_bytes": _APACHE_2_CANONICAL_SIZE,
                "sha256": _APACHE_2_CANONICAL_SHA256,
            }
        },
        "notice": {
            "path": COREML_VLM_BUNDLE_NOTICE,
            "size_bytes": len(notice),
            "sha256": hashlib.sha256(notice).hexdigest(),
        },
        "provenance": {
            "model": {
                "repo": SMOLVLM2_500M_REPO,
                "revision": SMOLVLM2_500M_REVISION,
                "license": "Apache-2.0",
            },
            "transformers": {
                "repo": "https://github.com/huggingface/transformers",
                "commit": COREML_VLM_TRANSFORMERS_COMMIT,
                "version": COREML_VLM_TRANSFORMERS_VERSION,
                "license": "Apache-2.0",
            },
            "bundle_runtime": {
                "project": "LibreYOLO",
                "license": "MIT",
            },
        },
    }


def _copy_regular_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as output:
        shutil.copyfileobj(source_handle, output, length=1024 * 1024)
    shutil.copystat(source, destination, follow_symlinks=False)


def build_coreml_vlm_bundle(
    package_path: str | os.PathLike[str],
    *,
    processor_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    move_model: bool = False,
    coremltools_module: Any | None = None,
) -> str:
    """Build one portable, no-overwrite SmolVLM2 Core ML bundle.

    ``move_model=False`` preserves an existing package and copies it into the
    bundle. Export orchestration may pass ``move_model=True`` for a freshly
    staged package; if publication fails after the move, the package is moved
    back to its original path.
    """

    source_package = Path(package_path)
    source_processor = Path(processor_dir)
    destination = Path(output_path)
    _validate_bundle_destination(destination)

    if source_package.is_symlink() or source_processor.is_symlink():
        raise ValueError("Core ML VLM bundle sources must not be symbolic links.")
    package_root = source_package.resolve(strict=True)
    processor_root = source_processor.resolve(strict=True)
    destination_parent = destination.parent.resolve(strict=False)
    for left, right, message in (
        (
            package_root,
            processor_root,
            "package and processor roots must not contain one another",
        ),
        (
            processor_root,
            package_root,
            "package and processor roots must not contain one another",
        ),
        (
            destination_parent,
            package_root,
            "bundle destination must not be inside the source package",
        ),
        (
            destination_parent,
            processor_root,
            "bundle destination must not be inside the processor snapshot",
        ),
    ):
        try:
            left.relative_to(right)
        except ValueError:
            continue
        raise ValueError(message + ".")
    destination.parent.mkdir(parents=True, exist_ok=True)

    ct = _package_ct(coremltools_module)
    profile, metadata, _ = _load_package_contract(
        package_root,
        coremltools_module=ct,
    )
    package_files = _safe_tree_files(
        package_root,
        label="Core ML VLM source package",
    )
    processor_files = _safe_tree_files(
        processor_root,
        label="SmolVLM2 processor snapshot",
    )
    validate_smolvlm2_500m_processor_assets(
        processor_root,
        revision=SMOLVLM2_500M_REVISION,
        transformers_version=COREML_VLM_TRANSFORMERS_VERSION,
    )
    processor_by_name = {
        relative.as_posix(): path for relative, path in processor_files
    }
    apache_license = _apache_2_license_bytes()
    bundle_notice = _bundle_notice_bytes()
    if set(processor_by_name) < set(SMOLVLM2_500M_REQUIRED_ASSETS):
        missing = sorted(
            set(SMOLVLM2_500M_REQUIRED_ASSETS) - set(processor_by_name)
        )
        raise FileNotFoundError(
            f"SmolVLM2 processor snapshot is missing {missing}."
        )

    payload: dict[str, dict[str, Any]] = {}
    for relative, path in package_files:
        if path.name == SMOLVLM2_500M_WEIGHTS_FILENAME:
            raise ValueError(
                "Refusing to bundle the 2 GB source safetensors payload; only "
                "converted Core ML weights belong inside Model.mlpackage."
            )
        name = (
            PurePosixPath(COREML_VLM_BUNDLE_MODEL_ROOT) / relative.as_posix()
        ).as_posix()
        _validate_payload_name(name)
        record = _payload_record(path)
        if _is_source_weight_record(record):
            raise ValueError(
                "Refusing to bundle the 2 GB source safetensors payload, even "
                "under a renamed package path."
            )
        payload[name] = record
    for name in sorted(SMOLVLM2_500M_REQUIRED_ASSETS):
        source = processor_by_name[name]
        bundle_name = (
            PurePosixPath(COREML_VLM_BUNDLE_PROCESSOR_ROOT) / name
        ).as_posix()
        _validate_payload_name(bundle_name)
        payload[bundle_name] = _payload_record(source)
    payload[COREML_VLM_BUNDLE_APACHE_LICENSE] = {
        "size_bytes": len(apache_license),
        "sha256": hashlib.sha256(apache_license).hexdigest(),
    }
    payload[COREML_VLM_BUNDLE_NOTICE] = {
        "size_bytes": len(bundle_notice),
        "sha256": hashlib.sha256(bundle_notice).hexdigest(),
    }

    manifest = {
        **_manifest_without_payload(profile, metadata),
        "payload_files": payload,
    }
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=".libreyolo-coreml-vlm-bundle-",
            dir=str(destination.parent),
        )
    )
    staged_model = temporary_root / COREML_VLM_BUNDLE_MODEL_ROOT
    model_was_moved = False
    remove_temporary_root = True
    try:
        processor_destination = (
            temporary_root / COREML_VLM_BUNDLE_PROCESSOR_ROOT
        )
        for name in sorted(SMOLVLM2_500M_REQUIRED_ASSETS):
            _copy_regular_file(
                processor_by_name[name],
                processor_destination / Path(name),
            )
        manifest_path = temporary_root / COREML_VLM_BUNDLE_MANIFEST
        manifest_path.write_text(
            _canonical_json(manifest) + "\n",
            encoding="utf-8",
        )
        license_path = temporary_root / Path(COREML_VLM_BUNDLE_APACHE_LICENSE)
        license_path.parent.mkdir(parents=True)
        with license_path.open("xb") as handle:
            handle.write(apache_license)
        with (temporary_root / COREML_VLM_BUNDLE_NOTICE).open("xb") as handle:
            handle.write(bundle_notice)
        if move_model:
            package_root.rename(staged_model)
            model_was_moved = True
        else:
            staged_model.mkdir()
            for relative, source in package_files:
                _copy_regular_file(source, staged_model / relative)

        _verify_staged_payload(temporary_root, payload)
        _validate_bundle_destination(destination)
        _publish_directory_no_replace(temporary_root, destination)
        model_was_moved = False
    except Exception as exc:
        if model_was_moved and staged_model.exists():
            if package_root.exists() or package_root.is_symlink():
                remove_temporary_root = False
                raise RuntimeError(
                    "Core ML VLM bundle publication failed and the moved model "
                    "could not be restored because its source path was "
                    f"reoccupied. The original is preserved at {staged_model}."
                ) from exc
            try:
                staged_model.rename(package_root)
                model_was_moved = False
            except Exception as restore_exc:
                remove_temporary_root = False
                raise RuntimeError(
                    "Core ML VLM bundle publication failed and the moved model "
                    f"could not be restored. It is preserved at {staged_model}."
                ) from restore_exc
        raise
    finally:
        if remove_temporary_root and temporary_root.exists():
            shutil.rmtree(temporary_root)
    return str(destination)


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Bundle manifest repeats key {key!r}.")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Core ML VLM bundle manifest is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError("Core ML VLM bundle manifest must be a JSON object.")
    return value


def _validate_payload_name(name: Any) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"Invalid Core ML VLM bundle payload path {name!r}.")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe Core ML VLM bundle payload path {name!r}.")
    canonical = path.as_posix()
    if canonical != name:
        raise ValueError(f"Non-canonical Core ML VLM bundle path {name!r}.")
    return canonical


def _validate_payload_record(name: str, value: Any) -> tuple[int, str]:
    if not isinstance(value, dict) or set(value) != {"size_bytes", "sha256"}:
        raise ValueError(f"Bundle payload record for {name!r} is malformed.")
    size = value["size_bytes"]
    digest = value["sha256"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"Bundle payload size for {name!r} is invalid.")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"Bundle payload SHA-256 for {name!r} is invalid.")
    return size, digest


@dataclass(frozen=True)
class CoreMLVLMBundleInfo:
    """Validated paths and exact contract for one portable VLM bundle."""

    path: Path
    model_path: Path
    processor_path: Path
    profile: CoreMLVLMProfile
    metadata: dict[str, Any]
    manifest: dict[str, Any]


def validate_coreml_vlm_bundle(
    bundle_path: str | os.PathLike[str],
    *,
    coremltools_module: Any | None = None,
) -> CoreMLVLMBundleInfo:
    """Validate paths, hashes, provenance, processor, and Core ML package ABI."""

    root = Path(bundle_path)
    if root.is_symlink():
        raise ValueError("Core ML VLM bundle root must not be a symbolic link.")
    if not root.is_dir() or root.suffix.lower() != COREML_VLM_BUNDLE_SUFFIX:
        raise ValueError(
            "Core ML VLM runtime requires a "
            f"{COREML_VLM_BUNDLE_SUFFIX} directory: {root}."
        )
    resolved_root = root.resolve(strict=True)
    all_files = _safe_tree_files(resolved_root, label="Core ML VLM bundle")
    by_name = {
        relative.as_posix(): path for relative, path in all_files
    }
    if COREML_VLM_BUNDLE_MANIFEST not in by_name:
        raise FileNotFoundError("Core ML VLM bundle is missing manifest.json.")
    manifest = _load_json_object(by_name[COREML_VLM_BUNDLE_MANIFEST])
    if set(manifest) != _COREML_VLM_BUNDLE_KEYS:
        missing = sorted(_COREML_VLM_BUNDLE_KEYS - set(manifest))
        extra = sorted(set(manifest) - _COREML_VLM_BUNDLE_KEYS)
        raise ValueError(
            "Core ML VLM bundle manifest keys changed: "
            f"missing={missing}, extra={extra}."
        )
    if (
        manifest["bundle_format"] != COREML_VLM_BUNDLE_FORMAT
        or isinstance(manifest["bundle_schema_version"], bool)
        or manifest["bundle_schema_version"]
        != COREML_VLM_BUNDLE_SCHEMA_VERSION
        or manifest["component_contract"] != SMOLVLM2_500M_COMPONENT_CONTRACT
        or manifest["model_path"] != COREML_VLM_BUNDLE_MODEL_ROOT
        or manifest["processor_path"] != COREML_VLM_BUNDLE_PROCESSOR_ROOT
        or manifest["source_weights_included"] is not False
    ):
        raise ValueError("Core ML VLM bundle identity contract changed.")

    raw_profile = manifest["profile"]
    if not isinstance(raw_profile, dict) or "context_length" not in raw_profile:
        raise ValueError("Core ML VLM bundle profile is malformed.")
    profile = _runtime_profile(raw_profile["context_length"])
    expected_metadata = smolvlm2_500m_coreml_metadata(profile)
    expected_manifest = _manifest_without_payload(profile, expected_metadata)
    for key, expected in expected_manifest.items():
        if manifest[key] != expected:
            raise ValueError(
                f"Core ML VLM bundle manifest field {key!r} changed."
            )

    raw_payload = manifest["payload_files"]
    if not isinstance(raw_payload, dict) or not raw_payload:
        raise ValueError("Core ML VLM bundle payload manifest is empty.")
    declared: dict[str, tuple[int, str]] = {}
    for raw_name, value in raw_payload.items():
        name = _validate_payload_name(raw_name)
        if name in declared:
            raise ValueError(f"Core ML VLM bundle repeats payload {name!r}.")
        declared[name] = _validate_payload_record(name, value)
    actual_names = set(by_name) - {COREML_VLM_BUNDLE_MANIFEST}
    if set(declared) != actual_names:
        missing = sorted(set(declared) - actual_names)
        extra = sorted(actual_names - set(declared))
        raise ValueError(
            "Core ML VLM bundle payload inventory changed: "
            f"missing={missing}, extra={extra}."
        )
    expected_processor_names = {
        (
            PurePosixPath(COREML_VLM_BUNDLE_PROCESSOR_ROOT) / name
        ).as_posix()
        for name in SMOLVLM2_500M_REQUIRED_ASSETS
    }
    actual_processor_names = {
        name
        for name in declared
        if name.startswith(COREML_VLM_BUNDLE_PROCESSOR_ROOT + "/")
    }
    if actual_processor_names != expected_processor_names:
        raise ValueError(
            "Core ML VLM bundle must contain exactly the 11 pinned processor "
            "and tokenizer assets."
        )
    license_record = declared.get(COREML_VLM_BUNDLE_APACHE_LICENSE)
    if license_record != (
        _APACHE_2_CANONICAL_SIZE,
        _APACHE_2_CANONICAL_SHA256,
    ):
        raise ValueError(
            "Core ML VLM bundle is missing the canonical Apache-2.0 license."
        )
    notice = _bundle_notice_bytes()
    notice_record = declared.get(COREML_VLM_BUNDLE_NOTICE)
    if notice_record != (
        len(notice),
        hashlib.sha256(notice).hexdigest(),
    ):
        raise ValueError(
            "Core ML VLM bundle is missing its exact third-party notice."
        )
    model_prefix = COREML_VLM_BUNDLE_MODEL_ROOT + "/"
    allowed_names = expected_processor_names | {
        COREML_VLM_BUNDLE_APACHE_LICENSE,
        COREML_VLM_BUNDLE_NOTICE,
    }
    unknown_payload = sorted(
        name
        for name in declared
        if name not in allowed_names and not name.startswith(model_prefix)
    )
    if unknown_payload:
        raise ValueError(
            "Core ML VLM bundle contains payload outside its approved roots: "
            f"{unknown_payload}."
        )
    if not any(name.startswith(model_prefix) for name in declared):
        raise ValueError("Core ML VLM bundle contains no model package payload.")
    if any(
        PurePosixPath(name).name == SMOLVLM2_500M_WEIGHTS_FILENAME
        for name in declared
    ):
        raise ValueError(
            "Core ML VLM bundle illegally contains the source safetensors file."
        )
    if any(
        _is_source_weight_record(value)
        for value in raw_payload.values()
        if isinstance(value, Mapping)
    ):
        raise ValueError(
            "Core ML VLM bundle illegally contains the exact source weight "
            "payload under a renamed path."
        )
    for name, (expected_size, expected_hash) in declared.items():
        path = by_name[name]
        actual_size = int(path.stat().st_size)
        if actual_size != expected_size:
            raise ValueError(
                f"Core ML VLM bundle payload {name!r} changed byte length."
            )
        actual_hash = _file_sha256(path)
        if not hmac.compare_digest(actual_hash, expected_hash):
            raise ValueError(
                f"Core ML VLM bundle payload {name!r} failed SHA-256 validation."
            )

    model_path = _ensure_contained(
        resolved_root / COREML_VLM_BUNDLE_MODEL_ROOT,
        resolved_root,
        label="Core ML VLM model path",
    )
    processor_path = _ensure_contained(
        resolved_root / COREML_VLM_BUNDLE_PROCESSOR_ROOT,
        resolved_root,
        label="Core ML VLM processor path",
    )
    validate_smolvlm2_500m_processor_assets(
        processor_path,
        revision=SMOLVLM2_500M_REVISION,
        transformers_version=COREML_VLM_TRANSFORMERS_VERSION,
    )
    ct = _package_ct(coremltools_module)
    package_profile, package_metadata, _ = _load_package_contract(
        model_path,
        coremltools_module=ct,
    )
    if package_profile != profile or package_metadata != expected_metadata:
        raise ValueError(
            "Core ML VLM bundle manifest and package metadata disagree."
        )
    return CoreMLVLMBundleInfo(
        path=resolved_root,
        model_path=model_path,
        processor_path=processor_path,
        profile=profile,
        metadata=package_metadata,
        manifest=manifest,
    )


def apply_coreml_vlm_repetition_penalty(
    logits: Any,
    token_ids: Any,
    *,
    penalty: float = SMOLVLM2_500M_REPETITION_PENALTY,
) -> np.ndarray:
    """Apply deterministic greedy-generation repetition penalty on the host."""

    scores = np.asarray(logits)
    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError(
            f"Core ML VLM logits must have shape [1, V], got {scores.shape}."
        )
    if not np.issubdtype(scores.dtype, np.floating):
        raise ValueError("Core ML VLM logits must be floating point.")
    if not bool(np.isfinite(scores).all()):
        raise ValueError("Core ML VLM logits contain NaN or infinity.")
    factor = float(penalty)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Core ML VLM repetition penalty must be positive.")
    ids = np.asarray(token_ids)
    if ids.size and not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Core ML VLM repetition history must be integral.")
    flat_ids = ids.reshape(-1).astype(np.int64, copy=False)
    if flat_ids.size and (
        int(flat_ids.min()) < 0 or int(flat_ids.max()) >= scores.shape[1]
    ):
        raise ValueError("Core ML VLM repetition history contains an invalid token.")
    adjusted = np.array(scores, copy=True, order="C")
    if flat_ids.size:
        unique = np.unique(flat_ids)
        selected = adjusted[0, unique]
        # PyTorch applies a Python scalar with FP32 arithmetic to FP16 values
        # and casts the scattered result back to the logits dtype.
        working = (
            selected.astype(np.float32)
            if selected.dtype == np.float16
            else selected
        )
        penalized = np.where(
            working < 0,
            working * factor,
            working / factor,
        )
        adjusted[0, unique] = penalized.astype(adjusted.dtype, copy=False)
    return np.ascontiguousarray(adjusted)


def _compute_unit(ct: Any, value: str) -> Any:
    key = str(value).strip().lower()
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    if key not in mapping:
        raise ValueError(
            f"Invalid Core ML compute_units {value!r}; expected one of "
            f"{sorted(mapping)}."
        )
    return mapping[key]


def _load_smolvlm2_processor(path: Path) -> Any:
    require_coreml_vlm_transformers_toolchain()
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "SmolVLM2 Core ML runtime requires Transformers "
            f"{COREML_VLM_TRANSFORMERS_VERSION}."
        ) from exc
    return AutoProcessor.from_pretrained(
        str(path),
        trust_remote_code=False,
        local_files_only=True,
    )


def _only_fp16_output(
    result: Any,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(result, Mapping) or set(result) != {name}:
        actual = sorted(result) if isinstance(result, Mapping) else []
        raise RuntimeError(
            f"Core ML VLM runtime output names changed: expected {[name]}, "
            f"got {actual}."
        )
    value = np.asarray(result[name])
    if value.dtype not in (np.float16, np.float32):
        raise RuntimeError(
            f"Core ML VLM output {name!r} must materialize as float16 or "
            f"float32, got {value.dtype}."
        )
    if tuple(value.shape) != shape:
        raise RuntimeError(
            f"Core ML VLM output {name!r} shape changed: expected {shape}, "
            f"got {tuple(value.shape)}."
        )
    if not bool(np.isfinite(value).all()):
        raise RuntimeError(
            f"Core ML VLM output {name!r} contains NaN or infinity."
        )
    normalized = np.ascontiguousarray(value, dtype=np.float16)
    if not bool(np.isfinite(normalized).all()):
        raise RuntimeError(
            f"Core ML VLM output {name!r} exceeds the declared FP16 range."
        )
    return normalized


class _DecodeRequest:
    """Exactly one Core ML state paired with exactly one append-only cursor."""

    def __init__(self, model: Any, profile: CoreMLVLMProfile) -> None:
        self._model = model
        self._state = model.make_state()
        self._cursor: CoreMLVLMDecodeCursor | None = CoreMLVLMDecodeCursor(profile)
        self._profile = profile

    @property
    def active(self) -> bool:
        return self._state is not None and self._cursor is not None

    def discard(self) -> None:
        self._state = None
        self._cursor = None
        self._model = None

    def predict(self, token_embeddings: np.ndarray) -> np.ndarray:
        if not self.active:
            raise RuntimeError("Core ML VLM decode state has been discarded.")
        if (
            token_embeddings.dtype != np.float16
            or token_embeddings.ndim != 3
            or token_embeddings.shape[0] != 1
            or token_embeddings.shape[2] != self._profile.hidden_size
            or token_embeddings.shape[1] <= 0
            or token_embeddings.shape[1] > self._profile.context_length
        ):
            raise ValueError(
                "Core ML VLM decoder embeddings must have FP16 shape "
                f"[1, Q, {self._profile.hidden_size}] with bounded Q."
            )
        if not bool(np.isfinite(token_embeddings).all()):
            raise ValueError(
                "Core ML VLM decoder embeddings contain NaN or infinity."
            )
        cursor = self._cursor
        assert cursor is not None
        query_length = int(token_embeddings.shape[1])
        causal_mask, position_ids = cursor.controls(query_length=query_length)
        inputs = {
            COREML_VLM_TOKEN_EMBEDDINGS_INPUT: token_embeddings,
            COREML_VLM_CAUSAL_MASK_INPUT: causal_mask,
            COREML_VLM_POSITION_IDS_INPUT: position_ids,
        }
        try:
            result = self._model.predict(inputs, state=self._state)
            logits = _only_fp16_output(
                result,
                name=COREML_VLM_LAST_LOGITS_OUTPUT,
                shape=(1, self._profile.vocab_size),
            )
            cursor.commit(
                causal_mask=causal_mask,
                position_ids=position_ids,
            )
            return logits
        except Exception:
            self.discard()
            raise


class CoreMLVLMRuntime:
    """Host-orchestrated runtime for one strict SmolVLM2 Core ML bundle."""

    def __init__(
        self,
        bundle_path: str | os.PathLike[str],
        *,
        compute_units: str = "validated",
        coremltools_module: Any | None = None,
    ) -> None:
        if sys.platform != "darwin":
            raise RuntimeError(
                "Core ML VLM inference requires macOS 15 or later. Current "
                f"platform: {sys.platform}."
            )
        ct = _package_ct(coremltools_module)
        info = validate_coreml_vlm_bundle(
            bundle_path,
            coremltools_module=ct,
        )
        resolved_compute_units = resolve_coreml_runtime_compute_units(
            compute_units,
            info.metadata,
        )
        processor = _load_smolvlm2_processor(info.processor_path)
        unit = _compute_unit(ct, resolved_compute_units)
        models: dict[str, Any] = {}
        try:
            for function_name in COREML_VLM_FUNCTION_NAMES:
                model = ct.models.MLModel(
                    str(info.model_path),
                    function_name=function_name,
                    compute_units=unit,
                )
                runtime_metadata = {
                    str(key): str(value)
                    for key, value in dict(
                        getattr(model, "user_defined_metadata", {}) or {}
                    ).items()
                }
                validate_coreml_vlm_metadata(runtime_metadata)
                expected_runtime_metadata = stringify_coreml_vlm_metadata(
                    info.metadata
                )
                for key, expected in expected_runtime_metadata.items():
                    if runtime_metadata.get(key) != expected:
                        raise ValueError(
                            "Core ML VLM runtime metadata differs from the "
                            f"package contract at {key!r}."
                        )
                models[function_name] = model
        except Exception:
            models.clear()
            raise
        self.bundle_path = str(info.path)
        self.model_path = str(info.model_path)
        self.metadata = info.metadata
        self.profile = info.profile
        self.processor = processor
        self._models: dict[str, Any] | None = models
        self._request_lock = threading.Lock()
        self._active_decode: _DecodeRequest | None = None

    @property
    def closed(self) -> bool:
        return self._models is None

    def close(self) -> None:
        with self._request_lock:
            if self._active_decode is not None:
                self._active_decode.discard()
                self._active_decode = None
            if self._models is not None:
                self._models.clear()
                self._models = None
            self.processor = None

    def __enter__(self) -> "CoreMLVLMRuntime":
        if self.closed:
            raise RuntimeError("Core ML VLM runtime is closed.")
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _model(self, function_name: str) -> Any:
        if self._models is None:
            raise RuntimeError("Core ML VLM runtime is closed.")
        return self._models[function_name]

    def _embed(self, input_ids: np.ndarray) -> np.ndarray:
        if (
            input_ids.dtype != np.int32
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or input_ids.shape[1] <= 0
            or input_ids.shape[1] > self.profile.context_length
        ):
            raise ValueError(
                "Core ML VLM input_ids must have INT32 shape [1, Q] with "
                "bounded Q."
            )
        if np.any(input_ids < 0) or np.any(input_ids >= self.profile.vocab_size):
            raise ValueError("Core ML VLM input_ids contain an invalid token.")
        result = self._model(COREML_VLM_EMBED_TOKENS_FUNCTION).predict(
            {COREML_VLM_INPUT_IDS_INPUT: np.ascontiguousarray(input_ids)}
        )
        return _only_fp16_output(
            result,
            name=COREML_VLM_TOKEN_EMBEDDINGS_OUTPUT,
            shape=(1, int(input_ids.shape[1]), self.profile.hidden_size),
        )

    def _encode_image(self, pixel_values: np.ndarray) -> np.ndarray:
        expected_shape = (
            1,
            self.profile.image_crops,
            self.profile.image_channels,
            self.profile.image_height,
            self.profile.image_width,
        )
        if (
            pixel_values.dtype != np.float16
            or tuple(pixel_values.shape) != expected_shape
        ):
            raise ValueError(
                "Core ML VLM pixel_values must have FP16 shape "
                f"{expected_shape}, got {pixel_values.dtype}/"
                f"{tuple(pixel_values.shape)}."
            )
        if not bool(np.isfinite(pixel_values).all()):
            raise ValueError(
                "Core ML VLM pixel_values contain NaN or infinity."
            )
        result = self._model(COREML_VLM_ENCODE_IMAGE_FUNCTION).predict(
            {
                COREML_VLM_PIXEL_VALUES_INPUT: np.ascontiguousarray(
                    pixel_values
                )
            }
        )
        return _only_fp16_output(
            result,
            name=COREML_VLM_IMAGE_EMBEDDINGS_OUTPUT,
            shape=(
                1,
                self.profile.image_token_count,
                self.profile.hidden_size,
            ),
        )

    def _generate_batch(
        self,
        batch: Mapping[str, Any],
        *,
        max_new_tokens: int,
    ) -> list[int]:
        prepared = prepare_smolvlm2_500m_coreml_processor_batch(
            self.profile,
            batch,
            max_new_tokens=max_new_tokens,
        )
        input_ids = prepared[COREML_VLM_INPUT_IDS_INPUT]
        pixel_values = prepared[COREML_VLM_PIXEL_VALUES_INPUT]
        image_embeddings = self._encode_image(pixel_values)
        token_embeddings = self._embed(input_ids)
        merged = merge_coreml_vlm_image_embeddings(
            self.profile,
            input_ids=input_ids,
            token_embeddings=token_embeddings,
            image_embeddings=image_embeddings,
        )

        if self._active_decode is not None:
            raise RuntimeError(
                "Core ML VLM runtime already has an active decode state."
            )
        request = _DecodeRequest(
            self._model(COREML_VLM_DECODE_FUNCTION),
            self.profile,
        )
        self._active_decode = request
        generated: list[int] = []
        history = input_ids.reshape(-1).astype(np.int64).tolist()
        try:
            logits = request.predict(merged)
            for index in range(max_new_tokens):
                adjusted = apply_coreml_vlm_repetition_penalty(
                    logits,
                    np.asarray(history, dtype=np.int64),
                    penalty=SMOLVLM2_500M_REPETITION_PENALTY,
                )
                next_token = int(np.argmax(adjusted[0]))
                generated.append(next_token)
                history.append(next_token)
                if next_token == SMOLVLM2_500M_EOS_TOKEN_ID:
                    break
                if index + 1 == max_new_tokens:
                    break
                next_ids = np.asarray([[next_token]], dtype=np.int32)
                logits = request.predict(self._embed(next_ids))
            return generated
        except Exception:
            request.discard()
            raise
        finally:
            request.discard()
            self._active_decode = None

    def chat(
        self,
        image: ImageInput,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        color_format: str = "auto",
    ) -> str:
        """Generate text for one image and prompt with a fresh KV state."""

        if self.closed:
            raise RuntimeError("Core ML VLM runtime is closed.")
        budget = (
            self.profile.max_new_tokens
            if max_new_tokens is None
            else max_new_tokens
        )
        if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
            raise ValueError("max_new_tokens must be an integer.")
        budget = int(budget)
        if budget <= 0:
            raise ValueError("max_new_tokens must be positive.")
        loaded = ImageLoader.load(image, color_format=color_format)
        canonical = preprocess_smolvlm2_500m_coreml_image(loaded)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": canonical},
                    {"type": "text", "text": str(prompt)},
                ],
            }
        ]
        with self._request_lock:
            if self.closed:
                raise RuntimeError("Core ML VLM runtime is closed.")
            batch = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            )
            token_ids = self._generate_batch(
                batch,
                max_new_tokens=budget,
            )
            decoded = self.processor.batch_decode(
                np.asarray([token_ids], dtype=np.int64),
                skip_special_tokens=True,
            )
        if (
            not isinstance(decoded, (list, tuple))
            or len(decoded) != 1
            or not isinstance(decoded[0], str)
        ):
            raise RuntimeError(
                "SmolVLM2 processor returned an invalid decoded response."
            )
        return decoded[0]

    def detect(
        self,
        image: ImageInput,
        prompt: str,
        *,
        name_to_id: Mapping[str, int],
        max_new_tokens: int | None = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        max_det: int = 300,
        classes: list[int] | None = None,
        color_format: str = "auto",
    ) -> dict[str, Any]:
        """Generate and parse the SmolVLM2 normalized-``bbox`` JSON contract."""

        if not isinstance(name_to_id, Mapping) or not name_to_id:
            raise ValueError("name_to_id must be a non-empty label mapping.")
        normalized: dict[str, int] = {}
        used_ids: set[int] = set()
        for raw_label, raw_id in name_to_id.items():
            label = str(raw_label).strip().lower()
            if not label:
                raise ValueError("Core ML VLM detection labels must not be empty.")
            if isinstance(raw_id, bool) or not isinstance(
                raw_id,
                (int, np.integer),
            ):
                raise ValueError("Core ML VLM detection class IDs must be integers.")
            class_id = int(raw_id)
            if class_id < 0:
                raise ValueError(
                    "Core ML VLM detection class IDs must be non-negative."
                )
            if label in normalized or class_id in used_ids:
                raise ValueError(
                    "Core ML VLM detection labels and IDs must be unique."
                )
            normalized[label] = class_id
            used_ids.add(class_id)

        loaded = ImageLoader.load(image, color_format=color_format)
        text = self.chat(
            loaded,
            prompt,
            max_new_tokens=max_new_tokens,
            color_format="rgb",
        )
        return build_detection_dict(
            extract_detections(text),
            normalized,
            loaded.size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            classes=classes,
            default_score=1.0,
            bbox_key="bbox",
            coord_divisor=1.0,
            box_format="xyxy",
        )


__all__ = [
    "COREML_VLM_BUNDLE_FORMAT",
    "COREML_VLM_BUNDLE_APACHE_LICENSE",
    "COREML_VLM_BUNDLE_MANIFEST",
    "COREML_VLM_BUNDLE_MODEL_ROOT",
    "COREML_VLM_BUNDLE_NOTICE",
    "COREML_VLM_BUNDLE_PROCESSOR_ROOT",
    "COREML_VLM_BUNDLE_SCHEMA_VERSION",
    "COREML_VLM_BUNDLE_SUFFIX",
    "COREML_VLM_RUNTIME_CONTEXTS",
    "CoreMLVLMBundleInfo",
    "CoreMLVLMRuntime",
    "apply_coreml_vlm_repetition_penalty",
    "build_coreml_vlm_bundle",
    "validate_coreml_vlm_bundle",
]
