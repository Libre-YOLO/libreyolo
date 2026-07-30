"""Strict portable bundle and host runtime for Florence-2-base Core ML.

The runtime is intentionally separate from LibreYOLO's one-shot Core ML
backend.  Florence needs a pinned processor, a stateless encoder, a stateful
three-beam decoder, request-local cache initialization, and host-side beam
scoring.

Provenance
----------
The model and processor are the MIT-licensed
``florence-community/Florence-2-base`` snapshot at
``00921df66db728a9ceb750f5eca43e5c203a2051``.  Beam semantics are implemented
in :mod:`libreyolo.backends.coreml_florence_beam` from the Apache-2.0
Transformers 5.12.1 reference pinned there.  This host/bundle implementation is
LibreYOLO code under MIT.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import importlib.metadata
import json
import os
import shutil
import stat
import sys
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Self

import numpy as np

from ..export.coreml_florence import (
    FLORENCE2_BASE_REPO,
    FLORENCE2_BASE_REQUIRED_ASSETS,
    FLORENCE2_BASE_REVISION,
    FLORENCE2_BASE_WEIGHTS_FILENAME,
    FLORENCE2_BASE_WEIGHTS_SHA256,
    FLORENCE2_BASE_WEIGHTS_SIZE,
    FLORENCE2_DECODER_START_TOKEN_ID,
    FLORENCE2_TASK,
    FLORENCE_BEAM_PARENT_INDICES_INPUT,
    FLORENCE_CAUSAL_MASK_INPUT,
    FLORENCE_COREML_COMPONENT_CONTRACT,
    FLORENCE_COREML_TRANSFORMERS_COMMIT,
    FLORENCE_COREML_TRANSFORMERS_VERSION,
    FLORENCE_CROSS_ATTENTION_MASK_INPUT,
    FLORENCE_CROSS_KEY_CACHE_STATE,
    FLORENCE_CROSS_KEY_OUTPUT,
    FLORENCE_CROSS_VALUE_CACHE_STATE,
    FLORENCE_CROSS_VALUE_OUTPUT,
    FLORENCE_DECODE_FUNCTION,
    FLORENCE_DECODER_INPUT_IDS_INPUT,
    FLORENCE_ENCODE_FUNCTION,
    FLORENCE_ENCODER_ATTENTION_MASK_INPUT,
    FLORENCE_ENCODER_INPUT_IDS_INPUT,
    FLORENCE_FUNCTION_NAMES,
    FLORENCE_LAST_LOGITS_OUTPUT,
    FLORENCE_PIXEL_VALUES_INPUT,
    FLORENCE_POSITION_IDS_INPUT,
    FlorenceCoreMLProfile,
    FlorenceDecodeCursor,
    florence2_base_coreml_metadata,
    florence2_base_coreml_profile,
    prepare_florence2_base_processor_batch,
    require_florence_coreml_toolchain,
    require_florence_transformers_toolchain,
    stringify_florence_coreml_metadata,
    validate_florence2_base_processor_assets,
    validate_florence_coreml_metadata,
    validate_florence_multifunction_spec,
)
from ..export.coreml_profiles import resolve_coreml_runtime_compute_units
from ..utils.image_loader import ImageInput, ImageLoader
from .coreml_florence_beam import Florence2BeamSearch

COREML_FLORENCE_BUNDLE_FORMAT = "libreyolo_coreml_florence_bundle"
COREML_FLORENCE_BUNDLE_SCHEMA_VERSION = 1
COREML_FLORENCE_BUNDLE_SUFFIX = ".coremlvlm"
COREML_FLORENCE_BUNDLE_MANIFEST = "manifest.json"
COREML_FLORENCE_BUNDLE_MODEL_ROOT = "Model.mlpackage"
COREML_FLORENCE_BUNDLE_PROCESSOR_ROOT = "Processor"
COREML_FLORENCE_BUNDLE_MIT_LICENSE = "LICENSES/MIT-Florence.txt"
COREML_FLORENCE_BUNDLE_APACHE_LICENSE = "LICENSES/Apache-2.0.txt"
COREML_FLORENCE_BUNDLE_NOTICE = "NOTICE.txt"

_APACHE_2_CANONICAL_SIZE = 11_357
_APACHE_2_CANONICAL_SHA256 = (
    "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
)

_COREML_FLORENCE_BUNDLE_KEYS = frozenset(
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
    candidates = [Path(__file__).resolve().parents[2] / "licenses" / "Apache-2.0.txt"]
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
        if len(canonical) == _APACHE_2_CANONICAL_SIZE and hmac.compare_digest(
            hashlib.sha256(canonical).hexdigest(),
            _APACHE_2_CANONICAL_SHA256,
        ):
            return canonical
    raise RuntimeError(
        "LibreYOLO's canonical Apache-2.0 license asset is missing or modified."
    )


def _florence_mit_license_bytes() -> bytes:
    return (
        "MIT License\n"
        "\n"
        "Copyright (c) Microsoft Corporation.\n"
        "\n"
        "Permission is hereby granted, free of charge, to any person obtaining "
        "a copy\n"
        'of this software and associated documentation files (the "Software"), '
        "to deal\n"
        "in the Software without restriction, including without limitation "
        "the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or "
        "sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n"
        "\n"
        "The above copyright notice and this permission notice shall be "
        "included in all\n"
        "copies or substantial portions of the Software.\n"
        "\n"
        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, '
        "EXPRESS OR\n"
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF "
        "MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT "
        "SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR "
        "OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, "
        "ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER "
        "DEALINGS IN THE\n"
        "SOFTWARE.\n"
    ).encode("ascii")


def _bundle_notice_bytes() -> bytes:
    return (
        "LibreYOLO Florence-2-base Core ML bundle\n"
        "\n"
        "Converted model and processor:\n"
        f"  {FLORENCE2_BASE_REPO}\n"
        f"  revision {FLORENCE2_BASE_REVISION}\n"
        "  license MIT\n"
        "  original copyright Microsoft Corporation\n"
        "\n"
        "Conversion equations and beam semantics reference:\n"
        "  https://github.com/huggingface/transformers\n"
        f"  commit {FLORENCE_COREML_TRANSFORMERS_COMMIT}\n"
        f"  version {FLORENCE_COREML_TRANSFORMERS_VERSION}\n"
        "  license Apache-2.0\n"
        "\n"
        "The bundle contains converted Core ML weights and exactly ten pinned\n"
        "processor/tokenizer assets. It does not contain model.safetensors.\n"
        "See LICENSES/ for the full MIT and Apache-2.0 terms.\n"
    ).encode("ascii")


def _ensure_contained(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its declared root: {path}.") from exc
    return resolved


def _safe_tree_files(root: Path, *, label: str) -> list[tuple[Path, Path]]:
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


def _validate_payload_name(name: Any) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"Invalid Florence bundle payload path {name!r}.")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe Florence bundle payload path {name!r}.")
    if path.as_posix() != name:
        raise ValueError(f"Non-canonical Florence bundle path {name!r}.")
    return name


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Florence bundle manifest repeats key {key!r}.")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Florence bundle manifest is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise TypeError("Florence bundle manifest must be a JSON object.")
    return value


def _payload_record(path: Path) -> dict[str, Any]:
    return {
        "size_bytes": int(path.stat().st_size),
        "sha256": _file_sha256(path),
    }


def _validate_payload_record(name: str, value: Any) -> tuple[int, str]:
    if not isinstance(value, dict) or set(value) != {"size_bytes", "sha256"}:
        raise ValueError(f"Florence payload record for {name!r} is malformed.")
    size = value["size_bytes"]
    digest = value["sha256"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"Florence payload size for {name!r} is invalid.")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"Florence payload SHA-256 for {name!r} is invalid.")
    return size, digest


def _validate_bundle_destination(path: Path) -> None:
    if path.suffix.lower() != COREML_FLORENCE_BUNDLE_SUFFIX:
        raise ValueError(
            f"Florence bundle output must end in {COREML_FLORENCE_BUNDLE_SUFFIX!r}."
        )
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite Florence bundle: {path}.")


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        source.rename(destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    function = None
    arguments: tuple[Any, ...] = ()
    if sys.platform == "darwin":
        function = getattr(libc, "renameatx_np", None)
        if function is not None:
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            function.restype = ctypes.c_int
            arguments = (
                -2,
                os.fsencode(source),
                -2,
                os.fsencode(destination),
                0x00000004,
            )
    elif sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        if function is not None:
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            function.restype = ctypes.c_int
            arguments = (
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                0x00000001,
            )
    if function is not None:
        ctypes.set_errno(0)
        if function(*arguments) == 0:
            return
        error = ctypes.get_errno()
        if error in {errno.EEXIST, getattr(errno, "ENOTEMPTY", errno.EEXIST)}:
            raise FileExistsError(
                error,
                "Florence bundle destination already exists",
                str(destination),
            )
        unsupported = {
            errno.EINVAL,
            errno.ENOSYS,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if error not in unsupported:
            raise OSError(
                error,
                "Failed to publish Florence bundle",
                str(destination),
            )
    raise RuntimeError(
        "The destination filesystem lacks atomic no-replace directory "
        "publication. Refusing an unsafe Florence bundle rename."
    )


def _copy_regular_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as output:
        shutil.copyfileobj(source_handle, output, length=1024 * 1024)
    shutil.copystat(source, destination, follow_symlinks=False)


def _metadata_from_spec(spec: Any) -> dict[str, str]:
    description = getattr(spec, "description", None)
    metadata = getattr(description, "metadata", None)
    values = getattr(metadata, "userDefined", None)
    if values is None:
        return {}
    return {str(key): str(value) for key, value in dict(values).items()}


def _validate_apple_package_manifest(
    package_root: Path,
    package_files: list[tuple[Path, Path]],
) -> None:
    by_name = {relative.as_posix(): path for relative, path in package_files}
    manifest_path = by_name.get("Manifest.json")
    if manifest_path is None:
        raise ValueError("Florence package is missing Apple's Manifest.json.")
    manifest = _load_json_object(manifest_path)
    if manifest.get("fileFormatVersion") != "1.0.0":
        raise ValueError("Florence package has an unknown file format version.")
    entries = manifest.get("itemInfoEntries")
    root_identifier = manifest.get("rootModelIdentifier")
    if (
        not isinstance(entries, dict)
        or not entries
        or not isinstance(root_identifier, str)
        or root_identifier not in entries
    ):
        raise ValueError("Florence package manifest entries are malformed.")
    covered = {"Manifest.json"}
    for identifier, raw_entry in entries.items():
        if not isinstance(identifier, str) or not isinstance(raw_entry, dict):
            raise TypeError("Florence package manifest entry is malformed.")
        relative = _validate_payload_name(raw_entry.get("path"))
        data_name = (PurePosixPath("Data") / relative).as_posix()
        data_path = package_root / Path(data_name)
        _ensure_contained(
            data_path,
            package_root,
            label="Florence package manifest path",
        )
        if data_path.is_file():
            covered.add(data_name)
        elif data_path.is_dir():
            prefix = data_name.rstrip("/") + "/"
            nested = {name for name in by_name if name.startswith(prefix)}
            if not nested:
                raise ValueError(
                    "Florence package manifest points to an empty directory."
                )
            covered.update(nested)
        else:
            raise ValueError(
                f"Florence package manifest payload is missing: {relative!r}."
            )
    if set(by_name) != covered:
        extras = sorted(set(by_name) - covered)
        raise ValueError(f"Florence package has unmanifested payload files: {extras}.")


def _package_ct(coremltools_module: Any | None = None) -> Any:
    if coremltools_module is None:
        try:
            import coremltools as ct
        except ImportError as exc:
            raise ImportError(
                "Florence Core ML bundles require coremltools 9.x."
            ) from exc
    else:
        ct = coremltools_module
    require_florence_coreml_toolchain(ct)
    return ct


def _load_package_contract(
    package_path: Path,
    *,
    coremltools_module: Any,
) -> tuple[FlorenceCoreMLProfile, dict[str, Any], dict[str, str]]:
    if package_path.is_symlink():
        raise ValueError("Florence package must not be a symbolic link.")
    if not package_path.is_dir() or package_path.suffix.lower() != ".mlpackage":
        raise ValueError(
            f"Florence model must be an .mlpackage directory: {package_path}."
        )
    files = _safe_tree_files(package_path, label="Florence Core ML package")
    _validate_apple_package_manifest(package_path, files)
    spec = coremltools_module.utils.load_spec(str(package_path))
    raw_metadata = _metadata_from_spec(spec)
    metadata = validate_florence_coreml_metadata(raw_metadata)
    profile = florence2_base_coreml_profile()
    if metadata["florence_profile"] != profile.as_dict():
        raise ValueError("Florence package metadata profile changed.")
    validate_florence_multifunction_spec(spec, profile=profile)
    return profile, metadata, raw_metadata


def _is_source_weight_record(record: Mapping[str, Any]) -> bool:
    return (
        record.get("size_bytes") == FLORENCE2_BASE_WEIGHTS_SIZE
        and record.get("sha256") == FLORENCE2_BASE_WEIGHTS_SHA256
    )


def _manifest_without_payload(
    profile: FlorenceCoreMLProfile,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    mit = _florence_mit_license_bytes()
    apache = _apache_2_license_bytes()
    notice = _bundle_notice_bytes()
    return {
        "bundle_format": COREML_FLORENCE_BUNDLE_FORMAT,
        "bundle_schema_version": COREML_FLORENCE_BUNDLE_SCHEMA_VERSION,
        "component_contract": FLORENCE_COREML_COMPONENT_CONTRACT,
        "model_path": COREML_FLORENCE_BUNDLE_MODEL_ROOT,
        "processor_path": COREML_FLORENCE_BUNDLE_PROCESSOR_ROOT,
        "profile": profile.as_dict(),
        "coreml_contract_sha256": metadata["coreml_florence_contract_sha256"],
        "processor": metadata["processor"],
        "source_weights_included": False,
        "licenses": {
            "MIT": {
                "path": COREML_FLORENCE_BUNDLE_MIT_LICENSE,
                "spdx": "MIT",
                "size_bytes": len(mit),
                "sha256": hashlib.sha256(mit).hexdigest(),
            },
            "Apache-2.0": {
                "path": COREML_FLORENCE_BUNDLE_APACHE_LICENSE,
                "spdx": "Apache-2.0",
                "size_bytes": len(apache),
                "sha256": hashlib.sha256(apache).hexdigest(),
            },
        },
        "notice": {
            "path": COREML_FLORENCE_BUNDLE_NOTICE,
            "size_bytes": len(notice),
            "sha256": hashlib.sha256(notice).hexdigest(),
        },
        "provenance": {
            "model": {
                "repo": FLORENCE2_BASE_REPO,
                "revision": FLORENCE2_BASE_REVISION,
                "license": "MIT",
            },
            "transformers": {
                "repo": "https://github.com/huggingface/transformers",
                "commit": FLORENCE_COREML_TRANSFORMERS_COMMIT,
                "version": FLORENCE_COREML_TRANSFORMERS_VERSION,
                "license": "Apache-2.0",
            },
            "bundle_runtime": {
                "project": "LibreYOLO",
                "license": "MIT",
            },
        },
    }


def _verify_staged_payload(
    root: Path,
    payload: Mapping[str, Mapping[str, Any]],
) -> None:
    files = _safe_tree_files(root, label="staged Florence bundle")
    by_name = {
        relative.as_posix(): path
        for relative, path in files
        if relative.as_posix() != COREML_FLORENCE_BUNDLE_MANIFEST
    }
    if set(by_name) != set(payload):
        raise RuntimeError("Staged Florence bundle inventory changed.")
    for name, expected in payload.items():
        if _payload_record(by_name[name]) != expected:
            raise RuntimeError(f"Staged Florence bundle payload {name!r} changed.")


def build_coreml_florence_bundle(
    package_path: str | os.PathLike[str],
    *,
    processor_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    move_model: bool = False,
    coremltools_module: Any | None = None,
) -> str:
    """Build one portable, hash-bound, no-overwrite Florence bundle.

    ``move_model=False`` preserves the source package and copies it into the
    bundle.  Export orchestration may pass ``move_model=True`` for a freshly
    staged package.  A failed publication then restores that package to its
    original path, unless that path has been independently reoccupied.
    """

    source_package = Path(package_path)
    source_processor = Path(processor_dir)
    destination = Path(output_path)
    _validate_bundle_destination(destination)
    if source_package.is_symlink() or source_processor.is_symlink():
        raise ValueError("Florence bundle sources must not be symbolic links.")

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
        label="Florence source package",
    )
    processor_files = _safe_tree_files(
        processor_root,
        label="Florence processor snapshot",
    )
    validate_florence2_base_processor_assets(
        processor_root,
        revision=FLORENCE2_BASE_REVISION,
        transformers_version=FLORENCE_COREML_TRANSFORMERS_VERSION,
    )
    processor_by_name = {
        relative.as_posix(): path for relative, path in processor_files
    }
    required_processor = set(FLORENCE2_BASE_REQUIRED_ASSETS)
    if not required_processor.issubset(processor_by_name):
        missing = sorted(required_processor - set(processor_by_name))
        raise FileNotFoundError(f"Florence processor snapshot is missing {missing}.")

    mit_license = _florence_mit_license_bytes()
    apache_license = _apache_2_license_bytes()
    bundle_notice = _bundle_notice_bytes()
    payload: dict[str, dict[str, Any]] = {}
    for relative, path in package_files:
        if path.name == FLORENCE2_BASE_WEIGHTS_FILENAME:
            raise ValueError(
                "Refusing to bundle the source safetensors payload; only "
                "converted Core ML weights belong inside Model.mlpackage."
            )
        name = (
            PurePosixPath(COREML_FLORENCE_BUNDLE_MODEL_ROOT) / relative.as_posix()
        ).as_posix()
        _validate_payload_name(name)
        record = _payload_record(path)
        if _is_source_weight_record(record):
            raise ValueError(
                "Refusing to bundle the exact source safetensors payload "
                "under a renamed package path."
            )
        payload[name] = record
    for name in sorted(required_processor):
        source = processor_by_name[name]
        bundle_name = (
            PurePosixPath(COREML_FLORENCE_BUNDLE_PROCESSOR_ROOT) / name
        ).as_posix()
        _validate_payload_name(bundle_name)
        payload[bundle_name] = _payload_record(source)
    for name, value in (
        (COREML_FLORENCE_BUNDLE_MIT_LICENSE, mit_license),
        (COREML_FLORENCE_BUNDLE_APACHE_LICENSE, apache_license),
        (COREML_FLORENCE_BUNDLE_NOTICE, bundle_notice),
    ):
        payload[name] = {
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }

    manifest = {
        **_manifest_without_payload(profile, metadata),
        "payload_files": payload,
    }
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=".libreyolo-coreml-florence-bundle-",
            dir=str(destination.parent),
        )
    )
    staged_model = temporary_root / COREML_FLORENCE_BUNDLE_MODEL_ROOT
    model_was_moved = False
    remove_temporary_root = True
    try:
        processor_destination = temporary_root / COREML_FLORENCE_BUNDLE_PROCESSOR_ROOT
        for name in sorted(required_processor):
            _copy_regular_file(
                processor_by_name[name],
                processor_destination / Path(name),
            )
        manifest_path = temporary_root / COREML_FLORENCE_BUNDLE_MANIFEST
        manifest_path.write_text(
            _canonical_json(manifest) + "\n",
            encoding="utf-8",
        )
        for name, value in (
            (COREML_FLORENCE_BUNDLE_MIT_LICENSE, mit_license),
            (COREML_FLORENCE_BUNDLE_APACHE_LICENSE, apache_license),
            (COREML_FLORENCE_BUNDLE_NOTICE, bundle_notice),
        ):
            path = temporary_root / Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(value)
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
                    "Florence bundle publication failed and the moved model "
                    "could not be restored because its source path was "
                    f"reoccupied. The original is preserved at {staged_model}."
                ) from exc
            try:
                staged_model.rename(package_root)
                model_was_moved = False
            except Exception as restore_exc:
                remove_temporary_root = False
                raise RuntimeError(
                    "Florence bundle publication failed and the moved model "
                    f"could not be restored. It is preserved at {staged_model}."
                ) from restore_exc
        raise
    finally:
        if remove_temporary_root and temporary_root.exists():
            shutil.rmtree(temporary_root)
    return str(destination)


@dataclass(frozen=True)
class CoreMLFlorenceBundleInfo:
    """Validated paths and exact contract for one Florence Core ML bundle."""

    path: Path
    model_path: Path
    processor_path: Path
    profile: FlorenceCoreMLProfile
    metadata: dict[str, Any]
    manifest: dict[str, Any]


def validate_coreml_florence_bundle(
    bundle_path: str | os.PathLike[str],
    *,
    coremltools_module: Any | None = None,
) -> CoreMLFlorenceBundleInfo:
    """Validate paths, hashes, provenance, processor, and package ABI."""

    root = Path(bundle_path)
    if root.is_symlink():
        raise ValueError("Florence Core ML bundle root must not be a symbolic link.")
    if not root.is_dir() or root.suffix.lower() != COREML_FLORENCE_BUNDLE_SUFFIX:
        raise ValueError(
            "Florence runtime requires a "
            f"{COREML_FLORENCE_BUNDLE_SUFFIX} directory: {root}."
        )
    resolved_root = root.resolve(strict=True)
    all_files = _safe_tree_files(
        resolved_root,
        label="Florence Core ML bundle",
    )
    by_name = {relative.as_posix(): path for relative, path in all_files}
    manifest_path = by_name.get(COREML_FLORENCE_BUNDLE_MANIFEST)
    if manifest_path is None:
        raise FileNotFoundError("Florence Core ML bundle is missing manifest.json.")
    manifest = _load_json_object(manifest_path)
    if set(manifest) != _COREML_FLORENCE_BUNDLE_KEYS:
        missing = sorted(_COREML_FLORENCE_BUNDLE_KEYS - set(manifest))
        extra = sorted(set(manifest) - _COREML_FLORENCE_BUNDLE_KEYS)
        raise ValueError(
            f"Florence bundle manifest keys changed: missing={missing}, extra={extra}."
        )
    if (
        manifest["bundle_format"] != COREML_FLORENCE_BUNDLE_FORMAT
        or isinstance(manifest["bundle_schema_version"], bool)
        or manifest["bundle_schema_version"] != COREML_FLORENCE_BUNDLE_SCHEMA_VERSION
        or manifest["component_contract"] != FLORENCE_COREML_COMPONENT_CONTRACT
        or manifest["model_path"] != COREML_FLORENCE_BUNDLE_MODEL_ROOT
        or manifest["processor_path"] != COREML_FLORENCE_BUNDLE_PROCESSOR_ROOT
        or manifest["source_weights_included"] is not False
    ):
        raise ValueError("Florence bundle identity contract changed.")

    profile = florence2_base_coreml_profile()
    expected_metadata = florence2_base_coreml_metadata(profile)
    expected_manifest = _manifest_without_payload(profile, expected_metadata)
    for key, expected in expected_manifest.items():
        if manifest[key] != expected:
            raise ValueError(f"Florence bundle manifest field {key!r} changed.")

    raw_payload = manifest["payload_files"]
    if not isinstance(raw_payload, dict) or not raw_payload:
        raise ValueError("Florence bundle payload manifest is empty.")
    declared: dict[str, tuple[int, str]] = {}
    for raw_name, value in raw_payload.items():
        name = _validate_payload_name(raw_name)
        if name in declared:
            raise ValueError(f"Florence bundle repeats payload {name!r}.")
        declared[name] = _validate_payload_record(name, value)
    actual_names = set(by_name) - {COREML_FLORENCE_BUNDLE_MANIFEST}
    if set(declared) != actual_names:
        missing = sorted(set(declared) - actual_names)
        extra = sorted(actual_names - set(declared))
        raise ValueError(
            "Florence bundle payload inventory changed: "
            f"missing={missing}, extra={extra}."
        )

    expected_processor_names = {
        (PurePosixPath(COREML_FLORENCE_BUNDLE_PROCESSOR_ROOT) / name).as_posix()
        for name in FLORENCE2_BASE_REQUIRED_ASSETS
    }
    actual_processor_names = {
        name
        for name in declared
        if name.startswith(COREML_FLORENCE_BUNDLE_PROCESSOR_ROOT + "/")
    }
    if actual_processor_names != expected_processor_names:
        raise ValueError(
            "Florence bundle must contain exactly the ten pinned processor "
            "and tokenizer assets."
        )

    expected_fixed_payloads = {
        COREML_FLORENCE_BUNDLE_MIT_LICENSE: _florence_mit_license_bytes(),
        COREML_FLORENCE_BUNDLE_APACHE_LICENSE: _apache_2_license_bytes(),
        COREML_FLORENCE_BUNDLE_NOTICE: _bundle_notice_bytes(),
    }
    for name, value in expected_fixed_payloads.items():
        if declared.get(name) != (
            len(value),
            hashlib.sha256(value).hexdigest(),
        ):
            raise ValueError(f"Florence bundle is missing its exact {name!r} payload.")

    model_prefix = COREML_FLORENCE_BUNDLE_MODEL_ROOT + "/"
    allowed_names = expected_processor_names | set(expected_fixed_payloads)
    unknown_payload = sorted(
        name
        for name in declared
        if name not in allowed_names and not name.startswith(model_prefix)
    )
    if unknown_payload:
        raise ValueError(
            "Florence bundle contains payload outside approved roots: "
            f"{unknown_payload}."
        )
    if not any(name.startswith(model_prefix) for name in declared):
        raise ValueError("Florence bundle contains no model package payload.")
    if any(
        PurePosixPath(name).name == FLORENCE2_BASE_WEIGHTS_FILENAME for name in declared
    ):
        raise ValueError(
            "Florence bundle illegally contains the source safetensors file."
        )
    if any(
        _is_source_weight_record(value)
        for value in raw_payload.values()
        if isinstance(value, Mapping)
    ):
        raise ValueError(
            "Florence bundle illegally contains the exact source weight "
            "payload under a renamed path."
        )
    for name, (expected_size, expected_hash) in declared.items():
        path = by_name[name]
        if int(path.stat().st_size) != expected_size:
            raise ValueError(f"Florence bundle payload {name!r} changed byte length.")
        if not hmac.compare_digest(_file_sha256(path), expected_hash):
            raise ValueError(
                f"Florence bundle payload {name!r} failed SHA-256 validation."
            )

    model_path = _ensure_contained(
        resolved_root / COREML_FLORENCE_BUNDLE_MODEL_ROOT,
        resolved_root,
        label="Florence model path",
    )
    processor_path = _ensure_contained(
        resolved_root / COREML_FLORENCE_BUNDLE_PROCESSOR_ROOT,
        resolved_root,
        label="Florence processor path",
    )
    validate_florence2_base_processor_assets(
        processor_path,
        revision=FLORENCE2_BASE_REVISION,
        transformers_version=FLORENCE_COREML_TRANSFORMERS_VERSION,
    )
    ct = _package_ct(coremltools_module)
    package_profile, package_metadata, _ = _load_package_contract(
        model_path,
        coremltools_module=ct,
    )
    if package_profile != profile or package_metadata != expected_metadata:
        raise ValueError("Florence bundle manifest and package metadata disagree.")
    return CoreMLFlorenceBundleInfo(
        path=resolved_root,
        model_path=model_path,
        processor_path=processor_path,
        profile=profile,
        metadata=package_metadata,
        manifest=manifest,
    )


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
            f"Invalid Florence Core ML compute_units {value!r}; expected one "
            f"of {sorted(mapping)}."
        )
    return mapping[key]


def _load_florence_processor(path: Path) -> Any:
    require_florence_transformers_toolchain()
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Florence Core ML runtime requires Transformers "
            f"{FLORENCE_COREML_TRANSFORMERS_VERSION}."
        ) from exc
    return AutoProcessor.from_pretrained(
        str(path),
        trust_remote_code=False,
        local_files_only=True,
    )


def _fp16_output(
    result: Any,
    *,
    expected_names: set[str],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(result, Mapping) or set(result) != expected_names:
        actual = sorted(result) if isinstance(result, Mapping) else []
        raise RuntimeError(
            "Florence Core ML runtime output names changed: expected "
            f"{sorted(expected_names)}, got {actual}."
        )
    value = np.asarray(result[name])
    if value.dtype not in (np.float16, np.float32):
        raise RuntimeError(
            f"Florence output {name!r} must materialize as float16 or float32, "
            f"got {value.dtype}."
        )
    if tuple(value.shape) != shape:
        raise RuntimeError(
            f"Florence output {name!r} shape changed: expected {shape}, "
            f"got {tuple(value.shape)}."
        )
    if not bool(np.isfinite(value).all()):
        raise RuntimeError(f"Florence output {name!r} contains NaN or infinity.")
    normalized = np.ascontiguousarray(value, dtype=np.float16)
    if not bool(np.isfinite(normalized).all()):
        raise RuntimeError(
            f"Florence output {name!r} exceeds the declared FP16 range."
        )
    return normalized


def _validated_state_value(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float16 or tuple(array.shape) != shape:
        raise RuntimeError(
            f"Florence state {name!r} requires FP16 shape {shape}, got "
            f"{array.dtype}/{tuple(array.shape)}."
        )
    if not bool(np.isfinite(array).all()):
        raise RuntimeError(f"Florence state {name!r} contains NaN or infinity.")
    return np.ascontiguousarray(array)


class _FlorenceDecodeRequest:
    """One fresh MLState, seeded cross cache, and append-only cursor."""

    def __init__(
        self,
        model: Any,
        profile: FlorenceCoreMLProfile,
        *,
        cross_key_values: np.ndarray,
        cross_value_values: np.ndarray,
    ) -> None:
        self._model: Any | None = model
        self._profile = profile
        self._state: Any | None = None
        self._cursor: FlorenceDecodeCursor | None = None
        state = model.make_state()
        write_state = getattr(state, "write_state", None)
        if not callable(write_state):
            raise TypeError("Florence decoder MLState does not expose write_state().")
        single_shape = profile.single_cross_cache_shape
        key = _validated_state_value(
            cross_key_values,
            name=FLORENCE_CROSS_KEY_OUTPUT,
            shape=single_shape,
        )
        value = _validated_state_value(
            cross_value_values,
            name=FLORENCE_CROSS_VALUE_OUTPUT,
            shape=single_shape,
        )
        # A multifunction package with the numerically required FP32 encoder
        # materializes writable decoder state as FP32 on Apple runtime, even
        # though the public state feature remains declared as FP16. Preserve
        # the FP16 cache values exactly while widening them for MLState.
        key = np.ascontiguousarray(
            np.repeat(key, profile.num_beams, axis=1),
            dtype=np.float32,
        )
        value = np.ascontiguousarray(
            np.repeat(value, profile.num_beams, axis=1),
            dtype=np.float32,
        )
        if key.shape != profile.cross_cache_shape:
            raise RuntimeError("Florence cross-key state shape changed.")
        if value.shape != profile.cross_cache_shape:
            raise RuntimeError("Florence cross-value state shape changed.")
        try:
            # Apple documents MLState.write_state(name=..., value=...) as the
            # host path for initializing state buffers before prediction.
            write_state(name=FLORENCE_CROSS_KEY_CACHE_STATE, value=key)
            write_state(name=FLORENCE_CROSS_VALUE_CACHE_STATE, value=value)
        except Exception:
            self._model = None
            raise
        self._state = state
        self._cursor = FlorenceDecodeCursor(profile)

    @property
    def active(self) -> bool:
        return (
            self._model is not None
            and self._state is not None
            and self._cursor is not None
        )

    def discard(self) -> None:
        self._state = None
        self._cursor = None
        self._model = None

    def predict(
        self,
        input_ids: np.ndarray,
        *,
        cross_attention_mask: np.ndarray,
        beam_parent_indices: np.ndarray,
    ) -> np.ndarray:
        if not self.active:
            raise RuntimeError("Florence decode state has been discarded.")
        expected_ids = (self._profile.num_beams, 1)
        if input_ids.dtype != np.int32 or tuple(input_ids.shape) != expected_ids:
            raise ValueError(
                f"Florence decoder input IDs must have INT32 shape {expected_ids}."
            )
        if np.any(input_ids < 0) or np.any(input_ids >= self._profile.vocab_size):
            raise ValueError("Florence decoder input IDs contain an invalid token.")
        expected_cross = (
            self._profile.num_beams,
            1,
            1,
            self._profile.encoder_context_length,
        )
        if (
            cross_attention_mask.dtype != np.float16
            or tuple(cross_attention_mask.shape) != expected_cross
            or not bool(np.isfinite(cross_attention_mask).all())
        ):
            raise ValueError(
                "Florence cross-attention mask must have finite FP16 shape "
                f"{expected_cross}."
            )
        expected_parents = (self._profile.num_beams,)
        if (
            beam_parent_indices.dtype != np.int32
            or tuple(beam_parent_indices.shape) != expected_parents
            or np.any(beam_parent_indices < 0)
            or np.any(beam_parent_indices >= self._profile.num_beams)
        ):
            raise ValueError(
                f"Florence beam parents must have valid INT32 shape {expected_parents}."
            )

        cursor = self._cursor
        model = self._model
        state = self._state
        assert cursor is not None and model is not None and state is not None
        causal_mask, position_ids = cursor.controls()
        inputs = {
            FLORENCE_DECODER_INPUT_IDS_INPUT: np.ascontiguousarray(input_ids),
            FLORENCE_CAUSAL_MASK_INPUT: causal_mask,
            FLORENCE_CROSS_ATTENTION_MASK_INPUT: np.ascontiguousarray(
                cross_attention_mask
            ),
            FLORENCE_POSITION_IDS_INPUT: position_ids,
            FLORENCE_BEAM_PARENT_INDICES_INPUT: np.ascontiguousarray(
                beam_parent_indices
            ),
        }
        try:
            result = model.predict(inputs, state=state)
            logits = _fp16_output(
                result,
                expected_names={FLORENCE_LAST_LOGITS_OUTPUT},
                name=FLORENCE_LAST_LOGITS_OUTPUT,
                shape=(
                    self._profile.num_beams,
                    self._profile.vocab_size,
                ),
            )
            cursor.commit(
                causal_mask=causal_mask,
                position_ids=position_ids,
            )
            return logits
        except Exception:
            self.discard()
            raise


def _validated_names(
    names: Mapping[int, str] | list[str] | tuple[str, ...],
) -> dict[int, str]:
    if isinstance(names, Mapping):
        raw = dict(names)
        if any(
            isinstance(key, bool) or not isinstance(key, (int, np.integer))
            for key in raw
        ) or {int(key) for key in raw} != set(range(len(raw))):
            raise ValueError(
                "Florence class-name mapping keys must be contiguous IDs "
                "starting at zero."
            )
        ordered = [raw[index] for index in range(len(raw))]
    elif isinstance(names, (list, tuple)):
        ordered = list(names)
    else:
        raise TypeError(
            "Florence classes must be a list, tuple, or integer-key mapping."
        )
    normalized = [str(value).strip() for value in ordered]
    if not normalized or any(not value for value in normalized):
        raise ValueError("Florence requires at least one non-empty class name.")
    lowered = [value.lower() for value in normalized]
    if len(set(lowered)) != len(lowered):
        raise ValueError("Florence class names must be unique ignoring case.")
    return dict(enumerate(normalized))


def _strict_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _strict_finite_number(name: str, value: Any, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}.")
    return result


class CoreMLFlorenceRuntime:
    """Host-orchestrated open-vocabulary Florence-2-base runtime."""

    def __init__(
        self,
        bundle_path: str | os.PathLike[str],
        *,
        names: Mapping[int, str] | list[str] | tuple[str, ...],
        compute_units: str = "validated",
        coremltools_module: Any | None = None,
    ) -> None:
        if sys.platform != "darwin":
            raise RuntimeError(
                "Florence Core ML inference requires macOS 15 or later. "
                f"Current platform: {sys.platform}."
            )
        validated_names = _validated_names(names)
        ct = _package_ct(coremltools_module)
        info = validate_coreml_florence_bundle(
            bundle_path,
            coremltools_module=ct,
        )
        resolved_compute_units = resolve_coreml_runtime_compute_units(
            compute_units,
            info.metadata,
        )
        processor = _load_florence_processor(info.processor_path)
        unit = _compute_unit(ct, resolved_compute_units)
        models: dict[str, Any] = {}
        expected_runtime_metadata = stringify_florence_coreml_metadata(info.metadata)
        try:
            for function_name in FLORENCE_FUNCTION_NAMES:
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
                validate_florence_coreml_metadata(runtime_metadata)
                for key, expected in expected_runtime_metadata.items():
                    if runtime_metadata.get(key) != expected:
                        raise ValueError(
                            "Florence runtime metadata differs from the "
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
        self.names = validated_names
        self._name_to_id = {
            label.strip().lower(): class_id for class_id, label in self.names.items()
        }
        self._models: dict[str, Any] | None = models
        self._request_lock = threading.Lock()
        self._active_decode: _FlorenceDecodeRequest | None = None

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

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("Florence Core ML runtime is closed.")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _model(self, function_name: str) -> Any:
        if self._models is None:
            raise RuntimeError("Florence Core ML runtime is closed.")
        return self._models[function_name]

    def set_classes(
        self,
        names: Mapping[int, str] | list[str] | tuple[str, ...],
    ) -> None:
        """Replace the ordered open-vocabulary class list."""

        validated = _validated_names(names)
        with self._request_lock:
            if self.closed:
                raise RuntimeError("Florence Core ML runtime is closed.")
            self.names = validated
            self._name_to_id = {
                label.strip().lower(): class_id for class_id, label in validated.items()
            }

    def _encode(
        self,
        prepared: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        inputs = {
            FLORENCE_PIXEL_VALUES_INPUT: prepared[FLORENCE_PIXEL_VALUES_INPUT],
            FLORENCE_ENCODER_INPUT_IDS_INPUT: prepared[
                FLORENCE_ENCODER_INPUT_IDS_INPUT
            ],
            FLORENCE_ENCODER_ATTENTION_MASK_INPUT: prepared[
                FLORENCE_ENCODER_ATTENTION_MASK_INPUT
            ],
        }
        result = self._model(FLORENCE_ENCODE_FUNCTION).predict(inputs)
        expected_names = {
            FLORENCE_CROSS_KEY_OUTPUT,
            FLORENCE_CROSS_VALUE_OUTPUT,
        }
        key = _fp16_output(
            result,
            expected_names=expected_names,
            name=FLORENCE_CROSS_KEY_OUTPUT,
            shape=self.profile.single_cross_cache_shape,
        )
        value = _fp16_output(
            result,
            expected_names=expected_names,
            name=FLORENCE_CROSS_VALUE_OUTPUT,
            shape=self.profile.single_cross_cache_shape,
        )
        return key, value

    def _generate_prepared(
        self,
        prepared: Mapping[str, np.ndarray],
        *,
        max_new_tokens: int,
    ) -> tuple[tuple[int, ...], float]:
        cross_key, cross_value = self._encode(prepared)
        if self._active_decode is not None:
            raise RuntimeError("Florence runtime already has an active decode state.")
        request = _FlorenceDecodeRequest(
            self._model(FLORENCE_DECODE_FUNCTION),
            self.profile,
            cross_key_values=cross_key,
            cross_value_values=cross_value,
        )
        self._active_decode = request
        search = Florence2BeamSearch(
            max_new_tokens=max_new_tokens,
            vocab_size=self.profile.vocab_size,
        )
        token_ids = np.full(
            (self.profile.num_beams, 1),
            FLORENCE2_DECODER_START_TOKEN_ID,
            dtype=np.int32,
        )
        parent_indices = np.arange(
            self.profile.num_beams,
            dtype=np.int32,
        )
        cross_mask = prepared[FLORENCE_CROSS_ATTENTION_MASK_INPUT]
        try:
            while not search.done:
                logits = request.predict(
                    token_ids,
                    cross_attention_mask=cross_mask,
                    beam_parent_indices=parent_indices,
                )
                step = search.advance(logits)
                if step.done:
                    break
                assert (
                    step.next_token_ids is not None and step.parent_indices is not None
                )
                token_ids = np.ascontiguousarray(
                    np.asarray(step.next_token_ids, dtype=np.int32)[:, None]
                )
                parent_indices = np.ascontiguousarray(
                    np.asarray(step.parent_indices, dtype=np.int32)
                )
            return search.output_sequence, search.output_score
        except Exception:
            request.discard()
            raise
        finally:
            request.discard()
            self._active_decode = None

    def generate(
        self,
        image: ImageInput,
        *,
        max_new_tokens: int | None = None,
        color_format: str = "auto",
    ) -> dict[str, Any]:
        """Generate and parse one open-vocabulary detection response."""

        if self.closed:
            raise RuntimeError("Florence Core ML runtime is closed.")
        budget = (
            self.profile.max_new_tokens
            if max_new_tokens is None
            else _strict_int("max_new_tokens", max_new_tokens, minimum=1)
        )
        if budget > self.profile.max_new_tokens:
            raise ValueError(
                "max_new_tokens exceeds the pinned Florence decoder budget."
            )
        loaded = ImageLoader.load(image, color_format=color_format)
        with self._request_lock:
            if self.closed:
                raise RuntimeError("Florence Core ML runtime is closed.")
            class_names = [self.names[index] for index in range(len(self.names))]
            prepared = prepare_florence2_base_processor_batch(
                self.processor,
                loaded,
                class_names,
                profile=self.profile,
            )
            sequence, beam_score = self._generate_prepared(
                prepared,
                max_new_tokens=budget,
            )
            decoded = self.processor.batch_decode(
                np.asarray([sequence], dtype=np.int64),
                skip_special_tokens=False,
            )
            if (
                not isinstance(decoded, (list, tuple))
                or len(decoded) != 1
                or not isinstance(decoded[0], str)
            ):
                raise RuntimeError(
                    "Florence processor returned an invalid decoded response."
                )
            parsed = self.processor.post_process_generation(
                decoded[0],
                task=FLORENCE2_TASK,
                image_size=loaded.size,
            )
        if not isinstance(parsed, Mapping):
            raise TypeError("Florence processor returned an invalid parsed response.")
        return {
            "token_ids": list(sequence),
            "beam_score": beam_score,
            "text": decoded[0],
            "parsed": dict(parsed),
            "image_size": loaded.size,
        }

    def predict(
        self,
        source: ImageInput,
        *,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        classes: list[int] | tuple[int, ...] | None = None,
        max_new_tokens: int | None = None,
        color_format: str = "auto",
    ) -> dict[str, Any]:
        """Return LibreYOLO's detection dictionary for one image."""

        threshold = _strict_finite_number("conf", conf, minimum=0.0)
        _strict_finite_number("iou", iou, minimum=0.0)
        limit = _strict_int("max_det", max_det, minimum=0)
        allowed: set[int] | None = None
        if classes is not None:
            if not isinstance(classes, (list, tuple)):
                raise TypeError("classes must be a list or tuple of IDs.")
            allowed = set()
            for value in classes:
                class_id = _strict_int("class ID", value, minimum=0)
                allowed.add(class_id)
        generated = self.generate(
            source,
            max_new_tokens=max_new_tokens,
            color_format=color_format,
        )
        raw_task = generated["parsed"].get(FLORENCE2_TASK, {})
        if not isinstance(raw_task, Mapping):
            raise TypeError("Florence parsed task payload must be a mapping.")
        labels = raw_task.get(
            "bboxes_labels",
            raw_task.get("labels", []),
        )
        boxes = raw_task.get("bboxes", [])
        if not isinstance(labels, (list, tuple)) or not isinstance(
            boxes, (list, tuple)
        ):
            raise TypeError("Florence parsed boxes and labels must be sequences.")

        output_boxes: list[list[float]] = []
        output_scores: list[float] = []
        output_classes: list[int] = []
        if limit > 0 and threshold <= 1.0:
            for raw_box, raw_label in zip(boxes, labels):
                class_id = self._name_to_id.get(str(raw_label).strip().lower())
                if class_id is None or (
                    allowed is not None and class_id not in allowed
                ):
                    continue
                if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                    continue
                try:
                    box = [float(value) for value in raw_box]
                except (TypeError, ValueError):
                    continue
                if (
                    not all(np.isfinite(value) for value in box)
                    or box[2] <= box[0]
                    or box[3] <= box[1]
                ):
                    continue
                output_boxes.append(box)
                output_scores.append(1.0)
                output_classes.append(class_id)
                if len(output_boxes) >= limit:
                    break
        return {
            "boxes": output_boxes,
            "scores": output_scores,
            "classes": output_classes,
            "num_detections": len(output_boxes),
        }


__all__ = [
    "COREML_FLORENCE_BUNDLE_APACHE_LICENSE",
    "COREML_FLORENCE_BUNDLE_FORMAT",
    "COREML_FLORENCE_BUNDLE_MANIFEST",
    "COREML_FLORENCE_BUNDLE_MIT_LICENSE",
    "COREML_FLORENCE_BUNDLE_MODEL_ROOT",
    "COREML_FLORENCE_BUNDLE_NOTICE",
    "COREML_FLORENCE_BUNDLE_PROCESSOR_ROOT",
    "COREML_FLORENCE_BUNDLE_SCHEMA_VERSION",
    "COREML_FLORENCE_BUNDLE_SUFFIX",
    "CoreMLFlorenceBundleInfo",
    "CoreMLFlorenceRuntime",
    "build_coreml_florence_bundle",
    "validate_coreml_florence_bundle",
]
