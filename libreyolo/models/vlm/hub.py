"""Immutable Hugging Face Hub transport for LibreYOLO VLM artifacts.

VLM checkpoints are directory artifacts.  They cannot use the generic
single-file ``hf://`` checkpoint path without losing the processor and the
metadata which bind an adapter to its exact base model.  This module therefore
uses a separate, immutable URI:

``hf+vlm://owner/repository@0123456789abcdef...``

Only 40-character commit SHAs are accepted.  Downloads fetch the manifest
first and then fetch exactly the files named by that manifest.  Uploads accept
only a fully validated artifact, refuse existing repositories, create one
commit, and verify that commit through a fresh download.

``huggingface_hub`` remains an optional dependency and is imported lazily.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from contextlib import ExitStack
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

from .artifact import (
    VLM_ARTIFACT_MANIFEST,
    VLMArtifactInfo,
    read_vlm_artifact_manifest,
    validate_vlm_base_snapshot,
    validate_vlm_artifact,
)

VLM_HUB_URI_PREFIX = "hf+vlm://"

_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REPO_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?$")
_MAX_REPO_ID_LENGTH = 193
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_SNAPSHOT_COMPLETE_MARKER = ".libreyolo_snapshot_complete"

__all__ = [
    "VLM_HUB_URI_PREFIX",
    "VLMHubRef",
    "VLMBaseSnapshotInfo",
    "parse_vlm_hub_uri",
    "inspect_vlm_hub_artifact",
    "download_vlm_artifact",
    "ensure_vlm_base_snapshot",
    "push_vlm_artifact",
]


@dataclass(frozen=True)
class VLMHubRef:
    """A VLM artifact repository pinned to one immutable Hub commit."""

    repo_id: str
    revision: str

    @property
    def uri(self) -> str:
        """Return the canonical immutable VLM Hub URI."""
        return f"{VLM_HUB_URI_PREFIX}{self.repo_id}@{self.revision}"


@dataclass(frozen=True)
class VLMBaseSnapshotInfo:
    """A locally materialized base snapshot validated against an artifact."""

    root: Path
    identity: Mapping[str, object]


def _detached_json(value):
    """Copy recursively frozen validated JSON into ordinary containers."""
    if isinstance(value, Mapping):
        return {str(key): _detached_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_detached_json(item) for item in value]
    return copy.deepcopy(value)


def _valid_repo_id(repo_id: str) -> bool:
    if not repo_id or len(repo_id) > _MAX_REPO_ID_LENGTH:
        return False
    if repo_id.count("/") != 1 or "\\" in repo_id:
        return False
    if repo_id.casefold().endswith(".git"):
        return False
    owner, name = repo_id.split("/", 1)
    for segment in (owner, name):
        if (
            len(segment) > 96
            or not _REPO_SEGMENT_RE.fullmatch(segment)
            or "--" in segment
            or ".." in segment
        ):
            return False
    return True


def parse_vlm_hub_uri(source: str) -> VLMHubRef:
    """Parse a canonical, immutable ``hf+vlm://`` artifact URI.

    Mutable branches, tags, abbreviated hashes, bare repository ids, file
    suffixes, query strings, and fragments are intentionally unsupported.
    """
    if not isinstance(source, str):
        raise TypeError("A VLM Hub source must be a string.")
    if not source.startswith(VLM_HUB_URI_PREFIX):
        raise ValueError(
            "A VLM Hub source must use "
            f"'{VLM_HUB_URI_PREFIX}owner/repo@<40-character-commit-sha>'."
        )

    remainder = source[len(VLM_HUB_URI_PREFIX) :]
    if remainder.count("@") != 1:
        raise ValueError(
            "A VLM Hub source must contain exactly one '@' followed by an "
            "immutable 40-character commit SHA."
        )
    repo_id, revision = remainder.rsplit("@", 1)
    if not _valid_repo_id(repo_id):
        raise ValueError(
            f"Invalid Hugging Face repository id {repo_id!r}; expected 'owner/repo'."
        )
    if not _COMMIT_SHA_RE.fullmatch(revision):
        raise ValueError(
            "A VLM Hub source revision must be a lowercase, 40-character "
            "commit SHA; branches, tags, and abbreviated hashes are mutable."
        )

    ref = VLMHubRef(repo_id=repo_id, revision=revision)
    if ref.uri != source:
        # This also rejects whitespace, extra slashes, queries, and fragments
        # without trying to normalize an ambiguous remote reference.
        raise ValueError(f"Non-canonical VLM Hub source {source!r}; use {ref.uri!r}.")
    return ref


def _validate_repo_id(repo_id: str) -> None:
    if not isinstance(repo_id, str) or not _valid_repo_id(repo_id):
        raise ValueError(
            f"Invalid Hugging Face repository id {repo_id!r}; expected 'owner/repo'."
        )


def _validate_token(token: str | None) -> None:
    if token is not None and (not isinstance(token, str) or not token):
        raise TypeError("token must be a non-empty string or None.")


def _require_hub():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "VLM Hub transport requires the optional huggingface_hub package. "
            "Install it with: pip install libreyolo[hf]"
        ) from exc
    return huggingface_hub


def _is_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _check_destination(output_dir: str | Path) -> Path:
    try:
        output = Path(output_dir)
    except TypeError as exc:
        raise TypeError("output_dir must be a filesystem path.") from exc
    output = Path(os.path.abspath(os.fspath(output)))

    if _lexists(output):
        raise FileExistsError(
            f"Refusing to replace existing VLM artifact destination: {output}"
        )
    parent = output.parent
    if not parent.is_dir():
        raise FileNotFoundError(
            f"VLM artifact destination parent does not exist: {parent}"
        )

    # Reject redirected parents before any network work.  On Windows a
    # junction is a reparse point but is not reliably reported by is_symlink.
    current = parent
    while True:
        if current.is_symlink() or _is_reparse_point(current):
            raise ValueError(
                "VLM artifact destination must not be below a symlink, "
                f"junction, or reparse point: {current}"
            )
        if current == current.parent:
            break
        current = current.parent
    return output


def _artifact_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        not name
        or relative.is_absolute()
        or "\\" in name
        or any(part in ("", ".", "..") for part in relative.parts)
        or ":" in relative.parts[0]
    ):
        raise ValueError(f"Unsafe path in VLM artifact manifest: {name!r}")
    return root.joinpath(*relative.parts)


def _copy_regular_file(
    source: str | Path,
    destination: Path,
    *,
    max_bytes: int | None = None,
    limit_label: str = "Downloaded file",
) -> None:
    source_path = Path(source)
    try:
        source_size = source_path.stat().st_size
    except OSError as exc:
        raise FileNotFoundError(
            f"Downloaded Hub file is unavailable: {source_path}"
        ) from exc
    if not source_path.is_file():
        raise ValueError(f"Downloaded Hub path is not a regular file: {source_path}")
    if max_bytes is not None and source_size > max_bytes:
        raise ValueError(f"{limit_label} exceeds its {max_bytes}-byte safety limit.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with source_path.open("rb") as source_handle, destination.open("xb") as output:
            created = True
            copied = 0
            while True:
                chunk = source_handle.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                copied += len(chunk)
                if max_bytes is not None and copied > max_bytes:
                    raise ValueError(
                        f"{limit_label} exceeds its {max_bytes}-byte safety "
                        "limit while reading."
                    )
                output.write(chunk)
    except FileExistsError:
        raise ValueError(
            f"Duplicate path while staging VLM artifact: {destination}"
        ) from None
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise


def _hub_download(
    hub,
    ref: VLMHubRef,
    filename: str,
    *,
    token: str | None,
    local_files_only: bool,
    cache_dir: Path | None = None,
    force_download: bool = False,
    max_bytes: int | None = None,
    expected_bytes: int | None = None,
) -> str:
    kwargs: dict[str, object] = {
        "repo_id": ref.repo_id,
        "filename": filename,
        "revision": ref.revision,
        "token": token,
        "local_files_only": local_files_only,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if force_download:
        kwargs["force_download"] = True
    if not local_files_only and max_bytes is not None:
        dry_run_kwargs = dict(kwargs)
        dry_run_kwargs.pop("force_download", None)
        dry_run_kwargs["dry_run"] = True
        try:
            remote_info = hub.hf_hub_download(**dry_run_kwargs)
        except Exception as exc:
            _raise_hub_download_error(exc, ref, filename)
        remote_size = getattr(remote_info, "file_size", None)
        if (
            isinstance(remote_size, bool)
            or not isinstance(remote_size, int)
            or remote_size < 0
        ):
            raise ConnectionError(
                f"Hugging Face did not report a valid size for VLM artifact "
                f"file '{filename}' at commit '{ref.revision}'."
            )
        if remote_size > max_bytes:
            raise ValueError(
                f"VLM artifact file '{filename}' exceeds its {max_bytes}-byte "
                "remote download limit."
            )
        if expected_bytes is not None and remote_size != expected_bytes:
            raise ValueError(
                f"VLM artifact file '{filename}' has remote size {remote_size}, "
                f"expected {expected_bytes}."
            )
    try:
        return str(hub.hf_hub_download(**kwargs))
    except Exception as exc:
        _raise_hub_download_error(exc, ref, filename)


def _raise_hub_download_error(exc: Exception, ref: VLMHubRef, filename: str) -> None:
    """Translate Hub failures without exposing credentials or response bodies."""
    kinds = {exception_type.__name__ for exception_type in type(exc).__mro__}
    if kinds & {"GatedRepoError", "RepositoryNotFoundError"}:
        raise PermissionError(
            f"Cannot access VLM artifact repository '{ref.repo_id}'. "
            "Check repository access and Hugging Face authentication."
        ) from None
    if "RevisionNotFoundError" in kinds:
        raise FileNotFoundError(
            f"Commit '{ref.revision}' was not found in '{ref.repo_id}'."
        ) from None
    if "LocalEntryNotFoundError" in kinds:
        raise FileNotFoundError(
            f"VLM artifact file '{filename}' is not available in the local "
            "Hugging Face cache."
        ) from None
    if "EntryNotFoundError" in kinds:
        raise FileNotFoundError(
            f"VLM artifact file '{filename}' was not found at commit "
            f"'{ref.revision}' in '{ref.repo_id}'."
        ) from None
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        raise PermissionError(
            f"Access to VLM artifact repository '{ref.repo_id}' was denied. "
            "Check repository access and Hugging Face authentication."
        ) from None
    raise ConnectionError(
        f"Could not download VLM artifact file '{filename}' from "
        f"'{ref.repo_id}' at commit '{ref.revision}'."
    ) from None


def _read_manifest_from_hub(
    hub,
    ref: VLMHubRef,
    temporary_root: Path,
    *,
    token: str | None,
    local_files_only: bool,
    cache_dir: Path | None,
    force_download: bool,
) -> VLMArtifactInfo:
    temporary_root.mkdir(parents=True, exist_ok=True)
    cached_manifest = _hub_download(
        hub,
        ref,
        VLM_ARTIFACT_MANIFEST,
        token=token,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        force_download=force_download,
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    manifest_root = temporary_root / "manifest"
    manifest_root.mkdir()
    _copy_regular_file(
        cached_manifest,
        manifest_root / VLM_ARTIFACT_MANIFEST,
        max_bytes=_MAX_MANIFEST_BYTES,
        limit_label="VLM artifact manifest",
    )
    info = read_vlm_artifact_manifest(manifest_root, require_payload=False)
    if not local_files_only:
        _validate_remote_repository_tree(hub, ref, info, token=token)
    return info


def _validate_remote_repository_tree(
    hub,
    ref: VLMHubRef,
    info: VLMArtifactInfo,
    *,
    token: str | None,
) -> None:
    """Require the immutable online repo tree to equal the manifest inventory."""
    try:
        api = hub.HfApi(token=token)
        remote_files = list(
            api.list_repo_files(
                ref.repo_id,
                revision=ref.revision,
                repo_type="model",
            )
        )
    except Exception as exc:
        kinds = {exception_type.__name__ for exception_type in type(exc).__mro__}
        if kinds & {"GatedRepoError", "RepositoryNotFoundError"}:
            raise PermissionError(
                f"Cannot inspect VLM artifact repository '{ref.repo_id}'. "
                "Check repository access and Hugging Face authentication."
            ) from None
        if "RevisionNotFoundError" in kinds:
            raise FileNotFoundError(
                f"Commit '{ref.revision}' was not found in '{ref.repo_id}'."
            ) from None
        raise ConnectionError(
            f"Could not inspect the exact repository tree for '{ref.repo_id}' "
            f"at commit '{ref.revision}'."
        ) from None

    if any(not isinstance(name, str) for name in remote_files):
        raise ValueError("Hugging Face returned a non-string repository path.")
    expected = {VLM_ARTIFACT_MANIFEST, *info.files}
    actual = set(remote_files)
    if len(actual) != len(remote_files) or actual != expected:
        missing = sorted(expected - actual, key=str.casefold)
        extra = sorted(actual - expected, key=str.casefold)
        raise ValueError(
            f"VLM artifact repository tree does not match its manifest: "
            f"missing={missing}, extra={extra}."
        )


def _declared_artifact_sizes(info: VLMArtifactInfo) -> dict[str, int]:
    try:
        entries = info.manifest["files"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "VLM artifact manifest has no validated file inventory."
        ) from exc
    if not isinstance(entries, Sequence) or isinstance(
        entries, (str, bytes, bytearray)
    ):
        raise ValueError("VLM artifact manifest file inventory is invalid.")
    sizes: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("VLM artifact manifest file entry is invalid.")
        name = entry.get("path")
        size = entry.get("size")
        if (
            not isinstance(name, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or name in sizes
        ):
            raise ValueError("VLM artifact manifest file entry is invalid.")
        sizes[name] = size
    if set(sizes) != set(info.files):
        raise ValueError("VLM artifact manifest inventory changed after validation.")
    return sizes


def _stage_vlm_artifact(
    hub,
    ref: VLMHubRef,
    temporary_root: Path,
    *,
    token: str | None,
    local_files_only: bool,
    cache_dir: Path | None = None,
    force_download: bool = False,
) -> VLMArtifactInfo:
    manifest_info = _read_manifest_from_hub(
        hub,
        ref,
        temporary_root,
        token=token,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        force_download=force_download,
    )

    content_root = temporary_root / "content" / manifest_info.aggregate_sha256
    content_root.mkdir(parents=True)
    _copy_regular_file(
        manifest_info.root / VLM_ARTIFACT_MANIFEST,
        content_root / VLM_ARTIFACT_MANIFEST,
        max_bytes=_MAX_MANIFEST_BYTES,
        limit_label="VLM artifact manifest",
    )

    declared_sizes = _declared_artifact_sizes(manifest_info)
    for filename in sorted(manifest_info.files, key=str.casefold):
        cached_file = _hub_download(
            hub,
            ref,
            filename,
            token=token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            max_bytes=declared_sizes[filename],
            expected_bytes=declared_sizes[filename],
        )
        _copy_regular_file(
            cached_file,
            _artifact_path(content_root, filename),
            max_bytes=declared_sizes[filename],
            limit_label=f"VLM artifact file {filename!r}",
        )

    downloaded = validate_vlm_artifact(content_root)
    if (
        downloaded.aggregate_sha256 != manifest_info.aggregate_sha256
        or downloaded.files != manifest_info.files
        or dict(downloaded.manifest) != dict(manifest_info.manifest)
    ):
        raise ValueError("Downloaded VLM artifact changed after manifest inspection.")
    return downloaded


def inspect_vlm_hub_artifact(
    source: str,
    *,
    token: str | None = None,
    local_files_only: bool = False,
) -> Mapping[str, object]:
    """Fetch and strictly validate only an immutable artifact manifest.

    No model, processor, or tensor payload is downloaded.  The returned
    mapping is a detached copy and does not expose the Hub cache path.  Online
    inspection also requires the exact repository tree to match the manifest.
    Offline cache inspection cannot prove that the remote commit has no extra
    files; it validates the cached manifest itself only.
    """
    ref = parse_vlm_hub_uri(source)
    _validate_token(token)
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a bool.")
    hub = _require_hub()
    with tempfile.TemporaryDirectory(prefix="libreyolo-vlm-manifest-") as temporary:
        info = _read_manifest_from_hub(
            hub,
            ref,
            Path(temporary).resolve(),
            token=token,
            local_files_only=local_files_only,
            cache_dir=None,
            force_download=False,
        )
        return _detached_json(info.manifest)


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        identity = path.lstat()
    except OSError as exc:
        raise ValueError(
            f"VLM publication staging directory disappeared: {path}"
        ) from exc
    if (
        not stat.S_ISDIR(identity.st_mode)
        or path.is_symlink()
        or _is_reparse_point(path)
    ):
        raise ValueError(f"VLM publication staging path changed: {path}")
    return identity.st_dev, identity.st_ino


def _same_artifact(left: VLMArtifactInfo, right: VLMArtifactInfo) -> bool:
    return (
        left.aggregate_sha256 == right.aggregate_sha256
        and left.files == right.files
        and left.manifest == right.manifest
    )


def _atomic_rename_create_only(
    source: Path,
    destination: Path,
    *,
    expected_source_identity: tuple[int, int] | None = None,
) -> None:
    """Atomically rename a directory without ever replacing ``destination``."""
    if (
        expected_source_identity is not None
        and _directory_identity(source) != expected_source_identity
    ):
        raise ValueError("VLM publication staging directory changed before rename")
    if os.name == "nt":
        # Windows MoveFile semantics used by os.rename reject any existing
        # destination.  os.replace is deliberately not used.
        try:
            os.rename(source, destination)
        except (FileExistsError, PermissionError):
            if _lexists(destination):
                raise FileExistsError(
                    f"Refusing to replace existing VLM destination: {destination}"
                ) from None
            raise
        return

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
        if renameat2 is None:
            raise RuntimeError(
                "This Linux runtime does not provide atomic create-only renameat2."
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,  # AT_FDCWD
            source_bytes,
            -100,
            destination_bytes,
            1,  # RENAME_NOREPLACE
        )
    elif sys.platform == "darwin":
        renamex_np = getattr(ctypes.CDLL(None, use_errno=True), "renamex_np", None)
        if renamex_np is None:
            raise RuntimeError(
                "This macOS runtime does not provide atomic create-only renamex_np."
            )
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(
            source_bytes,
            destination_bytes,
            0x00000004,  # RENAME_EXCL
        )
    else:
        # Falling back to check-then-rename would let a concurrent empty
        # directory be replaced on POSIX.  Fail closed on an unverified OS.
        raise RuntimeError(
            f"Atomic create-only VLM publication is unsupported on {sys.platform!r}."
        )

    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            f"Refusing to replace existing VLM destination: {destination}"
        ) from None
    raise OSError(error, os.strerror(error), os.fspath(destination))


def _publish_create_only(staged: VLMArtifactInfo, output: Path) -> VLMArtifactInfo:
    # Copy into a hidden same-filesystem directory, validate that copy, then
    # expose the complete tree in one create-only rename.
    _check_destination(output)
    publication_stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.publishing-", dir=output.parent)
    )
    stage_identity = _directory_identity(publication_stage)
    published = False
    try:
        names = sorted((VLM_ARTIFACT_MANIFEST, *staged.files), key=str.casefold)
        declared_sizes = _declared_artifact_sizes(staged)
        for filename in names:
            limit = (
                _MAX_MANIFEST_BYTES
                if filename == VLM_ARTIFACT_MANIFEST
                else declared_sizes[filename]
            )
            _copy_regular_file(
                _artifact_path(staged.root, filename),
                _artifact_path(publication_stage, filename),
                max_bytes=limit,
                limit_label=f"VLM artifact file {filename!r}",
            )
        validated_stage = validate_vlm_artifact(publication_stage)
        if not _same_artifact(validated_stage, staged):
            raise ValueError("VLM artifact changed while preparing publication")
        if _directory_identity(publication_stage) != stage_identity:
            raise ValueError(
                "VLM publication staging directory changed after validation"
            )
        _atomic_rename_create_only(
            publication_stage,
            output,
            expected_source_identity=stage_identity,
        )
        published = True
        if _directory_identity(output) != stage_identity:
            raise ValueError("VLM artifact destination changed during publication")
        published_artifact = validate_vlm_artifact(output)
        if not _same_artifact(published_artifact, staged):
            raise ValueError("VLM artifact changed during publication")
        if _directory_identity(output) != stage_identity:
            raise ValueError("VLM artifact destination changed during validation")
        return published_artifact
    finally:
        # Once the directory is visible, another process may legitimately add
        # or change entries. Never recursively delete a published path after a
        # validation failure; report the failure and leave it for explicit
        # operator inspection.
        if not published:
            cleanup = publication_stage
            try:
                current = cleanup.lstat()
                safe_to_remove = (
                    stat.S_ISDIR(current.st_mode)
                    and not cleanup.is_symlink()
                    and not _is_reparse_point(cleanup)
                    and (current.st_dev, current.st_ino) == stage_identity
                )
            except OSError:
                safe_to_remove = False
            if safe_to_remove:
                shutil.rmtree(cleanup, ignore_errors=True)


def download_vlm_artifact(
    source: str,
    output_dir: str | Path,
    *,
    token: str | None = None,
    local_files_only: bool = False,
) -> VLMArtifactInfo:
    """Download and validate an immutable VLM artifact into a new directory.

    The destination and all of its ancestors must already be safe local
    directories, and the destination itself must not exist.  Offline mode uses
    only ``hf_hub_download(local_files_only=True)`` and never constructs an
    ``HfApi`` client.  It therefore validates the materialized allowlisted view
    but cannot prove that the remote commit contains no unmanifested extras.
    """
    ref = parse_vlm_hub_uri(source)
    _validate_token(token)
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a bool.")
    output = _check_destination(output_dir)
    hub = _require_hub()
    with tempfile.TemporaryDirectory(prefix="libreyolo-vlm-download-") as temporary:
        staged = _stage_vlm_artifact(
            hub,
            ref,
            Path(temporary).resolve(),
            token=token,
            local_files_only=local_files_only,
        )
        return _publish_create_only(staged, output)


def _checked_weights_root(size: str) -> Path:
    names = {"2b": "LibreQwen3VL2b", "4b": "LibreQwen3VL4b"}
    try:
        directory_name = names[size]
    except KeyError:
        raise ValueError(
            f"VLM Hub base snapshot materialization does not support size {size!r}."
        ) from None

    weights = Path(os.path.abspath("weights"))
    if not _lexists(weights):
        # Check the existing lexical ancestry before creating the repository's
        # conventional weights directory.
        current = weights.parent
        while True:
            if current.is_symlink() or _is_reparse_point(current):
                raise ValueError(
                    "VLM base snapshots must not be stored below a symlink, "
                    f"junction, or reparse point: {current}"
                )
            if current == current.parent:
                break
            current = current.parent
        try:
            weights.mkdir()
        except FileExistsError:
            pass
    if not weights.is_dir() or weights.is_symlink() or _is_reparse_point(weights):
        raise ValueError(
            f"The lexical VLM weights root must be a regular local directory: {weights}"
        )
    return weights / directory_name


def _validated_snapshot_marker(root: Path, *, source: str, revision: str) -> None:
    marker_path = root / _SNAPSHOT_COMPLETE_MARKER
    try:
        if marker_path.stat().st_size > 4096:
            raise ValueError("marker is too large")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"VLM base snapshot {root} has no readable completion marker."
        ) from exc
    expected = {"repo": source, "revision": revision}
    if marker != expected:
        raise ValueError(
            f"VLM base snapshot {root} completion marker does not match its "
            "artifact identity."
        )


def _snapshot_result(root: Path, expected: Mapping[str, object]) -> VLMBaseSnapshotInfo:
    validated = validate_vlm_base_snapshot(root, expected)
    source = expected["source"]
    revision = expected["revision"]
    if not isinstance(source, str) or not isinstance(revision, str):
        raise ValueError("VLM artifact has an invalid base snapshot identity.")
    _validated_snapshot_marker(root, source=source, revision=revision)
    return VLMBaseSnapshotInfo(
        root=root,
        identity=_detached_json(validated),
    )


def ensure_vlm_base_snapshot(
    info: VLMArtifactInfo,
    *,
    token: str | None = None,
    local_files_only: bool = False,
) -> VLMBaseSnapshotInfo:
    """Materialize and verify the exact Qwen base bound by a VLM artifact.

    Existing lexical ``weights/LibreQwen3VL{size}`` roots are accepted only
    when every expected file and the completion marker validate.  An invalid
    existing root is never repaired or overwritten.  Missing roots are built
    in isolated same-filesystem staging and atomically published create-only.
    """
    _validate_token(token)
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a bool.")
    try:
        artifact_root = info.root
    except AttributeError as exc:
        raise TypeError("info must be a validated VLMArtifactInfo.") from exc

    # Revalidate the artifact at the trust boundary.  A caller-constructed
    # dataclass must not be able to choose arbitrary remote files or roots.
    current = validate_vlm_artifact(artifact_root)
    if (
        current.aggregate_sha256 != info.aggregate_sha256
        or current.files != info.files
        or dict(current.manifest) != dict(info.manifest)
    ):
        raise ValueError("VLM artifact changed before base snapshot acquisition.")
    try:
        identity = current.manifest["identity"]
        family = identity["family"]
        size = identity["size"]
        expected = current.base_snapshot
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "VLM artifact does not expose a validated base snapshot identity."
        ) from exc
    if family != "qwen3vl" or size not in {"2b", "4b"}:
        raise ValueError(
            "VLM Hub base snapshot acquisition currently supports only "
            "trainable Qwen3-VL 2B and 4B artifacts."
        )
    if not isinstance(expected, Mapping):
        raise ValueError("VLM artifact base snapshot identity must be a mapping.")
    source = expected.get("source")
    revision = expected.get("revision")
    entries = expected.get("files")
    if (
        not isinstance(source, str)
        or not _valid_repo_id(source)
        or not isinstance(revision, str)
        or not _COMMIT_SHA_RE.fullmatch(revision)
        or not isinstance(entries, Sequence)
        or isinstance(entries, (str, bytes, bytearray))
        or not entries
    ):
        raise ValueError("VLM artifact has an invalid base snapshot identity.")
    from .qwen3vl import LibreQwen3VL

    if source != LibreQwen3VL.HF_REPOS.get(
        size
    ) or revision != LibreQwen3VL.HF_REVISIONS.get(size):
        raise ValueError(
            "VLM artifact base snapshot does not match LibreYOLO's canonical "
            f"Qwen3-VL {size} repository and immutable revision."
        )

    destination = _checked_weights_root(size)
    if _lexists(destination):
        return _snapshot_result(destination, expected)

    hub = _require_hub()
    ref = VLMHubRef(repo_id=source, revision=revision)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.{expected.get('aggregate_sha256', 'snapshot')}.staging-",
            dir=destination.parent,
        )
    )
    stage_stat = stage.lstat()
    stage_identity = stage_stat.st_dev, stage_stat.st_ino
    try:
        names: list[str] = []
        declared_sizes: dict[str, int] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError(
                    "VLM artifact base snapshot contains an invalid file entry."
                )
            name = entry.get("path")
            size_bytes = entry.get("size")
            if (
                not isinstance(name, str)
                or isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes < 0
            ):
                raise ValueError(
                    "VLM artifact base snapshot contains an invalid file entry."
                )
            names.append(name)
            declared_sizes[name] = size_bytes
        if names != sorted(names, key=str.casefold) or len(names) != len(
            {name.casefold() for name in names}
        ):
            raise ValueError(
                "VLM artifact base snapshot file paths must be unique "
                "case-insensitively and sorted by casefolded path."
            )

        for filename in names:
            cached_file = _hub_download(
                hub,
                ref,
                filename,
                token=token,
                local_files_only=local_files_only,
                max_bytes=declared_sizes[filename],
                expected_bytes=declared_sizes[filename],
            )
            _copy_regular_file(
                cached_file,
                _artifact_path(stage, filename),
                max_bytes=declared_sizes[filename],
                limit_label=f"VLM base snapshot file {filename!r}",
            )
        validate_vlm_base_snapshot(stage, expected)
        marker = stage / _SNAPSHOT_COMPLETE_MARKER
        with marker.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {"repo": source, "revision": revision},
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
        staged_result = _snapshot_result(stage, expected)
        _atomic_rename_create_only(stage, destination)
        result = _snapshot_result(destination, staged_result.identity)
        if dict(result.identity) != dict(staged_result.identity):
            raise ValueError("VLM base snapshot changed during atomic publication.")
        return result
    finally:
        try:
            current_stage = stage.lstat()
            safe_to_remove = (
                stat.S_ISDIR(current_stage.st_mode)
                and not stage.is_symlink()
                and not _is_reparse_point(stage)
                and (current_stage.st_dev, current_stage.st_ino) == stage_identity
            )
        except OSError:
            safe_to_remove = False
        if safe_to_remove:
            shutil.rmtree(stage, ignore_errors=True)


def _authorized_namespaces(identity: object) -> tuple[str | None, set[str]]:
    if not isinstance(identity, Mapping):
        return None, set()
    identity_map = cast(Mapping[str, object], identity)
    username = identity_map.get("name")
    if not isinstance(username, str) or not username:
        username = None
    namespaces = {username} if username else set()
    organizations = identity_map.get("orgs", [])
    if isinstance(organizations, (list, tuple)):
        for organization in organizations:
            if not isinstance(organization, Mapping):
                continue
            name = organization.get("name")
            if isinstance(name, str) and name:
                namespaces.add(name)
    return username, namespaces


def _authenticate_namespace(api, repo_id: str) -> None:
    try:
        identity = api.whoami()
    except Exception:
        raise PermissionError(
            f"Could not authenticate a Hugging Face identity for '{repo_id}'. "
            "Run `hf auth login` or pass a write-scoped token."
        ) from None
    username, namespaces = _authorized_namespaces(identity)
    owner = repo_id.split("/", 1)[0]
    if owner.casefold() not in {name.casefold() for name in namespaces}:
        visible = sorted(namespaces, key=str.casefold)
        raise PermissionError(
            f"Cannot push to '{repo_id}': its owner is not the authenticated "
            f"user or one of that user's organizations (identity={username!r}, "
            f"namespaces={visible!r})."
        )


def _ensure_empty_repository(api, repo_id: str, *, private: bool) -> str:
    try:
        files = list(api.list_repo_files(repo_id, repo_type="model"))
    except Exception as exc:
        kinds = {exception_type.__name__ for exception_type in type(exc).__mro__}
        if "RepositoryNotFoundError" not in kinds:
            raise PermissionError(
                f"Could not inspect Hugging Face repository '{repo_id}' before "
                "upload. No files were uploaded."
            ) from None
        try:
            api.create_repo(
                repo_id,
                private=private,
                exist_ok=False,
                repo_type="model",
            )
        except Exception:
            raise PermissionError(
                f"Could not create empty Hugging Face repository '{repo_id}'. "
                "No files were uploaded."
            ) from None
        try:
            repository = api.repo_info(
                repo_id,
                repo_type="model",
                files_metadata=False,
            )
            parent_commit = getattr(repository, "sha", None)
            commits = list(api.list_repo_commits(repo_id, repo_type="model"))
            generated_files = list(api.list_repo_files(repo_id, repo_type="model"))
        except Exception:
            raise PermissionError(
                f"Could not verify the initial private state of newly created "
                f"repository '{repo_id}'. No artifact commit was created."
            ) from None
        if not isinstance(parent_commit, str) or not _COMMIT_SHA_RE.fullmatch(
            parent_commit
        ):
            raise RuntimeError(
                f"New Hugging Face repository '{repo_id}' has no immutable "
                "initial commit for optimistic concurrency control. No artifact "
                "commit was created."
            )
        commit_ids = [getattr(commit, "commit_id", None) for commit in commits]
        if commit_ids != [parent_commit]:
            raise FileExistsError(
                f"New Hugging Face repository '{repo_id}' changed during "
                "creation. No artifact commit was created."
            )
        if set(generated_files) - {".gitattributes"}:
            raise FileExistsError(
                f"New Hugging Face repository '{repo_id}' unexpectedly contains "
                f"files: {sorted(generated_files, key=str.casefold)}. No artifact "
                "commit was created."
            )
        return parent_commit

    # An existing empty repository can still be public.  Uploading to it while
    # private=True would silently publish the artifact, and a concurrent first
    # commit could race ours.  This transport is intentionally create-only at
    # the repository level as well as the file level.
    preview = sorted(str(name) for name in files)[:5]
    detail = f" Existing files include: {preview}" if preview else ""
    raise FileExistsError(
        f"Refusing to upload VLM artifact to existing repository '{repo_id}'."
        f"{detail} Choose a new repository id."
    )


def _stage_local_artifact_for_upload(
    artifact: VLMArtifactInfo, staging_root: Path
) -> VLMArtifactInfo:
    staging_root.mkdir()
    declared_sizes = _declared_artifact_sizes(artifact)
    for filename in sorted((VLM_ARTIFACT_MANIFEST, *artifact.files), key=str.casefold):
        limit = (
            _MAX_MANIFEST_BYTES
            if filename == VLM_ARTIFACT_MANIFEST
            else declared_sizes[filename]
        )
        _copy_regular_file(
            _artifact_path(artifact.root, filename),
            _artifact_path(staging_root, filename),
            max_bytes=limit,
            limit_label=f"VLM artifact file {filename!r}",
        )
    staged = validate_vlm_artifact(staging_root)
    if (
        staged.aggregate_sha256 != artifact.aggregate_sha256
        or staged.files != artifact.files
        or dict(staged.manifest) != dict(artifact.manifest)
    ):
        raise ValueError("VLM artifact changed while preparing its upload.")
    return staged


def push_vlm_artifact(
    path: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = True,
) -> str:
    """Upload one validated VLM directory artifact in one immutable commit.

    The repository must not already exist, so no files can be overwritten.
    New repositories start private; ``private=False``
    changes visibility only after a fresh remote verification succeeds.
    Success returns the canonical URI pinned to the verified commit, not a
    mutable repository URL.
    """
    _validate_repo_id(repo_id)
    _validate_token(token)
    if not isinstance(private, bool):
        raise TypeError("private must be a bool.")

    # This is intentionally before importing or calling huggingface_hub.  A
    # malformed local artifact must fail without authentication or network.
    artifact = validate_vlm_artifact(path)
    with tempfile.TemporaryDirectory(prefix="libreyolo-vlm-upload-") as upload_temp:
        upload_root = Path(upload_temp).resolve()
        stable = _stage_local_artifact_for_upload(artifact, upload_root / "artifact")

        # Network and authentication begin only after the isolated copy has
        # independently passed the complete artifact validator.
        hub = _require_hub()
        try:
            api = hub.HfApi(token=token)
        except Exception:
            raise PermissionError(
                f"Could not initialize Hugging Face access for '{repo_id}'."
            ) from None
        _authenticate_namespace(api, repo_id)
        # Even an explicitly public artifact starts private.  Visibility is
        # changed only after the exact committed tree and every payload hash
        # have passed a fresh remote verification.
        parent_commit = _ensure_empty_repository(api, repo_id, private=True)
        if ".gitattributes" not in stable.files:
            raise ValueError(
                "VLM artifact must inventory the canonical Hub .gitattributes file."
            )

        filenames = sorted((VLM_ARTIFACT_MANIFEST, *stable.files), key=str.casefold)
        with ExitStack() as open_files:
            operations = []
            for filename in filenames:
                # Hold an already-open descriptor to the private validated
                # copy.  Replacing or mutating the caller's original path
                # after validation cannot alter the bytes handed to the Hub.
                handle = open_files.enter_context(
                    _artifact_path(stable.root, filename).open("rb")
                )
                operations.append(
                    hub.CommitOperationAdd(
                        path_in_repo=filename,
                        path_or_fileobj=handle,
                    )
                )
            try:
                commit = api.create_commit(
                    repo_id=repo_id,
                    repo_type="model",
                    operations=operations,
                    commit_message="Upload LibreYOLO VLM artifact",
                    create_pr=False,
                    parent_commit=parent_commit,
                )
            except Exception:
                raise PermissionError(
                    f"Could not commit VLM artifact to '{repo_id}'. No existing "
                    "repository files were replaced."
                ) from None

        revision = getattr(commit, "oid", None)
        if not isinstance(revision, str) or not _COMMIT_SHA_RE.fullmatch(revision):
            raise RuntimeError(
                "Hugging Face did not return an immutable 40-character commit SHA "
                f"for the VLM artifact uploaded to '{repo_id}'."
            )
        ref = VLMHubRef(repo_id=repo_id, revision=revision)

        # A private, empty cache prevents this check from succeeding on files
        # read before the commit. force_download requires a remote fetch.
        downloaded = _stage_vlm_artifact(
            hub,
            ref,
            upload_root / "verification",
            token=token,
            local_files_only=False,
            cache_dir=upload_root / "hub-cache",
            force_download=True,
        )
        if (
            downloaded.aggregate_sha256 != stable.aggregate_sha256
            or downloaded.files != stable.files
            or dict(downloaded.manifest) != dict(stable.manifest)
        ):
            raise RuntimeError(
                f"Fresh verification of VLM artifact commit '{revision}' in "
                f"'{repo_id}' did not match the local artifact."
            )

        if not private:
            try:
                api.update_repo_settings(
                    repo_id,
                    private=False,
                    repo_type="model",
                )
            except Exception:
                raise PermissionError(
                    f"Verified VLM artifact '{ref.uri}' was uploaded but could "
                    "not be made public. It remains private."
                ) from None

        return ref.uri
