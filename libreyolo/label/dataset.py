"""Dataset session for LibreLabel: enumerate images and round-trip YOLO labels.

Thin wrapper over LibreYOLO's own ``load_data_config`` / ``img2label_paths`` so
that the image<->label mapping and ``data.yaml`` resolution are *identical* to
what training uses -- LibreLabel writes labels exactly where the trainer reads
them. No database; the filesystem dataset is the store.
"""

from __future__ import annotations

import hashlib
import errno
import json
import math
import os
import random
import re
import stat
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from libreyolo.data.utils import get_img_files, img2label_paths, load_data_config

from .labelio import (
    format_annotations,
    has_degenerate_polygon,
    has_obb_shaped_rows,
    has_out_of_bounds_coords,
    has_out_of_range_rows,
    has_unsupported_rows,
    has_zero_area_box,
    parse_annotations,
    sanitize_annotations,
)


_UPLOAD_LOCK = threading.RLock()
_LABEL_LOCK = threading.RLock()
_SIDECAR_LOCK = threading.RLock()


def _path_identity(path) -> str:
    """Canonical identity using the current platform's path case semantics."""
    value = Path(path).expanduser()
    try:
        value = value.resolve(strict=False)
    except (OSError, RuntimeError):
        value = Path(os.path.abspath(os.path.normpath(str(value))))
    return os.path.normcase(str(value))


def _portable_path_identity(path) -> str:
    """Case-folded identity for locks and portable collision detection."""
    return _path_identity(path).casefold()


def _windows_handle_identity_matches(
    expected: Tuple[int, int], observed: Tuple[int, int]
) -> bool:
    """Compare CPython stat identity with the full Windows handle identity."""
    expected_volume, expected_file = expected
    observed_volume, observed_file = observed
    if expected_file != observed_file:
        return False
    # CPython 3.10 exposes the legacy 32-bit Windows volume serial in st_dev;
    # newer CPython exposes the 64-bit FileIdInfo value used by the handle API.
    return expected_volume == observed_volume or (
        0 <= expected_volume <= 0xFFFFFFFF
        and expected_volume == observed_volume & 0xFFFFFFFF
    )


@contextmanager
def _interprocess_path_lock(path: Path, *, namespace: str = "librelabel-locks"):
    """Serialize one canonical filesystem target across LibreLabel processes."""
    lock_root = Path(tempfile.gettempdir()) / namespace
    lock_root.mkdir(parents=True, exist_ok=True)
    # Preserve the historical case-folded namespace so older LibreLabel
    # processes and case-insensitive filesystems lock the same target.
    identity = _portable_path_identity(path).encode("utf-8")
    lock_path = lock_root / (hashlib.sha256(identity).hexdigest() + ".lock")
    with open(lock_path, "a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _interprocess_upload_lock(dst: Path):
    """Serialize upload/finalize operations for one destination across processes."""
    # Keep the historical namespace so concurrently running older LibreLabel
    # processes still participate in the same upload/finalization lock.
    with _interprocess_path_lock(dst, namespace="librelabel-upload-locks"):
        yield


def _publish_no_clobber(temp_path: Path, target: Path) -> None:
    """Atomically replace an exclusively-created placeholder with a staged file."""
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(target, flags, 0o644)
    os.close(fd)
    try:
        os.replace(temp_path, target)
    except BaseException:
        try:
            target.unlink()
        except OSError:
            pass
        raise


def _names_to_list(names) -> List[str]:
    """Normalise ``names`` (dict ``{0: cat}`` or list) to an ordered list."""
    if names is None:
        return []
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return [str(n) for n in names]


def _resolve_data_arg(data: str) -> str:
    """Accept a ``data.yaml`` path, or a directory that contains one."""
    p = Path(data)
    if p.is_dir():
        for cand in ("data.yaml", "dataset.yaml"):
            if (p / cand).exists():
                return str(p / cand)
        yamls = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
        if yamls:
            return str(yamls[0])
        raise FileNotFoundError(f"No dataset YAML found in directory: {p}")
    return data


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + ``os.replace`` so a label is never half-written."""
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = 0o644
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def folder_yaml(folder: str) -> Optional[str]:
    """Return an existing dataset YAML inside ``folder`` (``data.yaml`` /
    ``dataset.yaml``, else the first ``*.yaml`` / ``*.yml``), or ``None``."""
    p = Path(folder)
    if not p.is_dir():
        return None
    for cand in ("data.yaml", "dataset.yaml"):
        if (p / cand).exists():
            return str(p / cand)
    ys = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
    return str(ys[0]) if ys else None


def count_images(folder: str) -> int:
    """How many supported images live under ``folder`` (recursive); 0 if none."""
    p = Path(folder)
    if not p.is_dir():
        return 0
    try:
        return len(get_img_files(p))
    except (FileNotFoundError, ValueError):
        return 0


def scaffold_data_yaml(folder: str, names: Optional[List[str]] = None,
                       task: Optional[str] = None) -> str:
    """Write a minimal LibreYOLO ``data.yaml`` for a bare folder of images.

    The folder of images *is* the dataset: the YAML is written beside the images
    (the exact layout ``libreyolo train`` reads) with a single ``train`` split
    pointing at the folder, so labels round-trip alongside the images and flow
    straight into training -- no export, no copy, nothing moved. The recursive
    scan means this works for a flat folder *and* an ``images/`` sub-tree (where
    the ``images``->``labels`` convention puts labels in a parallel ``labels/``).

    Returns the path to the written YAML. Raises ``FileNotFoundError`` if the
    folder is missing or holds no supported images, and ``FileExistsError`` if a
    dataset YAML is already there (open that instead of overwriting it).
    """
    p = Path(folder)
    if not p.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    existing = folder_yaml(folder)
    if existing:
        raise FileExistsError(existing)
    if not get_img_files(p):
        raise FileNotFoundError(f"No images found in {folder}")
    classes = [str(n).strip() for n in (names or []) if str(n).strip()]
    cfg = {
        "path": p.resolve().as_posix(),   # forward slashes: unambiguous in YAML on every OS
        "train": ".",
        "names": classes,
        "nc": len(classes),
    }
    # LibreLabel knows the user's selected authoring task; stamp even detection so
    # later transforming exports never have to guess an optional schema intent.
    cfg["task"] = str(task or "detect").strip().lower()
    text = (
        "# LibreLabel project -- created from a folder of images.\n"
        "# Labels are written next to the images, where `libreyolo train` reads them.\n"
        "# Add a `val:` split (a held-out set) before training for real.\n\n"
        + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    )
    out = p / "data.yaml"
    _atomic_write_text(out, text)
    return str(out)


def update_class_names(yaml_file: str, names: List[str]) -> None:
    """Rewrite a dataset YAML's ``names`` / ``nc`` in place, preserving every
    other key. Callers must only rename or append (never delete or reorder) so
    existing label class ids keep their meaning."""
    p = Path(yaml_file)
    original = p.read_text(encoding="utf-8")
    cfg = yaml.safe_load(original) or {}
    cfg["names"] = list(names)
    cfg["nc"] = len(names)
    # Keep the leading comment block (LibreLabel project hints) that safe_dump drops.
    comment = "\n".join(
        line for line in original.splitlines() if line.lstrip().startswith("#")
    )
    body = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    _atomic_write_text(p, (comment + "\n\n" + body) if comment else body)


# Image types the upload wizard accepts (matches what the labeler can render).
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _write_json_atomic(path: Path, obj) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False))


def _assert_unique_image_stems(paths, context: str) -> None:
    """Reject image sets whose derived ``<stem>.txt`` label names collide."""
    seen = {}
    for value in paths:
        path = Path(value)
        key = path.stem.casefold()
        if key in seen:
            raise ValueError(
                f"{context}: {seen[key].name} and {path.name} share the label basename "
                f"{path.stem!r}; rename one image before continuing."
            )
        seen[key] = path


def _move_validation_images(
    paths: List[Path], val_dir: Path
) -> Tuple[List[Tuple[Path, Path]], bool]:
    """Move a planned validation subset, rolling back the whole subset on error."""
    moved: List[Tuple[Path, Path]] = []
    created_val_dir = not val_dir.exists()
    try:
        val_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            dest = val_dir / src.name
            if dest.exists():
                raise FileExistsError(f"Validation destination already exists: {dest}")
            os.replace(src, dest)
            moved.append((src, dest))
    except BaseException:
        for src, dest in reversed(moved):
            if dest.exists() and not src.exists():
                os.replace(dest, src)
        if created_val_dir:
            try:
                val_dir.rmdir()
            except OSError:
                pass
        raise
    return moved, created_val_dir


def _rollback_validation_images(
    moved: List[Tuple[Path, Path]], val_dir: Path, *, remove_val_dir: bool
) -> None:
    for src, dest in reversed(moved):
        if dest.exists() and not src.exists():
            os.replace(dest, src)
    if remove_val_dir:
        try:
            val_dir.rmdir()
        except OSError:
            pass


def load_sidecar(base: str) -> dict:
    """Read the optional ``librelabel.json`` sidecar (project name + per-class
    colors) the New Project wizard writes next to ``data.yaml``. Convenience only;
    a missing or malformed sidecar is never fatal."""
    try:
        p = Path(base) / "librelabel.json"
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        pass
    return {}


def _project_root(data: str) -> Path:
    """The folder LibreLabel treats as a project: the dir holding ``data.yaml``."""
    p = Path(data).expanduser()
    if p.is_dir():
        return p
    if p.suffix.lower() in (".yaml", ".yml"):
        if not p.is_file():
            raise FileNotFoundError(f"Not a dataset YAML: {p}")
        return p.parent
    return p


def _is_link_or_reparse_point(path: Path) -> bool:
    """Return whether moving ``path`` would follow an alias to another tree."""
    try:
        info = path.lstat()
    except OSError:
        return False
    return path.is_symlink() or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _validated_project_root(
    lexical_root: Path,
    *,
    expected: Optional[Tuple[str, Tuple[int, int]]] = None,
) -> Tuple[Path, Tuple[str, Tuple[int, int]]]:
    """Resolve a non-linked project root and capture its filesystem identity."""
    if any(
        _is_link_or_reparse_point(path)
        for path in reversed([lexical_root, *lexical_root.parents])
    ):
        raise ValueError(
            "Refusing to trash a project opened through a directory link; "
            "reopen its real path first."
        )
    try:
        canonical = lexical_root.resolve(strict=True)
        info = canonical.stat()
    except (OSError, RuntimeError) as exc:
        raise ValueError("The project path changed during trash validation.") from exc
    if not stat.S_ISDIR(info.st_mode) or _is_link_or_reparse_point(canonical):
        raise ValueError("The project path changed during trash validation.")
    snapshot = (
        os.path.normcase(str(canonical)),
        (int(info.st_dev), int(info.st_ino)),
    )
    if expected is not None and snapshot != expected:
        raise ValueError("The project path changed during trash validation.")
    return canonical, snapshot


def _move_validated_project_root(
    source: Path,
    dest: Path,
    expected: Tuple[str, Tuple[int, int]],
    expected_trash_identity: Tuple[int, int],
) -> None:
    """Move the validated directory without resolving its source path again."""
    if os.name == "nt":
        _move_validated_project_root_windows(
            source, dest, expected[1], expected_trash_identity
        )
    else:
        _move_validated_project_root_posix(
            source, dest, expected[1], expected_trash_identity
        )


def _move_validated_project_root_posix(
    source: Path,
    dest: Path,
    expected_identity: Tuple[int, int],
    expected_trash_identity: Tuple[int, int],
) -> None:
    """Rename through held directory descriptors so ancestor swaps cannot redirect it."""
    if os.rename not in os.supports_dir_fd:
        raise OSError("Safe directory-relative rename is unavailable on this platform.")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)

    def open_directory_strict(path: Path) -> int:
        absolute = Path(os.path.abspath(os.fspath(path)))
        descriptor = os.open(absolute.anchor or os.sep, flags | nofollow)
        try:
            for component in absolute.parts[1:]:
                child = os.open(component, flags | nofollow, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    descriptors = []
    try:
        parent_fd = open_directory_strict(source.parent)
        descriptors.append(parent_fd)
        root_fd = os.open(source.name, flags | nofollow, dir_fd=parent_fd)
        descriptors.append(root_fd)
        trash_fd = open_directory_strict(dest.parent)
        descriptors.append(trash_fd)
        root_info = os.fstat(root_fd)
        entry_info = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        trash_info = os.fstat(trash_fd)
        root_identity = (int(root_info.st_dev), int(root_info.st_ino))
        entry_identity = (int(entry_info.st_dev), int(entry_info.st_ino))
        if (
            root_identity != expected_identity
            or entry_identity != expected_identity
            or not stat.S_ISDIR(root_info.st_mode)
            or not stat.S_ISDIR(entry_info.st_mode)
        ):
            raise ValueError("The project path changed before it could be trashed.")
        trash_identity = (int(trash_info.st_dev), int(trash_info.st_ino))
        if trash_identity != expected_trash_identity:
            raise ValueError("The trash directory changed before the project could be moved.")
        if trash_identity[0] != expected_identity[0]:
            raise ValueError(
                "Cannot safely trash a project across filesystem boundaries; "
                "move it manually instead."
            )
        try:
            os.stat(dest.name, dir_fd=trash_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Trash destination already exists: {dest}")
        try:
            os.rename(
                source.name,
                dest.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=trash_fd,
            )
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                raise ValueError(
                    "Cannot safely trash a project across filesystem boundaries; "
                    "move it manually instead."
                ) from exc
            raise
        moved_fd = os.open(dest.name, flags | nofollow, dir_fd=trash_fd)
        descriptors.append(moved_fd)
        moved_info = os.fstat(moved_fd)
        moved_identity = (int(moved_info.st_dev), int(moved_info.st_ino))
        if moved_identity != expected_identity:
            raise RuntimeError(
                "The project changed during trash; an unexpected directory was "
                f"quarantined at {dest}."
            )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _move_validated_project_root_windows(
    source: Path,
    dest: Path,
    expected_identity: Tuple[int, int],
    expected_trash_identity: Tuple[int, int],
) -> None:
    """Rename the exact validated Windows directory through its open handle."""
    import ctypes
    from ctypes import wintypes

    class _FileId128(ctypes.Structure):
        _fields_ = [("identifier", ctypes.c_ubyte * 16)]

    class _FileIdInfo(ctypes.Structure):
        _fields_ = [
            ("volume_serial_number", ctypes.c_ulonglong),
            ("file_id", _FileId128),
        ]

    class _FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("reparse_tag", wintypes.DWORD),
        ]

    class _FileRenameHeader(ctypes.Structure):
        _fields_ = [
            ("replace_if_exists", wintypes.DWORD),
            ("root_directory", wintypes.HANDLE),
            ("file_name_length", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    get_info = kernel32.GetFileInformationByHandleEx
    get_info.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    get_info.restype = wintypes.BOOL
    set_info = kernel32.SetFileInformationByHandle
    set_info.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    set_info.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    delete_access = 0x00010000
    read_attributes = 0x00000080
    share_read = 0x00000001
    share_write = 0x00000002
    open_existing = 3
    backup_semantics = 0x02000000
    open_reparse_point = 0x00200000
    file_attribute_directory = 0x00000010
    file_attribute_reparse_point = 0x00000400
    file_attribute_tag_info = 9
    file_rename_info = 3
    file_id_info = 18
    invalid_handle = ctypes.c_void_p(-1).value

    def extended_path(path: Path) -> str:
        value = os.path.abspath(os.fspath(path))
        if value.startswith("\\\\?\\"):
            return value
        if value.startswith("\\\\"):
            return "\\\\?\\UNC\\" + value[2:]
        return "\\\\?\\" + value

    def open_directory(path: Path, access: int):
        opened = create_file(
            extended_path(path),
            access,
            share_read | share_write,
            None,
            open_existing,
            backup_semantics | open_reparse_point,
            None,
        )
        if opened == invalid_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        return opened

    def directory_identity(opened, *, message: str) -> Tuple[int, int]:
        identity = _FileIdInfo()
        if not get_info(
            opened,
            file_id_info,
            ctypes.byref(identity),
            ctypes.sizeof(identity),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        attributes = _FileAttributeTagInfo()
        if not get_info(
            opened,
            file_attribute_tag_info,
            ctypes.byref(attributes),
            ctypes.sizeof(attributes),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if (
            not attributes.file_attributes & file_attribute_directory
            or attributes.file_attributes & file_attribute_reparse_point
        ):
            raise ValueError(message)
        return (
            int(identity.volume_serial_number),
            int.from_bytes(bytes(identity.file_id.identifier), "little"),
        )

    source_handle = open_directory(source, delete_access | read_attributes)
    trash_handle = None
    try:
        handle_identity = directory_identity(
            source_handle,
            message="The project path changed before it could be trashed.",
        )
        if not _windows_handle_identity_matches(expected_identity, handle_identity):
            raise ValueError("The project path changed before it could be trashed.")
        trash_handle = open_directory(dest.parent, read_attributes)
        trash_identity = directory_identity(
            trash_handle,
            message="The trash directory changed before the project could be moved.",
        )
        if not _windows_handle_identity_matches(
            expected_trash_identity, trash_identity
        ):
            raise ValueError("The trash directory changed before the project could be moved.")
        if trash_identity[0] != handle_identity[0]:
            raise ValueError(
                "Cannot safely trash a project across filesystem boundaries; "
                "move it manually instead."
            )

        encoded_dest = extended_path(dest).encode("utf-16-le")
        name_offset = (
            _FileRenameHeader.file_name_length.offset + ctypes.sizeof(wintypes.DWORD)
        )
        buffer = ctypes.create_string_buffer(name_offset + len(encoded_dest) + 2)
        header = _FileRenameHeader.from_buffer(buffer)
        header.replace_if_exists = 0
        header.root_directory = None
        header.file_name_length = len(encoded_dest)
        ctypes.memmove(
            ctypes.addressof(buffer) + name_offset,
            encoded_dest,
            len(encoded_dest),
        )
        if not set_info(source_handle, file_rename_info, buffer, len(buffer)):
            error = ctypes.get_last_error()
            if error == 17:
                raise ValueError(
                    "Cannot safely trash a project across filesystem boundaries; "
                    "move it manually instead."
                )
            raise ctypes.WinError(error)
    finally:
        if trash_handle is not None:
            close_handle(trash_handle)
        close_handle(source_handle)


def set_sidecar_name(data: str, name: str) -> str:
    """Update (or create) the project display name in the ``librelabel.json``
    sidecar next to ``data.yaml``. Returns the project root path."""
    return update_sidecar(data, name=str(name))


def update_sidecar(data: str, **fields) -> str:
    """Merge ``fields`` (name / description / instructions / ...) into the
    ``librelabel.json`` sidecar next to ``data.yaml``; ``None`` values are
    ignored. Returns the project root path."""
    root = _project_root(data)
    sidecar = root / "librelabel.json"
    with _SIDECAR_LOCK, _interprocess_path_lock(sidecar):
        sc = load_sidecar(str(root)) or {}
        for k, v in fields.items():
            if v is not None:
                sc[k] = v
        _write_json_atomic(sidecar, sc)
    return str(root)


def trash_project(data: str) -> str:
    """Soft-delete a project: move its whole folder to ``~/.librelabel/trash/``
    (recoverable) rather than erasing anything. Returns the trash path."""
    requested = Path(data).expanduser()
    if requested.suffix.lower() in (".yaml", ".yml") and _is_link_or_reparse_point(
        requested
    ):
        raise ValueError(
            "Refusing to trash a project opened through a filesystem link; "
            "reopen its real path first."
        )
    root = _project_root(data)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a folder: {root}")
    lexical_root = Path(os.path.abspath(str(root.expanduser())))
    canonical, snapshot = _validated_project_root(lexical_root)
    anchor = Path(canonical.anchor)
    home = Path.home().resolve(strict=False)
    registry = (home / ".librelabel").resolve(strict=False)
    if (
        canonical == anchor
        or canonical == home
        or home.is_relative_to(canonical)
        or registry == canonical
        or registry.is_relative_to(canonical)
    ):
        raise ValueError("Refusing to trash a filesystem, home, or LibreLabel root.")
    trash_root = Path.home() / ".librelabel" / "trash"
    trash_root.mkdir(parents=True, exist_ok=True)
    trash_lexical = Path(os.path.abspath(str(trash_root.expanduser())))
    trash, trash_snapshot = _validated_project_root(trash_lexical)
    if canonical == trash or canonical.is_relative_to(trash):
        raise ValueError("This project is already inside LibreLabel's trash.")
    dest = trash / (
        f"{time.time_ns()}-{uuid.uuid4().hex}-{root.name or 'dataset'}"
    )
    canonical, _ = _validated_project_root(lexical_root, expected=snapshot)
    _move_validated_project_root(canonical, dest, snapshot, trash_snapshot[1])
    return str(dest)


def save_uploaded_image(dst: str, name: str, data: bytes) -> str:
    """Write one browser-uploaded image into ``<dst>/images/train/`` (created on
    demand). The name is reduced to a safe basename and only known image
    extensions are accepted, so an upload can never escape the destination.
    Refuses to write into an existing dataset or over an existing file: uploads
    happen *before* the create-project guard runs, so without this a mistaken
    destination would silently overwrite images (whose labels then describe the
    wrong pixels)."""
    safe = Path(str(name)).name.strip()
    if not safe:
        raise ValueError("empty filename")
    if Path(safe).suffix.lower() not in IMG_EXTS:
        raise ValueError(f"unsupported file type: {safe}")
    if not data:
        raise ValueError("empty file")
    with _UPLOAD_LOCK, _interprocess_upload_lock(Path(dst)):
        if folder_yaml(dst):
            raise FileExistsError(
                "That folder is already a dataset (it has a data.yaml) - open it instead, "
                "or pick an empty folder for the new project.")
        out_dir = Path(dst) / "images" / "train"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / safe
        # Image suffixes disappear when the trainer derives ``<stem>.txt``. Treat
        # foo.jpg + foo.png as the same occupied label slot, including on
        # case-sensitive filesystems where the eventual training/export target may
        # still be case-insensitive.
        stem_key = out.stem.casefold()
        if any(
            p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem.casefold() == stem_key
            for p in out_dir.iterdir()
        ):
            raise FileExistsError(
                f"An image with label basename {out.stem!r} already exists in that folder - "
                "not overwriting or sharing its label file."
            )

        # Publish a fully-written unique temp through an exclusive placeholder and
        # atomic replacement. This preserves no-clobber semantics on filesystems
        # that do not support hard links (removable/network/cloud-backed storage).
        fd, tmp_name = tempfile.mkstemp(
            prefix=".librelabel-upload-", suffix=".tmp", dir=str(out_dir)
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.chmod(tmp, 0o644)
            try:
                _publish_no_clobber(tmp, out)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"{safe} already exists in that folder - not overwriting it."
                ) from exc
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        return str(out)


def create_uploaded_project(dst: str, *, name: Optional[str] = None,
                            description: str = "", color: str = "",
                            classes: Optional[List[str]] = None,
                            colors: Optional[List[str]] = None,
                            task: Optional[str] = None,
                            make_val: bool = False, val_frac: float = 0.2) -> str:
    """Create an uploaded project while excluding concurrent upload requests."""
    with _UPLOAD_LOCK, _interprocess_upload_lock(Path(dst)):
        return _create_uploaded_project_locked(
            dst,
            name=name,
            description=description,
            color=color,
            classes=classes,
            colors=colors,
            task=task,
            make_val=make_val,
            val_frac=val_frac,
        )


def _create_uploaded_project_locked(dst: str, *, name: Optional[str] = None,
                                    description: str = "", color: str = "",
                                    classes: Optional[List[str]] = None,
                                    colors: Optional[List[str]] = None,
                                    task: Optional[str] = None,
                                    make_val: bool = False, val_frac: float = 0.2) -> str:
    """Turn just-uploaded images (``<dst>/images/train``) into a real LibreYOLO
    dataset: an optional held-out ``val`` split, a ``data.yaml`` the trainer reads
    directly, and a ``librelabel.json`` sidecar (name + per-class colors).
    Returns the path to the written ``data.yaml``."""
    base = Path(dst)
    if folder_yaml(str(base)):
        raise FileExistsError(
            "That folder already has a dataset config (data.yaml). Pick an empty or new folder.")
    train_dir = base / "images" / "train"
    try:
        imgs = sorted(str(i) for i in get_img_files(train_dir)) if train_dir.is_dir() else []
    except (FileNotFoundError, ValueError):
        imgs = []
    if not imgs:
        raise FileNotFoundError("No uploaded images found - add some images first.")
    _assert_unique_image_stems(imgs, "Uploaded project")
    cls = [str(c).strip() for c in (classes or []) if str(c).strip()]
    cols = list(colors or [])

    has_val = False
    moved: List[Tuple[Path, Path]] = []
    remove_val_dir = False
    val_dir = base / "images" / "val"
    if make_val and len(imgs) >= 4:
        if val_dir.is_dir() and any(val_dir.iterdir()):
            raise ValueError(
                "Validation directory is not empty; use an empty upload folder so "
                "the wizard cannot merge unrelated images or label stems."
            )
        try:
            fraction = float(val_frac)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Validation fraction must be a finite number in [0, 1].") from exc
        if not 0.0 <= fraction <= 1.0 or not math.isfinite(fraction):
            raise ValueError("Validation fraction must be a finite number in [0, 1].")
        k = max(1, min(len(imgs) - 1, int(round(len(imgs) * fraction))))
        selected = [Path(imgs[i]) for i in random.Random(1234).sample(range(len(imgs)), k)]
        moved, remove_val_dir = _move_validation_images(selected, val_dir)
        has_val = True

    try:
        cfg = {"path": base.resolve().as_posix(), "train": "images/train"}
        if has_val:
            cfg["val"] = "images/val"
        cfg["names"] = cls
        cfg["nc"] = len(cls)
        cfg["task"] = str(task or "detect").strip().lower()
        text = (
            "# LibreLabel project -- created with the New Project wizard.\n"
            "# Labels are written next to the images, where `libreyolo train` reads them.\n\n"
            + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        )
        _atomic_write_text(base / "data.yaml", text)
    except BaseException:
        _rollback_validation_images(
            moved, val_dir, remove_val_dir=remove_val_dir
        )
        raise

    sidecar = {
        "name": name or base.name,
        "description": description or "",
        "color": color or "",
        "class_colors": {cls[i]: cols[i] for i in range(min(len(cls), len(cols))) if cols[i]},
    }
    try:
        _write_json_atomic(base / "librelabel.json", sidecar)
    except OSError:
        pass
    return str(base / "data.yaml")


def create_linked_project(src: str, *, name: Optional[str] = None,
                          classes: Optional[List[str]] = None,
                          colors: Optional[List[str]] = None,
                          task: Optional[str] = None,
                          projects_dir: Optional[str] = None) -> str:
    """Create a LINKED project: label a folder of images without writing anything
    into it -- not even ``data.yaml``. The project (config, labels, sidecar) lives
    in ``~/.librelabel/projects/<slug>``; the images are referenced through an
    absolute-path manifest and never copied, moved, or annotated in place.

    The tradeoff (documented in the yaml comment): because the labels are not
    next to the images, training needs a copy Export first. Returns the path to
    the managed ``data.yaml``."""
    srcp = Path(src)
    if not srcp.is_dir():
        raise FileNotFoundError(f"Not a folder: {src}")
    imgs = sorted(str(Path(i).resolve()) for i in get_img_files(srcp))
    if not imgs:
        raise FileNotFoundError(f"No images found in {src}")

    base_dir = Path(projects_dir) if projects_dir else Path.home() / ".librelabel" / "projects"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", srcp.name).strip("-") or "project"
    proj = base_dir / f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}"
    n = 1
    while proj.exists():
        n += 1
        proj = base_dir / f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}-{n}"
    proj.mkdir(parents=True)

    manifest = proj / "images.txt"
    _atomic_write_text(manifest, "\n".join(Path(i).as_posix() for i in imgs) + "\n")

    cls = [str(c).strip() for c in (classes or []) if str(c).strip()]
    cfg = {"path": proj.resolve().as_posix(), "train": "images.txt",
           "names": cls, "nc": len(cls)}
    cfg["task"] = str(task or "detect").strip().lower()
    text = (
        "# LibreLabel LINKED project -- the images stay in place, untouched:\n"
        f"#   {srcp.resolve().as_posix()}\n"
        "# Labels live here (labels/), NOT next to the images, so training this\n"
        "# yaml directly finds no labels: use LibreLabel's Export to produce a\n"
        "# self-contained training copy.\n\n"
        + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    )
    _atomic_write_text(proj / "data.yaml", text)

    cols = list(colors or [])
    sidecar = {
        "name": name or srcp.name,
        "linked": True,
        "source": str(srcp.resolve()),
        "class_colors": {cls[i]: cols[i] for i in range(min(len(cls), len(cols))) if cols[i]},
    }
    _write_json_atomic(proj / "librelabel.json", sidecar)
    return str(proj / "data.yaml")


class DatasetSession:
    """An open dataset: ordered images across train/val/test + label R/W."""

    def __init__(self, data: str):
        resolved = _resolve_data_arg(str(data))
        cfg = load_data_config(resolved, autodownload=False)
        self.yaml_file = cfg.get("yaml_file", resolved)
        self.root = cfg.get("path") or cfg.get("root") or ""
        self.names = _names_to_list(cfg.get("names"))
        nc = cfg.get("nc")
        self.nc = int(nc) if nc else len(self.names)
        # Optional wizard sidecar: project display name + per-class colors.
        self._sidecar = load_sidecar(str(self.root)) or load_sidecar(
            str(Path(self.yaml_file).parent))
        # Linked project: images stay in their source folder; every label lives
        # under the managed project dir instead of being derived from the image
        # path (the hash suffix keeps same-named images from sharing a file).
        self.linked = bool(isinstance(self._sidecar, dict) and self._sidecar.get("linked"))
        _linked_lab = Path(self.yaml_file).parent / "labels"

        self._items: List[Tuple[Path, Path, str]] = []
        seen: dict = {}                # native label path -> native image path
        portable_seen: dict = {}       # portable label path -> native image path
        self._path_splits: dict = {}   # normalized label path -> {splits it appears in}
        label_clash: Optional[Tuple[str, str]] = None   # two DIFFERENT images, one label file
        for split in ("train", "val", "test"):
            imgs = cfg.get(f"{split}_img_files") or []
            labels = cfg.get(f"{split}_label_files") or img2label_paths(
                [Path(i) for i in imgs]
            )
            for ip, lp in zip(imgs, labels, strict=True):
                if self.linked:
                    h = hashlib.sha1(
                        _portable_path_identity(ip).encode("utf-8")
                    ).hexdigest()[:8]
                    lp = _linked_lab / split / f"{Path(ip).stem}-{h}.txt"
                # A yaml may reuse a folder across splits; expose each label file
                # once so a single image can't be saved twice under two ids -- but
                # remember every split it was in, for exact-overlap leakage detection.
                key = _path_identity(lp)
                ikey = _path_identity(ip)
                portable_key = _portable_path_identity(lp)
                self._path_splits.setdefault(key, set()).add(split)
                if key in seen:
                    # Same label file again. Same image -> split overlap (leakage,
                    # handled by insights). DIFFERENT image (a.jpg + a.png) -> both
                    # would round-trip through one .txt, silently clobbering each
                    # other; remember it so the session goes read-only below.
                    if label_clash is None and seen[key] != ikey:
                        label_clash = (Path(seen[key]).name, Path(ip).name)
                    continue
                portable_image = portable_seen.get(portable_key)
                if (
                    label_clash is None
                    and portable_image is not None
                    and portable_image != ikey
                ):
                    label_clash = (Path(portable_image).name, Path(ip).name)
                seen[key] = ikey
                portable_seen.setdefault(portable_key, ikey)
                self._items.append((Path(ip), Path(lp), split))

        # Raw split sources (resolved paths/lists) so the duplicate fixer can
        # refuse .txt-manifest splits, where deleting a file leaves a dangling row.
        self._split_sources = {
            s: cfg.get(s) for s in ("train", "val", "test") if cfg.get(s)
        }
        self.writable, self.reason = self._check_writable()
        self._label_clash = label_clash is not None
        if self.writable and label_clash:
            self.writable = False
            self.reason = (
                "Two different images resolve to the same label file "
                f"({label_clash[0]} and {label_clash[1]} share a name): saving one "
                "would overwrite the other's labels. Rename one and reopen."
            )
        # Resolve an omitted task only when on-disk geometry makes it unambiguous.
        # The common schema does not require ``task``; non-quad polygons can only be
        # segmentation, while a four-corner row remains segment-vs-OBB ambiguous.
        declared_task = str(cfg.get("task") or "").strip().lower()
        has_polygon = False
        has_nonquad_polygon = False
        for _ip, label_path, _split in self._items:
            try:
                annotations = parse_annotations(label_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, UnicodeError):
                continue
            for annotation in annotations:
                if annotation.get("type") != "poly":
                    continue
                has_polygon = True
                if len(annotation.get("points") or []) != 8:
                    has_nonquad_polygon = True
        inferred_task = "segment" if not declared_task and has_nonquad_polygon else ""
        task = declared_task or inferred_task
        self._task_declared_or_inferred = bool(task)
        self._task_ambiguous = bool(not task and has_polygon)
        if self.writable and self._task_ambiguous:
            self.writable = False
            self.reason = (
                "Four-corner labels are ambiguous without task: segment or task: "
                "obb. Declare the task before editing so box and OBB geometry cannot "
                "be mixed in one dataset."
            )

        # Pose (kpt_shape), semantic-seg (masks_dir) and depth datasets store dense
        # labels LibreLabel can't edit; writing YOLO boxes would pollute them. The
        # `task:` key alone is enough to know -- a depth yaml may omit depths_dir
        # (the loader defaults it), a classify yaml uses no .txt labels at all, and
        # an OBB yaml's 9-field rows are oriented rectangles we'd corrupt if saved as
        # arbitrary polygons -- so treat those tasks as view-only on the task key too.
        self._task = task   # used to disambiguate 4-point (OBB-vs-polygon) rows on read/write
        # obb is editable: its 9-field rows are 4-corner quads that round-trip
        # byte-identically as 4-vertex polygons (see labelio), so LibreLabel can
        # author oriented boxes without corrupting them.
        native_annotations = any(
            cfg.get(f"{split}_annotation_file") for split in ("train", "val", "test")
        )
        self._native_annotations = bool(native_annotations)
        unsupported_task = bool(task and task not in ("detect", "segment", "obb"))
        task_specific_markers = any(
            key in cfg
            for key in (
                "panoptic_dir",
                "label_mapping",
                "depth_scale",
                "depth_mask_suffix",
                "depth_stem_suffix",
                "input_dir",
                "target_dir",
                "target_stem_suffix",
                "target_stem_suffixes",
                "val_mattes",
                "train_mattes",
                "mattes_dir",
                "flip_idx",
            )
        ) or ("images" in cfg and "labels" in cfg)
        root_path = Path(self.root) if self.root else Path(self.yaml_file).parent
        obvious_paired_layout = (
            ((root_path / "inputs").is_dir() and (root_path / "targets").is_dir())
            or (
                (root_path / "images").is_dir()
                and any(
                    (root_path / directory).is_dir()
                    for directory in (
                        "depths",
                        "mattes",
                        "matte",
                        "gt",
                        "masks",
                        "mask",
                        "alpha",
                    )
                )
            )
            or (
                (root_path / "labels").is_dir()
                and any((root_path / "labels").glob("*.jsonl"))
            )
        )
        dense = (cfg.get("kpt_shape") or cfg.get("masks_dir")
                 or cfg.get("depths_dir") or cfg.get("depth")
                 or unsupported_task or task_specific_markers
                 or obvious_paired_layout)
        # The generic exporter only knows image + YOLO box/polygon pairs. Keep a
        # separate flag from ``writable``: an ambiguous but ordinary detection
        # layout can still be copied safely, while dense/task-specific datasets
        # must never be exported with their masks/keypoints/targets omitted.
        self._lossy_export = bool(dense or native_annotations)
        if self.writable and native_annotations:
            self.writable = False
            self.reason = (
                "Native COCO-JSON annotations are view-only in LibreLabel; this "
                "session does not load or rewrite the JSON and must not create "
                "parallel empty YOLO labels."
            )
        elif self.writable and dense:
            self.writable = False
            self.reason = (
                "Keypoint / mask / depth / restore / task-specific dataset: "
                "view-only in LibreLabel - it edits detection boxes, segmentation "
                "polygons, and OBB corners only; saving would create parallel or "
                "incomplete labels."
            )
        self._deleted: set = set()  # ids of duplicates removed this session (tombstones)

    # -- safety ------------------------------------------------------------
    def _check_writable(self) -> Tuple[bool, str]:
        """Guard against the greedy ``images``->``labels`` substring swap.

        ``img2label_paths`` replaces *every* ``images`` path segment, so a root
        that itself contains ``images`` (e.g. ``my/images/proj/images/train``)
        derives a wrong label path and would silently corrupt the dataset.
        Detect the ambiguity up front and make the session read-only.
        """
        if self.linked:
            return True, ""   # labels live in the managed dir; no derivation traps apply
        root = None
        if self.root:
            try:
                root = Path(self.root).resolve()
            except Exception:  # noqa: BLE001
                root = None
        for ip, lp, _ in self._items:
            # img2label_paths rewrites every "<sep>images" prefix, so a component that
            # *starts with* "images" (e.g. "images_2026" -> "labels_2026") mis-derives
            # the label path. A component that merely *contains* "images" but doesn't
            # start with it (e.g. "my_images") is NOT rewritten -> still writable.
            risky = [p for p in ip.parts if p.startswith("images")]
            if len(risky) > 1 or (len(risky) == 1 and risky[0] != "images"):
                return (
                    False,
                    "Ambiguous dataset layout: a path segment contains 'images' in a "
                    "way that makes the label path ambiguous, so saving could write "
                    "outside the dataset. Rename the ancestor (e.g. to 'imgs/') and reopen.",
                )
            if lp == ip:
                return False, f"Could not derive a label path for {ip}."
            # A single 'images' segment is fine *inside* the dataset (the conventional
            # images/->labels/ sibling layout), but if it sits ABOVE the root -- e.g. a
            # flat folder /home/me/images/cats opened as `train: .` -- the rewrite still
            # fires and sends labels OUTSIDE the dataset. Require the label path to stay
            # within the root for any image that is itself under the root.
            if root is not None:
                try:
                    ip.resolve().relative_to(root)
                except ValueError:
                    continue   # image not under the dataset root (unusual) -> don't second-guess
                try:
                    lp.resolve().relative_to(root)
                except ValueError:
                    return (
                        False,
                        "Saving would write labels outside the dataset folder: an ancestor "
                        "path segment named 'images' gets rewritten to 'labels'. Move the "
                        "images into an 'images/' subfolder (or rename the ancestor) and reopen.",
                    )
        return True, ""

    # -- queries -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def meta(self) -> dict:
        return {
            "root": str(self.root),
            "yaml": str(self.yaml_file),
            "names": self.names,
            "nc": self.nc,
            "count": len(self._items),
            "writable": self.writable,
            "reason": self.reason,
            "task": self._task or "detect",
            "linked": self.linked,
            "source": (self._sidecar.get("source") or "") if self.linked else "",
            "has_val": any(s in ("val", "test") for _, _, s in self._items),
            "name": (self._sidecar.get("name") or "") if isinstance(self._sidecar, dict) else "",
            "description": (self._sidecar.get("description") or "") if isinstance(self._sidecar, dict) else "",
            "instructions": (self._sidecar.get("instructions") or "") if isinstance(self._sidecar, dict) else "",
            "colors": [
                (self._sidecar.get("class_colors", {}) or {}).get(n)
                for n in self.names
            ] if isinstance(self._sidecar, dict) else [],
        }

    def _status(self, lp: Path) -> str:
        if not lp.exists():
            return "unlabeled"
        try:
            return "labeled" if lp.stat().st_size > 0 else "empty"
        except OSError:
            return "unlabeled"

    def list_images(self) -> List[dict]:
        rows = []
        for i, (ip, lp, split) in enumerate(self._items):
            status = "deleted" if i in self._deleted else self._status(lp)
            rows.append({"id": i, "name": ip.name, "split": split, "status": status})
        return rows

    def stats(self) -> dict:
        """Aggregate the on-disk (accepted) labels into a dataset-health summary."""
        from collections import Counter

        counts: Counter = Counter()
        labeled = empty = total_boxes = 0
        for i, (_ip, lp, _split) in enumerate(self._items):
            if i in self._deleted or not lp.exists():
                continue
            try:
                text = lp.read_text(encoding="utf-8")
            except OSError:
                continue
            anns = parse_annotations(text)
            if anns:
                labeled += 1
                total_boxes += len(anns)
                for a in anns:
                    counts[a["cls"]] += 1
            else:
                empty += 1
        n = len(self.names)
        top = [
            [self.names[c] if 0 <= c < n else str(c), cnt]
            for c, cnt in counts.most_common(12)
        ]
        live = len(self._items) - len(self._deleted)
        return {
            "total": live,
            "labeled": labeled,
            "empty": empty,
            "unlabeled": live - labeled - empty,
            "boxes": total_boxes,
            "classes": top,
        }

    def insights(self) -> dict:
        """Dataset intelligence: dimension stats + perceptual-hash duplicates.

        Decodes each image once (downscaled) to compute a dHash and read its
        size. Cached for the session. Surfaces the data-quality issues that
        matter most for YOLO training: duplicate images and, especially,
        train/val *leakage* (the same image in two splits).
        """
        if getattr(self, "_insights_cache", None) is not None:
            return self._insights_cache

        from collections import Counter

        from PIL import Image

        dims: list = []          # (w, h, idx, split)
        hashes: dict = {}        # dhash -> [idx, ...]
        failed = 0
        for i, (ip, _lp, split) in enumerate(self._items):
            if i in self._deleted:
                continue
            try:
                with Image.open(ip) as im:
                    w, h = im.size
                    g = list(im.convert("L").resize((9, 8)).getdata())
            except Exception:  # noqa: BLE001
                failed += 1
                continue
            dims.append((w, h, i, split))
            bits = 0
            for row in range(8):
                base = row * 9
                for col in range(8):
                    bits = (bits << 1) | (1 if g[base + col] > g[base + col + 1] else 0)
            hashes.setdefault(bits, []).append(i)

        def _stat(vals):
            if not vals:
                return {"min": 0, "max": 0, "mean": 0, "median": 0}
            s = sorted(vals)
            return {
                "min": s[0], "max": s[-1],
                "mean": round(sum(s) / len(s)),
                "median": s[len(s) // 2],
            }

        ws = [d[0] for d in dims]
        hs = [d[1] for d in dims]
        mp = [round(w * h / 1e6, 2) for w, h, _i, _s in dims]
        res_top = Counter((w, h) for w, h, _i, _s in dims).most_common(6)
        name = lambda i: self._items[i][0].name  # noqa: E731
        split_of = lambda i: self._items[i][2]    # noqa: E731

        dup_groups = []
        leak_groups = []
        for ids in hashes.values():
            if len(ids) < 2:
                continue
            grp = {"ids": ids, "names": [name(i) for i in ids],
                   "splits": sorted({split_of(i) for i in ids})}
            dup_groups.append(grp)
            if len(grp["splits"]) > 1:
                leak_groups.append(grp)
        dup_groups.sort(key=lambda g: -len(g["ids"]))
        # Exact same label-path listed in >1 split (deduped out of _items, so the
        # dHash pass above can't see it) -- still real train/val leakage.
        kidx = {_path_identity(self._items[i][1]): i
                for i in range(len(self._items)) if i not in self._deleted}
        for key, splits in self._path_splits.items():
            if len(splits) > 1 and key in kidx:
                i = kidx[key]
                leak_groups.append({"ids": [i], "names": [name(i)],
                                    "splits": sorted(splits), "exact": True})

        self._insights_cache = {
            "count": len(self._items) - len(self._deleted),
            "measured": len(dims),
            "failed": failed,
            "width": _stat(ws),
            "height": _stat(hs),
            "megapixels": _stat(mp) if mp else {"min": 0, "max": 0, "mean": 0, "median": 0},
            "top_resolutions": [[w, h, c] for (w, h), c in res_top],
            "duplicate_groups": dup_groups[:50],
            "duplicate_image_count": sum(len(g["ids"]) for g in dup_groups),
            "leakage_groups": leak_groups[:50],
        }
        return self._insights_cache

    def quality(self, imgsz: int = 640) -> dict:
        """Geometry-lint accepted labels: tiny / sliver / full-frame boxes.

        Surfaces annotations a detector physically can't learn from at ``imgsz``
        (a few-pixel box), plus absurd aspect ratios and whole-frame boxes that
        are almost always slips. Reports only -- never edits a label.
        """
        from .quality import lint_annotations

        flagged: List[dict] = []
        counts = {"tiny": 0, "sliver": 0, "fullframe": 0}
        total_issues = 0
        for i, (ip, lp, _split) in enumerate(self._items):
            if i in self._deleted or not lp.exists():
                continue
            anns, editable = self.read_label(i)
            if not editable:
                continue   # view-only/dense (e.g. OBB) labels: don't lint a partial polygon view
            if not anns:
                continue
            issues = lint_annotations(anns, imgsz=imgsz)
            if issues:
                total_issues += len(issues)
                for it in issues:
                    counts[it["type"]] = counts.get(it["type"], 0) + 1
                flagged.append({"id": i, "name": ip.name, "count": len(issues),
                                "issues": issues})
        flagged.sort(key=lambda d: -d["count"])
        return {"imgsz": imgsz, "issues": total_issues, "counts": counts,
                "flagged": flagged[:100]}

    def resolve_duplicates(self, ids: List[int], *, purge: bool = False) -> dict:
        """Collapse a duplicate/leakage group to one survivor (reversible default).

        Keeps exactly one copy -- preferring the ``train`` copy so train/val
        leakage is eliminated -- and MOVES the rest (image + its label, together)
        into ``<root>/.librelabel_quarantine/`` so a probabilistic perceptual-hash
        match is never destructive. ``purge=True`` hard-deletes instead. Removed
        ids are tombstoned so open image ids stay stable for the UI. No-op +
        raises when the session is read-only; refuses ``.txt``-manifest splits
        (deleting a file there would leave a dangling manifest line).
        """
        import shutil

        if not self.writable:
            raise RuntimeError(self.reason)
        if self.linked:
            raise RuntimeError(
                "This is a linked project: the source images are never moved or "
                "deleted. Fix duplicates in the source folder, or export a copy.")
        valid = [i for i in dict.fromkeys(ids)
                 if 0 <= i < len(self._items) and i not in self._deleted]
        if len(valid) < 2:
            return {"removed": [], "kept": valid[0] if valid else None,
                    "quarantine": None}
        # Prefer a survivor that actually has labels (then train split, then lowest
        # id): keeping an unlabelled train copy while quarantining the only labelled
        # copy would silently turn a labelled image unlabelled after Fix.
        def _labelled(i):
            lp = self._items[i][1]
            try:
                return lp.exists() and lp.stat().st_size > 0
            except OSError:
                return False

        rank = {"train": 0, "val": 1, "test": 2}
        keep = min(valid, key=lambda i: (0 if _labelled(i) else 1,
                                         rank.get(self._items[i][2], 3), i))
        redundant = [i for i in valid if i != keep]
        # A split defined by an explicit file list (a .txt manifest OR an inline YAML
        # list of image files) still references the file we'd move/delete, leaving a
        # dangling entry on the next load -- refuse so the user fixes the list first.
        _listed = (".txt", ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
        for i in redundant:
            src = self._split_sources.get(self._items[i][2])
            srcs = src if isinstance(src, list) else [src]
            if any(str(x).lower().endswith(_listed) for x in srcs):
                raise RuntimeError(
                    "This split is defined by an explicit file list (a .txt manifest or "
                    "an inline YAML image list); update the list before pruning so no "
                    "dangling references are left.")
        qbase = Path(self.root) if self.root else Path(self.yaml_file).parent
        qroot = qbase / ".librelabel_quarantine"
        # A broad/recursive split (e.g. ``train: .``) rglob-scans the whole tree, so a
        # quarantine dir INSIDE it would be rediscovered on the next load/train and the
        # cleanup wouldn't stick. Refuse pruning such a split rather than silently
        # un-quarantining. (Purge has no quarantine dir, so it's unaffected.)
        if not purge:
            try:
                qres = qroot.resolve()
            except OSError:
                qres = qroot
            # Check EVERY split, not just the redundant ids' own: a broad split
            # elsewhere (e.g. `test: .`) would still rglob the quarantine dir back in.
            for src in self._split_sources.values():
                for s in (src if isinstance(src, list) else [src]):
                    if not s:
                        continue
                    try:
                        d = Path(s).resolve()
                    except OSError:
                        continue
                    if d.is_dir() and (qres == d or d in qres.parents):
                        raise RuntimeError(
                            "A split is a recursive directory that would re-scan the "
                            "quarantine folder; prune with purge, or point the split at a "
                            "narrower images/ subdirectory.")
        removed: List[dict] = []
        for i in redundant:
            ip, lp, split = self._items[i]
            try:
                if purge:
                    # Delete the IMAGE first: if that fails (lock/permission) the
                    # OSError below skips the tombstone with the labelled pair fully
                    # intact -- we never delete a label whose image survives (which
                    # would silently turn a labelled image unlabelled). Once the image
                    # is gone, the label cleanup is best-effort (an orphaned label is
                    # ignored by the loader, which iterates images).
                    if ip.exists():
                        ip.unlink()
                    try:
                        if lp.exists():
                            lp.unlink()
                    except OSError:
                        pass
                else:
                    dst_img = qroot / "images" / split / f"{i}_{ip.name}"   # id prefix: never collide
                    dst_lbl = qroot / "labels" / split / f"{i}_{lp.name}"
                    dst_img.parent.mkdir(parents=True, exist_ok=True)
                    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                    moved_img = False
                    if ip.exists():
                        shutil.move(str(ip), str(dst_img))
                        moved_img = True
                    try:
                        if lp.exists():
                            shutil.move(str(lp), str(dst_lbl))
                    except OSError:
                        if moved_img:  # roll back so the item stays consistent
                            shutil.move(str(dst_img), str(ip))
                        raise
            except OSError:
                continue
            self._deleted.add(i)
            removed.append({"id": i, "name": ip.name, "split": split})
        self._insights_cache = None  # dimensions / dup groups changed
        return {"removed": removed, "kept": keep,
                "kept_name": self._items[keep][0].name,
                "mode": "purge" if purge else "quarantine",
                "quarantine": None if purge else str(qroot)}

    def _check_index(self, idx: int) -> None:
        if not (0 <= idx < len(self._items)):
            raise IndexError(f"image id out of range: {idx}")

    def image_path(self, idx: int) -> Path:
        self._check_index(idx)
        return self._items[idx][0]

    def has_label_file(self, idx: int) -> bool:
        """Whether a label ``.txt`` exists on disk (an empty file = reviewed background)."""
        self._check_index(idx)
        return self._items[idx][1].exists()

    def read_label(self, idx: int) -> Tuple[List[dict], bool]:
        """Return ``(annotations, editable)`` - mixed box/polygon annotations.

        ``editable`` is ``False`` for files holding keypoint/pose or malformed rows
        (which we don't parse), so a save never silently drops those fields.
        """
        self._check_index(idx)
        lp = self._items[idx][1]
        with _LABEL_LOCK, _interprocess_path_lock(lp):
            return self._read_label_unlocked(idx)

    def _read_label_unlocked(self, idx: int) -> Tuple[List[dict], bool]:
        """Read one label while the caller owns its canonical filesystem lock."""
        lp = self._items[idx][1]
        if not lp.exists():
            return [], self.writable   # a read-only session stays inert even for unlabeled images
        text = lp.read_text(encoding="utf-8")
        annotations = parse_annotations(text)
        has_boxes = any(a.get("type") == "box" for a in annotations)
        has_polygons = any(a.get("type") == "poly" for a in annotations)
        # A file is editable only if the whole dataset is writable (a dense/pose/OBB
        # dataset's box-shaped rows are a partial view we must never round-trip) AND
        # the file has no rows a save would silently alter: keypoint/malformed rows,
        # an out-of-[0,nc) class, or out-of-[0,1] coordinates the writer rejects.
        editable = self.writable and not (
            has_unsupported_rows(text)
            or has_out_of_range_rows(text, self.nc)
            or has_out_of_bounds_coords(text)
            or has_degenerate_polygon(text)
            or has_zero_area_box(text)
            or (self._task == "detect" and has_polygons)
            or (self._task == "obb" and has_boxes)
            # 4-point rows are OBB-or-polygon-ambiguous only when the dataset declares
            # no task; segment (free polygons) and obb (oriented boxes) both edit them
            or (self._task not in ("segment", "obb") and has_obb_shaped_rows(text)))
        return annotations, editable

    def read_label_with_rev(self, idx: int) -> tuple[List[dict], bool, int]:
        """Return annotations, editability, and one matching content revision."""
        self._check_index(idx)
        lp = self._items[idx][1]
        with _LABEL_LOCK, _interprocess_path_lock(lp):
            annotations, editable = self._read_label_unlocked(idx)
            return annotations, editable, self._label_rev_unlocked(idx)

    def label_rev(self, idx: int) -> int:
        """A content revision token for optimistic label-write concurrency.

        Filesystem timestamps are too coarse on some removable/network filesystems
        for a compare-and-swap contract.  A fixed digest changes whenever the label
        bytes change; zero remains the sentinel for an absent file.
        """
        self._check_index(idx)
        lp = self._items[idx][1]
        with _LABEL_LOCK, _interprocess_path_lock(lp):
            return self._label_rev_unlocked(idx)

    def _label_rev_unlocked(self, idx: int) -> int:
        """Return a content revision while the caller owns the label lock."""
        lp = self._items[idx][1]
        try:
            data = lp.read_bytes()
        except FileNotFoundError:
            return 0
        digest = hashlib.blake2b(data, digest_size=8, person=b"LibreLbl").digest()
        return int.from_bytes(digest, "big") + 1

    # -- mutation ----------------------------------------------------------
    def write_label(self, idx: int, annotations: List[dict], expected_rev: Optional[int] = None) -> int:
        """Write annotations (boxes and/or polygons) atomically. Returns count.

        ``expected_rev`` (a :meth:`label_rev` token) enables optimistic concurrency:
        if the file was rewritten since the caller loaded it (another teammate saved),
        the write is refused so collaborative edits don't clobber each other.
        """
        self._check_index(idx)
        lp = self._items[idx][1]
        with _LABEL_LOCK, _interprocess_path_lock(lp):
            return self._write_label_unlocked(idx, annotations, expected_rev)

    def write_label_with_rev(
        self,
        idx: int,
        annotations: List[dict],
        expected_rev: Optional[int] = None,
    ) -> tuple[int, int]:
        """Atomically compare, write, and return the resulting content revision."""
        self._check_index(idx)
        lp = self._items[idx][1]
        with _LABEL_LOCK, _interprocess_path_lock(lp):
            count = self._write_label_unlocked(idx, annotations, expected_rev)
            return count, self._label_rev_unlocked(idx)

    def _write_label_unlocked(
        self,
        idx: int,
        annotations: List[dict],
        expected_rev: Optional[int],
    ) -> int:
        """Validate and write while the caller owns the canonical label lock."""
        if idx in self._deleted:
            # Tombstoned by duplicate/leakage cleanup: a stale client must not be
            # able to recreate a label file for a removed image.
            raise RuntimeError("This image was removed during duplicate cleanup; it is no longer editable.")
        if not self.writable:
            raise RuntimeError(self.reason)
        lp = self._items[idx][1]
        if expected_rev is not None and self._label_rev_unlocked(idx) != expected_rev:
            raise RuntimeError("This image was changed by someone else since you opened it; "
                               "reload it before saving so their labels aren't overwritten.")
        if lp.exists():
            existing = lp.read_text(encoding="utf-8")
            if has_unsupported_rows(existing):
                raise RuntimeError("This label file has keypoint/unsupported rows; it is read-only.")
            if has_out_of_range_rows(existing, self.nc):
                raise RuntimeError("This label file has class ids outside the dataset's nc; it is read-only.")
            if has_out_of_bounds_coords(existing):
                raise RuntimeError("This label file has coordinates outside [0, 1]; it is read-only.")
            if has_degenerate_polygon(existing):
                raise RuntimeError("This label file has a zero-area (collinear/collapsed) polygon; it is read-only.")
            if has_zero_area_box(existing):
                raise RuntimeError("This label file has a zero-width/height box; it is read-only.")
            parsed_existing = parse_annotations(existing)
            if self._task == "detect" and any(
                annotation.get("type") == "poly"
                for annotation in parsed_existing
            ):
                raise RuntimeError(
                    "This detection label file contains polygon rows; it is read-only."
                )
            if self._task == "obb" and any(
                annotation.get("type") == "box"
                for annotation in parsed_existing
            ):
                raise RuntimeError(
                    "This OBB label file contains axis-aligned box rows; it is read-only."
                )
            if self._task not in ("segment", "obb") and has_obb_shaped_rows(existing):
                raise RuntimeError("This label file has 4-point (OBB/quad) rows; without task: segment "
                                   "or task: obb they're ambiguous and kept read-only.")
        clean = sanitize_annotations(
            annotations, self.nc, task=self._task or "detect"
        )
        lp.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(lp, format_annotations(clean))
        return len(clean)
