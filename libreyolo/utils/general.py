"""Shared general utility functions."""

import atexit
import hashlib
import logging
import math
import os
import secrets
import stat
from dataclasses import dataclass
from itertools import count
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from typing import Dict, List, Tuple, Union
from urllib.parse import urlparse

import torch

from ..postprocess.common import postprocess_detections  # noqa: F401  (moved; re-exported for backward compatibility)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================


def increment_path(
    path: Union[str, Path], exist_ok: bool = False, sep: str = "", mkdir: bool = False
) -> Path:
    """
    Return an incremented path if it already exists.

    E.g. runs/detect/predict -> runs/detect/predict2 -> runs/detect/predict3, etc.

    Args:
        path: Base path to increment.
        exist_ok: If True, return the path as-is even if it exists.
        sep: Separator between base name and number (default: "").
        mkdir: Create the directory if True.

    Returns:
        Incremented Path.
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not Path(p).exists():
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


# COCO class names (80 classes)
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# =============================================================================
# Box Utilities
# =============================================================================


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).

    Args:
        boxes: Boxes in cxcywh format (..., 4)

    Returns:
        Boxes in xyxy format (..., 4)
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


# =============================================================================
# Path Utilities
# =============================================================================

_save_dir_cache: Dict[str, Path] = {}


@dataclass(frozen=True)
class _SavePathReservation:
    """Ownership record for one filesystem-backed save-path claim."""

    lock_path: Path
    token: str
    owner_pid: int
    identity: tuple[int, int]


_save_path_reservations: dict[str, _SavePathReservation] = {}
_save_path_lock = RLock()
_SAVE_PATH_OWNER_FILE = "owner"
_SAVE_PATH_LOCK_PREFIX = ".libreyolo-save-"
_SAVE_PATH_LOCK_SUFFIX = ".reserve"


def get_safe_stem(path: Union[str, Path]) -> str:
    path_str = str(path)
    if path_str.startswith(("http://", "https://", "s3://", "gs://")):
        parsed = urlparse(path_str)
        filename = Path(parsed.path).name
        return Path(filename).stem if filename else "inference"
    return Path(path_str).stem


def resolve_save_path(
    output_path: Union[str, Path, None],
    image_path: Union[str, Path, None],
    prefix: str = "",
    ext: str = "jpg",
    default_dir: str = "runs/detect",
    exist_ok: bool = False,
    force_ext: bool = False,
) -> Path:
    """
    Generate a save path handling both directory and file output paths.

    Uses an auto-incrementing directory scheme: runs/detect/predict,
    runs/detect/predict2, etc. The original filename is preserved.
    Within a single process, all images are saved to the same directory.
    Save names are atomically reserved across threads and processes so repeated
    inputs, duplicate basenames, and explicit-file batch outputs cannot
    overwrite one another before the previous write becomes visible.

    Args:
        output_path: User-provided output path (file or directory) or None
        image_path: Source image path for deriving filename
        prefix: Optional prefix for the filename (e.g., "tiled_")
        ext: File extension without dot (default: "jpg")
        default_dir: Default directory if output_path is None
        exist_ok: If True, reuse existing predict/ directory without incrementing
        force_ext: If True, replace an explicit file suffix with ``ext``. This is
            used by formats such as matte cutouts that require PNG output.

    Returns:
        Resolved Path object ready for saving
    """
    ext = str(ext).lower().lstrip(".")
    if not ext:
        raise ValueError("Save extension must not be empty.")

    # Get filename from image path or use default
    if image_path is not None:
        stem = get_safe_stem(image_path)
    else:
        stem = "inference"

    filename = f"{prefix}{stem}.{ext}"

    if output_path is None:
        if default_dir not in _save_dir_cache:
            _save_dir_cache[default_dir] = increment_path(
                Path(default_dir) / "predict", exist_ok=exist_ok, mkdir=True
            )
        candidate = _save_dir_cache[default_dir] / filename
        return _reserve_available_save_path(candidate)

    raw_output_path = str(output_path)
    save_path = Path(output_path)
    directory_hint = (
        save_path.is_dir()
        or raw_output_path.endswith(("/", "\\"))
        or save_path.suffix == ""
    )

    if directory_hint:
        save_path.mkdir(parents=True, exist_ok=True)
        candidate = save_path / filename
    else:
        if force_ext:
            save_path = save_path.with_suffix(f".{ext}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        candidate = save_path

    return _reserve_available_save_path(candidate)


def reserve_save_directory(path: Union[str, Path]) -> Path:
    """Create and return a collision-free directory across processes."""
    path = Path(path)
    for index in count(1):
        candidate = path if index == 1 else Path(f"{path}{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if os.path.lexists(candidate):
                continue
            raise
        return candidate


def _reserve_available_save_path(path: Union[str, Path]) -> Path:
    """Reserve a collision-free file path across threads and processes.

    A same-directory reservation marker is claimed with atomic ``mkdir``.
    Unlike a process-local set, that operation arbitrates independent Python
    processes without creating or truncating the user's final artifact. The
    first collision uses the established ``name2.ext`` convention from
    :func:`increment_path`.
    """
    path = Path(path)
    stem_path = path.with_suffix("")
    suffix = path.suffix

    with _save_path_lock:
        for index in count(1):
            candidate = path if index == 1 else Path(f"{stem_path}{index}{suffix}")
            reservation_key, lock_path = _save_path_reservation_location(candidate)
            if os.path.lexists(candidate):
                continue

            try:
                lock_path.mkdir(mode=0o700)
            except FileExistsError:
                continue

            token = secrets.token_hex(16)
            owner_file = lock_path / _SAVE_PATH_OWNER_FILE
            try:
                owner_file.write_text(token, encoding="ascii")
                lock_stat = os.stat(lock_path, follow_symlinks=False)
            except BaseException:
                _remove_new_reservation_marker(lock_path, owner_file)
                raise

            reservation = _SavePathReservation(
                lock_path=lock_path,
                token=token,
                owner_pid=os.getpid(),
                identity=(lock_stat.st_dev, lock_stat.st_ino),
            )
            _save_path_reservations[reservation_key] = reservation

            # A writer that does not use this allocator may have published the
            # candidate between our existence check and marker acquisition.
            # Release only our marker and keep searching rather than overwrite.
            if os.path.lexists(candidate):
                _release_save_path_reservation_key(reservation_key)
                continue
            return candidate


def _save_path_reservation_location(path: Path) -> tuple[str, Path]:
    """Return the physical-path key and bounded marker path for ``path``."""
    resolved = path.resolve(strict=False)
    reservation_key = os.path.normcase(str(resolved))
    # Conservatively case-fold the marker identity on every platform. This
    # prevents duplicate claims on case-insensitive macOS volumes without
    # making platform-specific filesystem guesses. Case-sensitive filesystems
    # may increment simultaneous case-only variants, which is always safe.
    marker_key = reservation_key.casefold()
    digest = hashlib.sha256(os.fsencode(marker_key)).hexdigest()
    lock_name = f"{_SAVE_PATH_LOCK_PREFIX}{digest}{_SAVE_PATH_LOCK_SUFFIX}"
    return reservation_key, resolved.parent / lock_name


def _remove_new_reservation_marker(lock_path: Path, owner_file: Path) -> None:
    """Best-effort cleanup for a marker this call has just created."""
    try:
        owner_file.unlink(missing_ok=True)
        lock_path.rmdir()
    except OSError:
        # Leaving a reservation marker is safer than deleting an unknown path.
        pass


def _reservation_is_owned(reservation: _SavePathReservation) -> bool:
    """Return whether the current process still owns this exact marker."""
    if reservation.owner_pid != os.getpid():
        return False

    try:
        marker_stat = os.stat(reservation.lock_path, follow_symlinks=False)
        if not stat.S_ISDIR(marker_stat.st_mode):
            return False
        if (marker_stat.st_dev, marker_stat.st_ino) != reservation.identity:
            return False
        owner = reservation.lock_path / _SAVE_PATH_OWNER_FILE
        return secrets.compare_digest(
            owner.read_text(encoding="ascii"), reservation.token
        )
    except (OSError, UnicodeError):
        return False


def _release_save_path_reservation_key(reservation_key: str) -> bool:
    """Release one owned marker without ever unlinking the output artifact."""
    with _save_path_lock:
        reservation = _save_path_reservations.get(reservation_key)
        if reservation is None or not _reservation_is_owned(reservation):
            return False

        # Move the deterministic marker aside first. A new allocator can then
        # claim the deterministic name, while cleanup operates only on our
        # token-specific tombstone and cannot remove the new reservation.
        tombstone = reservation.lock_path.with_name(
            f"{reservation.lock_path.name}.{reservation.token}.cleanup"
        )
        try:
            reservation.lock_path.rename(tombstone)
        except OSError:
            return False

        _save_path_reservations.pop(reservation_key, None)
        try:
            (tombstone / _SAVE_PATH_OWNER_FILE).unlink(missing_ok=True)
            tombstone.rmdir()
        except OSError:
            # A token-specific tombstone no longer blocks allocations. Avoid
            # recursive deletion because unexpected contents are not ours.
            pass
        return True


def release_save_path_reservation(path: Union[str, Path]) -> bool:
    """Release this process's reservation marker for ``path``.

    The final artifact is never deleted. Ownership is checked using the
    creating PID, marker identity, and random token before the marker is moved,
    so a forked child or another process cannot release the owner's claim.
    """
    try:
        reservation_key, _ = _save_path_reservation_location(Path(path))
    except (OSError, RuntimeError):
        return False
    return _release_save_path_reservation_key(reservation_key)


def _cleanup_save_path_reservations() -> None:
    """Release markers still owned by this process during orderly shutdown."""
    with _save_path_lock:
        reservation_keys = tuple(_save_path_reservations)
    for reservation_key in reservation_keys:
        _release_save_path_reservation_key(reservation_key)


def _reset_save_path_state_after_fork() -> None:
    """Prevent a child process from inheriting its parent's local ownership."""
    global _save_path_lock
    _save_path_reservations.clear()
    _save_path_lock = RLock()


atexit.register(_cleanup_save_path_reservations)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_save_path_state_after_fork)


def log_saved_result(result, save_path: Union[str, Path]) -> str:
    """Attach and log the path where an inference result was saved."""
    saved_path = str(save_path)
    release_save_path_reservation(save_path)
    result.saved_path = saved_path
    logger.info("Results saved to %s", saved_path)
    return saved_path


# =============================================================================
# Image Tiling
# =============================================================================


def get_slice_bboxes(
    image_width: int,
    image_height: int,
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate tile coordinates for slicing a large image.

    Args:
        image_width: Width of the original image.
        image_height: Height of the original image.
        slice_size: Size of each square tile (default: 640).
        overlap_ratio: Fractional overlap between tiles (default: 0.2).

    Returns:
        List of (x1, y1, x2, y2) tuples representing tile coordinates.
    """
    for name, value in (
        ("image_width", image_width),
        ("image_height", image_height),
        ("slice_size", slice_size),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}.")
    if isinstance(overlap_ratio, bool) or not isinstance(overlap_ratio, Real):
        raise TypeError(
            "overlap_ratio must be a real number, "
            f"got {type(overlap_ratio).__name__}."
        )
    overlap_ratio = float(overlap_ratio)
    if not math.isfinite(overlap_ratio) or not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(
            "overlap_ratio must be finite and in [0, 1), "
            f"got {overlap_ratio!r}."
        )

    slices = []
    overlap = int(slice_size * overlap_ratio)
    step = slice_size - overlap
    if step <= 0:
        raise ValueError(
            "Tile step must be positive; reduce overlap_ratio or increase slice_size."
        )

    y = 0
    while y < image_height:
        x = 0
        while x < image_width:
            x2 = min(x + slice_size, image_width)
            y2 = min(y + slice_size, image_height)
            # Ensure full tile size when near edges by adjusting start position
            x1 = max(0, x2 - slice_size) if x2 == image_width else x
            y1 = max(0, y2 - slice_size) if y2 == image_height else y
            slices.append((x1, y1, x2, y2))
            x += step
            if x2 == image_width:
                break
        y += step
        if y2 == image_height:
            break
    return slices


# =============================================================================
# Detection Post-processing
# =============================================================================


def make_anchors(
    feats: List[torch.Tensor], strides: List[int], grid_cell_offset: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate anchor points from feature map sizes.

    Args:
        feats: List of feature tensors from different scales
        strides: List of stride values corresponding to each feature map
        grid_cell_offset: Offset for grid cell centers (default: 0.5)

    Returns:
        Tuple of (anchor_points, stride_tensor)
    """
    centers_by_level = []
    stride_by_level = []

    for feature, stride in zip(feats, strides):
        dtype, device = feature.dtype, feature.device
        height, width = feature.shape[-2:]
        y_coords = torch.arange(height, device=device, dtype=dtype).add(
            grid_cell_offset
        )
        x_coords = torch.arange(width, device=device, dtype=dtype).add(
            grid_cell_offset
        )
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        centers = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
        stride_value = torch.as_tensor(stride, device=device, dtype=dtype)

        centers_by_level.append(centers)
        stride_by_level.append(stride_value.expand(centers.shape[0], 1))

    return torch.cat(centers_by_level, dim=0), torch.cat(stride_by_level, dim=0)
