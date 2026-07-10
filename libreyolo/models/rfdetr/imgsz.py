"""RF-DETR image-size validation helpers."""

from __future__ import annotations

from typing import Any


def resolve_patch_window(
    model: Any,
    *,
    default_patch_size: int = 16,
    default_num_windows: int = 2,
) -> tuple[int, int]:
    """Return ``(patch_size, num_windows)`` from common wrapper shapes.

    Real RF-DETR modules always expose ``patch_size``/``num_windows``; the
    defaults only cover wrappers that hide them and mirror every detection
    size config (patch 16 x 2 windows, block 32).
    """

    candidates = []
    current = model
    for _ in range(4):
        if current is None or current in candidates:
            break
        candidates.append(current)
        current = getattr(current, "module", None) or getattr(current, "model", None)

    patch_size = default_patch_size
    num_windows = default_num_windows
    for candidate in candidates:
        if hasattr(candidate, "patch_size") or hasattr(candidate, "num_windows"):
            patch_size = int(getattr(candidate, "patch_size", patch_size))
            num_windows = int(getattr(candidate, "num_windows", num_windows))
            break
    return patch_size, num_windows


def _suggest_sizes(value: int, block_size: int) -> str:
    lo = (value // block_size) * block_size
    hi = lo + block_size
    choices = [str(size) for size in (lo, hi) if size > 0]
    if not choices:
        choices = [str(block_size)]
    if len(choices) == 1:
        return choices[0]
    return f"{choices[0]} or {choices[1]}"


def _validate_one(
    value: int,
    *,
    name: str,
    patch_size: int,
    num_windows: int,
) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")

    block_size = int(patch_size) * int(num_windows)
    if value % block_size != 0:
        raise ValueError(
            f"{name}={value} is not divisible by {block_size} "
            f"(patch_size={patch_size} x num_windows={num_windows}). "
            f"Use {_suggest_sizes(value, block_size)}."
        )
    return value


def validate_imgsz(
    imgsz: int | tuple[int, int] | list[int],
    *,
    patch_size: int,
    num_windows: int,
    name: str = "imgsz",
) -> int | tuple[int, int]:
    """Validate RF-DETR spatial divisibility and return normalized ints."""

    if isinstance(imgsz, (tuple, list)):
        if len(imgsz) != 2:
            raise ValueError(f"{name} must be (height, width), got {imgsz}.")
        h = _validate_one(
            imgsz[0],
            name=f"{name} height",
            patch_size=patch_size,
            num_windows=num_windows,
        )
        w = _validate_one(
            imgsz[1],
            name=f"{name} width",
            patch_size=patch_size,
            num_windows=num_windows,
        )
        return h, w
    return _validate_one(
        imgsz,
        name=name,
        patch_size=patch_size,
        num_windows=num_windows,
    )
