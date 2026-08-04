"""Screen-capture source utilities for LibreYOLO.

A screen source is a string of the form ``"screen"``, optionally followed by a
monitor index and an optional capture box::

    "screen"                    # every monitor, merged
    "screen 1"                  # monitor 1 (the primary display)
    "screen 100 200 512 256"    # box on the merged desktop
    "screen 1 100 200 512 256"  # box on monitor 1

Box coordinates are ``left top width height`` and are relative to the chosen
monitor's top-left corner.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np

# "screen" plus zero or more whitespace-separated integers.
_SCREEN_RE = re.compile(r"^screen(?P<nums>(?:\s+-?\d+)*)\s*$", re.IGNORECASE)

DEFAULT_SCREEN_FPS = 30.0


@dataclass(frozen=True)
class ScreenRegion:
    """A monitor index plus an optional capture box within that monitor."""

    monitor: int = 0
    left: Optional[int] = None
    top: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    @property
    def has_box(self) -> bool:
        return all(
            value is not None
            for value in (self.left, self.top, self.width, self.height)
        )

    def __str__(self) -> str:
        if self.has_box:
            return (
                f"screen {self.monitor} {self.left} {self.top} "
                f"{self.width} {self.height}"
            )
        return f"screen {self.monitor}"


def is_screen_source(source) -> bool:
    """Check whether *source* is a screen-capture source string."""
    if not isinstance(source, (str, Path)):
        return False
    return _SCREEN_RE.match(str(source)) is not None


def parse_screen_source(source: Union[str, Path]) -> ScreenRegion:
    """Parse a screen source string into a :class:`ScreenRegion`.

    Raises:
        ValueError: If *source* is not a screen source, or carries a number of
            integers other than 0 (whole monitor), 1 (monitor index), 4 (box on
            the merged desktop), or 5 (monitor index plus box).
    """
    match = _SCREEN_RE.match(str(source))
    if match is None:
        raise ValueError(f"Not a screen source: {source!r}")

    nums = [int(tok) for tok in match.group("nums").split()]

    if len(nums) == 0:
        return ScreenRegion()
    if len(nums) == 1:
        return ScreenRegion(monitor=nums[0])
    if len(nums) == 4:
        left, top, width, height = nums
        return ScreenRegion(monitor=0, left=left, top=top, width=width, height=height)
    if len(nums) == 5:
        monitor, left, top, width, height = nums
        return ScreenRegion(
            monitor=monitor, left=left, top=top, width=width, height=height
        )

    raise ValueError(
        f"Invalid screen source: {source!r}. Expected 'screen', "
        "'screen <monitor>', 'screen <left> <top> <width> <height>', or "
        "'screen <monitor> <left> <top> <width> <height>'."
    )


def _import_mss():
    try:
        import mss
    except ImportError:
        raise ImportError(
            "Screen capture requires the 'mss' package. "
            "Install it with: pip install mss"
        ) from None
    return mss


def _resolve_box(sct, region: ScreenRegion) -> dict:
    """Turn a :class:`ScreenRegion` into an absolute mss capture box."""
    monitors = sct.monitors
    if not 0 <= region.monitor < len(monitors):
        raise ValueError(
            f"Monitor {region.monitor} does not exist. "
            f"{len(monitors) - 1} monitor(s) detected; valid indices are "
            f"0 (all monitors merged) to {len(monitors) - 1}."
        )

    monitor = monitors[region.monitor]
    coordinates = (region.left, region.top, region.width, region.height)
    if all(value is None for value in coordinates):
        return dict(monitor)
    if any(value is None for value in coordinates):
        raise ValueError("Screen capture box requires left, top, width, and height")

    left = region.left
    top = region.top
    width = region.width
    height = region.height
    assert left is not None and top is not None
    assert width is not None and height is not None

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Screen capture box must have positive size, got "
            f"width={width}, height={height}."
        )

    # Box coordinates are relative to the monitor's own origin.
    return {
        "left": monitor["left"] + left,
        "top": monitor["top"] + top,
        "width": width,
        "height": height,
    }


def grab_screen(source: Union[str, Path, ScreenRegion] = "screen") -> np.ndarray:
    """Capture the screen once and return an RGB ``uint8`` array (HWC).

    Args:
        source: A screen source string or a :class:`ScreenRegion`.

    Returns:
        RGB image array of shape ``(height, width, 3)``.
    """
    region = source if isinstance(source, ScreenRegion) else parse_screen_source(source)
    mss = _import_mss()

    with mss.mss() as sct:
        box = _resolve_box(sct, region)
        shot = sct.grab(box)
        # mss returns BGRA; drop alpha and flip to RGB.
        return np.asarray(shot, dtype=np.uint8)[:, :, :3][:, :, ::-1].copy()


class ScreenSource:
    """Iterate over screen captures, mirroring :class:`~.video.VideoSource`.

    Supports use as a context manager::

        with ScreenSource("screen 1", max_frames=100) as src:
            for frame_bgr, frame_idx in src:
                ...

    Args:
        source: Screen source string or :class:`ScreenRegion`.
        vid_stride: Emit every N-th capture (default ``1`` = every capture).
        max_frames: Stop after this many emitted frames. ``None`` (default)
            captures until the consumer stops asking, which for a live screen
            means forever.

    Note:
        Frames are yielded as BGR arrays so the shared frame-inference loop can
        treat a screen and a video file identically.
    """

    def __init__(
        self,
        source: Union[str, Path, ScreenRegion] = "screen",
        vid_stride: int = 1,
        max_frames: Optional[int] = None,
    ):
        self.region = (
            source if isinstance(source, ScreenRegion) else parse_screen_source(source)
        )
        self._vid_stride = max(1, int(vid_stride))
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be non-negative or None")
        self._max_frames = max_frames
        self._iterated = False

        mss = _import_mss()
        self._sct = mss.mss()
        try:
            self._box = _resolve_box(self._sct, self.region)
        except Exception:
            self.release()
            raise

        self.fps: float = DEFAULT_SCREEN_FPS / self._vid_stride
        # A live screen has no frame count; max_frames bounds it when given.
        self.total_frames: int = int(max_frames or 0)
        self.width: int = int(self._box["width"])
        self.height: int = int(self._box["height"])
        # Stem used when an annotated capture is written to disk.
        self.save_name: str = "screen"

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ScreenSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int]]:
        if self._sct is None or self._iterated:
            raise RuntimeError(
                "ScreenSource has been consumed or released. "
                "Create a new instance to iterate again."
            )
        self._iterated = True

        emitted = 0
        frame_idx = 0
        while self._max_frames is None or emitted < self._max_frames:
            shot = self._sct.grab(self._box)
            if frame_idx % self._vid_stride == 0:
                frame_bgra = np.asarray(shot, dtype=np.uint8)
                yield frame_bgra[:, :, :3].copy(), frame_idx
                emitted += 1
            frame_idx += 1

    def release(self):
        """Close the underlying mss handle. Safe to call multiple times."""
        sct = getattr(self, "_sct", None)
        if sct is not None:
            try:
                sct.close()
            finally:
                self._sct = None

    def __repr__(self) -> str:
        return (
            f"ScreenSource(region='{self.region}', "
            f"size={self.width}x{self.height}, "
            f"vid_stride={self._vid_stride})"
        )
