"""Unit tests for screen-capture source parsing and frame iteration."""

from types import SimpleNamespace

import numpy as np
import pytest

from libreyolo.utils import screen
from libreyolo.utils.screen import (
    ScreenRegion,
    ScreenSource,
    _resolve_box,
    grab_screen,
    is_screen_source,
    parse_screen_source,
)

pytestmark = pytest.mark.unit


class _FakeCapture:
    def __init__(self):
        self.monitors = [
            {"left": -4, "top": 0, "width": 8, "height": 3},
            {"left": 0, "top": 0, "width": 4, "height": 3},
        ]
        self.grabs = []
        self.closed = False

    def grab(self, box):
        self.grabs.append(dict(box))
        frame = np.empty((box["height"], box["width"], 4), dtype=np.uint8)
        frame[...] = (10, 20, 30, 255)  # BGRA
        return frame

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def fake_mss(monkeypatch):
    captures = []

    def make_capture():
        capture = _FakeCapture()
        captures.append(capture)
        return capture

    monkeypatch.setattr(
        screen, "_import_mss", lambda: SimpleNamespace(mss=make_capture)
    )
    return captures


@pytest.mark.parametrize(
    "value",
    ["screen", "SCREEN", "screen 1", "screen -2 3 4 5", "screen 1 2 3 4 5"],
)
def test_is_screen_source_accepts_documented_forms(value):
    assert is_screen_source(value)


@pytest.mark.parametrize("value", [None, 0, "screenshot", "screen one", "screen 1.5"])
def test_is_screen_source_rejects_other_inputs(value):
    assert not is_screen_source(value)


def test_parse_screen_source_forms():
    assert parse_screen_source("screen") == ScreenRegion()
    assert parse_screen_source("screen 2") == ScreenRegion(monitor=2)
    assert parse_screen_source("screen 10 20 30 40") == ScreenRegion(
        monitor=0, left=10, top=20, width=30, height=40
    )
    assert parse_screen_source("screen 2 10 20 30 40") == ScreenRegion(
        monitor=2, left=10, top=20, width=30, height=40
    )


def test_parse_screen_source_rejects_bad_arity():
    with pytest.raises(ValueError, match="Invalid screen source"):
        parse_screen_source("screen 1 2")
    with pytest.raises(ValueError, match="Not a screen source"):
        parse_screen_source("camera")


def test_resolve_box_uses_monitor_relative_coordinates():
    capture = _FakeCapture()

    box = _resolve_box(
        capture, ScreenRegion(monitor=1, left=1, top=2, width=2, height=1)
    )

    assert box == {"left": 1, "top": 2, "width": 2, "height": 1}


def test_resolve_box_rejects_invalid_monitor_and_size():
    capture = _FakeCapture()
    with pytest.raises(ValueError, match="Monitor -1"):
        _resolve_box(capture, ScreenRegion(monitor=-1))
    with pytest.raises(ValueError, match="positive size"):
        _resolve_box(capture, ScreenRegion(left=0, top=0, width=0, height=2))
    with pytest.raises(ValueError, match="requires left, top, width, and height"):
        _resolve_box(capture, ScreenRegion(width=2, height=2))


def test_grab_screen_returns_rgb(fake_mss):
    image = grab_screen("screen 1")

    assert image.shape == (3, 4, 3)
    assert image.dtype == np.uint8
    assert image[0, 0].tolist() == [30, 20, 10]
    assert fake_mss[0].closed is True


def test_screen_source_yields_bgr_with_stride_and_closes(fake_mss):
    source = ScreenSource("screen 1", vid_stride=2, max_frames=2)

    with source:
        frames = list(source)

    assert [index for _, index in frames] == [0, 2]
    assert all(frame[0, 0].tolist() == [10, 20, 30] for frame, _ in frames)
    assert len(fake_mss[0].grabs) == 3
    assert source.fps == 15.0
    assert source.total_frames == 2
    assert source.save_name == "screen"
    assert fake_mss[0].closed is True


def test_screen_source_rejects_negative_max_frames():
    with pytest.raises(ValueError, match="max_frames"):
        ScreenSource("screen", max_frames=-1)
