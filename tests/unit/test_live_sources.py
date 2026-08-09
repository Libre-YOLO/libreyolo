"""Local-only coverage for live prediction-source dispatch and capture."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libreyolo.utils import source as source_module
from libreyolo.utils.source import (
    MultiStreamSource,
    SourceKind,
    StreamFrame,
    StreamSource,
    classify_source,
    redact_source,
    resolve_youtube_stream,
)
from libreyolo.utils.results import Probs, Results
from libreyolo.utils.video import run_video_inference

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [0, 2, "0", "3"])
def test_webcam_indices_dispatch_as_streams(value):
    spec = classify_source(value)
    assert spec.kind == SourceKind.STREAM
    assert spec.items == (int(value),)


@pytest.mark.parametrize(
    "value",
    [
        "rtsp://127.0.0.1:8554/camera",
        "rtmp://127.0.0.1/live/camera",
        "tcp://127.0.0.1:9000",
        "udp://127.0.0.1:9000",
        "https://example.test/live/camera.m3u8",
    ],
)
def test_network_urls_dispatch_before_image_paths(value):
    spec = classify_source(value)
    assert spec.kind == SourceKind.STREAM
    assert spec.items == (value,)


@pytest.mark.parametrize(
    "value",
    [
        "https://www.youtube.com/watch?v=abc123",
        "https://youtu.be/abc123",
    ],
)
def test_youtube_pages_dispatch_as_streams(value):
    assert classify_source(value).kind == SourceKind.STREAM


def test_image_url_remains_an_image():
    assert classify_source("https://example.test/camera.jpg").kind == SourceKind.IMAGE


def test_stream_list_file_supports_comments_webcams_and_rtsp(tmp_path):
    path = tmp_path / "cameras.streams"
    path.write_text(
        "# loading dock\n0\n\nrtsp://127.0.0.1:8554/warehouse\n",
        encoding="utf-8",
    )
    spec = classify_source(path)
    assert spec.kind == SourceKind.STREAMS
    assert spec.items == (0, "rtsp://127.0.0.1:8554/warehouse")


def test_multi_source_list_cannot_mix_images_and_streams():
    with pytest.raises(TypeError, match="cannot mix"):
        classify_source([0, np.zeros((4, 4, 3), dtype=np.uint8)])


def test_rtsp_credentials_are_redacted_from_result_label():
    label = redact_source("rtsp://alice:secret@camera.test:8554/live")
    assert "secret" not in label
    assert label == "rtsp://alice:***@camera.test:8554/live"


def test_youtube_resolution_uses_yt_dlp_without_downloading(monkeypatch):
    calls = {}

    class FakeYDL:
        def __init__(self, options):
            calls["options"] = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def extract_info(self, url, download):
            calls["url"] = url
            calls["download"] = download
            return {"url": "https://media.test/direct.mp4"}

    monkeypatch.setattr(
        source_module,
        "_import_yt_dlp",
        lambda: SimpleNamespace(YoutubeDL=FakeYDL),
    )
    resolved = resolve_youtube_stream("https://youtu.be/example")
    assert resolved == "https://media.test/direct.mp4"
    assert calls["download"] is False
    assert calls["options"]["noplaylist"] is True


class _FakeCapture:
    def __init__(self, source, frames, *, first_read_barrier=None):
        self.source = source
        self.frames = list(frames)
        self.first_read_barrier = first_read_barrier
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2

        values = {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_WIDTH: 8,
            cv2.CAP_PROP_FRAME_HEIGHT: 6,
        }
        return values.get(prop, 0)

    def read(self):
        if self.read_count == 0 and self.first_read_barrier is not None:
            self.first_read_barrier.wait(timeout=2)
        self.read_count += 1
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def _frames(source_index, count=3):
    return [
        np.full((6, 8, 3), source_index * 10 + i, dtype=np.uint8) for i in range(count)
    ]


def test_single_webcam_capture_runs_on_reader_thread(monkeypatch):
    import cv2

    captures = []

    def fake_video_capture(source, *args):
        capture = _FakeCapture(source, _frames(0))
        captures.append(capture)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", fake_video_capture)
    with StreamSource(0, buffer=True) as stream:
        packets = list(stream)

    assert [packet.frame_idx for packet in packets] == [0, 1, 2]
    assert [int(packet.frame_bgr[0, 0, 0]) for packet in packets] == [0, 1, 2]
    assert captures[0].released is True
    assert stream._thread is not None
    assert stream._thread.name == "libreyolo-stream-0"


def test_multiple_cameras_open_and_read_independently(monkeypatch):
    import cv2

    barrier = threading.Barrier(2)
    captures = {}

    def fake_video_capture(source, *args):
        capture = _FakeCapture(
            source,
            _frames(int(source), count=2),
            first_read_barrier=barrier,
        )
        captures[source] = capture
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", fake_video_capture)
    with MultiStreamSource([0, 1], buffer=True) as streams:
        packets = list(streams)

    assert {(packet.source_index, packet.frame_idx) for packet in packets} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert {packet.source_label for packet in packets} == {"0", "1"}
    assert all(capture.released for capture in captures.values())


def test_existing_numeric_filename_is_not_claimed_as_webcam(tmp_path, monkeypatch):
    numeric_file = tmp_path / "0"
    numeric_file.write_bytes(b"not an image")
    monkeypatch.chdir(tmp_path)
    assert classify_source("0").kind == SourceKind.IMAGE


def test_missing_stream_list_fails_at_dispatch(tmp_path):
    path = Path(tmp_path) / "missing.streams"
    with pytest.raises(FileNotFoundError, match="Stream list not found"):
        classify_source(path)


def test_shared_video_loop_preserves_per_camera_path_and_frame_index():
    class FakeMultiSource:
        total_frames = 0
        fps = 25.0
        width = 8
        height = 6
        save_name = "streams"
        num_streams = 2

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def __iter__(self):
            for source_index, label in enumerate(("camera-a", "camera-b")):
                yield StreamFrame(
                    frame_bgr=np.zeros((6, 8, 3), dtype=np.uint8),
                    frame_idx=4 + source_index,
                    source_index=source_index,
                    source_label=label,
                    fps=25.0,
                )

    def predict_frame(_image):
        return Results(
            boxes=None,
            orig_shape=(6, 8),
            probs=Probs(torch.tensor([1.0])),
            names={0: "frame"},
        )

    results = list(
        run_video_inference(
            FakeMultiSource(),
            predict_frame,
            progress=False,
        )
    )

    assert [(result.path, result.frame_idx) for result in results] == [
        ("camera-a", 4),
        ("camera-b", 5),
    ]
