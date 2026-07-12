"""Unit tests for video support utilities."""

from pathlib import Path

import numpy as np
import pytest

from libreyolo.utils.general import release_save_path_reservation
from libreyolo.utils.video import (
    VideoSource,
    VideoWriter,
    _codec_candidates,
    _normalize_capture_fps,
    _processed_frame_count,
    _validate_vid_stride,
    collect_video_results,
    is_video_file,
    resolve_video_save_path,
    run_video_inference,
)

pytestmark = pytest.mark.unit

cv2 = pytest.importorskip("cv2", reason="opencv-python required for video tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_video(tmp_path):
    """Create a tiny 10-frame 64x64 video for testing."""
    path = str(tmp_path / "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (64, 64))
    for i in range(10):
        # Each frame has a different shade so they're distinguishable
        frame = np.full((64, 64, 3), fill_value=i * 25, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ---------------------------------------------------------------------------
# is_video_file
# ---------------------------------------------------------------------------


class TestIsVideoFile:
    def test_video_extensions(self):
        assert is_video_file("clip.mp4")
        assert is_video_file("clip.avi")
        assert is_video_file("clip.mkv")
        assert is_video_file(Path("clip.mov"))

    def test_image_extensions(self):
        assert not is_video_file("photo.jpg")
        assert not is_video_file("photo.png")

    def test_non_string(self):
        assert not is_video_file(42)
        assert not is_video_file(None)
        assert not is_video_file(np.zeros((3, 3, 3)))

    def test_case_insensitive(self):
        assert is_video_file("CLIP.MP4")
        assert is_video_file("Clip.Avi")


# ---------------------------------------------------------------------------
# VideoSource
# ---------------------------------------------------------------------------


class TestVideoSource:
    def test_metadata(self, sample_video):
        vs = VideoSource(sample_video)
        assert vs.width == 64
        assert vs.height == 64
        assert vs.total_frames == 10
        assert vs.fps == pytest.approx(10.0, abs=1.0)
        vs.release()

    def test_iterate_all_frames(self, sample_video):
        vs = VideoSource(sample_video)
        frames = list(vs)
        assert len(frames) == 10
        # Each element is (frame_bgr, frame_idx)
        for frame, idx in frames:
            assert frame.shape == (64, 64, 3)
            assert frame.dtype == np.uint8

    def test_frame_indices_sequential(self, sample_video):
        vs = VideoSource(sample_video)
        indices = [idx for _, idx in vs]
        assert indices == list(range(10))

    def test_vid_stride(self, sample_video):
        vs = VideoSource(sample_video, vid_stride=3)
        frames = list(vs)
        indices = [idx for _, idx in frames]
        # With stride 3 and 10 frames (0-9): should get frames 0, 3, 6, 9
        assert indices == [0, 3, 6, 9]

    def test_vid_stride_2(self, sample_video):
        vs = VideoSource(sample_video, vid_stride=2)
        frames = list(vs)
        indices = [idx for _, idx in frames]
        assert indices == [0, 2, 4, 6, 8]

    def test_vid_stride_larger_than_total(self, sample_video):
        vs = VideoSource(sample_video, vid_stride=100)
        frames = list(vs)
        # Only frame 0 matches (0 % 100 == 0), all others are skipped
        assert len(frames) == 1
        assert frames[0][1] == 0

    def test_invalid_path(self):
        with pytest.raises(ValueError, match="Cannot open video"):
            VideoSource("/nonexistent/video.mp4")

    def test_re_iteration_raises(self, sample_video):
        vs = VideoSource(sample_video)
        list(vs)  # consume once
        with pytest.raises(RuntimeError, match="consumed or released"):
            list(vs)

    def test_context_manager(self, sample_video):
        with VideoSource(sample_video) as vs:
            frames = list(vs)
        assert len(frames) == 10
        # After exiting context, cap should be released
        assert vs._cap is None

    def test_double_release_safe(self, sample_video):
        vs = VideoSource(sample_video)
        vs.release()
        vs.release()  # should not raise

    def test_repr(self, sample_video):
        vs = VideoSource(sample_video)
        r = repr(vs)
        assert "VideoSource" in r
        assert "64x64" in r
        vs.release()

    @pytest.mark.parametrize("stride", [0, -1, 1.5, "2", float("nan"), True])
    def test_invalid_vid_stride_is_rejected(self, sample_video, stride):
        with pytest.raises((TypeError, ValueError), match="vid_stride"):
            VideoSource(sample_video, vid_stride=stride)


def test_processed_frame_count_uses_ceil_semantics():
    assert _processed_frame_count(10, 3) == 4
    assert _processed_frame_count(10, 2) == 5
    assert _processed_frame_count(0, 3) == 0


def test_capture_fps_falls_back_for_nonfinite_and_nonpositive_metadata():
    assert _normalize_capture_fps(float("nan")) == 30.0
    assert _normalize_capture_fps(float("inf")) == 30.0
    assert _normalize_capture_fps(-1.0) == 30.0
    assert _normalize_capture_fps(24.0) == 24.0


def test_validate_vid_stride_accepts_numpy_integer():
    assert _validate_vid_stride(np.int64(3)) == 3


# ---------------------------------------------------------------------------
# VideoWriter
# ---------------------------------------------------------------------------


class TestVideoWriter:
    def test_mp4_prefers_h264_codec(self, tmp_path, monkeypatch):
        assert _codec_candidates("output.mp4")[0] == "avc1"

        calls = []
        avc1 = cv2.VideoWriter_fourcc(*"avc1")

        class FakeWriter:
            def __init__(self, opened):
                self.opened = opened
                self.released = False

            def isOpened(self):
                return self.opened

            def release(self):
                self.released = True

        def fake_video_writer(path, fourcc, fps, size):
            calls.append(fourcc)
            return FakeWriter(opened=fourcc == avc1)

        monkeypatch.setattr(cv2, "VideoWriter", fake_video_writer)

        writer = VideoWriter(tmp_path / "output.mp4", fps=10.0, width=32, height=32)
        writer.release()

        assert writer.codec == "avc1"
        assert calls == [avc1]

    def test_mp4_falls_back_to_mp4v(self, tmp_path, monkeypatch, caplog):
        calls = []
        avc1 = cv2.VideoWriter_fourcc(*"avc1")
        mp4v = cv2.VideoWriter_fourcc(*"mp4v")

        class FakeWriter:
            def __init__(self, opened):
                self.opened = opened
                self.released = False

            def isOpened(self):
                return self.opened

            def release(self):
                self.released = True

        def fake_video_writer(path, fourcc, fps, size):
            calls.append(fourcc)
            return FakeWriter(opened=fourcc == mp4v)

        monkeypatch.setattr(cv2, "VideoWriter", fake_video_writer)

        writer = VideoWriter(tmp_path / "output.mp4", fps=10.0, width=32, height=32)
        writer.release()

        assert writer.codec == "mp4v"
        assert calls == [avc1, mp4v]
        assert "falling back to mp4v" in caplog.text

    def test_write_and_read_back(self, tmp_path):
        out_path = str(tmp_path / "output.mp4")
        writer = VideoWriter(out_path, fps=10.0, width=32, height=32)

        for i in range(5):
            frame = np.full((32, 32, 3), fill_value=i * 50, dtype=np.uint8)
            writer.write_frame(frame)
        writer.release()

        # Read back and verify
        cap = cv2.VideoCapture(out_path)
        assert cap.isOpened()
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        assert count == 5

    def test_creates_parent_dirs(self, tmp_path):
        out_path = str(tmp_path / "sub" / "dir" / "output.mp4")
        writer = VideoWriter(out_path, fps=10.0, width=32, height=32)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        writer.write_frame(frame)
        writer.release()
        assert Path(out_path).exists()

    def test_context_manager(self, tmp_path):
        out_path = str(tmp_path / "ctx_output.mp4")
        with VideoWriter(out_path, fps=10.0, width=32, height=32) as writer:
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            writer.write_frame(frame)
        assert Path(out_path).exists()
        assert writer._writer is None  # released

    def test_open_failure_releases_reserved_path_for_retry(
        self, tmp_path, monkeypatch
    ):
        class ClosedWriter:
            @staticmethod
            def isOpened():
                return False

            @staticmethod
            def release():
                pass

        monkeypatch.setattr(cv2, "VideoWriter", lambda *_args: ClosedWriter())
        explicit = tmp_path / "prediction.mp4"
        reserved = Path(resolve_video_save_path("clip.mov", explicit))

        with pytest.raises(ValueError, match="Cannot open video writer"):
            VideoWriter(reserved, fps=10.0, width=32, height=32)

        retry = Path(resolve_video_save_path("clip.mov", explicit))
        assert retry == explicit
        assert release_save_path_reservation(retry) is True

    def test_release_without_created_file_reclaims_reserved_path(self, tmp_path):
        class UnmaterializedWriter:
            @staticmethod
            def release():
                pass

        explicit = tmp_path / "prediction.mp4"
        reserved = Path(resolve_video_save_path("clip.mov", explicit))
        writer = VideoWriter.__new__(VideoWriter)
        writer._path = str(reserved)
        writer._writer = UnmaterializedWriter()

        writer.release()

        retry = Path(resolve_video_save_path("clip.mov", explicit))
        assert retry == explicit
        assert release_save_path_reservation(retry) is True

    @pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_fps(self, tmp_path, fps):
        with pytest.raises(ValueError, match="FPS"):
            VideoWriter(tmp_path / "invalid.mp4", fps=fps, width=32, height=32)


class TestResolveVideoSavePath:
    def test_directory_output_gets_real_video_filename(self, tmp_path):
        output_dir = tmp_path / "predict"
        output_dir.mkdir()

        resolved = Path(resolve_video_save_path("clip.mov", output_dir))

        assert resolved == output_dir / "clip.mp4"

    def test_suffix_bearing_directory_gets_real_video_filename(self, tmp_path):
        output_dir = tmp_path / "predict.v1"
        output_dir.mkdir()

        resolved = Path(resolve_video_save_path("clip.mov", output_dir))

        assert resolved.parent == output_dir
        assert resolved.name == "clip.mp4"

    def test_duplicate_explicit_file_is_collision_safe(self, tmp_path):
        output_file = tmp_path / "prediction.mp4"

        first = Path(resolve_video_save_path("clip.mov", output_file))
        second = Path(resolve_video_save_path("clip.mov", output_file))

        assert first == output_file
        assert second == tmp_path / "prediction2.mp4"


# ---------------------------------------------------------------------------
# Results.frame_idx
# ---------------------------------------------------------------------------


class TestResultsFrameIdx:
    def test_default_none(self):
        import torch

        from libreyolo.utils.results import Boxes, Results

        boxes = Boxes(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,)))
        result = Results(boxes=boxes, orig_shape=(480, 640))
        assert result.frame_idx is None

    def test_set_frame_idx(self):
        import torch

        from libreyolo.utils.results import Boxes, Results

        boxes = Boxes(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,)))
        result = Results(boxes=boxes, orig_shape=(480, 640), frame_idx=42)
        assert result.frame_idx == 42

    def test_cpu_preserves_frame_idx(self):
        import torch

        from libreyolo.utils.results import Boxes, Results

        boxes = Boxes(torch.zeros((1, 4)), torch.zeros((1,)), torch.zeros((1,)))
        result = Results(boxes=boxes, orig_shape=(480, 640), frame_idx=7)
        cpu_result = result.cpu()
        assert cpu_result.frame_idx == 7

    def test_repr_includes_frame_idx(self):
        import torch

        from libreyolo.utils.results import Boxes, Results

        boxes = Boxes(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,)))
        result = Results(boxes=boxes, orig_shape=(480, 640), frame_idx=3)
        assert "frame_idx=3" in repr(result)

    def test_repr_omits_frame_idx_when_none(self):
        import torch

        from libreyolo.utils.results import Boxes, Results

        boxes = Boxes(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,)))
        result = Results(boxes=boxes, orig_shape=(480, 640))
        assert "frame_idx" not in repr(result)


class TestRunVideoInference:
    def test_progress_initialization_failure_releases_output(
        self, sample_video, tmp_path, monkeypatch
    ):
        import tqdm as tqdm_module

        explicit = tmp_path / "prediction.mp4"

        def fail_progress(*_args, **_kwargs):
            raise RuntimeError("synthetic progress failure")

        monkeypatch.setattr(tqdm_module, "tqdm", fail_progress)

        with pytest.raises(RuntimeError, match="synthetic progress failure"):
            list(
                run_video_inference(
                    sample_video,
                    lambda _image: None,
                    save=True,
                    output_path=str(explicit),
                    progress=True,
                )
            )

        retry = Path(resolve_video_save_path(sample_video, explicit))
        assert retry == explicit
        assert release_save_path_reservation(retry) is True

    def test_prediction_failure_releases_unmaterialized_output(
        self, sample_video, tmp_path
    ):
        explicit = tmp_path / "prediction.mp4"

        def fail_prediction(_image):
            raise RuntimeError("synthetic prediction failure")

        with pytest.raises(RuntimeError, match="synthetic prediction failure"):
            list(
                run_video_inference(
                    sample_video,
                    fail_prediction,
                    save=True,
                    output_path=str(explicit),
                    progress=False,
                )
            )

        retry = Path(resolve_video_save_path(sample_video, explicit))
        assert retry == explicit
        assert release_save_path_reservation(retry) is True

    def test_save_video_renders_gaze_direction(self, sample_video, tmp_path, monkeypatch):
        import torch

        from libreyolo.utils import drawing
        from libreyolo.utils.results import Boxes, Gaze, Results

        calls = []
        original = drawing.draw_gaze_arrows

        def record_gaze(*args, **kwargs):
            calls.append((args[2], args[3]))
            return original(*args, **kwargs)

        monkeypatch.setattr(drawing, "draw_gaze_arrows", record_gaze)

        def predict_frame(pil_img):
            width, height = pil_img.size
            boxes = Boxes(
                torch.tensor([[2.0, 2.0, width - 2.0, height - 2.0]]),
                torch.tensor([0.9]),
                torch.tensor([0.0]),
            )
            return Results(
                boxes,
                (height, width),
                names={0: "face"},
                gaze=Gaze(torch.tensor([[0.1, -0.2]])),
                task="gaze",
            )

        results = list(
            run_video_inference(
                sample_video,
                predict_frame,
                save=True,
                output_path=str(tmp_path / "gaze.mp4"),
                progress=False,
            )
        )

        assert len(calls) == len(results) == 10

    def test_save_classification_results_without_boxes(self, sample_video, tmp_path):
        import torch

        from libreyolo.utils.results import Probs, Results

        def predict_frame(pil_img):
            width, height = pil_img.size
            return Results(
                boxes=None,
                orig_shape=(height, width),
                names={0: "a", 1: "b"},
                probs=Probs(torch.tensor([0.7, 0.3])),
            )

        output_path = tmp_path / "classified.mp4"
        results = list(
            run_video_inference(
                sample_video,
                predict_frame,
                save=True,
                output_path=str(output_path),
                progress=False,
            )
        )

        assert len(results) == 10
        assert output_path.exists()
        assert {r.saved_path for r in results} == {str(output_path)}

    def test_save_point_results_without_boxes(self, sample_video, tmp_path):
        import torch

        from libreyolo.utils.results import Points, Results

        def predict_frame(pil_img):
            width, height = pil_img.size
            return Results(
                boxes=None,
                orig_shape=(height, width),
                names={0: "person"},
                points=Points(torch.tensor([[width / 2, height / 2, 0.0, 0.9]])),
            )

        output_path = tmp_path / "points.mp4"
        results = list(
            run_video_inference(
                sample_video,
                predict_frame,
                save=True,
                output_path=str(output_path),
                progress=False,
            )
        )

        assert len(results) == 10
        assert output_path.exists()
        assert {r.saved_path for r in results} == {str(output_path)}

    @pytest.mark.parametrize("task", ["semantic", "panoptic", "ocr"])
    def test_save_boxless_dense_and_ocr_video_results(
        self, sample_video, tmp_path, task
    ):
        import torch

        from libreyolo.utils.results import (
            OCRRegions,
            PanopticSegmentation,
            Results,
            SemanticMask,
        )

        def predict_frame(pil_img):
            width, height = pil_img.size
            kwargs = {}
            if task == "semantic":
                mask = torch.zeros((height, width), dtype=torch.int64)
                mask[:, width // 2 :] = 1
                kwargs["semantic_mask"] = SemanticMask(mask)
            elif task == "panoptic":
                segment_map = torch.ones((height, width), dtype=torch.int64)
                kwargs["panoptic"] = PanopticSegmentation(
                    segment_map,
                    [{"id": 1, "category_id": 0, "isthing": True}],
                )
            else:
                polygon = torch.tensor(
                    [[[2.0, 2.0], [20.0, 2.0], [20.0, 12.0], [2.0, 12.0]]]
                )
                kwargs["ocr"] = OCRRegions(
                    polygon,
                    ["text"],
                    torch.tensor([0.9]),
                    torch.tensor([0.8]),
                )
            return Results(
                boxes=None,
                orig_shape=(height, width),
                names={0: "thing", 1: "other"},
                **kwargs,
            )

        output_path = tmp_path / f"{task}.mp4"
        results = list(
            run_video_inference(
                sample_video,
                predict_frame,
                save=True,
                output_path=str(output_path),
                progress=False,
            )
        )

        assert len(results) == 10
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        assert {r.saved_path for r in results} == {str(output_path)}


class TestCollectVideoResults:
    """Tests for collect_video_results()."""

    def test_collects_all_items(self, sample_video):
        """Generator items are collected into a list."""
        items = ["a", "b", "c"]
        result = collect_video_results(iter(items), sample_video, vid_stride=1)
        assert result == ["a", "b", "c"]

    def test_empty_generator(self, sample_video):
        result = collect_video_results(iter([]), sample_video, vid_stride=1)
        assert result == []

    def test_warns_on_large_video(self, tmp_path):
        """Videos with many frames should emit a warning."""
        # Create a video that reports many frames (>500 threshold)
        path = str(tmp_path / "long.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Write 600 frames
        writer = cv2.VideoWriter(path, fourcc, 30.0, (16, 16))
        for _ in range(600):
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        writer.release()

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            collect_video_results(iter([1, 2, 3]), path, vid_stride=1)
            memory_warnings = [x for x in w if "stream=True" in str(x.message)]
            assert len(memory_warnings) == 1
