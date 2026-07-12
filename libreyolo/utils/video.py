"""Video source utilities for LibreYOLO."""

import logging
import math
import operator
import warnings
from pathlib import Path
from typing import Callable, Generator, Iterator, Tuple, Union

import numpy as np

from .general import release_save_path_reservation, resolve_save_path

logger = logging.getLogger(__name__)

MP4_CODEC_CANDIDATES = ("avc1", "mp4v")

# Video extensions supported via OpenCV's VideoCapture
VIDEO_EXTENSIONS = {
    ".asf",
    ".avi",
    ".gif",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".wmv",
    ".webm",
}


def is_video_file(source) -> bool:
    """Check whether *source* looks like a path to a video file."""
    if not isinstance(source, (str, Path)):
        return False
    return Path(source).suffix.lower() in VIDEO_EXTENSIONS


def _codec_candidates(path: Union[str, Path]) -> Tuple[str, ...]:
    if Path(path).suffix.lower() == ".mp4":
        return MP4_CODEC_CANDIDATES
    return ("mp4v",)


def resolve_video_save_path(
    source: Union[str, Path], output_path: Union[str, None]
) -> str:
    """Determine the output path for a saved video.

    Directory outputs receive a collision-safe ``<source>.mp4`` filename;
    explicit file outputs are also allocated without overwriting an existing
    artifact. When omitted, the standard ``runs/detect/predict*/`` directory
    is used.
    """
    return str(
        resolve_save_path(
            output_path,
            source,
            ext="mp4",
            default_dir="runs/detect",
        )
    )


def _validate_vid_stride(value: int) -> int:
    """Return a positive integer video stride without lossy coercion."""
    if isinstance(value, bool):
        raise TypeError("vid_stride must be a positive integer, not bool.")
    try:
        stride = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"vid_stride must be a positive integer, got {value!r}."
        ) from exc
    if stride <= 0:
        raise ValueError(f"vid_stride must be >= 1, got {stride}.")
    return stride


def _processed_frame_count(total_frames: int, vid_stride: int) -> int:
    """Return the number of frame indices selected from ``range(total)``."""
    if total_frames <= 0:
        return 0
    return (total_frames + vid_stride - 1) // vid_stride


def _normalize_capture_fps(value: object, *, fallback: float = 30.0) -> float:
    """Return a finite positive capture FPS, falling back for bad metadata."""
    try:
        fps = float(value)
    except (TypeError, ValueError):
        fps = math.nan
    if math.isfinite(fps) and fps > 0:
        return fps
    logger.warning("Invalid video FPS metadata %r; defaulting to %.1f", value, fallback)
    return fallback


class VideoSource:
    """Iterate over video frames using OpenCV.

    Supports use as a context manager::

        with VideoSource("clip.mp4", vid_stride=2) as src:
            for frame_bgr, frame_idx in src:
                ...

    Args:
        path: Path to a video file.
        vid_stride: Process every N-th frame (default ``1`` = every frame).

    Note:
        A ``VideoSource`` instance can only be iterated **once**. After
        iteration completes (or the source is released), create a new
        instance to iterate again.
    """

    def __init__(self, path: Union[str, Path], vid_stride: int = 1):
        self._vid_stride = _validate_vid_stride(vid_stride)
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "Video support requires 'opencv-python'. "
                "Install it with: pip install opencv-python"
            )

        self._path = str(path)

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            self._cap.release()
            raise ValueError(f"Cannot open video file: {self._path}")

        self._iterated = False

        detected_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = _normalize_capture_fps(detected_fps)
        raw_total_frames = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        try:
            valid_total = float(raw_total_frames)
        except (TypeError, ValueError):
            valid_total = math.nan
        self.total_frames = (
            int(valid_total) if math.isfinite(valid_total) and valid_total >= 0 else 0
        )
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int]]:
        if self._cap is None or self._iterated:
            raise RuntimeError(
                "VideoSource has been consumed or released. "
                "Create a new instance to iterate again."
            )
        self._iterated = True

        frame_idx = 0
        while self._cap.isOpened():
            grabbed = self._cap.grab()
            if not grabbed:
                break

            # Only decode on the stride boundary
            if frame_idx % self._vid_stride == 0:
                ok, frame = self._cap.retrieve()
                if ok:
                    yield frame, frame_idx
                else:
                    logger.warning(
                        "Failed to decode frame %d in %s, skipping",
                        frame_idx,
                        self._path,
                    )

            frame_idx += 1

    def release(self):
        """Release the underlying VideoCapture. Safe to call multiple times."""
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def __repr__(self) -> str:
        return (
            f"VideoSource(path='{self._path}', "
            f"fps={self.fps:.1f}, "
            f"frames={self.total_frames}, "
            f"size={self.width}x{self.height}, "
            f"vid_stride={self._vid_stride})"
        )


class VideoWriter:
    """Write annotated frames to a video file using OpenCV.

    Supports use as a context manager::

        with VideoWriter("out.mp4", fps=25, width=1920, height=1080) as w:
            w.write_frame(frame_bgr)

    Args:
        path: Output video file path (should end in ``.mp4``).
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    def __init__(self, path: Union[str, Path], fps: float, width: int, height: int):
        self._path = str(path)
        try:
            self._open(fps, width, height)
        except BaseException:
            release_save_path_reservation(self._path)
            raise

    def _open(self, fps: float, width: int, height: int) -> None:
        """Validate parameters and open the first available video codec."""
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "Video writing requires 'opencv-python'. "
                "Install it with: pip install opencv-python"
            )

        if isinstance(fps, bool):
            raise TypeError("Video writer FPS must be a finite positive number.")
        try:
            fps = float(fps)
        except (TypeError, ValueError) as exc:
            raise TypeError("Video writer FPS must be a finite positive number.") from exc
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video writer FPS must be finite and > 0, got {fps!r}.")

        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        self.codec = None
        self._writer = None
        for codec in _codec_candidates(self._path):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(self._path, fourcc, fps, (width, height))
            if writer.isOpened():
                self.codec = codec
                self._writer = writer
                break
            writer.release()

        if self._writer is None:
            raise ValueError(f"Cannot open video writer for: {self._path}")

        if self.codec != "avc1" and Path(self._path).suffix.lower() == ".mp4":
            logger.warning(
                "Could not open H.264 video writer; falling back to %s for %s",
                self.codec,
                self._path,
            )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # ------------------------------------------------------------------

    def write_frame(self, frame_bgr: np.ndarray):
        """Write a single BGR frame."""
        self._writer.write(frame_bgr)

    def release(self):
        """Flush and close the writer. Safe to call multiple times."""
        try:
            if self._writer is not None:
                try:
                    self._writer.release()
                finally:
                    self._writer = None
        finally:
            release_save_path_reservation(self._path)

    def __repr__(self) -> str:
        return f"VideoWriter(path='{self._path}')"


# ---------------------------------------------------------------------------
# Shared video inference helpers
# ---------------------------------------------------------------------------

_LARGE_VIDEO_THRESHOLD = 500


def collect_video_results(
    gen: Generator,
    source: Union[str, Path],
    vid_stride: int = 1,
) -> list:
    """Collect all video results into a list, warning for large videos."""
    stride = _validate_vid_stride(vid_stride)
    vs = VideoSource(source, vid_stride=stride)
    est_frames = _processed_frame_count(vs.total_frames, stride)
    vs.release()

    if est_frames > _LARGE_VIDEO_THRESHOLD:
        warnings.warn(
            f"Video has ~{est_frames} frames to process. "
            f"Consider using stream=True to avoid high memory usage.",
            stacklevel=3,
        )
    return list(gen)


def run_video_inference(
    source: Union[str, Path],
    predict_frame_fn: Callable,
    *,
    vid_stride: int = 1,
    save: bool = False,
    show: bool = False,
    output_path: Union[str, None] = None,
    annotate_fn: Union[Callable, None] = None,
    progress: bool = True,
) -> Generator:
    """Generic video inference loop shared by all backends.

    Args:
        source: Path to video file.
        predict_frame_fn: Callable that takes a PIL RGB image and returns
            a ``Results`` object.
        vid_stride: Process every N-th frame.
        save: Write annotated output video.
        show: Display frames in a cv2 window.
        output_path: Output path for saved video.
        annotate_fn: Optional callable ``(pil_img, result) -> pil_img`` for
            custom annotation (e.g. tracking labels). When *None*, the default
            ``draw_boxes()`` annotation is used.
        progress: Show a tqdm progress bar (frames processed, fps).

    Yields:
        ``Results`` for each processed frame.
    """
    import cv2
    import torch
    from PIL import Image
    from tqdm import tqdm

    from .drawing import (
        draw_boxes,
        draw_depth_map,
        draw_gaze_arrows,
        draw_keypoints,
        draw_masks,
        draw_matte,
        draw_obb,
        draw_ocr_regions,
        draw_panoptic,
        draw_points,
        draw_semantic_mask,
    )

    stride = _validate_vid_stride(vid_stride)
    with VideoSource(source, vid_stride=stride) as video_src:
        writer = None
        out_path = None
        effective_fps = None
        pbar = None
        try:
            if save:
                out_path = resolve_video_save_path(source, output_path)
                effective_fps = video_src.fps / stride
                # The writer is created lazily from the first output frame instead
                # of the source dimensions: restore/super-resolution results render
                # on a canvas ``restore_scale`` times the source frame.

            total = _processed_frame_count(video_src.total_frames, stride) or None
            pbar = (
                tqdm(
                    total=total,
                    desc=Path(source).name,
                    unit="frame",
                    dynamic_ncols=True,
                )
                if progress
                else None
            )

            for frame_bgr, frame_idx in video_src:
                # Convert BGR frame to PIL RGB for the model pipeline
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # Run model-specific inference
                result = predict_frame_fn(pil_img)
                result.frame_idx = frame_idx

                # Annotate frame for save/show
                if save or show:
                    if annotate_fn is not None:
                        annotated_pil = annotate_fn(pil_img, result)
                    elif (
                        result.boxes is None
                        and getattr(result, "probs", None) is not None
                    ):
                        annotated_pil = pil_img
                    elif (
                        result.boxes is None
                        and getattr(result, "points", None) is not None
                    ):
                        if len(result.points) > 0:
                            annotated_pil = draw_points(
                                pil_img,
                                result.points.xy.tolist(),
                                result.points.conf.tolist(),
                                result.points.cls.tolist(),
                                class_names=result.names,
                            )
                        else:
                            annotated_pil = pil_img
                    elif (
                        result.boxes is None
                        and getattr(result, "restored", None) is not None
                    ):
                        annotated_pil = Image.fromarray(
                            result.restored.array, mode="RGB"
                        )
                    elif (
                        result.boxes is None
                        and getattr(result, "matte", None) is not None
                    ):
                        # Checkerboard-composited cutout preview (video frames
                        # cannot carry an alpha channel, so the transparency is
                        # visualized instead).
                        annotated_pil = draw_matte(pil_img, result.matte.array)
                    elif (
                        result.boxes is None
                        and getattr(result, "depth_map", None) is not None
                    ):
                        depth_np = result.depth_map.data
                        if isinstance(depth_np, torch.Tensor):
                            depth_np = depth_np.cpu().numpy()
                        annotated_pil = draw_depth_map(pil_img, depth_np)
                    elif (
                        result.boxes is None
                        and getattr(result, "semantic_mask", None) is not None
                    ):
                        semantic_np = result.semantic_mask.data
                        if isinstance(semantic_np, torch.Tensor):
                            semantic_np = semantic_np.cpu().numpy()
                        annotated_pil = draw_semantic_mask(pil_img, semantic_np)
                    elif (
                        result.boxes is None
                        and getattr(result, "panoptic", None) is not None
                    ):
                        panoptic_np = result.panoptic.data
                        if isinstance(panoptic_np, torch.Tensor):
                            panoptic_np = panoptic_np.cpu().numpy()
                        annotated_pil = draw_panoptic(
                            pil_img,
                            panoptic_np,
                            result.panoptic.segments_info,
                            class_names=result.names,
                        )
                    elif (
                        result.boxes is None
                        and getattr(result, "ocr", None) is not None
                    ):
                        if len(result.ocr) > 0:
                            ocr_np = result.ocr.numpy()
                            annotated_pil = draw_ocr_regions(
                                pil_img,
                                ocr_np.data,
                                ocr_np.texts,
                                ocr_np.conf,
                            )
                        else:
                            annotated_pil = pil_img
                    elif result.boxes is not None and len(result.boxes) > 0:
                        annotated_pil = pil_img
                        if result.masks is not None:
                            masks_np = result.masks.data
                            if isinstance(masks_np, torch.Tensor):
                                masks_np = masks_np.cpu().numpy()
                            annotated_pil = draw_masks(
                                annotated_pil,
                                masks_np,
                                result.boxes.cls.tolist(),
                            )
                        if result.obb is not None:
                            annotated_pil = draw_obb(
                                annotated_pil,
                                result.obb.xywhr.tolist(),
                                result.obb.conf.tolist(),
                                result.obb.cls.tolist(),
                                class_names=result.names,
                                track_ids=(
                                    result.obb.id.tolist()
                                    if result.obb.id is not None
                                    else None
                                ),
                            )
                        else:
                            annotated_pil = draw_boxes(
                                annotated_pil,
                                result.boxes.xyxy.tolist(),
                                result.boxes.conf.tolist(),
                                result.boxes.cls.tolist(),
                                class_names=result.names,
                            )
                        if result.keypoints is not None:
                            kpts_np = result.keypoints.data
                            if isinstance(kpts_np, torch.Tensor):
                                kpts_np = kpts_np.cpu().numpy()
                            annotated_pil = draw_keypoints(annotated_pil, kpts_np)
                        if result.gaze is not None:
                            boxes_np = result.boxes.numpy()
                            gaze_np = result.gaze.numpy()
                            annotated_pil = draw_gaze_arrows(
                                annotated_pil,
                                boxes_np.xyxy.tolist(),
                                gaze_np.pitch.tolist(),
                                gaze_np.yaw.tolist(),
                            )
                    else:
                        annotated_pil = pil_img

                    annotated_bgr = cv2.cvtColor(
                        np.array(annotated_pil), cv2.COLOR_RGB2BGR
                    )

                    if save:
                        if writer is None:
                            frame_h, frame_w = annotated_bgr.shape[:2]
                            writer = VideoWriter(
                                out_path, effective_fps, frame_w, frame_h
                            )
                        writer.write_frame(annotated_bgr)

                    if show:
                        cv2.imshow("LibreYOLO", annotated_bgr)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                if save:
                    result.saved_path = str(out_path)

                if pbar is not None:
                    n_dets = len(result) if result is not None else 0
                    pbar.set_postfix(dets=n_dets, refresh=False)
                    pbar.update(1)

                yield result

        finally:
            try:
                if pbar is not None:
                    pbar.close()
            finally:
                try:
                    if writer is not None:
                        writer.release()
                        logger.info("Video saved to %s", out_path)
                finally:
                    if out_path is not None:
                        release_save_path_reservation(out_path)
                    if show:
                        cv2.destroyAllWindows()
