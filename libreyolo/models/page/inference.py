"""Inference orchestrator for LibrePAGE gaze-target estimation.

Wraps the two-stage pipeline (head detection -> PaGE forward -> heatmap
decode) behind the same ``__call__`` shape that ``InferenceRunner``
provides for detection models, so ``LibrePAGE`` integrates with the rest
of the framework via the standard ``BaseModel`` runner property.

Head boxes can be supplied directly (``head_boxes=[...]``, xyxy pixels or
normalized), or found automatically with any detector implementing the
shared face-detector protocol from the L2CS gaze family. Detector face
boxes are expanded by ``head_expand`` (default 1.4) because gaze-target
models are trained on full-head boxes, which are larger than the tight
face boxes detectors produce.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Generator, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from ...utils.general import log_saved_result, resolve_save_path
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.results import Boxes, GazeTargets, Results
from ...utils.video import collect_video_results, is_video_file, run_video_inference
from ..l2cs.face import FaceDetector, default_face_detector, resolve_face_detector
from .utils import (
    clamp_pixel_box,
    crop_boxes_from_faces,
    decode_heatmaps,
    head_rects_grid,
    preprocess_scene_and_heads,
)

if TYPE_CHECKING:
    from .model import LibrePAGE


logger = logging.getLogger(__name__)


class PageInferenceRunner:
    """Runs gaze-target inference for a ``LibrePAGE`` model."""

    def __init__(self, model: "LibrePAGE"):
        self.model = model

    # =========================================================================
    # Public entry point
    # =========================================================================

    def __call__(
        self,
        source: ImageInput | None = None,
        *,
        head_boxes: Optional[Sequence] = None,
        head_detector: Optional[FaceDetector] = None,
        head_conf: float = 0.5,
        head_expand: float = 1.4,
        save: bool = False,
        output_path: Optional[str] = None,
        color_format: str = "auto",
        stream: bool = False,
        vid_stride: int = 1,
        show: bool = False,
        output_file_format: Optional[str] = None,
        device: Optional[str] = None,
        # Rejected with a clear message so detection-shaped kwargs fail loudly
        augment: bool = False,
        tiling: bool = False,
        **_: object,
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        if augment:
            raise ValueError(
                "TTA (augment=True) is not meaningful for gaze-target "
                "inference; a horizontal flip mirrors the target and there is "
                "nothing sensible to merge."
            )
        if tiling:
            raise ValueError(
                "Tiled inference is not supported for gaze-target estimation "
                "(the scene context is global by construction)."
            )
        if output_file_format is not None:
            output_file_format = output_file_format.lower().lstrip(".")
            if output_file_format not in ("jpg", "jpeg", "png", "webp"):
                raise ValueError(
                    f"Invalid output_file_format: {output_file_format}. "
                    "Must be one of: 'jpg', 'png', 'webp'."
                )
        if device is not None:
            self._set_device(device)

        detector = self._resolve_runtime_detector(head_detector, head_boxes)

        if is_video_file(source):
            gen = self._predict_video(
                source,
                detector=detector,
                head_conf=head_conf,
                head_expand=head_expand,
                save=save,
                show=show,
                vid_stride=vid_stride,
                output_path=output_path,
                output_file_format=output_file_format,
            )
            if stream:
                return gen
            return collect_video_results(gen, source, vid_stride)

        if isinstance(source, (str, Path)) and Path(source).is_dir():
            image_paths = ImageLoader.collect_images(source)
            return [
                self._predict_single(
                    p,
                    detector=detector,
                    head_boxes=None,
                    head_conf=head_conf,
                    head_expand=head_expand,
                    save=save,
                    output_path=output_path,
                    color_format=color_format,
                    output_file_format=output_file_format,
                )
                for p in image_paths
            ]

        return self._predict_single(
            source,
            detector=detector,
            head_boxes=head_boxes,
            head_conf=head_conf,
            head_expand=head_expand,
            save=save,
            output_path=output_path,
            color_format=color_format,
            output_file_format=output_file_format,
        )

    # =========================================================================
    # Single-frame path
    # =========================================================================

    def _predict_single(
        self,
        image: ImageInput,
        *,
        detector: Optional[FaceDetector],
        head_boxes: Optional[Sequence],
        head_conf: float,
        head_expand: float,
        save: bool,
        output_path: Optional[str],
        color_format: str,
        output_file_format: Optional[str],
    ) -> Results:
        image_path = image if isinstance(image, (str, Path)) else None
        pil = ImageLoader.load(image, color_format=color_format)
        rgb_np = np.asarray(pil)
        h, w = rgb_np.shape[:2]

        boxes, confs = self._collect_heads(
            rgb_np, detector, head_boxes, head_conf, head_expand
        )
        result = self._run_gazetarget(pil, boxes, confs, (h, w), image_path)

        if save:
            ext = (output_file_format or "jpg").lower().lstrip(".")
            save_path = resolve_save_path(output_path, image_path, ext=ext)
            self._save_annotated_image(result, pil, save_path)

        return result

    def _predict_video(
        self,
        source: Union[str, Path],
        *,
        detector: Optional[FaceDetector],
        head_conf: float,
        head_expand: float,
        save: bool,
        show: bool,
        vid_stride: int,
        output_path: Optional[str],
        output_file_format: Optional[str],
    ) -> Generator[Results, None, None]:
        def predict_frame(pil_img: Image.Image) -> Results:
            rgb_np = np.asarray(pil_img)
            h, w = rgb_np.shape[:2]
            boxes, confs = self._collect_heads(
                rgb_np, detector, None, head_conf, head_expand
            )
            return self._run_gazetarget(pil_img, boxes, confs, (h, w), str(source))

        def annotate(pil_img: Image.Image, result: Results) -> Image.Image:
            return self._annotate(pil_img, result)

        yield from run_video_inference(
            source,
            predict_frame,
            vid_stride=vid_stride,
            save=save,
            show=show,
            output_path=output_path,
            annotate_fn=annotate,
        )

    # =========================================================================
    # Internals
    # =========================================================================

    def _resolve_runtime_detector(
        self,
        explicit: Optional[FaceDetector],
        head_boxes: Optional[Sequence],
    ) -> Optional[FaceDetector]:
        if head_boxes is not None:
            return None
        if explicit is not None:
            return resolve_face_detector(explicit)
        if self.model.head_detector is not None:
            return self.model.head_detector
        detector = default_face_detector()
        logger.info(
            "No head detector provided; using OpenCV %s as a fallback and "
            "expanding face boxes into head boxes. Pass head_detector=... or "
            "head_boxes=[...] to control head localization.",
            type(detector).__name__,
        )
        self.model.head_detector = detector
        return self.model.head_detector

    def _collect_heads(
        self,
        image_rgb: np.ndarray,
        detector: Optional[FaceDetector],
        head_boxes: Optional[Sequence],
        head_conf: float,
        head_expand: float,
    ):
        h, w = image_rgb.shape[:2]
        if head_boxes is not None:
            boxes = []
            confs = []
            seq = list(head_boxes)
            if len(seq) == 4 and not any(isinstance(v, (list, tuple)) for v in seq):
                seq = [seq]
            for box in seq:
                values = [float(v) for v in box]
                if all(0.0 <= v <= 1.0 for v in values):
                    values = [values[0] * w, values[1] * h, values[2] * w, values[3] * h]
                pixel, _ = clamp_pixel_box(values, w, h)
                boxes.append(pixel)
                confs.append(1.0)
            return boxes, confs
        if detector is None:
            raise RuntimeError(
                "LibrePAGE has no head source. Pass head_boxes=[...] for BYO "
                "bboxes or head_detector=... (a callable, a LibreYOLO model, "
                "or a RetinaFaceAdapter) when constructing or calling the model."
            )
        faces = [f for f in detector(image_rgb) if f.score >= head_conf]
        boxes = crop_boxes_from_faces(faces, w, h, head_expand)
        confs = [float(f.score) for f in faces]
        return boxes, confs

    def _run_gazetarget(
        self,
        pil: Image.Image,
        pixel_boxes: List,
        confs: List[float],
        orig_shape: tuple,
        image_path: Optional[Union[str, Path]],
    ) -> Results:
        names = {0: "person"}
        h, w = orig_shape

        def _empty() -> Results:
            return Results(
                boxes=Boxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                ),
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=names,
                gazetarget=GazeTargets(
                    torch.zeros((0, 3), dtype=torch.float32),
                    orig_shape=orig_shape,
                ),
            )

        if not pixel_boxes:
            return _empty()

        norm_boxes = [
            (x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in pixel_boxes
        ]
        scene, heads = preprocess_scene_and_heads(pil, pixel_boxes)
        rects = head_rects_grid(norm_boxes)

        device = self.model.device
        with torch.no_grad():
            heatmap_logits, inout_logits = self.model.model(
                scene.to(device), heads.to(device), rects.to(device)
            )
        heatmaps = torch.sigmoid(heatmap_logits).cpu()
        inout = torch.sigmoid(inout_logits).cpu()

        points = decode_heatmaps(heatmaps, w, h)
        targets = torch.cat([points, inout.unsqueeze(1)], dim=1)

        xyxy = torch.tensor([list(b) for b in pixel_boxes], dtype=torch.float32)
        conf = torch.tensor(confs, dtype=torch.float32)
        cls = torch.zeros(len(pixel_boxes), dtype=torch.float32)
        return Results(
            boxes=Boxes(xyxy, conf, cls),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=names,
            gazetarget=GazeTargets(targets, heatmaps, orig_shape=orig_shape),
        )

    # =========================================================================
    # Rendering
    # =========================================================================

    def _annotate(self, pil_img: Image.Image, result: Results) -> Image.Image:
        if result.boxes is None or len(result.boxes) == 0:
            return pil_img
        from ...utils.drawing import draw_gaze_targets

        gt = result.gazetarget
        if gt is None or len(gt) == 0:
            return pil_img
        gt_np = gt.numpy() if isinstance(gt.data, torch.Tensor) else gt
        return draw_gaze_targets(
            pil_img,
            result.boxes.xyxy.tolist(),
            gt_np.xy.tolist(),
            inout=gt_np.inout.tolist(),
            heatmaps=gt_np.heatmaps,
        )

    def _save_annotated_image(
        self,
        result: Results,
        original_img: Image.Image,
        save_path: Path,
    ) -> None:
        annotated = self._annotate(original_img, result)
        annotated.save(save_path)
        log_saved_result(result, save_path)

    def _set_device(self, device: str) -> None:
        device_str = str(device).strip().lower()
        if device_str in ("", "auto"):
            return
        if device_str.isdigit():
            device_str = f"cuda:{device_str}"
        target = torch.device(device_str)
        if target != self.model.device:
            self.model.device = target
            if hasattr(self.model.model, "to"):
                self.model.model.to(target)
