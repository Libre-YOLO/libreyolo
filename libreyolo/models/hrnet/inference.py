"""Two-stage inference orchestration for top-down HRNet pose."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Sequence

import numpy as np
import torch
from PIL import Image

from ...postprocess.hrnet import flip_back_tensor, postprocess_hrnet
from ...utils.general import log_saved_result, resolve_save_path
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.results import Boxes, Keypoints, Results
from ...utils.video import collect_video_results, is_video_file, run_video_inference
from .detector import (
    LibreYOLOPersonDetector,
    PersonBox,
    PersonDetector,
    default_person_detector,
    normalize_person_boxes,
    resolve_person_detector,
)
from .utils import preprocess_box_numpy, size_hw

if TYPE_CHECKING:
    from .model import LibreHRNet

logger = logging.getLogger(__name__)


class HRNetPoseInferenceRunner:
    """Run person detection, affine crops, HRNet, and heatmap decoding."""

    def __init__(self, model: "LibreHRNet") -> None:
        self.model = model

    def __call__(
        self,
        source: ImageInput | Sequence[ImageInput] | None = None,
        *,
        person_boxes: Sequence | np.ndarray | None = None,
        person_detector: PersonDetector | object | None = None,
        cropped: bool = False,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        keypoint_threshold: float = 0.2,
        oks_threshold: float = 0.9,
        flip_test: bool = False,
        imgsz: int | Sequence[int] | None = None,
        classes: Sequence[int] | None = None,
        save: bool = False,
        output_path: str | None = None,
        color_format: str = "auto",
        stream: bool = False,
        vid_stride: int = 1,
        show: bool = False,
        output_file_format: str | None = None,
        device: str | None = None,
        augment: bool = False,
        tiling: bool = False,
        **_unused: object,
    ) -> Results | list[Results] | Generator[Results, None, None]:
        if augment:
            raise ValueError(
                "Use flip_test=True for HRNet's supported horizontal-flip test; "
                "generic augment=True is not defined for top-down pose."
            )
        if tiling:
            raise ValueError(
                "Tiled HRNet inference is not supported because person crops may "
                "cross tile boundaries. Pair HRNet with a tiled detector instead."
            )
        if cropped and person_boxes is not None:
            raise ValueError("cropped=True and person_boxes are mutually exclusive")
        if cropped and person_detector is not None:
            raise ValueError("cropped=True and person_detector are mutually exclusive")
        if not 0.0 <= float(conf) < 1.0:
            raise ValueError(f"conf must be in [0, 1), got {conf}")
        if not 0.0 < float(iou) <= 1.0:
            raise ValueError(f"iou must be in (0, 1], got {iou}")
        if not 0.0 <= float(keypoint_threshold) <= 1.0:
            raise ValueError(
                "keypoint_threshold must be in [0, 1], got "
                f"{keypoint_threshold}"
            )
        if not 0.0 <= float(oks_threshold) <= 1.0:
            raise ValueError(f"oks_threshold must be in [0, 1], got {oks_threshold}")
        if int(max_det) < 0:
            raise ValueError(f"max_det must be non-negative, got {max_det}")
        if imgsz is not None and size_hw(imgsz) != size_hw(self.model._get_input_size()):
            raise ValueError(
                f"HRNet {self.model.size} has fixed crop size "
                f"{self.model._get_input_size()}, got imgsz={imgsz!r}"
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

        include_person = classes is None or 0 in {int(value) for value in classes}
        detector = self._resolve_runtime_detector(
            person_detector,
            person_boxes=person_boxes,
            cropped=cropped,
            iou=float(iou),
            max_det=int(max_det),
        )

        if is_video_file(source):
            if person_boxes is not None:
                raise ValueError(
                    "Static person_boxes are ambiguous for video. Pass a "
                    "person_detector, or use cropped=True for person-only video."
                )
            generator = self._predict_video(
                source,
                detector=detector,
                cropped=cropped,
                conf=float(conf),
                max_det=int(max_det),
                keypoint_threshold=float(keypoint_threshold),
                oks_threshold=float(oks_threshold),
                flip_test=flip_test,
                include_person=include_person,
                save=save,
                show=show,
                vid_stride=vid_stride,
                output_path=output_path,
            )
            if stream:
                return generator
            return collect_video_results(generator, source, vid_stride)

        image_sources = None
        if isinstance(source, (list, tuple)):
            image_sources = list(source)
        elif isinstance(source, (str, Path)) and Path(source).is_dir():
            image_sources = ImageLoader.collect_images(source)

        if image_sources is not None:
            if person_boxes is not None:
                raise ValueError(
                    "person_boxes apply to one image only; use a person_detector "
                    "for multi-image inference."
                )
            return [
                self._predict_single(
                    image_path,
                    detector=detector,
                    person_boxes=None,
                    cropped=cropped,
                    conf=float(conf),
                    max_det=int(max_det),
                    keypoint_threshold=float(keypoint_threshold),
                    oks_threshold=float(oks_threshold),
                    flip_test=flip_test,
                    include_person=include_person,
                    save=save,
                    output_path=output_path,
                    color_format=color_format,
                    output_file_format=output_file_format,
                )
                for image_path in image_sources
            ]

        return self._predict_single(
            source,
            detector=detector,
            person_boxes=person_boxes,
            cropped=cropped,
            conf=float(conf),
            max_det=int(max_det),
            keypoint_threshold=float(keypoint_threshold),
            oks_threshold=float(oks_threshold),
            flip_test=flip_test,
            include_person=include_person,
            save=save,
            output_path=output_path,
            color_format=color_format,
            output_file_format=output_file_format,
        )

    def _resolve_runtime_detector(
        self,
        explicit: object | None,
        *,
        person_boxes: Sequence | np.ndarray | None,
        cropped: bool,
        iou: float,
        max_det: int,
    ) -> PersonDetector | None:
        if cropped or person_boxes is not None:
            return None
        if explicit is not None:
            detector = resolve_person_detector(explicit, device=str(self.model.device))
        elif self.model.person_detector is not None:
            detector = self.model.person_detector
        else:
            try:
                detector = default_person_detector(device=str(self.model.device))
            except Exception as error:
                raise RuntimeError(
                    "Automatic HRNet person-detector setup failed. Pass "
                    "person_boxes=[...], cropped=True, or person_detector=<callable "
                    "or LibreYOLO detector> to bypass the default LibreYOLO9t pair."
                ) from error
            logger.info(
                "No person detector provided; pairing HRNet with LibreYOLO9t. "
                "Pass person_boxes=... or person_detector=... to control this stage."
            )
            self.model.person_detector = detector
        if isinstance(detector, LibreYOLOPersonDetector):
            detector.iou = iou
            detector.max_det = max_det
        return detector

    def _predict_single(
        self,
        image: ImageInput,
        *,
        detector: PersonDetector | None,
        person_boxes: Sequence | np.ndarray | None,
        cropped: bool,
        conf: float,
        max_det: int,
        keypoint_threshold: float,
        oks_threshold: float,
        flip_test: bool,
        include_person: bool,
        save: bool,
        output_path: str | None,
        color_format: str,
        output_file_format: str | None,
    ) -> Results:
        image_path = image if isinstance(image, (str, Path)) else None
        pil_image = ImageLoader.load(image, color_format=color_format)
        image_rgb = np.asarray(pil_image)
        height, width = image_rgb.shape[:2]
        people = self._collect_people(
            image_rgb,
            detector=detector,
            person_boxes=person_boxes,
            cropped=cropped,
            conf=conf,
        )
        if not include_person:
            people = []
        result = self._run_pose(
            image_rgb,
            people,
            orig_shape=(height, width),
            image_path=image_path,
            max_det=max_det,
            keypoint_threshold=keypoint_threshold,
            oks_threshold=oks_threshold,
            flip_test=flip_test,
        )

        if save:
            extension = output_file_format or "jpg"
            save_path = resolve_save_path(
                output_path,
                image_path,
                ext=extension,
                default_dir="runs/pose",
            )
            annotated = self._annotate(
                pil_image,
                result,
                keypoint_threshold=keypoint_threshold,
            )
            annotated.save(save_path)
            log_saved_result(result, save_path)
        return result

    def _predict_video(
        self,
        source: str | Path,
        *,
        detector: PersonDetector | None,
        cropped: bool,
        conf: float,
        max_det: int,
        keypoint_threshold: float,
        oks_threshold: float,
        flip_test: bool,
        include_person: bool,
        save: bool,
        show: bool,
        vid_stride: int,
        output_path: str | None,
    ) -> Generator[Results, None, None]:
        def predict_frame(pil_image: Image.Image) -> Results:
            image_rgb = np.asarray(pil_image)
            height, width = image_rgb.shape[:2]
            people = self._collect_people(
                image_rgb,
                detector=detector,
                person_boxes=None,
                cropped=cropped,
                conf=conf,
            )
            if not include_person:
                people = []
            return self._run_pose(
                image_rgb,
                people,
                orig_shape=(height, width),
                image_path=source,
                max_det=max_det,
                keypoint_threshold=keypoint_threshold,
                oks_threshold=oks_threshold,
                flip_test=flip_test,
            )

        def annotate(pil_image: Image.Image, result: Results) -> Image.Image:
            return self._annotate(
                pil_image,
                result,
                keypoint_threshold=keypoint_threshold,
            )

        yield from run_video_inference(
            source,
            predict_frame,
            vid_stride=vid_stride,
            save=save,
            show=show,
            output_path=output_path,
            annotate_fn=annotate,
        )

    @staticmethod
    def _collect_people(
        image_rgb: np.ndarray,
        *,
        detector: PersonDetector | None,
        person_boxes: Sequence | np.ndarray | None,
        cropped: bool,
        conf: float,
    ) -> list[PersonBox]:
        if cropped:
            height, width = image_rgb.shape[:2]
            return [PersonBox((0.0, 0.0, float(width), float(height)), 1.0)]
        if person_boxes is not None:
            return normalize_person_boxes(person_boxes, min_score=conf)
        if detector is None:
            raise RuntimeError(
                "HRNet has no person source. Pass person_boxes=[...], "
                "cropped=True, or person_detector=..."
            )
        return [person for person in detector(image_rgb) if person.score >= conf]

    def _run_pose(
        self,
        image_rgb: np.ndarray,
        people: list[PersonBox],
        *,
        orig_shape: tuple[int, int],
        image_path: str | Path | None,
        max_det: int,
        keypoint_threshold: float,
        oks_threshold: float,
        flip_test: bool,
    ) -> Results:
        if not people or max_det == 0:
            return self._empty_result(orig_shape, image_path)

        people = sorted(people, key=lambda person: person.score, reverse=True)[:max_det]
        tensors: list[np.ndarray] = []
        centers: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        kept_people: list[PersonBox] = []
        for person in people:
            try:
                tensor, center, scale = preprocess_box_numpy(
                    image_rgb,
                    person.xyxy,
                    self.model._get_input_size(),
                )
            except ValueError as error:
                logger.warning("Skipping degenerate HRNet person box: %s", error)
                continue
            tensors.append(tensor)
            centers.append(center)
            scales.append(scale)
            kept_people.append(person)
        if not tensors:
            return self._empty_result(orig_shape, image_path)

        batch = torch.from_numpy(np.stack(tensors)).to(self.model.device)
        with torch.inference_mode():
            heatmaps = self.model.model(batch)
            if flip_test:
                flipped = self.model.model(batch.flip(3))
                flipped = flip_back_tensor(flipped, shift=True)
                heatmaps = (heatmaps + flipped) * 0.5

        decoded = postprocess_hrnet(
            heatmaps,
            centers=np.stack(centers),
            scales=np.stack(scales),
            boxes=np.asarray([person.xyxy for person in kept_people], dtype=np.float32),
            box_scores=np.asarray(
                [person.score for person in kept_people],
                dtype=np.float32,
            ),
            keypoint_threshold=keypoint_threshold,
            oks_threshold=oks_threshold,
            max_det=max_det,
        )
        boxes = torch.from_numpy(np.ascontiguousarray(decoded["boxes"])).float()
        scores = torch.from_numpy(np.ascontiguousarray(decoded["scores"])).float()
        classes = torch.from_numpy(np.ascontiguousarray(decoded["classes"])).float()
        keypoints = torch.from_numpy(
            np.ascontiguousarray(decoded["keypoints"])
        ).float()
        return Results(
            boxes=Boxes(boxes, scores, classes, orig_shape=orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path is not None else None,
            names={0: "person"},
            keypoints=Keypoints(keypoints, orig_shape),
        )

    @staticmethod
    def _empty_result(
        orig_shape: tuple[int, int],
        image_path: str | Path | None,
    ) -> Results:
        return Results(
            boxes=Boxes(
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                orig_shape=orig_shape,
            ),
            orig_shape=orig_shape,
            path=str(image_path) if image_path is not None else None,
            names={0: "person"},
            keypoints=Keypoints(
                torch.zeros((0, 17, 3), dtype=torch.float32),
                orig_shape,
            ),
        )

    @staticmethod
    def _annotate(
        image: Image.Image,
        result: Results,
        *,
        keypoint_threshold: float,
    ) -> Image.Image:
        if result.boxes is None or len(result.boxes) == 0:
            return image.copy()
        from ...utils.drawing import draw_boxes, draw_keypoints

        annotated = draw_boxes(
            image,
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
            class_names=result.names,
        )
        keypoints = result.keypoints.data
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        return draw_keypoints(
            annotated,
            keypoints,
            conf_thres=keypoint_threshold,
        )

    def _set_device(self, device: str) -> None:
        device_string = str(device).strip().lower()
        if device_string in ("", "auto"):
            return
        if device_string.isdigit():
            device_string = f"cuda:{device_string}"
        target = torch.device(device_string)
        if target != self.model.device:
            self.model.device = target
            self.model.model.to(target)
