"""Base class for LibreYOLO inference backends."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..models.yolo9.utils import (
    _YOLO9_MAX_NMS_CANDIDATES,
    postprocess as yolo9_postprocess,
    preprocess_image,
)
from ..models.yolonas.utils import (
    YOLO_NAS_PRE_NMS_TOP_K,
    YOLO_NAS_POSE_RESIZE_SIZE,
    YOLO_NAS_RESIZE_SIZE,
    preprocess_image as yolonas_preprocess_image,
    preprocess_pose_image as yolonas_preprocess_pose_image,
)
from ..models.yolox.utils import preprocess_image as yolox_preprocess_image
from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from ..utils.drawing import (
    draw_boxes,
    draw_depth_map,
    draw_gaze_arrows,
    draw_keypoints,
    draw_masks,
    draw_obb,
    draw_ocr_regions,
    draw_panoptic,
    draw_points,
    draw_semantic_mask,
)
from ..utils.general import (
    COCO_CLASSES,
    get_safe_stem,
    log_saved_result,
    resolve_save_path,
    save_path_write_guard,
)
from ..utils.image_loader import ImageLoader
from ..utils.model_info import build_model_info, format_model_info
from ..utils.predict_args import normalize_predict_kwargs, validate_predict_inputs
from ..utils.results import (
    Boxes,
    DepthMap,
    Gaze,
    Keypoints,
    Matte,
    Masks,
    OBB,
    Points,
    Probs,
    Results,
    RestoredImage,
    SemanticMask,
)
from ..utils.video import collect_video_results, is_video_file, run_video_inference

logger = logging.getLogger(__name__)

ImageSize = Union[int, Tuple[int, int]]
_RECTANGULAR_BACKEND_FAMILIES = {
    "yolo9",
    "yolo9_e2e",
    "yolo9_p2",
    "nafnet",
    "realesrgan",
}

# Real-ESRGAN integer upscale factor per size, used by scale-aware restore decode.
_REALESRGAN_BACKEND_SCALE = {"x4": 4, "x2": 2, "x4t": 4}
_SWINIR_BACKEND_SCALE = {"s": 4, "m": 4, "l": 4}
_REALESRGAN_BACKEND_PAD_MULTIPLE = {"x4": 1, "x2": 2, "x4t": 1}

# Families removed from LibreYOLO. An exported artifact whose metadata still names
# one of these must fail loudly instead of being silently parsed as YOLO9.
_REMOVED_FAMILIES = {"damoyolo"}


class _BackendEvalProxy:
    def eval(self):
        return self


def _imgsz_hw(imgsz: ImageSize) -> Tuple[int, int]:
    if isinstance(imgsz, tuple):
        if len(imgsz) != 2:
            raise ValueError(f"imgsz must be int or (height, width), got {imgsz}")
        h, w = int(imgsz[0]), int(imgsz[1])
    else:
        h = w = int(imgsz)
    if h <= 0 or w <= 0:
        raise ValueError(f"imgsz values must be positive, got {(h, w)}")
    return h, w


def _normalize_imgsz(imgsz: ImageSize) -> ImageSize:
    h, w = _imgsz_hw(imgsz)
    return h if h == w else (h, w)


def _is_rectangular_imgsz(imgsz: ImageSize) -> bool:
    h, w = _imgsz_hw(imgsz)
    return h != w


class MetadataImageSizeError(ValueError):
    """Raised when exported input-size metadata is malformed."""


def _read_metadata_imgsz(
    meta: dict,
    model_family: Optional[str],
    *,
    artifact: str,
) -> ImageSize | None:
    """Read exported-runtime input size metadata.

    ``imgsz`` stays as the legacy square scalar. ``imgsz_h``/``imgsz_w`` are
    only allowed to describe rectangular runtime inputs for backend families
    that explicitly support them.
    """
    has_imgsz_h = "imgsz_h" in meta
    has_imgsz_w = "imgsz_w" in meta
    if has_imgsz_h != has_imgsz_w:
        raise MetadataImageSizeError(
            f"{artifact} must define both imgsz_h and imgsz_w, or neither."
        )

    if has_imgsz_h and has_imgsz_w:
        try:
            imgsz = _normalize_imgsz((int(meta["imgsz_h"]), int(meta["imgsz_w"])))
        except (TypeError, ValueError) as e:
            raise MetadataImageSizeError(
                f"{artifact} has invalid imgsz_h/imgsz_w metadata."
            ) from e
        if (
            _is_rectangular_imgsz(imgsz)
            and (model_family or "").lower() not in _RECTANGULAR_BACKEND_FAMILIES
        ):
            raise NotImplementedError(
                "Rectangular exported-backend inference is currently supported "
                "for YOLO9-family and NAFNet exports only. "
                f"{artifact} declares model_family={model_family or 'unknown'!r}."
            )
        return imgsz

    if "imgsz" in meta:
        try:
            return _normalize_imgsz(int(meta["imgsz"]))
        except (TypeError, ValueError) as e:
            raise MetadataImageSizeError(
                f"{artifact} has invalid imgsz metadata."
            ) from e

    return None


def _read_pose_metadata(meta: dict) -> dict[str, Any]:
    """Extract shared pose metadata from embedded or sidecar export metadata."""
    pose_meta: dict[str, Any] = {}
    if "num_keypoints" in meta:
        pose_meta["num_keypoints"] = int(meta["num_keypoints"])
    if "keypoint_dim" in meta:
        pose_meta["keypoint_dim"] = int(meta["keypoint_dim"])
    if "num_keypoints_per_class" in meta:
        raw_schema = meta["num_keypoints_per_class"]
        if isinstance(raw_schema, str):
            raw_schema = json.loads(raw_schema)
        if raw_schema is not None:
            pose_meta["num_keypoints_per_class"] = [int(count) for count in raw_schema]
    return pose_meta


def _nms_numpy(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45
) -> list:
    """Numpy-based Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return keep


def _batched_nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
) -> list:
    """Class-aware NMS matching torchvision.ops.batched_nms ordering."""
    keep = []
    for cls in np.unique(class_ids):
        cls_indices = np.where(class_ids == cls)[0]
        cls_keep = _nms_numpy(boxes[cls_indices], scores[cls_indices], iou_threshold)
        keep.extend(cls_indices[cls_keep].tolist())

    if not keep:
        return []
    keep = np.asarray(keep, dtype=np.int64)
    return keep[np.argsort(scores[keep])[::-1]].tolist()


def _is_pytorch_cuda_device(device_str: str) -> bool:
    """Return True only when device_str is a valid PyTorch CUDA device string.

    Non-PyTorch runtimes (OpenVINO "gpu", CoreML "coreml", ncnn "ncnn") store
    backend-specific device identifiers in self.device that are not parseable
    by torch.device(); calling torch.device() on them raises RuntimeError.
    """
    try:
        return torch.device(device_str).type == "cuda"
    except RuntimeError:
        return False


def _is_nms_free_family(model_family: Optional[str]) -> bool:
    """Whether backend outputs should bypass generic NMS.

    DETR-style families already emit a ranked set prediction after top-k
    selection. Applying YOLO-style IoU suppression on top of that can remove
    valid detections and make exported runtimes diverge from native PyTorch.
    """
    return model_family in {
        "dfine",
        "deim",
        "deimv2",
        "ec",
        "rfdetr",
        "rtdetr",
        "rtdetrv2",
        "rtdetrv4",
        "yolo9_e2e",
    }


def _rfdetr_num_select(task: str, model_size: Optional[str]) -> int:
    """Return RF-DETR's configured top-k selection for exported backends."""
    if task == "segment":
        return {"n": 100, "s": 100, "m": 200, "l": 200}.get(model_size or "", 300)
    if task == "pose" and model_size == "x":
        return 100
    return 300


def _logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(max_values, axis=axis) + np.log(
        np.sum(np.exp(values - max_values), axis=axis)
    )


def _rfdetr_keypoint_log_mean_trace_np(active_keypoints: np.ndarray) -> np.ndarray:
    log_l11 = active_keypoints[..., 4]
    l21 = active_keypoints[..., 5]
    log_l22 = active_keypoints[..., 6]
    w_find = 1.0 / (1.0 + np.exp(-active_keypoints[..., 2]))
    log_t1 = -2.0 * log_l11
    log_t2 = -2.0 * log_l22
    log_t3 = 2.0 * np.log(np.clip(np.abs(l21), 1e-12, None)) + log_t1 + log_t2
    log_trace_sigma = _logsumexp_np(
        np.stack([log_t1, log_t2, log_t3], axis=-1),
        axis=-1,
    )
    log_w_find = np.log(np.clip(w_find, 1e-12, None))
    return _logsumexp_np(log_trace_sigma + log_w_find, axis=-1) - _logsumexp_np(
        log_w_find,
        axis=-1,
    )


class BaseBackend(ABC):
    """Abstract base class for all inference backends.

    Subclasses must:
    1. Implement ``__init__`` to load the runtime-specific model, then call
       ``super().__init__(...)`` with the resolved common attributes.
    2. Implement ``_run_inference`` to execute the model and return raw outputs.
    """

    def __init__(
        self,
        *,
        model_path: str,
        nb_classes: int,
        device: str,
        imgsz: ImageSize,
        model_family: Optional[str],
        names: Dict[int, str],
        model_size: Optional[str] = None,
        task: str | None = None,
        supported_tasks=None,
        default_task: str | None = None,
        crop_pct: float | None = None,
        interpolation: str | None = None,
        num_keypoints: int | None = None,
        keypoint_dim: int | None = None,
        num_keypoints_per_class: list[int] | None = None,
        num_bins: int | None = None,
        bin_width_deg: float | None = None,
        offset_deg: float | None = None,
    ):
        self.model_path = model_path
        self.nb_classes = nb_classes
        self.device = device
        self.imgsz = _normalize_imgsz(imgsz)
        self.model_family = model_family
        self.family = model_family
        # DAMO-YOLO was removed; reject its exported artifacts loudly instead of
        # silently mis-parsing them as YOLO9 (DAMO used different pre/post-processing).
        if model_family in _REMOVED_FAMILIES:
            raise ValueError(
                f"model_family={model_family!r} is no longer supported: the "
                f"{model_family} family was removed from LibreYOLO. Re-export this "
                "model with a supported family, or pin an older LibreYOLO release "
                "to run an existing export."
            )
        self.model_size = model_size
        self.DEFAULT_TASK = normalize_task(default_task, default="detect")
        self.SUPPORTED_TASKS = normalize_supported_tasks(
            supported_tasks or (self.DEFAULT_TASK,)
        )
        self.task = resolve_task(
            explicit_task=task,
            default_task=self.DEFAULT_TASK,
            supported_tasks=self.SUPPORTED_TASKS,
        )
        if self.model_family == "yolo9" and self.task == "segment":
            raise NotImplementedError(
                "YOLO9 segmentation support was removed. Use a supported "
                "segmentation family instead of loading YOLO9 segment exports."
            )
        self.names = names
        self.FAMILY = model_family or "export"
        try:
            self.size = model_size or "export"
        except AttributeError:
            # Some concrete backends expose size as a computed read-only property.
            pass
        self.input_size = self.imgsz
        # Classification eval preprocessing (from export metadata); defaults keep
        # legacy behavior. Lets exported-backend classify inference match native.
        self.crop_pct = crop_pct if crop_pct is not None else 0.875
        self.interpolation = interpolation or "bilinear"
        # Set by backends that load a model with NMS baked into the graph; such
        # models emit final (1, max_det, 6) detections instead of raw tensors.
        if not hasattr(self, "embedded_nms"):
            self.embedded_nms = False
        if not hasattr(self, "embedded_nms_raw_output_index"):
            self.embedded_nms_raw_output_index = None
        if num_keypoints is not None:
            self.num_keypoints = int(num_keypoints)
        if keypoint_dim is not None:
            self.keypoint_dim = int(keypoint_dim)
        if num_keypoints_per_class is not None:
            self.num_keypoints_per_class = [
                int(count) for count in num_keypoints_per_class
            ]
        self.num_bins = int(num_bins if num_bins is not None else 90)
        self.bin_width_deg = float(bin_width_deg if bin_width_deg is not None else 4.0)
        self.offset_deg = float(offset_deg if offset_deg is not None else -180.0)
        if not hasattr(self, "model"):
            self.model = _BackendEvalProxy()

    # =========================================================================
    # Abstract interface
    # =========================================================================

    @abstractmethod
    def _run_inference(self, blob: np.ndarray) -> list:
        """Run backend-specific inference.

        Args:
            blob: Preprocessed input array of shape ``(1, C, H, W)``.

        Returns:
            List of numpy arrays, one per model output tensor.
        """

    # =========================================================================
    # Preprocessing
    # =========================================================================

    def _preprocess(self, image, effective_imgsz, color_format):
        """Dispatch to model-family-specific preprocessing.

        Returns:
            Tuple of (input_tensor, original_img, original_size, ratio).
        """
        if self.task == "restore" or self.model_family == "nafnet":
            if self.model_family == "realesrgan":
                return self._preprocess_restore_native(image, color_format)
            return self._preprocess_restore(image, effective_imgsz, color_format)
        if self.task == "depth":
            return self._preprocess_depth(image, effective_imgsz, color_format)
        if self.task == "matte":
            return self._preprocess_matte(image, effective_imgsz, color_format)
        if self.task == "gaze":
            return self._preprocess_gaze(image, effective_imgsz, color_format)
        if self.task == "classify":
            return self._preprocess_classify(image, effective_imgsz, color_format)
        if self.task == "point" and self.model_family == "fomo":
            from ..models.fomo.utils import preprocess_image as fomo_preprocess_image

            h, w = _imgsz_hw(effective_imgsz)
            if h != w:
                raise NotImplementedError(
                    "FOMO exported inference requires square imgsz."
                )
            return fomo_preprocess_image(image, h, color_format=color_format)
        if self.task == "semantic":
            return self._preprocess_semantic(image, effective_imgsz, color_format)
        if self.model_family == "yolox":
            return yolox_preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )
        elif self.model_family == "yolonas":
            if self.task == "pose":
                return yolonas_preprocess_pose_image(
                    image, input_size=effective_imgsz, color_format=color_format
                )
            return yolonas_preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )
        elif self.model_family == "rfdetr":
            tensor, img, size = self._preprocess_rfdetr(
                image,
                effective_imgsz,
                color_format,
                task=self.task,
            )
            return tensor, img, size, 1.0
        elif self.model_family in ("dfine", "rtdetrv4"):
            tensor, img, size = self._preprocess_dfine(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family == "deim":
            tensor, img, size = self._preprocess_deim(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family == "deimv2":
            tensor, img, size = self._preprocess_deimv2(
                image, effective_imgsz, color_format, self.model_size
            )
            return tensor, img, size, 1.0
        elif self.model_family == "ec":
            tensor, img, size = self._preprocess_ec(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family in ("rtdetr", "rtdetrv2"):
            tensor, img, size = self._preprocess_rtdetr(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family == "picodet":
            tensor, img, size = self._preprocess_picodet(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family == "rtmdet":
            tensor, img, size, ratio = self._preprocess_rtmdet(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, ratio
        elif self.model_family in ("yolo2", "yolo3", "yolo4"):
            from ..models.darknet.preprocess import preprocess_image as _dk_pre

            sz = (
                effective_imgsz
                if isinstance(effective_imgsz, int)
                else max(effective_imgsz)
            )
            return _dk_pre(image, input_size=sz, color_format=color_format)
        elif self.model_family == "yolo7":
            from ..models.yolo7.utils import preprocess_image as _y7_pre

            sz = (
                effective_imgsz
                if isinstance(effective_imgsz, int)
                else max(effective_imgsz)
            )
            return _y7_pre(image, input_size=sz, color_format=color_format)
        else:
            tensor, img, size = preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )
            return tensor, img, size, 1.0

    def _preprocess_classify(self, image, input_size, color_format):
        """Classification preprocessing: ImageNet-style resize/crop/normalize.

        Uses the per-family ``crop_pct``/``interpolation`` recorded in export
        metadata so exported-backend inference matches native predict()/val().
        """
        from ..data.classify_dataset import build_classify_transforms

        h, w = _imgsz_hw(input_size)
        if h != w:
            raise NotImplementedError(
                "Classification exported-backend inference supports square imgsz only."
            )

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        transform = build_classify_transforms(
            h,
            augment=False,
            crop_pct=getattr(self, "crop_pct", 0.875),
            interpolation=getattr(self, "interpolation", "bilinear"),
        )
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor, img, original_size, 1.0

    def _preprocess_semantic(self, image, input_size, color_format):
        """Dense semantic preprocessing for fixed-canvas exported graphs."""
        input_h, input_w = _imgsz_hw(input_size)
        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()
        arr = np.asarray(img.convert("RGB"))
        if self.model_family == "pidnet":
            from ..models.pidnet.model import preprocess_numpy

            chw, ratio = preprocess_numpy(arr, (input_h, input_w))
        else:
            resized = cv2.resize(
                arr, (input_w, input_h), interpolation=cv2.INTER_LINEAR
            )
            chw = np.ascontiguousarray(
                resized.astype(np.float32).transpose(2, 0, 1) / 255.0
            )
            ratio = 1.0
        return (
            torch.from_numpy(chw).unsqueeze(0).float(),
            original_img,
            original_size,
            ratio,
        )

    @staticmethod
    def _preprocess_matte(image, input_size, color_format):
        """BiRefNet fixed-canvas ImageNet-normalized matte preprocessing."""
        from ..models.birefnet.utils import preprocess_numpy

        input_h, input_w = _imgsz_hw(input_size)
        if input_h != input_w:
            raise NotImplementedError(
                "Matte exported-runtime inference requires square imgsz."
            )
        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        chw, ratio = preprocess_numpy(np.asarray(img.convert("RGB")), input_h)
        return (
            torch.from_numpy(chw).unsqueeze(0).float(),
            img.copy(),
            original_size,
            ratio,
        )

    @staticmethod
    def _preprocess_gaze(image, input_size, color_format):
        """Preprocess one already-cropped face for the L2CS gaze head."""
        from ..models.l2cs.utils import preprocess_face_crops

        input_h, input_w = _imgsz_hw(input_size)
        if (input_h, input_w) != (448, 448):
            raise ValueError(
                "L2CS exported inference requires the fixed 448x448 contract."
            )
        img = ImageLoader.load(image, color_format=color_format)
        return preprocess_face_crops([img]), img.copy(), img.size, 1.0

    @staticmethod
    def _preprocess_restore(image, input_size, color_format):
        """Restoration preprocessing for fixed-shape exported runtimes.

        Native NAFNet prediction runs at the input image's own resolution and
        reflect-pads only to the network stride. Exported runtimes use a fixed
        graph shape, so backend prediction accepts images that fit inside the
        exported canvas, pads bottom/right without resizing, and crops the
        restored output back to the original canvas.
        """
        input_h, input_w = _imgsz_hw(input_size)
        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()
        orig_w, orig_h = original_size
        if orig_h > input_h or orig_w > input_w:
            raise ValueError(
                "Restoration exported-runtime inference is fixed-resolution. "
                f"Input image is {orig_w}x{orig_h}, but the exported canvas is "
                f"{input_w}x{input_h}. Use a native .pt model for native-size "
                "large-image prediction, or export a matching fixed size."
            )

        arr = np.asarray(img, dtype=np.float32) / 255.0
        pad_h = input_h - orig_h
        pad_w = input_w - orig_w
        if pad_h or pad_w:
            mode = (
                "reflect"
                if orig_h > 1 and orig_w > 1 and pad_h < orig_h and pad_w < orig_w
                else "edge"
            )
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
        img_tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return img_tensor.unsqueeze(0).float(), original_img, original_size, 1.0

    @staticmethod
    def _preprocess_depth(image, input_size, color_format):
        """Depth preprocessing for fixed-shape exported runtimes.

        Native depth prediction keeps the aspect ratio (short side to the
        model's native resolution). Exported runtimes use a fixed graph shape,
        so backend prediction stretch-resizes to the exported canvas and the
        depth map is resized back to the original canvas after inference
        (ADR 0006). Padding is deliberately avoided: padded pixels would leak
        fake depth context into real pixels through the receptive field.
        """
        input_h, input_w = _imgsz_hw(input_size)
        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()
        arr = np.asarray(img, dtype=np.uint8)
        resized = cv2.resize(arr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        img_tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
        return img_tensor, original_img, original_size, 1.0

    @property
    def restore_scale(self) -> int:
        """Integer upscale factor for restore backends (1 unless super-resolution)."""

        if self.model_family == "realesrgan":
            return _REALESRGAN_BACKEND_SCALE.get(str(self.model_size), 1)
        if self.model_family == "swinir":
            return _SWINIR_BACKEND_SCALE.get(str(self.model_size), 1)
        return 1

    def _preprocess_restore_native(self, image, color_format):
        """Native-resolution restore preprocessing for dynamic Real-ESRGAN graphs.

        Loads RGB [0, 1], reflect-pads bottom/right to the network divisibility
        factor (2 for the x2 pixel-unshuffle variant, 1 otherwise). The dynamic
        ONNX graph accepts any spatial size, so no fixed canvas is imposed.
        """

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()
        arr = np.asarray(img, dtype=np.float32) / 255.0
        multiple = _REALESRGAN_BACKEND_PAD_MULTIPLE.get(str(self.model_size), 1)
        if multiple > 1:
            orig_h, orig_w = arr.shape[:2]
            pad_h = (multiple - orig_h % multiple) % multiple
            pad_w = (multiple - orig_w % multiple) % multiple
            if pad_h or pad_w:
                mode = (
                    "reflect"
                    if orig_h > 1 and orig_w > 1 and pad_h < orig_h and pad_w < orig_w
                    else "edge"
                )
                arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
        img_tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return img_tensor.unsqueeze(0).float(), original_img, original_size, 1.0

    @staticmethod
    def _preprocess_rfdetr(image, input_size, color_format, task=None):
        """RF-DETR preprocessing: direct resize + ImageNet normalization."""
        from ..models.rfdetr.utils import (
            IMAGENET_MEAN,
            IMAGENET_STD,
            preprocess_numpy as rfdetr_preprocess_numpy,
        )

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size  # (W, H)
        original_img = img.copy()

        if task == "pose":
            h, w = _imgsz_hw(input_size)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            img_tensor = F.interpolate(
                img_tensor,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
            return (img_tensor - mean) / std, original_img, original_size

        img_chw, _ = rfdetr_preprocess_numpy(np.array(img), input_size)
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_dfine(image, input_size, color_format):
        """D-FINE preprocessing: plain resize + RGB + /255, no ImageNet norm."""
        from ..models.dfine.utils import preprocess_numpy as dfine_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = dfine_preprocess_numpy(np.array(img), input_size)
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0)

        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_deim(image, input_size, color_format):
        """DEIM-D-FINE preprocessing: plain resize + RGB + /255."""
        from ..models.deim.utils import preprocess_numpy as deim_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = deim_preprocess_numpy(np.array(img), input_size)
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0)

        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_deimv2(image, input_size, color_format, model_size=None):
        """DEIMv2 preprocessing; DINO-backed sizes use ImageNet normalization."""
        from ..models.deimv2.nn import DINO_SIZES
        from ..models.deimv2.utils import preprocess_numpy as deimv2_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = deimv2_preprocess_numpy(
            np.array(img), input_size, imagenet_norm=model_size in DINO_SIZES
        )
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0)

        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_ec(image, input_size, color_format):
        """EC preprocessing: plain resize + RGB + /255 + ImageNet (mean, std)."""
        from ..models.ec.postprocess import (
            preprocess_numpy as ec_preprocess_numpy,
        )

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = ec_preprocess_numpy(np.array(img), input_size)
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_picodet(image, input_size, color_format):
        """PICODET preprocessing: simple resize + RGB + ImageNet mean/std (0-255 space)."""
        from ..models.picodet.utils import preprocess_numpy as picodet_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = picodet_preprocess_numpy(np.array(img), input_size)
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_rtmdet(image, input_size, color_format):
        """RTMDet preprocessing: BGR letterbox + mmdet mean/std normalization."""
        from ..models.rtmdet.utils import preprocess_numpy as rtmdet_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size  # (W, H)
        original_img = img.copy()

        img_chw, ratio = rtmdet_preprocess_numpy(np.array(img), input_size)
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
        return img_tensor, original_img, original_size, ratio

    @staticmethod
    def _preprocess_rtdetr(image, input_size, color_format):
        """RT-DETR preprocessing: direct resize + normalize to [0,1]."""
        from ..models.rtdetr.utils import preprocess_numpy as rtdetr_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size  # (W, H)
        original_img = img.copy()

        img_chw, _ = rtdetr_preprocess_numpy(np.array(img), input_size)
        img_tensor = torch.from_numpy(img_chw)
        return img_tensor, original_img, original_size

    # =========================================================================
    # Output parsing
    # =========================================================================

    def _parse_outputs(
        self,
        all_outputs: list,
        effective_imgsz: ImageSize,
        original_size: tuple,
        conf: float,
        ratio: float | None = None,
        iou: float = 0.45,
        max_det: int = 300,
    ):
        """Parse raw outputs into boxes, scores, classes, masks, OBB, and keypoints."""
        orig_w, orig_h = original_size

        if getattr(self, "embedded_nms", False):
            raw_index = getattr(self, "embedded_nms_raw_output_index", None)
            if (
                self.model_family == "yolo9"
                and isinstance(raw_index, int)
                and raw_index < len(all_outputs)
            ):
                boxes, scores, cls = self._parse_yolo9(
                    [all_outputs[raw_index]],
                    effective_imgsz,
                    orig_w,
                    orig_h,
                    conf,
                    iou=iou,
                    max_det=max_det,
                )
                return boxes, scores, cls, None
            boxes, scores, cls = self._parse_embedded_nms(
                all_outputs, effective_imgsz, orig_w, orig_h, conf
            )
            return boxes, scores, cls, None

        if self.model_family == "yolox":
            boxes, scores, cls = self._parse_yolox(
                all_outputs, effective_imgsz, orig_w, orig_h, conf, ratio
            )
            return boxes, scores, cls, None
        elif self.model_family == "yolonas":
            if self.task == "pose":
                return self._parse_yolonas_pose(
                    all_outputs,
                    effective_imgsz,
                    orig_w,
                    orig_h,
                    conf,
                    ratio=ratio,
                    max_det=max_det,
                )
            boxes, scores, cls = self._parse_yolonas(
                all_outputs, effective_imgsz, orig_w, orig_h, conf, ratio=ratio
            )
            return boxes, scores, cls, None
        elif self.model_family == "rfdetr":
            return self._parse_rfdetr(
                all_outputs,
                orig_w,
                orig_h,
                conf,
                max_det=max_det,
            )
        elif self.model_family in ("dfine", "rtdetrv4"):
            if self.model_family == "dfine" and self.task == "segment":
                return self._parse_dfine_segment(
                    all_outputs, orig_w, orig_h, conf, max_det=max_det
                )
            boxes, scores, cls = self._parse_dfine(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family == "deim":
            boxes, scores, cls = self._parse_dfine(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family == "deimv2":
            boxes, scores, cls = self._parse_dfine(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family == "ec":
            if self.task == "segment":
                return self._parse_ec_segment(
                    all_outputs, orig_w, orig_h, conf, max_det=max_det
                )
            if self.task == "pose":
                return self._parse_ec_pose(
                    all_outputs, orig_w, orig_h, conf, max_det=max_det
                )
            boxes, scores, cls = self._parse_dfine(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family in ("rtdetr", "rtdetrv2"):
            boxes, scores, cls = self._parse_rtdetr(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family == "picodet":
            boxes, scores, cls = self._parse_picodet(
                all_outputs, effective_imgsz, orig_w, orig_h, conf
            )
            return boxes, scores, cls, None
        elif self.model_family == "rtmdet":
            boxes, scores, cls = self._parse_rtmdet(
                all_outputs, effective_imgsz, orig_w, orig_h, conf, ratio
            )
            return boxes, scores, cls, None
        else:
            parsed = self._parse_yolo9(
                all_outputs, effective_imgsz, orig_w, orig_h, conf, iou, max_det
            )
            if len(parsed) == 6:
                return parsed
            if len(parsed) == 5:
                return parsed
            if len(parsed) == 4:
                return parsed
            boxes, scores, cls = parsed
            return boxes, scores, cls, None

    def _parse_yolox(
        self, all_outputs, effective_imgsz, orig_w, orig_h, conf, ratio=1.0
    ):
        """Parse YOLOX output: (B, N, 5+nc) — cxcywh + objectness + class_scores."""
        outputs = all_outputs[0][0]  # (N, 5+nc)

        cx, cy, w, h = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
        objectness = outputs[:, 4]
        class_scores = outputs[:, 5:]

        max_class_scores = np.max(class_scores, axis=1)
        max_scores = objectness * max_class_scores
        class_ids = np.argmax(class_scores, axis=1)

        mask = max_scores > conf
        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        max_scores, class_ids = max_scores[mask], class_ids[mask]

        if len(max_scores) == 0:
            return np.empty((0, 4)), max_scores, class_ids

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        if ratio is None or ratio == 1.0:
            input_h, input_w = _imgsz_hw(effective_imgsz)
            ratio = min(input_h / orig_h, input_w / orig_w)
        boxes /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_boxes]
        max_scores = max_scores[valid_boxes]
        class_ids = class_ids[valid_boxes]

        return boxes, max_scores, class_ids

    def _parse_rtmdet(
        self, all_outputs, effective_imgsz, orig_w, orig_h, conf, ratio=1.0
    ):
        """Parse RTMDet export-mode output: (B, N, 4 + nc) — xyxy (input-canvas pixels) + sigmoid scores.

        RTMDet exports use letterbox preprocessing, so the inverse scale is a
        single ``ratio`` (aspect-preserving), like YOLOX.
        """
        outputs = all_outputs[0][0]  # (N, 4 + nc)
        boxes_all = outputs[:, :4]
        scores = outputs[:, 4:]

        valid = scores > conf
        if not valid.any():
            return (
                np.empty((0, 4), dtype=boxes_all.dtype),
                np.empty((0,), dtype=scores.dtype),
                np.empty((0,), dtype=np.int64),
            )

        box_indices, class_ids = np.nonzero(valid)
        max_scores = scores[box_indices, class_ids]

        input_h, input_w = _imgsz_hw(effective_imgsz)
        strides = (8, 16, 32)
        level_sizes = [
            int(np.ceil(input_h / stride)) * int(np.ceil(input_w / stride))
            for stride in strides
        ]
        level_offsets = np.cumsum([0, *level_sizes])
        if level_offsets[-1] == boxes_all.shape[0]:
            nms_pre = 30000
            keep_parts = []
            for start, end in zip(level_offsets[:-1], level_offsets[1:]):
                level_mask = (box_indices >= start) & (box_indices < end)
                level_indices = np.nonzero(level_mask)[0]
                if level_indices.size > nms_pre:
                    level_scores = max_scores[level_indices]
                    keep = np.argpartition(-level_scores, nms_pre - 1)[:nms_pre]
                    keep = keep[np.argsort(-level_scores[keep])]
                    level_indices = level_indices[keep]
                keep_parts.append(level_indices)
            keep_indices = (
                np.concatenate(keep_parts)
                if keep_parts
                else np.empty((0,), dtype=np.int64)
            )
        else:
            nms_pre = min(30000, max_scores.size)
            keep_indices = np.argpartition(-max_scores, nms_pre - 1)[:nms_pre]
            keep_indices = keep_indices[np.argsort(-max_scores[keep_indices])]

        box_indices = box_indices[keep_indices]
        max_scores = max_scores[keep_indices]
        class_ids = class_ids[keep_indices]
        boxes = boxes_all[box_indices].astype(np.float32, copy=True)

        if len(boxes) == 0:
            return boxes, max_scores, class_ids

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_h)
        if ratio is None or ratio == 1.0:
            ratio = min(input_h / orig_h, input_w / orig_w)
        boxes = boxes / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_boxes]
        max_scores = max_scores[valid_boxes]
        class_ids = class_ids[valid_boxes]

        return boxes, max_scores, class_ids

    def _parse_picodet(self, all_outputs, effective_imgsz, orig_w, orig_h, conf):
        """Parse PICODET output: (B, N, 4+nc) — xyxy (input-canvas pixels) + sigmoid scores.

        PICODET exports use simple resize (not letterbox), so the inverse
        scale is independent x/y ratios from input canvas back to the
        original image.
        """
        outputs = all_outputs[0][0]  # (N, 4+nc)
        boxes_all = outputs[:, :4]
        scores = outputs[:, 4:]

        # Multi-label per anchor (every (anchor, class) pair above conf), matching the native
        # postprocess (postprocess/picodet.py). argmax kept only the best class per anchor and
        # dropped secondary-class detections, costing ~0.7 mAP vs native.
        valid = scores > conf
        if not valid.any():
            return (
                np.empty((0, 4), dtype=boxes_all.dtype),
                np.empty((0,), dtype=scores.dtype),
                np.empty((0,), dtype=np.int64),
            )

        box_indices, class_ids = np.nonzero(valid)
        max_scores = scores[box_indices, class_ids]

        # Per-level top-k (nms_pre), matching native postprocess/picodet.py: each FPN level is
        # capped separately so a busy level can't crowd out detections from other levels. The
        # exported output concatenates the 4 PicoDet levels (strides 8/16/32/64) in order, so we
        # map each candidate's anchor index to its level via the cumulative grid sizes. Falls back
        # to a single global cap if the layout doesn't match (unexpected stride/imgsz). The cap
        # also keeps numpy NMS fast (the uncapped multi-label flood at conf=0.001 was ~1.6-12 s/img).
        nms_pre = 1000
        # Ceil division: feature maps from stride-2 convs round up, so e.g. PicoDet-m (416) has a
        # 7x7 stride-64 P6 (416//64=6 would mismatch N and silently fall back to the global cap).
        level_sizes = [((effective_imgsz + s - 1) // s) ** 2 for s in (8, 16, 32, 64)]
        if sum(level_sizes) == scores.shape[0]:
            bounds = np.cumsum([0] + level_sizes)
            keep = []
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                idx = np.nonzero((box_indices >= lo) & (box_indices < hi))[0]
                if idx.size > nms_pre:
                    idx = idx[np.argpartition(max_scores[idx], -nms_pre)[-nms_pre:]]
                keep.append(idx)
            keep = np.concatenate(keep) if keep else np.empty(0, dtype=np.int64)
            box_indices, class_ids, max_scores = (
                box_indices[keep],
                class_ids[keep],
                max_scores[keep],
            )
        elif max_scores.shape[0] > nms_pre:
            top = np.argpartition(max_scores, -nms_pre)[-nms_pre:]
            box_indices, class_ids, max_scores = (
                box_indices[top],
                class_ids[top],
                max_scores[top],
            )

        boxes = boxes_all[box_indices].copy()

        scale_x = orig_w / effective_imgsz
        scale_y = orig_h / effective_imgsz
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return boxes, max_scores, class_ids

    def _parse_embedded_nms(self, all_outputs, effective_imgsz, orig_w, orig_h, conf):
        """Parse a graph-embedded-NMS detection output.

        Shape ``(1, max_det, 6)`` with rows ``[x1, y1, x2, y2, score, class]`` in
        input-canvas (letterbox) pixels. NMS already ran in the graph; here we
        drop zero-padding / sub-``conf`` rows and undo the letterbox scaling.
        """
        det = np.asarray(all_outputs[0], dtype=np.float32)
        if det.ndim == 3:
            det = det[0]  # (max_det, 6)
        keep = det[:, 4] > conf
        det = det[keep]
        if det.shape[0] == 0:
            empty = np.empty((0, 4), dtype=np.float32)
            return empty, np.empty((0,), np.float32), np.empty((0,), np.int64)

        boxes = det[:, :4].copy()
        scores = det[:, 4].astype(np.float32)
        class_ids = det[:, 5].astype(np.int64)

        input_h, input_w = _imgsz_hw(effective_imgsz)
        ratio = min(input_h / orig_h, input_w / orig_w)
        boxes /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        return boxes, scores, class_ids

    def _parse_yolo9(
        self,
        all_outputs,
        effective_imgsz,
        orig_w,
        orig_h,
        conf,
        iou: float = 0.45,
        max_det: int = 300,
    ):
        """Parse YOLO9 output: (B, 4+nc, N) — xyxy + class_scores."""
        if self.task == "obb":
            output = torch.from_numpy(np.asarray(all_outputs[0]))
            parsed = yolo9_postprocess(
                {"predictions": output, "obb": True},
                conf_thres=conf,
                iou_thres=iou,
                input_size=effective_imgsz,
                original_size=(orig_w, orig_h),
                max_det=max_det,
                letterbox=True,
            )
            boxes = np.asarray(parsed["boxes"], dtype=np.float32).reshape(-1, 4)
            max_scores = np.asarray(parsed["scores"], dtype=np.float32)
            class_ids = np.asarray(parsed["classes"], dtype=np.int64)
            obb = np.asarray(parsed["obb"], dtype=np.float32).reshape(-1, 7)
            return boxes, max_scores, class_ids, None, obb

        outputs = all_outputs[0][0].T  # (N, 4+nc)

        boxes_input_all = outputs[:, :4]
        scores = outputs[:, 4:]
        keypoints = None
        keypoints_all = None
        if self.task == "pose" and len(all_outputs) >= 2:
            keypoints_all = np.asarray(all_outputs[1][0], dtype=np.float32)

        if self.model_family == "yolo9_e2e" and self.task == "detect":
            topk_anchors = min(max_det, scores.shape[0])
            if topk_anchors == 0 or scores.shape[-1] == 0:
                return (
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                )

            anchor_scores = np.max(scores, axis=1)
            anchor_idx = np.argpartition(-anchor_scores, topk_anchors - 1)[
                :topk_anchors
            ]
            anchor_idx = anchor_idx[np.argsort(-anchor_scores[anchor_idx])]
            boxes_subset = boxes_input_all[anchor_idx]
            scores_subset = scores[anchor_idx]

            flat_scores = scores_subset.reshape(-1)
            topk_scores = min(max_det, flat_scores.size)
            flat_idx = np.argpartition(-flat_scores, topk_scores - 1)[:topk_scores]
            flat_idx = flat_idx[np.argsort(-flat_scores[flat_idx])]
            class_ids = flat_idx % scores_subset.shape[-1]
            box_indices = flat_idx // scores_subset.shape[-1]
            boxes_input = boxes_subset[box_indices]
            max_scores = flat_scores[flat_idx]
            keep = max_scores > conf
            boxes_input = boxes_input[keep]
            max_scores = max_scores[keep]
            class_ids = class_ids[keep]
        else:
            anchor_idx, class_ids = np.nonzero(scores > conf)
            boxes_input = boxes_input_all[anchor_idx]
            max_scores = scores[anchor_idx, class_ids]
            if keypoints_all is not None:
                keypoints = keypoints_all[anchor_idx].copy()
            max_nms = max(max_det, _YOLO9_MAX_NMS_CANDIDATES)
            if max_scores.size > max_nms:
                keep = np.argpartition(-max_scores, max_nms - 1)[:max_nms]
                keep = keep[np.argsort(-max_scores[keep])]
                boxes_input = boxes_input[keep]
                max_scores = max_scores[keep]
                class_ids = class_ids[keep]
                if keypoints is not None:
                    keypoints = keypoints[keep]

        boxes = boxes_input.copy()

        if len(boxes) == 0:
            if self.task == "pose" and keypoints_all is not None:
                return boxes, max_scores, class_ids, None, None, keypoints_all[:0]
            return boxes, max_scores, class_ids

        input_h, input_w = _imgsz_hw(effective_imgsz)
        ratio = min(input_h / orig_h, input_w / orig_w)
        boxes[:, :4] /= ratio
        if keypoints is not None:
            keypoints[..., :2] /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        if keypoints is not None:
            keypoints[..., 0] = np.clip(keypoints[..., 0], 0, orig_w)
            keypoints[..., 1] = np.clip(keypoints[..., 1], 0, orig_h)
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not valid_boxes.any():
            if self.task == "pose" and keypoints is not None:
                return (
                    boxes[:0],
                    max_scores[:0],
                    class_ids[:0],
                    None,
                    None,
                    keypoints[:0],
                )
            return boxes[:0], max_scores[:0], class_ids[:0]
        if not valid_boxes.all():
            boxes = boxes[valid_boxes]
            boxes_input = boxes_input[valid_boxes]
            max_scores = max_scores[valid_boxes]
            class_ids = class_ids[valid_boxes]
            if keypoints is not None:
                keypoints = keypoints[valid_boxes]

        if self.task == "pose" and keypoints is not None:
            return boxes, max_scores, class_ids, None, None, keypoints

        return boxes, max_scores, class_ids

    def _parse_yolonas(
        self,
        all_outputs,
        effective_imgsz,
        orig_w,
        orig_h,
        conf,
        ratio: Optional[float] = None,
    ):
        """Parse YOLO-NAS output: [boxes(B,N,4), scores(B,N,nc)] in input pixels."""
        first = all_outputs[0][0]
        second = all_outputs[1][0]
        if first.shape[-1] == 4 and second.shape[-1] != 4:
            boxes = first
            scores = second
        elif second.shape[-1] == 4 and first.shape[-1] != 4:
            boxes = second
            scores = first
        else:
            boxes = first
            scores = second

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = max_scores > conf
        boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]

        if len(boxes) == 0:
            return boxes, max_scores, class_ids

        boxes = boxes.astype(np.float32, copy=True)
        if YOLO_NAS_PRE_NMS_TOP_K and max_scores.size > YOLO_NAS_PRE_NMS_TOP_K:
            keep = np.argpartition(-max_scores, YOLO_NAS_PRE_NMS_TOP_K - 1)[
                :YOLO_NAS_PRE_NMS_TOP_K
            ]
            keep = keep[np.argsort(-max_scores[keep])]
            boxes = boxes[keep]
            max_scores = max_scores[keep]
            class_ids = class_ids[keep]

        input_h, input_w = _imgsz_hw(effective_imgsz)
        if ratio is None or ratio <= 0:
            resize_size = min(YOLO_NAS_RESIZE_SIZE, input_h, input_w)
            ratio = min(resize_size / orig_h, resize_size / orig_w)
        new_w = round(orig_w * ratio)
        new_h = round(orig_h * ratio)
        offset_x = (input_w - new_w) // 2
        offset_y = (input_h - new_h) // 2
        boxes[:, 0::2] = (boxes[:, 0::2] - offset_x) / ratio
        boxes[:, 1::2] = (boxes[:, 1::2] - offset_y) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_boxes]
        max_scores = max_scores[valid_boxes]
        class_ids = class_ids[valid_boxes]
        return boxes, max_scores, class_ids

    def _parse_yolonas_pose(
        self,
        all_outputs,
        effective_imgsz,
        orig_w,
        orig_h,
        conf,
        ratio: Optional[float] = None,
        max_det=300,
    ):
        """Parse YOLO-NAS pose: boxes, scores, keypoint xy, keypoint confidence."""
        boxes = all_outputs[0][0]
        scores = all_outputs[1][0]
        keypoints_xy = all_outputs[2][0]
        keypoints_conf = all_outputs[3][0]

        # scores: [A, nc]. Single-class pose keeps the historical squeeze;
        # multi-class pose takes the top-scoring class per anchor.
        if scores.ndim > 1 and scores.shape[-1] > 1:
            class_ids_full = scores.argmax(axis=-1).astype(np.int64)
            scores = scores.max(axis=-1)
        else:
            scores = scores.squeeze(-1)
            class_ids_full = None

        mask = scores >= conf
        boxes = boxes[mask].astype(np.float32, copy=True)
        max_scores = scores[mask].astype(np.float32, copy=False)
        keypoints_xy = keypoints_xy[mask].astype(np.float32, copy=True)
        keypoints_conf = keypoints_conf[mask].astype(np.float32, copy=False)
        if class_ids_full is not None:
            class_ids = class_ids_full[mask]
        else:
            class_ids = np.zeros((max_scores.shape[0],), dtype=np.int64)

        if len(boxes) == 0:
            keypoints = np.zeros((0, keypoints_xy.shape[-2], 3), dtype=np.float32)
            return boxes, max_scores, class_ids, None, None, keypoints

        pre_nms_top_k = max(1000, int(max_det))
        if max_scores.size > pre_nms_top_k:
            keep = np.argpartition(-max_scores, pre_nms_top_k - 1)[:pre_nms_top_k]
            keep = keep[np.argsort(-max_scores[keep])]
            boxes = boxes[keep]
            max_scores = max_scores[keep]
            keypoints_xy = keypoints_xy[keep]
            keypoints_conf = keypoints_conf[keep]
            class_ids = class_ids[keep]

        scale = ratio
        if scale is None or scale <= 0:
            scale = min(
                YOLO_NAS_POSE_RESIZE_SIZE / orig_h,
                YOLO_NAS_POSE_RESIZE_SIZE / orig_w,
            )
        boxes[:, 0::2] /= scale
        boxes[:, 1::2] /= scale
        keypoints_xy[..., 0] /= scale
        keypoints_xy[..., 1] /= scale

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not valid.all():
            boxes = boxes[valid]
            max_scores = max_scores[valid]
            class_ids = class_ids[valid]
            keypoints_xy = keypoints_xy[valid]
            keypoints_conf = keypoints_conf[valid]

        keypoints = np.concatenate([keypoints_xy, keypoints_conf[..., None]], axis=-1)
        return boxes, max_scores, class_ids, None, None, keypoints

    def _parse_dfine(self, all_outputs, orig_w, orig_h, conf, max_det: int = 300):
        """Parse D-FINE outputs: pred_logits (B, Q, nc) + pred_boxes (B, Q, 4) cxcywh [0,1].

        Matches the upstream DFINEPostProcessor (use_focal_loss=True): sigmoid →
        topk over (queries × classes) flattened → labels = topk_idx % nc, query_idx
        = topk_idx // nc. No NMS (DETR set-prediction).
        """
        pred_logits = all_outputs[0][0]  # (Q, nc)
        pred_boxes = all_outputs[1][0]  # (Q, 4)

        Q, nc = pred_logits.shape
        prob = 1.0 / (1.0 + np.exp(-pred_logits.astype(np.float64)))
        prob = prob.astype(np.float32)

        flat = prob.reshape(-1)  # (Q * nc,)
        k = min(max_det, flat.size)
        # Top-k via argpartition (faster than full sort).
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]

        scores = flat[idx]
        query_idx = idx // nc
        class_ids = idx % nc

        # cxcywh -> xyxy in [0,1], then gather + scale.
        cx, cy, w, h = (
            pred_boxes[:, 0],
            pred_boxes[:, 1],
            pred_boxes[:, 2],
            pred_boxes[:, 3],
        )
        boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes = boxes_xyxy[query_idx]

        boxes[:, [0, 2]] *= orig_w
        boxes[:, [1, 3]] *= orig_h
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        mask = scores > conf
        return boxes[mask], scores[mask], class_ids[mask].astype(np.int64)

    def _parse_dfine_segment(
        self, all_outputs, orig_w, orig_h, conf, max_det: int = 300
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Parse D-FINE-seg raw exports into boxes, classes, and masks."""
        pred_logits = all_outputs[0][0]
        pred_boxes = all_outputs[1][0]
        pred_masks = all_outputs[2][0] if len(all_outputs) >= 3 else None

        _, nc = pred_logits.shape
        prob = 1.0 / (1.0 + np.exp(-pred_logits.astype(np.float64)))
        prob = prob.astype(np.float32)
        flat = prob.reshape(-1)
        k = min(max_det, flat.size)
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]

        scores = flat[idx]
        query_idx = idx // nc
        class_ids = idx % nc

        boxes = self._scale_cxcywh_boxes(
            pred_boxes[query_idx],
            orig_w,
            orig_h,
            clip=True,
        )
        keep = scores > conf
        boxes = boxes[keep]
        scores = scores[keep]
        query_idx = query_idx[keep]
        class_ids = class_ids[keep]

        masks_out = None
        if pred_masks is not None and query_idx.size > 0:
            masks_t = torch.from_numpy(pred_masks[query_idx]).unsqueeze(1).float()
            in_h, in_w = _imgsz_hw(self.input_size)
            masks_t = F.interpolate(
                masks_t,
                size=(int(in_h), int(in_w)),
                mode="bilinear",
                align_corners=False,
            )
            masks_t = F.interpolate(
                masks_t,
                size=(int(orig_h), int(orig_w)),
                mode="bilinear",
                align_corners=False,
            )[:, 0].clamp_(0, 1)
            boxes_t = torch.from_numpy(boxes).to(dtype=masks_t.dtype)
            if boxes_t.numel() > 0:
                ys = torch.arange(int(orig_h), dtype=masks_t.dtype)[None, :, None]
                xs = torch.arange(int(orig_w), dtype=masks_t.dtype)[None, None, :]
                x1, y1, x2, y2 = boxes_t.T
                inside = (
                    (xs >= x1[:, None, None])
                    & (xs < x2[:, None, None])
                    & (ys >= y1[:, None, None])
                    & (ys < y2[:, None, None])
                )
                masks_t = masks_t * inside.to(dtype=masks_t.dtype)
            masks_out = (masks_t >= 0.5).numpy()

        return boxes, scores, class_ids.astype(np.int64), masks_out

    def _parse_ec_segment(
        self, all_outputs, orig_w, orig_h, conf, max_det=300
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Parse EC segmentation outputs: logits, normalized cxcywh boxes, masks."""
        pred_logits = all_outputs[0][0]
        pred_boxes = all_outputs[1][0]
        pred_masks = all_outputs[2][0] if len(all_outputs) >= 3 else None

        query_idx, class_ids, scores = self._ec_topk(pred_logits, max_det=max_det)
        keep = scores > conf
        query_idx = query_idx[keep]
        class_ids = class_ids[keep]
        max_scores = scores[keep]

        boxes = self._scale_cxcywh_boxes(
            pred_boxes[query_idx],
            orig_w,
            orig_h,
            clip=False,
        )
        masks_out = None
        if pred_masks is not None and query_idx.size > 0:
            masks_t = torch.from_numpy(pred_masks[query_idx]).unsqueeze(1).float()
            masks_t = F.interpolate(
                masks_t,
                size=(int(orig_h), int(orig_w)),
                mode="bilinear",
                align_corners=False,
            )
            masks_out = (masks_t[:, 0] > 0.0).numpy()

        return boxes, max_scores, class_ids.astype(np.int64), masks_out

    def _parse_ec_pose(self, all_outputs, orig_w, orig_h, conf, max_det=300):
        """Parse EC pose outputs: logits and normalized flattened keypoints."""
        pred_logits = all_outputs[0][0]
        pred_boxes = None
        pred_keypoints = all_outputs[1][0]
        if len(all_outputs) >= 3:
            maybe_boxes = all_outputs[1][0]
            maybe_keypoints = all_outputs[2][0]
            if maybe_boxes.shape[-1] == 4:
                pred_boxes = maybe_boxes
                pred_keypoints = maybe_keypoints

        scores_per_class = 1.0 / (1.0 + np.exp(-pred_logits.astype(np.float64)))
        scores_per_class = scores_per_class.astype(np.float32)
        # Person class is the LAST logit (index 1 of ECPose's 2-class head); keep
        # this in lockstep with ``postprocess_pose`` so .pt and ONNX agree.
        query_scores = scores_per_class[..., -1]
        k = min(max_det, query_scores.size)
        query_idx = np.argpartition(-query_scores, k - 1)[:k]
        query_idx = query_idx[np.argsort(-query_scores[query_idx])]
        scores = query_scores[query_idx]
        keep = scores >= conf
        query_idx = query_idx[keep]
        max_scores = scores[keep]
        class_ids = np.zeros((max_scores.shape[0],), dtype=np.int64)

        if pred_keypoints.ndim >= 3 and pred_keypoints.shape[-1] == 2:
            num_keypoints = int(pred_keypoints.shape[-2])
        else:
            num_keypoints = int(pred_keypoints.shape[-1]) // 2
        if num_keypoints <= 0:
            num_keypoints = int(getattr(self, "num_keypoints", 17) or 17)
        if query_idx.size == 0:
            empty_boxes = np.zeros((0, 4), dtype=np.float32)
            empty_keypoints = np.zeros((0, num_keypoints, 3), dtype=np.float32)
            return empty_boxes, max_scores, class_ids, None, None, empty_keypoints

        keypoints_xy = pred_keypoints[query_idx].reshape(-1, num_keypoints, 2)
        keypoints_xy = keypoints_xy.astype(np.float32, copy=True)
        keypoints_xy[..., 0] *= float(orig_w)
        keypoints_xy[..., 1] *= float(orig_h)

        if pred_boxes is not None:
            boxes = self._scale_cxcywh_boxes(pred_boxes[query_idx], orig_w, orig_h)
        else:
            x_min = keypoints_xy[..., 0].min(axis=1)
            y_min = keypoints_xy[..., 1].min(axis=1)
            x_max = keypoints_xy[..., 0].max(axis=1)
            y_max = keypoints_xy[..., 1].max(axis=1)
            boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
        visibility = np.ones((*keypoints_xy.shape[:-1], 1), dtype=np.float32)
        keypoints = np.concatenate([keypoints_xy, visibility], axis=-1)
        return boxes, max_scores, class_ids, None, None, keypoints

    @staticmethod
    def _ec_topk(pred_logits, max_det: int):
        scores = 1.0 / (1.0 + np.exp(-pred_logits.astype(np.float64)))
        scores = scores.astype(np.float32)
        num_classes = scores.shape[-1]
        flat = scores.reshape(-1)
        k = min(max_det, flat.size)
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]
        query_idx = idx // num_classes
        class_ids = idx % num_classes
        return query_idx, class_ids, flat[idx]

    @staticmethod
    def _scale_cxcywh_boxes(boxes_cxcywh, orig_w, orig_h, *, clip: bool = True):
        cx, cy, w, h = (
            boxes_cxcywh[:, 0],
            boxes_cxcywh[:, 1],
            boxes_cxcywh[:, 2],
            boxes_cxcywh[:, 3],
        )
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes = boxes.astype(np.float32, copy=False)
        boxes[:, [0, 2]] *= orig_w
        boxes[:, [1, 3]] *= orig_h
        if clip:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        return boxes

    def _normalize_rfdetr_keypoint_output(
        self,
        raw_keypoint_output,
        *,
        query_count: int,
        num_classes: int,
    ) -> np.ndarray:
        raw = np.asarray(raw_keypoint_output)
        if raw.ndim >= 3 and raw.shape[0] == 1 and raw.shape[1] == query_count:
            raw = raw[0]
        elif raw.ndim == 4 and raw.shape[0] == 1:
            raw = raw[0]

        if raw.ndim == 2:
            schema = getattr(self, "num_keypoints_per_class", None)
            if schema:
                schema_counts = np.asarray(
                    [int(count) for count in schema], dtype=np.int64
                )
                if schema_counts.size != num_classes or schema_counts.max() <= 0:
                    raise ValueError(
                        "Invalid RF-DETR GroupPose num_keypoints_per_class metadata "
                        f"for {num_classes} classes: {list(schema_counts)}"
                    )
                slots = int(schema_counts.size * schema_counts.max())
                if slots <= 0 or raw.shape[-1] % slots != 0:
                    raise ValueError(
                        "RF-DETR GroupPose flattened keypoint output cannot be "
                        f"reshaped with schema {list(schema_counts)}: {raw.shape}"
                    )
                pred_dim = raw.shape[-1] // slots
                raw = raw.reshape(raw.shape[0], slots, pred_dim)
            else:
                keypoint_dim = int(getattr(self, "keypoint_dim", 3) or 3)
                if keypoint_dim not in (2, 3) or raw.shape[-1] % keypoint_dim != 0:
                    raise ValueError(
                        "RF-DETR flattened keypoint output cannot be reshaped "
                        f"with keypoint_dim={keypoint_dim}: {raw.shape}"
                    )
                raw = raw.reshape(
                    raw.shape[0], raw.shape[-1] // keypoint_dim, keypoint_dim
                )

        if raw.ndim != 3:
            raise ValueError(f"Unexpected RF-DETR keypoint output shape: {raw.shape}")
        return raw

    def _parse_rfdetr(self, all_outputs, orig_w, orig_h, conf, max_det=300):
        """Parse RF-DETR output: boxes (B,300,4) cxcywh [0,1] + logits (B,300,nc).

        For segmentation models a third output is present:
        masks (B,300,Hm,Wm) raw mask logits at model resolution.
        For pose models a third output is present:
        keypoints (B,300,K,3) with normalized xy and visibility logits.
        For OBB models a third output is present:
        angles (B,300,1) in radians.
        """
        first = all_outputs[0][0]
        second = all_outputs[1][0]
        if first.shape[-1] == 4:
            boxes_all = first
            logits = second
        else:
            logits = first
            boxes_all = second
        raw_masks = None
        raw_keypoints = None
        raw_keypoint_output = None
        raw_angles = None
        grouppose_active_keypoints = None
        if len(all_outputs) >= 3:
            if self.task == "obb":
                raw_angles = all_outputs[2][0]
            elif self.task == "pose":
                raw_keypoint_output = all_outputs[2]
            else:
                raw_masks = all_outputs[2][0]

        if raw_keypoint_output is not None and not getattr(
            self, "num_keypoints_per_class", None
        ):
            public_classes = int(self.nb_classes)
            if 0 < public_classes < logits.shape[-1]:
                logits = logits[:, :public_classes]
        scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64))).astype(np.float32)
        num_queries, num_classes = scores.shape
        if raw_keypoint_output is not None:
            raw_keypoints = self._normalize_rfdetr_keypoint_output(
                raw_keypoint_output,
                query_count=num_queries,
                num_classes=num_classes,
            )
        model_size = self.model_size or getattr(self, "size", None)
        num_select = (
            _rfdetr_num_select(self.task, model_size)
            if int(max_det) == 300
            else int(max_det)
        )
        k = min(
            num_select,
            num_queries * num_classes,
        )
        flat_indexes = np.argpartition(scores.reshape(-1), -k)[-k:]
        flat_indexes = flat_indexes[np.argsort(scores.reshape(-1)[flat_indexes])[::-1]]
        max_scores = scores.reshape(-1)[flat_indexes]
        query_idx = flat_indexes // num_classes
        class_ids = flat_indexes % num_classes
        boxes_raw = boxes_all[query_idx]
        angles_raw = raw_angles[query_idx] if raw_angles is not None else None
        keypoints_raw = (
            raw_keypoints[query_idx].copy() if raw_keypoints is not None else None
        )
        if raw_masks is not None:
            raw_masks = raw_masks[query_idx]

        if (
            self.task == "pose"
            and keypoints_raw is not None
            and keypoints_raw.ndim == 3
            and keypoints_raw.shape[-1] >= 7
            and num_classes > 1
            and keypoints_raw.shape[1] % num_classes == 0
        ):
            schema = getattr(self, "num_keypoints_per_class", None)
            keypoint_counts = None
            if schema:
                schema_counts = np.asarray(
                    [int(count) for count in schema], dtype=np.int64
                )
                if (
                    schema_counts.size == num_classes
                    and schema_counts.max() > 0
                    and keypoints_raw.shape[1]
                    == schema_counts.size * int(schema_counts.max())
                ):
                    keypoint_counts = schema_counts
                    max_num_keypoints = int(schema_counts.max())
                else:
                    raise ValueError(
                        "Invalid RF-DETR GroupPose num_keypoints_per_class metadata "
                        f"for keypoint output {keypoints_raw.shape}: {list(schema_counts)}"
                    )
            else:
                max_num_keypoints = keypoints_raw.shape[1] // num_classes
            grouped = keypoints_raw.reshape(
                keypoints_raw.shape[0],
                num_classes,
                max_num_keypoints,
                keypoints_raw.shape[-1],
            )
            selected = grouped[np.arange(len(class_ids)), class_ids]

            # GroupPose exports use internal class 0 for no-keypoint detections
            # and keypoint-bearing classes after it. Public pose labels are
            # contiguous over only the keypoint-bearing classes (person -> 0).
            if keypoint_counts is None:
                keypoint_counts = np.full(
                    num_classes, max_num_keypoints, dtype=np.int64
                )
                if self.nb_classes == num_classes - 1:
                    keypoint_counts[0] = 0
            active_counts = keypoint_counts[class_ids]
            valid_pose_class = active_counts > 0

            if np.any(valid_pose_class):
                trace_alpha = 0.2
                log_mean_traces = np.zeros(len(selected), dtype=np.float32)
                for class_idx, active_count in enumerate(keypoint_counts):
                    if active_count <= 0:
                        continue
                    class_mask = class_ids == class_idx
                    if not np.any(class_mask):
                        continue
                    log_mean_traces[class_mask] = _rfdetr_keypoint_log_mean_trace_np(
                        selected[class_mask, :active_count]
                    )
                max_scores = max_scores * np.exp(-trace_alpha * log_mean_traces)

            keypoints_selected = np.zeros(
                (len(selected), max_num_keypoints, 3),
                dtype=np.float32,
            )
            active_keypoint_mask = np.zeros(
                (len(selected), max_num_keypoints),
                dtype=bool,
            )
            for row_idx, active_count in enumerate(active_counts):
                if active_count <= 0:
                    continue
                keypoints_selected[row_idx, :active_count, :3] = selected[
                    row_idx,
                    :active_count,
                    :3,
                ]
                active_keypoint_mask[row_idx, :active_count] = True

            kp_classes = np.flatnonzero(keypoint_counts > 0)
            remap = np.full(num_classes, -1, dtype=class_ids.dtype)
            remap[kp_classes] = np.arange(len(kp_classes), dtype=class_ids.dtype)

            boxes_raw = boxes_raw[valid_pose_class]
            max_scores = max_scores[valid_pose_class]
            class_ids = remap[class_ids[valid_pose_class]]
            if angles_raw is not None:
                angles_raw = angles_raw[valid_pose_class]
            if keypoints_raw is not None:
                keypoints_raw = keypoints_selected[valid_pose_class]
                grouppose_active_keypoints = active_keypoint_mask[valid_pose_class]
            if raw_masks is not None:
                raw_masks = raw_masks[valid_pose_class]

        mask = max_scores > conf
        boxes_raw = boxes_raw[mask]
        max_scores, class_ids = max_scores[mask], class_ids[mask]
        if angles_raw is not None:
            angles_raw = angles_raw[mask]
        if keypoints_raw is not None:
            keypoints_raw = keypoints_raw[mask]
            if grouppose_active_keypoints is not None:
                grouppose_active_keypoints = grouppose_active_keypoints[mask]
        if raw_masks is not None:
            raw_masks = raw_masks[mask]

        if len(boxes_raw) == 0:
            if self.task == "obb":
                return (
                    boxes_raw,
                    max_scores,
                    class_ids,
                    None,
                    np.zeros((0, 7), dtype=np.float32),
                )
            if self.task == "pose" and keypoints_raw is not None:
                return boxes_raw, max_scores, class_ids, None, None, keypoints_raw
            return boxes_raw, max_scores, class_ids, None

        # COCO 91→80 class mapping
        if num_classes == 91 and self.nb_classes == 80:
            from ..models.rfdetr.model import _COCO91_TO_COCO80

            mapped = np.array([_COCO91_TO_COCO80.get(int(c), -1) for c in class_ids])
            valid = mapped >= 0
            boxes_raw = boxes_raw[valid]
            max_scores = max_scores[valid]
            class_ids = mapped[valid]
            if angles_raw is not None:
                angles_raw = angles_raw[valid]
            if keypoints_raw is not None:
                keypoints_raw = keypoints_raw[valid]
                if grouppose_active_keypoints is not None:
                    grouppose_active_keypoints = grouppose_active_keypoints[valid]
            if raw_masks is not None:
                raw_masks = raw_masks[valid]

        if len(boxes_raw) == 0:
            if self.task == "obb":
                return (
                    boxes_raw,
                    max_scores,
                    class_ids,
                    None,
                    np.zeros((0, 7), dtype=np.float32),
                )
            if self.task == "pose" and keypoints_raw is not None:
                return (
                    boxes_raw,
                    max_scores,
                    class_ids,
                    None,
                    None,
                    keypoints_raw,
                )
            return boxes_raw, max_scores, class_ids, None

        cx, cy, w, h = (
            boxes_raw[:, 0],
            boxes_raw[:, 1],
            boxes_raw[:, 2],
            boxes_raw[:, 3],
        )
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        obb_out = None
        if angles_raw is not None:
            angles = np.asarray(angles_raw, dtype=np.float32).reshape(-1)
            obb_out = np.stack(
                [
                    cx * orig_w,
                    cy * orig_h,
                    w * orig_w,
                    h * orig_h,
                    angles,
                    max_scores,
                    class_ids.astype(np.float32),
                ],
                axis=1,
            ).astype(np.float32, copy=False)

        # Resize and threshold masks to original image resolution
        masks_out = None
        if raw_masks is not None and len(raw_masks) > 0:
            masks_t = torch.from_numpy(raw_masks).unsqueeze(1).float()
            masks_t = F.interpolate(
                masks_t,
                size=(int(orig_h), int(orig_w)),
                mode="bilinear",
                align_corners=False,
            )
            masks_out = (masks_t[:, 0] > 0.0).numpy()  # (N, H, W)

        keypoints_out = None
        if keypoints_raw is not None:
            keypoints_out = np.asarray(keypoints_raw, dtype=np.float32).copy()
            keypoints_out[..., 0] *= float(orig_w)
            keypoints_out[..., 1] *= float(orig_h)
            if keypoints_out.shape[-1] == 2:
                visibility = np.ones((*keypoints_out.shape[:-1], 1), dtype=np.float32)
                keypoints_out = np.concatenate([keypoints_out, visibility], axis=-1)
            else:
                keypoints_out[..., 2] = 1.0 / (1.0 + np.exp(-keypoints_out[..., 2]))
                keypoints_out = keypoints_out[..., :3]
            if grouppose_active_keypoints is not None:
                keypoints_out[~grouppose_active_keypoints] = 0.0

        if self.task == "obb":
            return boxes, max_scores, class_ids, masks_out, obb_out
        if self.task == "pose":
            return boxes, max_scores, class_ids, masks_out, None, keypoints_out
        return boxes, max_scores, class_ids, masks_out

    def _parse_rtdetr(self, all_outputs, orig_w, orig_h, conf, max_det: int = 300):
        """Parse RT-DETR output: pred_boxes (B,Q,4) cxcywh [0,1] + pred_logits (B,Q,C).

        RTDETR outputs are already in the correct class indices (no COCO 91->80 mapping needed).
        """
        # all_outputs order depends on ONNX output naming; try both orderings
        first = all_outputs[0][0]  # (Q, 4) or (Q, C)
        second = all_outputs[1][0]  # (Q, C) or (Q, 4)

        # Detect which is boxes and which is logits by shape
        if first.shape[1] == 4 and len(second.shape) == 2 and second.shape[1] != 4:
            boxes_raw = first  # (Q, 4) normalized cxcywh
            logits = second  # (Q, C) raw logits
        elif second.shape[1] == 4 and len(first.shape) == 2 and first.shape[1] != 4:
            boxes_raw = second
            logits = first
        else:
            # Fallback: assume pred_logits has more columns (num_classes typically > 4)
            if first.shape[1] > second.shape[1]:
                logits = first
                boxes_raw = second
            else:
                logits = second
                boxes_raw = first

        # Match upstream RTDETRPostProcessor (and _parse_dfine): top-K across the
        # flattened (Q*nc) score matrix, allowing multiple classes per query.
        # Per-query argmax (the previous logic) silently dropped valid non-max
        # detections and cost ~0.7-0.9 mAP on COCO val2017.
        Q, nc = logits.shape
        prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        prob = prob.astype(np.float32)

        flat = prob.reshape(-1)
        k = min(max_det, flat.size)
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]

        scores = flat[idx]
        query_idx = idx // nc
        class_ids = idx % nc

        cx, cy, w, h = (
            boxes_raw[:, 0],
            boxes_raw[:, 1],
            boxes_raw[:, 2],
            boxes_raw[:, 3],
        )
        boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes = boxes_xyxy[query_idx]
        boxes[:, [0, 2]] *= orig_w
        boxes[:, [1, 3]] *= orig_h
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        mask = scores > conf
        return boxes[mask], scores[mask], class_ids[mask]

    # =========================================================================
    # Result building
    # =========================================================================

    @staticmethod
    def _parse_classify_probs(all_outputs) -> torch.Tensor:
        logits = np.asarray(all_outputs[0])
        if logits.ndim == 1:
            logits = logits[None, :]
        if logits.ndim != 2:
            raise ValueError(
                "Classification backend output must have shape (batch, classes), "
                f"got {tuple(logits.shape)}."
            )
        logits_t = torch.from_numpy(logits).float()
        return torch.softmax(logits_t, dim=1)[0]

    @staticmethod
    def _parse_restore_output(
        all_outputs, original_size: Tuple[int, int], scale: int = 1
    ) -> np.ndarray:
        """Decode backend restoration output to HWC uint8 RGB.

        For super-resolution the valid canvas is ``scale`` times the input, so
        the output is cropped to ``scale`` x the original size.
        """
        restored = np.asarray(all_outputs[0])
        if restored.ndim == 4:
            restored = restored[0]
        if restored.ndim == 3 and restored.shape[0] == 3:
            restored = np.transpose(restored, (1, 2, 0))
        if restored.ndim != 3 or restored.shape[-1] != 3:
            raise ValueError(
                "Restoration backend output must have shape [B, 3, H, W] "
                f"or [H, W, 3], got {tuple(restored.shape)}."
            )
        orig_w, orig_h = original_size
        restored = restored[: orig_h * int(scale), : orig_w * int(scale), :]
        return (np.clip(restored, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    def _build_classify_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        image_path,
    ) -> Results:
        return Results(
            boxes=None,
            probs=Probs(self._parse_classify_probs(all_outputs)),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
            task=self.task,
        )

    @staticmethod
    def _parse_depth_output(
        all_outputs, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Decode backend depth output to an (H, W) float map on the original canvas."""
        depth = np.asarray(all_outputs[0], dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[None, None]
        elif depth.ndim == 3:
            depth = depth[:, None] if depth.shape[0] == 1 else depth[None]
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(
                "Depth backend output must have shape [B, 1, H, W], "
                f"got {tuple(np.asarray(all_outputs[0]).shape)}."
            )
        orig_w, orig_h = original_size
        depth_t = torch.from_numpy(np.ascontiguousarray(depth))
        # align_corners=True matches the native depth families' postprocess.
        depth_t = F.interpolate(
            depth_t, size=(orig_h, orig_w), mode="bilinear", align_corners=True
        )
        return depth_t[0, 0]

    def _build_depth_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        original_size: Tuple[int, int],
        image_path,
    ) -> Results:
        depth = self._parse_depth_output(all_outputs, original_size)
        return Results(
            boxes=None,
            depth_map=DepthMap(depth, orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
            task=self.task,
        )

    def _parse_semantic_output(
        self,
        all_outputs,
        original_size: Tuple[int, int],
        effective_imgsz: ImageSize,
        ratio: float,
    ) -> torch.Tensor:
        logits = np.asarray(all_outputs[0], dtype=np.float32)
        if logits.ndim == 3:
            logits = logits[None]
        if logits.ndim != 4:
            raise ValueError(
                "Semantic backend output must have shape [B, C, H, W], "
                f"got {tuple(np.asarray(all_outputs[0]).shape)}."
            )
        orig_w, orig_h = original_size
        logits_t = torch.from_numpy(np.ascontiguousarray(logits))
        align_corners = False
        if self.model_family == "pidnet":
            input_h, input_w = _imgsz_hw(effective_imgsz)
            scale_y = logits_t.shape[-2] / input_h
            scale_x = logits_t.shape[-1] / input_w
            valid_h = min(
                logits_t.shape[-2], max(int(round(orig_h * ratio * scale_y)), 1)
            )
            valid_w = min(
                logits_t.shape[-1], max(int(round(orig_w * ratio * scale_x)), 1)
            )
            logits_t = logits_t[..., :valid_h, :valid_w]
            align_corners = True
        logits_t = F.interpolate(
            logits_t,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=align_corners,
        )
        return logits_t.argmax(dim=1)[0]

    def _build_semantic_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        original_size: Tuple[int, int],
        effective_imgsz: ImageSize,
        ratio: float,
        image_path,
    ) -> Results:
        semantic = self._parse_semantic_output(
            all_outputs, original_size, effective_imgsz, ratio
        )
        return Results(
            boxes=None,
            semantic_mask=SemanticMask(semantic, orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
            task=self.task,
        )

    @staticmethod
    def _parse_matte_output(
        all_outputs, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Decode matte logits to a soft alpha map on the original canvas."""
        logits = np.asarray(all_outputs[-1], dtype=np.float32)
        if logits.ndim == 2:
            logits = logits[None, None]
        elif logits.ndim == 3:
            logits = logits[:, None] if logits.shape[0] == 1 else logits[None]
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(
                "Matte backend output must have shape [B, 1, H, W], "
                f"got {tuple(np.asarray(all_outputs[-1]).shape)}."
            )
        orig_w, orig_h = original_size
        matte = torch.sigmoid(torch.from_numpy(np.ascontiguousarray(logits)))
        matte = F.interpolate(
            matte,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        return matte[0, 0].clamp(0.0, 1.0)

    def _build_matte_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        original_size: Tuple[int, int],
        image_path,
    ) -> Results:
        matte = self._parse_matte_output(all_outputs, original_size)
        return Results(
            boxes=None,
            matte=Matte(matte, orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
            task=self.task,
        )

    def _build_gaze_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        image_path,
    ) -> Results:
        """Decode L2CS yaw/pitch logits for a single face-crop input."""
        if len(all_outputs) != 2:
            raise ValueError(
                f"Gaze backend requires yaw and pitch logits, got {len(all_outputs)} outputs."
            )
        from ..models.l2cs.utils import bin_logits_to_angles

        yaw = torch.from_numpy(
            np.ascontiguousarray(np.asarray(all_outputs[0], dtype=np.float32))
        )
        pitch = torch.from_numpy(
            np.ascontiguousarray(np.asarray(all_outputs[1], dtype=np.float32))
        )
        angles = bin_logits_to_angles(
            yaw,
            pitch,
            num_bins=self.num_bins,
            bin_width_deg=self.bin_width_deg,
            offset_deg=self.offset_deg,
        )
        orig_h, orig_w = orig_shape
        boxes = Boxes(
            torch.tensor([[0.0, 0.0, float(orig_w), float(orig_h)]]),
            torch.ones(1),
            torch.zeros(1),
            orig_shape=orig_shape,
        )
        return Results(
            boxes=boxes,
            gaze=Gaze(angles, orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
            task=self.task,
        )

    def _build_point_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        original_size: Tuple[int, int],
        effective_imgsz: ImageSize,
        conf: float,
        max_det: int,
        image_path,
    ) -> Results:
        if self.model_family != "fomo":
            raise NotImplementedError(
                f"Exported point parsing is not implemented for {self.model_family!r}."
            )
        from ..models.fomo.utils import postprocess as postprocess_fomo

        heatmap = torch.from_numpy(
            np.ascontiguousarray(np.asarray(all_outputs[0], dtype=np.float32))
        )
        input_h, input_w = _imgsz_hw(effective_imgsz)
        if input_h != input_w:
            raise NotImplementedError("FOMO exported inference requires square imgsz.")
        decoded = postprocess_fomo(
            heatmap,
            conf_thres=conf,
            input_size=input_h,
            original_size=original_size,
            max_det=max_det,
        )["points"]
        return Results(
            boxes=None,
            points=Points(decoded, orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
            task=self.task,
        )

    def _build_restore_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        original_size: Tuple[int, int],
        image_path,
    ) -> Results:
        scale = self.restore_scale
        restored = self._parse_restore_output(all_outputs, original_size, scale)
        restored_hw = (int(restored.shape[0]), int(restored.shape[1]))
        return Results(
            boxes=None,
            restored=RestoredImage(torch.from_numpy(restored), restored_hw),
            orig_shape=orig_shape,
            restore_scale=scale,
            path=str(image_path) if image_path else None,
            names=self.names,
            task=self.task,
        )

    def _build_result(
        self,
        boxes: np.ndarray,
        max_scores: np.ndarray,
        class_ids: np.ndarray,
        *,
        masks: "np.ndarray | None" = None,
        obb: "np.ndarray | None" = None,
        keypoints: "np.ndarray | None" = None,
        orig_shape: Tuple[int, int],
        image_path,
        iou: float,
        classes: Optional[List[int]],
        max_det: int,
    ) -> Results:
        """Apply family-appropriate suppression/max_det/filtering and wrap."""
        if len(boxes) == 0:
            masks_obj = None
            if masks is not None:
                masks_array = np.asarray(masks)
                if masks_array.size == 0 and masks_array.ndim != 3:
                    masks_array = np.zeros((0, *orig_shape), dtype=bool)
                masks_obj = Masks(
                    torch.as_tensor(masks_array).bool(),
                    orig_shape,
                )
            elif self.task == "segment":
                masks_obj = Masks(
                    torch.zeros((0, *orig_shape), dtype=torch.bool),
                    orig_shape,
                )

            keypoints_obj = None
            if keypoints is not None:
                keypoints_array = np.asarray(keypoints)
                if keypoints_array.size == 0 and keypoints_array.ndim != 3:
                    num_keypoints = int(getattr(self, "num_keypoints", 0) or 0)
                    keypoint_dim = int(getattr(self, "keypoint_dim", 3) or 3)
                    keypoints_array = np.zeros(
                        (0, num_keypoints, keypoint_dim), dtype=np.float32
                    )
                keypoints_obj = Keypoints(
                    torch.as_tensor(keypoints_array, dtype=torch.float32),
                    orig_shape,
                )
            return Results(
                boxes=Boxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                ),
                obb=OBB(torch.zeros((0, 7), dtype=torch.float32), orig_shape)
                if self.task == "obb"
                else None,
                masks=masks_obj,
                keypoints=keypoints_obj,
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=self.names,
                task=self.task,
            )

        if obb is None and not _is_nms_free_family(self.model_family):
            # YOLO9 needs class-aware NMS so multi-label detections
            # on a shared anchor (same box, different class) survive, matching
            # the native batched_nms path. Class-agnostic NMS would drop the
            # lower-scored class and make exported runtimes disagree with native.
            # ONNX models with graph-embedded NMS still pass through this after
            # backend clipping so letterboxed-image behavior stays aligned with
            # native YOLO9 postprocess.
            if self.model_family in (
                "picodet",
                "rtmdet",
                "yolo9",
                "yolonas",
                "yolox",
            ):
                keep = _batched_nms_numpy(boxes, max_scores, class_ids, iou)
            else:
                keep = _nms_numpy(boxes, max_scores, iou)
            boxes, max_scores, class_ids = (
                boxes[keep],
                max_scores[keep],
                class_ids[keep],
            )
            if masks is not None:
                masks = masks[keep]
            if keypoints is not None:
                keypoints = keypoints[keep]

        if len(boxes) > max_det:
            top_indices = np.argsort(max_scores)[::-1][:max_det]
            boxes = boxes[top_indices]
            max_scores = max_scores[top_indices]
            class_ids = class_ids[top_indices]
            if masks is not None:
                masks = masks[top_indices]
            if obb is not None:
                obb = obb[top_indices]
            if keypoints is not None:
                keypoints = keypoints[top_indices]

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        conf_t = torch.tensor(max_scores, dtype=torch.float32)
        cls_t = torch.tensor(class_ids, dtype=torch.float32)
        obb_t = torch.tensor(obb, dtype=torch.float32) if obb is not None else None

        if classes is not None and len(boxes_t) > 0:
            cls_mask = torch.zeros(len(cls_t), dtype=torch.bool)
            for cid in classes:
                cls_mask |= cls_t == cid
            boxes_t = boxes_t[cls_mask]
            conf_t = conf_t[cls_mask]
            cls_t = cls_t[cls_mask]
            if masks is not None:
                masks = masks[cls_mask.numpy()]
            if obb_t is not None:
                obb_t = obb_t[cls_mask]
            if keypoints is not None:
                keypoints = keypoints[cls_mask.numpy()]

        masks_obj = None
        if masks is not None:
            masks_array = np.asarray(masks)
            if masks_array.size == 0 and masks_array.ndim != 3:
                masks_array = np.zeros((0, *orig_shape), dtype=bool)
            masks_obj = Masks(
                torch.as_tensor(masks_array).bool(), orig_shape=orig_shape
            )

        keypoints_obj = None
        if keypoints is not None:
            keypoints_array = np.asarray(keypoints)
            if keypoints_array.size == 0 and keypoints_array.ndim != 3:
                num_keypoints = int(getattr(self, "num_keypoints", 0) or 0)
                keypoint_dim = int(getattr(self, "keypoint_dim", 3) or 3)
                keypoints_array = np.zeros(
                    (0, num_keypoints, keypoint_dim), dtype=np.float32
                )
            keypoints_obj = Keypoints(
                torch.as_tensor(keypoints_array, dtype=torch.float32),
                orig_shape,
            )

        obb_obj = None
        if obb_t is not None:
            obb_obj = OBB(obb_t, orig_shape)

        return Results(
            boxes=Boxes(boxes_t, conf_t, cls_t),
            masks=masks_obj,
            keypoints=keypoints_obj,
            obb=obb_obj,
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
            task=self.task,
        )

    # =========================================================================
    # Save
    # =========================================================================

    def _save_annotated(self, result, original_img, image_path, output_path):
        """Save annotated image to disk."""
        is_matte = result.boxes is None and getattr(result, "matte", None) is not None
        ext = Path(str(image_path).split("?", 1)[0]).suffix.lstrip(".")
        if not ext:
            ext = "jpg"
        if is_matte:
            ext = "png"

        if output_path:
            final_path = resolve_save_path(
                output_path,
                image_path,
                ext=ext,
                force_ext=is_matte,
            )
        else:
            stem = get_safe_stem(image_path) if image_path else "inference"
            model_tag = Path(self.model_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            candidate = (
                Path("runs/detections") / f"{stem}_{model_tag}_{timestamp}.{ext}"
            )
            final_path = resolve_save_path(
                candidate,
                image_path,
                ext=ext,
                force_ext=is_matte,
            )

        with save_path_write_guard(final_path):
            self._write_annotated_result(
                result,
                original_img,
                final_path,
                is_matte=is_matte,
            )

    def _write_annotated_result(
        self, result, original_img, final_path: Path, *, is_matte: bool
    ) -> None:
        """Render and write one already-reserved backend result path."""
        annotated_img = original_img.copy()
        if result.boxes is None and getattr(result, "probs", None) is not None:
            pass
        elif result.boxes is None and getattr(result, "restored", None) is not None:
            annotated_img = Image.fromarray(result.restored.array, mode="RGB")
        elif result.boxes is None and getattr(result, "depth_map", None) is not None:
            depth_data = result.depth_map.data
            if isinstance(depth_data, torch.Tensor):
                depth_data = depth_data.cpu().numpy()
            annotated_img = draw_depth_map(original_img, depth_data)
        elif result.boxes is None and getattr(result, "semantic_mask", None) is not None:
            semantic_data = result.semantic_mask.data
            if isinstance(semantic_data, torch.Tensor):
                semantic_data = semantic_data.cpu().numpy()
            annotated_img = draw_semantic_mask(original_img, semantic_data)
        elif result.boxes is None and getattr(result, "panoptic", None) is not None:
            panoptic_data = result.panoptic.data
            if isinstance(panoptic_data, torch.Tensor):
                panoptic_data = panoptic_data.cpu().numpy()
            annotated_img = draw_panoptic(
                original_img,
                panoptic_data,
                result.panoptic.segments_info,
                class_names=result.names,
            )
        elif is_matte:
            result.save(final_path, image=original_img)
            log_saved_result(result, final_path)
            return
        elif result.boxes is None and getattr(result, "ocr", None) is not None:
            if len(result.ocr) > 0:
                ocr_data = result.ocr.numpy()
                annotated_img = draw_ocr_regions(
                    original_img,
                    ocr_data.data,
                    ocr_data.texts,
                    ocr_data.conf,
                )
        elif result.boxes is None and getattr(result, "points", None) is not None:
            if len(result.points) > 0:
                annotated_img = draw_points(
                    original_img,
                    result.points.xy.tolist(),
                    result.points.conf.tolist(),
                    result.points.cls.tolist(),
                    class_names=result.names,
                )
        elif result.boxes is not None and len(result.boxes) > 0:
            if result.masks is not None:
                masks_data = result.masks.data
                if isinstance(masks_data, torch.Tensor):
                    masks_data = masks_data.cpu().numpy()
                annotated_img = draw_masks(
                    annotated_img,
                    masks_data,
                    result.boxes.cls.tolist(),
                )
            if result.obb is not None:
                annotated_img = draw_obb(
                    annotated_img,
                    result.obb.xywhr.tolist(),
                    result.obb.conf.tolist(),
                    result.obb.cls.tolist(),
                    class_names=self.names,
                )
            else:
                annotated_img = draw_boxes(
                    annotated_img,
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                    class_names=self.names,
                )
            if result.keypoints is not None:
                kpts_np = result.keypoints.data
                if isinstance(kpts_np, torch.Tensor):
                    kpts_np = kpts_np.cpu().numpy()
                annotated_img = draw_keypoints(annotated_img, kpts_np)
            if result.gaze is not None:
                boxes_np = result.boxes.numpy()
                gaze_np = result.gaze.numpy()
                annotated_img = draw_gaze_arrows(
                    annotated_img,
                    boxes_np.xyxy.tolist(),
                    gaze_np.pitch.tolist(),
                    gaze_np.yaw.tolist(),
                )

        annotated_img.save(final_path)
        log_saved_result(result, final_path)

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def build_names(nb_classes: int) -> Dict[int, str]:
        """Build a class names dict — COCO for 80 classes, generic otherwise."""
        if nb_classes == 80:
            return {i: n for i, n in enumerate(COCO_CLASSES)}
        return {i: f"class_{i}" for i in range(nb_classes)}

    def eval(self):
        return self

    def _get_model_name(self) -> str:
        return self.model_family or "export"

    def _get_input_size(self) -> ImageSize:
        return self.imgsz

    def _get_val_preprocessor(self, img_size: ImageSize | None = None):
        if img_size is None:
            img_size = self._get_input_size()

        from ..validation.preprocessors import (
            DEIMValPreprocessor,
            DEIMv2DINOValPreprocessor,
            DEIMv2ValPreprocessor,
            DFINEValPreprocessor,
            ECValPreprocessor,
            PICODETValPreprocessor,
            RFDETRValPreprocessor,
            RTDETRValPreprocessor,
            RTDETRv2ValPreprocessor,
            RTMDetValPreprocessor,
            StandardValPreprocessor,
            YOLO9E2EValPreprocessor,
            YOLO9ValPreprocessor,
            YOLONASValPreprocessor,
            YOLOXValPreprocessor,
        )

        if self.model_family == "deimv2":
            from ..models.deimv2.nn import DINO_SIZES

            model_size = self.model_size or getattr(self, "size", None)
            preprocessor_cls = (
                DEIMv2DINOValPreprocessor
                if model_size in DINO_SIZES
                else DEIMv2ValPreprocessor
            )
            return preprocessor_cls(img_size=_imgsz_hw(img_size))

        preprocessor_cls = {
            "deim": DEIMValPreprocessor,
            "dfine": DFINEValPreprocessor,
            "ec": ECValPreprocessor,
            "picodet": PICODETValPreprocessor,
            "rfdetr": RFDETRValPreprocessor,
            "rtdetr": RTDETRValPreprocessor,
            "rtdetrv2": RTDETRv2ValPreprocessor,
            "rtdetrv4": DFINEValPreprocessor,
            "rtmdet": RTMDetValPreprocessor,
            "yolo9": YOLO9ValPreprocessor,
            "yolo9_e2e": YOLO9E2EValPreprocessor,
            "yolo9_p2": YOLO9ValPreprocessor,
            "yolonas": YOLONASValPreprocessor,
            "yolox": YOLOXValPreprocessor,
        }.get(self.model_family, StandardValPreprocessor)
        return preprocessor_cls(img_size=_imgsz_hw(img_size))

    def _resolve_predict_imgsz(self, imgsz: ImageSize | None = None) -> ImageSize:
        effective = _normalize_imgsz(imgsz if imgsz is not None else self.imgsz)
        if (
            _is_rectangular_imgsz(effective)
            and (self.model_family or "").lower() not in _RECTANGULAR_BACKEND_FAMILIES
        ):
            raise NotImplementedError(
                "Rectangular imgsz backend inference is currently supported "
                "for YOLO9-family and NAFNet exports only."
            )
        return effective

    def _forward(self, input_tensor: torch.Tensor):
        blob = input_tensor.detach().cpu().numpy()
        try:
            outputs = self._run_inference(blob)
        except Exception:
            if blob.shape[0] <= 1:
                raise
            per_image_outputs = [
                self._run_inference(blob[i : i + 1]) for i in range(blob.shape[0])
            ]
            outputs = [
                np.concatenate(
                    [np.asarray(item[j]) for item in per_image_outputs], axis=0
                )
                for j in range(len(per_image_outputs[0]))
            ]
        return [torch.from_numpy(np.asarray(output)) for output in outputs]

    @staticmethod
    def _as_numpy_outputs(output) -> list:
        if isinstance(output, torch.Tensor):
            return [output.detach().cpu().numpy()]
        if isinstance(output, np.ndarray):
            return [output]
        if isinstance(output, (list, tuple)):
            arrays = []
            for item in output:
                if isinstance(item, torch.Tensor):
                    arrays.append(item.detach().cpu().numpy())
                else:
                    arrays.append(np.asarray(item))
            return arrays
        return [np.asarray(output)]

    @staticmethod
    def _unpack_parsed_outputs(parsed):
        if len(parsed) == 6:
            return parsed
        if len(parsed) == 5:
            boxes, max_scores, class_ids, masks, obb = parsed
            return boxes, max_scores, class_ids, masks, obb, None
        boxes, max_scores, class_ids, masks = parsed
        return boxes, max_scores, class_ids, masks, None, None

    def _postprocess(
        self,
        output,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        input_size: ImageSize | None = None,
        letterbox: bool = False,
        max_det: int = 300,
        ratio: float | None = None,
        **kwargs,
    ) -> Dict:
        effective_imgsz = self._resolve_predict_imgsz(input_size)
        outputs = self._as_numpy_outputs(output)
        if self.task == "classify":
            return {"probs": self._parse_classify_probs(outputs)}
        if self.task == "restore":
            restored = np.asarray(outputs[0])
            if restored.ndim == 4:
                orig_w, orig_h = original_size
                restored = restored[:, :, :orig_h, :orig_w]
            return {"restored": torch.from_numpy(restored).float().clamp(0.0, 1.0)}
        if self.task == "depth":
            return {"depth": self._parse_depth_output(outputs, original_size)}
        if self.task == "matte":
            return {"matte": self._parse_matte_output(outputs, original_size)}
        if self.task == "gaze":
            result = self._build_gaze_result(
                outputs,
                orig_shape=(int(original_size[1]), int(original_size[0])),
                image_path=None,
            )
            return {"gaze": result.gaze.data}
        if self.task == "semantic":
            return {
                "semantic": self._parse_semantic_output(
                    outputs,
                    original_size,
                    effective_imgsz,
                    float(ratio or 1.0),
                )
            }
        if self.task == "point":
            result = self._build_point_result(
                outputs,
                orig_shape=(int(original_size[1]), int(original_size[0])),
                original_size=original_size,
                effective_imgsz=effective_imgsz,
                conf=conf_thres,
                max_det=max_det,
                image_path=None,
            )
            return {"points": result.points.data}
        parsed = self._parse_outputs(
            outputs,
            effective_imgsz,
            original_size,
            conf_thres,
            ratio=ratio,
            iou=iou_thres,
            max_det=max_det,
        )
        boxes, max_scores, class_ids, masks, obb, keypoints = (
            self._unpack_parsed_outputs(parsed)
        )
        result = self._build_result(
            boxes,
            max_scores,
            class_ids,
            masks=masks,
            obb=obb,
            keypoints=keypoints,
            orig_shape=(int(original_size[1]), int(original_size[0])),
            image_path=None,
            iou=iou_thres,
            classes=None,
            max_det=max_det,
        )

        det: Dict[str, object] = {
            "num_detections": len(result),
            "boxes": result.boxes.xyxy,
            "scores": result.boxes.conf,
            "classes": result.boxes.cls.to(torch.int64),
        }
        if result.masks is not None:
            det["masks"] = result.masks.data
        if result.keypoints is not None:
            det["keypoints"] = result.keypoints.data
        if result.obb is not None:
            det["obb"] = result.obb.data
        return det

    def val(
        self,
        data: str | None = None,
        batch: int = 16,
        imgsz: ImageSize | None = None,
        conf: float = 0.001,
        iou: float = 0.6,
        workers: int = 4,
        allow_download_scripts: bool = False,
        device: str | None = None,
        split: str = "val",
        augment: bool = False,
        save_json: bool = False,
        verbose: bool = True,
        *,
        plots: bool | None = None,
        **kwargs,
    ) -> Dict:
        from ..validation import (
            ClassifyValidator,
            DepthValidator,
            DetectionValidator,
            OBBValidator,
            PointValidator,
            PoseValidator,
            RestoreValidator,
            SemanticValidator,
            SegmentationValidator,
            ValidationConfig,
            MatteValidator,
        )

        if augment:
            raise ValueError(
                "Augmented validation is not supported for exported backends"
            )
        if imgsz is None:
            imgsz = self._get_input_size()
        imgsz = self._resolve_predict_imgsz(imgsz)
        if _is_rectangular_imgsz(imgsz):
            raise NotImplementedError(
                "Rectangular exported-backend validation is not supported yet."
            )
        if plots is not None and "save_plots" not in kwargs:
            kwargs["save_plots"] = plots

        validation_device = device or (
            self.device
            if _is_pytorch_cuda_device(self.device) and torch.cuda.is_available()
            else "cpu"
        )
        config = ValidationConfig(
            data=data,
            batch_size=batch,
            imgsz=imgsz,
            conf_thres=conf,
            iou_thres=iou,
            num_workers=workers,
            allow_download_scripts=allow_download_scripts,
            device=validation_device,
            split=split,
            augment=augment,
            save_json=save_json,
            verbose=verbose,
            **kwargs,
        )
        if self.task == "classify":
            validator_cls = ClassifyValidator
        elif self.task == "point":
            validator_cls = PointValidator
        elif self.task == "segment":
            validator_cls = SegmentationValidator
        elif self.task == "pose":
            validator_cls = PoseValidator
        elif self.task == "obb":
            validator_cls = OBBValidator
        elif self.task == "restore":
            validator_cls = RestoreValidator
        elif self.task == "semantic":
            validator_cls = SemanticValidator
        elif self.task == "depth":
            validator_cls = DepthValidator
        elif self.task == "matte":
            validator_cls = MatteValidator
        elif self.task == "gaze":
            raise NotImplementedError(
                "Exported gaze validation requires a gaze-labelled dataset contract."
            )
        else:
            validator_cls = DetectionValidator
        validator = validator_cls(model=self, config=config)
        return validator()

    # =========================================================================
    # Inference pipeline
    # =========================================================================

    def _predict_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[ImageSize] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        save_stem: Optional[str] = None,
    ) -> Results:
        """Run inference on a single image.

        ``save_stem`` overrides the saved filename stem for in-memory images
        (which have no path to derive one from).
        """
        image_path = image if isinstance(image, (str, Path)) else None
        effective_imgsz = self._resolve_predict_imgsz(imgsz)

        input_tensor, original_img, original_size, ratio = self._preprocess(
            image, effective_imgsz, color_format
        )

        blob = input_tensor.numpy()

        all_outputs = self._run_inference(blob)

        orig_w, orig_h = original_size
        orig_shape = (orig_h, orig_w)
        if self.task == "classify":
            result = self._build_classify_result(
                all_outputs,
                orig_shape=orig_shape,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "restore":
            result = self._build_restore_result(
                all_outputs,
                orig_shape=orig_shape,
                original_size=original_size,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "depth":
            result = self._build_depth_result(
                all_outputs,
                orig_shape=orig_shape,
                original_size=original_size,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "matte":
            result = self._build_matte_result(
                all_outputs,
                orig_shape=orig_shape,
                original_size=original_size,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "gaze":
            result = self._build_gaze_result(
                all_outputs,
                orig_shape=orig_shape,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "semantic":
            result = self._build_semantic_result(
                all_outputs,
                orig_shape=orig_shape,
                original_size=original_size,
                effective_imgsz=effective_imgsz,
                ratio=float(ratio or 1.0),
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "point":
            result = self._build_point_result(
                all_outputs,
                orig_shape=orig_shape,
                original_size=original_size,
                effective_imgsz=effective_imgsz,
                conf=conf,
                max_det=max_det,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result

        parsed = self._parse_outputs(
            all_outputs,
            effective_imgsz,
            original_size,
            conf,
            ratio=ratio,
            iou=iou,
            max_det=max_det,
        )
        boxes, max_scores, class_ids, masks, obb, keypoints = (
            self._unpack_parsed_outputs(parsed)
        )

        result = self._build_result(
            boxes,
            max_scores,
            class_ids,
            masks=masks,
            obb=obb,
            keypoints=keypoints,
            orig_shape=orig_shape,
            image_path=image_path,
            iou=iou,
            classes=classes,
            max_det=max_det,
        )

        if save:
            self._save_annotated(
                result,
                original_img,
                image_path if image_path is not None else save_stem,
                output_path,
            )

        return result

    def _supports_batched_inference(self) -> bool:
        """Whether ``_run_inference`` accepts stacked (N, C, H, W) blobs.

        Default False: traced/compiled runtimes are typically baked to
        batch 1. Backends whose artifact declares a dynamic batch axis
        (ONNX, OpenVINO) override this; TensorRT manages batching itself
        in its own ``_process_in_batches``.
        """
        return False

    def _process_in_batches(
        self,
        images: List,
        batch: int = 1,
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[ImageSize] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
    ) -> List[Results]:
        """Process multiple images (file paths or in-memory).

        When ``batch > 1`` and the runtime accepts stacked blobs, each chunk
        of ``batch`` images runs as a single forward pass; otherwise images
        run sequentially.
        """
        use_batched = (
            batch > 1
            and self._supports_batched_inference()
            # Latched by _predict_batch after a runtime rejects a stacked
            # blob, so a long list does not retry (and warn) once per chunk.
            and not getattr(self, "_batched_inference_failed", False)
        )
        if use_batched:
            results = []
            for start in range(0, len(images), batch):
                results.extend(
                    self._predict_batch(
                        images[start : start + batch],
                        start_idx=start,
                        save=save,
                        output_path=output_path,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        classes=classes,
                        max_det=max_det,
                        color_format=color_format,
                    )
                )
            return results

        results = []
        for idx, image in enumerate(images):
            results.append(
                self._predict_single(
                    image,
                    save=save,
                    output_path=output_path,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    classes=classes,
                    max_det=max_det,
                    color_format=color_format,
                    save_stem=(
                        None if isinstance(image, (str, Path)) else f"image{idx}"
                    ),
                )
            )
        return results

    def _predict_batch(
        self,
        chunk: List,
        start_idx: int,
        *,
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[ImageSize] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
    ) -> List[Results]:
        """Run one stacked forward pass over a chunk of images.

        Mirrors ``_predict_single`` step for step, except the preprocessed
        tensors are concatenated into a single blob for ``_run_inference``
        and the outputs are sliced back per image (``[i : i + 1]`` keeps the
        batch dim, which every output parser already expects). Falls back to
        the sequential path if the blob cannot be stacked or the runtime
        rejects the batched call.
        """
        effective_imgsz = self._resolve_predict_imgsz(imgsz)

        preprocessed = []
        for image in chunk:
            input_tensor, original_img, original_size, ratio = self._preprocess(
                image, effective_imgsz, color_format
            )
            image_path = image if isinstance(image, (str, Path)) else None
            preprocessed.append(
                (input_tensor, original_img, original_size, ratio, image_path)
            )

        tensors = [item[0] for item in preprocessed]
        all_outputs = None
        stackable = all(
            isinstance(t, torch.Tensor) and t.dim() == 4 and t.shape == tensors[0].shape
            for t in tensors
        )
        if stackable and not getattr(self, "_batched_inference_failed", False):
            blob = np.concatenate([t.numpy() for t in tensors], axis=0)
            try:
                all_outputs = self._run_inference(blob)
            except Exception as e:
                self._batched_inference_failed = True
                logger.warning(
                    "Batched inference failed for %s (%s); falling back to "
                    "sequential processing.",
                    Path(self.model_path).name,
                    e,
                )
        if all_outputs is None:
            return [
                self._predict_single(
                    image,
                    save=save,
                    output_path=output_path,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    classes=classes,
                    max_det=max_det,
                    color_format=color_format,
                    save_stem=(
                        None
                        if isinstance(image, (str, Path))
                        else f"image{start_idx + offset}"
                    ),
                )
                for offset, image in enumerate(chunk)
            ]

        results = []
        for offset, (_, original_img, original_size, ratio, image_path) in enumerate(
            preprocessed
        ):
            per_image = [
                np.asarray(output)[offset : offset + 1] for output in all_outputs
            ]
            save_name = (
                image_path if image_path is not None else f"image{start_idx + offset}"
            )
            orig_w, orig_h = original_size
            orig_shape = (orig_h, orig_w)

            if self.task == "classify":
                result = self._build_classify_result(
                    per_image,
                    orig_shape=orig_shape,
                    image_path=image_path,
                )
            elif self.task == "restore":
                result = self._build_restore_result(
                    per_image,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=image_path,
                )
            elif self.task == "depth":
                result = self._build_depth_result(
                    per_image,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=image_path,
                )
            elif self.task == "matte":
                result = self._build_matte_result(
                    per_image,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=image_path,
                )
            elif self.task == "gaze":
                result = self._build_gaze_result(
                    per_image,
                    orig_shape=orig_shape,
                    image_path=image_path,
                )
            elif self.task == "semantic":
                result = self._build_semantic_result(
                    per_image,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    effective_imgsz=effective_imgsz,
                    ratio=float(ratio or 1.0),
                    image_path=image_path,
                )
            elif self.task == "point":
                result = self._build_point_result(
                    per_image,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    effective_imgsz=effective_imgsz,
                    conf=conf,
                    max_det=max_det,
                    image_path=image_path,
                )
            else:
                parsed = self._parse_outputs(
                    per_image,
                    effective_imgsz,
                    original_size,
                    conf,
                    ratio=ratio,
                    iou=iou,
                    max_det=max_det,
                )
                boxes, max_scores, class_ids, masks, obb, keypoints = (
                    self._unpack_parsed_outputs(parsed)
                )
                result = self._build_result(
                    boxes,
                    max_scores,
                    class_ids,
                    masks=masks,
                    obb=obb,
                    keypoints=keypoints,
                    orig_shape=orig_shape,
                    image_path=image_path,
                    iou=iou,
                    classes=classes,
                    max_det=max_det,
                )

            if save:
                self._save_annotated(result, original_img, save_name, output_path)
            results.append(result)
        return results

    # =========================================================================
    # Public API
    # =========================================================================

    def info(self, detailed: bool = False, verbose: bool = True) -> Dict:
        """Return exported-runtime metadata and lightweight counts."""
        data = build_model_info(self, detailed=detailed)
        if verbose:
            logger.info(format_model_info(data))
        return data

    def __call__(
        self,
        source: Union[
            str, Path, Image.Image, np.ndarray, torch.Tensor, list, tuple, None
        ] = None,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[ImageSize] = None,
        device: str | None = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        batch: int = 1,
        # video parameters
        stream: bool = False,
        vid_stride: int = 1,
        show: bool = False,
        output_path: str | None = None,
        color_format: str = "auto",
        **kwargs,
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Run inference on an image, list of images, directory, or video."""
        normalize_predict_kwargs(kwargs)
        classes = validate_predict_inputs(
            names=getattr(self, "names", None),
            conf=conf,
            iou=iou,
            classes=classes,
            max_det=max_det,
            batch=batch,
            vid_stride=vid_stride,
        )
        if device not in (None, "", "auto", self.device):
            logger.warning(
                "Backend was loaded on device=%s; predict(device=%s) is ignored. "
                "Load the backend with device=%s to change runtime device.",
                self.device,
                device,
                device,
            )

        if isinstance(source, (np.ndarray, torch.Tensor)) and source.ndim == 4:
            return self._process_in_batches(
                [source[index] for index in range(source.shape[0])],
                batch=batch,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
            )

        # Handle video input
        if is_video_file(source):
            gen = self._predict_video(
                source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                save=save,
                show=show,
                vid_stride=vid_stride,
                output_path=output_path,
            )
            if stream:
                return gen
            return collect_video_results(gen, source, vid_stride)

        # Handle in-memory batch input (list/tuple of images)
        if isinstance(source, (list, tuple)):
            return self._process_in_batches(
                list(source),
                batch=batch,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
            )

        if isinstance(source, (str, Path)) and Path(source).is_dir():
            image_paths = ImageLoader.collect_images(source)
            if not image_paths:
                return []
            return self._process_in_batches(
                image_paths,
                batch=batch,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
            )

        return self._predict_single(
            source,
            save=save,
            output_path=output_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
            max_det=max_det,
            color_format=color_format,
        )

    def predict(
        self, *args, **kwargs
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Alias for __call__ method."""
        return self(*args, **kwargs)

    def _predict_video(
        self,
        source: Union[str, Path],
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[ImageSize] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        show: bool = False,
        vid_stride: int = 1,
        output_path: Optional[str] = None,
    ) -> Generator[Results, None, None]:
        """Run inference on a video file, yielding per-frame Results."""
        effective_imgsz = self._resolve_predict_imgsz(imgsz)

        def predict_frame(pil_img):
            input_tensor, original_img, original_size, ratio = self._preprocess(
                pil_img, effective_imgsz, "rgb"
            )
            blob = input_tensor.numpy()
            all_outputs = self._run_inference(blob)
            orig_w, orig_h = original_size
            orig_shape = (orig_h, orig_w)
            if self.task == "classify":
                return self._build_classify_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    image_path=str(source),
                )
            if self.task == "restore":
                return self._build_restore_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=str(source),
                )
            if self.task == "depth":
                return self._build_depth_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=str(source),
                )
            if self.task == "matte":
                return self._build_matte_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=str(source),
                )
            if self.task == "gaze":
                return self._build_gaze_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    image_path=str(source),
                )
            if self.task == "semantic":
                return self._build_semantic_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    effective_imgsz=effective_imgsz,
                    ratio=float(ratio or 1.0),
                    image_path=str(source),
                )
            if self.task == "point":
                return self._build_point_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    effective_imgsz=effective_imgsz,
                    conf=conf,
                    max_det=max_det,
                    image_path=str(source),
                )
            parsed = self._parse_outputs(
                all_outputs,
                effective_imgsz,
                original_size,
                conf,
                ratio=ratio,
                iou=iou,
                max_det=max_det,
            )
            boxes, max_scores, class_ids, masks, obb, keypoints = (
                self._unpack_parsed_outputs(parsed)
            )
            return self._build_result(
                boxes,
                max_scores,
                class_ids,
                masks=masks,
                obb=obb,
                keypoints=keypoints,
                orig_shape=orig_shape,
                image_path=str(source),
                iou=iou,
                classes=classes,
                max_det=max_det,
            )

        yield from run_video_inference(
            source,
            predict_frame,
            vid_stride=vid_stride,
            save=save,
            show=show,
            output_path=output_path,
        )
