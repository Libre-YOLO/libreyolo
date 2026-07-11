"""Face-detector layer for L2CS gaze inference.

L2CS-Net needs face bounding boxes upstream of the gaze head. Rather than
bundling a specific face detector as a hard dependency, this module defines
a small ``FaceDetector`` protocol and four adapters:

* ``CallableFaceDetector`` — wraps any ``image -> list[FaceBox]`` callable.
* ``LibreYOLOFaceDetector`` — adapts an existing LibreYOLO detector (e.g.
  a YOLO9 model fine-tuned on faces) into the protocol.
* ``HaarCascadeFaceDetector`` — OpenCV 4's bundled frontal-face Haar cascade;
  fully offline (``face_detector="haar"``). OpenCV 5 removed this API.
* ``YuNetFaceDetector`` — OpenCV's ``FaceDetectorYN``; one-time ~230 KB model
  fetch from the official OpenCV zoo, MIT (``face_detector="yunet"``).
  ``default_face_detector()`` picks Haar on OpenCV 4 and YuNet on OpenCV 5,
  and is the automatic fallback when no other face source is given
  (``face_detector="auto"``).
* ``RetinaFaceAdapter`` — optional, lazy-imports ``face_detection`` for
  parity with upstream L2CS-Net's pipeline. Behind the ``gaze-retinaface``
  optional extra; never imported eagerly.

Callers may also pre-compute face boxes and pass them directly to
``LibreL2CS(...)``, bypassing the protocol entirely (the cleanest path
for composition with external detectors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol

import numpy as np


@dataclass
class FaceBox:
    """A single face detection.

    Attributes
    ----------
    xyxy : tuple
        ``(x1, y1, x2, y2)`` in pixel coordinates of the source image.
    score : float
        Detector confidence in [0, 1]. Used for filtering only.
    landmarks : np.ndarray | None
        Optional ``(K, 2)`` landmarks (e.g. 5-pt for RetinaFace). May be None.
    """

    xyxy: tuple
    score: float = 1.0
    landmarks: Optional[np.ndarray] = None


class FaceDetector(Protocol):
    """Anything callable that maps a numpy RGB image to a list of ``FaceBox``."""

    def __call__(self, image_rgb: np.ndarray) -> List[FaceBox]:  # pragma: no cover - protocol
        ...


@dataclass
class CallableFaceDetector:
    """Adapt an arbitrary ``image -> iterable`` callable into a ``FaceDetector``.

    The callable may return any of:

    * a list of ``FaceBox`` instances,
    * a list of ``(x1, y1, x2, y2)`` tuples (score defaults to 1.0),
    * a list of ``(x1, y1, x2, y2, score)`` tuples,
    * a numpy array of shape ``(N, 4)`` or ``(N, 5)``.
    """

    fn: Callable[[np.ndarray], Any]
    min_score: float = 0.0

    def __call__(self, image_rgb: np.ndarray) -> List[FaceBox]:
        raw = self.fn(image_rgb)
        return _normalize_boxes(raw, self.min_score)


@dataclass
class LibreYOLOFaceDetector:
    """Adapt a LibreYOLO detector that emits face boxes into a ``FaceDetector``.

    Any LibreYOLO model that returns ``Results`` with ``.boxes.xyxy`` works.
    By default all detected boxes are treated as faces; pass ``face_class``
    to filter to a specific class id when the underlying model is multi-class.
    """

    model: Any
    conf: float = 0.4
    face_class: Optional[int] = None
    imgsz: Optional[int] = None

    def __call__(self, image_rgb: np.ndarray) -> List[FaceBox]:
        # Avoid the import cost when not used
        from PIL import Image

        pil = Image.fromarray(image_rgb)
        kwargs = {"conf": self.conf}
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        if self.face_class is not None:
            kwargs["classes"] = [self.face_class]
        result = self.model(pil, **kwargs)
        if isinstance(result, list):
            result = result[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
        conf = boxes.conf
        if hasattr(conf, "cpu"):
            conf = conf.cpu().numpy()
        return [
            FaceBox(xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                    score=float(s))
            for b, s in zip(xyxy, conf)
        ]


@dataclass
class HaarCascadeFaceDetector:
    """Zero-download frontal-face detector using OpenCV 4's bundled Haar cascade.

    opencv-python 4.x ships both the ``CascadeClassifier`` API and the cascade
    XML, so this works fully offline. OpenCV 5 removed the Haar API from the
    main wheel; there, use :class:`YuNetFaceDetector` (what
    :func:`default_face_detector` picks automatically). Quality is below a
    learned detector (frontal faces only, no landmarks, no confidence scores).
    """

    min_size_frac: float = 0.05  # smallest face, as a fraction of min(H, W)
    scale_factor: float = 1.1
    min_neighbors: int = 5
    _impl: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._impl is not None:
            return
        import cv2

        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                f"OpenCV {cv2.__version__} does not include the Haar cascade "
                "API (removed from the main wheel in OpenCV 5). Use "
                "face_detector='yunet' or default_face_detector() instead."
            )
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        impl = cv2.CascadeClassifier(str(cascade_path))
        if impl.empty():
            raise RuntimeError(
                f"Failed to load OpenCV Haar cascade from {cascade_path}."
            )
        self._impl = impl

    def __call__(self, image_rgb: np.ndarray) -> List[FaceBox]:
        self._load()
        import cv2

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        min_side = int(min(gray.shape[:2]) * self.min_size_frac)
        faces = self._impl.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(max(24, min_side), max(24, min_side)),
        )
        return [
            FaceBox(xyxy=(float(x), float(y), float(x + w), float(y + h)), score=1.0)
            for (x, y, w, h) in faces
        ]


_YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
# Official OpenCV model zoo copy (Git LFS, hence the media host). MIT license
# (Shiqi Yu, libfacedetection): https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE
_YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/" + _YUNET_FILENAME
)


@dataclass
class YuNetFaceDetector:
    """Face detector using OpenCV's ``FaceDetectorYN`` (YuNet).

    Works on both OpenCV 4.5.4+ and OpenCV 5. The ~230 KB YuNet ONNX model is
    not bundled with opencv-python; it is fetched once from the official
    OpenCV model zoo (MIT license) into ``weights/`` unless ``model_path``
    points at an existing copy.
    """

    model_path: Optional[str] = None
    score_threshold: float = 0.6
    nms_threshold: float = 0.3
    _impl: Any = field(default=None, init=False, repr=False)

    def _resolve_model(self) -> Path:
        if self.model_path is not None:
            path = Path(self.model_path)
            if not path.exists():
                raise FileNotFoundError(f"YuNet model not found: {path}")
            return path
        dest = Path("weights") / _YUNET_FILENAME
        if dest.exists():
            return dest
        import logging

        import requests

        logging.getLogger(__name__).info(
            "Downloading the YuNet face detector (~230 KB, MIT license) from "
            "the OpenCV model zoo to %s ...", dest
        )
        resp = requests.get(_YUNET_URL, timeout=60)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".onnx.part")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)
        return dest

    def _load(self) -> None:
        if self._impl is not None:
            return
        import cv2

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                f"OpenCV {cv2.__version__} does not include FaceDetectorYN "
                "(needs opencv-python >= 4.5.4)."
            )
        model = self._resolve_model()
        self._impl = cv2.FaceDetectorYN.create(
            str(model), "", (320, 320), self.score_threshold, self.nms_threshold
        )

    def __call__(self, image_rgb: np.ndarray) -> List[FaceBox]:
        self._load()
        h, w = image_rgb.shape[:2]
        self._impl.setInputSize((w, h))
        _, faces = self._impl.detect(image_rgb[:, :, ::-1].copy())  # expects BGR
        if faces is None:
            return []
        out: List[FaceBox] = []
        for row in faces:
            x, y, bw, bh = (float(v) for v in row[:4])
            out.append(
                FaceBox(
                    xyxy=(x, y, x + bw, y + bh),
                    score=float(row[14]),
                    landmarks=np.asarray(row[4:14], dtype=np.float32).reshape(5, 2),
                )
            )
        return out


def default_face_detector() -> FaceDetector:
    """Best zero-configuration face detector for the installed OpenCV.

    OpenCV 4 wheels bundle the Haar cascade, so that path needs no download at
    all; OpenCV 5 removed it, so YuNet (one-time ~230 KB fetch) is used there.
    """
    import cv2

    if hasattr(cv2, "CascadeClassifier"):
        return HaarCascadeFaceDetector()
    return YuNetFaceDetector()


@dataclass
class RetinaFaceAdapter:
    """Optional adapter around ``face_detection.RetinaFace`` for upstream parity.

    Requires manually installing the git-only ``face_detection`` package. Not
    imported eagerly.
    """

    min_score: float = 0.5
    gpu_id: Optional[int] = None
    _impl: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._impl is not None:
            return
        try:
            from face_detection import RetinaFace
        except ImportError as e:  # pragma: no cover - depends on env
            raise ImportError(
                "RetinaFaceAdapter requires the optional 'face_detection' package. "
                "Install it manually from https://github.com/elliottzheng/face-detection "
                "or pass another face detector."
            ) from e
        self._impl = RetinaFace() if self.gpu_id is None else RetinaFace(gpu_id=self.gpu_id)

    def __call__(self, image_rgb: np.ndarray) -> List[FaceBox]:
        self._load()
        # face_detection expects BGR
        image_bgr = image_rgb[:, :, ::-1].copy()
        faces = self._impl(image_bgr)
        if faces is None:
            return []
        out: List[FaceBox] = []
        for box, landmark, score in faces:
            if float(score) < self.min_score:
                continue
            out.append(
                FaceBox(
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    score=float(score),
                    landmarks=np.asarray(landmark, dtype=np.float32),
                )
            )
        return out


def _normalize_boxes(raw: Any, min_score: float) -> List[FaceBox]:
    """Coerce a flexible callable's output into a list of FaceBox."""
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        return _from_array(raw, min_score)
    boxes: List[FaceBox] = []
    for item in raw:
        if isinstance(item, FaceBox):
            if item.score >= min_score:
                boxes.append(item)
            continue
        seq = list(item)
        if len(seq) == 4:
            boxes.append(FaceBox(xyxy=(float(seq[0]), float(seq[1]),
                                       float(seq[2]), float(seq[3])),
                                 score=1.0))
        elif len(seq) >= 5:
            score = float(seq[4])
            if score < min_score:
                continue
            boxes.append(FaceBox(xyxy=(float(seq[0]), float(seq[1]),
                                       float(seq[2]), float(seq[3])),
                                 score=score))
        else:
            raise ValueError(
                f"Unsupported face-detector tuple length {len(seq)}; expected 4 or 5+."
            )
    return boxes


def _from_array(arr: np.ndarray, min_score: float) -> List[FaceBox]:
    if arr.ndim != 2 or arr.shape[1] not in (4, 5):
        raise ValueError(
            f"Expected face-detector array of shape (N, 4) or (N, 5), got {arr.shape}."
        )
    boxes: List[FaceBox] = []
    for row in arr:
        if arr.shape[1] == 5:
            score = float(row[4])
            if score < min_score:
                continue
            boxes.append(FaceBox(xyxy=(float(row[0]), float(row[1]),
                                       float(row[2]), float(row[3])),
                                 score=score))
        else:
            boxes.append(FaceBox(xyxy=(float(row[0]), float(row[1]),
                                       float(row[2]), float(row[3])),
                                 score=1.0))
    return boxes


def resolve_face_detector(spec: Any) -> Optional[FaceDetector]:
    """Coerce a user-supplied detector spec into a ``FaceDetector`` or None.

    Accepts: None (no detector), a ``FaceDetector`` instance, a LibreYOLO
    detection model, or a plain callable ``image -> boxes``.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        name = spec.lower()
        if name == "auto":
            return default_face_detector()
        if name == "haar":
            return HaarCascadeFaceDetector()
        if name == "yunet":
            return YuNetFaceDetector()
        raise ValueError(
            f"Unknown face_detector name {spec!r}; named detectors are "
            "'auto', 'haar', and 'yunet'."
        )
    if isinstance(
        spec,
        (
            CallableFaceDetector,
            HaarCascadeFaceDetector,
            YuNetFaceDetector,
            LibreYOLOFaceDetector,
            RetinaFaceAdapter,
        ),
    ):
        return spec
    # A LibreYOLO detection model — wrap so its boxes feed the gaze head.
    # Detected precisely via the base class rather than duck-typing on a
    # ``.predict`` attribute, which unrelated objects also expose.
    from ..base.model import BaseModel

    if isinstance(spec, BaseModel):
        return LibreYOLOFaceDetector(model=spec)
    if callable(spec):
        return CallableFaceDetector(fn=spec)
    raise TypeError(
        f"Unsupported face_detector spec: {type(spec).__name__}. "
        "Provide a callable, a LibreYOLO model, or a FaceDetector instance."
    )
