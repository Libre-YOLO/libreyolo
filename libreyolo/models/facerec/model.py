"""LibreYOLO face-embedding model (task ``embed`` / facial-recognition).

Inference-only, two-stage and ONNX-backed — structurally a sibling of the gaze
(L2CS) task: a face detector locates faces (and ideally 5 landmarks), each face
is aligned to a canonical 112x112 crop, and a recognition head produces an
L2-normalized identity embedding. Verification/identification is then a cosine
similarity (== dot product of unit vectors).

The recognition head is consumed as an opaque ONNX graph via onnxruntime, so no
third-party architecture code is ported. Training and dataset validation are out
of scope here (exactly like L2CS) — embeddings are an inference product.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ..l2cs.face import FaceBox, resolve_face_detector
from .preprocess import PreprocCfg, l2_normalize, preprocess_aligned, resolve_preproc

logger = logging.getLogger(__name__)


class LibreFaceEmbedder:
    """Face-embedding model: image -> per-face (box + L2-normalized embedding).

    Args:
        model_path: Path to a face-recognition ONNX model that maps an
            ``(N, 3, 112, 112)`` aligned-face batch to ``(N, D)`` embeddings.
        preproc: A ``PreprocCfg``, a preset name (``"arcface"`` | ``"raw_bgr"``),
            or ``None`` (defaults to the ArcFace convention: RGB,
            ``(x-127.5)/127.5``).
        device: ``"auto"`` | ``"cpu"`` | ``"cuda"``.
        face_detector: Optional default detector (a callable, a LibreYOLO
            detection model, or a ``FaceDetector``). May also be supplied per
            call, or bypassed entirely with ``face_boxes=...``.
    """

    FAMILY = "facerec"
    FILENAME_PREFIX = "librefacerec-"
    INPUT_SIZES = {"l": 112}
    TASK_INPUT_SIZES = {}
    SUPPORTED_TASKS = ("embed",)
    DEFAULT_TASK = "embed"

    def __init__(
        self,
        model_path: str,
        *,
        preproc: "PreprocCfg | str | None" = None,
        device: str = "auto",
        face_detector: Any = None,
        names: Optional[dict] = None,
        task: str | None = "embed",
        compute_units: str = "all",
    ):
        from ...tasks import normalize_task

        resolved_task = normalize_task(task or self.DEFAULT_TASK)
        if resolved_task != "embed":
            raise ValueError(
                "LibreFaceEmbedder supports only the 'embed' "
                f"(facial-recognition) task, got task={task!r}."
            )

        if not Path(model_path).exists():
            from .weights import is_facerec_weight_name, resolve_facerec_weight

            if is_facerec_weight_name(model_path):
                model_path = resolve_facerec_weight(model_path)
            else:
                raise FileNotFoundError(
                    f"Face-embedding model not found: {model_path}"
                )

        path = Path(model_path)
        is_coreml = path.is_dir() and path.suffix.lower() == ".mlpackage"
        if path.is_file() and path.suffix.lower() == ".onnx":
            from .weights import verify_facerec_weight_file

            verify_facerec_weight_file(path)
        explicit_cfg = None if preproc is None else resolve_preproc(preproc)
        if is_coreml:
            from ...backends.coreml_facerec import CoreMLFaceSession

            self.session = CoreMLFaceSession(
                str(path),
                compute_units=compute_units,
            )
            payload = self.session.preprocess
            artifact_cfg = PreprocCfg(
                size=int(payload["size"]),
                color_order=str(payload["color_order"]),
                mean=float(payload["mean"]),
                scale=float(payload["scale"]),
                layout=str(payload["layout"]),
            )
            if explicit_cfg is not None and explicit_cfg != artifact_cfg:
                raise ValueError(
                    "Explicit face preprocessing conflicts with the Core ML "
                    f"artifact contract: {explicit_cfg!r} != {artifact_cfg!r}."
                )
            self.cfg = artifact_cfg
            self.device = "coreml"
            self.input_name = self.session.get_inputs()[0].name
            self._dim = int(self.session.embedding_dim)
            self.size = str(self.session.metadata.get("size", "custom"))
        else:
            if path.suffix.lower() != ".onnx" or not path.is_file():
                raise ValueError(
                    "LibreFaceEmbedder requires an ONNX file or a face "
                    f"Core ML .mlpackage, got {path}."
                )
            try:
                import onnxruntime as ort
            except ImportError as e:  # pragma: no cover - env dependent
                raise ImportError(
                    "Face embedding requires onnxruntime. "
                    'Install with: pip install "libreyolo[onnx]"'
                ) from e

            available = ort.get_available_providers()
            if (
                device in ("auto", "cuda", "gpu")
                and "CUDAExecutionProvider" in available
            ):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.device = "cuda"
            else:
                providers = ["CPUExecutionProvider"]
                self.device = "cpu"

            so = ort.SessionOptions()
            so.log_severity_level = 3
            self.session = ort.InferenceSession(
                str(path),
                sess_options=so,
                providers=providers,
            )
            self.input_name = self.session.get_inputs()[0].name
            self.cfg = explicit_cfg or PreprocCfg.arcface()
            out_dim = self.session.get_outputs()[0].shape[-1]
            self._dim = int(out_dim) if isinstance(out_dim, int) else None
            self.size = (
                "l"
                if path.name.lower() == "librefacerec-l.onnx"
                else "custom"
            )

        self.model_path = str(path)
        self.family = self.FAMILY
        self.task = "embed"
        self.names = names or {0: "face"}
        self.face_detector = (
            resolve_face_detector(face_detector) if face_detector is not None else None
        )
        self._default_detector_instance = None

    # ------------------------------------------------------------------
    @property
    def dim(self) -> Optional[int]:
        """Embedding dimension (``None`` if an ONNX graph declares it dynamic)."""
        return self._dim

    def _get_model_name(self) -> str:
        return self.FAMILY

    def _get_input_size(self) -> int:
        return int(self.cfg.size)

    @classmethod
    def get_download_url(cls, filename: str) -> str | None:
        from .weights import FACEREC_WEIGHT_URLS

        return FACEREC_WEIGHT_URLS.get(Path(filename).name.lower())

    def embed_aligned(self, aligned_crops: List[np.ndarray]) -> np.ndarray:
        """List of aligned HxWx3 RGB uint8 crops -> ``(N, D)`` L2-normalized."""
        if not aligned_crops:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        outs = []
        for crop in aligned_crops:
            blob = preprocess_aligned(crop, self.cfg)
            out = self.session.run(None, {self.input_name: blob})[0]
            outs.append(np.asarray(out, dtype=np.float32).reshape(1, -1))
        emb = np.concatenate(outs, axis=0)
        self._dim = emb.shape[1]
        return l2_normalize(emb, axis=1)

    def default_face_detector(self):
        """The family's default detector (auto-downloaded on first use)."""
        if self._default_detector_instance is None:
            from .weights import default_face_detector

            self._default_detector_instance = default_face_detector()
        return self._default_detector_instance

    # ------------------------------------------------------------------
    @property
    def _runner(self):
        if getattr(self, "_runner_instance", None) is None:
            from .inference import FaceEmbedRunner

            self._runner_instance = FaceEmbedRunner(self)
        return self._runner_instance

    def __call__(self, source=None, **kwargs):
        return self._runner(source, **kwargs)

    def predict(self, *args, **kwargs):
        return self(*args, **kwargs)

    def verify(
        self,
        image_a,
        image_b,
        *,
        threshold: float = 0.4,
        face_detector: Any = None,
        face_boxes_a: Any = None,
        face_boxes_b: Any = None,
    ) -> dict:
        """Compare the most prominent face in two images.

        Returns ``{'similarity', 'same_person', 'threshold'}`` where similarity
        is cosine similarity in ``[-1, 1]``.
        """
        ra = self(image_a, face_detector=face_detector, face_boxes=face_boxes_a)
        rb = self(image_b, face_detector=face_detector, face_boxes=face_boxes_b)
        ea = _top_embedding(ra)
        eb = _top_embedding(rb)
        if ea is None or eb is None:
            raise RuntimeError(
                "verify() needs at least one face in each image "
                f"(image_a faces={0 if ea is None else 1}, image_b faces={0 if eb is None else 1})."
            )
        sim = float(np.dot(ea, eb))
        return {"similarity": sim, "same_person": sim >= threshold, "threshold": threshold}

    # ------------------------------------------------------------------
    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training is out of scope for LibreFaceEmbedder (inference-only). "
            "Train a recognition head upstream and export it to ONNX."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            "Dataset validation is out of scope for LibreFaceEmbedder. Evaluate "
            "verification accuracy with the `compare` API on labeled pairs."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        normalized = str(format).strip().lower()
        if normalized not in {"coreml", "mlpackage"}:
            raise NotImplementedError(
                "LibreFaceEmbedder supports only the mechanical Core ML "
                "deployment conversion of its loaded ONNX graph."
            )
        from ...export.coreml_facerec import export_facerec_coreml

        return export_facerec_coreml(self, kwargs)


def _top_embedding(result) -> Optional[np.ndarray]:
    """Highest-confidence face embedding from a Results, or None."""
    emb = getattr(result, "embeddings", None)
    if emb is None or len(emb) == 0:
        return None
    data = emb.numpy().data if hasattr(emb, "numpy") else emb.data
    data = np.asarray(data, dtype=np.float32)
    boxes = result.boxes
    if boxes is not None and len(boxes) == len(data):
        conf = np.asarray(boxes.numpy().conf, dtype=np.float32)
        idx = int(np.argmax(conf)) if conf.size else 0
    else:
        idx = 0
    return l2_normalize(data[idx])


class OpenCVFaceDetector:
    """Face detector + 5 landmarks via OpenCV's bundled ``cv2.FaceDetectorYN``.

    Returns ``FaceBox`` with 5-point landmarks, ready for canonical alignment.
    The detector API ships with OpenCV (a core dependency, Apache-2.0); the
    ``.onnx`` model file is supplied by the caller. Lazy — cv2/model are only
    touched on first use.
    """

    def __init__(self, model_path: str, score_threshold: float = 0.6, nms_threshold: float = 0.3):
        self.model_path = str(model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._impl = None

    def _load(self):
        if self._impl is None:
            import cv2

            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Face-detector model not found: {self.model_path}")
            self._impl = cv2.FaceDetectorYN.create(
                self.model_path, "", (320, 320),
                score_threshold=self.score_threshold, nms_threshold=self.nms_threshold,
            )

    def __call__(self, image_rgb: np.ndarray) -> List[FaceBox]:
        self._load()
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        h, w = bgr.shape[:2]
        self._impl.setInputSize((w, h))
        _, faces = self._impl.detect(bgr)
        if faces is None:
            return []
        out: List[FaceBox] = []
        for f in faces:
            x, y, bw, bh = f[:4]
            out.append(
                FaceBox(
                    xyxy=(float(x), float(y), float(x + bw), float(y + bh)),
                    score=float(f[-1]),
                    landmarks=f[4:14].reshape(5, 2).astype(np.float32),
                )
            )
        return out
