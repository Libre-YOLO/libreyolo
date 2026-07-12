"""INT8 calibration utilities for TensorRT export."""

import logging
from pathlib import Path
from typing import Iterator, Tuple, Union

import cv2
import numpy as np

from libreyolo.data.utils import load_data_config, get_img_files

logger = logging.getLogger(__name__)

ImageSize = Union[int, Tuple[int, int]]


def _imgsz_hw(imgsz: ImageSize) -> tuple[int, int]:
    if isinstance(imgsz, tuple):
        if len(imgsz) != 2:
            raise ValueError(f"imgsz must be int or (height, width), got {imgsz}")
        h, w = int(imgsz[0]), int(imgsz[1])
    else:
        h = w = int(imgsz)
    if h <= 0 or w <= 0:
        raise ValueError(f"imgsz values must be positive, got {(h, w)}")
    return h, w


class CalibrationDataLoader:
    """
    Calibration data provider for INT8 quantization.

    Loads images from a dataset config and provides batches of preprocessed
    numpy arrays suitable for TensorRT calibration.

    Example::

        calib_loader = CalibrationDataLoader(
            data="coco8.yaml",
            imgsz=640,
            batch=8,
            fraction=0.5,
        )
        for batch in calib_loader:
            # batch is np.ndarray of shape (B, 3, H, W), dtype float32
            ...
    """

    def __init__(
        self,
        data: str,
        imgsz: ImageSize = 640,
        batch: int = 8,
        fraction: float = 1.0,
        preprocess_fn=None,
        allow_download_scripts: bool = False,
        model_family: str | None = None,
        task: str | None = None,
        model_size: str | None = None,
        input_shape: tuple[int, int, int, int] | None = None,
    ):
        """
        Initialize calibration data loader.

        Args:
            data: Path to data.yaml configuration file or built-in dataset name.
            imgsz: Input image size as an int or ``(height, width)`` tuple.
            batch: Batch size for calibration.
            fraction: Fraction of dataset to use (0.0-1.0). Use smaller values
                     for faster calibration with slight accuracy tradeoff.
            preprocess_fn: Callable ``(img_rgb_hwc, input_size) -> (chw_float32, ratio)``.
                Obtained from ``model._get_preprocess_numpy()``.
            allow_download_scripts: Allow embedded Python in dataset YAML downloads.
            model_family: Canonical family used to select task-specific runtime
                preprocessing where the generic model callback is insufficient.
            task: Export task used to select dense/pose runtime preprocessing.
            model_size: Canonical model size (reserved for family runtime contracts).
            input_shape: Concrete NCHW trace shape. Calibration samples are
                validated against its CHW portion before batching.
        """
        if int(batch) < 1:
            raise ValueError(f"Calibration batch must be positive, got {batch}.")
        self.imgsz = imgsz
        self.batch = int(batch)
        self.fraction = max(0.0, min(1.0, fraction))
        self._preprocess_fn = preprocess_fn
        self.model_family = str(model_family or "").lower() or None
        self.task = str(task or "").lower() or None
        self.model_size = model_size

        if input_shape is None:
            input_h, input_w = _imgsz_hw(imgsz)
            self.input_shape = (self.batch, 3, input_h, input_w)
        else:
            self.input_shape = tuple(int(dim) for dim in input_shape)
            if len(self.input_shape) != 4 or any(dim <= 0 for dim in self.input_shape):
                raise ValueError(
                    "Calibration input_shape must be a positive NCHW tuple, got "
                    f"{self.input_shape}."
                )
            if self.input_shape[0] != self.batch or self.input_shape[1] != 3:
                raise ValueError(
                    "Calibration input_shape must match batch and contain three "
                    f"image channels, got batch={self.batch}, shape={self.input_shape}."
                )
        self._sample_shape = self.input_shape[1:]

        if self._preprocess_fn is None and not self._uses_runtime_preprocessor:
            raise ValueError(
                "Calibration requires preprocess_fn for this model family/task so "
                "samples match exported-runtime preprocessing."
            )

        # Load dataset config (handles resolve, download, path resolution)
        data_config = load_data_config(
            data,
            autodownload=True,
            allow_scripts=allow_download_scripts,
        )

        # Get train images (preferred for calibration - more diverse) or val
        root = Path(data_config.get("path", "."))

        # Check for pre-resolved image files (from .txt format datasets)
        if "train_img_files" in data_config:
            self.img_files = [Path(f) for f in data_config["train_img_files"]]
        elif "val_img_files" in data_config:
            self.img_files = [Path(f) for f in data_config["val_img_files"]]
        else:
            # Resolve from directory/file path - prefer train for more diversity
            train_path = data_config.get("train") or data_config.get("val")
            if train_path is None:
                raise ValueError("Dataset config must have 'train' or 'val' key")

            self.img_files = get_img_files(train_path, prefix=str(root))

        if len(self.img_files) == 0:
            raise ValueError(f"No images found in dataset: {data}")

        total = len(self.img_files)
        self.num_samples = max(1, int(total * self.fraction))
        self.img_files = self.img_files[: self.num_samples]
        self._num_batches = (self.num_samples + self.batch - 1) // self.batch

    @property
    def _uses_runtime_preprocessor(self) -> bool:
        return (
            self.task == "depth"
            or self.task == "restore"
            or self.model_family in {"nafnet", "realesrgan", "swinir"}
            or (self.model_family == "yolonas" and self.task == "pose")
        )

    def _coerce_sample(self, sample) -> np.ndarray:
        """Convert a preprocessor result to one contiguous CHW float32 sample."""
        if hasattr(sample, "detach"):
            sample = sample.detach().cpu().numpy()
        array = np.asarray(sample)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            raise ValueError(
                "Calibration preprocessing must return CHW or 1xCHW data, got "
                f"shape {array.shape}."
            )
        if tuple(array.shape) != tuple(self._sample_shape):
            raise ValueError(
                "Calibration preprocessing must match the exported runtime input "
                f"shape {self._sample_shape}, got {tuple(array.shape)}."
            )
        return np.ascontiguousarray(array, dtype=np.float32)

    def _preprocess_runtime(self, img_rgb: np.ndarray):
        """Apply task-specific preprocessing used by exported backends."""
        _, target_h, target_w = self._sample_shape
        if self.task == "depth":
            from libreyolo.backends.base import BaseBackend

            tensor, *_ = BaseBackend._preprocess_depth(
                img_rgb,
                (target_h, target_w),
                "rgb",
            )
            return tensor

        if self.task == "restore" or self.model_family in {
            "nafnet",
            "realesrgan",
            "swinir",
        }:
            from libreyolo.backends.base import BaseBackend

            source_h, source_w = img_rgb.shape[:2]
            if source_h > target_h or source_w > target_w:
                scale = min(target_h / source_h, target_w / source_w)
                resized_h = max(1, min(target_h, int(round(source_h * scale))))
                resized_w = max(1, min(target_w, int(round(source_w * scale))))
                img_rgb = cv2.resize(
                    img_rgb,
                    (resized_w, resized_h),
                    interpolation=cv2.INTER_AREA,
                )
            tensor, *_ = BaseBackend._preprocess_restore(
                img_rgb,
                (target_h, target_w),
                "rgb",
            )
            return tensor

        if self.model_family == "yolonas" and self.task == "pose":
            if target_h != target_w:
                raise ValueError(
                    "YOLO-NAS pose calibration requires a square runtime input, "
                    f"got {(target_h, target_w)}."
                )
            from libreyolo.models.yolonas.utils import preprocess_pose_image

            tensor, *_ = preprocess_pose_image(
                img_rgb,
                input_size=target_h,
                color_format="rgb",
            )
            return tensor

        raise RuntimeError("No built-in calibration runtime preprocessor selected.")

    def _preprocess_array(self, img_rgb: np.ndarray) -> np.ndarray:
        """Preprocess one RGB image and enforce the trace-time CHW contract."""
        if self._uses_runtime_preprocessor:
            result = self._preprocess_runtime(img_rgb)
        else:
            result = self._preprocess_fn(img_rgb, self.imgsz)
            if isinstance(result, tuple):
                result = result[0]
        return self._coerce_sample(result)

    def _preprocess(self, img_path: Path) -> np.ndarray:
        """Read and preprocess one calibration image."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._preprocess_array(img_rgb)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield batches of calibration data as numpy arrays."""
        batch_data = []
        valid_samples = 0

        for img_path in self.img_files:
            try:
                img = self._preprocess(img_path)
                batch_data.append(img)
                valid_samples += 1
            except Exception as e:
                logger.warning("Skipping %s: %s", img_path, e)
                continue

            if len(batch_data) == self.batch:
                yield np.stack(batch_data, axis=0)
                batch_data = []

        # Pad last batch to full size (required by TensorRT)
        if batch_data:
            while len(batch_data) < self.batch:
                batch_data.append(batch_data[-1].copy())
            yield np.stack(batch_data, axis=0)
        elif valid_samples == 0:
            raise RuntimeError(
                "No calibration images matched the exported runtime input contract."
            )

    def __len__(self) -> int:
        """Return number of calibration batches."""
        return self._num_batches

    @property
    def shape(self) -> tuple:
        """Return shape of a single batch: (batch, channels, height, width)."""
        sample_shape = getattr(self, "_sample_shape", None)
        if sample_shape is None:
            h, w = _imgsz_hw(self.imgsz)
            sample_shape = (3, h, w)
        return (self.batch, *sample_shape)

    @property
    def dtype(self) -> np.dtype:
        """Return data type of calibration batches."""
        return np.float32


def get_calibration_dataloader(
    data: str,
    imgsz: ImageSize = 640,
    batch: int = 8,
    fraction: float = 1.0,
    preprocess_fn=None,
    allow_download_scripts: bool = False,
    model_family: str | None = None,
    task: str | None = None,
    model_size: str | None = None,
    input_shape: tuple[int, int, int, int] | None = None,
) -> CalibrationDataLoader:
    """
    Factory function for calibration data loader.

    Args:
        data: Path to data.yaml or built-in dataset name (e.g., "coco8").
        imgsz: Input image size.
        batch: Batch size for calibration.
        fraction: Fraction of dataset to use.
        preprocess_fn: Callable ``(img_rgb_hwc, input_size) -> (chw_float32, ratio)``.
        model_family: Canonical family for runtime preprocessing selection.
        task: Export task for runtime preprocessing selection.
        model_size: Canonical model size.
        input_shape: Concrete NCHW trace shape used for per-sample validation.

    Returns:
        CalibrationDataLoader instance.
    """
    return CalibrationDataLoader(
        data=data,
        imgsz=imgsz,
        batch=batch,
        fraction=fraction,
        preprocess_fn=preprocess_fn,
        allow_download_scripts=allow_download_scripts,
        model_family=model_family,
        task=task,
        model_size=model_size,
        input_shape=input_shape,
    )
