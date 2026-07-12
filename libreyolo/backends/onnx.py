"""ONNX runtime inference backend for LibreYOLO."""

import logging
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from ..utils.general import COCO_CLASSES
from ..utils.serialization import warn_on_metadata_schema_version
from .base import (
    BaseBackend,
    ImageSize,
    MetadataImageSizeError,
    _read_classification_metadata,
    _read_metadata_imgsz,
    _read_pose_metadata,
)

logger = logging.getLogger(__name__)


_ONNX_TENSOR_DTYPES = {
    "tensor(bool)": np.bool_,
    "tensor(double)": np.float64,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
}


def _iter_onnx_tensors(message):
    """Yield every TensorProto nested in an ONNX protobuf message."""
    for field, value in message.ListFields():
        if field.message_type is None:
            continue
        is_repeated = getattr(field, "is_repeated", None)
        if is_repeated is None:
            is_repeated = field.label == field.LABEL_REPEATED
        if is_repeated:
            children = (
                value.values() if field.message_type.GetOptions().map_entry else value
            )
        else:
            children = (value,)
        for child in children:
            descriptor = getattr(child, "DESCRIPTOR", None)
            if descriptor is None:
                continue
            if descriptor.full_name == "onnx.TensorProto":
                yield child
            else:
                yield from _iter_onnx_tensors(child)


def _parse_external_data_integer(
    value: str | None,
    *,
    key: str,
    tensor_name: str,
) -> int:
    """Parse a non-negative ONNX external-data offset or length."""
    if value in {None, ""}:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ONNX external tensor {tensor_name!r} has invalid {key}={value!r}."
        ) from exc
    if parsed < 0:
        raise ValueError(
            f"ONNX external tensor {tensor_name!r} has negative {key}={parsed}."
        )
    return parsed


def _load_validated_onnx_metadata(onnx_path: str) -> dict:
    """Parse metadata and confine every external tensor to the model directory."""
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "Safe ONNX inference requires the 'onnx' package in addition to "
            "onnxruntime. Install with: pip install 'libreyolo[onnx]'"
        ) from exc

    artifact = Path(onnx_path)
    if not artifact.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    try:
        model_proto = onnx.load(str(artifact), load_external_data=False)
    except Exception as exc:
        raise ValueError(f"Failed to parse ONNX model {onnx_path}: {exc}") from exc

    base_dir = artifact.parent.resolve()
    for tensor in _iter_onnx_tensors(model_proto):
        if not tensor.HasField("data_location") or (
            tensor.data_location != onnx.TensorProto.EXTERNAL
        ):
            continue

        tensor_name = tensor.name or "<unnamed>"
        entries = {}
        for entry in tensor.external_data:
            if entry.key in entries:
                raise ValueError(
                    f"ONNX external tensor {tensor_name!r} repeats "
                    f"external_data key {entry.key!r}."
                )
            entries[entry.key] = entry.value

        location = entries.get("location")
        if not location or "\x00" in location:
            raise ValueError(
                f"ONNX external tensor {tensor_name!r} has no valid location."
            )
        windows_path = PureWindowsPath(location)
        normalized_path = PurePosixPath(location.replace("\\", "/"))
        if windows_path.drive or normalized_path.is_absolute():
            raise ValueError(
                f"ONNX external tensor {tensor_name!r} uses an absolute location: "
                f"{location!r}."
            )

        candidate = (base_dir / Path(*normalized_path.parts)).resolve()
        if not candidate.is_relative_to(base_dir):
            raise ValueError(
                f"ONNX external tensor {tensor_name!r} escapes the model directory: "
                f"{location!r}."
            )
        if not candidate.is_file():
            raise FileNotFoundError(
                f"ONNX external tensor data not found for {tensor_name!r}: {candidate}"
            )

        offset = _parse_external_data_integer(
            entries.get("offset"), key="offset", tensor_name=tensor_name
        )
        length = _parse_external_data_integer(
            entries.get("length"), key="length", tensor_name=tensor_name
        )
        file_size = candidate.stat().st_size
        if offset > file_size or (length and length > file_size - offset):
            raise ValueError(
                f"ONNX external tensor {tensor_name!r} references bytes outside "
                f"{candidate.name!r} (offset={offset}, length={length}, "
                f"size={file_size})."
            )

    return {entry.key: entry.value for entry in model_proto.metadata_props}


def _resolve_onnx_providers(device, available_providers) -> tuple[list, str]:
    """Resolve ONNX Runtime providers without warning for an explicit CPU."""
    if isinstance(device, bool):
        raise ValueError(f"Invalid ONNX device {device!r}.")
    if isinstance(device, int):
        key = f"cuda:{device}"
    else:
        key = "auto" if device is None else str(device).strip().lower()
        if key.isdigit():
            key = f"cuda:{key}"
    if key == "auto":
        if "CUDAExecutionProvider" in available_providers:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
        return ["CPUExecutionProvider"], "cpu"
    if key == "cpu":
        return ["CPUExecutionProvider"], "cpu"
    if key in {"cuda", "gpu"} or key.startswith("cuda:"):
        indexed_device = None
        if key.startswith("cuda:"):
            try:
                indexed_device = int(key.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid ONNX CUDA device {device!r}.") from exc
            if indexed_device < 0:
                raise ValueError(f"Invalid ONNX CUDA device {device!r}.")
        if "CUDAExecutionProvider" in available_providers:
            cuda_provider = (
                "CUDAExecutionProvider"
                if indexed_device is None
                else ("CUDAExecutionProvider", {"device_id": indexed_device})
            )
            resolved = "cuda" if indexed_device is None else f"cuda:{indexed_device}"
            return [cuda_provider, "CPUExecutionProvider"], resolved
        logger.warning(
            "Requested device %r but CUDAExecutionProvider is not available; "
            "falling back to CPU.",
            device,
        )
        return ["CPUExecutionProvider"], "cpu"
    logger.warning(
        "Requested device %r is not supported by the ONNX backend; falling back to CPU.",
        device,
    )
    return ["CPUExecutionProvider"], "cpu"


class OnnxBackend(BaseBackend):
    """ONNX runtime inference backend for LibreYOLO models.

    Args:
        onnx_path: Path to the ONNX model file.
        nb_classes: Number of classes (default: 80 for COCO).
        device: Device for inference. "auto" (default) uses CUDA if available, else CPU.

    Example:
        >>> model = OnnxBackend("model.onnx")
        >>> result = model("image.jpg", save=True)
        >>> print(result.boxes.xyxy)
    """

    def __init__(
        self,
        onnx_path: str,
        nb_classes: int = 80,
        device: str = "auto",
        task: str | None = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "ONNX inference requires onnxruntime. "
                "Install with: pip install onnxruntime"
            ) from e

        validated_metadata = _load_validated_onnx_metadata(onnx_path)

        providers, resolved_device = _resolve_onnx_providers(
            device,
            ort.get_available_providers(),
        )

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_dtype = self._numpy_dtype_for_onnx_type(input_info.type)
        self.output_names = [output.name for output in self.session.get_outputs()]
        try:
            runtime_metadata = dict(
                self.session.get_modelmeta().custom_metadata_map or {}
            )
        except Exception:
            runtime_metadata = {}
        metadata = runtime_metadata or validated_metadata

        (
            model_family,
            model_size,
            metadata_task,
            supported_tasks,
            default_task,
            names,
            embedded_nms,
            metadata_imgsz,
        ) = self._read_onnx_metadata(onnx_path, nb_classes, runtime_metadata=metadata)
        pose_metadata = self._read_onnx_pose_metadata(
            onnx_path, runtime_metadata=metadata
        )
        # Models exported with nms=True emit final (1, max_det, 6) detections.
        # Newer YOLO9 ONNX exports also include a raw auxiliary output so the
        # LibreYOLO backend can apply native clipping/NMS for non-square images.
        self.embedded_nms = embedded_nms
        self.embedded_nms_raw_output_index = (
            self.output_names.index("raw")
            if embedded_nms and "raw" in self.output_names
            else None
        )
        input_shape = input_info.shape
        # Dynamic-batch exports carry a symbolic dim ("batch") or None at
        # axis 0; static exports carry an int.
        self._dynamic_batch_axis = bool(input_shape) and not isinstance(
            input_shape[0], int
        )
        dynamic_spatial = len(input_shape) == 4 and any(
            not isinstance(dim, int) for dim in input_shape[2:4]
        )
        static_imgsz = self._read_static_input_imgsz(input_shape)
        if static_imgsz is not None:
            imgsz = static_imgsz
        elif metadata_imgsz is not None:
            imgsz = metadata_imgsz
        else:
            imgsz = 640  # dynamic shape without metadata; use default
        resolved_task = resolve_task(
            explicit_task=task,
            checkpoint_task=metadata_task,
            default_task=default_task,
            supported_tasks=supported_tasks,
        )
        classification_metadata = (
            _read_classification_metadata(metadata)
            if resolved_task == "classify"
            else {}
        )

        super().__init__(
            model_path=onnx_path,
            nb_classes=nb_classes if names is None else len(names),
            device=resolved_device,
            imgsz=imgsz,
            model_family=model_family,
            names=names if names is not None else self.build_names(nb_classes),
            model_size=model_size,
            task=resolved_task,
            supported_tasks=supported_tasks,
            default_task=default_task,
            dynamic_spatial=dynamic_spatial,
            num_bins=(int(metadata["num_bins"]) if metadata.get("num_bins") else None),
            bin_width_deg=(
                float(metadata["bin_width_deg"])
                if metadata.get("bin_width_deg")
                else None
            ),
            offset_deg=(
                float(metadata["offset_deg"]) if metadata.get("offset_deg") else None
            ),
            **classification_metadata,
            **pose_metadata,
        )

    @staticmethod
    def _read_static_input_imgsz(input_shape) -> ImageSize | None:
        if len(input_shape) != 4:
            return None
        h, w = input_shape[2], input_shape[3]
        if not isinstance(h, int) or not isinstance(w, int) or h <= 0 or w <= 0:
            return None
        return h if h == w else (h, w)

    @staticmethod
    def _load_embedded_metadata(onnx_path: str) -> dict:
        """Load the complete metadata map directly from an ONNX artifact."""
        try:
            import onnx

            model_proto = onnx.load(onnx_path, load_external_data=False)
            return {entry.key: entry.value for entry in model_proto.metadata_props}
        except Exception as exc:
            logger.warning("Failed to load ONNX metadata from %s: %s", onnx_path, exc)
            return {}

    @staticmethod
    def _read_onnx_metadata(
        onnx_path: str,
        default_nb_classes: int,
        runtime_metadata: dict | None = None,
    ):
        """Read libreyolo metadata embedded in an ONNX model file.

        Returns:
            Tuple of (model_family, model_size, task, supported_tasks,
            default_task, names, embedded_nms, imgsz).
        """
        model_family = None
        model_size = None
        task = "detect"
        default_task = "detect"
        supported_tasks = ("detect",)
        names = None
        imgsz = None
        embedded_nms = False
        try:
            meta = (
                dict(runtime_metadata)
                if runtime_metadata is not None
                else OnnxBackend._load_embedded_metadata(onnx_path)
            )
            warn_on_metadata_schema_version(
                meta,
                artifact=f"ONNX metadata for {onnx_path}",
                logger=logger,
            )

            if "model_family" in meta:
                model_family = meta["model_family"]
            if "model_size" in meta or "size" in meta:
                model_size = meta.get("model_size") or meta.get("size")
            imgsz = _read_metadata_imgsz(
                meta,
                model_family,
                artifact=f"ONNX metadata for {onnx_path}",
            )
            if "default_task" in meta:
                default_task = normalize_task(meta["default_task"], default="detect")
            if "task" in meta:
                task = normalize_task(meta["task"], default=default_task)
            elif meta.get("segmentation") == "true":
                task = "segment"
            if "supported_tasks" in meta:
                supported_tasks = normalize_supported_tasks(meta["supported_tasks"])
            else:
                supported_tasks = normalize_supported_tasks((task,))

            if "names" in meta:
                import json

                names_raw = json.loads(meta["names"])
                names = {int(k): v for k, v in names_raw.items()}

            if ("nb_classes" in meta or "nc" in meta) and names is None:
                nc = int(meta.get("nb_classes", meta.get("nc")))
                if nc == 80:
                    names = {i: n for i, n in enumerate(COCO_CLASSES)}
                else:
                    names = {i: f"class_{i}" for i in range(nc)}

            embedded_nms = str(meta.get("nms", "")).lower() == "true"
        except (NotImplementedError, MetadataImageSizeError):
            raise
        except Exception as e:
            logger.warning("Failed to read ONNX metadata from %s: %s", onnx_path, e)

        return (
            model_family,
            model_size,
            task,
            supported_tasks,
            default_task,
            names,
            embedded_nms,
            imgsz,
        )

    @staticmethod
    def _read_onnx_pose_metadata(
        onnx_path: str,
        runtime_metadata: dict | None = None,
    ) -> dict:
        try:
            meta = (
                dict(runtime_metadata)
                if runtime_metadata is not None
                else OnnxBackend._load_embedded_metadata(onnx_path)
            )
        except Exception as e:
            logger.warning(
                "Failed to read ONNX pose metadata from %s: %s", onnx_path, e
            )
            return {}

        return _read_pose_metadata(meta)

    def _supports_batched_inference(self) -> bool:
        # Embedded-NMS graphs are exported batch-1; everything else with a
        # dynamic batch axis accepts stacked blobs directly.
        return self._dynamic_batch_axis and not self.embedded_nms

    @staticmethod
    def _numpy_dtype_for_onnx_type(type_name: str) -> np.dtype:
        """Translate an ONNX Runtime tensor type into its NumPy dtype."""
        try:
            return np.dtype(_ONNX_TENSOR_DTYPES[type_name])
        except KeyError as exc:
            raise TypeError(
                "Unsupported ONNX Runtime input type "
                f"{type_name!r}; this backend cannot construct a matching NumPy input."
            ) from exc

    def _run_inference(self, blob: np.ndarray) -> list:
        """Run ONNX Runtime inference."""
        runtime_blob = np.ascontiguousarray(blob, dtype=self.input_dtype)
        return self.session.run(None, {self.input_name: runtime_blob})
