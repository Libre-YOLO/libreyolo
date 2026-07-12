"""TensorRT inference backend for LibreYOLO."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from ..utils.serialization import warn_on_metadata_schema_version
from .base import (
    BaseBackend,
    ImageSize,
    _read_classification_metadata,
    _read_metadata_imgsz,
    _read_pose_metadata,
)

logger = logging.getLogger(__name__)


def _resolve_tensorrt_device(device) -> torch.device:
    """Resolve and validate the CUDA device used by a TensorRT runtime."""
    if isinstance(device, bool):
        raise ValueError(f"Invalid TensorRT CUDA device {device!r}.")
    if isinstance(device, int):
        index = device
    else:
        key = "auto" if device is None else str(device).strip().lower()
        if key in {"auto", "cuda", "gpu"}:
            index = int(torch.cuda.current_device())
        elif key.isdigit():
            index = int(key)
        elif key.startswith("cuda:"):
            try:
                index = int(key.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid TensorRT CUDA device {device!r}.") from exc
        else:
            raise ValueError(
                "TensorRT inference requires device=0, device='0', 'cuda', "
                "'cuda:N', 'gpu', or 'auto'; "
                f"got {device!r}."
            )
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"TensorRT CUDA device index {index} is unavailable; "
            f"detected {torch.cuda.device_count()} CUDA device(s)."
        )
    return torch.device("cuda", index)


class TensorRTBackend(BaseBackend):
    """TensorRT inference backend for LibreYOLO models.

    Args:
        engine_path: Path to the TensorRT engine file (.engine).
            If a JSON sidecar file exists at ``<engine_path>.json``, model
            metadata (nb_classes, class names, model family, etc.) is loaded
            from it automatically.
        nb_classes: Number of classes. When ``None`` (default), uses the value
            from the sidecar file if available, otherwise defaults to 80.
        device: CUDA device for inference. Accepts an integer index, numeric
            string, ``"cuda"``, ``"cuda:N"``, ``"gpu"``, ``"auto"``, or a
            CUDA ``torch.device``.

    Example:
        >>> model = TensorRTBackend("model.engine")
        >>> result = model("image.jpg", save=True)
        >>> print(result.boxes.xyxy)
    """

    def __init__(
        self,
        engine_path: str,
        nb_classes: int | None = None,
        device: str | int | torch.device = "auto",
        task: str | None = None,
    ):
        try:
            import tensorrt as trt
        except ImportError as e:
            raise ImportError(
                "TensorRT inference requires tensorrt. "
                "Install with: pip install tensorrt"
            ) from e

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT requires CUDA. No CUDA-capable GPU detected.")
        resolved_device = _resolve_tensorrt_device(device)
        torch.cuda.set_device(resolved_device)
        self.device = resolved_device

        if not Path(engine_path).exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.model_path = str(engine_path)
        sidecar_path = Path(str(engine_path) + ".json")
        self._metadata = {}
        if sidecar_path.exists():
            with open(sidecar_path) as f:
                self._metadata = json.load(f)
        warn_on_metadata_schema_version(
            self._metadata,
            artifact=f"TensorRT metadata sidecar {sidecar_path}",
            logger=logger,
        )

        # Priority: explicit arg > sidecar > default (80)
        resolved_nb_classes = (
            nb_classes
            if nb_classes is not None
            else self._metadata.get("nb_classes", self._metadata.get("nc", 80))
        )
        model_family = self._metadata.get("model_family")
        default_task = normalize_task(
            self._metadata.get("default_task"), default="detect"
        )
        metadata_task = normalize_task(self._metadata.get("task"), default=default_task)
        supported_tasks = normalize_supported_tasks(
            self._metadata.get("supported_tasks", (metadata_task,))
        )
        pose_metadata = _read_pose_metadata(self._metadata)
        self._sidecar_size = self._metadata.get("model_size") or self._metadata.get(
            "size"
        )

        sidecar_names = self._metadata.get("names")
        if sidecar_names is not None and nb_classes is None:
            names: Dict[int, str] = {int(k): v for k, v in sidecar_names.items()}
        else:
            names = self.build_names(resolved_nb_classes)

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_names: List[str] = []
        self.input_shape = None
        self.output_shapes: Dict[str, Tuple] = {}
        self.input_numpy_dtype = None
        self.input_torch_dtype = None
        self.output_numpy_dtypes: Dict[str, np.dtype] = {}
        self.output_torch_dtypes: Dict[str, torch.dtype] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)
            numpy_dtype, torch_dtype = self._binding_dtypes(
                trt,
                self.engine.get_tensor_dtype(name),
                tensor_name=name,
            )

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = tuple(shape)
                self.input_numpy_dtype = numpy_dtype
                self.input_torch_dtype = torch_dtype
            else:
                self.output_names.append(name)
                self.output_shapes[name] = tuple(shape)
                self.output_numpy_dtypes[name] = numpy_dtype
                self.output_torch_dtypes[name] = torch_dtype

        if self.input_name is None:
            raise RuntimeError("No input tensor found in TensorRT engine")

        self._dynamic_input = any(dim == -1 for dim in self.input_shape)
        self._dynamic_batch = self.input_shape[0] == -1  # -1 = dynamic batch
        self._min_batch, self._max_batch = self._detect_batch_limits()
        dynamic_spatial = self._detect_dynamic_spatial_profile()

        if model_family is None:
            model_family = self._detect_model_family()
        metadata_imgsz = _read_metadata_imgsz(
            self._metadata,
            model_family,
            artifact=f"TensorRT metadata sidecar {sidecar_path}",
        )
        imgsz = self._read_static_input_imgsz(self.input_shape) or metadata_imgsz or 640
        self._allocate_buffers(self._initial_input_shape(imgsz))
        if not self._metadata:
            inferred_task = self._detect_task_from_filename()
            if inferred_task is not None:
                metadata_task = inferred_task
                default_task = inferred_task
                supported_tasks = (inferred_task,)

        resolved_task = resolve_task(
            explicit_task=task,
            checkpoint_task=metadata_task,
            default_task=default_task,
            supported_tasks=supported_tasks,
        )
        classification_metadata = (
            _read_classification_metadata(self._metadata)
            if resolved_task == "classify"
            else {}
        )

        super().__init__(
            model_path=engine_path,
            nb_classes=resolved_nb_classes,
            device=str(resolved_device),
            imgsz=imgsz,
            model_family=model_family,
            names=names,
            model_size=self._sidecar_size,
            task=resolved_task,
            supported_tasks=supported_tasks,
            default_task=default_task,
            dynamic_spatial=dynamic_spatial,
            **classification_metadata,
            **pose_metadata,
        )

    # =========================================================================
    # TensorRT-specific internals
    # =========================================================================

    @staticmethod
    def _read_static_input_imgsz(input_shape) -> ImageSize | None:
        if len(input_shape) != 4:
            return None
        h, w = input_shape[2], input_shape[3]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return h if h == w else (h, w)
        return None

    def _detect_dynamic_spatial_profile(self) -> bool:
        """Return whether the active TensorRT profile varies height or width."""
        if len(self.input_shape) != 4 or not any(
            dim == -1 for dim in self.input_shape[2:4]
        ):
            return False
        try:
            min_shape, _, max_shape = self.engine.get_tensor_profile_shape(
                self.input_name, 0
            )
            return any(
                int(min_shape[index]) != int(max_shape[index]) for index in (2, 3)
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    @staticmethod
    def _binding_dtypes(trt, trt_dtype, *, tensor_name: str):
        """Return NumPy and torch dtypes for a TensorRT I/O tensor."""
        try:
            numpy_dtype = np.dtype(trt.nptype(trt_dtype))
            torch_dtype = torch.from_numpy(np.empty((0,), dtype=numpy_dtype)).dtype
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                f"TensorRT tensor {tensor_name!r} uses unsupported dtype {trt_dtype!r}."
            ) from exc
        return numpy_dtype, torch_dtype

    def _initial_input_shape(self, imgsz: ImageSize) -> tuple[int, ...]:
        """Resolve a concrete startup shape, preferring profile-opt dimensions."""
        if not self._dynamic_input:
            return tuple(int(dim) for dim in self.input_shape)

        try:
            profile_shapes = self.engine.get_tensor_profile_shape(self.input_name, 0)
            opt_shape = tuple(int(dim) for dim in profile_shapes[1])
            if len(opt_shape) == len(self.input_shape) and all(
                dim > 0 for dim in opt_shape
            ):
                return opt_shape
        except (AttributeError, IndexError, TypeError, ValueError):
            pass

        if len(self.input_shape) != 4:
            raise RuntimeError(
                "TensorRT dynamic input has no usable optimization-profile shape: "
                f"{self.input_shape}."
            )
        if isinstance(imgsz, tuple):
            input_h, input_w = (int(imgsz[0]), int(imgsz[1]))
        else:
            input_h = input_w = int(imgsz)
        fallback = (1, 3, input_h, input_w)
        return tuple(
            int(engine_dim) if engine_dim > 0 else fallback[index]
            for index, engine_dim in enumerate(self.input_shape)
        )

    def _validate_input_shape(self, shape: tuple[int, ...]) -> None:
        """Validate a concrete runtime shape against the engine declaration."""
        if len(shape) != len(self.input_shape):
            raise ValueError(
                f"TensorRT input {self.input_name!r} expects {len(self.input_shape)} "
                f"dimensions, got shape {shape}."
            )
        for axis, (actual, declared) in enumerate(zip(shape, self.input_shape)):
            if actual <= 0:
                raise ValueError(f"TensorRT input shape must be positive, got {shape}.")
            if declared > 0 and actual != declared:
                raise ValueError(
                    f"TensorRT input {self.input_name!r} axis {axis} is fixed at "
                    f"{declared}, got {actual} (shape {shape})."
                )

    def _set_runtime_input_shape(self, shape: tuple[int, ...]) -> None:
        """Set a dynamic input shape and reject profiles that do not accept it."""
        self._validate_input_shape(shape)
        if not self._dynamic_input:
            return
        accepted = self.context.set_input_shape(self.input_name, shape)
        if accepted is False:
            raise ValueError(
                f"TensorRT optimization profile rejected input shape {shape} "
                f"for tensor {self.input_name!r}."
            )

    def _runtime_output_shape(self, name: str) -> tuple[int, ...]:
        """Read a fully resolved output shape after the input shape is set."""
        try:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
        except AttributeError:
            shape = tuple(int(dim) for dim in self.output_shapes[name])
        if not shape or any(dim <= 0 for dim in shape):
            raise RuntimeError(
                f"TensorRT output {name!r} has unresolved runtime shape {shape}; "
                "ensure the optimization profile covers the input shape."
            )
        return shape

    def _execute(self) -> None:
        """Execute the current bindings and fail if TensorRT rejects the launch."""
        executed = self.context.execute_async_v3(self.stream.cuda_stream)
        if not executed:
            raise RuntimeError(
                "TensorRT execute_async_v3 returned False; inference was not executed."
            )

    def _set_tensor_address(self, name: str, tensor: torch.Tensor) -> None:
        """Bind one I/O tensor and fail if TensorRT rejects its device pointer."""
        accepted = self.context.set_tensor_address(name, tensor.data_ptr())
        if accepted is False:
            raise RuntimeError(
                f"TensorRT rejected the device address for tensor {name!r}."
            )

    def _wait_for_input_copy(self) -> None:
        """Order TensorRT's stream after the PyTorch stream that filled input."""
        self.stream.wait_stream(torch.cuda.current_stream(self.device))

    def _allocate_buffers(self, input_shape: tuple[int, ...]):
        """Set the input shape, then allocate dtype-correct CUDA I/O buffers."""
        concrete_shape = tuple(int(dim) for dim in input_shape)
        self._set_runtime_input_shape(concrete_shape)

        self.inputs = {}
        self.outputs = {}
        if not hasattr(self, "stream"):
            self.stream = torch.cuda.Stream(device=self.device)
        self._current_input_shape = concrete_shape
        self._current_batch = concrete_shape[0]

        input_size = int(np.prod(concrete_shape))
        self.inputs[self.input_name] = torch.empty(
            input_size, dtype=self.input_torch_dtype, device=self.device
        )

        self._runtime_output_shapes: Dict[str, tuple[int, ...]] = {}
        for name in self.output_names:
            shape = self._runtime_output_shape(name)
            self._runtime_output_shapes[name] = shape
            size = int(np.prod(shape))
            self.outputs[name] = torch.empty(
                size,
                dtype=self.output_torch_dtypes[name],
                device=self.device,
            )

    def _detect_batch_limits(self) -> tuple[int, int]:
        """Return the smallest and largest batch sizes the engine can execute."""
        if not self._dynamic_batch:
            batch = int(self.input_shape[0])
            return batch, batch

        try:
            profile_shapes = self.engine.get_tensor_profile_shape(self.input_name, 0)
            return int(profile_shapes[0][0]), int(profile_shapes[2][0])
        except (AttributeError, IndexError, TypeError, ValueError):
            pass

        metadata_min = self._metadata.get("trt_min_batch")
        metadata_max = self._metadata.get("trt_max_batch")
        if metadata_min is not None or metadata_max is not None:
            try:
                minimum = max(1, int(metadata_min or 1))
                maximum = max(minimum, int(metadata_max or minimum))
                return minimum, maximum
            except (TypeError, ValueError):
                pass

        return 1, 1

    def _detect_model_family(self) -> Optional[str]:
        """Detect model family from output shapes when sidecar metadata is absent."""
        # DETR exports share ``pred_logits``/``pred_boxes`` output names; the
        # sidecar is authoritative, but filename hints keep sidecar-less engines
        # routed to the right family when the user keeps LibreYOLO's names.
        stem = Path(self.model_path).stem.lower()
        if "deimv2" in stem:
            return "deimv2"
        # "ec" must be a whole token, not a bare substring (else "detector"/
        # "detection" would falsely match and route a YOLO tensor through EC's
        # sigmoid/top-k). The LibreYOLO default naming is ``LibreEC*`` (see
        # LibreEC.FILENAME_PREFIX), so also honor that prefix explicitly.
        stem_tokens = re.split(r"[_\-.]+", stem)
        if stem.startswith("libreec") or "ec" in stem_tokens:
            return "ec"
        if "dfine" in stem:
            return "dfine"
        if "deim" in stem:
            return "deim"
        if "rtdetr" in stem or "rt-detr" in stem:
            return "rtdetr"
        if "rfdetr" in stem or "rf-detr" in stem:
            return "rfdetr"

        # Without metadata or filename hints, this two-output schema is known
        # to be DETR-style detection but cannot distinguish D-FINE/DEIM/DEIMv2.
        # Keep the historical fallback for compatibility.
        if "pred_logits" in self.output_names and "pred_boxes" in self.output_names:
            return "dfine"
        if "output" in self.output_shapes:
            shape = self.output_shapes["output"]
            if len(shape) == 3 and shape[2] == 4 and len(self.output_names) == 2:
                return "rfdetr"
            elif len(shape) == 3:
                return "yolo9"
            elif len(shape) == 4:
                return "yolox"
        else:
            yolox_outputs = [n for n in self.output_names if n.startswith("cat_")]
            if yolox_outputs:
                return "yolox"
            # RTDETR has pred_logits and pred_boxes outputs
            has_pred_logits = any("pred_logits" in n for n in self.output_names)
            has_pred_boxes = any("pred_boxes" in n for n in self.output_names)
            if has_pred_logits and has_pred_boxes:
                return "rtdetr"
        return None

    def _detect_task_from_filename(self) -> Optional[str]:
        stem = Path(self.model_path).stem.lower()
        if re.search(r"(?:^|[_-])obb(?:[_-]|$)", stem):
            return "obb"
        if re.search(r"(?:^|[_-])(?:seg|segment)(?:[_-]|$)", stem):
            return "segment"
        if re.search(
            r"(?:rfdetr|rf-detr)[_-]?(?:xx|2xl|xl|[nsmlx])[_-]?(?:seg|segment)",
            stem,
        ):
            return "segment"
        return None

    def _infer(self, input_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Run TensorRT inference.

        Args:
            input_array: Input tensor of shape (B, C, H, W) or (C, H, W).

        Returns:
            Dict mapping output tensor names to numpy arrays.
        """
        with torch.cuda.device(self.device):
            return self._infer_current_device(input_array)

    def _infer_current_device(self, input_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference while the engine's CUDA device is current."""
        input_array = np.asarray(input_array)
        if input_array.ndim == 3:
            input_array = input_array[np.newaxis]
        requested_batch = int(input_array.shape[0])
        if requested_batch < 1:
            raise ValueError("TensorRT inference requires at least one input image.")
        if requested_batch < self._min_batch:
            padding = np.repeat(
                input_array[-1:], self._min_batch - requested_batch, axis=0
            )
            input_array = np.concatenate((input_array, padding), axis=0)
        actual_shape = tuple(int(dim) for dim in input_array.shape)
        self._validate_input_shape(actual_shape)
        if actual_shape[0] > self._max_batch:
            raise ValueError(
                f"TensorRT engine supports at most batch {self._max_batch}, "
                f"got batch {actual_shape[0]}."
            )
        if actual_shape != self._current_input_shape:
            # Dynamic output dimensions are resolved only after the actual
            # input shape is applied, so shape changes must precede allocation.
            self._allocate_buffers(actual_shape)

        runtime_input = np.ascontiguousarray(
            input_array,
            dtype=self.input_numpy_dtype,
        )
        input_tensor = torch.from_numpy(runtime_input).to(self.device).flatten()
        self.inputs[self.input_name].copy_(input_tensor)
        self._wait_for_input_copy()

        self._set_tensor_address(self.input_name, self.inputs[self.input_name])
        for name in self.output_names:
            self._set_tensor_address(name, self.outputs[name])

        self._execute()
        self.stream.synchronize()

        results = {}
        for name in self.output_names:
            shape = self._runtime_output_shapes[name]
            output = self.outputs[name].cpu().numpy().reshape(shape)
            if output.ndim > 0 and output.shape[0] == actual_shape[0]:
                output = output[:requested_batch]
            results[name] = output

        return results

    # =========================================================================
    # BaseBackend interface
    # =========================================================================

    def _run_inference(self, blob: np.ndarray) -> list:
        """Run TensorRT inference and return outputs as a list."""
        outputs_dict = self._infer(blob)
        return [outputs_dict[name] for name in self.output_names]

    def _supports_batched_inference(self) -> bool:
        """Use the shared task-complete batch path for dynamic-batch engines."""
        return self._max_batch > 1

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
    ) -> list:
        """Clamp chunks to the engine profile, then use shared task dispatch."""
        if self._dynamic_batch:
            effective_batch = min(batch, self._max_batch)
        else:
            effective_batch = self._max_batch
        return super()._process_in_batches(
            images,
            batch=effective_batch,
            save=save,
            output_path=output_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
            max_det=max_det,
            color_format=color_format,
        )

    # =========================================================================
    # Metadata helpers
    # =========================================================================

    @property
    def size(self) -> str:
        """Return model size from sidecar metadata or engine filename."""
        if self._sidecar_size is not None:
            return self._sidecar_size
        stem = Path(self.model_path).stem.lower()
        for pattern in (
            r"(?:rfdetr|rf-detr)[_-]?(xx|2xl|xl|[nsmlx])(?:[_-]|$|seg|segment)",
            r"(?:rfdetr|rf-detr)[_-]?(?:seg|segment)[_-]?(xx|2xl|xl|[nsmlx])(?:[_-]|$)",
        ):
            rfdetr_match = re.search(pattern, stem)
            if rfdetr_match is not None:
                size = rfdetr_match.group(1)
                return {"xl": "x", "2xl": "xx"}.get(size, size)

        token_match = re.search(r"(?:^|[_-])(xx|[ntsmlxc])(?:[_-]|$)", stem)
        if token_match is not None:
            return token_match.group(1)
        return "unknown"

    def _get_model_name(self) -> str:
        """Return model name for compatibility."""
        if self.model_family == "yolo9":
            return "yolo9"
        elif self.model_family == "yolox":
            return "yolox"
        elif self.model_family == "rfdetr":
            return "rfdetr"
        elif self.model_family == "dfine":
            return "dfine"
        elif self.model_family == "deim":
            return "deim"
        elif self.model_family == "deimv2":
            return "deimv2"
        elif self.model_family == "rtdetr":
            return "rtdetr"
        elif self.model_family == "ec":
            return "ec"
        return "libreyolo"

    def _get_input_size(self) -> ImageSize:
        """Return model input size."""
        return self.imgsz
