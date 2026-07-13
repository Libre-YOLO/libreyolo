"""TensorRT export implementation."""

import hashlib
import json
import logging
import os
import shutil
import stat
import uuid
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .calibration import CalibrationDataLoader
from .config import TensorRTExportConfig

logger = logging.getLogger(__name__)


def check_tensorrt_available() -> None:
    """Check if TensorRT is available and raise helpful error if not."""
    try:
        import tensorrt as trt

        _ = trt.__version__
    except ImportError:
        raise ImportError(
            "TensorRT export requires the 'tensorrt' package.\n\n"
            "Installation options:\n"
            "  1. pip install tensorrt  (requires NVIDIA GPU + CUDA)\n"
            "  2. pip install nvidia-tensorrt  (alternative package name)\n\n"
            "Requirements:\n"
            "  - NVIDIA GPU with compute capability >= 5.0\n"
            "  - CUDA toolkit installed\n"
            "  - cuDNN installed\n\n"
            "For Jetson devices, TensorRT is pre-installed with JetPack."
        )


def _create_calibrator_class():
    """Create calibrator class that inherits from TensorRT base.

    The class is created at runtime so that importing this module does not
    require TensorRT to be installed.
    """
    try:
        import tensorrt as trt

        class _TensorRTCalibratorImpl(trt.IInt8EntropyCalibrator2):
            """INT8 entropy calibrator for TensorRT engine builds."""

            def __init__(
                self,
                data_loader,
                cache_file="calibration.cache",
                *,
                device_index=0,
            ):
                super().__init__()
                self.data_loader = data_loader
                self.cache_file = Path(cache_file) if cache_file else None
                self.batch_iter = None
                self._device_input = None
                self._allocation_backend = None
                self._pycuda_context = None
                self._device_index = int(device_index)
                self._batch_size = data_loader.batch
                self._batch_idx = 0

            def get_batch_size(self):
                return self._batch_size

            def get_batch(self, names):
                if self.batch_iter is None:
                    self.batch_iter = iter(self.data_loader)

                try:
                    batch = next(self.batch_iter)
                    self._batch_idx += 1
                    total = len(self.data_loader)
                    logger.info("Calibrating: batch %d/%d", self._batch_idx, total)
                    device_ptr = self._ensure_cuda_memory(batch)
                    return [device_ptr]
                except StopIteration:
                    return None

            def _ensure_cuda_memory(self, batch):
                batch = np.ascontiguousarray(batch, dtype=np.float32)

                # Try cuda-python/cuda-bindings first (newer, better maintained)
                try:
                    from cuda.bindings import runtime as cudart

                    (err,) = cudart.cudaSetDevice(self._device_index)
                    if err != cudart.cudaError_t.cudaSuccess:
                        raise RuntimeError(f"cudaSetDevice failed: {err}")
                    if self._device_input is None:
                        err, self._device_input = cudart.cudaMalloc(batch.nbytes)
                        if err != cudart.cudaError_t.cudaSuccess:
                            raise RuntimeError(f"cudaMalloc failed: {err}")
                        self._allocation_backend = "cuda-bindings"
                    (err,) = cudart.cudaMemcpy(
                        self._device_input,
                        batch.ctypes.data,
                        batch.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    )
                    if err != cudart.cudaError_t.cudaSuccess:
                        raise RuntimeError(f"cudaMemcpy failed: {err}")
                    return int(self._device_input)
                except ImportError:
                    pass

                # Fall back to pycuda
                try:
                    import pycuda.driver as cuda

                    if self._pycuda_context is None:
                        cuda.init()
                        self._pycuda_context = cuda.Device(
                            self._device_index
                        ).retain_primary_context()
                    self._pycuda_context.push()
                    try:
                        if self._device_input is None:
                            self._device_input = cuda.mem_alloc(batch.nbytes)
                            self._allocation_backend = "pycuda"

                        cuda.memcpy_htod(self._device_input, batch)
                        return int(self._device_input)
                    finally:
                        self._pycuda_context.pop()
                except ImportError:
                    raise ImportError(
                        "INT8 calibration requires cuda-python or pycuda.\n"
                        "Install with: pip install cuda-python\n"
                        "Or: pip install pycuda (requires python3-dev)"
                    )

            def _release_cuda_memory(self):
                try:
                    if (
                        self._device_input is not None
                        and self._allocation_backend == "cuda-bindings"
                    ):
                        from cuda.bindings import runtime as cudart

                        (err,) = cudart.cudaSetDevice(self._device_index)
                        if err == cudart.cudaError_t.cudaSuccess:
                            cudart.cudaFree(self._device_input)
                    elif (
                        self._device_input is not None
                        and self._allocation_backend == "pycuda"
                        and self._pycuda_context is not None
                    ):
                        self._pycuda_context.push()
                        try:
                            self._device_input.free()
                        finally:
                            self._pycuda_context.pop()
                except Exception:
                    pass
                finally:
                    self._device_input = None
                    self._allocation_backend = None
                    if self._pycuda_context is not None:
                        try:
                            self._pycuda_context.detach()
                        except Exception:
                            pass
                        self._pycuda_context = None

            def __del__(self):
                self._release_cuda_memory()

            def read_calibration_cache(self):
                if self.cache_file is not None and self.cache_file.exists():
                    logger.info("Loading calibration cache: %s", self.cache_file)
                    with open(self.cache_file, "rb") as f:
                        return f.read()
                return None

            def write_calibration_cache(self, cache):
                if self.cache_file is None:
                    return
                logger.info("Saving calibration cache: %s", self.cache_file)
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, "wb") as f:
                    f.write(cache)

        return _TensorRTCalibratorImpl

    except ImportError:
        raise ImportError(
            "INT8 calibration requires TensorRT.\nInstall with: pip install tensorrt"
        )


def get_calibrator_class():
    """Get the appropriate calibrator class based on TensorRT availability."""
    return _create_calibrator_class()


def _validate_batch_profile(min_batch: int, opt_batch: int, max_batch: int) -> None:
    """Validate TensorRT's ordered, positive dynamic-batch profile bounds."""
    if not (1 <= min_batch <= opt_batch <= max_batch):
        raise ValueError(
            "TensorRT dynamic batch bounds must satisfy "
            "1 <= min_batch <= opt_batch <= max_batch, got "
            f"{min_batch}, {opt_batch}, {max_batch}."
        )


def _resolve_hardware_compatibility_level(trt, requested: str):
    """Resolve a TensorRT compatibility enum without assuming API availability."""
    levels = getattr(trt, "HardwareCompatibilityLevel", None)
    if levels is None:
        return None
    attribute = {
        "ampere_plus": "AMPERE_PLUS",
        "same_compute_capability": "SAME_COMPUTE_CAPABILITY",
    }.get(requested)
    return getattr(levels, attribute, None) if attribute is not None else None


def _select_tensorrt_device(device: int) -> int:
    """Select the requested CUDA device before creating TensorRT objects."""
    if isinstance(device, bool):
        raise ValueError(
            f"TensorRT device must be a non-negative integer, got {device!r}."
        )
    try:
        if isinstance(device, str):
            if not device.strip().isdigit():
                raise ValueError
            index = int(device.strip())
        else:
            index = device.__index__()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"TensorRT device must be a non-negative integer, got {device!r}."
        ) from exc
    if index < 0:
        raise ValueError(f"TensorRT device must be non-negative, got {index}.")
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT export requires a CUDA-capable GPU.")
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"TensorRT CUDA device index {index} is unavailable; "
            f"detected {torch.cuda.device_count()} CUDA device(s)."
        )
    torch.cuda.set_device(index)
    logger.info("Using GPU device: %d (%s)", index, torch.cuda.get_device_name(index))
    return index


def _publish_tensorrt_artifacts(
    serialized_engine,
    output_path: str | Path,
    metadata: dict | None,
) -> str:
    """Stage and transactionally publish an engine and optional sidecar."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = Path(str(output) + ".json")

    with TemporaryDirectory(
        prefix="libreyolo-tensorrt-publish-", dir=output.parent
    ) as workspace_name:
        workspace = Path(workspace_name)
        staged: dict[Path, Path | None] = {output: workspace / output.name}
        with open(staged[output], "xb") as file:
            file.write(serialized_engine)
            file.flush()
            os.fsync(file.fileno())

        if metadata is not None:
            staged[sidecar] = workspace / sidecar.name
            with open(staged[sidecar], "x", encoding="utf-8") as file:
                json.dump(metadata, file, allow_nan=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
        else:
            # A metadata-free rebuild must not leave metadata from an older
            # engine describing the newly-published artifact.
            staged[sidecar] = None

        backups: dict[Path, Path | None] = {}
        target_modes: dict[Path, int] = {}
        pending_backup: Path | None = None
        try:
            for target in staged:
                if not target.exists():
                    backups[target] = None
                    continue
                target_modes[target] = stat.S_IMODE(target.stat().st_mode)
                pending_backup = output.parent / (
                    f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.bak"
                )
                try:
                    os.link(target, pending_backup)
                except OSError:
                    # Keep restrictive source modes on a backup that may need
                    # to survive an incomplete rollback.
                    shutil.copy2(target, pending_backup)
                backups[target] = pending_backup
                pending_backup = None
        except BaseException:
            if pending_backup is not None:
                pending_backup.unlink(missing_ok=True)
            for backup in backups.values():
                if backup is not None:
                    backup.unlink(missing_ok=True)
            raise

        promoted = []
        retained_backups = set()
        try:
            for target, temporary in staged.items():
                if temporary is None:
                    if not target.exists():
                        continue
                    target.unlink()
                else:
                    if target in target_modes:
                        temporary.chmod(target_modes[target])
                    os.replace(temporary, target)
                promoted.append(target)
        except BaseException as promotion_error:
            rollback_errors = []
            for target in reversed(promoted):
                backup = backups[target]
                try:
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        os.replace(backup, target)
                        backups[target] = None
                except OSError as rollback_error:
                    if backup is not None:
                        retained_backups.add(backup)
                    rollback_errors.append(
                        f"{target}: {rollback_error}; previous artifact retained "
                        f"at {backup}"
                    )
            if rollback_errors:
                raise RuntimeError(
                    "TensorRT artifact publication failed and rollback was "
                    "incomplete: " + "; ".join(rollback_errors)
                ) from promotion_error
            raise
        finally:
            for backup in backups.values():
                if backup is None or backup in retained_backups:
                    continue
                backup.unlink(missing_ok=True)

    return str(output)


def _calibration_cache_fingerprint(calibration_data) -> str | None:
    """Hash exact preprocessed batches, disabling reuse if they are not stable."""
    try:
        if iter(calibration_data) is calibration_data:
            return None
    except TypeError:
        return None

    def hash_pass() -> str | None:
        digest = hashlib.sha256()
        count = 0
        try:
            for batch in calibration_data:
                array = np.ascontiguousarray(np.asarray(batch))
                if array.size == 0 or array.dtype.hasobject:
                    return None
                digest.update(array.dtype.str.encode("ascii"))
                digest.update(json.dumps(array.shape).encode("ascii"))
                digest.update(array.tobytes())
                count += 1
        except Exception:
            return None
        return digest.hexdigest() if count else None

    first = hash_pass()
    second = hash_pass()
    return first if first is not None and first == second else None


def _calibration_cache_path(
    output_path: str | Path,
    onnx_data: bytes,
    calibration_data,
    *,
    enabled: bool,
) -> Path | None:
    """Return a cache path keyed by both graph and calibration identity."""
    if not enabled:
        return None
    calibration_hash = _calibration_cache_fingerprint(calibration_data)
    if calibration_hash is None:
        return None
    digest = hashlib.sha256(onnx_data)
    digest.update(calibration_hash.encode("ascii"))
    cache_out = Path(output_path)
    return cache_out.with_name(f"{cache_out.stem}.{digest.hexdigest()[:16]}.cache")


def _validate_dynamic_calibration_contract(
    calibration_data,
    *,
    int8: bool,
    dynamic: bool,
    opt_batch: int,
    traced_shape: tuple[int, ...] | None,
) -> None:
    """Require TensorRT dynamic INT8 calibration to use the OPT batch-1 shape."""
    if not int8 or not dynamic:
        return
    if calibration_data is None:
        raise ValueError("INT8 quantization requires calibration data.")
    try:
        calibration_shape = tuple(int(dim) for dim in calibration_data.shape)
        calibration_batch = int(calibration_data.batch)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Dynamic TensorRT INT8 calibration requires a loader with batch and "
            "NCHW shape metadata."
        ) from exc
    if opt_batch != 1 or calibration_batch != 1 or calibration_shape[0] != 1:
        raise ValueError(
            "Dynamic TensorRT INT8 calibration requires opt_batch=1 and a "
            f"batch-1 calibration loader, got opt_batch={opt_batch}, "
            f"loader batch={calibration_batch}, shape={calibration_shape}."
        )
    if traced_shape is not None and calibration_shape[1:] != traced_shape[1:]:
        raise ValueError(
            "Dynamic TensorRT INT8 calibration shape must match the traced input "
            f"CHW dimensions, got {calibration_shape} versus {traced_shape}."
        )


def _validate_static_calibration_contract(
    calibration_data,
    *,
    int8: bool,
    dynamic: bool,
    traced_shape: tuple[int, ...] | None,
    network_shape: tuple[int, ...] | None = None,
) -> None:
    """Require a static INT8 loader to match its traced and parsed NCHW input."""
    if not int8 or dynamic:
        return
    if calibration_data is None:
        raise ValueError("INT8 quantization requires calibration data.")
    try:
        calibration_shape = tuple(int(dim) for dim in calibration_data.shape)
        calibration_batch = int(calibration_data.batch)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Static TensorRT INT8 calibration requires a loader with batch and "
            "NCHW shape metadata."
        ) from exc
    if (
        len(calibration_shape) != 4
        or any(dim <= 0 for dim in calibration_shape)
        or calibration_batch <= 0
        or calibration_shape[0] != calibration_batch
    ):
        raise ValueError(
            "Static TensorRT INT8 calibration requires positive NCHW shape "
            "metadata whose first axis equals loader.batch, got "
            f"batch={calibration_batch}, shape={calibration_shape}."
        )

    if traced_shape is not None and calibration_shape != traced_shape:
        raise ValueError(
            "Static TensorRT INT8 calibration shape must match the traced input "
            f"NCHW shape, got {calibration_shape} versus {traced_shape}."
        )

    if network_shape is None:
        return
    declared = tuple(int(dim) for dim in network_shape)
    if len(declared) != 4:
        raise ValueError(
            "Static TensorRT INT8 calibration requires a rank-4 network input, "
            f"got shape {declared}."
        )
    for axis, (actual, fixed) in enumerate(zip(calibration_shape, declared)):
        if fixed > 0 and actual != fixed:
            raise ValueError(
                "Static TensorRT INT8 calibration shape does not match the parsed "
                f"network input at axis {axis}: got {calibration_shape}, network "
                f"declares {declared}."
            )


def _profile_shape(
    declared_shape,
    traced_shape: tuple[int, ...] | None,
    *,
    batch: int,
) -> tuple[int, ...]:
    """Resolve one profile shape without replacing non-batch axes by batch."""
    declared = tuple(int(dim) for dim in declared_shape)
    traced = None if traced_shape is None else tuple(int(dim) for dim in traced_shape)
    if traced is not None:
        if len(traced) != len(declared) or any(dim <= 0 for dim in traced):
            raise ValueError(
                "TensorRT traced input_shape must contain one positive value per "
                f"network axis, got {traced} for declared shape {declared}."
            )
        for axis, (actual, fixed) in enumerate(zip(traced, declared)):
            if fixed > 0 and actual != fixed:
                raise ValueError(
                    f"TensorRT traced input_shape axis {axis} is {actual}, but the "
                    f"ONNX input fixes it at {fixed}."
                )

    resolved = []
    for axis, dim in enumerate(declared):
        if dim > 0:
            resolved.append(dim)
        elif axis == 0:
            resolved.append(batch)
        elif traced is not None:
            resolved.append(traced[axis])
        else:
            raise ValueError(
                "TensorRT cannot derive a dynamic non-batch profile axis without "
                f"the traced input_shape; ONNX input shape is {declared}."
            )
    return tuple(resolved)


def export_tensorrt(
    onnx_path: str,
    output_path: str,
    *,
    half: bool = True,
    int8: bool = False,
    workspace: float = 4.0,
    calibration_data: Optional[CalibrationDataLoader] = None,
    dynamic: bool = False,
    verbose: bool = False,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 8,
    hardware_compatibility: str = "none",
    device: int = 0,
    config: Optional[Union[str, Path, dict, TensorRTExportConfig]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    input_shape: tuple[int, int, int, int] | None = None,
) -> str:
    """Export ONNX model to TensorRT engine.

    The engine is optimized for the GPU it's built on. By default, engines are NOT
    portable between different GPU architectures (e.g., an engine built
    on RTX 4090 won't work on RTX 3080). Use hardware_compatibility to
    enable portability at the cost of some performance.

    Args:
        onnx_path: Path to ONNX model file.
        output_path: Output path for .engine file.
        half: Enable FP16 precision (default: True). Provides ~2x speedup
              on most GPUs with minimal accuracy loss.
        int8: Enable INT8 precision (default: False). Requires calibration_data.
              Provides additional speedup but may impact accuracy.
        workspace: GPU workspace size in GiB for kernel optimization (default: 4.0).
                  Larger values may find faster kernels but use more GPU memory
                  during build. Does not affect inference memory usage.
        calibration_data: CalibrationDataLoader for INT8 quantization.
                         Required when int8=True.
        dynamic: Enable dynamic batch size (default: False). When True, the
                engine supports batch sizes from min_batch to max_batch.
        verbose: Enable verbose TensorRT logging (default: False).
        min_batch: Minimum batch size for dynamic batching (default: 1).
        opt_batch: Optimal batch size for dynamic batching (default: 1).
                  TensorRT optimizes kernels for this batch size.
        max_batch: Maximum batch size for dynamic batching (default: 8).
        hardware_compatibility: Hardware compatibility level (default: "none").
            - "none": Optimize for current GPU only (fastest, not portable)
            - "ampere_plus": Works on Ampere (RTX 30xx, A100) and newer GPUs
            - "same_compute_capability": Works on GPUs with same SM version
        device: GPU device ID for multi-GPU systems (default: 0).
        config: Optional TensorRTExportConfig or path to YAML config file.
               If provided, overrides individual parameters.
        metadata: Optional dict of model metadata to write as a JSON sidecar
                 file alongside the engine (e.g. model_family, nb_classes, names).
        input_shape: Concrete NCHW shape of the tensor used to trace the ONNX
                     graph. Required when a non-batch input axis is dynamic.

    Returns:
        Path to exported .engine file.

    Raises:
        ImportError: If TensorRT is not installed.
        ValueError: If int8=True but calibration_data is not provided.
        RuntimeError: If engine building fails.

    Example::

        # FP16 export (recommended for most use cases)
        export_tensorrt("model.onnx", "model.engine", half=True)

        # INT8 export with calibration
        from libreyolo.export.calibration import get_calibration_dataloader
        calib = get_calibration_dataloader("coco8.yaml", imgsz=640)
        export_tensorrt("model.onnx", "model_int8.engine", int8=True,
                       calibration_data=calib)

        # Export with config file
        export_tensorrt("model.onnx", "model.engine",
                       config="tensorrt_default.yaml")
    """
    use_calibration_cache = True
    if config is not None:
        from .config import load_export_config

        cfg = load_export_config(config)
        half = cfg.half
        int8 = cfg.int8
        workspace = cfg.workspace
        verbose = cfg.verbose
        hardware_compatibility = cfg.hardware_compatibility
        device = cfg.device
        use_calibration_cache = bool(cfg.int8_calibration.cache)
        dynamic = bool(cfg.dynamic.enabled)
        if dynamic:
            min_batch = cfg.dynamic.min_batch
            opt_batch = cfg.dynamic.opt_batch
            max_batch = cfg.dynamic.max_batch
        if cfg.int8:
            # int8_calibration dataset/fraction are parsed and validated by
            # TensorRTExportConfig but not consumed here: calibration comes
            # from the pre-built ``calibration_data`` loader (the export()
            # data=/fraction= arguments). Warn rather than silently drop it.
            warnings.warn(
                "TensorRTExportConfig.int8_calibration dataset/fraction are "
                "currently ignored: INT8 "
                "calibration is driven by the export() data=/fraction= "
                "arguments (passed here as calibration_data). Set those to "
                "control INT8 calibration."
            )

    min_batch = int(min_batch)
    opt_batch = int(opt_batch)
    max_batch = int(max_batch)
    _validate_batch_profile(min_batch, opt_batch, max_batch)
    traced_input_shape = (
        None if input_shape is None else tuple(int(dim) for dim in input_shape)
    )
    if traced_input_shape is not None and (
        len(traced_input_shape) != 4 or any(dim <= 0 for dim in traced_input_shape)
    ):
        raise ValueError(
            "TensorRT input_shape must be a positive NCHW tuple, got "
            f"{traced_input_shape}."
        )

    if int8 and calibration_data is None:
        raise ValueError(
            "INT8 quantization requires calibration data.\n"
            "Provide calibration_data parameter or use data='coco8.yaml' "
            "in the export() call."
        )
    _validate_dynamic_calibration_contract(
        calibration_data,
        int8=int8,
        dynamic=dynamic,
        opt_batch=opt_batch,
        traced_shape=traced_input_shape,
    )
    _validate_static_calibration_contract(
        calibration_data,
        int8=int8,
        dynamic=dynamic,
        traced_shape=traced_input_shape,
    )

    if metadata is not None:
        metadata = dict(metadata)
        if dynamic:
            metadata.update(
                {
                    "trt_min_batch": int(min_batch),
                    "trt_opt_batch": int(opt_batch),
                    "trt_max_batch": int(max_batch),
                }
            )

    check_tensorrt_available()
    import tensorrt as trt

    device = _select_tensorrt_device(device)

    if half and int8:
        warnings.warn(
            "Both half=True and int8=True specified. Using INT8 precision "
            "(INT8 includes FP16 fallback for unsupported layers)."
        )

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    log_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    trt_logger = trt.Logger(log_level)

    builder = trt.Builder(trt_logger)
    # Explicit batch mode (required for modern TensorRT)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Parsing ONNX model: %s", onnx_path)
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        error_msgs = []
        for i in range(parser.num_errors):
            error_msgs.append(str(parser.get_error(i)))
        raise RuntimeError("Failed to parse ONNX model:\n" + "\n".join(error_msgs))

    logger.info(
        "Network: %d inputs, %d outputs, %d layers",
        network.num_inputs,
        network.num_outputs,
        network.num_layers,
    )
    if int8:
        if network.num_inputs != 1:
            raise ValueError(
                "TensorRT INT8 calibration currently supports exactly one network "
                f"input, but the parsed ONNX graph has {network.num_inputs}."
            )
        _validate_static_calibration_contract(
            calibration_data,
            int8=int8,
            dynamic=dynamic,
            traced_shape=traced_input_shape,
            network_shape=tuple(int(dim) for dim in network.get_input(0).shape),
        )

    builder_config = builder.create_builder_config()

    workspace_bytes = int(workspace * (1 << 30))  # GiB to bytes
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    logger.info("Workspace: %s GiB", workspace)

    if hardware_compatibility != "none":
        try:
            compat_level = _resolve_hardware_compatibility_level(
                trt, hardware_compatibility
            )

            if compat_level is not None:
                builder_config.hardware_compatibility_level = compat_level
                logger.info("Hardware compatibility: %s", hardware_compatibility)
            elif hardware_compatibility in {
                "ampere_plus",
                "same_compute_capability",
            }:
                warnings.warn(
                    f"hardware_compatibility={hardware_compatibility!r} is not "
                    "supported by this TensorRT API. The engine will use the "
                    "default current-GPU compatibility."
                )
            else:
                warnings.warn(
                    f"Unknown hardware_compatibility '{hardware_compatibility}'. "
                    f"Using default (none)."
                )
        except AttributeError:
            warnings.warn(
                "TensorRT version does not support hardware_compatibility_level. "
                "Engine will only work on current GPU architecture."
            )

    # Precision
    precision_str = "FP32"
    # Canonical precision actually realized by the build (may differ from the
    # requested precision when the GPU lacks fast FP16/INT8). Threaded into the
    # sidecar metadata so the artifact never claims a precision it isn't.
    actual_precision = "fp32"

    if half or int8:
        if builder.platform_has_fast_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
            precision_str = "FP16"
            actual_precision = "fp16"
            # ViT backbones (DINOv2/v3) overflow in FP16 -> NaN. The FP16 builder flag above is
            # enabled for both half and int8 builds (INT8 still runs FP16-precision layers), so
            # pin the ViT backbone whenever FP16 is active, not only for half. Detect a ViT
            # backbone (LayerNorm/Erf under model/backbone) and pin its float compute layers to
            # FP32 (mixed precision); no-op for CNN backbones.
            try:
                import onnx as _onnx

                _vit = any(
                    n.op_type in ("LayerNormalization", "Erf")
                    and "backbone" in (n.name or "")
                    for n in _onnx.load(onnx_path).graph.node
                )
            except Exception:
                _vit = False
            if _vit:
                builder_config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                _npin = 0
                for _i in range(network.num_layers):
                    _lyr = network.get_layer(_i)
                    if (
                        "backbone" not in (_lyr.name or "")
                        or _lyr.type == trt.LayerType.SHAPE
                    ):
                        continue
                    try:
                        _outs = [_lyr.get_output(_j) for _j in range(_lyr.num_outputs)]
                        if any(
                            o is None or o.dtype not in (trt.float32, trt.float16)
                            for o in _outs
                        ):
                            continue
                        _lyr.precision = trt.float32
                        for _j in range(_lyr.num_outputs):
                            _lyr.set_output_type(_j, trt.float32)
                        _npin += 1
                    except Exception:
                        continue
                if _npin > 0:
                    precision_str = "FP16 (FP32 ViT backbone)"
                    logger.info("ViT backbone: pinned %d float layers to FP32", _npin)
                else:
                    logger.warning(
                        "ViT backbone detected in ONNX but no matching TRT layers found; "
                        "FP32 pinning skipped"
                    )
        else:
            warnings.warn("GPU does not support fast FP16. Falling back to FP32.")

    if int8:
        if builder.platform_has_fast_int8:
            builder_config.set_flag(trt.BuilderFlag.INT8)
            precision_str = "INT8"
            actual_precision = "int8"

            CalibratorClass = get_calibrator_class()
            # Key the calibration cache on model identity (ONNX content hash) so
            # calibration scales for one model are never reused for another that
            # happens to export to the same output path.
            cache_file = _calibration_cache_path(
                output_path,
                onnx_data,
                calibration_data,
                enabled=use_calibration_cache,
            )
            if use_calibration_cache and cache_file is None:
                warnings.warn(
                    "TensorRT calibration cache reuse is disabled because a "
                    "complete calibration dataset/preprocessor identity could not "
                    "be established."
                )
            calibrator = CalibratorClass(
                calibration_data,
                cache_file=str(cache_file) if cache_file is not None else None,
                device_index=device,
            )
            builder_config.int8_calibrator = calibrator
            logger.info(
                "INT8 calibration: %d batches, batch size %d",
                len(calibration_data),
                calibration_data.batch,
            )
        else:
            warnings.warn("GPU does not support fast INT8. Falling back to FP16.")

    logger.info("Precision: %s", precision_str)

    # Dynamic profiles. Non-batch dynamic axes are fixed to the concrete shape
    # used for tracing; they must never be substituted with a batch value.
    network_input_shapes = [
        tuple(int(dim) for dim in network.get_input(i).shape)
        for i in range(network.num_inputs)
    ]
    has_dynamic_axes = any(dim == -1 for shape in network_input_shapes for dim in shape)

    if has_dynamic_axes:
        profile = builder.create_optimization_profile()
        if dynamic:
            profile_batches = (min_batch, opt_batch, max_batch)
        else:
            fixed_batch = traced_input_shape[0] if traced_input_shape else 1
            profile_batches = (fixed_batch, fixed_batch, fixed_batch)

        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_name = input_tensor.name
            declared_shape = network_input_shapes[i]
            concrete_shape = traced_input_shape if i == 0 else None
            min_shape = _profile_shape(
                declared_shape,
                concrete_shape,
                batch=profile_batches[0],
            )
            opt_shape = _profile_shape(
                declared_shape,
                concrete_shape,
                batch=profile_batches[1],
            )
            max_shape = _profile_shape(
                declared_shape,
                concrete_shape,
                batch=profile_batches[2],
            )

            accepted = profile.set_shape(
                input_name,
                min_shape,
                opt_shape,
                max_shape,
            )
            if accepted is False:
                raise ValueError(
                    f"TensorRT rejected optimization profile for {input_name!r}: "
                    f"min={min_shape}, opt={opt_shape}, max={max_shape}."
                )
            logger.info(
                "Dynamic input '%s': min=%s, opt=%s, max=%s",
                input_name,
                min_shape,
                opt_shape,
                max_shape,
            )

        builder_config.add_optimization_profile(profile)
    elif dynamic:
        logger.info(
            "Note: ONNX inputs are static, using static optimization (%s)",
            network_input_shapes,
        )

    logger.info("Building TensorRT engine... (this may take several minutes)")

    serialized_engine = builder.build_serialized_network(network, builder_config)

    if serialized_engine is None:
        raise RuntimeError(
            "TensorRT engine build failed. Common causes:\n"
            "  - Unsupported ONNX operations\n"
            "  - Insufficient GPU memory\n"
            "  - CUDA/cuDNN version mismatch\n"
            "Try running with verbose=True for detailed error messages."
        )

    if metadata is not None:
        # Reflect the precision actually realized by the build (e.g. fp16 after
        # an INT8→FP16 fallback) instead of the pre-build request.
        metadata["precision"] = actual_precision
    result = _publish_tensorrt_artifacts(serialized_engine, output_path, metadata)
    if metadata is not None:
        logger.info("Metadata sidecar: %s.json", output_path)

    engine_size_mb = Path(result).stat().st_size / (1024 * 1024)
    logger.info("Engine saved: %s (%.1f MB)", result, engine_size_mb)

    return result
