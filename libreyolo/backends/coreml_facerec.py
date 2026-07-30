"""Strict Core ML session facade for the face-embedding component."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..export.coreml_facerec import (
    FACEREC_COREML_INPUT_NAME,
    FACEREC_COREML_OUTPUT_NAME,
    FACEREC_COREML_REQUIRED_COMPUTE_UNITS,
    validate_facerec_coreml_metadata,
)
from ..export.coreml_profiles import resolve_coreml_runtime_compute_units

_COREML_FLOAT32 = 65568


@dataclass(frozen=True)
class CoreMLFaceIO:
    name: str
    shape: tuple[int, ...]


def _metadata_from_spec(spec: Any) -> dict[str, str]:
    description = getattr(spec, "description", None)
    metadata = getattr(description, "metadata", None)
    values = getattr(metadata, "userDefined", None)
    if values is None:
        return {}
    return {str(key): str(value) for key, value in dict(values).items()}


def load_coreml_package_metadata(path: str | Path) -> dict[str, str]:
    """Read package metadata without compiling or executing the model."""
    try:
        import coremltools as ct
    except ImportError as exc:
        raise ImportError(
            "Core ML package loading requires coremltools. Install with: "
            "pip install 'libreyolo[coreml]'"
        ) from exc
    package = Path(path)
    if not package.is_dir() or package.suffix.lower() != ".mlpackage":
        raise ValueError(f"Core ML model must be an .mlpackage directory: {path}")
    spec = ct.utils.load_spec(str(package))
    return _metadata_from_spec(spec)


def coreml_package_family(path: str | Path) -> str | None:
    value = load_coreml_package_metadata(path).get("model_family")
    normalized = str(value or "").strip().lower()
    return normalized or None


def _fixed_multiarray(
    feature: Any,
    *,
    label: str,
) -> CoreMLFaceIO:
    feature_type = getattr(feature, "type", None)
    which = getattr(feature_type, "WhichOneof", None)
    if not callable(which) or which("Type") != "multiArrayType":
        raise ValueError(
            f"Face Core ML {label} must be a fixed FP32 MultiArray."
        )
    if bool(getattr(feature_type, "isOptional", False)):
        raise ValueError(f"Face Core ML {label} must not be optional.")
    array = getattr(feature_type, "multiArrayType", None)
    flexibility = getattr(array, "WhichOneof", None)
    if callable(flexibility) and flexibility("ShapeFlexibility") is not None:
        raise ValueError(f"Face Core ML {label} must have a fixed shape.")
    shape = tuple(int(value) for value in getattr(array, "shape", ()))
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"Face Core ML {label} shape is invalid: {shape}.")
    if int(getattr(array, "dataType", 0) or 0) != _COREML_FLOAT32:
        raise ValueError(f"Face Core ML {label} must expose FP32 values.")
    return CoreMLFaceIO(str(feature.name), shape)


def validate_facerec_coreml_spec(
    spec: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check protobuf IO with the tamper-evident metadata contract."""
    parsed = validate_facerec_coreml_metadata(metadata)
    description = getattr(spec, "description", None)
    inputs = list(getattr(description, "input", ()) or ())
    outputs = list(getattr(description, "output", ()) or ())
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            "Face Core ML packages require exactly one input and one output."
        )
    input_info = _fixed_multiarray(inputs[0], label="input")
    output_info = _fixed_multiarray(outputs[0], label="output")
    preprocess = parsed["preprocess"]
    size = int(preprocess["size"])
    expected_input_shape = (
        (1, 3, size, size)
        if preprocess["layout"] == "NCHW"
        else (1, size, size, 3)
    )
    expected_output_shape = (1, int(parsed["embedding_dim"]))
    if (
        input_info.name != FACEREC_COREML_INPUT_NAME
        or input_info.shape != expected_input_shape
    ):
        raise ValueError(
            "Face Core ML input disagrees with its aligned-face contract: "
            f"expected {FACEREC_COREML_INPUT_NAME!r}/{expected_input_shape}, "
            f"got {input_info.name!r}/{input_info.shape}."
        )
    if (
        output_info.name != FACEREC_COREML_OUTPUT_NAME
        or output_info.shape != expected_output_shape
    ):
        raise ValueError(
            "Face Core ML output disagrees with its embedding contract: "
            f"expected {FACEREC_COREML_OUTPUT_NAME!r}/{expected_output_shape}, "
            f"got {output_info.name!r}/{output_info.shape}."
        )
    try:
        declared_outputs = json.loads(
            str(metadata.get("coreml_output_names", ""))
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Face Core ML output-name metadata must be valid JSON."
        ) from exc
    if declared_outputs != [FACEREC_COREML_OUTPUT_NAME]:
        raise ValueError(
            "Face Core ML output-name metadata was modified."
        )
    return {
        **parsed,
        "input": input_info,
        "output": output_info,
    }


def _compute_unit(ct: Any, value: str):
    key = str(value).strip().lower()
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    if key not in mapping:
        raise ValueError(
            f"Invalid Core ML compute_units {value!r}; expected one of "
            f"{sorted(mapping)}."
        )
    return mapping[key]


class CoreMLFaceSession:
    """ONNX Runtime-shaped facade over one strict Core ML embed component."""

    def __init__(
        self,
        model_path: str,
        *,
        compute_units: str = "cpu_only",
    ) -> None:
        if sys.platform != "darwin":
            raise RuntimeError(
                f"Core ML inference requires macOS. Current platform: "
                f"{sys.platform}."
            )
        try:
            import coremltools as ct
        except ImportError as exc:
            raise ImportError(
                "Core ML inference requires coremltools. Install with: "
                "pip install 'libreyolo[coreml]'"
            ) from exc

        path = Path(model_path)
        if not path.is_dir() or path.suffix.lower() != ".mlpackage":
            raise ValueError(
                f"Face Core ML model must be an .mlpackage directory: {path}"
            )
        spec = ct.utils.load_spec(str(path))
        metadata = _metadata_from_spec(spec)
        from ..export.coreml_identity import (
            COREML_DEPLOYMENT_ABI_SCHEMA,
            validate_coreml_deployment_abi,
        )
        from ..export.coreml_profiles import (
            COREML_EXECUTION_PROFILE_VERSION,
        )

        declared_profile_version = str(
            metadata.get("coreml_execution_profile_version", "")
        ).strip()
        declared_abi_schema = str(
            metadata.get("coreml_profile_abi_schema", "")
        ).strip()
        if (
            declared_profile_version == COREML_EXECUTION_PROFILE_VERSION
            or declared_abi_schema == COREML_DEPLOYMENT_ABI_SCHEMA
        ):
            validate_coreml_deployment_abi(spec, metadata)
        contract = validate_facerec_coreml_spec(spec, metadata)
        requested_compute_units = resolve_coreml_runtime_compute_units(
            compute_units,
            metadata,
        )
        if (
            requested_compute_units
            != FACEREC_COREML_REQUIRED_COMPUTE_UNITS
        ):
            raise NotImplementedError(
                "Face Core ML runtime is validated only with "
                "compute_units='cpu_only'. FP16 failed raw-embedding parity, "
                "and other compute-unit planners have not passed the hardware "
                "gate."
            )
        self.model = ct.models.MLModel(
            str(path),
            compute_units=_compute_unit(ct, requested_compute_units),
        )
        runtime_metadata = {
            str(key): str(value)
            for key, value in dict(
                getattr(self.model, "user_defined_metadata", {}) or {}
            ).items()
        }
        if runtime_metadata and runtime_metadata != metadata:
            raise ValueError(
                "Face Core ML runtime metadata differs from the package spec."
            )
        if (
            declared_profile_version == COREML_EXECUTION_PROFILE_VERSION
            or declared_abi_schema == COREML_DEPLOYMENT_ABI_SCHEMA
        ):
            validate_coreml_deployment_abi(
                self.model.get_spec(),
                runtime_metadata,
            )
        self.model_path = str(path)
        self.metadata = metadata
        self.preprocess = contract["preprocess"]
        self.embedding_dim = int(contract["embedding_dim"])
        self._input = contract["input"]
        self._output = contract["output"]

    def get_inputs(self) -> list[CoreMLFaceIO]:
        return [self._input]

    def get_outputs(self) -> list[CoreMLFaceIO]:
        return [self._output]

    def run(
        self,
        output_names: list[str] | None,
        inputs: Mapping[str, Any],
    ) -> list[np.ndarray]:
        if output_names not in (None, [self._output.name]):
            raise ValueError(
                "Face Core ML session exposes only output "
                f"{self._output.name!r}."
            )
        if set(inputs) != {self._input.name}:
            raise ValueError(
                "Face Core ML session input names changed: expected "
                f"{[self._input.name]}, got {sorted(inputs)}."
            )
        array = np.asarray(inputs[self._input.name], dtype=np.float32)
        if tuple(array.shape) != self._input.shape:
            raise ValueError(
                "Face Core ML input shape mismatch: expected "
                f"{self._input.shape}, got {tuple(array.shape)}."
            )
        if not bool(np.isfinite(array).all()):
            raise ValueError("Face Core ML input contains NaN or infinity.")
        result = self.model.predict(
            {self._input.name: np.ascontiguousarray(array)}
        )
        if not isinstance(result, Mapping) or set(result) != {self._output.name}:
            names = sorted(result) if isinstance(result, Mapping) else []
            raise RuntimeError(
                "Face Core ML runtime output names changed: expected "
                f"{[self._output.name]}, got {names}."
            )
        output = np.asarray(result[self._output.name], dtype=np.float32)
        if tuple(output.shape) != self._output.shape:
            raise RuntimeError(
                "Face Core ML output shape mismatch: expected "
                f"{self._output.shape}, got {tuple(output.shape)}."
            )
        if not bool(np.isfinite(output).all()):
            raise RuntimeError("Face Core ML output contains NaN or infinity.")
        return [np.ascontiguousarray(output)]


__all__ = [
    "CoreMLFaceIO",
    "CoreMLFaceSession",
    "coreml_package_family",
    "load_coreml_package_metadata",
    "validate_facerec_coreml_spec",
]
