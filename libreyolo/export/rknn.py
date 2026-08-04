"""Rockchip RKNN export and host-simulator parity helpers.

Provenance
----------
This module was implemented against LibreYOLO's exporter contract and the
public Rockchip ``rknn_model_zoo`` repository at commit
``bad6c7334531becaf90a561988519b7bec34d0ab`` (Apache-2.0).  The relevant API
references were ``examples/LPRNet/python/convert.py`` and
``py_utils/rknn_executor.py``.  No Ultralytics source code was inspected or
used.

``rknn-toolkit2`` itself is a separately installed vendor SDK governed by
Rockchip's SDK license.  LibreYOLO does not bundle or redistribute it.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RKNN_TARGET = "rk3588"

# These are deliberately exact model variants, not family-wide promises.
# Each entry passed Toolkit2 2.3.2 RK3588 compilation, host-simulator raw
# output checks, and task-level detection comparison on a real image.
RKNN_SIMULATOR_VALIDATED_MODELS = frozenset(
    {
        ("picodet", "s", "detect"),
        ("yolo9", "t", "detect"),
        ("yolo9_e2e", "t", "detect"),
        ("yolonas", "s", "detect"),
    }
)
RKNN_SIMULATOR_VALIDATED_TARGETS = frozenset({DEFAULT_RKNN_TARGET})

# Toolkit2's floating build commonly lowers internal tensors to float16, so
# decoded detector tensors are not elementwise-allclose to ONNX Runtime even
# when final boxes/classes remain stable. These output-scale-independent gates
# separated every validated detector above from the failed G0/G1 candidates.
DEFAULT_RKNN_MIN_COSINE = 0.9999
DEFAULT_RKNN_MAX_NORMALIZED_RMSE = 0.02


def validate_rknn_export_request(
    *,
    model_family: str,
    model_size: str,
    task: str,
    target_platform: str,
) -> None:
    """Reject combinations that have not passed the recorded RKNN gates."""
    model_key = (
        str(model_family).strip().lower(),
        str(model_size).strip().lower(),
        str(task).strip().lower(),
    )
    if model_key not in RKNN_SIMULATOR_VALIDATED_MODELS:
        validated = ", ".join(
            f"{family}-{size}/{validated_task}"
            for family, size, validated_task in sorted(
                RKNN_SIMULATOR_VALIDATED_MODELS
            )
        )
        rendered = f"{model_key[0]}-{model_key[1]}/{model_key[2]}"
        raise NotImplementedError(
            f"RKNN export is not validated for {rendered}. "
            f"Simulator-validated variants: {validated}. Compile-only "
            "results are intentionally not advertised as support."
        )

    target = str(target_platform).strip().lower()
    if target not in RKNN_SIMULATOR_VALIDATED_TARGETS:
        validated_targets = ", ".join(sorted(RKNN_SIMULATOR_VALIDATED_TARGETS))
        raise NotImplementedError(
            f"RKNN target {target!r} is not validated. "
            f"Simulator-validated targets: {validated_targets}."
        )


def resolve_rknn_target(
    *,
    name: str | None = None,
    target: str | None = None,
    target_platform: str | None = None,
) -> str:
    """Resolve supported target aliases to one lowercase platform name."""
    supplied = [
        (key, value.strip().lower())
        for key, value in (
            ("name", name),
            ("target", target),
            ("target_platform", target_platform),
        )
        if isinstance(value, str) and value.strip()
    ]
    if not supplied:
        return DEFAULT_RKNN_TARGET

    values = {value for _, value in supplied}
    if len(values) != 1:
        rendered = ", ".join(f"{key}={value!r}" for key, value in supplied)
        raise ValueError(f"Conflicting RKNN targets: {rendered}")
    return supplied[0][1]


def _load_rknn_class():
    if sys.platform != "linux":
        raise ImportError(
            "RKNN export requires Linux x86_64. On Windows, run LibreYOLO "
            "inside WSL2 or a Linux Docker container with Rockchip's "
            "rknn-toolkit2 wheel installed."
        )
    try:
        from rknn.api import RKNN
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "RKNN export requires Rockchip rknn-toolkit2, which LibreYOLO "
            "cannot redistribute. Install the vendor wheel in a Linux x86_64 "
            "environment after reviewing its SDK license."
        ) from exc
    return RKNN


def check_rknn_available() -> None:
    """Raise a focused error unless the vendor compiler can be imported."""
    _load_rknn_class()
    try:
        import onnx
    except ImportError as exc:
        raise ImportError("RKNN export requires ONNX.") from exc
    if not hasattr(onnx, "mapping"):
        raise ImportError(
            "rknn-toolkit2 2.3.2 is incompatible with ONNX 1.19 and newer "
            "because the vendor compiler still imports onnx.mapping. "
            "Install onnx==1.18.0 in the isolated RKNN environment."
        )


def _check_status(stage: str, status: Any) -> None:
    if status != 0:
        raise RuntimeError(f"RKNN {stage} failed with status {status!r}.")


def _write_calibration_dataset(
    calibration_data: Iterable[np.ndarray], directory: Path
) -> Path:
    """Materialize LibreYOLO calibration batches as RKNN ``.npy`` inputs."""
    entries: list[str] = []
    for index, batch in enumerate(calibration_data):
        array = np.asarray(batch, dtype=np.float32)
        if array.ndim != 4:
            raise ValueError(
                "RKNN calibration batches must have NCHW rank 4, "
                f"got shape {array.shape}."
            )
        sample_path = directory / f"calibration_{index:05d}.npy"
        np.save(sample_path, array)
        entries.append(str(sample_path.resolve()))

    if not entries:
        raise ValueError("RKNN INT8 calibration data produced no usable batches.")

    dataset_path = directory / "dataset.txt"
    dataset_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return dataset_path


def _write_metadata_sidecar(output_path: Path, metadata: Mapping[str, Any]) -> None:
    sidecar = Path(f"{output_path}.metadata.json")
    temporary = sidecar.with_suffix(f"{sidecar.suffix}.tmp")
    temporary.write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(sidecar)


def _compile_rknn(
    *,
    onnx_path: str,
    output_path: str,
    target_platform: str = DEFAULT_RKNN_TARGET,
    int8: bool = False,
    calibration_data: Iterable[np.ndarray] | None = None,
    metadata: Mapping[str, Any] | None = None,
    verbose: bool = False,
    config: Mapping[str, Any] | None = None,
    build: Mapping[str, Any] | None = None,
    simulator_inputs: np.ndarray | Sequence[np.ndarray] | None = None,
) -> tuple[str, list[np.ndarray] | None]:
    source = Path(onnx_path)
    if not source.is_file():
        raise FileNotFoundError(f"ONNX model not found: {source}")

    output = Path(output_path)
    if output.suffix.lower() != ".rknn":
        raise ValueError(f"RKNN output path must end in .rknn, got {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    target_platform = resolve_rknn_target(target_platform=target_platform)
    config_options = dict(config or {})
    if "target_platform" in config_options:
        raise ValueError(
            "Pass the RKNN target through name=, target=, or target_platform=; "
            "do not set rknn_config['target_platform']."
        )
    build_options = dict(build or {})
    reserved_build = {"do_quantization", "dataset"}.intersection(build_options)
    if reserved_build:
        names = ", ".join(sorted(reserved_build))
        raise ValueError(f"RKNN build options are managed by LibreYOLO: {names}")
    if int8 and calibration_data is None:
        raise ValueError("RKNN INT8 export requires calibration data.")

    RKNN = _load_rknn_class()
    rknn = RKNN(verbose=bool(verbose))
    completed = False
    simulator_outputs = None
    try:
        # The public vendor examples treat config() as exception-based and do
        # not require an integer status (some Toolkit2 versions return None).
        rknn.config(target_platform=target_platform, **config_options)
        _check_status("ONNX load", rknn.load_onnx(model=str(source)))

        if int8:
            assert calibration_data is not None
            with tempfile.TemporaryDirectory(prefix="libreyolo_rknn_calib_") as tmp:
                dataset = _write_calibration_dataset(
                    calibration_data,
                    Path(tmp),
                )
                _check_status(
                    "build",
                    rknn.build(
                        do_quantization=True,
                        dataset=str(dataset),
                        **build_options,
                    ),
                )
        else:
            _check_status(
                "build",
                rknn.build(do_quantization=False, **build_options),
            )

        _check_status("export", rknn.export_rknn(str(output)))
        if simulator_inputs is not None:
            input_list = (
                list(simulator_inputs)
                if isinstance(simulator_inputs, (list, tuple))
                else [simulator_inputs]
            )
            # Toolkit2's x86 simulator operates on the in-memory graph created
            # by load_onnx()+build(). Serialized .rknn artifacts are target
            # binaries and cannot be reloaded into the PC simulator.
            _check_status("simulator initialization", rknn.init_runtime())
            outputs = rknn.inference(inputs=input_list, data_format="nchw")
            if outputs is None:
                raise RuntimeError("RKNN simulator returned no outputs.")
            simulator_outputs = [np.asarray(value) for value in outputs]
        completed = True
    finally:
        rknn.release()

    if completed and metadata is not None:
        _write_metadata_sidecar(output, metadata)
    return str(output), simulator_outputs


def export_rknn(
    *,
    onnx_path: str,
    output_path: str,
    target_platform: str = DEFAULT_RKNN_TARGET,
    int8: bool = False,
    calibration_data: Iterable[np.ndarray] | None = None,
    metadata: Mapping[str, Any] | None = None,
    verbose: bool = False,
    config: Mapping[str, Any] | None = None,
    build: Mapping[str, Any] | None = None,
) -> str:
    """Compile a static ONNX graph into an RKNN model.

    The ONNX graph keeps LibreYOLO's native NCHW float input contract. Optional
    vendor configuration can be supplied through ``config`` and ``build``;
    target, quantization, and calibration keys remain owned by this wrapper.
    """
    result, _ = _compile_rknn(
        onnx_path=onnx_path,
        output_path=output_path,
        target_platform=target_platform,
        int8=int8,
        calibration_data=calibration_data,
        metadata=metadata,
        verbose=verbose,
        config=config,
        build=build,
    )
    return result


def export_rknn_with_simulator(
    *,
    onnx_path: str,
    output_path: str,
    simulator_inputs: np.ndarray | Sequence[np.ndarray],
    target_platform: str = DEFAULT_RKNN_TARGET,
    int8: bool = False,
    calibration_data: Iterable[np.ndarray] | None = None,
    metadata: Mapping[str, Any] | None = None,
    verbose: bool = False,
    config: Mapping[str, Any] | None = None,
    build: Mapping[str, Any] | None = None,
) -> tuple[str, list[np.ndarray]]:
    """Compile/export an RKNN model and run its in-memory PC simulator graph."""
    result, outputs = _compile_rknn(
        onnx_path=onnx_path,
        output_path=output_path,
        target_platform=target_platform,
        int8=int8,
        calibration_data=calibration_data,
        metadata=metadata,
        verbose=verbose,
        config=config,
        build=build,
        simulator_inputs=simulator_inputs,
    )
    assert outputs is not None
    return result, outputs


def run_rknn_simulator(
    onnx_path: str,
    inputs: np.ndarray | Sequence[np.ndarray],
    *,
    target_platform: str = DEFAULT_RKNN_TARGET,
    verbose: bool = False,
    config: Mapping[str, Any] | None = None,
    build: Mapping[str, Any] | None = None,
) -> list[np.ndarray]:
    """Compile an ONNX graph and run Toolkit2's board-free host simulator."""
    with tempfile.TemporaryDirectory(prefix="libreyolo_rknn_sim_") as tmp:
        _, outputs = export_rknn_with_simulator(
            onnx_path=onnx_path,
            output_path=str(Path(tmp) / "simulator.rknn"),
            simulator_inputs=inputs,
            target_platform=target_platform,
            verbose=verbose,
            config=config,
            build=build,
        )
    return outputs


def compare_rknn_outputs(
    reference_outputs: Sequence[np.ndarray],
    rknn_outputs: Sequence[np.ndarray],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    raise_on_failure: bool = True,
) -> list[dict[str, Any]]:
    """Measure tensor-level parity and optionally raise when it fails."""
    if len(reference_outputs) != len(rknn_outputs):
        raise AssertionError(
            "RKNN output count mismatch: "
            f"reference={len(reference_outputs)}, simulator={len(rknn_outputs)}"
        )

    metrics: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, (reference, actual) in enumerate(zip(reference_outputs, rknn_outputs)):
        expected = np.asarray(reference)
        observed = np.asarray(actual)
        if expected.shape != observed.shape:
            raise AssertionError(
                f"RKNN output {index} shape mismatch: "
                f"reference={expected.shape}, simulator={observed.shape}"
            )

        expected_float = expected.astype(np.float64, copy=False)
        observed_float = observed.astype(np.float64, copy=False)
        absolute = np.abs(expected_float - observed_float)
        denominator = np.maximum(np.abs(expected_float), np.finfo(np.float64).eps)
        relative = absolute / denominator
        squared = np.square(expected_float - observed_float)
        rmse = float(np.sqrt(squared.mean())) if squared.size else 0.0
        expected_rms = (
            float(np.sqrt(np.square(expected_float).mean()))
            if expected_float.size
            else 0.0
        )
        normalized_rmse = rmse / max(expected_rms, np.finfo(np.float64).eps)
        expected_flat = expected_float.ravel()
        observed_flat = observed_float.ravel()
        norm_product = float(
            np.linalg.norm(expected_flat) * np.linalg.norm(observed_flat)
        )
        if norm_product:
            cosine_similarity = float(
                np.dot(expected_flat, observed_flat) / norm_product
            )
        else:
            cosine_similarity = float(np.array_equal(expected_flat, observed_flat))
        within_tolerance = bool(
            np.allclose(
                expected_float,
                observed_float,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        )
        item = {
            "index": index,
            "shape": list(expected.shape),
            "max_abs_error": float(absolute.max(initial=0.0)),
            "mean_abs_error": float(absolute.mean()) if absolute.size else 0.0,
            "rmse": rmse,
            "normalized_rmse": normalized_rmse,
            "cosine_similarity": cosine_similarity,
            "max_rel_error": float(relative.max(initial=0.0)),
            "within_tolerance": within_tolerance,
        }
        metrics.append(item)
        if not within_tolerance:
            failures.append(
                f"output {index}: max_abs={item['max_abs_error']:.6g}, "
                f"max_rel={item['max_rel_error']:.6g}"
            )

    if failures and raise_on_failure:
        detail = "; ".join(failures)
        raise AssertionError(
            f"RKNN simulator parity failed (rtol={rtol}, atol={atol}): {detail}"
        )
    return metrics


def evaluate_rknn_metrics(
    metrics: Sequence[Mapping[str, Any]],
    *,
    min_cosine: float = DEFAULT_RKNN_MIN_COSINE,
    max_normalized_rmse: float = DEFAULT_RKNN_MAX_NORMALIZED_RMSE,
) -> tuple[bool, list[dict[str, Any]]]:
    """Apply the recorded floating-build acceptance gate to raw metrics.

    Strict elementwise ``allclose`` remains visible as ``within_tolerance``.
    An output is accepted when it passes that strict check or when both its
    cosine similarity and normalized RMSE pass the scale-independent gate.
    """
    evaluated: list[dict[str, Any]] = []
    for raw_item in metrics:
        item = dict(raw_item)
        cosine = float(item.get("cosine_similarity", float("nan")))
        normalized_rmse = float(item.get("normalized_rmse", float("nan")))
        scale_independent_pass = bool(
            np.isfinite(cosine)
            and np.isfinite(normalized_rmse)
            and cosine >= float(min_cosine)
            and normalized_rmse <= float(max_normalized_rmse)
        )
        item["accepted"] = bool(
            item.get("within_tolerance", False) or scale_independent_pass
        )
        item["scale_independent_pass"] = scale_independent_pass
        evaluated.append(item)
    return all(item["accepted"] for item in evaluated), evaluated


def verify_rknn_simulator_parity(
    onnx_path: str,
    input_data: np.ndarray,
    *,
    target_platform: str = DEFAULT_RKNN_TARGET,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    raise_on_failure: bool = True,
    verbose: bool = False,
    config: Mapping[str, Any] | None = None,
    build: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compile ONNX and compare its outputs with the RKNN host simulator."""
    reference = _run_onnx_reference(onnx_path, input_data)
    simulated = run_rknn_simulator(
        onnx_path,
        input_data,
        target_platform=target_platform,
        verbose=verbose,
        config=config,
        build=build,
    )
    return compare_rknn_outputs(
        reference,
        simulated,
        rtol=rtol,
        atol=atol,
        raise_on_failure=raise_on_failure,
    )


def _run_onnx_reference(
    onnx_path: str, input_data: np.ndarray
) -> list[np.ndarray]:
    """Run a single-input ONNX model on CPU for RKNN parity."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "RKNN parity requires onnxruntime. Install LibreYOLO's ONNX extra."
        ) from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model_inputs = session.get_inputs()
    if len(model_inputs) != 1:
        raise NotImplementedError(
            "The RKNN parity helper currently supports single-input models only."
        )
    array = np.asarray(input_data)
    reference = [
        np.asarray(output)
        for output in session.run(None, {model_inputs[0].name: array})
    ]
    return reference
