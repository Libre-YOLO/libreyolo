"""Internal process-level runner for the Qwen VLM confidence benchmark.

Run one benchmark per process, then compare the two persisted validator reports
in a separate invocation.  This module intentionally is not exported from the
validation package and does not change the public ``LibreVLM.val()`` contract.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import cv2
import PIL

from libreyolo.models.vlm.qwen3vl import LibreQwen3VL
from libreyolo.data import load_data_config

from .config import ValidationConfig
from .vlm_benchmark_dataset import (
    BenchmarkDatasetError,
    VerifiedBenchmarkRunInputs,
    verify_benchmark_run_inputs,
)
from .vlm_confidence_report import (
    PersistedRepeatComparison,
    VLMConfidenceReportError,
    compare_confidence_reports,
    read_confidence_report_identity,
)
from .vlm_confidence_validator import VLMConfidenceValidator

_MODEL_FAMILY = "qwen3vl"
_MODEL_SIZE = "2b"
_IMAGE_SIZE = 1024
_DEFAULT_CONF = 0.25
_CONFIDENCE_IOU = 0.5
_REPORT_NAME = "vlm_confidence_report.json"
_ENVELOPE_NAME = "vlm_confidence_run.json"
_RUN_SCHEMA = "libreyolo.vlm-confidence-benchmark-run.v2"
_STATUS_SCHEMA = "libreyolo.vlm-confidence-benchmark-status.v1"
_CONTEXT_SCHEMA = "libreyolo.vlm-confidence-benchmark-context.v2"
_DATASET_CONTEXT_SCHEMA = "libreyolo.vlm-confidence-benchmark-dataset.v1"
_DATASET_MANIFEST_SCHEMA = "libreyolo.vlm-benchmark-dataset.v1"
_REVIEW_SCHEMA = "libreyolo.vlm-benchmark-dataset-review.v1"
_REQUIRED_PARTITION_NAME = "promotion500"
_REQUIRED_PARTITION_ROLE = "zero_shot_confidence_promotion"
_REQUIRED_PARTITION_START = 0
_REQUIRED_PARTITION_STOP = 500
_REQUIRED_ANNOTATION_ARTIFACT = "annotations/instances_val2017_promotion500.json"
_REQUIRED_CLASS_COUNT = 80
_REVIEW_CHECKS = {
    "canonical_source",
    "image_attribution_sufficiency",
    "annotation_license_and_redistribution",
    "privacy_and_pii",
    "visual_quality",
    "selection_salt_freeze",
    "benchmark_suitability",
    "publication_upload_authorization",
}
_UTC_RFC3339 = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_MAX_SAFE_INTEGER = (1 << 53) - 1
_MAX_ENVELOPE_BYTES = 16 * 1024 * 1024
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_RUN_IDENTIFIER = re.compile(r"^[0-9a-f]{32}$")
_PROCESS_IDENTIFIER = secrets.token_hex(16)
_FASTER_COCO_ENV = "LIBREYOLO_FASTER_COCO_EVAL"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_OUTPUT_EXISTS = 3
EXIT_INVALID_REPORT = 4
EXIT_NOT_REPRODUCIBLE = 5
EXIT_RUN_FAILED = 6


class BenchmarkInputError(ValueError):
    """A benchmark request is invalid before execution starts."""


class BenchmarkOutputExistsError(FileExistsError):
    """The requested immutable output location is already occupied."""


class _CLIUsageError(ValueError):
    """An argparse failure that the CLI can return as strict JSON."""


@dataclass(frozen=True)
class BenchmarkArtifacts:
    """Paths and normalized metrics produced by one benchmark process."""

    output_dir: Path
    report_path: Path
    envelope_path: Path
    metrics: dict[str, float | None]
    nonfinite_metrics: tuple[str, ...]


@dataclass(frozen=True)
class _ValidatedEnvelope:
    run_id: str
    process_id: str
    report_sha256: str


class _JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CLIUsageError(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_exists(path: Path) -> bool:
    """Return true for ordinary paths and broken symlinks."""

    return os.path.lexists(os.fspath(path))


def _validate_seed(value: Any) -> int:
    if isinstance(value, bool):
        raise BenchmarkInputError("seed must be an integer in [0, 2**32 - 1]")
    try:
        seed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BenchmarkInputError("seed must be an integer in [0, 2**32 - 1]") from exc
    if str(seed) != str(value).strip() or not 0 <= seed <= 0xFFFFFFFF:
        raise BenchmarkInputError("seed must be an integer in [0, 2**32 - 1]")
    return seed


def _arg_seed(value: str) -> int:
    try:
        return _validate_seed(value)
    except BenchmarkInputError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _arg_tolerance(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            "tolerance must be a finite non-negative number"
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError(
            "tolerance must be a finite non-negative number"
        )
    return result


def _hash_randomization_enabled() -> bool:
    return bool(sys.flags.hash_randomization)


def configure_determinism(seed: int) -> dict[str, Any]:
    """Set process-wide deterministic RNG and Torch behavior.

    This must be called before constructing the model.  It deliberately does
    not restore prior settings because a benchmark invocation owns its process.
    """

    seed = _validate_seed(seed)
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed != "0" or _hash_randomization_enabled():
        raise BenchmarkInputError(
            "PYTHONHASHSEED must be set to 0 before starting the benchmark "
            "Python process"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": seed,
        "python_hash_seed": python_hash_seed,
        "python_hash_randomization": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _git_context() -> dict[str, Any]:
    root = _repo_root()
    try:
        commit_process = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status_process = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("could not inspect the benchmark git revision") from exc
    commit = commit_process.stdout.strip().lower()
    if (
        commit_process.returncode != 0
        or not 40 <= len(commit) <= 64
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise RuntimeError("could not resolve the benchmark git commit")
    if status_process.returncode != 0:
        raise RuntimeError("could not inspect the benchmark worktree state")
    return {"commit": commit, "dirty": bool(status_process.stdout)}


def _runtime_context(*, requested_device: str, resolved_device: str) -> dict[str, Any]:
    cudnn_version = torch.backends.cudnn.version()
    package_versions = {}
    for package in (
        "transformers",
        "huggingface_hub",
        "tokenizers",
        "safetensors",
        "pycocotools",
    ):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"cannot identify required benchmark package {package!r}"
            ) from exc
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "pillow": str(PIL.__version__),
        "opencv": str(cv2.__version__),
        "packages": package_versions,
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn": None if cudnn_version is None else int(cudnn_version),
        "nvidia_driver": _nvidia_driver_version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_device": requested_device,
        "resolved_device": resolved_device,
    }


def _nvidia_driver_version() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot identify the NVIDIA driver version") from exc
    versions = sorted(
        {line.strip() for line in process.stdout.splitlines() if line.strip()}
    )
    if process.returncode != 0 or not versions:
        raise RuntimeError("cannot identify the NVIDIA driver version")
    return ",".join(versions)


def _attention_backends(model: Any) -> dict[str, str]:
    target = getattr(model, "model", None)
    root = getattr(target, "config", None)
    configs = {
        "model": root,
        "text": getattr(root, "text_config", None),
        "vision": getattr(root, "vision_config", None),
    }
    result = {}
    for name, config in configs.items():
        if config is None:
            continue
        for field in (
            "_attn_implementation",
            "_attn_implementation_internal",
            "attn_implementation",
        ):
            value = getattr(config, field, None)
            if isinstance(value, str) and value:
                result[name] = value
                break
    if not result:
        raise RuntimeError("cannot identify the model attention backend")
    if any("flash" in value.lower() for value in result.values()):
        raise BenchmarkInputError(
            "FlashAttention is disabled for the reproducibility benchmark"
        )
    return result


def _resolved_model_device(model: Any) -> str:
    target = getattr(model, "model", None)
    tensors = []
    if target is not None:
        tensors.extend(target.parameters())
        tensors.extend(target.buffers())
    devices = {tensor.device for tensor in tensors}
    if len(devices) > 1:
        raise RuntimeError("benchmark model spans multiple devices")
    device = next(iter(devices), getattr(model, "device", None))
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return str(device)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_metrics(
    metrics: Mapping[str, Any],
) -> tuple[dict[str, float | None], tuple[str, ...]]:
    normalized: dict[str, float | None] = {}
    nonfinite = []
    for key, value in sorted(metrics.items()):
        if not isinstance(key, str):
            raise TypeError("benchmark metric names must be strings")
        if isinstance(value, bool):
            raise TypeError(f"benchmark metric {key!r} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"benchmark metric {key!r} must be numeric") from exc
        if math.isfinite(number):
            normalized[key] = number
        else:
            normalized[key] = None
            nonfinite.append(key)
    return normalized, tuple(nonfinite)


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(_json_text(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _staged_output(destination: Path) -> Iterator[Path]:
    if _path_exists(destination):
        raise BenchmarkOutputExistsError(
            f"benchmark output already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(f".{destination.name}.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise BenchmarkOutputExistsError(
            f"benchmark output is reserved by another process: {destination}"
        ) from exc
    stage: Path | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(f"{os.getpid()}\n")
        stage = Path(
            tempfile.mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.tmp-",
            )
        )
        yield stage
        if _path_exists(destination):
            raise BenchmarkOutputExistsError(
                f"benchmark output appeared during execution: {destination}"
            )
        stage.rename(destination)
        stage = None
    finally:
        if stage is not None and stage.is_dir():
            shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


def _require_pycocotools() -> None:
    """Import the fixed evaluator before any model construction or download."""

    try:
        import_module("pycocotools.coco")
    except ImportError as exc:
        raise BenchmarkInputError(
            "pycocotools is required for the VLM confidence benchmark"
        ) from exc


def _portable_dataset_context(
    verified: VerifiedBenchmarkRunInputs,
) -> dict[str, Any]:
    """Build the path-free dataset identity embedded in the benchmark report."""

    try:
        annotation_artifact = verified.annotation_path.relative_to(
            verified.manifest_path.parent
        ).as_posix()
    except ValueError as exc:
        raise BenchmarkInputError(
            "verified benchmark annotation artifact must be inside the manifest bundle"
        ) from exc

    review = dict(verified.review_attestation)
    context = {
        "schema": _DATASET_CONTEXT_SCHEMA,
        "manifest": {
            "schema": _DATASET_MANIFEST_SCHEMA,
            "sha256": verified.manifest_sha256,
        },
        "source": {
            "canonical_annotation_sha256": verified.source_canonical_sha256,
            "file_sha256": verified.source_file_sha256,
            "file_size_bytes": verified.source_file_size_bytes,
            "selected_image_identity_sha256": (verified.selected_image_identity_sha256),
        },
        "partition": {
            "name": verified.partition_name,
            "role": verified.partition_role,
            "start": verified.partition_start,
            "stop": verified.partition_stop,
            "image_count": verified.partition_stop - verified.partition_start,
            "annotation_artifact": annotation_artifact,
            "annotation_size_bytes": verified.annotation_size_bytes,
            "annotation_sha256": verified.annotation_sha256,
        },
        "classes": {
            "count": len(verified.class_names),
            "names": list(verified.class_names),
            "category_ids": [
                int(category["id"]) for category in verified.expected_categories
            ],
        },
        "review": {
            "schema": review.get("schema"),
            "sha256": verified.review_attestation_sha256,
            "manifest_sha256": review.get("manifest_sha256"),
            "partition_role": review.get("partition_role"),
            "status": review.get("status"),
            "reviewer": review.get("reviewer"),
            "reviewed_at": review.get("reviewed_at"),
            "checks": review.get("checks"),
        },
    }
    try:
        return _validate_dataset_context(
            context, "verified_inputs", "$.execution_context.dataset"
        )
    except VLMConfidenceReportError as exc:
        raise BenchmarkInputError(
            f"verified benchmark inputs violate the runner contract: {exc}"
        ) from exc


def _resolved_existing_path(value: Any, label: str, *, directory: bool) -> Path:
    if not isinstance(value, str) or not value:
        raise BenchmarkInputError(f"resolved benchmark {label} must be a path string")
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise BenchmarkInputError(
            f"resolved benchmark {label} does not exist: {path}"
        ) from exc
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise BenchmarkInputError(
            f"resolved benchmark {label} must be an existing {kind}: {resolved}"
        )
    return resolved


@contextmanager
def _temporary_verified_dataset_yaml(
    verified: VerifiedBenchmarkRunInputs,
) -> Iterator[Path]:
    """Create and preflight the only dataset config the runner may execute."""

    payload = {
        "path": str(verified.images_dir),
        "val": str(verified.images_dir),
        "annotations": {"val": str(verified.annotation_path)},
        "nc": len(verified.class_names),
        "names": list(verified.class_names),
    }
    with tempfile.TemporaryDirectory(
        prefix="libreyolo-vlm-confidence-dataset-"
    ) as temporary_root:
        dataset_yaml = Path(temporary_root) / "verified_dataset.yaml"
        _write_json_atomic(dataset_yaml, payload)
        try:
            resolved = load_data_config(
                str(dataset_yaml), autodownload=False, allow_scripts=False
            )
        except (OSError, TypeError, ValueError) as exc:
            raise BenchmarkInputError(
                f"verified benchmark dataset config could not be resolved: {exc}"
            ) from exc

        if not isinstance(resolved, Mapping):
            raise BenchmarkInputError(
                "generated benchmark dataset config did not resolve to a mapping"
            )
        if "download" in resolved:
            raise BenchmarkInputError(
                "generated benchmark dataset config must not define a download"
            )
        annotations = resolved.get("annotations")
        if not isinstance(annotations, Mapping) or set(annotations) != {"val"}:
            raise BenchmarkInputError(
                "generated benchmark dataset config must define only annotations.val"
            )
        image_root = _resolved_existing_path(
            resolved.get("val"), "validation image root", directory=True
        )
        annotation = _resolved_existing_path(
            resolved.get("val_annotation_file"),
            "validation annotation artifact",
            directory=False,
        )
        expected_image_root = verified.images_dir.resolve(strict=True)
        expected_annotation = verified.annotation_path.resolve(strict=True)
        root = _resolved_existing_path(
            resolved.get("root"), "dataset root", directory=True
        )
        configured_annotation = _resolved_existing_path(
            annotations["val"], "configured annotation artifact", directory=False
        )
        if root != expected_image_root:
            raise BenchmarkInputError(
                "resolved dataset root does not match the verified image root"
            )
        if image_root != expected_image_root:
            raise BenchmarkInputError(
                "resolved validation image root does not match the verified input"
            )
        if annotation != expected_annotation:
            raise BenchmarkInputError(
                "resolved validation annotation does not match the verified artifact"
            )
        if configured_annotation != expected_annotation:
            raise BenchmarkInputError(
                "configured validation annotation does not match the verified artifact"
            )
        if type(resolved.get("nc")) is not int or resolved["nc"] != len(
            verified.class_names
        ):
            raise BenchmarkInputError(
                "resolved benchmark class count does not match the verified input"
            )
        if resolved.get("names") != list(verified.class_names):
            raise BenchmarkInputError(
                "resolved benchmark class names do not match the verified input"
            )
        yield dataset_yaml


def run_benchmark(
    manifest: str | os.PathLike[str],
    source_annotations: str | os.PathLike[str],
    images_dir: str | os.PathLike[str],
    review_attestation: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    seed: int = 0,
    device: str = "auto",
) -> BenchmarkArtifacts:
    """Run one fresh Qwen3-VL-2B confidence benchmark into an immutable directory."""

    seed = _validate_seed(seed)
    if not isinstance(device, str) or not device.strip():
        raise BenchmarkInputError("device must be a non-empty string")
    faster_coco_override = os.environ.get(_FASTER_COCO_ENV)
    if faster_coco_override is not None and faster_coco_override.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise BenchmarkInputError(
            f"{_FASTER_COCO_ENV} must not enable faster-coco-eval for this benchmark"
        )
    requested_destination = Path(output_root).expanduser()
    if _path_exists(requested_destination):
        raise BenchmarkOutputExistsError(
            f"benchmark output already exists: {requested_destination}"
        )
    destination = requested_destination.resolve(strict=False)
    repository = _repo_root()
    if destination == repository or repository in destination.parents:
        raise BenchmarkInputError(
            "benchmark output must be outside the git worktree so generated "
            "artifacts cannot change the recorded source state"
        )
    if _path_exists(destination):
        raise BenchmarkOutputExistsError(
            f"benchmark output already exists: {destination}"
        )
    git_context = _git_context()
    if git_context["dirty"]:
        raise BenchmarkInputError(
            "the benchmark requires a clean git worktree so its code identity is "
            "immutable"
        )

    try:
        verified = verify_benchmark_run_inputs(
            manifest,
            source_annotations,
            images_dir,
            review_attestation,
            required_role=_REQUIRED_PARTITION_ROLE,
        )
    except BenchmarkDatasetError as exc:
        raise BenchmarkInputError(f"invalid benchmark dataset evidence: {exc}") from exc
    dataset_context = _portable_dataset_context(verified)
    _require_pycocotools()

    normalized_metrics: dict[str, float | None]
    nonfinite_metrics: tuple[str, ...]
    with _temporary_verified_dataset_yaml(verified) as dataset_yaml:
        with _staged_output(destination) as stage:
            determinism = configure_determinism(seed)
            model = LibreQwen3VL(size=_MODEL_SIZE, device=device)
            resolved_device = _resolved_model_device(model)
            runtime_context = _runtime_context(
                requested_device=device, resolved_device=resolved_device
            )
            runtime_context["attention_backends"] = _attention_backends(model)
            execution_context = {
                "schema": _CONTEXT_SCHEMA,
                "git": git_context,
                "runtime": runtime_context,
                "determinism": determinism,
                "dataset": dataset_context,
            }
            config = ValidationConfig(
                data=str(dataset_yaml),
                split="val",
                batch_size=1,
                imgsz=_IMAGE_SIZE,
                device=device,
                save_dir=str(stage),
                num_workers=0,
                allow_download_scripts=False,
                save_json=True,
                save_plots=True,
                faster_coco_eval=False,
            )
            validator = VLMConfidenceValidator(
                model,
                config,
                seed=seed,
                default_conf=_DEFAULT_CONF,
                confidence_iou=_CONFIDENCE_IOU,
                benchmark_context=execution_context,
                verified_dataset=verified,
            )
            metrics = validator.run()
            if not isinstance(metrics, Mapping):
                raise TypeError("VLM confidence validator must return a metric mapping")
            normalized_metrics, nonfinite_metrics = _normalized_metrics(metrics)
            if _git_context() != git_context:
                raise RuntimeError(
                    "the benchmark git revision or worktree changed during execution"
                )

            report = stage / _REPORT_NAME
            self_comparison = compare_confidence_reports(report, report)
            if not self_comparison.reproducible:
                raise RuntimeError("persisted benchmark report failed self-comparison")
            envelope = {
                "schema": _RUN_SCHEMA,
                "run_id": secrets.token_hex(16),
                "process_id": _PROCESS_IDENTIFIER,
                "request": {
                    "manifest": str(verified.manifest_path),
                    "annotations": str(verified.source_annotations),
                    "images_dir": str(verified.images_dir),
                    "review_attestation": str(verified.review_attestation_path),
                    "seed": seed,
                    "model_family": _MODEL_FAMILY,
                    "model_size": _MODEL_SIZE,
                    "device": device,
                    "imgsz": _IMAGE_SIZE,
                    "default_conf": _DEFAULT_CONF,
                    "confidence_iou": _CONFIDENCE_IOU,
                },
                # The same context is also embedded in benchmark_config so the
                # strict report comparator treats code/runtime/dataset drift as
                # a different configuration instead of merely recording it here.
                "execution_context": execution_context,
                "report": {
                    "path": _REPORT_NAME,
                    "sha256": _file_sha256(report),
                },
                "metrics": normalized_metrics,
                "nonfinite_metrics": list(nonfinite_metrics),
            }
            _write_json_atomic(stage / _ENVELOPE_NAME, envelope)
            staged_envelope = _load_runner_envelope(report, "staged_run")
            if (
                staged_envelope.run_id != envelope["run_id"]
                or staged_envelope.process_id != envelope["process_id"]
                or staged_envelope.report_sha256 != envelope["report"]["sha256"]
            ):
                raise RuntimeError(
                    "persisted benchmark envelope failed its writer-reader round trip"
                )

    return BenchmarkArtifacts(
        output_dir=destination,
        report_path=destination / _REPORT_NAME,
        envelope_path=destination / _ENVELOPE_NAME,
        metrics=normalized_metrics,
        nonfinite_metrics=nonfinite_metrics,
    )


def _envelope_error(label: str, path: str, message: str) -> VLMConfidenceReportError:
    return VLMConfidenceReportError(f"{label}:{path}: {message}")


def _exact_object(
    value: Any, expected: set[str], label: str, path: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _envelope_error(label, path, "must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unsupported " + ", ".join(extra))
        raise _envelope_error(label, path, "; ".join(details))
    return value


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VLMConfidenceReportError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise VLMConfidenceReportError(f"non-finite JSON constant {value!r} is forbidden")


def _load_envelope_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise _envelope_error(label, "$", f"missing companion {_ENVELOPE_NAME}")
    with path.open("rb") as stream:
        payload = stream.read(_MAX_ENVELOPE_BYTES + 1)
    if len(payload) > _MAX_ENVELOPE_BYTES:
        raise _envelope_error(
            label, "$", f"exceeds the {_MAX_ENVELOPE_BYTES}-byte limit"
        )
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _envelope_error(label, "$", "is not strict UTF-8") from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_json_constant,
        )
    except RecursionError as exc:
        raise _envelope_error(label, "$", "JSON nesting is too deep") from exc
    except json.JSONDecodeError as exc:
        raise _envelope_error(label, "$", f"invalid JSON: {exc.msg}") from exc
    except VLMConfidenceReportError as exc:
        raise _envelope_error(label, "$", str(exc)) from exc
    except ValueError as exc:
        raise _envelope_error(label, "$", "contains an invalid JSON value") from exc
    if not isinstance(decoded, dict):
        raise _envelope_error(label, "$", "top level must be a JSON object")
    return decoded


def _finite_number(value: Any, label: str, path: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise _envelope_error(label, path, "must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _envelope_error(label, path, "must be a finite number") from exc
    if not math.isfinite(result):
        raise _envelope_error(label, path, "must be a finite number")
    return result


def _validate_dataset_context(value: Any, label: str, path: str) -> dict[str, Any]:
    dataset = _exact_object(
        value,
        {"schema", "manifest", "source", "partition", "classes", "review"},
        label,
        path,
    )
    if dataset["schema"] != _DATASET_CONTEXT_SCHEMA:
        raise _envelope_error(
            label,
            f"{path}.schema",
            f"must equal {_DATASET_CONTEXT_SCHEMA!r}",
        )

    def digest(raw: Any, field_path: str) -> str:
        if not isinstance(raw, str) or not _HEX_DIGEST.fullmatch(raw):
            raise _envelope_error(
                label, field_path, "must be a lowercase SHA256 digest"
            )
        return raw

    manifest = _exact_object(
        dataset["manifest"], {"schema", "sha256"}, label, f"{path}.manifest"
    )
    if manifest["schema"] != _DATASET_MANIFEST_SCHEMA:
        raise _envelope_error(
            label,
            f"{path}.manifest.schema",
            f"must equal {_DATASET_MANIFEST_SCHEMA!r}",
        )
    manifest_sha256 = digest(manifest["sha256"], f"{path}.manifest.sha256")

    source = _exact_object(
        dataset["source"],
        {
            "canonical_annotation_sha256",
            "file_sha256",
            "file_size_bytes",
            "selected_image_identity_sha256",
        },
        label,
        f"{path}.source",
    )
    digest(
        source["canonical_annotation_sha256"],
        f"{path}.source.canonical_annotation_sha256",
    )
    digest(source["file_sha256"], f"{path}.source.file_sha256")
    if (
        type(source["file_size_bytes"]) is not int
        or source["file_size_bytes"] <= 0
        or source["file_size_bytes"] > _MAX_SAFE_INTEGER
    ):
        raise _envelope_error(
            label,
            f"{path}.source.file_size_bytes",
            "must be a positive exact JSON integer",
        )
    digest(
        source["selected_image_identity_sha256"],
        f"{path}.source.selected_image_identity_sha256",
    )

    partition = _exact_object(
        dataset["partition"],
        {
            "name",
            "role",
            "start",
            "stop",
            "image_count",
            "annotation_artifact",
            "annotation_size_bytes",
            "annotation_sha256",
        },
        label,
        f"{path}.partition",
    )
    expected_partition = {
        "name": _REQUIRED_PARTITION_NAME,
        "role": _REQUIRED_PARTITION_ROLE,
        "start": _REQUIRED_PARTITION_START,
        "stop": _REQUIRED_PARTITION_STOP,
        "image_count": _REQUIRED_PARTITION_STOP - _REQUIRED_PARTITION_START,
        "annotation_artifact": _REQUIRED_ANNOTATION_ARTIFACT,
    }
    for field, expected in expected_partition.items():
        actual = partition[field]
        if type(actual) is not type(expected) or actual != expected:
            raise _envelope_error(
                label,
                f"{path}.partition.{field}",
                f"must equal {expected!r}",
            )
    if (
        type(partition["annotation_size_bytes"]) is not int
        or partition["annotation_size_bytes"] <= 0
        or partition["annotation_size_bytes"] > _MAX_SAFE_INTEGER
    ):
        raise _envelope_error(
            label,
            f"{path}.partition.annotation_size_bytes",
            "must be a positive exact JSON integer",
        )
    digest(partition["annotation_sha256"], f"{path}.partition.annotation_sha256")

    classes = _exact_object(
        dataset["classes"],
        {"count", "names", "category_ids"},
        label,
        f"{path}.classes",
    )
    names = classes["names"]
    if type(classes["count"]) is not int or classes["count"] != _REQUIRED_CLASS_COUNT:
        raise _envelope_error(
            label,
            f"{path}.classes.count",
            f"must equal {_REQUIRED_CLASS_COUNT}",
        )
    if (
        not isinstance(names, list)
        or len(names) != _REQUIRED_CLASS_COUNT
        or len(set(names)) != len(names)
        or any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in names
        )
    ):
        raise _envelope_error(
            label,
            f"{path}.classes.names",
            "must contain 80 unique normalized non-empty class names",
        )
    category_ids = classes["category_ids"]
    if (
        not isinstance(category_ids, list)
        or len(category_ids) != _REQUIRED_CLASS_COUNT
        or any(
            type(category_id) is not int or not 0 <= category_id <= _MAX_SAFE_INTEGER
            for category_id in category_ids
        )
        or category_ids != sorted(set(category_ids))
    ):
        raise _envelope_error(
            label,
            f"{path}.classes.category_ids",
            "must contain 80 unique increasing non-negative exact JSON integers",
        )

    review = _exact_object(
        dataset["review"],
        {
            "schema",
            "sha256",
            "manifest_sha256",
            "partition_role",
            "status",
            "reviewer",
            "reviewed_at",
            "checks",
        },
        label,
        f"{path}.review",
    )
    if review["schema"] != _REVIEW_SCHEMA:
        raise _envelope_error(
            label,
            f"{path}.review.schema",
            f"must equal {_REVIEW_SCHEMA!r}",
        )
    digest(review["sha256"], f"{path}.review.sha256")
    if (
        digest(review["manifest_sha256"], f"{path}.review.manifest_sha256")
        != manifest_sha256
    ):
        raise _envelope_error(
            label,
            f"{path}.review.manifest_sha256",
            "must match the verified manifest digest",
        )
    if review["partition_role"] != _REQUIRED_PARTITION_ROLE:
        raise _envelope_error(
            label,
            f"{path}.review.partition_role",
            f"must equal {_REQUIRED_PARTITION_ROLE!r}",
        )
    if review["status"] != "approved":
        raise _envelope_error(label, f"{path}.review.status", "must equal 'approved'")
    reviewer = review["reviewer"]
    if (
        not isinstance(reviewer, str)
        or not reviewer
        or reviewer != reviewer.strip()
        or len(reviewer) > 256
    ):
        raise _envelope_error(
            label,
            f"{path}.review.reviewer",
            "must be a normalized non-empty string of at most 256 characters",
        )
    reviewed_at = review["reviewed_at"]
    if not isinstance(reviewed_at, str) or not _UTC_RFC3339.fullmatch(reviewed_at):
        raise _envelope_error(
            label,
            f"{path}.review.reviewed_at",
            "must be a UTC RFC3339 timestamp ending in Z",
        )
    try:
        parsed_reviewed_at = datetime.fromisoformat(
            reviewed_at.removesuffix("Z") + "+00:00"
        )
    except ValueError as exc:
        raise _envelope_error(
            label, f"{path}.review.reviewed_at", "is not a valid timestamp"
        ) from exc
    if parsed_reviewed_at.utcoffset() != timezone.utc.utcoffset(parsed_reviewed_at):
        raise _envelope_error(label, f"{path}.review.reviewed_at", "must use UTC")
    checks = _exact_object(
        review["checks"], _REVIEW_CHECKS, label, f"{path}.review.checks"
    )
    if any(value is not True for value in checks.values()):
        raise _envelope_error(
            label,
            f"{path}.review.checks",
            "every required manual-review check must be true",
        )
    return dict(dataset)


def _validate_execution_context(
    value: Any, request: Mapping[str, Any], label: str
) -> dict[str, Any]:
    context = _exact_object(
        value,
        {"schema", "git", "runtime", "determinism", "dataset"},
        label,
        "$.execution_context",
    )
    if context["schema"] != _CONTEXT_SCHEMA:
        raise _envelope_error(
            label, "$.execution_context.schema", f"must equal {_CONTEXT_SCHEMA!r}"
        )
    _validate_dataset_context(context["dataset"], label, "$.execution_context.dataset")
    git = _exact_object(
        context["git"], {"commit", "dirty"}, label, "$.execution_context.git"
    )
    if not isinstance(git["commit"], str) or not _GIT_COMMIT.fullmatch(git["commit"]):
        raise _envelope_error(
            label, "$.execution_context.git.commit", "must be a lowercase git digest"
        )
    if git["dirty"] is not False:
        raise _envelope_error(label, "$.execution_context.git.dirty", "must be false")
    runtime = _exact_object(
        context["runtime"],
        {
            "python",
            "implementation",
            "platform",
            "torch",
            "numpy",
            "pillow",
            "opencv",
            "packages",
            "cuda_runtime",
            "cudnn",
            "nvidia_driver",
            "cuda_available",
            "requested_device",
            "resolved_device",
            "attention_backends",
        },
        label,
        "$.execution_context.runtime",
    )
    for field in (
        "python",
        "implementation",
        "platform",
        "torch",
        "numpy",
        "pillow",
        "opencv",
        "requested_device",
        "resolved_device",
    ):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise _envelope_error(
                label,
                f"$.execution_context.runtime.{field}",
                "must be a non-empty string",
            )
    packages = _exact_object(
        runtime["packages"],
        {
            "transformers",
            "huggingface_hub",
            "tokenizers",
            "safetensors",
            "pycocotools",
        },
        label,
        "$.execution_context.runtime.packages",
    )
    for package, version in packages.items():
        if not isinstance(version, str) or not version:
            raise _envelope_error(
                label,
                f"$.execution_context.runtime.packages.{package}",
                "must be a non-empty version string",
            )
    if runtime["cuda_runtime"] is not None and not isinstance(
        runtime["cuda_runtime"], str
    ):
        raise _envelope_error(
            label,
            "$.execution_context.runtime.cuda_runtime",
            "must be null or a string",
        )
    if runtime["cudnn"] is not None and (
        isinstance(runtime["cudnn"], bool) or not isinstance(runtime["cudnn"], int)
    ):
        raise _envelope_error(
            label, "$.execution_context.runtime.cudnn", "must be null or an integer"
        )
    if runtime["nvidia_driver"] is not None and (
        not isinstance(runtime["nvidia_driver"], str) or not runtime["nvidia_driver"]
    ):
        raise _envelope_error(
            label,
            "$.execution_context.runtime.nvidia_driver",
            "must be null or a non-empty string",
        )
    if not isinstance(runtime["cuda_available"], bool):
        raise _envelope_error(
            label,
            "$.execution_context.runtime.cuda_available",
            "must be boolean",
        )
    attention_backends = runtime["attention_backends"]
    if (
        not isinstance(attention_backends, Mapping)
        or not attention_backends
        or any(
            key not in {"model", "text", "vision"}
            or not isinstance(value, str)
            or not value
            for key, value in attention_backends.items()
        )
        or any("flash" in value.lower() for value in attention_backends.values())
    ):
        raise _envelope_error(
            label,
            "$.execution_context.runtime.attention_backends",
            "must identify one or more model attention backends",
        )
    if request["device"] != runtime["requested_device"]:
        raise _envelope_error(
            label,
            "$.request.device",
            "must equal execution_context.runtime.requested_device",
        )
    determinism = _exact_object(
        context["determinism"],
        {
            "seed",
            "python_hash_seed",
            "python_hash_randomization",
            "cublas_workspace_config",
            "torch_deterministic_algorithms",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
        },
        label,
        "$.execution_context.determinism",
    )
    if type(determinism["seed"]) is not int or determinism["seed"] != request["seed"]:
        raise _envelope_error(
            label,
            "$.execution_context.determinism.seed",
            "must equal request.seed",
        )
    expected_determinism = {
        "python_hash_seed": "0",
        "python_hash_randomization": False,
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    for field, expected in expected_determinism.items():
        if (
            type(determinism[field]) is not type(expected)
            or determinism[field] != expected
        ):
            raise _envelope_error(
                label,
                f"$.execution_context.determinism.{field}",
                f"must equal {expected!r}",
            )
    return dict(context)


def _load_runner_envelope(report_value: Any, label: str) -> _ValidatedEnvelope:
    if isinstance(report_value, bool) or not isinstance(
        report_value, (str, os.PathLike)
    ):
        raise TypeError(f"{label} must be a filesystem path")
    report_path = Path(report_value).expanduser().resolve(strict=False)
    envelope_path = report_path.parent / _ENVELOPE_NAME
    envelope = _load_envelope_json(envelope_path, label)
    root = _exact_object(
        envelope,
        {
            "schema",
            "run_id",
            "process_id",
            "request",
            "execution_context",
            "report",
            "metrics",
            "nonfinite_metrics",
        },
        label,
        "$",
    )
    if root["schema"] != _RUN_SCHEMA:
        raise _envelope_error(label, "$.schema", f"must equal {_RUN_SCHEMA!r}")
    for field, pattern in (
        ("run_id", _RUN_IDENTIFIER),
        ("process_id", _RUN_IDENTIFIER),
    ):
        if not isinstance(root[field], str) or not pattern.fullmatch(root[field]):
            raise _envelope_error(
                label, f"$.{field}", "must be a 32-character lowercase hex identifier"
            )
    request = _exact_object(
        root["request"],
        {
            "manifest",
            "annotations",
            "images_dir",
            "review_attestation",
            "seed",
            "model_family",
            "model_size",
            "device",
            "imgsz",
            "default_conf",
            "confidence_iou",
        },
        label,
        "$.request",
    )
    for field in (
        "manifest",
        "annotations",
        "images_dir",
        "review_attestation",
        "device",
    ):
        if not isinstance(request[field], str) or not request[field]:
            raise _envelope_error(label, f"$.request.{field}", "must be non-empty")
    for field in ("manifest", "annotations", "images_dir", "review_attestation"):
        if not Path(request[field]).is_absolute():
            raise _envelope_error(
                label, f"$.request.{field}", "must be an absolute operational path"
            )
    if request["model_family"] != _MODEL_FAMILY or request["model_size"] != _MODEL_SIZE:
        raise _envelope_error(
            label, "$.request", "must identify the fixed Qwen3-VL-2B benchmark"
        )
    if (
        isinstance(request["seed"], bool)
        or not isinstance(request["seed"], int)
        or not 0 <= request["seed"] <= 0xFFFFFFFF
    ):
        raise _envelope_error(label, "$.request.seed", "must be a 32-bit integer")
    if type(request["imgsz"]) is not int or request["imgsz"] != _IMAGE_SIZE:
        raise _envelope_error(label, "$.request.imgsz", f"must equal {_IMAGE_SIZE}")
    if (
        _finite_number(request["default_conf"], label, "$.request.default_conf")
        != _DEFAULT_CONF
    ):
        raise _envelope_error(
            label, "$.request.default_conf", f"must equal {_DEFAULT_CONF}"
        )
    if (
        _finite_number(request["confidence_iou"], label, "$.request.confidence_iou")
        != _CONFIDENCE_IOU
    ):
        raise _envelope_error(
            label, "$.request.confidence_iou", f"must equal {_CONFIDENCE_IOU}"
        )
    execution_context = _validate_execution_context(
        root["execution_context"], request, label
    )
    report = _exact_object(root["report"], {"path", "sha256"}, label, "$.report")
    if report["path"] != _REPORT_NAME or report_path.name != _REPORT_NAME:
        raise _envelope_error(
            label, "$.report.path", f"must identify companion {_REPORT_NAME}"
        )
    if not isinstance(report["sha256"], str) or not _HEX_DIGEST.fullmatch(
        report["sha256"]
    ):
        raise _envelope_error(
            label, "$.report.sha256", "must be a lowercase SHA256 digest"
        )
    metrics = root["metrics"]
    if not isinstance(metrics, Mapping) or any(
        not isinstance(key, str) for key in metrics
    ):
        raise _envelope_error(label, "$.metrics", "must be a string-keyed object")
    null_metrics = []
    for key, value in metrics.items():
        if value is None:
            null_metrics.append(key)
        else:
            _finite_number(value, label, f"$.metrics.{key}")
    nonfinite = root["nonfinite_metrics"]
    if (
        not isinstance(nonfinite, list)
        or any(not isinstance(key, str) for key in nonfinite)
        or nonfinite != sorted(set(nonfinite))
        or nonfinite != sorted(null_metrics)
    ):
        raise _envelope_error(
            label,
            "$.nonfinite_metrics",
            "must exactly list the null metric keys in sorted order",
        )
    report_digest, benchmark_config, report_metrics = read_confidence_report_identity(
        report_path, label=f"{label}_report"
    )
    if report_digest != report["sha256"]:
        raise _envelope_error(
            label, "$.report.sha256", "does not match the companion report bytes"
        )
    report_context = benchmark_config.get("benchmark_run")
    if not isinstance(report_context, Mapping) or _json_text(
        report_context
    ) != _json_text(execution_context):
        raise _envelope_error(
            label,
            "$.execution_context",
            "does not match benchmark_config.benchmark_run in the report",
        )
    if (
        benchmark_config.get("family") != _MODEL_FAMILY
        or benchmark_config.get("size") != _MODEL_SIZE
    ):
        raise _envelope_error(
            label,
            "$.request",
            "does not match the model identity in the report",
        )
    report_class_names = benchmark_config.get("class_names")
    expected_class_names = execution_context["dataset"]["classes"]["names"]
    if not isinstance(report_class_names, (list, tuple)) or list(
        report_class_names
    ) != list(expected_class_names):
        raise _envelope_error(
            label,
            "$.execution_context.dataset.classes.names",
            "does not match benchmark_config.class_names in the report",
        )
    if benchmark_config.get("seed") != request["seed"]:
        raise _envelope_error(
            label, "$.request.seed", "does not match the report configuration"
        )
    evaluation = benchmark_config.get("evaluation")
    expected_category_map = {
        str(index): category_id
        for index, category_id in enumerate(
            execution_context["dataset"]["classes"]["category_ids"]
        )
    }
    actual_category_map = (
        evaluation.get("label_to_category_id")
        if isinstance(evaluation, Mapping)
        else None
    )
    if actual_category_map != expected_category_map:
        raise _envelope_error(
            label,
            "$.execution_context.dataset.classes.category_ids",
            "does not match benchmark_config.evaluation.label_to_category_id in the report",
        )
    report_imgsz = evaluation.get("imgsz") if isinstance(evaluation, Mapping) else None
    scalar_imgsz_matches = type(report_imgsz) is int and report_imgsz == _IMAGE_SIZE
    pair_imgsz_matches = (
        isinstance(report_imgsz, list)
        and len(report_imgsz) == 2
        and all(type(value) is int for value in report_imgsz)
        and report_imgsz == [_IMAGE_SIZE, _IMAGE_SIZE]
    )
    if not isinstance(evaluation, Mapping) or not (
        scalar_imgsz_matches or pair_imgsz_matches
    ):
        raise _envelope_error(
            label, "$.request.imgsz", "does not match the report configuration"
        )
    if evaluation.get("faster_coco_eval") is not False:
        raise _envelope_error(
            label,
            "$.execution_context",
            "report must use the pinned pycocotools evaluator path",
        )
    evaluator_backend = evaluation.get("backend")
    if not isinstance(evaluator_backend, str) or not (
        evaluator_backend.startswith("pycocotools ")
        or evaluator_backend == "not-run:no-predictions"
    ):
        raise _envelope_error(
            label,
            "$.report.benchmark_config.evaluation.backend",
            "must record the pycocotools evaluator backend",
        )
    confidence_evaluation = benchmark_config.get("confidence_evaluation")
    if not isinstance(confidence_evaluation, Mapping):
        raise _envelope_error(
            label,
            "$.request",
            "confidence thresholds do not match the report configuration",
        )
    report_default_conf = _finite_number(
        confidence_evaluation.get("default_conf"),
        label,
        "$.report.benchmark_config.confidence_evaluation.default_conf",
    )
    report_confidence_iou = _finite_number(
        confidence_evaluation.get("iou_threshold"),
        label,
        "$.report.benchmark_config.confidence_evaluation.iou_threshold",
    )
    if report_default_conf != _DEFAULT_CONF or report_confidence_iou != _CONFIDENCE_IOU:
        raise _envelope_error(
            label,
            "$.request",
            "confidence thresholds do not match the report configuration",
        )
    if (
        benchmark_config.get("device")
        != execution_context["runtime"]["resolved_device"]
    ):
        raise _envelope_error(
            label,
            "$.execution_context.runtime.resolved_device",
            "does not match the report configuration",
        )
    if _json_text(dict(metrics)) != _json_text(report_metrics):
        raise _envelope_error(
            label, "$.metrics", "does not match the validated report metrics"
        )
    return _ValidatedEnvelope(
        run_id=root["run_id"],
        process_id=root["process_id"],
        report_sha256=report_digest,
    )


def compare_benchmarks(
    first_report: str | os.PathLike[str],
    second_report: str | os.PathLike[str],
    *,
    score_atol: float = 0.0,
    metric_atol: float = 0.0,
    map_atol: float = 0.0,
) -> PersistedRepeatComparison:
    """Strictly compare reports created by two independent run processes."""

    first_path = Path(first_report).expanduser().resolve(strict=False)
    second_path = Path(second_report).expanduser().resolve(strict=False)
    same_file = first_path == second_path
    if not same_file and first_path.is_file() and second_path.is_file():
        same_file = os.path.samefile(first_path, second_path)
    if same_file:
        raise VLMConfidenceReportError(
            "benchmark comparison requires reports from two distinct run processes"
        )
    first_envelope = _load_runner_envelope(first_path, "first_run")
    second_envelope = _load_runner_envelope(second_path, "second_run")
    if first_envelope.run_id == second_envelope.run_id:
        raise VLMConfidenceReportError(
            "benchmark comparison requires distinct run identifiers"
        )
    if first_envelope.process_id == second_envelope.process_id:
        raise VLMConfidenceReportError(
            "benchmark comparison requires two fresh Python processes"
        )
    comparison = compare_confidence_reports(
        first_report,
        second_report,
        score_atol=score_atol,
        metric_atol=metric_atol,
        map_atol=map_atol,
    )
    if (
        comparison.first_report_sha256 != first_envelope.report_sha256
        or comparison.second_report_sha256 != second_envelope.report_sha256
    ):
        raise VLMConfidenceReportError(
            "a benchmark report changed while it was being compared"
        )
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = _JSONArgumentParser(
        prog="python -m libreyolo.validation.vlm_confidence_benchmark"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run_parser = subparsers.add_parser(
        "run", help="run one fresh Qwen3-VL-2B benchmark process"
    )
    run_parser.add_argument("--manifest", required=True, type=Path)
    run_parser.add_argument("--annotations", required=True, type=Path)
    run_parser.add_argument("--images-dir", required=True, type=Path)
    run_parser.add_argument("--review-attestation", required=True, type=Path)
    run_parser.add_argument("--output-root", required=True, type=Path)
    run_parser.add_argument("--seed", type=_arg_seed, default=0)
    run_parser.add_argument("--device", default="auto")

    compare_parser = subparsers.add_parser(
        "compare", help="compare two independently persisted reports"
    )
    compare_parser.add_argument("first_report", type=Path)
    compare_parser.add_argument("second_report", type=Path)
    compare_parser.add_argument("--score-atol", type=_arg_tolerance, default=0.0)
    compare_parser.add_argument("--metric-atol", type=_arg_tolerance, default=0.0)
    compare_parser.add_argument("--map-atol", type=_arg_tolerance, default=0.0)
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments without executing a benchmark."""

    return build_parser().parse_args(argv)


def _emit_status(value: Mapping[str, Any]) -> None:
    sys.stdout.write(_json_text(value))


def _error_status(*, mode: str | None, code: int, kind: str, message: str) -> int:
    _emit_status(
        {
            "schema": _STATUS_SCHEMA,
            "status": "error",
            "mode": mode,
            "code": code,
            "error": {"kind": kind, "message": message},
        }
    )
    return code


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    mode_hint = (
        arguments[0] if arguments and arguments[0] in {"run", "compare"} else None
    )
    try:
        args = parse_cli_args(arguments)
    except _CLIUsageError as exc:
        return _error_status(
            mode=mode_hint, code=EXIT_USAGE, kind="usage", message=str(exc)
        )

    try:
        if args.mode == "run":
            with redirect_stdout(sys.stderr):
                artifacts = run_benchmark(
                    args.manifest,
                    args.annotations,
                    args.images_dir,
                    args.review_attestation,
                    args.output_root,
                    seed=args.seed,
                    device=args.device,
                )
            _emit_status(
                {
                    "schema": _STATUS_SCHEMA,
                    "status": "ok",
                    "mode": "run",
                    "code": EXIT_OK,
                    "output_root": artifacts.output_dir,
                    "report": artifacts.report_path,
                    "envelope": artifacts.envelope_path,
                    "nonfinite_metrics": artifacts.nonfinite_metrics,
                }
            )
            return EXIT_OK

        with redirect_stdout(sys.stderr):
            comparison = compare_benchmarks(
                args.first_report,
                args.second_report,
                score_atol=args.score_atol,
                metric_atol=args.metric_atol,
                map_atol=args.map_atol,
            )
        comparison_payload = _json_value(comparison)
        reproducible = bool(comparison_payload["reproducible"])
        code = EXIT_OK if reproducible else EXIT_NOT_REPRODUCIBLE
        _emit_status(
            {
                "schema": _STATUS_SCHEMA,
                "status": "reproducible" if reproducible else "different",
                "mode": "compare",
                "code": code,
                "tolerances": {
                    "score_atol": args.score_atol,
                    "metric_atol": args.metric_atol,
                    "map_atol": args.map_atol,
                },
                "comparison": comparison_payload,
            }
        )
        return code
    except BenchmarkInputError as exc:
        return _error_status(
            mode=args.mode, code=EXIT_USAGE, kind="input", message=str(exc)
        )
    except BenchmarkOutputExistsError as exc:
        return _error_status(
            mode=args.mode,
            code=EXIT_OUTPUT_EXISTS,
            kind="output_exists",
            message=str(exc),
        )
    except VLMConfidenceReportError as exc:
        return _error_status(
            mode=args.mode,
            code=EXIT_INVALID_REPORT,
            kind="invalid_report",
            message=str(exc),
        )
    except Exception as exc:
        return _error_status(
            mode=args.mode,
            code=EXIT_RUN_FAILED,
            kind="execution",
            message=str(exc),
        )


if __name__ == "__main__":
    raise SystemExit(main())
