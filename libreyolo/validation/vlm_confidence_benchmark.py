"""Internal process-level runner for the Qwen VLM confidence benchmark.

Run one benchmark per process, then compare the two persisted validator reports
in a separate invocation.  This module intentionally is not exported from the
validation package and does not change the public ``LibreVLM.val()`` contract.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
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
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader

from libreyolo.data import load_data_config
from libreyolo.data.dataset import COCODataset
from libreyolo.models.vlm.qwen3vl import LibreQwen3VL

from .coco_evaluator import COCOEvaluator
from .config import ValidationConfig
from .detection_validator import val_collate_fn
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
_MODEL_SIZES = {"2b", "4b"}
_IMAGE_SIZE = 1024
_DEFAULT_CONF = 0.25
_CONFIDENCE_IOU = 0.5
_REPORT_NAME = "vlm_confidence_report.json"
_ENVELOPE_NAME = "vlm_confidence_run.json"
_RUN_SCHEMA = "libreyolo.vlm-confidence-benchmark-run.v3"
_STATUS_SCHEMA = "libreyolo.vlm-confidence-benchmark-status.v1"
_CONTEXT_SCHEMA = "libreyolo.vlm-confidence-benchmark-context.v3"
_PREFLIGHT_SCHEMA = "libreyolo.vlm-confidence-benchmark-preflight.v2"
_CHECKPOINT_CONTEXT_SCHEMA = "libreyolo.vlm-confidence-checkpoint-identity.v1"
_CHECKPOINT_IDENTITY_SCHEMA = "libreyolo.vlm-checkpoint-identity.v1"
_DATASET_CONTEXT_SCHEMA = "libreyolo.vlm-confidence-benchmark-dataset.v1"
_DATASET_MANIFEST_SCHEMA = "libreyolo.vlm-benchmark-dataset.v1"
_REVIEW_SCHEMA = "libreyolo.vlm-benchmark-dataset-review.v1"
_BASE_PARTITION_ROLE = "zero_shot_confidence_promotion"
_CHECKPOINT_PARTITION_ROLE = "fine_tune_validation"
_PARTITION_REQUIREMENTS = {
    _BASE_PARTITION_ROLE: {
        "name": "promotion500",
        "role": _BASE_PARTITION_ROLE,
        "start": 0,
        "stop": 500,
        "image_count": 500,
        "annotation_artifact": "annotations/instances_val2017_promotion500.json",
    },
    _CHECKPOINT_PARTITION_ROLE: {
        "name": "holdout100",
        "role": _CHECKPOINT_PARTITION_ROLE,
        "start": 0,
        "stop": 100,
        "image_count": 100,
        "annotation_artifact": "annotations/instances_val2017_holdout100.json",
    },
}
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
_CHECKPOINT_TEMP_PREFIX = "libreyolo-vlm-confidence-checkpoint-"
_BASE_SNAPSHOT_TEMP_PREFIX = "libreyolo-vlm-confidence-base-"
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}
_QWEN_2B_REPO = "Qwen/Qwen3-VL-2B-Instruct"
_QWEN_2B_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
# Audited 2026-08-16 from the official revision API and its /resolve/... blobs:
# https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct/revision/89644892e4d85e24eaac8bacfd4f463576704203
# https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203
_QWEN_2B_SNAPSHOT_IDENTITY = {
    "kind": "pinned_hf_snapshot",
    "schema": "libreyolo.vlm-hf-snapshot-identity.v1",
    "source": _QWEN_2B_REPO,
    "revision": _QWEN_2B_REVISION,
    "format": "safetensors_single",
    "artifacts": [
        {
            "path": "config.json",
            "size_bytes": 1_505,
            "sha256": "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
        },
        {
            "path": "model.safetensors",
            "size_bytes": 4_255_140_312,
            "sha256": "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
        },
    ],
    "sha256": "ed2f80a94ea529acc7c3192ca7d1c4a8cfc28f69de472417e0402a59fdb6cd07",
    "files": 2,
    "size_bytes": 4_255_141_817,
    "weight_files": ["model.safetensors"],
}
_QWEN_2B_PROCESSOR_CONTENT_IDENTITY = {
    "source": _QWEN_2B_REPO,
    "revision": _QWEN_2B_REVISION,
    "sha256": "f6626ce88ba637238391c175ea0b8f57a58f7bfcbe8b9876ceaded603185826d",
    "files": 9,
}
_QWEN_4B_REPO = "Qwen/Qwen3-VL-4B-Instruct"
_QWEN_4B_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
_QWEN_4B_SNAPSHOT_IDENTITY = {
    "kind": "pinned_hf_snapshot",
    "schema": "libreyolo.vlm-hf-snapshot-identity.v1",
    "source": _QWEN_4B_REPO,
    "revision": _QWEN_4B_REVISION,
    "format": "safetensors_sharded",
    "artifacts": [
        {
            "path": "config.json",
            "size_bytes": 1_505,
            "sha256": "edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d",
        },
        {
            "path": "model-00001-of-00002.safetensors",
            "size_bytes": 4_967_229_296,
            "sha256": "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
        },
        {
            "path": "model-00002-of-00002.safetensors",
            "size_bytes": 3_908_490_048,
            "sha256": "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
        },
        {
            "path": "model.safetensors.index.json",
            "size_bytes": 64_742,
            "sha256": "58a7841d7bff2548dd91577d216274a83cf1b500bc6a534b809d6c1b1707cf2b",
        },
    ],
    "sha256": "e03ebe7bd6863419b7e823e83480310bc1b3b9ac6a23444e495acefcbc592efe",
    "files": 4,
    "size_bytes": 8_875_785_591,
    "weight_files": [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ],
}
_QWEN_4B_PROCESSOR_CONTENT_IDENTITY = {
    "source": _QWEN_4B_REPO,
    "revision": _QWEN_4B_REVISION,
    "sha256": "d74426dfb45d795a6f1e1967c5790df088d009b5c35e0cb7846ea7247e3d9f30",
    "files": 10,
}
_QWEN_BASE_PINS = {
    "2b": (_QWEN_2B_REPO, _QWEN_2B_REVISION),
    "4b": (_QWEN_4B_REPO, _QWEN_4B_REVISION),
}
_QWEN_SNAPSHOT_IDENTITIES = {
    "2b": _QWEN_2B_SNAPSHOT_IDENTITY,
    "4b": _QWEN_4B_SNAPSHOT_IDENTITY,
}
_QWEN_PROCESSOR_CONTENT_IDENTITIES = {
    "2b": _QWEN_2B_PROCESSOR_CONTENT_IDENTITY,
    "4b": _QWEN_4B_PROCESSOR_CONTENT_IDENTITY,
}
_CHECKPOINT_FILE_ROLES = {
    "checkpoint_contract",
    "adapter_config",
    "adapter_weights",
    "processor",
}

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
class BenchmarkPreflight:
    """Fully checked, process-local readiness evidence for one benchmark run."""

    output_dir: Path
    model_size: str
    checkpoint_root: Path | None
    checkpoint_identity: dict[str, Any] | None
    snapshot_root: Path
    snapshot_identity: dict[str, Any]
    processor_content_identity: dict[str, Any]
    dataset_context: dict[str, Any]
    git_context: dict[str, Any]
    determinism: dict[str, Any]
    runtime_context: dict[str, Any]
    offline_context: dict[str, Any]


@dataclass(frozen=True)
class _PreModelInputs:
    destination: Path
    model_size: str
    checkpoint_request_path: Path | None
    checkpoint_source_root: Path | None
    checkpoint_source_identity: Any | None
    checkpoint_load_root: Path | None
    checkpoint_load_identity: Any | None
    checkpoint_context: dict[str, Any] | None
    requested_device: str
    resolved_device: str
    verified: VerifiedBenchmarkRunInputs
    dataset_yaml: Path
    portable_dataset_context: dict[str, Any]
    native_dataset_context: dict[str, Any]
    git_context: dict[str, Any]
    determinism: dict[str, Any]
    runtime_context: dict[str, Any]
    device_probe: dict[str, Any]
    offline_context: dict[str, Any]
    snapshot_root: Path
    snapshot_load_root: Path | None
    snapshot_identity: dict[str, Any]
    processor_content_identity: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkRunIdentity:
    """Strict path-free identity read from one report and its sibling envelope."""

    run_id: str
    process_id: str
    report_sha256: str
    envelope_sha256: str
    execution_context: dict[str, Any]
    benchmark_config: dict[str, Any]
    metrics: dict[str, Any]
    nonfinite_metrics: tuple[str, ...]


@dataclass(frozen=True)
class _ValidatedEnvelope(BenchmarkRunIdentity):
    """Internal validated envelope result used by writer and public reader."""


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


def _validate_device_argument(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkInputError("device must be a non-empty string")
    requested = value.strip()
    if requested == "auto":
        return requested
    candidate = f"cuda:{int(requested)}" if requested.isdigit() else requested
    try:
        torch.device(candidate)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise BenchmarkInputError(f"invalid benchmark device: {requested!r}") from exc
    return requested


def _validate_output_destination(output_root: str | os.PathLike[str]) -> Path:
    requested = Path(output_root).expanduser()
    if _path_exists(requested):
        raise BenchmarkOutputExistsError(
            f"benchmark output already exists: {requested}"
        )
    destination = Path(os.path.abspath(os.fspath(requested)))
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

    ancestor = destination.parent
    while not _path_exists(ancestor):
        parent = ancestor.parent
        if parent == ancestor:
            raise BenchmarkInputError(
                f"benchmark output has no existing directory ancestor: {destination}"
            )
        ancestor = parent
    try:
        lexical_ancestor, resolved_ancestor = (
            VLMConfidenceValidator._strict_local_directory_root(
                ancestor, "Benchmark output ancestor"
            )
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise BenchmarkInputError(f"invalid benchmark output parent: {exc}") from exc

    missing_parent_parts = destination.parent.relative_to(lexical_ancestor).parts
    try:
        with tempfile.TemporaryDirectory(
            dir=resolved_ancestor, prefix=".libreyolo-vlm-output-probe-"
        ) as temporary_root:
            simulated_parent = Path(temporary_root).joinpath(*missing_parent_parts)
            if missing_parent_parts:
                simulated_parent.mkdir(parents=True, exist_ok=False)
            simulated_destination = simulated_parent / destination.name
            lock = simulated_parent / f".{destination.name}.lock"
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            stage = Path(tempfile.mkdtemp(dir=simulated_parent, prefix=".stage-"))
            stage.rename(simulated_destination)
    except (OSError, ValueError) as exc:
        raise BenchmarkInputError(
            f"benchmark output cannot be staged at the requested location: {exc}"
        ) from exc
    return destination


def _inspect_checkpoint_identity(checkpoint_dir: str | os.PathLike[str]) -> Any:
    """Return the strict local identity without importing it on base-only runs."""

    try:
        from libreyolo.models.vlm.training.checkpoint import (
            inspect_vlm_checkpoint_identity,
        )
    except ImportError as exc:
        raise BenchmarkInputError(
            "the installed LibreYOLO runtime cannot inspect VLM checkpoint identities"
        ) from exc
    try:
        return inspect_vlm_checkpoint_identity(checkpoint_dir)
    except (OSError, TypeError, ValueError) as exc:
        raise BenchmarkInputError(f"invalid local VLM checkpoint: {exc}") from exc


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_processor_sha256(files: Sequence[Mapping[str, Any]]) -> str:
    processor_files = [
        {
            "path": entry["path"],
            "size": entry["size"],
            "sha256": entry["sha256"],
        }
        for entry in files
        if entry["role"] == "processor"
    ]
    return _canonical_json_sha256(processor_files)


def _checkpoint_aggregate_sha256(
    identity: Mapping[str, Any], files: Sequence[Mapping[str, Any]]
) -> str:
    payload = {
        "schema": _CHECKPOINT_IDENTITY_SCHEMA,
        "family": identity["family"],
        "size": identity["size"],
        "task": identity["task"],
        "base_repo": identity["base_repo"],
        "base_revision": identity["base_revision"],
        "files": list(files),
        "adapter_weights_sha256": identity["adapter_weights_sha256"],
        "adapter_config_sha256": identity["adapter_config_sha256"],
        "checkpoint_contract_sha256": identity["checkpoint_contract_sha256"],
        "processor_sha256": identity["processor_sha256"],
    }
    return _canonical_json_sha256(payload)


def _checkpoint_context(identity: Any) -> dict[str, Any]:
    """Build the portable, path-free form persisted in reports and envelopes."""

    scalar_fields = (
        "family",
        "size",
        "task",
        "base_repo",
        "base_revision",
        "aggregate_sha256",
        "adapter_weights_sha256",
        "adapter_config_sha256",
        "checkpoint_contract_sha256",
        "processor_sha256",
    )
    values = {field: getattr(identity, field, None) for field in scalar_fields}
    if values["family"] != _MODEL_FAMILY:
        raise BenchmarkInputError(
            "the confidence benchmark checkpoint must belong to qwen3vl"
        )
    if values["size"] not in _MODEL_SIZES:
        raise BenchmarkInputError(
            "the confidence benchmark checkpoint size must be '2b' or '4b'"
        )
    if values["task"] != "detect":
        raise BenchmarkInputError(
            "the confidence benchmark checkpoint task must be 'detect'"
        )
    expected_repo, expected_revision = _QWEN_BASE_PINS[values["size"]]
    if (
        values["base_repo"] != expected_repo
        or values["base_revision"] != expected_revision
    ):
        raise BenchmarkInputError(
            "the checkpoint contract does not bind the official pinned Qwen3-VL base"
        )
    for field in (
        "aggregate_sha256",
        "adapter_weights_sha256",
        "adapter_config_sha256",
        "checkpoint_contract_sha256",
        "processor_sha256",
    ):
        if not isinstance(values[field], str) or not _HEX_DIGEST.fullmatch(
            values[field]
        ):
            raise BenchmarkInputError(
                f"the checkpoint identity has an invalid {field} digest"
            )

    raw_files = getattr(identity, "files", None)
    if not isinstance(raw_files, tuple) or not raw_files:
        raise BenchmarkInputError(
            "the checkpoint identity must contain a non-empty frozen file inventory"
        )
    files = []
    for entry in raw_files:
        path = getattr(entry, "path", None)
        role = getattr(entry, "role", None)
        size = getattr(entry, "size", None)
        digest = getattr(entry, "sha256", None)
        if (
            not isinstance(path, str)
            or not path
            or Path(path).name != path
            or "/" in path
            or "\\" in path
            or any(ord(character) < 32 for character in path)
        ):
            raise BenchmarkInputError(
                "the checkpoint identity contains an unsafe file basename"
            )
        if role not in _CHECKPOINT_FILE_ROLES:
            raise BenchmarkInputError(
                "the checkpoint identity contains an unsupported file role"
            )
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= _MAX_SAFE_INTEGER
        ):
            raise BenchmarkInputError(
                "the checkpoint identity contains an invalid file size"
            )
        if not isinstance(digest, str) or not _HEX_DIGEST.fullmatch(digest):
            raise BenchmarkInputError(
                "the checkpoint identity contains an invalid file digest"
            )
        files.append({"path": path, "role": role, "size": size, "sha256": digest})
    paths = [entry["path"] for entry in files]
    if paths != sorted(paths, key=str.casefold) or len(
        {path.casefold() for path in paths}
    ) != len(paths):
        raise BenchmarkInputError(
            "the checkpoint identity file inventory must be sorted and unique"
        )
    roles = [entry["role"] for entry in files]
    if (
        any(
            roles.count(role) != 1
            for role in ("checkpoint_contract", "adapter_config", "adapter_weights")
        )
        or roles.count("processor") < 1
    ):
        raise BenchmarkInputError(
            "the checkpoint identity has an incomplete or ambiguous role inventory"
        )
    adapter_file = next(entry for entry in files if entry["role"] == "adapter_weights")
    if adapter_file["sha256"] != values["adapter_weights_sha256"]:
        raise BenchmarkInputError(
            "the checkpoint adapter weights digest does not match its file record"
        )
    if _checkpoint_processor_sha256(files) != values["processor_sha256"]:
        raise BenchmarkInputError(
            "the checkpoint processor digest does not match its processor file records"
        )
    if _checkpoint_aggregate_sha256(values, files) != values["aggregate_sha256"]:
        raise BenchmarkInputError(
            "the checkpoint aggregate digest does not match its identity payload"
        )

    return {
        "schema": _CHECKPOINT_CONTEXT_SCHEMA,
        "kind": "qwen3vl_lora_checkpoint",
        **values,
        "files": files,
    }


def _prepare_checkpoint(
    checkpoint_dir: str | os.PathLike[str] | None,
    destination: Path,
) -> tuple[Path | None, Path | None, Any | None, dict[str, Any] | None]:
    if checkpoint_dir is None:
        return None, None, None, None
    if isinstance(checkpoint_dir, bool) or not isinstance(
        checkpoint_dir, (str, os.PathLike)
    ):
        raise BenchmarkInputError("checkpoint_dir must be a filesystem path")
    try:
        checkpoint_request_path = Path(
            os.path.abspath(Path(checkpoint_dir).expanduser())
        )
    except (OSError, TypeError, ValueError) as exc:
        raise BenchmarkInputError(
            "checkpoint_dir must be a valid filesystem path"
        ) from exc
    identity = _inspect_checkpoint_identity(checkpoint_dir)
    root = getattr(identity, "root", None)
    if not isinstance(root, Path) or not root.is_absolute():
        raise BenchmarkInputError(
            "the checkpoint identity must expose an absolute local root"
        )
    try:
        resolved_root = root.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise BenchmarkInputError(
            f"the checkpoint identity root is unavailable: {root}"
        ) from exc
    if resolved_root != root or not root.is_dir():
        raise BenchmarkInputError(
            "the checkpoint identity root must be a canonical local directory"
        )
    if (
        destination == root
        or root in destination.parents
        or destination in root.parents
    ):
        raise BenchmarkInputError(
            "benchmark output and checkpoint directories must not overlap"
        )
    return checkpoint_request_path, root, identity, _checkpoint_context(identity)


@contextmanager
def _isolated_checkpoint_snapshot(
    source_root: Path | None,
    source_identity: Any | None,
    source_context: dict[str, Any] | None,
) -> Iterator[tuple[Path | None, Any | None]]:
    """Hold a stable private copy for the complete model-loading lifetime."""

    if source_root is None or source_identity is None or source_context is None:
        if any(
            value is not None
            for value in (source_root, source_identity, source_context)
        ):
            raise RuntimeError("checkpoint isolation state is internally inconsistent")
        yield None, None
        return

    try:
        from libreyolo.models.vlm import artifact as artifact_module
    except ImportError as exc:
        raise BenchmarkInputError(
            "the installed LibreYOLO runtime cannot isolate a VLM checkpoint"
        ) from exc

    try:
        temporary_checkpoint = tempfile.TemporaryDirectory(
            prefix=_CHECKPOINT_TEMP_PREFIX
        )
    except OSError as exc:
        raise BenchmarkInputError(
            f"could not create a stable isolated checkpoint copy: {exc}"
        ) from exc

    body_completed = False
    try:
        try:
            isolated_root = Path(temporary_checkpoint.name).resolve() / "checkpoint"
            isolated_root.mkdir(mode=0o700)
            for record in source_identity.files:
                artifact_module._copy_file_stable(
                    source_root / record.path,
                    isolated_root / record.path,
                )

            isolated_identity = _inspect_checkpoint_identity(isolated_root)
            if getattr(isolated_identity, "root", None) != isolated_root:
                raise BenchmarkInputError(
                    "the isolated checkpoint inspector returned a different root"
                )
            isolated_context = _checkpoint_context(isolated_identity)
            if isolated_context != source_context:
                raise BenchmarkInputError(
                    "the isolated checkpoint identity does not match the requested "
                    "checkpoint"
                )
            _require_checkpoint_stable(
                source_root,
                source_identity,
                phase="while creating its isolated copy",
                input_error=True,
            )
        except BenchmarkInputError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise BenchmarkInputError(
                f"could not create a stable isolated checkpoint copy: {exc}"
            ) from exc
        yield isolated_root, isolated_identity
        body_completed = True
    finally:
        try:
            temporary_checkpoint.cleanup()
        except OSError as exc:
            if body_completed:
                raise BenchmarkInputError(
                    f"could not remove the isolated checkpoint copy: {exc}"
                ) from exc


def _require_checkpoint_stable(
    root: Path | None,
    expected: Any | None,
    *,
    phase: str,
    input_error: bool,
) -> None:
    if root is None or expected is None:
        if root is not None or expected is not None:
            raise RuntimeError("checkpoint readiness state is internally inconsistent")
        return
    try:
        actual = _inspect_checkpoint_identity(root)
    except BenchmarkInputError:
        if input_error:
            raise
        raise RuntimeError(f"VLM checkpoint became invalid {phase}") from None
    if actual != expected:
        error = BenchmarkInputError if input_error else RuntimeError
        raise error(f"VLM checkpoint identity changed {phase}")


def _configure_offline_environment() -> dict[str, Any]:
    for name, value in _OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value
    try:
        hub_constants = import_module("huggingface_hub.constants")
        transformers_hub = import_module("transformers.utils.hub")
    except ImportError as exc:
        raise BenchmarkInputError(
            "cannot import the required Hugging Face offline-mode runtime"
        ) from exc
    hub_offline = getattr(hub_constants, "HF_HUB_OFFLINE", None)
    telemetry_disabled = getattr(hub_constants, "HF_HUB_DISABLE_TELEMETRY", None)
    transformers_offline = getattr(transformers_hub, "is_offline_mode", None)
    transformers_offline = (
        transformers_offline() if callable(transformers_offline) else None
    )
    if (
        hub_offline is not True
        or telemetry_disabled is not True
        or transformers_offline is not True
    ):
        raise BenchmarkInputError(
            "Hugging Face offline mode and telemetry suppression must be enabled "
            "before importing huggingface_hub or transformers"
        )
    return {
        "hf_hub_offline": os.environ["HF_HUB_OFFLINE"],
        "transformers_offline": os.environ["TRANSFORMERS_OFFLINE"],
        "hf_hub_disable_telemetry": os.environ["HF_HUB_DISABLE_TELEMETRY"],
        "do_not_track": os.environ["DO_NOT_TRACK"],
        "hub_runtime_offline": hub_offline,
        "transformers_runtime_offline": transformers_offline,
        "hub_runtime_telemetry_disabled": telemetry_disabled,
    }


def _reject_faster_coco_override() -> None:
    value = os.environ.get(_FASTER_COCO_ENV)
    if value is not None and value.strip().lower() in {"1", "true", "yes", "on"}:
        raise BenchmarkInputError(
            f"{_FASTER_COCO_ENV} must not enable faster-coco-eval for this benchmark"
        )


def _required_package_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "transformers",
        "huggingface_hub",
        "tokenizers",
        "safetensors",
        "pycocotools",
    ):
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise BenchmarkInputError(
                f"cannot identify required benchmark package {package!r}"
            ) from exc
        if not isinstance(version, str) or not version.strip():
            raise BenchmarkInputError(
                f"cannot identify required benchmark package {package!r}"
            )
        versions[package] = version
    return versions


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _resolve_and_probe_device(requested_device: str) -> tuple[str, dict[str, Any]]:
    """Resolve BaseModel device syntax and exercise the selected device locally."""

    requested_device = _validate_device_argument(requested_device)
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        elif _mps_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif requested_device.isdigit():
        device = torch.device("cuda", int(requested_device))
    else:
        device = torch.device(requested_device)

    if device.type not in {"cpu", "cuda", "mps"}:
        raise BenchmarkInputError(
            "the VLM confidence benchmark supports only cpu, cuda, and mps devices"
        )

    index: int | None = device.index
    name: str | None = None
    capability: list[int] | None = None
    total_memory: int | None = None
    free_memory: int | None = None
    bf16_supported: bool | None = None
    driver: str | None = None
    try:
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise BenchmarkInputError(
                    f"requested CUDA device is unavailable: {requested_device!r}"
                )
            count = int(torch.cuda.device_count())
            if index is None:
                index = int(torch.cuda.current_device())
            if index < 0 or index >= count:
                raise BenchmarkInputError(
                    f"requested CUDA device index {index} is outside the available "
                    f"range 0..{max(count - 1, 0)}"
                )
            device = torch.device("cuda", index)
            properties = torch.cuda.get_device_properties(index)
            name = str(properties.name)
            capability = [
                int(value) for value in torch.cuda.get_device_capability(index)
            ]
            total_memory = int(properties.total_memory)
            free_memory, reported_total = (
                int(value) for value in torch.cuda.mem_get_info(index)
            )
            if reported_total != total_memory:
                total_memory = reported_total
            with torch.cuda.device(index):
                bf16_supported = bool(torch.cuda.is_bf16_supported())
                probe = torch.ones(1, device=device, dtype=torch.float32)
                if float((probe + 1.0).item()) != 2.0:
                    raise RuntimeError("CUDA tensor probe returned an invalid result")
            torch.cuda.synchronize(index)
            driver = _nvidia_driver_version()
        elif device.type == "mps":
            if not _mps_available():
                raise BenchmarkInputError("requested MPS device is unavailable")
            probe = torch.ones(1, device=device, dtype=torch.float32)
            if float((probe + 1.0).item()) != 2.0:
                raise RuntimeError("MPS tensor probe returned an invalid result")
            mps_module = getattr(torch, "mps", None)
            synchronize = getattr(mps_module, "synchronize", None)
            if callable(synchronize):
                synchronize()
            try:
                bf16_probe = torch.ones(1, device=device, dtype=torch.bfloat16)
                bf16_supported = bool(float((bf16_probe + 1.0).float().item()) == 2.0)
            except (RuntimeError, TypeError):
                bf16_supported = False
            name = platform.processor() or platform.machine()
        else:
            device = torch.device("cpu")
            probe = torch.ones(1, device=device, dtype=torch.float32)
            if float((probe + 1.0).item()) != 2.0:
                raise RuntimeError("CPU tensor probe returned an invalid result")
            name = platform.processor() or platform.machine()
    except BenchmarkInputError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise BenchmarkInputError(
            f"could not execute a probe on benchmark device {device}: {exc}"
        ) from exc

    resolved = str(device)
    return resolved, {
        "requested_device": requested_device,
        "resolved_device": resolved,
        "type": device.type,
        "index": index,
        "name": name,
        "capability": capability,
        "total_memory_bytes": total_memory,
        "free_memory_bytes": free_memory,
        "bf16_supported": bf16_supported,
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "nvidia_driver": driver,
        "tiny_tensor_probe": "ok",
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


def _runtime_context(
    *,
    requested_device: str,
    resolved_device: str,
    package_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    cudnn_version = torch.backends.cudnn.version()
    package_versions = (
        _required_package_versions()
        if package_versions is None
        else dict(package_versions)
    )
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
            context,
            "verified_inputs",
            "$.execution_context.dataset",
            required_role=verified.partition_role,
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


class _RecordingValPreprocessor:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.input_dimensions: list[tuple[int, int]] = []

    @property
    def wants_unresized_image(self) -> bool:
        return bool(getattr(self.delegate, "wants_unresized_image", False))

    def __call__(self, image: np.ndarray, targets: np.ndarray, input_size: Any):
        self.input_dimensions.append((int(image.shape[0]), int(image.shape[1])))
        return self.delegate(image, targets, input_size)


def _normalized_native_annotation(row: Mapping[str, Any]) -> dict[str, Any]:
    bbox = row.get("bbox")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes, bytearray)):
        raise ValueError("COCO annotation bbox must be a sequence")
    if len(bbox) != 4:
        raise ValueError("COCO annotation bbox must contain four values")
    normalized_bbox = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in normalized_bbox):
        raise ValueError("COCO annotation bbox values must be finite")
    area = float(row.get("area", normalized_bbox[2] * normalized_bbox[3]))
    if not math.isfinite(area):
        raise ValueError("COCO annotation area must be finite")
    return {
        "id": int(row["id"]),
        "image_id": int(row["image_id"]),
        "category_id": int(row["category_id"]),
        "bbox": normalized_bbox,
        "area": area,
        "iscrowd": int(row.get("iscrowd", 0)),
        "ignore": int(row.get("ignore", 0)),
    }


def _exercise_native_evaluator(
    dataset: COCODataset,
    annotations: Sequence[Mapping[str, Any]],
    expected_category_map: Mapping[int, int],
) -> str:
    """Run one real, deterministic pycocotools evaluation against verified GT."""

    category_to_label = {
        int(category_id): int(label)
        for label, category_id in expected_category_map.items()
    }
    candidate = next(
        (
            row
            for row in annotations
            if int(row["iscrowd"]) == 0
            and int(row["ignore"]) == 0
            and float(row["bbox"][2]) > 0.0
            and float(row["bbox"][3]) > 0.0
            and int(row["category_id"]) in category_to_label
        ),
        None,
    )
    if candidate is None:
        raise ValueError(
            "native COCO evaluator self-test has no usable ground-truth annotation"
        )

    x, y, width, height = (float(value) for value in candidate["bbox"])
    probe = COCOEvaluator(
        dataset.coco,
        iou_type="bbox",
        label_to_category_id=expected_category_map,
        faster_coco_eval=False,
    )
    probe.update(
        {
            "boxes": np.asarray([[x, y, x + width, y + height]], dtype=np.float32),
            "scores": np.asarray([1.0], dtype=np.float32),
            "classes": np.asarray(
                [category_to_label[int(candidate["category_id"])]], dtype=np.int64
            ),
        },
        int(candidate["image_id"]),
    )
    try:
        metrics = probe.compute()
    except Exception as exc:
        raise ValueError(f"native COCO evaluator self-test failed: {exc}") from exc
    if not isinstance(metrics, Mapping):
        raise ValueError("native COCO evaluator self-test returned no metric mapping")
    required_metrics = {"mAP", "mAP50", "mAP75", "map_5095", "ar_100"}
    if not required_metrics.issubset(metrics):
        raise ValueError("native COCO evaluator self-test omitted required metrics")
    try:
        normalized_metrics = {
            str(name): float(value) for name, value in metrics.items()
        }
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "native COCO evaluator self-test returned a non-numeric metric"
        ) from exc
    if not all(math.isfinite(value) for value in normalized_metrics.values()):
        raise ValueError("native COCO evaluator self-test returned a non-finite metric")
    map50 = normalized_metrics["mAP50"]
    if not 0.0 < map50 <= 1.0 + 1e-12:
        raise ValueError(
            "native COCO evaluator self-test did not match its perfect prediction"
        )
    if not isinstance(probe.last_backend, str) or not probe.last_backend.startswith(
        "pycocotools "
    ):
        raise ValueError(
            "native COCO evaluator self-test did not execute the pycocotools backend"
        )
    return probe.last_backend


def _preflight_native_dataset(
    verified: VerifiedBenchmarkRunInputs,
) -> dict[str, Any]:
    """Exercise the exact native dataset/evaluator path without loading a model."""

    try:
        expected_images = list(verified.expected_images)
        expected_count = verified.partition_stop - verified.partition_start
        if expected_count <= 0 or len(expected_images) != expected_count:
            raise ValueError(
                "verified image rows do not match the benchmark partition bounds"
            )
        if len(verified.class_names) != _REQUIRED_CLASS_COUNT:
            raise ValueError("verified class count is not the fixed 80-class contract")

        preprocessor_class = LibreQwen3VL.val_preprocessor_class
        preprocessor = _RecordingValPreprocessor(
            preprocessor_class(img_size=(_IMAGE_SIZE, _IMAGE_SIZE))
        )
        dataset = COCODataset(
            data_dir=str(verified.images_dir),
            json_file=str(verified.annotation_path),
            name=str(verified.images_dir),
            img_size=(_IMAGE_SIZE, _IMAGE_SIZE),
            preproc=preprocessor,
            num_classes=len(verified.class_names),
            names=list(verified.class_names),
        )

        expected_ids = [int(row["image_id"]) for row in expected_images]
        actual_ids = [int(value) for value in dataset.ids]
        if actual_ids != expected_ids:
            raise ValueError("native COCO dataset image order changed")
        if tuple(dataset._classes) != tuple(verified.class_names):
            raise ValueError("native COCO dataset class names changed")

        expected_categories = [
            {"id": int(row["id"]), "name": str(row["name"])}
            for row in verified.expected_categories
        ]
        actual_categories = [
            {"id": int(row["id"]), "name": str(row["name"])} for row in dataset.cats
        ]
        if actual_categories != expected_categories:
            raise ValueError("native COCO category rows changed")
        expected_category_map = {
            index: int(category["id"])
            for index, category in enumerate(expected_categories)
        }
        if dict(dataset.label_to_category_id) != expected_category_map:
            raise ValueError("native COCO class-to-category mapping changed")

        annotation_rows = dataset.coco.dataset.get("annotations")
        if not isinstance(annotation_rows, list):
            raise ValueError("native COCO ground truth has no annotation rows")
        actual_annotations = [
            _normalized_native_annotation(row) for row in annotation_rows
        ]
        expected_annotations = [
            _normalized_native_annotation(row) for row in verified.expected_annotations
        ]
        if actual_annotations != expected_annotations:
            raise ValueError("native COCO ground-truth annotation rows changed")

        image_rows = dataset.coco.dataset.get("images")
        if not isinstance(image_rows, list):
            raise ValueError("native COCO ground truth has no image rows")
        actual_image_rows = [
            {
                "image_id": int(row["id"]),
                "file_name": str(row["file_name"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
            for row in image_rows
        ]
        expected_image_rows = [
            {
                "image_id": int(row["image_id"]),
                "file_name": str(row["file_name"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
            for row in expected_images
        ]
        if actual_image_rows != expected_image_rows:
            raise ValueError("native COCO ground-truth image rows changed")

        image_root = verified.images_dir.resolve(strict=True)
        for index, row in enumerate(expected_images):
            candidate = dataset._image_path(index)
            if candidate.is_symlink():
                raise ValueError(f"native COCO image path is a symlink: {candidate}")
            resolved = candidate.resolve(strict=True)
            expected = (image_root / str(row["file_name"])).resolve(strict=True)
            if resolved != expected or resolved.parent != image_root:
                raise ValueError(
                    f"native COCO image path changed for image {row['image_id']}"
                )

        evaluator = COCOEvaluator(
            dataset.coco,
            iou_type="bbox",
            label_to_category_id=dataset.label_to_category_id,
            faster_coco_eval=False,
        )
        if evaluator.faster_coco_eval is not False:
            raise ValueError("native COCO evaluator selected faster-coco-eval")
        if evaluator.label_to_category_id != expected_category_map:
            raise ValueError("native COCO evaluator category mapping changed")
        evaluator_self_test_backend = _exercise_native_evaluator(
            dataset, actual_annotations, expected_category_map
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=val_collate_fn,
            drop_last=False,
        )
        observed_ids = []
        for index, batch in enumerate(dataloader):
            if index >= expected_count:
                raise ValueError("native COCO dataloader produced extra images")
            images, targets, image_info, image_ids = batch
            if tuple(images.shape) != (1, 3, _IMAGE_SIZE, _IMAGE_SIZE):
                raise ValueError(
                    "native COCO dataloader produced an unexpected image tensor shape"
                )
            if int(targets.shape[0]) != 1:
                raise ValueError(
                    "native COCO dataloader violated the batch-size-one contract"
                )
            raw_id = image_ids[0]
            if isinstance(raw_id, torch.Tensor):
                if raw_id.numel() != 1:
                    raise ValueError("native COCO image id is not scalar")
                raw_id = raw_id.item()
            image_id = int(raw_id)
            row = expected_images[index]
            if image_id != int(row["image_id"]):
                raise ValueError("native COCO dataloader image order changed")
            info = tuple(int(value) for value in image_info[0])
            expected_info = (int(row["height"]), int(row["width"]))
            if info != expected_info:
                raise ValueError(
                    f"native COCO decoded dimensions changed for image {image_id}"
                )
            observed_ids.append(image_id)

        if observed_ids != expected_ids or len(preprocessor.input_dimensions) != len(
            expected_ids
        ):
            raise ValueError(
                "native COCO dataloader did not iterate the full partition"
            )
        for row, observed in zip(expected_images, preprocessor.input_dimensions):
            raw_height, raw_width = int(row["height"]), int(row["width"])
            if preprocessor.wants_unresized_image:
                expected_input = (raw_height, raw_width)
            else:
                ratio = min(_IMAGE_SIZE / raw_height, _IMAGE_SIZE / raw_width)
                expected_input = (int(raw_height * ratio), int(raw_width * ratio))
            if observed != expected_input:
                raise ValueError(
                    "native COCO preprocessor input dimensions changed for image "
                    f"{row['image_id']}"
                )

        order_payload = [
            {
                "image_id": int(row["image_id"]),
                "file_name": str(row["file_name"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
            for row in expected_images
        ]
        return {
            "dataset_class": (
                f"{type(dataset).__module__}.{type(dataset).__qualname__}"
            ),
            "preprocessor_class": (
                f"{preprocessor_class.__module__}.{preprocessor_class.__qualname__}"
            ),
            "evaluator_class": (
                f"{type(evaluator).__module__}.{type(evaluator).__qualname__}"
            ),
            "batch_size": 1,
            "num_workers": 0,
            "faster_coco_eval": False,
            "evaluator_self_test": "passed",
            "evaluator_self_test_backend": evaluator_self_test_backend,
            "image_count": len(observed_ids),
            "category_count": len(actual_categories),
            "annotation_count": len(actual_annotations),
            "image_order_sha256": hashlib.sha256(
                _json_text(order_payload).encode("utf-8")
            ).hexdigest(),
            "ground_truth_sha256": hashlib.sha256(
                _json_text(
                    {
                        "images": actual_image_rows,
                        "categories": actual_categories,
                        "annotations": actual_annotations,
                    }
                ).encode("utf-8")
            ).hexdigest(),
        }
    except BenchmarkInputError:
        raise
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise BenchmarkInputError(
            f"native benchmark dataset preflight failed: {exc}"
        ) from exc


_VERIFIED_INPUT_FIELDS = (
    "manifest_path",
    "manifest_sha256",
    "source_annotations",
    "source_canonical_sha256",
    "source_file_sha256",
    "source_file_size_bytes",
    "images_dir",
    "selected_image_identity_sha256",
    "partition_name",
    "partition_role",
    "partition_start",
    "partition_stop",
    "annotation_path",
    "annotation_sha256",
    "annotation_size_bytes",
    "class_names",
    "expected_images",
    "expected_categories",
    "expected_annotations",
    "review_attestation_path",
    "review_attestation_sha256",
    "review_attestation",
)


def _verified_inputs_identity(verified: VerifiedBenchmarkRunInputs) -> str:
    return _json_text(
        {name: getattr(verified, name) for name in _VERIFIED_INPUT_FIELDS}
    )


def _verify_run_inputs(
    manifest: str | os.PathLike[str],
    source_annotations: str | os.PathLike[str],
    images_dir: str | os.PathLike[str],
    review_attestation: str | os.PathLike[str],
    *,
    required_role: str,
) -> VerifiedBenchmarkRunInputs:
    if required_role not in _PARTITION_REQUIREMENTS:
        raise RuntimeError("benchmark partition selection is internally inconsistent")
    try:
        return verify_benchmark_run_inputs(
            manifest,
            source_annotations,
            images_dir,
            review_attestation,
            required_role=required_role,
        )
    except BenchmarkDatasetError as exc:
        raise BenchmarkInputError(f"invalid benchmark dataset evidence: {exc}") from exc


def _snapshot_evidence(
    model_size: str = _MODEL_SIZE,
    *,
    expected_repo: str | None = None,
    expected_revision: str | None = None,
    root: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    prefix = LibreQwen3VL.FILENAME_PREFIX
    repos = LibreQwen3VL.HF_REPOS
    revisions = LibreQwen3VL.HF_REVISIONS
    if (
        not isinstance(prefix, str)
        or not prefix
        or model_size not in _MODEL_SIZES
        or model_size not in repos
        or model_size not in revisions
    ):
        raise BenchmarkInputError("cannot derive the pinned Qwen snapshot identity")
    root = (
        Path.cwd() / "weights" / f"{prefix}{model_size}" if root is None else Path(root)
    )
    repo = str(repos[model_size])
    revision = str(revisions[model_size])
    official_repo, official_revision = _QWEN_BASE_PINS[model_size]
    if repo != official_repo or revision != official_revision:
        raise BenchmarkInputError(
            "the benchmark Qwen repository or revision differs from its official pin"
        )
    if expected_repo is not None and expected_repo != repo:
        raise BenchmarkInputError(
            "the checkpoint base repository differs from the benchmark model pin"
        )
    if expected_revision is not None and expected_revision != revision:
        raise BenchmarkInputError(
            "the checkpoint base revision differs from the benchmark model pin"
        )
    try:
        snapshot = VLMConfidenceValidator._base_snapshot_identity_from_root(
            root, repo, revision
        )
        processor = VLMConfidenceValidator._processor_content_identity_from_root(
            root, repo, revision
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise BenchmarkInputError(
            f"invalid local Qwen benchmark snapshot: {exc}"
        ) from exc
    if snapshot != _QWEN_SNAPSHOT_IDENTITIES[model_size]:
        raise BenchmarkInputError(
            "local Qwen benchmark weights do not match the official pinned bytes"
        )
    if processor != _QWEN_PROCESSOR_CONTENT_IDENTITIES[model_size]:
        raise BenchmarkInputError(
            "local Qwen benchmark processor content does not match the official "
            "pinned bytes"
        )
    return root, snapshot, processor


def _require_snapshot_stable(
    root: Path,
    model_size: str,
    expected_snapshot: Mapping[str, Any],
    expected_processor: Mapping[str, Any],
    *,
    phase: str,
    input_error: bool,
) -> None:
    base_repo, base_revision = _QWEN_BASE_PINS[model_size]
    try:
        actual_root, snapshot, processor = _snapshot_evidence(
            model_size,
            expected_repo=base_repo,
            expected_revision=base_revision,
            root=root,
        )
    except BenchmarkInputError:
        if input_error:
            raise
        raise RuntimeError(f"VLM base snapshot became invalid {phase}") from None
    if (
        actual_root != root
        or snapshot != expected_snapshot
        or processor != expected_processor
    ):
        error = BenchmarkInputError if input_error else RuntimeError
        raise error(f"VLM base snapshot identity changed {phase}")


def _audited_base_snapshot_files(
    source_root: Path,
    snapshot_identity: Mapping[str, Any],
    processor_identity: Mapping[str, Any],
) -> tuple[Path, tuple[tuple[str, Path], ...]]:
    try:
        _lexical, canonical_root = VLMConfidenceValidator._strict_local_directory_root(
            source_root, "Local base snapshot directory"
        )
        inventory = VLMConfidenceValidator._snapshot_files(canonical_root)
        processor_files = VLMConfidenceValidator._directory_identity_files(
            canonical_root, processor_artifacts=True
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise BenchmarkInputError(
            f"invalid local Qwen benchmark snapshot: {exc}"
        ) from exc

    expected_processor_files = processor_identity.get("files")
    if (
        isinstance(expected_processor_files, bool)
        or not isinstance(expected_processor_files, int)
        or expected_processor_files < 1
        or len(processor_files) != expected_processor_files
    ):
        raise BenchmarkInputError(
            "local Qwen benchmark processor inventory does not match its identity"
        )

    required = {".libreyolo_snapshot_complete"}
    artifacts = snapshot_identity.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise BenchmarkInputError(
            "local Qwen benchmark snapshot has no audited artifact inventory"
        )
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise BenchmarkInputError(
                "local Qwen benchmark snapshot has a malformed artifact inventory"
            )
        name = record.get("path")
        if not isinstance(name, str) or not name:
            raise BenchmarkInputError(
                "local Qwen benchmark snapshot has an unsafe artifact path"
            )
        relative = Path(name)
        if (
            relative.is_absolute()
            or name != relative.as_posix()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise BenchmarkInputError(
                "local Qwen benchmark snapshot has an unsafe artifact path"
            )
        required.add(name)
    required.update(
        path.relative_to(canonical_root).as_posix() for path in processor_files
    )
    missing = sorted(required - set(inventory))
    if missing:
        raise BenchmarkInputError(
            "local Qwen benchmark snapshot is missing audited load inputs: "
            + ", ".join(missing)
        )
    return canonical_root, tuple((name, inventory[name]) for name in sorted(required))


@contextmanager
def _isolated_base_snapshot(
    source_root: Path,
    model_size: str,
    snapshot_identity: Mapping[str, Any],
    processor_identity: Mapping[str, Any],
    *,
    enabled: bool,
) -> Iterator[Path | None]:
    """Retain an exact private model-load input for the complete run lifetime."""

    if not enabled:
        yield None
        return

    try:
        from libreyolo.models.vlm import artifact as artifact_module
    except ImportError as exc:
        raise BenchmarkInputError(
            "the installed LibreYOLO runtime cannot isolate the Qwen base snapshot"
        ) from exc

    try:
        temporary_snapshot = tempfile.TemporaryDirectory(
            prefix=_BASE_SNAPSHOT_TEMP_PREFIX
        )
    except OSError as exc:
        raise BenchmarkInputError(
            f"could not create a stable isolated base snapshot: {exc}"
        ) from exc

    body_completed = False
    try:
        try:
            isolated_root = Path(temporary_snapshot.name).resolve() / "snapshot"
            isolated_root.mkdir(mode=0o700)
            _canonical_source, files = _audited_base_snapshot_files(
                source_root, snapshot_identity, processor_identity
            )
            for relative, source in files:
                destination = isolated_root.joinpath(*Path(relative).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                artifact_module._copy_file_stable(
                    source,
                    destination,
                )
            _require_snapshot_stable(
                isolated_root,
                model_size,
                snapshot_identity,
                processor_identity,
                phase="while validating its isolated copy",
                input_error=True,
            )
            _require_snapshot_stable(
                source_root,
                model_size,
                snapshot_identity,
                processor_identity,
                phase="while creating its isolated copy",
                input_error=True,
            )
        except BenchmarkInputError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise BenchmarkInputError(
                f"could not create a stable isolated base snapshot: {exc}"
            ) from exc
        yield isolated_root
        body_completed = True
    finally:
        try:
            temporary_snapshot.cleanup()
        except OSError as exc:
            if body_completed:
                raise BenchmarkInputError(
                    f"could not remove the isolated base snapshot: {exc}"
                ) from exc


def _construct_benchmark_model(
    *,
    model_size: str,
    requested_device: str,
    snapshot_load_root: Path,
    checkpoint_load_root: Path | None,
):
    """Construct Qwen while forcing every base load to the retained private root."""

    base_class = LibreQwen3VL

    class _IsolatedSnapshotQwen3VL(base_class):
        def _ensure_weights(self) -> str:
            return str(snapshot_load_root)

    model_kwargs: dict[str, Any] = {
        "size": model_size,
        "device": requested_device,
    }
    if checkpoint_load_root is not None:
        model_kwargs["checkpoint_dir"] = str(checkpoint_load_root)
    return _IsolatedSnapshotQwen3VL(**model_kwargs)


@contextmanager
def _verified_pre_model_inputs(
    manifest: str | os.PathLike[str],
    source_annotations: str | os.PathLike[str],
    images_dir: str | os.PathLike[str],
    review_attestation: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    seed: int,
    device: str,
    checkpoint_dir: str | os.PathLike[str] | None,
    isolate_base_for_load: bool,
) -> Iterator[_PreModelInputs]:
    seed = _validate_seed(seed)
    requested_device = _validate_device_argument(device)
    destination = _validate_output_destination(output_root)
    (
        checkpoint_request_path,
        checkpoint_source_root,
        checkpoint_source_identity,
        checkpoint_context,
    ) = _prepare_checkpoint(checkpoint_dir, destination)
    model_size = (
        _MODEL_SIZE if checkpoint_context is None else checkpoint_context["size"]
    )
    required_partition_role = (
        _BASE_PARTITION_ROLE
        if checkpoint_context is None
        else _CHECKPOINT_PARTITION_ROLE
    )
    base_repo, base_revision = _QWEN_BASE_PINS[model_size]

    try:
        git_context = _git_context()
    except RuntimeError as exc:
        raise BenchmarkInputError(str(exc)) from exc
    if git_context["dirty"]:
        raise BenchmarkInputError(
            "the benchmark requires a clean git worktree so its code identity is "
            "immutable"
        )

    determinism = configure_determinism(seed)
    offline_context = _configure_offline_environment()
    _reject_faster_coco_override()
    package_versions = _required_package_versions()
    _require_pycocotools()
    resolved_device, device_probe = _resolve_and_probe_device(requested_device)
    try:
        runtime_context = _runtime_context(
            requested_device=requested_device,
            resolved_device=resolved_device,
            package_versions=package_versions,
        )
    except RuntimeError as exc:
        raise BenchmarkInputError(str(exc)) from exc

    verified = _verify_run_inputs(
        manifest,
        source_annotations,
        images_dir,
        review_attestation,
        required_role=required_partition_role,
    )
    dataset_context = _portable_dataset_context(verified)
    initial_verified_identity = _verified_inputs_identity(verified)

    with (
        _isolated_checkpoint_snapshot(
            checkpoint_source_root,
            checkpoint_source_identity,
            checkpoint_context,
        ) as (checkpoint_load_root, checkpoint_load_identity),
        _temporary_verified_dataset_yaml(verified) as dataset_yaml,
    ):
        native_dataset_context = _preflight_native_dataset(verified)
        snapshot_root, snapshot_identity, processor_identity = _snapshot_evidence(
            model_size,
            expected_repo=base_repo,
            expected_revision=base_revision,
        )
        with _isolated_base_snapshot(
            snapshot_root,
            model_size,
            snapshot_identity,
            processor_identity,
            enabled=isolate_base_for_load,
        ) as snapshot_load_root:
            stable_verified = _verify_run_inputs(
                manifest,
                source_annotations,
                images_dir,
                review_attestation,
                required_role=required_partition_role,
            )
            if _verified_inputs_identity(stable_verified) != initial_verified_identity:
                raise BenchmarkInputError(
                    "benchmark dataset evidence changed during preflight"
                )
            try:
                final_git_context = _git_context()
            except RuntimeError as exc:
                raise BenchmarkInputError(str(exc)) from exc
            if final_git_context != git_context:
                raise BenchmarkInputError(
                    "the benchmark git revision or worktree changed during preflight"
                )
            _require_checkpoint_stable(
                checkpoint_source_root,
                checkpoint_source_identity,
                phase="during preflight",
                input_error=True,
            )
            _require_checkpoint_stable(
                checkpoint_load_root,
                checkpoint_load_identity,
                phase="during isolated-checkpoint preflight",
                input_error=True,
            )
            _require_snapshot_stable(
                snapshot_root,
                model_size,
                snapshot_identity,
                processor_identity,
                phase="during preflight",
                input_error=True,
            )
            if snapshot_load_root is not None:
                _require_snapshot_stable(
                    snapshot_load_root,
                    model_size,
                    snapshot_identity,
                    processor_identity,
                    phase="during isolated-snapshot preflight",
                    input_error=True,
                )

            yield _PreModelInputs(
                destination=destination,
                model_size=model_size,
                checkpoint_request_path=checkpoint_request_path,
                checkpoint_source_root=checkpoint_source_root,
                checkpoint_source_identity=checkpoint_source_identity,
                checkpoint_load_root=checkpoint_load_root,
                checkpoint_load_identity=checkpoint_load_identity,
                checkpoint_context=checkpoint_context,
                requested_device=requested_device,
                resolved_device=resolved_device,
                verified=stable_verified,
                dataset_yaml=dataset_yaml,
                portable_dataset_context=dataset_context,
                native_dataset_context=native_dataset_context,
                git_context=git_context,
                determinism=determinism,
                runtime_context=runtime_context,
                device_probe=device_probe,
                offline_context=offline_context,
                snapshot_root=snapshot_root,
                snapshot_load_root=snapshot_load_root,
                snapshot_identity=snapshot_identity,
                processor_content_identity=processor_identity,
            )


def preflight_benchmark(
    manifest: str | os.PathLike[str],
    source_annotations: str | os.PathLike[str],
    images_dir: str | os.PathLike[str],
    review_attestation: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    seed: int = 0,
    device: str = "auto",
    checkpoint_dir: str | os.PathLike[str] | None = None,
) -> BenchmarkPreflight:
    """Verify that a benchmark run is ready without constructing its model."""

    with _verified_pre_model_inputs(
        manifest,
        source_annotations,
        images_dir,
        review_attestation,
        output_root,
        seed=seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
        isolate_base_for_load=False,
    ) as prepared:
        return BenchmarkPreflight(
            output_dir=prepared.destination,
            model_size=prepared.model_size,
            checkpoint_root=prepared.checkpoint_request_path,
            checkpoint_identity=prepared.checkpoint_context,
            snapshot_root=prepared.snapshot_root,
            snapshot_identity=prepared.snapshot_identity,
            processor_content_identity=prepared.processor_content_identity,
            dataset_context={
                "identity": prepared.portable_dataset_context,
                "native": prepared.native_dataset_context,
            },
            git_context=prepared.git_context,
            determinism=prepared.determinism,
            runtime_context={
                **prepared.runtime_context,
                "device_probe": prepared.device_probe,
            },
            offline_context=prepared.offline_context,
        )


def run_benchmark(
    manifest: str | os.PathLike[str],
    source_annotations: str | os.PathLike[str],
    images_dir: str | os.PathLike[str],
    review_attestation: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    seed: int = 0,
    device: str = "auto",
    checkpoint_dir: str | os.PathLike[str] | None = None,
) -> BenchmarkArtifacts:
    """Run one fresh pinned Qwen3-VL confidence benchmark into an immutable directory."""

    normalized_metrics: dict[str, float | None]
    nonfinite_metrics: tuple[str, ...]
    with ExitStack() as pre_model_stack:
        prepared = pre_model_stack.enter_context(
            _verified_pre_model_inputs(
                manifest,
                source_annotations,
                images_dir,
                review_attestation,
                output_root,
                seed=seed,
                device=device,
                checkpoint_dir=checkpoint_dir,
                isolate_base_for_load=True,
            )
        )
        seed = int(prepared.determinism["seed"])
        destination = prepared.destination
        verified = prepared.verified
        with _staged_output(destination) as stage:
            if prepared.snapshot_load_root is None:
                raise RuntimeError(
                    "benchmark run has no isolated base snapshot load root"
                )
            _require_checkpoint_stable(
                prepared.checkpoint_source_root,
                prepared.checkpoint_source_identity,
                phase="before model construction",
                input_error=False,
            )
            _require_checkpoint_stable(
                prepared.checkpoint_load_root,
                prepared.checkpoint_load_identity,
                phase="before model construction from the isolated copy",
                input_error=False,
            )
            _require_snapshot_stable(
                prepared.snapshot_root,
                prepared.model_size,
                prepared.snapshot_identity,
                prepared.processor_content_identity,
                phase="before model construction",
                input_error=False,
            )
            _require_snapshot_stable(
                prepared.snapshot_load_root,
                prepared.model_size,
                prepared.snapshot_identity,
                prepared.processor_content_identity,
                phase="before model construction from the isolated copy",
                input_error=False,
            )
            model = _construct_benchmark_model(
                model_size=prepared.model_size,
                requested_device=prepared.requested_device,
                snapshot_load_root=prepared.snapshot_load_root,
                checkpoint_load_root=prepared.checkpoint_load_root,
            )
            resolved_device = _resolved_model_device(model)
            if resolved_device != prepared.resolved_device:
                raise RuntimeError(
                    "benchmark model resolved to a different device than the "
                    "pre-model device probe"
                )
            _require_checkpoint_stable(
                prepared.checkpoint_source_root,
                prepared.checkpoint_source_identity,
                phase="during model construction",
                input_error=False,
            )
            _require_checkpoint_stable(
                prepared.checkpoint_load_root,
                prepared.checkpoint_load_identity,
                phase="during model construction from the isolated copy",
                input_error=False,
            )
            _require_snapshot_stable(
                prepared.snapshot_root,
                prepared.model_size,
                prepared.snapshot_identity,
                prepared.processor_content_identity,
                phase="during model construction",
                input_error=False,
            )
            _require_snapshot_stable(
                prepared.snapshot_load_root,
                prepared.model_size,
                prepared.snapshot_identity,
                prepared.processor_content_identity,
                phase="during model construction from the isolated copy",
                input_error=False,
            )
            snapshot_identity = prepared.snapshot_identity
            processor_identity = prepared.processor_content_identity

            runtime_context = dict(prepared.runtime_context)
            runtime_context["attention_backends"] = _attention_backends(model)
            execution_context = {
                "schema": _CONTEXT_SCHEMA,
                "git": prepared.git_context,
                "runtime": runtime_context,
                "determinism": prepared.determinism,
                "dataset": prepared.portable_dataset_context,
                "checkpoint": prepared.checkpoint_context,
            }
            config = ValidationConfig(
                data=str(prepared.dataset_yaml),
                split="val",
                batch_size=1,
                imgsz=_IMAGE_SIZE,
                device=prepared.requested_device,
                save_dir=str(stage),
                num_workers=0,
                allow_download_scripts=False,
                save_json=True,
                save_plots=True,
                faster_coco_eval=False,
            )
            identity_expectations: dict[str, Any]
            if prepared.checkpoint_load_identity is None:
                identity_expectations = {
                    "expected_snapshot_identity": snapshot_identity,
                    "expected_processor_content_identity": processor_identity,
                }
            else:
                identity_expectations = {
                    "expected_checkpoint_identity": prepared.checkpoint_load_identity,
                    "expected_snapshot_identity": snapshot_identity,
                    "expected_processor_content_identity": processor_identity,
                }
            identity_expectations["expected_snapshot_root"] = (
                prepared.snapshot_load_root
            )
            validator = VLMConfidenceValidator(
                model,
                config,
                seed=seed,
                default_conf=_DEFAULT_CONF,
                confidence_iou=_CONFIDENCE_IOU,
                benchmark_context=execution_context,
                verified_dataset=verified,
                **identity_expectations,
            )
            metrics = validator.run()
            _require_checkpoint_stable(
                prepared.checkpoint_source_root,
                prepared.checkpoint_source_identity,
                phase="during generation",
                input_error=False,
            )
            _require_checkpoint_stable(
                prepared.checkpoint_load_root,
                prepared.checkpoint_load_identity,
                phase="during generation from the isolated copy",
                input_error=False,
            )
            _require_snapshot_stable(
                prepared.snapshot_root,
                prepared.model_size,
                prepared.snapshot_identity,
                prepared.processor_content_identity,
                phase="during generation",
                input_error=False,
            )
            _require_snapshot_stable(
                prepared.snapshot_load_root,
                prepared.model_size,
                prepared.snapshot_identity,
                prepared.processor_content_identity,
                phase="during generation from the isolated copy",
                input_error=False,
            )
            if not isinstance(metrics, Mapping):
                raise TypeError("VLM confidence validator must return a metric mapping")
            normalized_metrics, nonfinite_metrics = _normalized_metrics(metrics)
            if _git_context() != prepared.git_context:
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
                    "model_size": prepared.model_size,
                    "checkpoint_dir": (
                        None
                        if prepared.checkpoint_request_path is None
                        else str(prepared.checkpoint_request_path)
                    ),
                    "device": prepared.requested_device,
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
            del validator, model, metrics
            gc.collect()
            # Release every pre-model temporary resource, especially the isolated
            # checkpoint and base snapshot, before staged output publication.
            pre_model_stack.close()

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


def _load_envelope_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
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
    return decoded, payload


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


def _validate_dataset_context(
    value: Any,
    label: str,
    path: str,
    *,
    required_role: str,
) -> dict[str, Any]:
    expected_partition = _PARTITION_REQUIREMENTS.get(required_role)
    if expected_partition is None:
        raise RuntimeError("benchmark partition selection is internally inconsistent")
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
    if review["partition_role"] != required_role:
        raise _envelope_error(
            label,
            f"{path}.review.partition_role",
            f"must equal {required_role!r}",
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


def _validate_checkpoint_context(
    value: Any, request: Mapping[str, Any], label: str
) -> dict[str, Any] | None:
    checkpoint_dir = request["checkpoint_dir"]
    if value is None:
        if checkpoint_dir is not None:
            raise _envelope_error(
                label,
                "$.execution_context.checkpoint",
                "must identify request.checkpoint_dir",
            )
        if request["model_size"] != _MODEL_SIZE:
            raise _envelope_error(
                label,
                "$.request.model_size",
                f"must equal {_MODEL_SIZE!r} without a checkpoint",
            )
        return None
    if checkpoint_dir is None:
        raise _envelope_error(
            label,
            "$.execution_context.checkpoint",
            "must be null when request.checkpoint_dir is null",
        )
    path = "$.execution_context.checkpoint"
    checkpoint = _exact_object(
        value,
        {
            "schema",
            "kind",
            "family",
            "size",
            "task",
            "base_repo",
            "base_revision",
            "aggregate_sha256",
            "adapter_weights_sha256",
            "adapter_config_sha256",
            "checkpoint_contract_sha256",
            "processor_sha256",
            "files",
        },
        label,
        path,
    )
    if checkpoint["schema"] != _CHECKPOINT_CONTEXT_SCHEMA:
        raise _envelope_error(
            label,
            f"{path}.schema",
            f"must equal {_CHECKPOINT_CONTEXT_SCHEMA!r}",
        )
    if checkpoint["kind"] != "qwen3vl_lora_checkpoint":
        raise _envelope_error(
            label, f"{path}.kind", "must equal 'qwen3vl_lora_checkpoint'"
        )
    if checkpoint["family"] != _MODEL_FAMILY or checkpoint["task"] != "detect":
        raise _envelope_error(
            label, path, "must identify a Qwen3-VL detection checkpoint"
        )
    size = checkpoint["size"]
    if size not in _MODEL_SIZES or size != request["model_size"]:
        raise _envelope_error(label, f"{path}.size", "must equal request.model_size")
    expected_repo, expected_revision = _QWEN_BASE_PINS[size]
    if (
        checkpoint["base_repo"] != expected_repo
        or checkpoint["base_revision"] != expected_revision
    ):
        raise _envelope_error(
            label, path, "must bind the official pinned Qwen3-VL base"
        )
    for field in (
        "aggregate_sha256",
        "adapter_weights_sha256",
        "adapter_config_sha256",
        "checkpoint_contract_sha256",
        "processor_sha256",
    ):
        if not isinstance(checkpoint[field], str) or not _HEX_DIGEST.fullmatch(
            checkpoint[field]
        ):
            raise _envelope_error(
                label, f"{path}.{field}", "must be a lowercase SHA256 digest"
            )
    files = checkpoint["files"]
    if not isinstance(files, list) or not files:
        raise _envelope_error(
            label, f"{path}.files", "must be a non-empty file inventory"
        )
    normalized_files = []
    for index, raw in enumerate(files):
        file_path = f"{path}.files[{index}]"
        entry = _exact_object(raw, {"path", "role", "size", "sha256"}, label, file_path)
        name = entry["path"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or any(ord(character) < 32 for character in name)
        ):
            raise _envelope_error(
                label, f"{file_path}.path", "must be a safe file basename"
            )
        if entry["role"] not in _CHECKPOINT_FILE_ROLES:
            raise _envelope_error(
                label, f"{file_path}.role", "is not a supported checkpoint role"
            )
        size_bytes = entry["size"]
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or not 0 < size_bytes <= _MAX_SAFE_INTEGER
        ):
            raise _envelope_error(
                label, f"{file_path}.size", "must be a positive safe integer"
            )
        if not isinstance(entry["sha256"], str) or not _HEX_DIGEST.fullmatch(
            entry["sha256"]
        ):
            raise _envelope_error(
                label, f"{file_path}.sha256", "must be a lowercase SHA256 digest"
            )
        normalized_files.append(dict(entry))
    paths = [entry["path"] for entry in normalized_files]
    if paths != sorted(paths, key=str.casefold) or len(
        {name.casefold() for name in paths}
    ) != len(paths):
        raise _envelope_error(
            label,
            f"{path}.files",
            "must be sorted by path and unique case-insensitively",
        )
    roles = [entry["role"] for entry in normalized_files]
    if (
        any(
            roles.count(role) != 1
            for role in ("checkpoint_contract", "adapter_config", "adapter_weights")
        )
        or roles.count("processor") < 1
    ):
        raise _envelope_error(
            label,
            f"{path}.files",
            "must contain one contract, adapter config, adapter payload, and processor",
        )
    adapter_file = next(
        entry for entry in normalized_files if entry["role"] == "adapter_weights"
    )
    if adapter_file["sha256"] != checkpoint["adapter_weights_sha256"]:
        raise _envelope_error(
            label,
            f"{path}.adapter_weights_sha256",
            "must match the adapter_weights file record",
        )
    if _checkpoint_processor_sha256(normalized_files) != checkpoint["processor_sha256"]:
        raise _envelope_error(
            label,
            f"{path}.processor_sha256",
            "must match the processor file records",
        )
    if (
        _checkpoint_aggregate_sha256(checkpoint, normalized_files)
        != checkpoint["aggregate_sha256"]
    ):
        raise _envelope_error(
            label,
            f"{path}.aggregate_sha256",
            "must match the source checkpoint identity payload",
        )
    return {**dict(checkpoint), "files": normalized_files}


def _validate_execution_context(
    value: Any, request: Mapping[str, Any], label: str
) -> dict[str, Any]:
    context = _exact_object(
        value,
        {"schema", "git", "runtime", "determinism", "dataset", "checkpoint"},
        label,
        "$.execution_context",
    )
    if context["schema"] != _CONTEXT_SCHEMA:
        raise _envelope_error(
            label, "$.execution_context.schema", f"must equal {_CONTEXT_SCHEMA!r}"
        )
    required_partition_role = (
        _BASE_PARTITION_ROLE
        if request["checkpoint_dir"] is None
        else _CHECKPOINT_PARTITION_ROLE
    )
    _validate_dataset_context(
        context["dataset"],
        label,
        "$.execution_context.dataset",
        required_role=required_partition_role,
    )
    _validate_checkpoint_context(context["checkpoint"], request, label)
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
    envelope, envelope_payload = _load_envelope_json(envelope_path, label)
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
            "checkpoint_dir",
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
    checkpoint_dir = request["checkpoint_dir"]
    if checkpoint_dir is not None and (
        not isinstance(checkpoint_dir, str)
        or not checkpoint_dir
        or not Path(checkpoint_dir).is_absolute()
    ):
        raise _envelope_error(
            label,
            "$.request.checkpoint_dir",
            "must be null or an absolute operational path",
        )
    if (
        request["model_family"] != _MODEL_FAMILY
        or request["model_size"] not in _MODEL_SIZES
    ):
        raise _envelope_error(
            label, "$.request", "must identify a supported Qwen3-VL benchmark"
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
        or benchmark_config.get("size") != request["model_size"]
    ):
        raise _envelope_error(
            label,
            "$.request",
            "does not match the model identity in the report",
        )
    checkpoint_context = execution_context["checkpoint"]
    expected_repo, expected_revision = _QWEN_BASE_PINS[request["model_size"]]
    if (
        benchmark_config.get("base_repo") != expected_repo
        or benchmark_config.get("base_revision") != expected_revision
    ):
        raise _envelope_error(
            label,
            "$.execution_context.checkpoint",
            "does not match the pinned base identity in the report",
        )
    native_checkpoint = benchmark_config.get("checkpoint")
    if checkpoint_context is not None:
        if not isinstance(native_checkpoint, Mapping) or _json_text(
            native_checkpoint
        ) != _json_text(checkpoint_context):
            raise _envelope_error(
                label,
                "$.execution_context.checkpoint",
                "does not match benchmark_config.checkpoint in the report",
            )
        report_processor = benchmark_config.get("processor")
        expected_processor_files = sum(
            entry["role"] == "processor" for entry in checkpoint_context["files"]
        )
        if (
            not isinstance(report_processor, Mapping)
            or report_processor.get("source") != "checkpoint"
            or report_processor.get("revision") is not None
            or report_processor.get("sha256") != checkpoint_context["processor_sha256"]
            or report_processor.get("files") != expected_processor_files
        ):
            raise _envelope_error(
                label,
                "$.execution_context.checkpoint.processor_sha256",
                "does not match the strict checkpoint processor in the report",
            )
    elif not isinstance(native_checkpoint, Mapping) or (
        native_checkpoint.get("kind") != "pinned_hf_snapshot"
    ):
        raise _envelope_error(
            label,
            "$.execution_context.checkpoint",
            "base-only runs must report their pinned base snapshot identity",
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
        envelope_sha256=hashlib.sha256(envelope_payload).hexdigest(),
        execution_context=json.loads(_json_text(execution_context)),
        benchmark_config=json.loads(_json_text(benchmark_config)),
        metrics=json.loads(_json_text(dict(metrics))),
        nonfinite_metrics=tuple(nonfinite),
    )


def read_benchmark_run_identity(
    report_path: str | os.PathLike[str], *, label: str = "benchmark_run"
) -> BenchmarkRunIdentity:
    """Validate one report with its sibling runner envelope and return its identity."""

    if not isinstance(label, str) or not label:
        raise TypeError("label must be a non-empty string")
    validated = _load_runner_envelope(report_path, label)
    return BenchmarkRunIdentity(
        run_id=validated.run_id,
        process_id=validated.process_id,
        report_sha256=validated.report_sha256,
        envelope_sha256=validated.envelope_sha256,
        execution_context=json.loads(_json_text(validated.execution_context)),
        benchmark_config=json.loads(_json_text(validated.benchmark_config)),
        metrics=json.loads(_json_text(validated.metrics)),
        nonfinite_metrics=tuple(validated.nonfinite_metrics),
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


def _add_benchmark_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--review-attestation", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--seed", type=_arg_seed, default=0)
    parser.add_argument("--device", default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = _JSONArgumentParser(
        prog="python -m libreyolo.validation.vlm_confidence_benchmark"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run_parser = subparsers.add_parser(
        "run", help="run one fresh pinned Qwen3-VL benchmark process"
    )
    _add_benchmark_request_arguments(run_parser)
    preflight_parser = subparsers.add_parser(
        "preflight",
        help="verify one pinned Qwen3-VL benchmark request without loading the model",
    )
    _add_benchmark_request_arguments(preflight_parser)

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
        arguments[0]
        if arguments and arguments[0] in {"preflight", "run", "compare"}
        else None
    )
    try:
        args = parse_cli_args(arguments)
    except _CLIUsageError as exc:
        return _error_status(
            mode=mode_hint, code=EXIT_USAGE, kind="usage", message=str(exc)
        )

    try:
        if args.mode == "preflight":
            with redirect_stdout(sys.stderr):
                preflight = preflight_benchmark(
                    args.manifest,
                    args.annotations,
                    args.images_dir,
                    args.review_attestation,
                    args.output_root,
                    seed=args.seed,
                    device=args.device,
                    checkpoint_dir=args.checkpoint_dir,
                )
            _emit_status(
                {
                    "schema": _STATUS_SCHEMA,
                    "status": "ready",
                    "mode": "preflight",
                    "code": EXIT_OK,
                    "preflight": {
                        "schema": _PREFLIGHT_SCHEMA,
                        "request": {
                            "model_family": _MODEL_FAMILY,
                            "model_size": preflight.model_size,
                            "checkpoint_dir": preflight.checkpoint_root,
                            "seed": preflight.determinism["seed"],
                            "device": args.device.strip(),
                            "resolved_device": preflight.runtime_context[
                                "resolved_device"
                            ],
                            "output_root": preflight.output_dir,
                        },
                        "git": preflight.git_context,
                        "offline": preflight.offline_context,
                        "determinism": preflight.determinism,
                        "runtime": preflight.runtime_context,
                        "dataset": preflight.dataset_context,
                        "snapshot": {
                            "root": preflight.snapshot_root,
                            "weights": preflight.snapshot_identity,
                            "processor_content": (preflight.processor_content_identity),
                        },
                        "checkpoint": preflight.checkpoint_identity,
                    },
                }
            )
            return EXIT_OK

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
                    checkpoint_dir=args.checkpoint_dir,
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
