"""Helpers for safe torch loading and LibreYOLO checkpoint metadata."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path
from typing import Any

import torch

from ..tasks import normalize_task


SCHEMA_VERSION = "1.0"

REQUIRED_CHECKPOINT_METADATA_KEYS = (
    "model",
    "schema_version",
    "libreyolo_version",
    "model_family",
    "size",
    "task",
    "nc",
    "names",
    "imgsz",
)


class CheckpointMetadataError(ValueError):
    """Raised when a checkpoint does not satisfy the LibreYOLO metadata schema."""


class CheckpointLoadError(RuntimeError):
    """Raised when checkpoint tensors do not satisfy the declared load policy."""


@dataclass(frozen=True)
class CheckpointLoadPolicy:
    """Explicit contract for loading a state dict into a native model.

    Complete LibreYOLO checkpoints use an exact policy. Legacy checkpoints may
    opt into bounded missing-key compatibility, while deliberate transfer paths
    declare the only model tensors they are allowed to leave initialized.
    Unexpected keys and shape mismatches always require explicit allowlists.
    """

    name: str = "native-exact"
    allow_partial_missing: bool = False
    min_key_coverage: float = 1.0
    min_element_coverage: float = 1.0
    allowed_missing: tuple[str, ...] = ()
    allowed_unexpected: tuple[str, ...] = ()
    allowed_shape_mismatch: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("min_key_coverage", "min_element_coverage"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0, 1], got {value}.")

    def allowing(
        self,
        *,
        name: str | None = None,
        missing: tuple[str, ...] = (),
        unexpected: tuple[str, ...] = (),
        shape_mismatch: tuple[str, ...] = (),
    ) -> "CheckpointLoadPolicy":
        """Return a copy extended with narrow, named compatibility exceptions."""
        return replace(
            self,
            name=name or self.name,
            allowed_missing=self.allowed_missing + tuple(missing),
            allowed_unexpected=self.allowed_unexpected + tuple(unexpected),
            allowed_shape_mismatch=(
                self.allowed_shape_mismatch + tuple(shape_mismatch)
            ),
        )


NATIVE_CHECKPOINT_LOAD_POLICY = CheckpointLoadPolicy()
LEGACY_CHECKPOINT_LOAD_POLICY = CheckpointLoadPolicy(
    name="legacy-coverage",
    allow_partial_missing=True,
    min_key_coverage=0.90,
    min_element_coverage=0.90,
)


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Measured tensor coverage for one checkpoint load attempt."""

    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    allowed_missing_keys: tuple[str, ...]
    allowed_unexpected_keys: tuple[str, ...]
    allowed_shape_mismatches: tuple[str, ...]
    total_target_tensors: int
    loaded_elements: int
    total_target_elements: int
    allowed_unloaded_elements: int

    @property
    def eligible_target_tensors(self) -> int:
        return max(
            0,
            self.total_target_tensors
            - len(self.allowed_missing_keys)
            - len(self.allowed_shape_mismatches),
        )

    @property
    def eligible_target_elements(self) -> int:
        return max(0, self.total_target_elements - self.allowed_unloaded_elements)

    @property
    def key_coverage(self) -> float:
        if self.eligible_target_tensors == 0:
            return 1.0
        return len(self.loaded_keys) / self.eligible_target_tensors

    @property
    def element_coverage(self) -> float:
        if self.eligible_target_elements == 0:
            return 1.0
        return self.loaded_elements / self.eligible_target_elements

    def summary(self) -> str:
        return (
            f"loaded={len(self.loaded_keys)}/{self.eligible_target_tensors} eligible tensors "
            f"({self.key_coverage:.1%}), elements={self.loaded_elements}/"
            f"{self.eligible_target_elements} eligible ({self.element_coverage:.1%}), "
            f"missing={len(self.missing_keys)}, "
            f"unexpected={len(self.unexpected_keys)}, "
            f"shape_mismatched={len(self.shape_mismatches)}, "
            f"allowed_missing={len(self.allowed_missing_keys)}, "
            f"allowed_unexpected={len(self.allowed_unexpected_keys)}, "
            f"allowed_shape_mismatched={len(self.allowed_shape_mismatches)}"
        )


def _matches_any(key: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatchcase(key, pattern) for pattern in patterns)


def _state_value_elements(value: Any) -> int:
    numel = getattr(value, "numel", None)
    if not callable(numel):
        return 0
    try:
        return int(numel())
    except (TypeError, ValueError):
        return 0


def inspect_state_dict_load(
    module: torch.nn.Module,
    state_dict: dict[str, Any],
    *,
    policy: CheckpointLoadPolicy,
) -> CheckpointLoadReport:
    """Inspect a state dict without mutating ``module``."""
    if not isinstance(state_dict, dict):
        raise CheckpointLoadError("state_dict must be a dictionary.")

    target = module.state_dict()
    loaded: list[str] = []
    shape_mismatches: list[str] = []
    unexpected: list[str] = []

    for key, value in state_dict.items():
        if key not in target:
            unexpected.append(str(key))
            continue
        target_value = target[key]
        incoming_shape = getattr(value, "shape", None)
        target_shape = getattr(target_value, "shape", None)
        if incoming_shape is None or tuple(incoming_shape) != tuple(target_shape):
            shape_mismatches.append(str(key))
            continue
        loaded.append(str(key))

    loaded_set = set(loaded)
    mismatch_set = set(shape_mismatches)
    missing = sorted(set(target) - loaded_set - mismatch_set)

    allowed_missing = sorted(
        key for key in missing if _matches_any(key, policy.allowed_missing)
    )
    allowed_unexpected = sorted(
        key for key in unexpected if _matches_any(key, policy.allowed_unexpected)
    )
    allowed_mismatches = sorted(
        key
        for key in shape_mismatches
        if _matches_any(key, policy.allowed_shape_mismatch)
    )

    return CheckpointLoadReport(
        loaded_keys=tuple(sorted(loaded)),
        missing_keys=tuple(sorted(set(missing) - set(allowed_missing))),
        unexpected_keys=tuple(sorted(set(unexpected) - set(allowed_unexpected))),
        shape_mismatches=tuple(sorted(set(shape_mismatches) - set(allowed_mismatches))),
        allowed_missing_keys=tuple(allowed_missing),
        allowed_unexpected_keys=tuple(allowed_unexpected),
        allowed_shape_mismatches=tuple(allowed_mismatches),
        total_target_tensors=len(target),
        loaded_elements=sum(_state_value_elements(target[key]) for key in loaded),
        total_target_elements=sum(
            _state_value_elements(value) for value in target.values()
        ),
        allowed_unloaded_elements=sum(
            _state_value_elements(target[key])
            for key in (*allowed_missing, *allowed_mismatches)
        ),
    )


def inspect_load_state_dict_result(
    module: torch.nn.Module,
    result: Any,
    *,
    policy: CheckpointLoadPolicy,
) -> CheckpointLoadReport:
    """Build a report after a loader that adapts its module before loading.

    This is reserved for wrappers such as RF-DETR whose ``load_state_dict``
    first rebuilds schema-dependent heads. Ordinary loaders should use the
    mutation-free :func:`load_state_dict_checked` path.
    """
    target = module.state_dict()
    if hasattr(result, "missing_keys"):
        raw_missing = result.missing_keys
        raw_unexpected = result.unexpected_keys
    elif result:
        raw_missing, raw_unexpected = result
    else:
        raw_missing, raw_unexpected = (), ()
    missing = sorted(set(raw_missing or ()))
    unexpected = sorted(set(raw_unexpected or ()))
    allowed_missing = sorted(
        key for key in missing if _matches_any(key, policy.allowed_missing)
    )
    allowed_unexpected = sorted(
        key for key in unexpected if _matches_any(key, policy.allowed_unexpected)
    )
    loaded = sorted(set(target) - set(missing))

    # Tiny test doubles may report missing keys without exposing a state dict.
    # Count those reported targets so the report remains truthful.
    total_target_tensors = max(len(target), len(loaded) + len(missing))
    total_target_elements = sum(
        _state_value_elements(value) for value in target.values()
    )
    loaded_elements = sum(
        _state_value_elements(target[key]) for key in loaded if key in target
    )

    return CheckpointLoadReport(
        loaded_keys=tuple(loaded),
        missing_keys=tuple(sorted(set(missing) - set(allowed_missing))),
        unexpected_keys=tuple(sorted(set(unexpected) - set(allowed_unexpected))),
        shape_mismatches=(),
        allowed_missing_keys=tuple(allowed_missing),
        allowed_unexpected_keys=tuple(allowed_unexpected),
        allowed_shape_mismatches=(),
        total_target_tensors=total_target_tensors,
        loaded_elements=loaded_elements,
        total_target_elements=total_target_elements,
        allowed_unloaded_elements=sum(
            _state_value_elements(target[key])
            for key in allowed_missing
            if key in target
        ),
    )


def enforce_checkpoint_load_report(
    report: CheckpointLoadReport,
    *,
    policy: CheckpointLoadPolicy,
    context: str,
) -> None:
    """Raise when a measured load violates its explicit policy."""
    reasons: list[str] = []
    if report.unexpected_keys:
        reasons.append("unexpected checkpoint keys are not allowed")
    if report.shape_mismatches:
        reasons.append("tensor shape mismatches are not allowed")
    if report.missing_keys and not policy.allow_partial_missing:
        reasons.append("required model tensors are missing")
    if policy.allow_partial_missing:
        if report.key_coverage < policy.min_key_coverage:
            reasons.append(
                f"key coverage {report.key_coverage:.1%} is below "
                f"{policy.min_key_coverage:.1%}"
            )
        if report.element_coverage < policy.min_element_coverage:
            reasons.append(
                f"element coverage {report.element_coverage:.1%} is below "
                f"{policy.min_element_coverage:.1%}"
            )

    if not reasons:
        return

    details: list[str] = []
    for label, keys in (
        ("missing", report.missing_keys),
        ("unexpected", report.unexpected_keys),
        ("shape_mismatched", report.shape_mismatches),
    ):
        if keys:
            preview = ", ".join(keys[:8])
            suffix = f" (+{len(keys) - 8} more)" if len(keys) > 8 else ""
            details.append(f"{label}=[{preview}]{suffix}")

    detail_text = f"; {'; '.join(details)}" if details else ""
    raise CheckpointLoadError(
        f"{context} violates checkpoint load policy {policy.name!r}: "
        f"{'; '.join(reasons)}. {report.summary()}{detail_text}"
    )


def load_state_dict_checked(
    module: torch.nn.Module,
    state_dict: dict[str, Any],
    *,
    policy: CheckpointLoadPolicy,
    context: str,
) -> CheckpointLoadReport:
    """Validate coverage and then load a state dict without silent partial state."""
    report = inspect_state_dict_load(module, state_dict, policy=policy)
    enforce_checkpoint_load_report(report, policy=policy, context=context)

    ignored = set(report.allowed_unexpected_keys) | set(report.allowed_shape_mismatches)
    filtered = {key: value for key, value in state_dict.items() if key not in ignored}
    module.load_state_dict(filtered, strict=False)
    return report


def get_libreyolo_version() -> str:
    """Return the installed LibreYOLO version, with an editable-install fallback."""
    try:
        return version("libreyolo")
    except PackageNotFoundError:
        return "0.0.0.dev0"


def _supports_weights_only() -> bool:
    """Return whether the installed torch.load supports ``weights_only``."""
    try:
        signature = inspect.signature(torch.load)
    except (TypeError, ValueError):
        return False
    return "weights_only" in signature.parameters


def _torch_load(
    path: str | Path,
    *,
    map_location: Any,
    weights_only: bool,
    context: str,
    safe_globals: list[Any] | tuple[Any, ...] | None = None,
):
    load_kwargs = {"map_location": map_location}

    if _supports_weights_only():
        load_kwargs["weights_only"] = weights_only
        if weights_only and safe_globals:
            safe_globals_context = getattr(
                getattr(torch, "serialization", None),
                "safe_globals",
                None,
            )
            if safe_globals_context is None:
                raise RuntimeError(
                    f"Safe loading for {context} requires a PyTorch build that "
                    "supports torch.serialization.safe_globals(...)."
                )
            with safe_globals_context(list(safe_globals)):
                return torch.load(path, **load_kwargs)
        return torch.load(path, **load_kwargs)

    if weights_only:
        raise RuntimeError(
            f"Safe loading for {context} requires a PyTorch build that supports "
            "torch.load(..., weights_only=...). Upgrade PyTorch or use a trusted "
            "checkpoint workflow."
        )

    return torch.load(path, **load_kwargs)


def load_untrusted_torch_file(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    context: str = "model weights",
    safe_globals: list[Any] | tuple[Any, ...] | None = None,
):
    """Safely load a user-supplied torch file."""
    return _torch_load(
        path,
        map_location=map_location,
        weights_only=True,
        context=context,
        safe_globals=safe_globals,
    )


def load_trusted_torch_file(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    context: str = "trusted checkpoint",
):
    """Load a trusted internal torch checkpoint with full metadata."""
    return _torch_load(
        path,
        map_location=map_location,
        weights_only=False,
        context=context,
    )


def build_class_names(nc: int) -> dict[int, str]:
    """Return COCO names for 80 classes, else a generic indexed mapping."""
    if nc == 80:
        from .general import COCO_CLASSES

        return {index: name for index, name in enumerate(COCO_CLASSES)}

    return {index: f"class_{index}" for index in range(nc)}


def normalize_checkpoint_names(
    names: Any,
    nc: int,
    *,
    allow_sparse: bool = True,
) -> dict[int, str]:
    """Normalize names, optionally padding sparse legacy reader metadata."""
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict):
        raise CheckpointMetadataError("names must be a dict[int, str] or list[str].")

    normalized: dict[int, str] = {}
    for key, value in names.items():
        if isinstance(key, bool):
            raise CheckpointMetadataError(
                f"names contains a non-integer class index: {key!r}."
            )
        if not allow_sparse and not isinstance(key, (int, str)):
            raise CheckpointMetadataError(
                f"names contains a non-integer class index: {key!r}."
            )
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise CheckpointMetadataError(
                f"names contains a non-integer class index: {key!r}."
            ) from exc
        if index in normalized:
            raise CheckpointMetadataError(
                "names contains duplicate class indices after normalization: "
                f"{key!r} resolves to {index}."
            )
        if not allow_sparse and not isinstance(value, str):
            raise CheckpointMetadataError(
                f"names[{index}] must be a string, got {type(value).__name__}."
            )
        normalized[index] = str(value)

    expected = set(range(nc))
    extra = sorted(index for index in normalized if index not in expected)
    if extra:
        raise CheckpointMetadataError(
            "names keys must be within class indices 0..nc-1 "
            f"(nc={nc}, got out-of-range keys {extra})."
        )

    missing = sorted(expected - set(normalized))
    if missing:
        if not allow_sparse:
            raise CheckpointMetadataError(
                "names must include every class index in 0..nc-1 "
                f"(missing indices {missing})."
            )
        warnings.warn(
            "names is missing class indices "
            f"{missing}; padding with generic class_i labels.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {index: normalized.get(index, f"class_{index}") for index in range(nc)}


def _infer_checkpoint_imgsz(
    *,
    model_family: str,
    size: str,
    task: str,
) -> int | None:
    """Infer square input size from the immutable public model manifest."""
    from ..models.manifest import get_artifact_spec

    artifact = get_artifact_spec(model_family, size, task)
    return artifact.native_imgsz if artifact is not None else None


def _checkpoint_imgsz_contract_error(
    *,
    model_family: str,
    size: str,
    task: str,
    imgsz: int,
) -> str | None:
    """Return a public family input-contract error without building a model."""
    from ..models.manifest import get_artifact_spec, load_family_class

    artifact = get_artifact_spec(model_family, size, task)
    if artifact is None:
        return None
    try:
        model_class = load_family_class(model_family)
    except (ImportError, ModuleNotFoundError):
        # Metadata inspection must remain available without optional model
        # extras. The installed reader performs its family-specific validation.
        return None

    model = object.__new__(model_class)
    model.input_size = artifact.native_imgsz
    model.size = artifact.size
    model.task = artifact.task
    if artifact.family == "rfdetr":
        from types import SimpleNamespace

        from ..models.rfdetr.nn import (
            RFDETR_CONFIGS,
            RFDETR_POSE_CONFIGS,
            RFDETR_SEG_CONFIGS,
        )

        configs = (
            RFDETR_SEG_CONFIGS
            if artifact.task == "segment"
            else RFDETR_POSE_CONFIGS
            if artifact.task == "pose"
            else RFDETR_CONFIGS
        )
        config = configs[artifact.size]
        model.model = SimpleNamespace(
            patch_size=config.patch_size,
            num_windows=config.num_windows,
        )
    try:
        model_class._validate_input_size(
            model,
            imgsz,
            context="checkpoint",
            allow_fixed_override=bool(
                getattr(model_class, "CHECKPOINT_INPUT_SIZE_OVERRIDE", False)
            ),
        )
    except (RuntimeError, ValueError) as exc:
        return str(exc)
    return None


def wrap_libreyolo_checkpoint(
    state_dict: dict[str, torch.Tensor],
    *,
    model_family: str,
    size: str,
    task: str,
    nc: int,
    names: dict[int, str] | list[str] | None = None,
    imgsz: int | None = None,
    libreyolo_version: str | None = None,
    schema_version: str = SCHEMA_VERSION,
    **extra_metadata: Any,
) -> dict[str, Any]:
    """Build a strict LibreYOLO v1.0 metadata-wrapped checkpoint."""
    reserved = sorted(set(extra_metadata) & set(REQUIRED_CHECKPOINT_METADATA_KEYS))
    if reserved:
        raise CheckpointMetadataError(
            "extra checkpoint metadata cannot override required fields: "
            + ", ".join(reserved)
        )

    normalized_task = normalize_task(task)
    if normalized_task is None:
        raise CheckpointMetadataError("task is required.")
    if isinstance(model_family, str):
        model_family = model_family.strip().casefold()
    if isinstance(size, str):
        size = size.strip().casefold()

    if names is None:
        names = build_class_names(nc)
    normalized_names = normalize_checkpoint_names(names, nc, allow_sparse=False)

    if imgsz is None:
        imgsz = _infer_checkpoint_imgsz(
            model_family=model_family,
            size=size,
            task=normalized_task,
        )
    if imgsz is None:
        raise CheckpointMetadataError(
            "imgsz is required and could not be inferred from model_family/size/task."
        )
    if isinstance(imgsz, bool) or not isinstance(imgsz, Integral) or imgsz <= 0:
        raise CheckpointMetadataError(
            f"imgsz must be a positive int, got {imgsz!r}."
        )
    imgsz = int(imgsz)
    imgsz_error = _checkpoint_imgsz_contract_error(
        model_family=model_family,
        size=size,
        task=normalized_task,
        imgsz=imgsz,
    )
    if imgsz_error is not None:
        raise CheckpointMetadataError(imgsz_error)

    checkpoint: dict[str, Any] = {
        "model": state_dict,
        "schema_version": schema_version,
        "libreyolo_version": libreyolo_version or get_libreyolo_version(),
        "model_family": model_family,
        "size": size,
        "task": normalized_task,
        "nc": nc,
        "names": normalized_names,
        "imgsz": imgsz,
    }
    # Optional fields with None are intentionally omitted from checkpoint files.
    checkpoint.update({k: v for k, v in extra_metadata.items() if v is not None})
    validate_checkpoint_metadata(checkpoint, strict=True)
    return checkpoint


def validate_checkpoint_metadata(
    checkpoint: Any,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate a LibreYOLO checkpoint wrapper against metadata schema v1.0.

    This function is intentionally non-mutating. Callers that need normalized
    values should do so explicitly through the schema construction helpers.
    """
    errors: list[str] = []
    if not isinstance(checkpoint, dict):
        errors.append("checkpoint must be a dict.")
    else:
        for key in REQUIRED_CHECKPOINT_METADATA_KEYS:
            if key not in checkpoint:
                errors.append(f"missing required key: {key}")

        model = checkpoint.get("model")
        if "model" in checkpoint and not isinstance(model, dict):
            errors.append("model must be a state_dict dictionary.")

        if checkpoint.get("schema_version") != SCHEMA_VERSION:
            errors.append(
                f"schema_version must be {SCHEMA_VERSION!r}, got "
                f"{checkpoint.get('schema_version')!r}."
            )

        libreyolo_version = checkpoint.get("libreyolo_version")
        if "libreyolo_version" in checkpoint and not (
            isinstance(libreyolo_version, str) and libreyolo_version
        ):
            errors.append("libreyolo_version must be a non-empty string.")

        model_family = checkpoint.get("model_family")
        if "model_family" in checkpoint and not (
            isinstance(model_family, str) and model_family
        ):
            errors.append("model_family must be a non-empty string.")

        size = checkpoint.get("size")
        if "size" in checkpoint and not (isinstance(size, str) and size):
            errors.append("size must be a non-empty string.")

        raw_task = checkpoint.get("task")
        try:
            task = normalize_task(raw_task)
            if task is None:
                errors.append("task is required.")
            elif raw_task != task:
                errors.append(
                    f"task must use canonical identifier {task!r}, got "
                    f"{raw_task!r}."
                )
        except ValueError as exc:
            task = None
            errors.append(str(exc))

        artifact = None
        if (
            isinstance(model_family, str)
            and model_family
            and isinstance(size, str)
            and size
            and task is not None
        ):
            from ..models.manifest import get_artifact_spec, get_family_spec

            family_spec = get_family_spec(model_family)
            if family_spec is None:
                errors.append(
                    f"model_family {model_family!r} is not declared in the "
                    "public model manifest."
                )
            else:
                if model_family != family_spec.family:
                    errors.append(
                        "model_family must use canonical identifier "
                        f"{family_spec.family!r}, got {model_family!r}."
                    )
                artifact = get_artifact_spec(model_family, size, task)
                if artifact is not None and size != artifact.size:
                    errors.append(
                        f"size must use canonical identifier {artifact.size!r}, "
                        f"got {size!r}."
                    )
            if family_spec is not None and artifact is None:
                errors.append(
                    "model_family/size/task must identify a declared model "
                    f"artifact, got {model_family!r}/{size!r}/{task!r}."
                )

        nc = checkpoint.get("nc")
        if not isinstance(nc, int) or isinstance(nc, bool) or nc <= 0:
            errors.append("nc must be a positive int.")
            nc_for_names = None
        else:
            nc_for_names = nc

        if "names" in checkpoint and nc_for_names is not None:
            try:
                normalize_checkpoint_names(
                    checkpoint["names"],
                    nc_for_names,
                    allow_sparse=False,
                )
            except CheckpointMetadataError as exc:
                errors.append(str(exc))

        imgsz = checkpoint.get("imgsz")
        if not isinstance(imgsz, int) or isinstance(imgsz, bool) or imgsz <= 0:
            errors.append("imgsz must be a positive int.")
        elif artifact is not None:
            imgsz_error = _checkpoint_imgsz_contract_error(
                model_family=model_family,
                size=size,
                task=task,
                imgsz=imgsz,
            )
            if imgsz_error is not None:
                errors.append(imgsz_error)

    if strict and errors:
        raise CheckpointMetadataError("; ".join(errors))
    return errors


def parse_checkpoint_metadata_for_load(
    checkpoint: Any,
    *,
    context: str = "checkpoint",
) -> tuple[dict[str, Any], bool]:
    """Parse checkpoint metadata without downgrading a malformed v1 wrapper.

    A checkpoint with an explicit ``schema_version`` is held to the complete v1
    contract, including rejection of unsupported schema versions.  A pre-schema
    LibreYOLO checkpoint may carry ``libreyolo_version`` without declaring a
    schema; that marker alone remains legacy metadata.  Metadata on older/raw
    checkpoints remains reader-compatible: it is warned about and sparse class
    names are padded, but it never makes a legacy checkpoint count as an exact
    native checkpoint.

    Returns a normalized copy of the top-level checkpoint and whether it is a
    validated native v1 wrapper.  Tensor dictionaries nested below ``model`` or
    ``state_dict`` are retained by reference.
    """
    if not isinstance(checkpoint, dict):
        raise CheckpointMetadataError(f"{context} must be a dictionary.")

    parsed = dict(checkpoint)
    claims_native_v1 = "schema_version" in checkpoint
    if claims_native_v1:
        validate_checkpoint_metadata(checkpoint, strict=True)
        parsed["task"] = normalize_task(checkpoint["task"])
        parsed["names"] = normalize_checkpoint_names(
            checkpoint["names"],
            checkpoint["nc"],
            allow_sparse=False,
        )
        return parsed, True

    metadata_keys = set(REQUIRED_CHECKPOINT_METADATA_KEYS) - {"model"}
    present_metadata = sorted(metadata_keys & set(checkpoint))
    if present_metadata:
        warnings.warn(
            f"{context} has legacy or incomplete metadata "
            f"({', '.join(present_metadata)}); loading through the legacy "
            "compatibility path.",
            RuntimeWarning,
            stacklevel=2,
        )

    nc = checkpoint.get("nc")
    names = checkpoint.get("names")
    if (
        names is not None
        and isinstance(nc, int)
        and not isinstance(nc, bool)
        and nc > 0
    ):
        parsed["names"] = normalize_checkpoint_names(
            names,
            nc,
            allow_sparse=True,
        )
    task = checkpoint.get("task")
    if task is not None:
        parsed["task"] = normalize_task(task)
    return parsed, False


def warn_on_metadata_schema_version(
    metadata: Any,
    *,
    artifact: str,
    logger: Any,
) -> None:
    """Warn when exported runtime metadata is legacy or from another schema."""
    if not isinstance(metadata, dict) or not metadata:
        return

    schema_version = metadata.get("schema_version")
    if schema_version is None:
        logger.warning(
            "%s metadata has no schema_version; treating it as legacy metadata.",
            artifact,
        )
        return

    if str(schema_version) != SCHEMA_VERSION:
        logger.warning(
            "%s metadata schema_version %r differs from supported %r.",
            artifact,
            schema_version,
            SCHEMA_VERSION,
        )


def is_libreyolo_checkpoint(checkpoint: Any) -> bool:
    """Return whether a loaded object carries complete LibreYOLO v1.0 metadata."""
    return not validate_checkpoint_metadata(checkpoint, strict=False)


def unwrap_libreyolo_checkpoint(
    loaded: Any,
    *,
    strict: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return ``(state_dict, metadata)`` from a LibreYOLO checkpoint wrapper."""
    if strict:
        validate_checkpoint_metadata(loaded, strict=True)

    if isinstance(loaded, dict) and isinstance(loaded.get("model"), dict):
        metadata = {k: v for k, v in loaded.items() if k != "model"}
        return loaded["model"], metadata

    if strict:
        raise CheckpointMetadataError("checkpoint does not contain a model state_dict.")
    if isinstance(loaded, dict):
        return loaded, {}
    raise CheckpointMetadataError("checkpoint must be a dict.")
