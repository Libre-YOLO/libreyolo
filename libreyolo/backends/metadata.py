"""Shared parsing for LibreYOLO exported-runtime metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..tasks import normalize_supported_tasks, normalize_task
from ..utils.serialization import (
    SCHEMA_VERSION,
    CheckpointMetadataError,
    build_class_names,
    normalize_checkpoint_names,
)
from .base import (
    ImageSize,
    _read_metadata_imgsz,
    _read_pose_metadata,
    _read_runtime_metadata,
)


class ExportMetadataError(ValueError):
    """Raised when exported-runtime metadata is missing or malformed."""


@dataclass(frozen=True)
class ExportMetadata:
    """Normalized metadata needed by inference backends."""

    model_family: str | None
    model_size: str | None
    task: str
    supported_tasks: tuple[str, ...]
    default_task: str
    nb_classes: int
    names: dict[int, str]
    imgsz: ImageSize | None
    embedded_nms: bool
    pose: dict[str, Any]
    runtime: dict[str, Any]


def _read_nonempty_string(meta: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _read_names(raw_names: Any, nc: int, *, strict: bool) -> dict[int, str]:
    if isinstance(raw_names, str):
        try:
            raw_names = json.loads(raw_names)
        except json.JSONDecodeError as exc:
            raise ExportMetadataError("names must contain valid JSON.") from exc

    try:
        names = normalize_checkpoint_names(raw_names, nc)
    except CheckpointMetadataError as exc:
        raise ExportMetadataError(str(exc)) from exc

    if strict:
        source_keys = (
            set(range(len(raw_names)))
            if isinstance(raw_names, list)
            else {int(key) for key in raw_names}
        )
        expected_keys = set(range(nc))
        if source_keys != expected_keys:
            raise ExportMetadataError(
                "names must define every class index 0..nc-1 for remote inference."
            )
    return names


def parse_export_metadata(
    metadata: dict[str, Any],
    *,
    artifact: str,
    default_nb_classes: int = 80,
    strict: bool = False,
) -> ExportMetadata:
    """Normalize flat metadata emitted by LibreYOLO runtime exporters.

    ``strict=True`` is intended for remote artifacts. A remote backend cannot
    safely infer a family or class schema from a filename, so the core runtime
    fields must be explicit.
    """
    if not isinstance(metadata, dict):
        raise ExportMetadataError(f"{artifact} must be a JSON object.")

    if strict:
        missing = []
        if not _read_nonempty_string(metadata, "schema_version"):
            missing.append("schema_version")
        if not _read_nonempty_string(metadata, "libreyolo_version"):
            missing.append("libreyolo_version")
        if not _read_nonempty_string(metadata, "model_family"):
            missing.append("model_family")
        if not _read_nonempty_string(metadata, "size", "model_size"):
            missing.append("size")
        if not _read_nonempty_string(metadata, "task"):
            missing.append("task")
        if metadata.get("nc", metadata.get("nb_classes")) is None:
            missing.append("nc")
        if metadata.get("names") is None:
            missing.append("names")
        if metadata.get("imgsz") is None and not (
            metadata.get("imgsz_h") is not None and metadata.get("imgsz_w") is not None
        ):
            missing.append("imgsz")
        if missing:
            raise ExportMetadataError(
                f"{artifact} is missing required LibreYOLO metadata: "
                + ", ".join(missing)
                + "."
            )
        if str(metadata["schema_version"]) != SCHEMA_VERSION:
            raise ExportMetadataError(
                f"{artifact} uses unsupported schema_version "
                f"{metadata['schema_version']!r}; expected {SCHEMA_VERSION!r}."
            )

    model_family = _read_nonempty_string(metadata, "model_family")
    model_size = _read_nonempty_string(metadata, "size", "model_size")
    default_task = normalize_task(
        metadata.get("default_task"),
        default="detect",
    )
    task = normalize_task(metadata.get("task"), default=default_task)
    if task is None:
        task = "detect"
    if metadata.get("task") is None and metadata.get("segmentation") == "true":
        task = "segment"
    supported_tasks = normalize_supported_tasks(
        metadata.get("supported_tasks", (task,))
    )
    if task not in supported_tasks:
        raise ExportMetadataError(
            f"{artifact} declares task={task!r}, but supported_tasks is "
            f"{supported_tasks!r}."
        )

    raw_nc = metadata.get("nc", metadata.get("nb_classes", default_nb_classes))
    try:
        nc = int(raw_nc)
    except (TypeError, ValueError) as exc:
        raise ExportMetadataError(f"{artifact} has invalid nc metadata.") from exc
    if nc <= 0:
        raise ExportMetadataError(f"{artifact} nc must be a positive integer.")

    if metadata.get("names") is None:
        names = build_class_names(nc)
    else:
        try:
            names = _read_names(metadata["names"], nc, strict=strict)
        except ExportMetadataError as exc:
            raise ExportMetadataError(
                f"{artifact} has invalid names metadata: {exc}"
            ) from exc

    imgsz = _read_metadata_imgsz(metadata, model_family, artifact=artifact)
    try:
        runtime = _read_runtime_metadata(metadata)
        pose = _read_pose_metadata(metadata)
    except (TypeError, ValueError, json.JSONDecodeError):
        if strict:
            raise
        # Legacy local artifacts historically ignored malformed optional
        # metadata while still loading the core family/shape contract.
        runtime = {"embedded_nms": str(metadata.get("nms", "")).lower() == "true"}
        pose = {}
    return ExportMetadata(
        model_family=model_family,
        model_size=model_size,
        task=task,
        supported_tasks=supported_tasks,
        default_task=default_task or "detect",
        nb_classes=nc,
        names=names,
        imgsz=imgsz,
        embedded_nms=bool(runtime["embedded_nms"]),
        pose=pose,
        runtime=runtime,
    )
