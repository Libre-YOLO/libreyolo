"""The VLM fine-tune checkpoint contract.

A VLM checkpoint is a directory, not a ``.pt``: the PEFT adapter tensors, the
processor snapshot, and ``libreyolo_vlm.json``, the contract file that makes
the directory loadable by ``LibreVLM(path)``. Base weights are never copied
into a checkpoint; the contract records which base repo (and pinned revision)
to resolve, so checkpoints stay megabytes and LibreYOLO never redistributes
upstream weights.

``libreyolo_vlm.json`` fields (schema 1):

- ``schema``: contract version, integer.
- ``family`` / ``size``: adapter class resolution keys.
- ``base_repo`` / ``base_revision``: the exact base the adapter was trained on.
- ``names``: ordered training vocabulary; pre-applied on load.
- ``bbox_key`` / ``coord_divisor`` / ``box_format`` / ``prompt``: the output
  convention the fine-tune was trained to emit, restored when it is loaded.
- ``task``: always ``detect`` today.
- ``metrics``: final metrics of the producing run.
- ``libreyolo_version``: producer version string.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONTRACT_FILENAME = "libreyolo_vlm.json"
CONTRACT_SCHEMA = 1
_REQUIRED_FIELDS = (
    "family",
    "size",
    "base_repo",
    "base_revision",
    "names",
    "bbox_key",
    "coord_divisor",
    "box_format",
    "prompt",
    "task",
)
_BOX_FORMATS = {"xyxy", "xywh", "cxcywh", "yxyx"}
_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")

__all__ = [
    "CONTRACT_FILENAME",
    "is_vlm_checkpoint",
    "read_contract",
    "save_vlm_checkpoint",
    "validate_vlm_checkpoint_artifact",
    "validate_lora_artifact",
]


def _finite_metric_value(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(converted)


def is_vlm_checkpoint(path) -> bool:
    """True when ``path`` is a directory carrying the VLM checkpoint contract."""
    try:
        p = Path(path)
    except TypeError:
        return False
    return p.is_dir() and (p / CONTRACT_FILENAME).is_file()


def read_contract(path) -> Dict[str, Any]:
    """Read and validate a schema-1 checkpoint contract."""
    contract_path = Path(path) / CONTRACT_FILENAME
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Unreadable VLM checkpoint contract {contract_path}: {exc}"
        ) from exc
    if not isinstance(contract, dict):
        raise ValueError(
            f"VLM checkpoint contract {contract_path} must contain a JSON object."
        )
    schema = contract.get("schema")
    if schema != CONTRACT_SCHEMA:
        raise ValueError(
            f"VLM checkpoint {path} uses contract schema {schema!r}; this "
            f"LibreYOLO understands schema {CONTRACT_SCHEMA}. Upgrade libreyolo."
        )
    for key in _REQUIRED_FIELDS:
        if key not in contract:
            raise ValueError(
                f"VLM checkpoint contract {contract_path} is missing {key!r}."
            )

    for key in ("family", "size", "base_repo", "bbox_key", "prompt"):
        value = contract[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"VLM checkpoint contract {contract_path} field {key!r} "
                "must be a non-empty string."
            )

    base_revision = contract["base_revision"]
    if base_revision is not None and (
        not isinstance(base_revision, str) or not _COMMIT_SHA.fullmatch(base_revision)
    ):
        raise ValueError(
            f"VLM checkpoint contract {contract_path} field 'base_revision' "
            "must be null or an immutable 40-character commit SHA."
        )

    names = contract["names"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
    ):
        raise ValueError(
            f"VLM checkpoint contract {contract_path} field 'names' must be "
            "a non-empty list of non-empty strings."
        )
    normalized_names = [name.strip().lower() for name in names]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError(
            f"VLM checkpoint contract {contract_path} field 'names' must be "
            "unique case-insensitively."
        )

    coord_divisor = contract["coord_divisor"]
    if not _finite_metric_value(coord_divisor) or coord_divisor <= 0:
        raise ValueError(
            f"VLM checkpoint contract {contract_path} field 'coord_divisor' "
            "must be a finite positive number."
        )

    box_format = contract["box_format"]
    if not isinstance(box_format, str) or box_format not in _BOX_FORMATS:
        raise ValueError(
            f"VLM checkpoint contract {contract_path} field 'box_format' must "
            f"be one of {sorted(_BOX_FORMATS)}, got {box_format!r}."
        )
    if contract["task"] != "detect":
        raise ValueError(
            f"VLM checkpoint contract {contract_path} field 'task' must be "
            f"'detect', got {contract['task']!r}."
        )
    metrics = contract.get("metrics", {})
    if not isinstance(metrics, dict) or any(
        not isinstance(key, str) or not key or not _finite_metric_value(value)
        for key, value in metrics.items()
    ):
        raise ValueError(
            f"VLM checkpoint contract {contract_path} field 'metrics' must map "
            "non-empty names to finite numbers."
        )
    return contract


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _read_weight_map(index_path: Path, *, kind: str) -> set[str]:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unreadable {kind} shard index {index_path}: {exc}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"{kind.title()} shard index {index_path} has no weight_map object."
        )
    raw_shard_names = list(weight_map.values())
    if any(
        not isinstance(name, str) or not name or Path(name).name != name
        for name in raw_shard_names
    ):
        raise ValueError(
            f"{kind.title()} shard index {index_path} contains an invalid path."
        )
    return set(raw_shard_names)


def _validate_indexed_payload(directory: Path, index_path: Path, *, kind: str) -> None:
    shard_names = _read_weight_map(index_path, kind=kind)
    missing = sorted(
        name for name in shard_names if not _nonempty_file(directory / name)
    )
    if missing:
        raise ValueError(
            f"{kind.title()} shard index {index_path} references missing or empty "
            f"shards: {missing[:3]}."
        )


def _validate_adapter_payload(directory: Path) -> None:
    singles = [
        path
        for path in (
            directory / "adapter_model.safetensors",
            directory / "adapter_model.bin",
        )
        if path.exists()
    ]
    indexes = [
        path
        for path in (
            directory / "adapter_model.safetensors.index.json",
            directory / "adapter_model.bin.index.json",
        )
        if path.exists()
    ]
    loose_shards = [
        path
        for pattern in ("adapter_model-*.safetensors", "adapter_model-*.bin")
        for path in directory.glob(pattern)
    ]
    if singles:
        if len(singles) != 1 or indexes or loose_shards:
            raise ValueError(
                f"VLM adapter checkpoint {directory} has ambiguous tensor payloads."
            )
        if not _nonempty_file(singles[0]):
            raise ValueError(
                f"VLM adapter checkpoint {directory} has an empty adapter tensor payload."
            )
        return
    if indexes or loose_shards:
        raise ValueError(
            f"VLM adapter checkpoint {directory} uses a sharded tensor payload, "
            "which the supported PEFT loader cannot load. Save one non-empty "
            "adapter_model.safetensors or adapter_model.bin file."
        )
    raise ValueError(
        f"VLM adapter checkpoint {directory} has no adapter tensor payload."
    )


def _validate_full_model_payload(directory: Path) -> None:
    singles = [
        path
        for path in (directory / "model.safetensors", directory / "pytorch_model.bin")
        if path.exists()
    ]

    indexes = [
        path
        for path in (
            directory / "model.safetensors.index.json",
            directory / "pytorch_model.bin.index.json",
        )
        if path.is_file()
    ]
    loose_shards = [
        path
        for pattern in ("model-*.safetensors", "pytorch_model-*.bin")
        for path in directory.glob(pattern)
    ]
    if singles:
        if len(singles) != 1 or indexes or loose_shards:
            raise ValueError(
                f"VLM full-model checkpoint {directory} has ambiguous tensor payloads."
            )
        if not _nonempty_file(singles[0]):
            raise ValueError(
                f"VLM full-model checkpoint {directory} has an empty model payload."
            )
        return
    if len(indexes) != 1:
        raise ValueError(
            f"VLM full-model checkpoint {directory} requires one non-empty "
            "model payload or one shard index."
        )
    index_path = indexes[0]
    _validate_indexed_payload(directory, index_path, kind="model")
    indexed = _read_weight_map(index_path, kind="model")
    unexpected = sorted(path.name for path in loose_shards if path.name not in indexed)
    if unexpected:
        raise ValueError(
            f"VLM full-model checkpoint {directory} has unindexed model shards: "
            f"{unexpected[:3]}."
        )


def validate_lora_artifact(directory) -> None:
    """Require a parseable PEFT config and an adapter tensor payload."""
    directory = Path(directory)
    config_path = directory / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"VLM adapter checkpoint {directory} has an unreadable "
            f"adapter_config.json: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError(
            f"VLM adapter checkpoint {directory} adapter_config.json must "
            "contain a JSON object."
        )
    if config.get("peft_type") != "LORA":
        raise ValueError(
            f"VLM adapter checkpoint {directory} must declare "
            "peft_type='LORA' in adapter_config.json."
        )
    _validate_adapter_payload(directory)


def validate_vlm_checkpoint_artifact(directory) -> str:
    """Require exactly one complete adapter or full-model representation."""
    directory = Path(directory)
    adapter_marker = (directory / "adapter_config.json").exists()
    full_marker = (directory / "config.json").exists()
    if adapter_marker and full_marker:
        raise ValueError(
            "VLM checkpoint save produced both adapter and full-model markers."
        )
    if adapter_marker:
        validate_lora_artifact(directory)
        return "adapter"
    if full_marker:
        config_path = directory / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"VLM full-model checkpoint {directory} has an unreadable "
                f"config.json: {exc}"
            ) from exc
        if not isinstance(config, dict):
            raise ValueError(
                f"VLM full-model checkpoint {directory} config.json must "
                "contain a JSON object."
            )
        _validate_full_model_payload(directory)
        return "full"
    raise ValueError(
        "VLM checkpoint save produced neither a complete LoRA adapter nor a "
        "full-model tensor payload."
    )


def save_vlm_checkpoint(
    directory,
    *,
    peft_model,
    processor,
    wrapper,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Stage and replace one checkpoint directory with one representation."""
    metric_values = dict(metrics or {})
    if any(
        not isinstance(key, str) or not key or not _finite_metric_value(value)
        for key, value in metric_values.items()
    ):
        raise ValueError("VLM checkpoint metrics must be finite numeric values.")
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError(f"Refusing to replace symlinked VLM checkpoint {directory}.")
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"VLM checkpoint target {directory} is not a directory.")

    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.staging-", dir=directory.parent)
    )
    staging = staging_root / "checkpoint"
    staging.mkdir()

    try:
        peft_model.save_pretrained(str(staging))
        representation = validate_vlm_checkpoint_artifact(staging)
        processor.save_pretrained(str(staging))
        if validate_vlm_checkpoint_artifact(staging) != representation:
            raise ValueError(
                "Processor save changed the VLM checkpoint representation."
            )

        try:
            from libreyolo import __version__ as libreyolo_version
        except ImportError:  # pragma: no cover
            libreyolo_version = "unknown"

        contract = {
            "schema": CONTRACT_SCHEMA,
            "family": wrapper.FAMILY,
            "size": wrapper.size,
            "base_repo": wrapper.HF_REPOS[wrapper.size],
            "base_revision": wrapper.HF_REVISIONS.get(wrapper.size),
            "names": [wrapper.names[i] for i in range(len(wrapper.names))],
            "bbox_key": wrapper.BBOX_KEY,
            "coord_divisor": float(wrapper.COORD_DIVISOR),
            "box_format": wrapper.BOX_FORMAT,
            "prompt": wrapper._detection_prompt(),
            "task": "detect",
            "metrics": metric_values,
            "libreyolo_version": libreyolo_version,
        }
        (staging / CONTRACT_FILENAME).write_text(
            json.dumps(contract, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        read_contract(staging)

        backup = directory.with_name(f".{directory.name}.backup-{uuid.uuid4().hex}")
        had_existing = directory.exists()
        if had_existing:
            os.replace(directory, backup)
        try:
            os.replace(staging, directory)
        except Exception:
            if had_existing:
                try:
                    os.replace(backup, directory)
                except Exception as rollback_exc:
                    raise RuntimeError(
                        "VLM checkpoint publication failed and rollback could not "
                        f"restore {directory}; the previous checkpoint remains at "
                        f"{backup}."
                    ) from rollback_exc
            raise
        if had_existing:
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                logger.warning(
                    "Saved VLM checkpoint but could not remove backup %s: %s",
                    backup,
                    exc,
                )
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
    return directory
