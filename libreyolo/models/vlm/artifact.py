"""Offline construction and validation of publishable LibreVLM artifacts.

The schema in this module deliberately covers one narrow, verified cohort:
Qwen3-VL 2B/4B LoRA detection adapters.  It is independent from Hub transport
and model loading.  In particular, importing this module does not import
Transformers, PEFT, Hugging Face Hub, or construct a model.

An artifact is a flat, immutable directory.  Its manifest inventories every
payload byte except the manifest itself (which cannot hash itself), while the
manifest aggregate binds the sorted inventory.  Publication evidence remains
a human-authored external JSON input; the builder validates and copies it but
never creates an approval.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO, Iterator
from urllib.parse import unquote_to_bytes, urlsplit

VLM_ARTIFACT_SCHEMA = "libreyolo.vlm-artifact.v1"
VLM_ARTIFACT_MANIFEST = "libreyolo_vlm_artifact.json"
VLM_ARTIFACT_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
VLM_ARTIFACT_MAX_PAYLOAD_BYTES = 384 * 1024 * 1024
PUBLICATION_EVIDENCE_SCHEMA = "libreyolo.vlm-publication-evidence.v2"
PUBLICATION_EVIDENCE_FILENAME = "publication_evidence.json"
VLM_BASE_SNAPSHOT_SCHEMA = "libreyolo.vlm-base-snapshot.v1"

_CONTRACT_FILENAME = "libreyolo_vlm.json"
_ADAPTER_CONFIG_FILENAME = "adapter_config.json"
_ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"
_APACHE_LICENSE_FILENAME = "LICENSE"
_NOTICE_FILENAME = "NOTICE"
_README_FILENAME = "README.md"
_GITATTRIBUTES_FILENAME = ".gitattributes"

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 2_000_000
_MAX_SAFE_INTEGER = (1 << 53) - 1
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_IDENTIFIER_RE = re.compile(r"^[0-9a-f]{32}$")
_VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+_-]{0,127}$")
_TOKEN_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z ._+:/-]{0,255}$")
_SPDX_TOKEN_RE = re.compile(
    r"(?:DocumentRef-[0-9A-Za-z.-]+:)?LicenseRef-[0-9A-Za-z.-]+"
    r"|[0-9A-Za-z][0-9A-Za-z.-]*\+?|AND|OR|WITH|\(|\)"
)
_SPDX_IDENTIFIER_RE = re.compile(
    r"^(?:(?:DocumentRef-[0-9A-Za-z.-]+:)?LicenseRef-[0-9A-Za-z.-]+"
    r"|[0-9A-Za-z][0-9A-Za-z.-]*\+?)$"
)
_HTTPS_HOST_RE = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_HTTPS_PATH_RE = re.compile(r"^(?:[0-9A-Za-z._~/-]|%[0-9A-F]{2})*$")
_UTC_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})T(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:\.\d{1,6})?Z$"
)
_LORA_TENSOR_RE = re.compile(
    r"^(?P<stem>base_model\.model\.model\.language_model\.layers\."
    r"(?P<layer>\d+)\.(?:self_attn\.(?P<attn>q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?P<mlp>gate_proj|up_proj|down_proj)))\.lora_(?P<side>[AB])\.weight$"
)

_PRODUCTION_QWEN_LORA_LAYOUT = {
    "2b": {"layers": 28, "hidden": 2048, "q": 2048, "kv": 1024, "intermediate": 6144},
    "4b": {"layers": 36, "hidden": 2560, "q": 4096, "kv": 1024, "intermediate": 9728},
}
_QWEN_LORA_LAYOUT = _PRODUCTION_QWEN_LORA_LAYOUT

_SUPPORTED_BASES = {
    "2b": (
        "Qwen/Qwen3-VL-2B-Instruct",
        "89644892e4d85e24eaac8bacfd4f463576704203",
    ),
    "4b": (
        "Qwen/Qwen3-VL-4B-Instruct",
        "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    ),
}
_PUBLICATION_DEPENDENCY_PINS = {
    "peft": "0.19.1",
    "transformers": "5.12.1",
}
_CONFIDENCE_REPORT_SCHEMA = "libreyolo.vlm-confidence-report.v2"
_CONFIDENCE_CONTEXT_SCHEMA = "libreyolo.vlm-confidence-benchmark-context.v3"
_CONFIDENCE_CHECKPOINT_SCHEMA = "libreyolo.vlm-confidence-checkpoint-identity.v1"
_CONFIDENCE_DATASET_SCHEMA = "libreyolo.vlm-confidence-benchmark-dataset.v1"
_CONFIDENCE_PARTITION_NAME = "holdout100"
_CONFIDENCE_PARTITION_ROLE = "fine_tune_validation"
_CONFIDENCE_PARTITION_START = 0
_CONFIDENCE_PARTITION_STOP = 100
_CONFIDENCE_PARTITION_IMAGE_COUNT = 100
_CONFIDENCE_PARTITION_ARTIFACT = "annotations/instances_val2017_holdout100.json"
_CONFIDENCE_BENCHMARK_ID = ":".join(
    (
        _CONFIDENCE_REPORT_SCHEMA,
        _CONFIDENCE_CONTEXT_SCHEMA,
        _CONFIDENCE_PARTITION_NAME,
        _CONFIDENCE_PARTITION_ROLE,
    )
)
_EVALUATION_CLAIM_SCHEMA = "libreyolo.vlm-publication-evaluation-claim.v2"
_REPEATABILITY_RECEIPT_SCHEMA = "libreyolo.vlm-confidence-repeatability-receipt.v1"
_REPEATABILITY_CLAIM_SCHEMA = "libreyolo.vlm-confidence-repeatability-claim.v1"
_CONFIDENCE_CONTEXT_KEYS = {
    "schema",
    "git",
    "runtime",
    "determinism",
    "dataset",
    "checkpoint",
}
_CONFIDENCE_DATASET_KEYS = {
    "schema",
    "manifest",
    "source",
    "partition",
    "classes",
    "review",
}
_CONFIDENCE_PARTITION_KEYS = {
    "name",
    "role",
    "start",
    "stop",
    "image_count",
    "annotation_artifact",
    "annotation_size_bytes",
    "annotation_sha256",
}
_CONFIDENCE_CHECKPOINT_KEYS = {
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
}
_CONFIDENCE_VALIDATION_METRICS = {
    "metrics/vlm_confidence/auroc",
    "metrics/vlm_confidence/candidate_mAP50",
    "metrics/vlm_confidence/candidate_mAP50-95",
    "metrics/vlm_confidence/constant_mAP50",
    "metrics/vlm_confidence/constant_mAP50-95",
    "metrics/vlm_confidence/default_conf_fp_retention",
    "metrics/vlm_confidence/default_conf_prediction_retention",
    "metrics/vlm_confidence/default_conf_tp_retention",
    "metrics/vlm_confidence/delta_mAP50",
    "metrics/vlm_confidence/delta_mAP50-95",
    "metrics/vlm_confidence/detection_score_coverage",
    "metrics/vlm_confidence/prediction_score_coverage",
    "metrics/vlm_confidence/ranking_ap",
    "metrics/vlm_confidence/response_score_coverage",
    "metrics/vlm_confidence/scored_prediction_brier",
    "metrics/vlm_confidence/scored_prediction_ece",
    "metrics/vlm_confidence/scored_prediction_mce",
}
_CONFIDENCE_DELTA_METRICS = {
    "metrics/vlm_confidence/delta_mAP50": (
        "metrics/vlm_confidence/candidate_mAP50",
        "metrics/vlm_confidence/constant_mAP50",
    ),
    "metrics/vlm_confidence/delta_mAP50-95": (
        "metrics/vlm_confidence/candidate_mAP50-95",
        "metrics/vlm_confidence/constant_mAP50-95",
    ),
}
_CONFIDENCE_PROBABILITY_METRICS = _CONFIDENCE_VALIDATION_METRICS - set(
    _CONFIDENCE_DELTA_METRICS
)

# Exact file metadata at the immutable upstream revisions above. Small-file
# SHA-256 values are hashes of the downloaded bytes; safetensors hashes are the
# Git LFS object SHA-256 values returned by the official Hugging Face API.
_CANONICAL_BASE_FILES = {
    "2b": (
        (
            ".gitattributes",
            1_519,
            "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
        ),
        (
            "chat_template.json",
            5_502,
            "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
        ),
        (
            "config.json",
            1_505,
            "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
        ),
        (
            "generation_config.json",
            269,
            "1e241830b48b397cb0900101421df5450baddc7adf01e5fc86b5615865f3bae4",
        ),
        (
            "merges.txt",
            1_671_839,
            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        ),
        (
            "model.safetensors",
            4_255_140_312,
            "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
        ),
        (
            "preprocessor_config.json",
            390,
            "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        ),
        (
            "README.md",
            7_136,
            "5fc5be1ca9a3910399bd6239ee5086ab5d82a2a59c5d2b00e887a8835cc110e4",
        ),
        (
            "tokenizer.json",
            7_032_403,
            "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
        ),
        (
            "tokenizer_config.json",
            10_868,
            "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
        ),
        (
            "video_preprocessor_config.json",
            385,
            "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
        ),
        (
            "vocab.json",
            2_776_833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
    ),
    "4b": (
        (
            ".gitattributes",
            1_519,
            "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
        ),
        (
            "chat_template.json",
            5_502,
            "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
        ),
        (
            "config.json",
            1_505,
            "edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d",
        ),
        (
            "generation_config.json",
            269,
            "8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44",
        ),
        (
            "merges.txt",
            1_671_839,
            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        ),
        (
            "model-00001-of-00002.safetensors",
            4_967_229_296,
            "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
        ),
        (
            "model-00002-of-00002.safetensors",
            3_908_490_048,
            "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
        ),
        (
            "model.safetensors.index.json",
            64_742,
            "58a7841d7bff2548dd91577d216274a83cf1b500bc6a534b809d6c1b1707cf2b",
        ),
        (
            "preprocessor_config.json",
            390,
            "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        ),
        (
            "README.md",
            7_133,
            "a884e5e78f7d6f7bfe237f909dbc41a126542e259dc79d8ab33cc8980580ff79",
        ),
        (
            "tokenizer.json",
            7_032_403,
            "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
        ),
        (
            "tokenizer_config.json",
            10_868,
            "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
        ),
        (
            "video_preprocessor_config.json",
            385,
            "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
        ),
        (
            "vocab.json",
            2_776_833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
    ),
}

# Processor serialization produced by Transformers 5.12.1 after the Qwen
# training collator fixes left padding, verified identical for best/last and
# applicable to both pinned sizes (their upstream processor inputs are equal).
_TRAINED_PROCESSOR_FILES = (
    (
        "chat_template.jinja",
        5_412,
        "24a1eb036569714fc3efe7908495159c19ac5138f652c9e524475e40ce87d716",
    ),
    (
        "processor_config.json",
        1_251,
        "f196d5698d1771c734bb3a24bd658ba75536fc4feafc5b83c035b7693511a2db",
    ),
    (
        "tokenizer.json",
        11_422_818,
        "8579e1ca7cc5d82a9e0202eed555529996f4ffe7f563c2979a0290cf3db452d3",
    ),
    (
        "tokenizer_config.json",
        765,
        "74ebcde921b7bcd0144e9d121243afa7894463dd5db77452fc99c65dbeae7ee3",
    ),
)
_CANONICAL_PROCESSOR_FILES = {
    "2b": _TRAINED_PROCESSOR_FILES,
    "4b": _TRAINED_PROCESSOR_FILES,
}

_PROCESSOR_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}
_ARTIFACT_FILE_LIMITS = {
    _CONTRACT_FILENAME: 1024 * 1024,
    _ADAPTER_CONFIG_FILENAME: 1024 * 1024,
    _ADAPTER_WEIGHTS_FILENAME: 256 * 1024 * 1024,
    "added_tokens.json": 4 * 1024 * 1024,
    "chat_template.jinja": 256 * 1024,
    "chat_template.json": 256 * 1024,
    "merges.txt": 8 * 1024 * 1024,
    "preprocessor_config.json": 4 * 1024 * 1024,
    "processor_config.json": 4 * 1024 * 1024,
    "special_tokens_map.json": 4 * 1024 * 1024,
    "tokenizer.json": 32 * 1024 * 1024,
    "tokenizer_config.json": 4 * 1024 * 1024,
    "video_preprocessor_config.json": 4 * 1024 * 1024,
    "vocab.json": 16 * 1024 * 1024,
    _APACHE_LICENSE_FILENAME: 32 * 1024,
    _NOTICE_FILENAME: 1024 * 1024,
    _README_FILENAME: 1024 * 1024,
    PUBLICATION_EVIDENCE_FILENAME: 4 * 1024 * 1024,
    _GITATTRIBUTES_FILENAME: 4 * 1024,
}
_REQUIRED_INPUT_FILES = {
    _CONTRACT_FILENAME,
    _ADAPTER_CONFIG_FILENAME,
    _ADAPTER_WEIGHTS_FILENAME,
    "tokenizer_config.json",
}
_SOURCE_ONLY_FILES = {_README_FILENAME}
_GENERATED_FILES = {
    _APACHE_LICENSE_FILENAME,
    _NOTICE_FILENAME,
    _README_FILENAME,
    PUBLICATION_EVIDENCE_FILENAME,
    _GITATTRIBUTES_FILENAME,
}
_ROLE_BY_FIXED_PATH = {
    _CONTRACT_FILENAME: "checkpoint_contract",
    _ADAPTER_CONFIG_FILENAME: "adapter_config",
    _ADAPTER_WEIGHTS_FILENAME: "adapter_weights",
    _APACHE_LICENSE_FILENAME: "license",
    _NOTICE_FILENAME: "notice",
    _README_FILENAME: "model_card",
    PUBLICATION_EVIDENCE_FILENAME: "publication_evidence",
    _GITATTRIBUTES_FILENAME: "hub_config",
}
_REQUIRED_SINGLE_ROLES = set(_ROLE_BY_FIXED_PATH.values())
_ALL_ROLES = _REQUIRED_SINGLE_ROLES | {"processor"}

_CONTRACT_KEYS = {
    "schema",
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
    "metrics",
    "libreyolo_version",
}
_EVIDENCE_KEYS = {
    "schema",
    "artifact_license",
    "base_model",
    "training_data",
    "evaluation",
    "code",
    "review",
}
_ARTIFACT_LICENSE_KEYS = {"spdx", "redistribution_decision"}
_BASE_MODEL_KEYS = {
    "repo",
    "revision",
    "license_spdx",
    "license_evidence_url",
    "weights_redistribution_decision",
    "processor_redistribution_decision",
    "snapshot",
}
_SNAPSHOT_KEYS = {
    "schema",
    "source",
    "revision",
    "files",
    "aggregate_sha256",
    "sha256",
}
_SNAPSHOT_FILE_KEYS = {"path", "size", "sha256"}
_TRAINING_DATA_KEYS = {
    "source",
    "version",
    "split",
    "license_spdx",
    "license_evidence_url",
    "manifest_sha256",
    "redistribution_decision",
}
_EVALUATION_KEYS = {
    "benchmark",
    "report_sha256",
    "envelope_sha256",
    "checkpoint_sha256",
    "metrics",
    "repeatability",
    "passed",
}
_REPEATABILITY_KEYS = {
    "schema",
    "receipt_sha256",
    "comparison_sha256",
    "runs",
    "tolerances",
    "reproducible",
}
_REPEATABILITY_RUN_KEYS = {
    "run_id",
    "process_id",
    "report_sha256",
    "envelope_sha256",
}
_REPEATABILITY_TOLERANCE_KEYS = {"score_atol", "metric_atol", "map_atol"}
_CODE_KEYS = {"repository", "revision", "clean", "recipe", "dependencies"}
_RECIPE_KEYS = {"id", "sha256"}
_TEMPLATE_TRAINING_DATA_KEYS = _TRAINING_DATA_KEYS - {"redistribution_decision"}
_TEMPLATE_CODE_KEYS = {"revision", "clean", "dependencies"}
_REVIEW_KEYS = {"approved", "reviewer", "reviewed_at", "bindings", "gates"}
_BINDING_KEYS = {
    "base_snapshot_sha256",
    "training_data_manifest_sha256",
    "evaluation_report_sha256",
    "evaluation_envelope_sha256",
    "evaluation_repeatability_receipt_sha256",
    "evaluation_repeatability_comparison_sha256",
    "evaluation_claim_sha256",
    "code_revision",
    "recipe_sha256",
    "adapter_weights_sha256",
    "adapter_config_sha256",
    "checkpoint_contract_sha256",
    "processor_sha256",
}
_GATE_KEYS = {
    "artifact_license_approved",
    "base_model_verified",
    "training_data_approved",
    "privacy_approved",
    "evaluation_approved",
    "code_provenance_approved",
}
_MANIFEST_KEYS = {
    "schema",
    "representation",
    "identity",
    "files",
    "aggregate_sha256",
}
_IDENTITY_KEYS = {
    "family",
    "size",
    "task",
    "base_repo",
    "base_revision",
    "artifact_license",
    "checkpoint_contract_sha256",
    "publication_evidence_sha256",
    "processor_sha256",
    "weights_sha256",
    "base_snapshot",
}
_FILE_KEYS = {"path", "role", "size", "sha256"}
_PEFT_CONFIG_KEYS = {
    "alora_invocation_tokens",
    "alpha_pattern",
    "arrow_config",
    "auto_mapping",
    "base_model_name_or_path",
    "bias",
    "corda_config",
    "ensure_weight_tying",
    "eva_config",
    "exclude_modules",
    "fan_in_fan_out",
    "inference_mode",
    "init_lora_weights",
    "layer_replication",
    "layers_pattern",
    "layers_to_transform",
    "loftq_config",
    "lora_alpha",
    "lora_bias",
    "lora_dropout",
    "lora_ga_config",
    "megatron_config",
    "megatron_core",
    "modules_to_save",
    "peft_type",
    "peft_version",
    "qalora_group_size",
    "r",
    "rank_pattern",
    "revision",
    "target_modules",
    "target_parameters",
    "task_type",
    "trainable_token_indices",
    "use_bdlora",
    "use_dora",
    "use_qalora",
    "use_rslora",
}

_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


class VLMArtifactError(ValueError):
    """Raised when a VLM artifact or its publication evidence is invalid."""


@dataclass(frozen=True)
class VLMArtifactInfo:
    """Validated identity and exact payload inventory for one VLM artifact."""

    root: Path
    manifest: Mapping[str, Any]
    aggregate_sha256: str
    files: tuple[str, ...]
    base_snapshot: Mapping[str, Any]


__all__ = [
    "PUBLICATION_EVIDENCE_FILENAME",
    "PUBLICATION_EVIDENCE_SCHEMA",
    "VLM_ARTIFACT_MANIFEST",
    "VLM_ARTIFACT_MAX_MANIFEST_BYTES",
    "VLM_ARTIFACT_MAX_PAYLOAD_BYTES",
    "VLM_ARTIFACT_SCHEMA",
    "VLM_BASE_SNAPSHOT_SCHEMA",
    "VLMArtifactError",
    "VLMArtifactInfo",
    "build_vlm_artifact",
    "create_vlm_publication_evidence_template",
    "read_vlm_artifact_manifest",
    "validate_vlm_base_snapshot",
    "validate_vlm_artifact",
]


def _path_exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _path_argument(value: Any, label: str) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{label} must be a filesystem path")
    return Path(value).expanduser()


def _is_link_or_junction(path: Path) -> bool:
    try:
        identity = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:
        raise VLMArtifactError(f"Could not inspect path entry: {path}") from exc
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(identity.st_mode) or bool(
        getattr(identity, "st_file_attributes", 0) & reparse
    )


def _absolute_lexical(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _assert_unlinked_components(path: Path, label: str, *, leaf_exists: bool) -> None:
    lexical = _absolute_lexical(path)
    parts = lexical.parts
    current = Path(lexical.anchor)
    stop = len(parts) if leaf_exists else len(parts) - 1
    for part in parts[1:stop]:
        current /= part
        if not _path_exists(current):
            raise VLMArtifactError(f"{label} has a missing parent: {current}")
        if _is_link_or_junction(current):
            raise VLMArtifactError(
                f"{label} must not contain a symlink or junction: {current}"
            )


def _required_directory(value: Any, label: str) -> Path:
    lexical = _absolute_lexical(_path_argument(value, label))
    _assert_unlinked_components(lexical, label, leaf_exists=True)
    try:
        resolved = lexical.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise VLMArtifactError(
            f"{label} is not an existing directory: {lexical}"
        ) from exc
    if not resolved.is_dir():
        raise VLMArtifactError(f"{label} is not an existing directory: {resolved}")
    return resolved


def _required_file(value: Any, label: str) -> Path:
    lexical = _absolute_lexical(_path_argument(value, label))
    _assert_unlinked_components(lexical, label, leaf_exists=True)
    try:
        resolved = lexical.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise VLMArtifactError(f"{label} is not an existing file: {lexical}") from exc
    if not resolved.is_file():
        raise VLMArtifactError(f"{label} is not an existing file: {resolved}")
    _assert_regular_unlinked_file(resolved, label)
    return resolved


def _new_directory_destination(value: Any, label: str) -> Path:
    lexical = _absolute_lexical(_path_argument(value, label))
    _assert_unlinked_components(lexical, label, leaf_exists=False)
    if _path_exists(lexical):
        raise FileExistsError(f"{label} already exists: {lexical}")
    parent = _required_directory(lexical.parent, f"{label} parent")
    resolved = parent / lexical.name
    if _path_exists(resolved):
        raise FileExistsError(f"{label} already exists: {resolved}")
    return resolved


def _new_file_destination(value: Any, label: str) -> Path:
    lexical = _absolute_lexical(_path_argument(value, label))
    _assert_unlinked_components(lexical, label, leaf_exists=False)
    if _path_exists(lexical):
        raise FileExistsError(f"{label} already exists: {lexical}")
    parent = _required_directory(lexical.parent, f"{label} parent")
    resolved = parent / lexical.name
    if _path_exists(resolved):
        raise FileExistsError(f"{label} already exists: {resolved}")
    return resolved


def _assert_regular_unlinked_file(path: Path, label: str) -> os.stat_result:
    try:
        identity = os.lstat(path)
    except OSError as exc:
        raise VLMArtifactError(f"Could not inspect {label}: {path}") from exc
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(identity.st_mode) or bool(
        getattr(identity, "st_file_attributes", 0) & reparse
    ):
        raise VLMArtifactError(f"{label} must not be a symlink or junction: {path}")
    if not stat.S_ISREG(identity.st_mode):
        raise VLMArtifactError(f"{label} must be a regular file: {path}")
    if getattr(identity, "st_nlink", 1) != 1:
        raise VLMArtifactError(f"{label} must not be a hard-linked file: {path}")
    return identity


def _assert_disjoint(source: Path, destination: Path) -> None:
    resolved_destination = destination.resolve(strict=False)
    if (
        resolved_destination == source
        or resolved_destination in source.parents
        or source in resolved_destination.parents
    ):
        raise VLMArtifactError("checkpoint and artifact directories must be disjoint")


def _same_file_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )


@contextmanager
def _open_stable_regular_file(
    path: Path,
    label: str,
    *,
    max_bytes: int | None = None,
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    """Open one unlinked regular file and bind the handle to its path identity."""

    before = _assert_regular_unlinked_file(path, label)
    if max_bytes is not None and before.st_size > max_bytes:
        raise VLMArtifactError(f"{label} exceeds the {max_bytes}-byte safety limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or getattr(opened_before, "st_nlink", 1) != 1
            or not _same_file_identity(before, opened_before)
        ):
            raise VLMArtifactError(f"{label} changed before it was opened")
        stream = os.fdopen(descriptor, "rb")
        descriptor = None
    except OSError as exc:
        raise VLMArtifactError(f"Could not open {label}: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

    try:
        yield stream, opened_before
    finally:
        try:
            opened_after = os.fstat(stream.fileno())
        except OSError as exc:
            raise VLMArtifactError(f"Could not recheck {label}: {path}") from exc
        finally:
            stream.close()
        after = _assert_regular_unlinked_file(path, label)
        if not _same_file_identity(
            opened_before, opened_after
        ) or not _same_file_identity(opened_before, after):
            raise VLMArtifactError(f"{label} changed while it was being read")


def _read_bounded(path: Path, *, max_bytes: int, label: str) -> bytes:
    try:
        with _open_stable_regular_file(path, label, max_bytes=max_bytes) as (
            stream,
            opened,
        ):
            payload = stream.read(max_bytes + 1)
            stream.seek(0)
            verified_payload = stream.read(max_bytes + 1)
    except VLMArtifactError:
        raise
    except OSError as exc:
        raise VLMArtifactError(f"Could not read {label}: {path}") from exc
    if len(payload) > max_bytes:
        raise VLMArtifactError(f"{label} exceeds the {max_bytes}-byte safety limit")
    if len(payload) != opened.st_size:
        raise VLMArtifactError(f"{label} changed while it was being read")
    if verified_payload != payload:
        raise VLMArtifactError(f"{label} changed while it was being read")
    return payload


def _parse_int(value: str) -> int:
    number = int(value)
    if abs(number) > _MAX_SAFE_INTEGER:
        raise VLMArtifactError("JSON integer exceeds the exact safe range")
    return number


def _parse_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise VLMArtifactError("JSON number must be finite")
    return number


def _reject_constant(value: str) -> None:
    raise VLMArtifactError(f"JSON constant {value!r} is not permitted")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VLMArtifactError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _validate_json_tree(value: Any, label: str) -> None:
    stack = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise VLMArtifactError(
                f"{label} exceeds the {_MAX_JSON_NODES}-node safety limit"
            )
        if depth > _MAX_JSON_DEPTH:
            raise VLMArtifactError(
                f"{label} exceeds the {_MAX_JSON_DEPTH}-level nesting limit"
            )
        if isinstance(current, Mapping):
            if any(not isinstance(key, str) for key in current):
                raise VLMArtifactError(f"{label} object keys must be strings")
            stack.extend((nested, depth + 1) for nested in current.values())
        elif isinstance(current, list):
            stack.extend((nested, depth + 1) for nested in current)
        elif current is None or isinstance(current, (str, bool)):
            continue
        elif type(current) is int:
            if abs(current) > _MAX_SAFE_INTEGER:
                raise VLMArtifactError(f"{label} integer exceeds the exact safe range")
        elif isinstance(current, float):
            if not math.isfinite(current):
                raise VLMArtifactError(f"{label} numbers must be finite")
        else:
            raise VLMArtifactError(
                f"{label} contains unsupported {type(current).__name__} data"
            )


def _decode_json(payload: bytes, label: str) -> Any:
    if len(payload) > _MAX_JSON_BYTES:
        raise VLMArtifactError(f"{label} exceeds the JSON safety limit")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VLMArtifactError(f"{label} must be UTF-8 JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_parse_int,
            parse_float=_parse_float,
            parse_constant=_reject_constant,
        )
    except VLMArtifactError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise VLMArtifactError(f"{label} is not valid bounded JSON") from exc
    _validate_json_tree(value, label)
    return value


def _load_json(path: Path, label: str) -> Any:
    return _decode_json(
        _read_bounded(path, max_bytes=_MAX_JSON_BYTES, label=label), label
    )


def _canonical_json(value: Any) -> bytes:
    _validate_json_tree(value, "JSON value")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_file_bytes(value: Any) -> bytes:
    return _canonical_json(value) + b"\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VLMArtifactError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if unknown:
            detail.append(f"unknown {unknown}")
        raise VLMArtifactError(f"{label} has invalid keys: {', '.join(detail)}")
    return dict(value)


def _require_string(value: Any, label: str, *, max_length: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > max_length
        or unicodedata.normalize("NFC", value) != value
        or any(
            ord(char) == 127
            or unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
            or (char != " " and char.isspace())
            for char in value
        )
    ):
        raise VLMArtifactError(
            f"{label} must be a safe, normalized, nonblank, trimmed string"
        )
    return value


def _require_token(value: Any, label: str) -> str:
    text = _require_string(value, label, max_length=256)
    if not _TOKEN_RE.fullmatch(text):
        raise VLMArtifactError(f"{label} contains unsafe characters")
    return text


def _require_spdx(value: Any, label: str) -> str:
    text = _require_string(value, label, max_length=256)
    tokens: list[str] = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        match = _SPDX_TOKEN_RE.match(text, position)
        if match is None:
            raise VLMArtifactError(f"{label} is not a valid SPDX expression")
        tokens.append(match.group(0))
        position = match.end()

    index = 0

    def parse_primary() -> None:
        nonlocal index
        if index >= len(tokens):
            raise VLMArtifactError(f"{label} is not a valid SPDX expression")
        token = tokens[index]
        if token == "(":
            index += 1
            parse_or_expression()
            if index >= len(tokens) or tokens[index] != ")":
                raise VLMArtifactError(f"{label} is not a valid SPDX expression")
            index += 1
        elif (
            token in {"AND", "OR", "WITH", ")", "NONE", "NOASSERTION"}
            or _SPDX_IDENTIFIER_RE.fullmatch(token) is None
        ):
            raise VLMArtifactError(f"{label} is not a valid SPDX expression")
        else:
            index += 1
        if index < len(tokens) and tokens[index] == "WITH":
            index += 1
            if (
                index >= len(tokens)
                or tokens[index] in {"AND", "OR", "WITH", "(", ")"}
                or _SPDX_IDENTIFIER_RE.fullmatch(tokens[index]) is None
            ):
                raise VLMArtifactError(f"{label} is not a valid SPDX expression")
            index += 1

    def parse_and_expression() -> None:
        nonlocal index
        parse_primary()
        while index < len(tokens) and tokens[index] == "AND":
            index += 1
            parse_primary()

    def parse_or_expression() -> None:
        nonlocal index
        parse_and_expression()
        while index < len(tokens) and tokens[index] == "OR":
            index += 1
            parse_and_expression()

    parse_or_expression()
    if index != len(tokens):
        raise VLMArtifactError(f"{label} is not a valid SPDX expression")
    return text


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise VLMArtifactError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise VLMArtifactError(f"{label} must be a lowercase 40-character commit SHA")
    return value


def _require_https_url(value: Any, label: str) -> str:
    text = _require_string(value, label, max_length=2048)
    try:
        parsed = urlsplit(text)
        port = parsed.port
        decoded_path = unquote_to_bytes(parsed.path).decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise VLMArtifactError(
            f"{label} must be a canonical, safely percent-encoded HTTPS URL"
        ) from exc
    encoded_unreserved = any(
        chr(int(match.group(0)[1:], 16)).isalnum()
        or chr(int(match.group(0)[1:], 16)) in "._~-"
        for match in re.finditer(r"%[0-9A-F]{2}", parsed.path)
        if int(match.group(0)[1:], 16) < 128
    )
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port is not None
        or parsed.hostname is None
        or parsed.hostname != parsed.hostname.lower()
        or _HTTPS_HOST_RE.fullmatch(parsed.hostname) is None
        or parsed.netloc != parsed.hostname
        or _HTTPS_PATH_RE.fullmatch(parsed.path) is None
        or encoded_unreserved
        or unicodedata.normalize("NFC", decoded_path) != decoded_path
        or any(
            ord(char) == 127
            or unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
            or (char != " " and char.isspace())
            for char in decoded_path
        )
        or "//" in parsed.path
        or any(segment in {".", ".."} for segment in parsed.path.split("/"))
    ):
        raise VLMArtifactError(
            f"{label} must be a canonical, safely percent-encoded HTTPS URL"
        )
    return text


def _require_true(value: Any, label: str) -> None:
    if value is not True:
        raise VLMArtifactError(f"{label} must be true")


def _require_finite_metrics(value: Any, label: str) -> dict[str, float | int]:
    if not isinstance(value, dict) or not value:
        raise VLMArtifactError(f"{label} must be a nonempty JSON object")
    for key, metric in value.items():
        _require_string(key, f"{label} key", max_length=128)
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise VLMArtifactError(f"{label}.{key} must be a finite number")
        if not math.isfinite(float(metric)):
            raise VLMArtifactError(f"{label}.{key} must be a finite number")
    return value


def _safe_inventory_path(value: Any, label: str, *, flat: bool) -> str:
    text = _require_string(value, label, max_length=512)
    if "\\" in text or ":" in text or text.startswith("/"):
        raise VLMArtifactError(f"{label} is not a safe portable relative path")
    path = PurePosixPath(text)
    if text != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise VLMArtifactError(f"{label} is not a canonical relative path")
    if flat and len(path.parts) != 1:
        raise VLMArtifactError(f"{label} must be a flat artifact path")
    return text


def _aggregate_entries(entries: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json(list(entries)))


def _canonical_base_snapshot(size: str) -> dict[str, Any]:
    repo, revision = _SUPPORTED_BASES[size]
    files = [
        {"path": path, "size": size_bytes, "sha256": digest}
        for path, size_bytes, digest in _CANONICAL_BASE_FILES[size]
    ]
    aggregate = _aggregate_entries(files)
    payload = {
        "schema": VLM_BASE_SNAPSHOT_SCHEMA,
        "source": repo,
        "revision": revision,
        "files": files,
        "aggregate_sha256": aggregate,
    }
    return {**payload, "sha256": _sha256_bytes(_canonical_json(payload))}


def _validate_snapshot_files(
    value: Any,
    *,
    expected_repo: str | None = None,
    expected_revision: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    snapshot = _require_exact_keys(value, _SNAPSHOT_KEYS, "base_model.snapshot")
    if snapshot["schema"] != VLM_BASE_SNAPSHOT_SCHEMA:
        raise VLMArtifactError(
            f"base_model.snapshot.schema must be {VLM_BASE_SNAPSHOT_SCHEMA!r}"
        )
    source = _require_string(snapshot["source"], "base_model.snapshot.source")
    revision = _require_commit(snapshot["revision"], "base_model.snapshot.revision")
    if expected_repo is not None and source != expected_repo:
        raise VLMArtifactError("base snapshot source does not match base_model.repo")
    if expected_revision is not None and revision != expected_revision:
        raise VLMArtifactError(
            "base snapshot revision does not match base_model.revision"
        )
    files = snapshot["files"]
    if not isinstance(files, (list, tuple)) or not files:
        raise VLMArtifactError("base_model.snapshot.files must be a nonempty list")
    checked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(files):
        item = _require_exact_keys(
            raw, _SNAPSHOT_FILE_KEYS, f"base_model.snapshot.files[{index}]"
        )
        path = _safe_inventory_path(
            item["path"], f"snapshot file {index} path", flat=False
        )
        folded = path.casefold()
        if folded in seen:
            raise VLMArtifactError(
                "base snapshot file paths must be unique case-insensitively"
            )
        seen.add(folded)
        size = item["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise VLMArtifactError(f"base snapshot file {path} size must be positive")
        digest = _require_sha256(item["sha256"], f"base snapshot file {path} sha256")
        if path.lower().endswith((".bin", ".pt", ".pth", ".py", ".pyc")):
            raise VLMArtifactError(f"base snapshot file {path} is not publish-safe")
        checked.append({"path": path, "size": size, "sha256": digest})
    if [entry["path"] for entry in checked] != sorted(
        (entry["path"] for entry in checked), key=str.casefold
    ):
        raise VLMArtifactError("base snapshot file inventory must be sorted by path")
    paths = {entry["path"] for entry in checked}
    if (
        "config.json" not in paths
        or "preprocessor_config.json" not in paths
        or "tokenizer_config.json" not in paths
    ):
        raise VLMArtifactError(
            "base snapshot inventory must include config.json, "
            "preprocessor_config.json, and tokenizer_config.json"
        )
    if not any(path.endswith(".safetensors") for path in paths):
        raise VLMArtifactError(
            "base snapshot inventory must include safetensors weights"
        )
    aggregate = _aggregate_entries(checked)
    if snapshot["aggregate_sha256"] != aggregate:
        raise VLMArtifactError(
            "base snapshot aggregate_sha256 does not match its files"
        )
    identity_payload = {
        "schema": VLM_BASE_SNAPSHOT_SCHEMA,
        "source": source,
        "revision": revision,
        "files": checked,
        "aggregate_sha256": aggregate,
    }
    if snapshot["sha256"] != _sha256_bytes(_canonical_json(identity_payload)):
        raise VLMArtifactError("base snapshot sha256 does not match its identity")
    matching_sizes = [
        size
        for size, identity in _SUPPORTED_BASES.items()
        if identity == (source, revision)
    ]
    if len(matching_sizes) != 1:
        raise VLMArtifactError(
            "base snapshot is not a supported immutable Qwen3-VL pin"
        )
    canonical = _canonical_base_snapshot(matching_sizes[0])
    if (
        checked != canonical["files"]
        or aggregate != canonical["aggregate_sha256"]
        or snapshot["sha256"] != canonical["sha256"]
    ):
        raise VLMArtifactError(
            "base snapshot does not match the exact official file inventory"
        )
    return checked, aggregate


def _recipe_sha256() -> str:
    recipe = Path(__file__).with_name("training") / "recipes.py"
    payload = recipe.read_bytes().replace(b"\r\n", b"\n")
    return _sha256_bytes(payload)


def _plain_json_object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be a plain JSON object")
    try:
        payload = _canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise VLMArtifactError(f"{label} must contain only plain JSON values") from exc
    if len(payload) > _ARTIFACT_FILE_LIMITS[PUBLICATION_EVIDENCE_FILENAME]:
        raise VLMArtifactError(f"{label} exceeds the publication evidence safety limit")
    normalized = _decode_json(payload, label)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded above
        raise VLMArtifactError(f"{label} must be a JSON object")
    return normalized


def _validate_template_context(
    training_data_value: Any,
    code_value: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training_data = _require_exact_keys(
        _plain_json_object(training_data_value, "training_data"),
        _TEMPLATE_TRAINING_DATA_KEYS,
        "training_data",
    )
    _require_https_url(training_data["source"], "training_data.source")
    _require_token(training_data["version"], "training_data.version")
    _require_token(training_data["split"], "training_data.split")
    _require_spdx(training_data["license_spdx"], "training_data.license_spdx")
    _require_https_url(
        training_data["license_evidence_url"],
        "training_data.license_evidence_url",
    )
    _require_sha256(training_data["manifest_sha256"], "training_data.manifest_sha256")

    code = _require_exact_keys(
        _plain_json_object(code_value, "code"),
        _TEMPLATE_CODE_KEYS,
        "code",
    )
    _require_commit(code["revision"], "code.revision")
    if type(code["clean"]) is not bool:
        raise VLMArtifactError("code.clean must be a boolean")
    dependencies = code["dependencies"]
    expected_dependencies = {"libreyolo", "peft", "torch", "transformers"}
    if type(dependencies) is not dict or set(dependencies) != expected_dependencies:
        raise VLMArtifactError(
            f"code.dependencies must contain exactly {sorted(expected_dependencies)}"
        )
    for name, version in dependencies.items():
        if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
            raise VLMArtifactError(f"code.dependencies.{name} has an invalid version")
    for name, expected in _PUBLICATION_DEPENDENCY_PINS.items():
        if dependencies[name] != expected:
            raise VLMArtifactError(
                f"code.dependencies.{name} must be {expected!r} for VLM artifact v1"
            )
    return training_data, code


def _read_confidence_benchmark_identity(path: str | os.PathLike[str]) -> Any:
    """Read one strict report/envelope pair without an eager validation import."""

    try:
        from libreyolo.validation.vlm_confidence_benchmark import (
            VLMConfidenceReportError,
            read_benchmark_run_identity,
        )
    except ImportError as exc:  # pragma: no cover - package integrity guard
        raise VLMArtifactError(
            "confidence benchmark validation is unavailable in this LibreYOLO runtime"
        ) from exc
    try:
        return read_benchmark_run_identity(path, label="publication_evaluation")
    except (OSError, TypeError, VLMConfidenceReportError) as exc:
        raise VLMArtifactError(f"invalid confidence benchmark run: {exc}") from exc


def _read_confidence_repeatability_identity(path: str | os.PathLike[str]) -> Any:
    """Read one strict two-run receipt without an eager validation import."""

    try:
        from libreyolo.validation.vlm_confidence_benchmark import (
            VLMConfidenceReportError,
            read_benchmark_repeatability_receipt,
        )
    except ImportError as exc:  # pragma: no cover - package integrity guard
        raise VLMArtifactError(
            "confidence repeatability validation is unavailable in this runtime"
        ) from exc
    try:
        return read_benchmark_repeatability_receipt(
            path, label="publication_repeatability"
        )
    except (OSError, TypeError, VLMConfidenceReportError) as exc:
        raise VLMArtifactError(
            f"invalid confidence repeatability receipt: {exc}"
        ) from exc


def _confidence_run_ref(identity: Any) -> dict[str, str]:
    return {
        "run_id": identity.run_id,
        "process_id": identity.process_id,
        "report_sha256": identity.report_sha256,
        "envelope_sha256": identity.envelope_sha256,
    }


def _repeatability_claim_from_receipt(
    receipt_path: str | os.PathLike[str], primary_run: Mapping[str, str]
) -> tuple[dict[str, Any], Any]:
    identity = _read_confidence_repeatability_identity(receipt_path)
    if not identity.comparison.reproducible:
        raise VLMArtifactError(
            "confidence repeatability receipt must record a reproducible comparison"
        )
    tolerances = dict(identity.tolerances)
    if tolerances != {"score_atol": 0.0, "metric_atol": 0.0, "map_atol": 0.0}:
        raise VLMArtifactError(
            "publication repeatability requires exact zero comparison tolerances"
        )
    runs = [
        {
            "run_id": run.run_id,
            "process_id": run.process_id,
            "report_sha256": run.report_sha256,
            "envelope_sha256": run.envelope_sha256,
        }
        for run in identity.runs
    ]
    if runs[0] != dict(primary_run):
        raise VLMArtifactError(
            "repeatability receipt runs[0] must match the confidence_report primary run"
        )
    claim = {
        "schema": _REPEATABILITY_CLAIM_SCHEMA,
        "receipt_sha256": identity.receipt_sha256,
        "comparison_sha256": identity.comparison_sha256,
        "runs": runs,
        "tolerances": tolerances,
        "reproducible": True,
    }
    return claim, identity


def _validate_confidence_validation_metrics(
    value: Any, label: str
) -> dict[str, float | int]:
    metrics = _require_exact_keys(value, _CONFIDENCE_VALIDATION_METRICS, label)
    _require_finite_metrics(metrics, label)
    for key in sorted(_CONFIDENCE_PROBABILITY_METRICS):
        metric = float(metrics[key])
        if metric < 0.0 or metric > 1.0:
            raise VLMArtifactError(f"{label}.{key} must be between 0 and 1")
    for delta_key, (candidate_key, constant_key) in _CONFIDENCE_DELTA_METRICS.items():
        delta = float(metrics[delta_key])
        if delta < -1.0 or delta > 1.0:
            raise VLMArtifactError(f"{label}.{delta_key} must be between -1 and 1")
        expected_delta = float(metrics[candidate_key]) - float(metrics[constant_key])
        if delta != expected_delta:
            raise VLMArtifactError(
                f"{label}.{delta_key} must equal {candidate_key} minus {constant_key}"
            )
    return metrics


def _evaluation_claim_sha256(evaluation: Mapping[str, Any]) -> str:
    """Hash the machine-derived evaluation claim, excluding human approval."""

    claim = {
        "schema": _EVALUATION_CLAIM_SCHEMA,
        "benchmark": evaluation["benchmark"],
        "report_sha256": evaluation["report_sha256"],
        "envelope_sha256": evaluation["envelope_sha256"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "metrics": evaluation["metrics"],
        "repeatability": evaluation["repeatability"],
    }
    return _sha256_bytes(_canonical_json(claim))


def _inspect_checkpoint_identity(path: Path) -> Any:
    """Inspect one strict checkpoint without adding an eager training import."""

    try:
        from libreyolo.models.vlm.training.checkpoint import (
            inspect_vlm_checkpoint_identity,
        )
    except ImportError as exc:  # pragma: no cover - package integrity guard
        raise VLMArtifactError(
            "VLM checkpoint identity inspection is unavailable in this runtime"
        ) from exc
    return inspect_vlm_checkpoint_identity(path)


def _checkpoint_report_context(identity: Any) -> dict[str, Any]:
    """Return the exact path-free identity expected in a v3 confidence report."""

    files = [
        {
            "path": record.path,
            "role": record.role,
            "size": record.size,
            "sha256": record.sha256,
        }
        for record in identity.files
    ]
    return {
        "schema": _CONFIDENCE_CHECKPOINT_SCHEMA,
        "kind": "qwen3vl_lora_checkpoint",
        "family": identity.family,
        "size": identity.size,
        "task": identity.task,
        "base_repo": identity.base_repo,
        "base_revision": identity.base_revision,
        "aggregate_sha256": identity.aggregate_sha256,
        "adapter_weights_sha256": identity.adapter_weights_sha256,
        "adapter_config_sha256": identity.adapter_config_sha256,
        "checkpoint_contract_sha256": identity.checkpoint_contract_sha256,
        "processor_sha256": identity.processor_sha256,
        "files": files,
    }


def _confidence_run_identity_bytes(identity: Any) -> bytes:
    """Return an exact path-free snapshot of the public strict run identity."""

    return _canonical_json(
        {
            "run_id": identity.run_id,
            "process_id": identity.process_id,
            "report_sha256": identity.report_sha256,
            "envelope_sha256": identity.envelope_sha256,
            "execution_context": identity.execution_context,
            "benchmark_config": identity.benchmark_config,
            "metrics": identity.metrics,
            "nonfinite_metrics": list(identity.nonfinite_metrics),
        }
    )


def _evaluation_from_confidence_report(
    report_path: str | os.PathLike[str], checkpoint_identity: Any
) -> tuple[dict[str, Any], bytes, dict[str, str]]:
    """Derive publication evaluation claims from a bound strict gate report."""

    run_identity = _read_confidence_benchmark_identity(report_path)
    report_sha = _require_sha256(
        run_identity.report_sha256, "confidence report SHA-256"
    )
    envelope_sha = _require_sha256(
        run_identity.envelope_sha256, "confidence run envelope SHA-256"
    )
    benchmark_config = _plain_json_object(
        run_identity.benchmark_config, "confidence report benchmark_config"
    )
    context = _require_exact_keys(
        run_identity.execution_context,
        _CONFIDENCE_CONTEXT_KEYS,
        "confidence run execution_context",
    )
    if _canonical_json(benchmark_config.get("benchmark_run")) != _canonical_json(
        context
    ):
        raise VLMArtifactError(
            "confidence report benchmark_run does not match the run envelope context"
        )
    if context["schema"] != _CONFIDENCE_CONTEXT_SCHEMA:
        raise VLMArtifactError(
            "confidence run execution_context schema must be "
            f"{_CONFIDENCE_CONTEXT_SCHEMA!r}"
        )
    dataset = _require_exact_keys(
        context["dataset"],
        _CONFIDENCE_DATASET_KEYS,
        "confidence run execution_context.dataset",
    )
    if dataset["schema"] != _CONFIDENCE_DATASET_SCHEMA:
        raise VLMArtifactError(
            "confidence report benchmark dataset schema must be "
            f"{_CONFIDENCE_DATASET_SCHEMA!r}"
        )
    partition = _require_exact_keys(
        dataset["partition"],
        _CONFIDENCE_PARTITION_KEYS,
        "confidence run execution_context.dataset.partition",
    )
    expected_partition = {
        "name": _CONFIDENCE_PARTITION_NAME,
        "role": _CONFIDENCE_PARTITION_ROLE,
        "start": _CONFIDENCE_PARTITION_START,
        "stop": _CONFIDENCE_PARTITION_STOP,
        "image_count": _CONFIDENCE_PARTITION_IMAGE_COUNT,
        "annotation_artifact": _CONFIDENCE_PARTITION_ARTIFACT,
    }
    if any(partition[key] != expected for key, expected in expected_partition.items()):
        raise VLMArtifactError(
            "confidence report must use the exact holdout100 "
            "fine_tune_validation partition"
        )

    expected_checkpoint = _checkpoint_report_context(checkpoint_identity)
    report_checkpoint = _require_exact_keys(
        context["checkpoint"],
        _CONFIDENCE_CHECKPOINT_KEYS,
        "confidence report benchmark_run.checkpoint",
    )
    if _canonical_json(report_checkpoint) != _canonical_json(expected_checkpoint):
        raise VLMArtifactError(
            "confidence report checkpoint identity does not match the checkpoint"
        )
    expected_duplicates = {
        "family": checkpoint_identity.family,
        "size": checkpoint_identity.size,
        "base_repo": checkpoint_identity.base_repo,
        "base_revision": checkpoint_identity.base_revision,
    }
    for field, expected in expected_duplicates.items():
        if benchmark_config.get(field) != expected:
            raise VLMArtifactError(
                f"confidence report benchmark_config.{field} does not match "
                "the checkpoint"
            )

    metrics_value = run_identity.metrics
    if type(metrics_value) is not dict:
        raise VLMArtifactError("confidence report metrics must be a JSON object")
    missing_required = sorted(
        key
        for key in _CONFIDENCE_VALIDATION_METRICS
        if key not in metrics_value or metrics_value[key] is None
    )
    if missing_required:
        raise VLMArtifactError(
            "confidence report has null or missing validation metrics: "
            + ", ".join(missing_required)
        )
    metrics = {
        key: metrics_value[key] for key in sorted(_CONFIDENCE_VALIDATION_METRICS)
    }
    _validate_confidence_validation_metrics(
        metrics, "confidence report validation metrics"
    )
    return (
        {
            "benchmark": _CONFIDENCE_BENCHMARK_ID,
            "report_sha256": report_sha,
            "envelope_sha256": envelope_sha,
            "metrics": metrics,
        },
        _confidence_run_identity_bytes(run_identity),
        _confidence_run_ref(run_identity),
    )


def _validate_repeatability_claim(
    value: Any,
    *,
    report_sha256: str,
    envelope_sha256: str,
) -> dict[str, Any]:
    claim = _require_exact_keys(value, _REPEATABILITY_KEYS, "evaluation.repeatability")
    if claim["schema"] != _REPEATABILITY_CLAIM_SCHEMA:
        raise VLMArtifactError(
            f"evaluation.repeatability.schema must be {_REPEATABILITY_CLAIM_SCHEMA!r}"
        )
    _require_sha256(claim["receipt_sha256"], "evaluation.repeatability.receipt_sha256")
    _require_sha256(
        claim["comparison_sha256"],
        "evaluation.repeatability.comparison_sha256",
    )
    runs_value = claim["runs"]
    if not isinstance(runs_value, list) or len(runs_value) != 2:
        raise VLMArtifactError(
            "evaluation.repeatability.runs must contain exactly two runs"
        )
    runs = []
    for index, value in enumerate(runs_value):
        run = _require_exact_keys(
            value,
            _REPEATABILITY_RUN_KEYS,
            f"evaluation.repeatability.runs[{index}]",
        )
        for field in ("run_id", "process_id"):
            if not isinstance(run[field], str) or not _RUN_IDENTIFIER_RE.fullmatch(
                run[field]
            ):
                raise VLMArtifactError(
                    f"evaluation.repeatability.runs[{index}].{field} must be a "
                    "32-character lowercase hex identifier"
                )
        for field in ("report_sha256", "envelope_sha256"):
            _require_sha256(
                run[field], f"evaluation.repeatability.runs[{index}].{field}"
            )
        runs.append(run)
    if runs[0]["run_id"] == runs[1]["run_id"]:
        raise VLMArtifactError("evaluation.repeatability run_id values must differ")
    if runs[0]["process_id"] == runs[1]["process_id"]:
        raise VLMArtifactError("evaluation.repeatability process_id values must differ")
    if (
        runs[0]["report_sha256"] != report_sha256
        or runs[0]["envelope_sha256"] != envelope_sha256
    ):
        raise VLMArtifactError(
            "evaluation.repeatability.runs[0] must match the primary evaluation run"
        )

    tolerances = _require_exact_keys(
        claim["tolerances"],
        _REPEATABILITY_TOLERANCE_KEYS,
        "evaluation.repeatability.tolerances",
    )
    for field in sorted(_REPEATABILITY_TOLERANCE_KEYS):
        value = tolerances[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != 0.0
        ):
            raise VLMArtifactError(
                f"evaluation.repeatability.tolerances.{field} must equal 0"
            )
    _require_true(claim["reproducible"], "evaluation.repeatability.reproducible")
    return claim


def _validate_publication_evidence(
    value: Any,
    *,
    expected_size: str | None = None,
    enforce_current_recipe: bool = False,
) -> dict[str, Any]:
    evidence = _require_exact_keys(value, _EVIDENCE_KEYS, "publication evidence")
    if evidence["schema"] != PUBLICATION_EVIDENCE_SCHEMA:
        raise VLMArtifactError(
            f"publication evidence schema must be {PUBLICATION_EVIDENCE_SCHEMA!r}"
        )

    artifact_license = _require_exact_keys(
        evidence["artifact_license"], _ARTIFACT_LICENSE_KEYS, "artifact_license"
    )
    if artifact_license["spdx"] != "Apache-2.0":
        raise VLMArtifactError("artifact_license.spdx must be 'Apache-2.0'")
    if artifact_license["redistribution_decision"] != "approved":
        raise VLMArtifactError(
            "artifact_license.redistribution_decision must be 'approved'"
        )

    base = _require_exact_keys(evidence["base_model"], _BASE_MODEL_KEYS, "base_model")
    base_repo = _require_string(base["repo"], "base_model.repo")
    base_revision = _require_commit(base["revision"], "base_model.revision")
    matches = [
        size
        for size, identity in _SUPPORTED_BASES.items()
        if identity == (base_repo, base_revision)
    ]
    if len(matches) != 1 or (expected_size is not None and matches[0] != expected_size):
        raise VLMArtifactError(
            "base_model does not match a supported immutable Qwen3-VL pin"
        )
    if base["license_spdx"] != "Apache-2.0":
        raise VLMArtifactError("base_model.license_spdx must be 'Apache-2.0'")
    _require_https_url(base["license_evidence_url"], "base_model.license_evidence_url")
    if base["weights_redistribution_decision"] != "reference-only":
        raise VLMArtifactError(
            "base_model.weights_redistribution_decision must be 'reference-only'"
        )
    if base["processor_redistribution_decision"] != "approved":
        raise VLMArtifactError(
            "base_model.processor_redistribution_decision must be 'approved'"
        )
    _validate_snapshot_files(
        base["snapshot"],
        expected_repo=base_repo,
        expected_revision=base_revision,
    )
    snapshot_sha = base["snapshot"]["sha256"]

    data = _require_exact_keys(
        evidence["training_data"], _TRAINING_DATA_KEYS, "training_data"
    )
    _require_https_url(data["source"], "training_data.source")
    _require_token(data["version"], "training_data.version")
    _require_token(data["split"], "training_data.split")
    _require_spdx(data["license_spdx"], "training_data.license_spdx")
    _require_https_url(
        data["license_evidence_url"], "training_data.license_evidence_url"
    )
    data_sha = _require_sha256(data["manifest_sha256"], "training_data.manifest_sha256")
    if data["redistribution_decision"] != "approved-for-derived-weights":
        raise VLMArtifactError(
            "training_data.redistribution_decision must be "
            "'approved-for-derived-weights'"
        )

    evaluation = _require_exact_keys(
        evidence["evaluation"], _EVALUATION_KEYS, "evaluation"
    )
    if evaluation["benchmark"] != _CONFIDENCE_BENCHMARK_ID:
        raise VLMArtifactError(
            f"evaluation.benchmark must be {_CONFIDENCE_BENCHMARK_ID!r}"
        )
    report_sha = _require_sha256(
        evaluation["report_sha256"], "evaluation.report_sha256"
    )
    envelope_sha = _require_sha256(
        evaluation["envelope_sha256"], "evaluation.envelope_sha256"
    )
    evaluation_checkpoint_sha = _require_sha256(
        evaluation["checkpoint_sha256"], "evaluation.checkpoint_sha256"
    )
    _validate_confidence_validation_metrics(evaluation["metrics"], "evaluation.metrics")
    repeatability = _validate_repeatability_claim(
        evaluation["repeatability"],
        report_sha256=report_sha,
        envelope_sha256=envelope_sha,
    )
    _require_true(evaluation["passed"], "evaluation.passed")
    evaluation_claim_sha = _evaluation_claim_sha256(evaluation)

    code = _require_exact_keys(evidence["code"], _CODE_KEYS, "code")
    if code["repository"] != "https://github.com/LibreYOLO/libreyolo":
        raise VLMArtifactError("code.repository must identify LibreYOLO")
    code_revision = _require_commit(code["revision"], "code.revision")
    _require_true(code["clean"], "code.clean")
    recipe = _require_exact_keys(code["recipe"], _RECIPE_KEYS, "code.recipe")
    if recipe["id"] != "qwen3vl-lora-v1":
        raise VLMArtifactError("code.recipe.id must be 'qwen3vl-lora-v1'")
    recipe_sha = _require_sha256(recipe["sha256"], "code.recipe.sha256")
    if enforce_current_recipe and recipe_sha != _recipe_sha256():
        raise VLMArtifactError("code.recipe.sha256 does not match the shipped recipe")
    dependencies = code["dependencies"]
    required_dependencies = {"libreyolo", "peft", "torch", "transformers"}
    if not isinstance(dependencies, dict) or set(dependencies) != required_dependencies:
        raise VLMArtifactError(
            f"code.dependencies must contain exactly {sorted(required_dependencies)}"
        )
    for name, version in dependencies.items():
        if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
            raise VLMArtifactError(f"code.dependencies.{name} has an invalid version")
    for name, expected in _PUBLICATION_DEPENDENCY_PINS.items():
        if dependencies[name] != expected:
            raise VLMArtifactError(
                f"code.dependencies.{name} must be {expected!r} for VLM artifact v1"
            )

    review = _require_exact_keys(evidence["review"], _REVIEW_KEYS, "review")
    _require_true(review["approved"], "review.approved")
    _require_string(review["reviewer"], "review.reviewer", max_length=200)
    reviewed_at = _require_string(review["reviewed_at"], "review.reviewed_at")
    match = _UTC_RE.fullmatch(reviewed_at)
    if match is None:
        raise VLMArtifactError("review.reviewed_at must be an RFC 3339 UTC timestamp")
    try:
        datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise VLMArtifactError(
            "review.reviewed_at is not a real UTC timestamp"
        ) from exc
    bindings = _require_exact_keys(review["bindings"], _BINDING_KEYS, "review.bindings")
    adapter_weights_sha = _require_sha256(
        bindings["adapter_weights_sha256"],
        "review.bindings.adapter_weights_sha256",
    )
    adapter_config_sha = _require_sha256(
        bindings["adapter_config_sha256"],
        "review.bindings.adapter_config_sha256",
    )
    checkpoint_contract_sha = _require_sha256(
        bindings["checkpoint_contract_sha256"],
        "review.bindings.checkpoint_contract_sha256",
    )
    processor_sha = _require_sha256(
        bindings["processor_sha256"], "review.bindings.processor_sha256"
    )
    if evaluation_checkpoint_sha != adapter_weights_sha:
        raise VLMArtifactError(
            "evaluation.checkpoint_sha256 must match the reviewed adapter weights"
        )
    expected_bindings = {
        "base_snapshot_sha256": snapshot_sha,
        "training_data_manifest_sha256": data_sha,
        "evaluation_report_sha256": report_sha,
        "evaluation_envelope_sha256": envelope_sha,
        "evaluation_repeatability_receipt_sha256": repeatability["receipt_sha256"],
        "evaluation_repeatability_comparison_sha256": repeatability[
            "comparison_sha256"
        ],
        "evaluation_claim_sha256": evaluation_claim_sha,
        "code_revision": code_revision,
        "recipe_sha256": recipe_sha,
        "adapter_weights_sha256": adapter_weights_sha,
        "adapter_config_sha256": adapter_config_sha,
        "checkpoint_contract_sha256": checkpoint_contract_sha,
        "processor_sha256": processor_sha,
    }
    if bindings != expected_bindings:
        raise VLMArtifactError("review.bindings do not match the reviewed evidence")
    gates = _require_exact_keys(review["gates"], _GATE_KEYS, "review.gates")
    for key, approved in gates.items():
        _require_true(approved, f"review.gates.{key}")
    return evidence


def _expected_prompt(names: list[str]) -> str:
    labels = ", ".join(names)
    return (
        f"Detect all instances of: {labels}. "
        "Output the result as a JSON array, one object per instance: "
        '[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
        "Only include objects that are actually visible; if there are none, "
        "respond with an empty array []."
    )


def _validate_contract(value: Any) -> dict[str, Any]:
    contract = _require_exact_keys(value, _CONTRACT_KEYS, "VLM checkpoint contract")
    if contract["schema"] != 1:
        raise VLMArtifactError("VLM checkpoint contract schema must be 1")
    if contract["family"] != "qwen3vl" or contract["size"] not in _SUPPORTED_BASES:
        raise VLMArtifactError("artifact publication supports only Qwen3-VL 2B/4B")
    repo, revision = _SUPPORTED_BASES[contract["size"]]
    if contract["base_repo"] != repo or contract["base_revision"] != revision:
        raise VLMArtifactError(
            "checkpoint contract does not use the immutable base pin"
        )
    if contract["task"] != "detect":
        raise VLMArtifactError("checkpoint contract task must be 'detect'")
    if contract["bbox_key"] != "bbox_2d":
        raise VLMArtifactError("checkpoint contract bbox_key must be 'bbox_2d'")
    if contract["box_format"] != "xyxy":
        raise VLMArtifactError("checkpoint contract box_format must be 'xyxy'")
    divisor = contract["coord_divisor"]
    if (
        isinstance(divisor, bool)
        or not isinstance(divisor, (int, float))
        or divisor != 1000
    ):
        raise VLMArtifactError("checkpoint contract coord_divisor must be 1000")
    names = contract["names"]
    if not isinstance(names, list) or not names:
        raise VLMArtifactError("checkpoint contract names must be nonblank strings")
    for index, name in enumerate(names):
        _require_string(name, f"checkpoint contract names[{index}]", max_length=200)
    if len({name.casefold() for name in names}) != len(names):
        raise VLMArtifactError(
            "checkpoint contract names must be unique case-insensitively"
        )
    if contract["prompt"] != _expected_prompt(names):
        raise VLMArtifactError(
            "checkpoint contract prompt does not match Qwen3-VL detection"
        )
    _require_finite_metrics(contract["metrics"], "checkpoint contract metrics")
    version = contract["libreyolo_version"]
    if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
        raise VLMArtifactError("checkpoint contract libreyolo_version is invalid")
    return contract


def _canonical_adapter_config(
    value: Any, contract: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VLMArtifactError("adapter_config.json must contain a JSON object")
    config = dict(value)
    if set(config) != _PEFT_CONFIG_KEYS:
        missing = sorted(_PEFT_CONFIG_KEYS - set(config))
        unknown = sorted(set(config) - _PEFT_CONFIG_KEYS)
        raise VLMArtifactError(
            f"adapter_config.json keys do not match PEFT 0.19.1: "
            f"missing={missing}, unknown={unknown}"
        )
    if config.get("peft_type") != "LORA":
        raise VLMArtifactError("adapter_config.json must declare peft_type='LORA'")
    expected_repo = contract["base_repo"]
    current_repo = config.get("base_model_name_or_path")
    local_cache_name = f"weights/LibreQwen3VL{contract['size']}"
    normalized_current_repo = (
        current_repo.replace("\\", "/")
        if isinstance(current_repo, str)
        else current_repo
    )
    if normalized_current_repo not in (None, "", expected_repo, local_cache_name):
        raise VLMArtifactError(
            "adapter_config.json base model does not match the contract"
        )
    current_revision = config.get("revision")
    if current_revision not in (None, contract["base_revision"]):
        raise VLMArtifactError(
            "adapter_config.json revision does not match the contract"
        )
    native_auto_mapping = {
        "base_model_class": "Qwen3VLForConditionalGeneration",
        "parent_library": "transformers.models.qwen3_vl.modeling_qwen3_vl",
    }
    if config.get("auto_mapping") not in (None, {}, native_auto_mapping):
        raise VLMArtifactError(
            "adapter_config.json must not request dynamic model code"
        )
    expected_recipe = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": (
            r".*language_model.*\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        ),
    }
    for key, expected in expected_recipe.items():
        if config.get(key) != expected:
            raise VLMArtifactError(
                f"adapter_config.json field {key!r} does not match the Qwen3-VL recipe"
            )
    safe_behavior = {
        "alora_invocation_tokens": None,
        "alpha_pattern": {},
        "arrow_config": None,
        "corda_config": None,
        "ensure_weight_tying": False,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_bias": False,
        "lora_ga_config": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_version": "0.19.1",
        "qalora_group_size": 16,
        "rank_pattern": {},
        "target_parameters": None,
        "task_type": None,
        "trainable_token_indices": None,
        "use_bdlora": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }
    for key, expected in safe_behavior.items():
        if config[key] != expected:
            raise VLMArtifactError(
                f"adapter_config.json field {key!r} is outside the fixed recipe"
            )
    config["base_model_name_or_path"] = expected_repo
    config["revision"] = contract["base_revision"]
    config["auto_mapping"] = None
    return config


def _expected_lora_shapes(size: str, module: str) -> tuple[list[int], list[int]]:
    layout = _QWEN_LORA_LAYOUT[size]
    hidden = layout["hidden"]
    if module == "q_proj":
        input_width, output_width = hidden, layout["q"]
    elif module in {"k_proj", "v_proj"}:
        input_width, output_width = hidden, layout["kv"]
    elif module == "o_proj":
        input_width, output_width = layout["q"], hidden
    elif module in {"gate_proj", "up_proj"}:
        input_width, output_width = hidden, layout["intermediate"]
    elif module == "down_proj":
        input_width, output_width = layout["intermediate"], hidden
    else:  # pragma: no cover - guarded by the tensor-name regex
        raise AssertionError(module)
    return [16, input_width], [output_width, 16]


def _validate_safetensors(path: Path, size: str) -> str:
    try:
        with _open_stable_regular_file(path, "adapter safetensors") as (
            stream,
            opened,
        ):
            if opened.st_size < 9:
                raise VLMArtifactError("adapter_model.safetensors is too small")
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise VLMArtifactError(
                    "adapter_model.safetensors has a truncated header"
                )
            header_size = struct.unpack("<Q", prefix)[0]
            if (
                header_size < 2
                or header_size > _MAX_SAFETENSORS_HEADER_BYTES
                or header_size % 8 != 0
                or 8 + header_size > opened.st_size
            ):
                raise VLMArtifactError(
                    "adapter_model.safetensors has an invalid header length"
                )
            header_bytes = stream.read(header_size)
            stream.seek(0)
            verified_header = stream.read(8 + header_size)
            if verified_header != prefix + header_bytes:
                raise VLMArtifactError(
                    "adapter_model.safetensors changed while its header was read"
                )
            stream.seek(0)
            bound_header = stream.read(8 + header_size)
            if bound_header != verified_header:
                raise VLMArtifactError(
                    "adapter_model.safetensors changed before it was fingerprinted"
                )
            payload_hasher = hashlib.sha256(bound_header)
            payload_size = len(bound_header)
            while True:
                chunk = stream.read(
                    min(_COPY_CHUNK_BYTES, opened.st_size - payload_size + 1)
                )
                if not chunk:
                    break
                payload_hasher.update(chunk)
                payload_size += len(chunk)
                if payload_size > opened.st_size:
                    raise VLMArtifactError(
                        "adapter_model.safetensors changed while fingerprinted"
                    )
            if payload_size != opened.st_size:
                raise VLMArtifactError(
                    "adapter_model.safetensors changed while fingerprinted"
                )
            payload_sha256 = payload_hasher.hexdigest()
    except OSError as exc:
        raise VLMArtifactError("Could not read adapter_model.safetensors") from exc
    header = _decode_json(header_bytes, "adapter safetensors header")
    if not isinstance(header, dict):
        raise VLMArtifactError("adapter safetensors header must contain an object")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in metadata.items()
        )
    ):
        raise VLMArtifactError(
            "adapter safetensors metadata must map strings to strings"
        )
    if not header:
        raise VLMArtifactError("adapter_model.safetensors contains no tensors")
    ranges: list[tuple[int, int, str]] = []
    lora_pairs: dict[str, dict[str, tuple[list[int], str]]] = {}
    layer_modules: dict[int, set[str]] = {}
    logical_pairs: set[tuple[int, str]] = set()
    tensor_dtypes: set[str] = set()
    for name, raw in header.items():
        _require_string(name, "safetensors tensor name", max_length=1024)
        name_match = _LORA_TENSOR_RE.fullmatch(name)
        if name_match is None:
            raise VLMArtifactError(
                f"adapter safetensors contains a non-LoRA tensor: {name!r}"
            )
        tensor = _require_exact_keys(
            raw, {"dtype", "shape", "data_offsets"}, f"safetensors tensor {name!r}"
        )
        dtype = tensor["dtype"]
        if dtype not in _SAFETENSORS_DTYPE_BYTES:
            raise VLMArtifactError(f"safetensors tensor {name!r} has unsupported dtype")
        shape = tensor["shape"]
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in shape
        ):
            raise VLMArtifactError(f"safetensors tensor {name!r} has invalid shape")
        offsets = tensor["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in offsets
            )
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise VLMArtifactError(f"safetensors tensor {name!r} has invalid offsets")
        elements = math.prod(shape)
        expected_bytes = elements * _SAFETENSORS_DTYPE_BYTES[dtype]
        if offsets[1] - offsets[0] != expected_bytes:
            raise VLMArtifactError(
                f"safetensors tensor {name!r} byte range is inconsistent"
            )
        ranges.append((offsets[0], offsets[1], name))
        stem = name_match.group("stem")
        side = name_match.group("side")
        layer_token = name_match.group("layer")
        layer = int(layer_token)
        if layer_token != str(layer):
            raise VLMArtifactError(
                f"adapter safetensors layer index is not canonical: {layer_token!r}"
            )
        module = name_match.group("attn") or name_match.group("mlp")
        pair = lora_pairs.setdefault(stem, {})
        if side in pair:
            raise VLMArtifactError(f"adapter safetensors duplicates LoRA side {name!r}")
        pair[side] = (shape, dtype)
        layer_modules.setdefault(layer, set()).add(module)
        logical_pairs.add((layer, module))
        tensor_dtypes.add(dtype)
    ranges.sort()
    cursor = 0
    for start, end, name in ranges:
        if start != cursor:
            raise VLMArtifactError(f"safetensors tensor {name!r} has a gap or overlap")
        cursor = end
    if cursor != opened.st_size - 8 - header_size:
        raise VLMArtifactError(
            "adapter_model.safetensors has trailing or missing tensor data"
        )
    for stem, pair in lora_pairs.items():
        if set(pair) != {"A", "B"}:
            raise VLMArtifactError(
                f"adapter safetensors has an incomplete LoRA pair: {stem}"
            )
        shape_a, dtype_a = pair["A"]
        shape_b, dtype_b = pair["B"]
        module = stem.rsplit(".", 1)[-1]
        expected_a, expected_b = _expected_lora_shapes(size, module)
        if shape_a != expected_a or shape_b != expected_b or dtype_a != dtype_b:
            raise VLMArtifactError(
                f"adapter safetensors LoRA pair is not rank-16 compatible: {stem}"
            )
    expected_modules = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    layers = sorted(layer_modules)
    expected_layers = _QWEN_LORA_LAYOUT.get(size, {}).get("layers")
    if expected_layers is None or layers != list(range(expected_layers)):
        raise VLMArtifactError(
            f"adapter safetensors must cover exactly {expected_layers} Qwen3-VL layers"
        )
    if any(modules != expected_modules for modules in layer_modules.values()):
        raise VLMArtifactError(
            "adapter safetensors must contain all seven supported LoRA modules per layer"
        )
    if len(header) != expected_layers * len(expected_modules) * 2 or len(
        logical_pairs
    ) != expected_layers * len(expected_modules):
        raise VLMArtifactError("adapter safetensors tensor inventory is not exact")
    if len(tensor_dtypes) != 1 or not tensor_dtypes <= {"BF16", "F32"}:
        raise VLMArtifactError(
            "adapter safetensors tensors must use one uniform BF16 or F32 dtype"
        )
    return payload_sha256


def _scan_flat_directory(root: Path, label: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    folded: set[str] = set()
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise VLMArtifactError(f"Could not inspect {label}: {root}") from exc
    for entry in entries:
        path = root / entry.name
        if _is_link_or_junction(path):
            raise VLMArtifactError(f"{label} must not contain links: {entry.name}")
        try:
            identity = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise VLMArtifactError(
                f"Could not inspect {label} entry: {entry.name}"
            ) from exc
        if not stat.S_ISREG(identity.st_mode):
            raise VLMArtifactError(
                f"{label} must contain only flat regular files: {entry.name}"
            )
        _assert_regular_unlinked_file(path, f"{label} entry {entry.name}")
        name = _safe_inventory_path(entry.name, f"{label} entry", flat=True)
        lower = name.casefold()
        if lower in folded:
            raise VLMArtifactError(f"{label} contains case-ambiguous paths")
        folded.add(lower)
        files[name] = path
    return files


def _validate_checkpoint_inventory(root: Path) -> dict[str, Path]:
    files = _scan_flat_directory(root, "VLM checkpoint")
    missing = sorted(_REQUIRED_INPUT_FILES - set(files))
    if missing:
        raise VLMArtifactError(f"VLM checkpoint is missing required files: {missing}")
    if not {"preprocessor_config.json", "processor_config.json"} & set(files):
        raise VLMArtifactError(
            "VLM checkpoint requires preprocessor_config.json or processor_config.json"
        )
    allowed = _REQUIRED_INPUT_FILES | _PROCESSOR_FILES | _SOURCE_ONLY_FILES
    unexpected = sorted(set(files) - allowed)
    if unexpected:
        raise VLMArtifactError(
            f"VLM checkpoint contains unsupported files: {unexpected}"
        )
    if any(name.lower().endswith((".bin", ".pt", ".pth")) for name in files):
        raise VLMArtifactError("VLM artifacts permit safetensors weights only")
    for name, path in files.items():
        limit = _ARTIFACT_FILE_LIMITS[name]
        if os.lstat(path).st_size > limit:
            raise VLMArtifactError(
                f"VLM checkpoint file {name} exceeds its {limit}-byte safety limit"
            )
    tokenizer_files = set(files) & {"tokenizer.json", "vocab.json", "merges.txt"}
    if (
        "tokenizer.json" not in tokenizer_files
        and not {"vocab.json", "merges.txt"} <= tokenizer_files
    ):
        raise VLMArtifactError(
            "VLM checkpoint requires tokenizer.json or both vocab.json and merges.txt"
        )
    # PEFT writes its own generic README. Publication always regenerates the
    # reviewed LibreYOLO model card, so source prose is inspected as a regular
    # file above but never copied.
    return {
        name: path for name, path in files.items() if name not in _SOURCE_ONLY_FILES
    }


def _validate_processor_files(root: Path, files: set[str], size: str) -> None:
    expected_records = [
        {"path": path, "size": size_bytes, "sha256": digest}
        for path, size_bytes, digest in _CANONICAL_PROCESSOR_FILES[size]
    ]
    expected_paths = {record["path"] for record in expected_records}
    if files != expected_paths:
        raise VLMArtifactError(
            "Qwen processor serialization must contain the exact audited file set"
        )
    observed_records = [
        _snapshot_file_record(root / record["path"], record["path"])
        for record in expected_records
    ]
    if observed_records != expected_records:
        raise VLMArtifactError(
            "Qwen processor serialization does not match the audited upstream assets"
        )
    tokenizer = _load_json(root / "tokenizer_config.json", "tokenizer_config.json")
    if not isinstance(tokenizer, dict):
        raise VLMArtifactError("tokenizer_config.json must contain an object")
    embedded_template = tokenizer.get("chat_template")
    if (
        "chat_template.json" not in files
        and "chat_template.jinja" not in files
        and (not isinstance(embedded_template, str) or not embedded_template.strip())
    ):
        raise VLMArtifactError("Qwen processor snapshot is missing its chat template")
    processor_name = (
        "processor_config.json"
        if "processor_config.json" in files
        else "preprocessor_config.json"
    )
    preprocessor = _load_json(root / processor_name, processor_name)
    if not isinstance(preprocessor, dict) or not preprocessor:
        raise VLMArtifactError(f"{processor_name} must contain a nonempty object")


def _copy_file_stable(source: Path, destination: Path) -> None:
    before = _assert_regular_unlinked_file(source, f"source file {source.name}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    hasher = hashlib.sha256()
    copied = 0
    try:
        descriptor = os.open(source, flags)
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or getattr(opened_before, "st_nlink", 1) != 1
            or not _same_file_identity(before, opened_before)
        ):
            raise VLMArtifactError(
                f"checkpoint file changed before it was opened: {source.name}"
            )
        source_stream = os.fdopen(descriptor, "rb")
        descriptor = None
        with source_stream as src, destination.open("xb") as dst:
            while True:
                chunk = src.read(
                    min(_COPY_CHUNK_BYTES, opened_before.st_size - copied + 1)
                )
                if not chunk:
                    break
                dst.write(chunk)
                hasher.update(chunk)
                copied += len(chunk)
                if copied > opened_before.st_size:
                    raise VLMArtifactError(
                        f"checkpoint file changed while copied: {source.name}"
                    )
            dst.flush()
            os.fsync(dst.fileno())
            opened_mid = os.fstat(src.fileno())
            if not _same_file_identity(opened_before, opened_mid):
                raise VLMArtifactError(
                    f"checkpoint file changed while copied: {source.name}"
                )

            src.seek(0)
            verified = hashlib.sha256()
            verified_size = 0
            while True:
                chunk = src.read(
                    min(
                        _COPY_CHUNK_BYTES,
                        opened_before.st_size - verified_size + 1,
                    )
                )
                if not chunk:
                    break
                verified.update(chunk)
                verified_size += len(chunk)
                if verified_size > opened_before.st_size:
                    raise VLMArtifactError(
                        f"checkpoint file changed while copied: {source.name}"
                    )
            opened_after = os.fstat(src.fileno())
            if (
                verified_size != copied
                or verified.digest() != hasher.digest()
                or not _same_file_identity(opened_before, opened_after)
            ):
                raise VLMArtifactError(
                    f"checkpoint file changed while copied: {source.name}"
                )
    except OSError as exc:
        raise VLMArtifactError(f"Could not copy checkpoint file {source.name}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    after = _assert_regular_unlinked_file(source, f"source file {source.name}")
    if copied != before.st_size or not _same_file_identity(before, after):
        raise VLMArtifactError(f"checkpoint file changed while copied: {source.name}")
    copied_identity = _fingerprint_file(destination, destination.name)
    if copied_identity["sha256"] != hasher.hexdigest():
        raise VLMArtifactError(f"staged file digest mismatch: {destination.name}")


def _write_create_only(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise VLMArtifactError(f"Could not create artifact file {path.name}") from exc


def _link_create_only(source: Path, destination: Path) -> None:
    """Atomically add one name for a fully written regular file."""

    os.link(source, destination, follow_symlinks=False)


def _write_bytes_atomic_create_only(
    destination: Path, payload: bytes, *, label: str
) -> None:
    """Publish complete bytes without replacing a concurrent destination."""

    parent = destination.parent
    parent_identity = os.lstat(parent)
    if _is_link_or_junction(parent) or not stat.S_ISDIR(parent_identity.st_mode):
        raise VLMArtifactError(f"{label} parent must be an unlinked directory")
    parent_seal = (parent_identity.st_dev, parent_identity.st_ino)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.staging-",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    temporary_identity = os.fstat(descriptor)
    temporary_seal = (temporary_identity.st_dev, temporary_identity.st_ino)
    linked = False
    complete = False

    def require_parent_seal() -> None:
        try:
            current = os.lstat(parent)
        except OSError as exc:
            raise VLMArtifactError(
                f"{label} parent changed during publication"
            ) from exc
        if (
            _is_link_or_junction(parent)
            or not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != parent_seal
        ):
            raise VLMArtifactError(f"{label} parent changed during publication")

    def entry_has_seal(path: Path, seal: tuple[int, int]) -> bool:
        try:
            current = os.lstat(path)
        except OSError:
            return False
        return (
            not _is_link_or_junction(path)
            and stat.S_ISREG(current.st_mode)
            and (current.st_dev, current.st_ino) == seal
        )

    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        require_parent_seal()
        if not entry_has_seal(temporary, temporary_seal):
            raise VLMArtifactError(f"{label} staging file changed during publication")
        try:
            _link_create_only(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"{label} already exists: {destination}") from exc
        except OSError as exc:
            raise VLMArtifactError(f"Could not publish {label}: {destination}") from exc
        linked = True
        require_parent_seal()
        if not entry_has_seal(destination, temporary_seal):
            raise VLMArtifactError(f"{label} changed during publication")

        os.close(descriptor)
        descriptor = -1
        if not entry_has_seal(temporary, temporary_seal):
            raise VLMArtifactError(f"{label} staging file changed during publication")
        temporary.unlink()
        require_parent_seal()
        observed = _read_bounded(
            destination,
            max_bytes=max(len(payload), 1),
            label=label,
        )
        if observed != payload:
            raise VLMArtifactError(f"{label} bytes changed during publication")
        complete = True
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if linked and not complete and entry_has_seal(destination, temporary_seal):
            try:
                destination.unlink()
            except OSError:
                pass
        if entry_has_seal(temporary, temporary_seal):
            try:
                temporary.unlink()
            except OSError:
                pass


def _rename_create_only(
    source: Path, destination: Path, expected_source_seal: tuple[int, int]
) -> None:
    """Atomically rename a directory while refusing an existing destination."""

    source_identity = os.lstat(source)
    if (
        _is_link_or_junction(source)
        or not stat.S_ISDIR(source_identity.st_mode)
        or (source_identity.st_dev, source_identity.st_ino) != expected_source_seal
    ):
        raise VLMArtifactError("artifact staging directory changed before publication")

    if os.name == "nt":
        # MoveFileEx without MOVEFILE_REPLACE_EXISTING is the behavior exposed
        # by os.rename on Windows.
        os.rename(source, destination)
        return

    import ctypes  # local: no startup cost on ordinary validation

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise VLMArtifactError(
                "atomic create-only directory publication requires renameat2"
            )
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            -100,  # AT_FDCWD
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
    elif sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise VLMArtifactError(
                "atomic create-only directory publication requires renamex_np"
            )
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(
            os.fsencode(source),
            os.fsencode(destination),
            0x00000004,  # RENAME_EXCL
        )
    else:
        raise VLMArtifactError(
            "atomic create-only directory publication is unsupported on this platform"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _canonicalize_json_file(source: Path, destination: Path, label: str) -> Any:
    value = _load_json(source, label)
    _write_create_only(destination, _json_file_bytes(value))
    return value


def _apache_license_bytes() -> bytes:
    # Reuse the standard text already shipped as package data. LF normalization
    # plus exactly one final newline is byte-identical to Qwen's Apache-2.0
    # license evidence and stable across Git checkout settings.
    license_path = Path(__file__).parents[1] / "picosam3" / "LICENSE"
    payload = license_path.read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n") + b"\n"
    expected = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    if _sha256_bytes(payload) != expected:
        raise VLMArtifactError(
            "packaged Apache-2.0 license text has an unexpected digest"
        )
    return payload


def _gitattributes_bytes() -> bytes:
    return b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n"


def _markdown_inline_code(value: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", value)), default=0)
    fence = "`" * (longest + 1)
    if longest:
        return f"{fence} {value} {fence}"
    return f"{fence}{value}{fence}"


def _markdown_text(value: str) -> str:
    return re.sub(r"([\\`*_[\]{}()<>#+.!|])", r"\\\1", value)


def _notice_bytes(contract: Mapping[str, Any], evidence: Mapping[str, Any]) -> bytes:
    base = evidence["base_model"]
    data = evidence["training_data"]
    text = (
        "LibreYOLO Qwen3-VL detection adapter\n\n"
        "This repository contains a LoRA adapter trained with LibreYOLO and "
        "licensed under Apache-2.0. It does not contain the upstream base-model "
        "weights. It does include Qwen processor, tokenizer, and chat-template "
        "assets redistributed under Apache-2.0.\n\n"
        f"Base model: {base['repo']}\n"
        f"Base revision: {base['revision']}\n"
        f"Base license: {base['license_spdx']}\n"
        f"Base license evidence: {base['license_evidence_url']}\n"
        f"Base weights redistribution: {base['weights_redistribution_decision']}\n"
        "Included processor assets redistribution: "
        f"{base['processor_redistribution_decision']}\n"
        f"Training data source: {data['source']}\n"
        f"Training data version: {data['version']}\n"
        f"Training data split: {data['split']}\n"
        f"Training data license: {data['license_spdx']}\n"
        f"Training data manifest SHA-256: {data['manifest_sha256']}\n"
        f"LibreYOLO code revision: {evidence['code']['revision']}\n"
        f"Model family/size/task: qwen3vl/{contract['size']}/detect\n"
    )
    return text.encode("utf-8")


def _readme_bytes(contract: Mapping[str, Any], evidence: Mapping[str, Any]) -> bytes:
    base = evidence["base_model"]
    data = evidence["training_data"]
    evaluation = evidence["evaluation"]
    metrics = "\n".join(
        f"- {_markdown_inline_code(name)}: {value}"
        for name, value in sorted(evaluation["metrics"].items())
    )
    labels = ", ".join(_markdown_inline_code(name) for name in contract["names"])
    training_label = _markdown_text(f"{data['version']} / {data['split']}")
    base_label = _markdown_inline_code(f"{base['repo']}@{base['revision']}")
    text = (
        "---\n"
        "license: apache-2.0\n"
        "library_name: libreyolo\n"
        "pipeline_tag: image-text-to-text\n"
        f"base_model: {base['repo']}\n"
        "base_model_relation: adapter\n"
        "tags:\n"
        "- object-detection\n"
        "- vlm\n"
        "- peft\n"
        "- lora\n"
        "- libreyolo\n"
        "---\n\n"
        f"# LibreYOLO Qwen3-VL {contract['size'].upper()} detection adapter\n\n"
        "This artifact is a safetensors-only LoRA adapter for LibreYOLO. "
        "It must be validated before it is loaded. The base model is referenced "
        "by an immutable revision and its weights are not redistributed here. "
        "Qwen processor, tokenizer, and chat-template assets from that revision "
        "are included under Apache-2.0.\n\n"
        "## Identity\n\n"
        f"- Base: {base_label}\n"
        f"- Task: {_markdown_inline_code('detect')}\n"
        f"- Vocabulary: {labels}\n"
        f"- Training data: [{training_label}]({data['source']})\n"
        f"- Training data license: {_markdown_inline_code(data['license_spdx'])}\n"
        f"- Evaluation: {_markdown_inline_code(evaluation['benchmark'])}\n\n"
        "## Recorded evaluation\n\n"
        f"{metrics}\n\n"
        "The evaluation record names the adapter digest it was reported against, "
        "but this package does not independently rerun the benchmark. The hashes "
        "and review fields in `publication_evidence.json` are "
        "consistency and provenance records, not cryptographic authentication "
        "of the reviewer or legal advice.\n"
    )
    return text.encode("utf-8")


def _role_for(path: str) -> str:
    return _ROLE_BY_FIXED_PATH.get(path, "processor")


def _fingerprint_file(path: Path, relative: str) -> dict[str, Any]:
    hasher = hashlib.sha256()
    read = 0
    try:
        with _open_stable_regular_file(path, f"artifact file {relative}") as (
            stream,
            opened,
        ):
            while True:
                chunk = stream.read(min(_COPY_CHUNK_BYTES, opened.st_size - read + 1))
                if not chunk:
                    break
                hasher.update(chunk)
                read += len(chunk)
                if read > opened.st_size:
                    raise VLMArtifactError(
                        f"artifact file changed while fingerprinted: {relative}"
                    )
    except OSError as exc:
        raise VLMArtifactError(
            f"Could not fingerprint artifact file {relative}"
        ) from exc
    if read != opened.st_size:
        raise VLMArtifactError(f"artifact file changed while fingerprinted: {relative}")
    return {
        "path": relative,
        "role": _role_for(relative),
        "size": read,
        "sha256": hasher.hexdigest(),
    }


def _build_manifest(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    paths = sorted(
        (path.name for path in root.iterdir() if path.name != VLM_ARTIFACT_MANIFEST),
        key=str.casefold,
    )
    entries = [_fingerprint_file(root / path, path) for path in paths]
    by_role = {
        entry["role"]: entry for entry in entries if entry["role"] != "processor"
    }
    processor_entries = [entry for entry in entries if entry["role"] == "processor"]
    processor_identity = [
        {"path": entry["path"], "size": entry["size"], "sha256": entry["sha256"]}
        for entry in processor_entries
    ]
    identity = {
        "family": "qwen3vl",
        "size": contract["size"],
        "task": "detect",
        "base_repo": contract["base_repo"],
        "base_revision": contract["base_revision"],
        "artifact_license": "Apache-2.0",
        "checkpoint_contract_sha256": by_role["checkpoint_contract"]["sha256"],
        "publication_evidence_sha256": by_role["publication_evidence"]["sha256"],
        "processor_sha256": _aggregate_entries(processor_identity),
        "weights_sha256": by_role["adapter_weights"]["sha256"],
        "base_snapshot": evidence["base_model"]["snapshot"],
    }
    return {
        "schema": VLM_ARTIFACT_SCHEMA,
        "representation": "lora_adapter",
        "identity": identity,
        "files": entries,
        "aggregate_sha256": _aggregate_entries(entries),
    }


def _validate_reviewed_artifact_bindings(
    manifest: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    identity = manifest["identity"]
    bindings = evidence["review"]["bindings"]
    adapter_config_sha = next(
        entry["sha256"]
        for entry in manifest["files"]
        if entry["role"] == "adapter_config"
    )
    expected = {
        "adapter_weights_sha256": identity["weights_sha256"],
        "adapter_config_sha256": adapter_config_sha,
        "checkpoint_contract_sha256": identity["checkpoint_contract_sha256"],
        "processor_sha256": identity["processor_sha256"],
    }
    mismatches = [key for key, value in expected.items() if bindings[key] != value]
    if mismatches:
        raise VLMArtifactError(
            "publication review does not bind the built artifact fields: "
            + ", ".join(mismatches)
        )


def _validate_manifest_structure(value: Any) -> dict[str, Any]:
    manifest = _require_exact_keys(value, _MANIFEST_KEYS, "VLM artifact manifest")
    if manifest["schema"] != VLM_ARTIFACT_SCHEMA:
        raise VLMArtifactError(f"manifest schema must be {VLM_ARTIFACT_SCHEMA!r}")
    if manifest["representation"] != "lora_adapter":
        raise VLMArtifactError("manifest representation must be 'lora_adapter'")
    identity = _require_exact_keys(
        manifest["identity"], _IDENTITY_KEYS, "manifest identity"
    )
    if identity["family"] != "qwen3vl" or identity["size"] not in _SUPPORTED_BASES:
        raise VLMArtifactError("manifest identity supports only Qwen3-VL 2B/4B")
    if identity["task"] != "detect" or identity["artifact_license"] != "Apache-2.0":
        raise VLMArtifactError("manifest task/license identity is invalid")
    repo, revision = _SUPPORTED_BASES[identity["size"]]
    if identity["base_repo"] != repo or identity["base_revision"] != revision:
        raise VLMArtifactError(
            "manifest base identity does not match the immutable pin"
        )
    for key in (
        "checkpoint_contract_sha256",
        "publication_evidence_sha256",
        "processor_sha256",
        "weights_sha256",
    ):
        _require_sha256(identity[key], f"manifest identity {key}")
    _validate_snapshot_files(
        identity["base_snapshot"],
        expected_repo=repo,
        expected_revision=revision,
    )

    raw_files = manifest["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise VLMArtifactError("manifest files must be a nonempty list")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_size = 0
    role_counts: dict[str, int] = {role: 0 for role in _ALL_ROLES}
    for index, raw in enumerate(raw_files):
        entry = _require_exact_keys(raw, _FILE_KEYS, f"manifest files[{index}]")
        path = _safe_inventory_path(
            entry["path"], f"manifest file {index} path", flat=True
        )
        folded = path.casefold()
        if folded in seen:
            raise VLMArtifactError("manifest paths must be unique case-insensitively")
        seen.add(folded)
        role = entry["role"]
        if role not in _ALL_ROLES or role != _role_for(path):
            raise VLMArtifactError(f"manifest file {path} has an invalid role")
        size = entry["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise VLMArtifactError(f"manifest file {path} size must be positive")
        limit = _ARTIFACT_FILE_LIMITS.get(path)
        if limit is None or size > limit:
            raise VLMArtifactError(
                f"manifest file {path} exceeds its {limit}-byte safety limit"
            )
        total_size += size
        digest = _require_sha256(entry["sha256"], f"manifest file {path} sha256")
        if path == VLM_ARTIFACT_MANIFEST:
            raise VLMArtifactError("manifest must not inventory itself")
        entries.append({"path": path, "role": role, "size": size, "sha256": digest})
        role_counts[role] += 1
    if total_size > VLM_ARTIFACT_MAX_PAYLOAD_BYTES:
        raise VLMArtifactError(
            "manifest payload exceeds the aggregate artifact safety limit"
        )
    if [entry["path"] for entry in entries] != sorted(
        (entry["path"] for entry in entries), key=str.casefold
    ):
        raise VLMArtifactError("manifest file inventory must be sorted by path")
    if any(role_counts[role] != 1 for role in _REQUIRED_SINGLE_ROLES):
        raise VLMArtifactError(
            "manifest must contain exactly one file for every required role"
        )
    if role_counts["processor"] < 3:
        raise VLMArtifactError("manifest requires a complete processor inventory")
    paths = {entry["path"] for entry in entries}
    required = _REQUIRED_INPUT_FILES | _GENERATED_FILES
    if not required <= paths:
        raise VLMArtifactError(
            f"manifest is missing required paths: {sorted(required - paths)}"
        )
    allowed = required | _PROCESSOR_FILES
    if paths - allowed:
        raise VLMArtifactError(
            f"manifest contains unsupported paths: {sorted(paths - allowed)}"
        )
    if "tokenizer.json" not in paths and not {"vocab.json", "merges.txt"} <= paths:
        raise VLMArtifactError("manifest processor inventory is missing tokenizer data")
    if not {"preprocessor_config.json", "processor_config.json"} & paths:
        raise VLMArtifactError("manifest processor inventory is missing processor data")
    by_path = {entry["path"]: entry for entry in entries}
    if identity["checkpoint_contract_sha256"] != by_path[_CONTRACT_FILENAME]["sha256"]:
        raise VLMArtifactError(
            "manifest contract identity does not match its file entry"
        )
    if (
        identity["publication_evidence_sha256"]
        != by_path[PUBLICATION_EVIDENCE_FILENAME]["sha256"]
    ):
        raise VLMArtifactError(
            "manifest evidence identity does not match its file entry"
        )
    if identity["weights_sha256"] != by_path[_ADAPTER_WEIGHTS_FILENAME]["sha256"]:
        raise VLMArtifactError(
            "manifest weights identity does not match its file entry"
        )
    processor_identity = [
        {"path": entry["path"], "size": entry["size"], "sha256": entry["sha256"]}
        for entry in entries
        if entry["role"] == "processor"
    ]
    expected_processor_identity = [
        {"path": path, "size": size_bytes, "sha256": digest}
        for path, size_bytes, digest in _CANONICAL_PROCESSOR_FILES[identity["size"]]
    ]
    if processor_identity != expected_processor_identity:
        raise VLMArtifactError(
            "manifest processor identity does not match the audited Qwen serialization"
        )
    if identity["processor_sha256"] != _aggregate_entries(processor_identity):
        raise VLMArtifactError("manifest processor identity does not match its files")
    if manifest["aggregate_sha256"] != _aggregate_entries(entries):
        raise VLMArtifactError("manifest aggregate_sha256 does not match its files")
    return manifest


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(nested) for key, nested in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(nested) for nested in value)
    return value


def _manifest_info(root: Path, manifest: Mapping[str, Any]) -> VLMArtifactInfo:
    frozen = _deep_freeze(manifest)
    return VLMArtifactInfo(
        root=root,
        manifest=frozen,
        aggregate_sha256=manifest["aggregate_sha256"],
        files=tuple(entry["path"] for entry in manifest["files"]),
        base_snapshot=frozen["identity"]["base_snapshot"],
    )


def _validate_payload(root: Path, manifest: Mapping[str, Any]) -> None:
    actual = _scan_flat_directory(root, "VLM artifact")
    expected_paths = {entry["path"] for entry in manifest["files"]} | {
        VLM_ARTIFACT_MANIFEST
    }
    if set(actual) != expected_paths:
        missing = sorted(expected_paths - set(actual))
        extra = sorted(set(actual) - expected_paths)
        raise VLMArtifactError(
            f"artifact inventory mismatch: missing={missing}, extra={extra}"
        )
    for entry in manifest["files"]:
        observed = _fingerprint_file(actual[entry["path"]], entry["path"])
        if observed != entry:
            raise VLMArtifactError(
                f"artifact file does not match manifest: {entry['path']}"
            )

    manifest_path = root / VLM_ARTIFACT_MANIFEST
    manifest_bytes = _read_bounded(
        manifest_path,
        max_bytes=VLM_ARTIFACT_MAX_MANIFEST_BYTES,
        label="VLM artifact manifest",
    )
    if manifest_bytes != _json_file_bytes(manifest):
        raise VLMArtifactError("VLM artifact manifest must use canonical JSON")

    contract_path = root / _CONTRACT_FILENAME
    contract = _validate_contract(_load_json(contract_path, "VLM checkpoint contract"))
    if _read_bounded(
        contract_path, max_bytes=_MAX_JSON_BYTES, label="VLM checkpoint contract"
    ) != _json_file_bytes(contract):
        raise VLMArtifactError("VLM checkpoint contract must use canonical JSON")
    identity = manifest["identity"]
    for key in ("family", "size", "task", "base_repo", "base_revision"):
        if identity[key] != contract[key]:
            raise VLMArtifactError(
                f"manifest identity {key} does not match the contract"
            )

    evidence_path = root / PUBLICATION_EVIDENCE_FILENAME
    evidence = _validate_publication_evidence(
        _load_json(evidence_path, "publication evidence"),
        expected_size=contract["size"],
    )
    if _read_bounded(
        evidence_path, max_bytes=_MAX_JSON_BYTES, label="publication evidence"
    ) != _json_file_bytes(evidence):
        raise VLMArtifactError("publication evidence must use canonical JSON")
    if evidence["code"]["dependencies"]["libreyolo"] != contract["libreyolo_version"]:
        raise VLMArtifactError(
            "publication evidence LibreYOLO version does not match contract"
        )
    if evidence["base_model"]["snapshot"] != identity["base_snapshot"]:
        raise VLMArtifactError(
            "manifest base snapshot identity does not match publication evidence"
        )
    _validate_reviewed_artifact_bindings(manifest, evidence)

    adapter_path = root / _ADAPTER_CONFIG_FILENAME
    adapter = _load_json(adapter_path, "adapter_config.json")
    canonical_adapter = _canonical_adapter_config(adapter, contract)
    if canonical_adapter["peft_version"] != evidence["code"]["dependencies"]["peft"]:
        raise VLMArtifactError(
            "adapter_config.json peft_version does not match publication evidence"
        )
    if _read_bounded(
        adapter_path, max_bytes=_MAX_JSON_BYTES, label="adapter_config.json"
    ) != _json_file_bytes(canonical_adapter):
        raise VLMArtifactError("adapter_config.json is not canonical")
    _validate_safetensors(root / _ADAPTER_WEIGHTS_FILENAME, contract["size"])

    processor_files = {
        entry["path"] for entry in manifest["files"] if entry["role"] == "processor"
    }
    _validate_processor_files(root, processor_files, contract["size"])
    for name in processor_files:
        if name.endswith(".json"):
            _load_json(root / name, name)

    expected_generated = {
        _GITATTRIBUTES_FILENAME: _gitattributes_bytes(),
        _APACHE_LICENSE_FILENAME: _apache_license_bytes(),
        _NOTICE_FILENAME: _notice_bytes(contract, evidence),
        _README_FILENAME: _readme_bytes(contract, evidence),
    }
    for name, expected in expected_generated.items():
        observed = _read_bounded(
            root / name, max_bytes=max(len(expected), 1), label=name
        )
        if observed != expected:
            raise VLMArtifactError(f"generated artifact file is not canonical: {name}")


def _scan_snapshot_tree(root: Path) -> tuple[dict[str, Path], set[str]]:
    """Return security-relevant snapshot files and directories.

    Hugging Face's root ``.cache`` directory and LibreYOLO's completion marker
    are transport metadata, not model inputs, so they are deliberately outside
    the immutable snapshot identity.
    """

    files: dict[str, Path] = {}
    directories: set[str] = set()
    pending = [root]
    folded: set[str] = set()
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(
                os.scandir(directory), key=lambda entry: entry.name.casefold()
            )
        except OSError as exc:
            raise VLMArtifactError(
                f"Could not inspect base snapshot: {directory}"
            ) from exc
        for entry in entries:
            path = directory / entry.name
            relative = path.relative_to(root).as_posix()
            if relative == ".cache" and entry.is_dir(follow_symlinks=False):
                if _is_link_or_junction(path):
                    raise VLMArtifactError("base snapshot .cache must not be a link")
                continue
            if relative == ".libreyolo_snapshot_complete":
                _assert_regular_unlinked_file(path, "base snapshot completion marker")
                continue
            if _is_link_or_junction(path):
                raise VLMArtifactError(
                    f"base snapshot must not contain a symlink or junction: {relative}"
                )
            try:
                identity = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise VLMArtifactError(
                    f"Could not inspect base snapshot entry: {relative}"
                ) from exc
            if stat.S_ISDIR(identity.st_mode):
                canonical = _safe_inventory_path(
                    relative, "base snapshot directory", flat=False
                )
                directories.add(canonical)
                pending.append(path)
                continue
            if not stat.S_ISREG(identity.st_mode):
                raise VLMArtifactError(
                    f"base snapshot contains a non-regular entry: {relative}"
                )
            canonical = _safe_inventory_path(relative, "base snapshot file", flat=False)
            casefolded = canonical.casefold()
            if casefolded in folded:
                raise VLMArtifactError(
                    "base snapshot contains case-ambiguous file paths"
                )
            folded.add(casefolded)
            _assert_regular_unlinked_file(path, f"base snapshot file {canonical}")
            files[canonical] = path
    return files, directories


def _snapshot_file_record(path: Path, relative: str) -> dict[str, Any]:
    fingerprint = _fingerprint_file(path, relative)
    return {
        "path": relative,
        "size": fingerprint["size"],
        "sha256": fingerprint["sha256"],
    }


def validate_vlm_base_snapshot(
    root: str | os.PathLike[str], expected_identity: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Validate an exact local base snapshot without network or model loading.

    ``expected_identity`` is ``VLMArtifactInfo.base_snapshot`` (also available
    at ``manifest["identity"]["base_snapshot"]``).  Only Hub cache metadata and
    LibreYOLO's completion marker are excluded from the exact-tree comparison.
    """

    if not isinstance(expected_identity, Mapping):
        raise TypeError("expected_identity must be a mapping")
    expected = dict(expected_identity)
    expected_files, _ = _validate_snapshot_files(expected)
    source = expected["source"]
    revision = expected["revision"]
    if (source, revision) not in _SUPPORTED_BASES.values():
        raise VLMArtifactError("base snapshot identity is not a supported Qwen3-VL pin")
    snapshot_root = _required_directory(root, "VLM base snapshot")
    actual_files, actual_directories = _scan_snapshot_tree(snapshot_root)
    expected_paths = {entry["path"] for entry in expected_files}
    if set(actual_files) != expected_paths:
        missing = sorted(expected_paths - set(actual_files))
        extra = sorted(set(actual_files) - expected_paths)
        raise VLMArtifactError(
            f"base snapshot inventory mismatch: missing={missing}, extra={extra}"
        )
    expected_directories = {
        parent.as_posix()
        for path in expected_paths
        for parent in PurePosixPath(path).parents
        if parent.as_posix() != "."
    }
    if actual_directories != expected_directories:
        raise VLMArtifactError(
            "base snapshot contains unexpected or missing directories"
        )
    actual_records = [
        _snapshot_file_record(actual_files[entry["path"]], entry["path"])
        for entry in expected_files
    ]
    if actual_records != expected_files:
        raise VLMArtifactError("base snapshot files do not match the expected identity")
    marker = snapshot_root / ".libreyolo_snapshot_complete"
    if _path_exists(marker):
        marker_value = _load_json(marker, "base snapshot completion marker")
        if marker_value != {"repo": source, "revision": revision}:
            raise VLMArtifactError(
                "base snapshot completion marker does not match the expected identity"
            )
    return MappingProxyType(expected)


def read_vlm_artifact_manifest(
    path_or_dir: str | os.PathLike[str], *, require_payload: bool = False
) -> VLMArtifactInfo:
    """Strictly read an artifact manifest, optionally validating every payload.

    ``require_payload=False`` is the manifest-first transport seam.  The
    manifest may be the only file present.  No network or model dependency is
    imported or called.  ``require_payload=True`` requires an exact artifact
    directory and is equivalent to :func:`validate_vlm_artifact`.
    """

    candidate = _path_argument(path_or_dir, "VLM artifact")
    direct_file = candidate.name == VLM_ARTIFACT_MANIFEST and _path_exists(candidate)
    if direct_file:
        manifest_path = _required_file(candidate, "VLM artifact manifest")
        root = manifest_path.parent
    else:
        root = _required_directory(candidate, "VLM artifact")
        manifest_path = _required_file(
            root / VLM_ARTIFACT_MANIFEST, "VLM artifact manifest"
        )
    raw = _read_bounded(
        manifest_path,
        max_bytes=VLM_ARTIFACT_MAX_MANIFEST_BYTES,
        label="VLM artifact manifest",
    )
    manifest = _validate_manifest_structure(_decode_json(raw, "VLM artifact manifest"))
    if raw != _json_file_bytes(manifest):
        raise VLMArtifactError("VLM artifact manifest must use canonical JSON")
    if require_payload:
        if direct_file:
            root = _required_directory(root, "VLM artifact")
        _validate_payload(root, manifest)
    return _manifest_info(root, manifest)


def validate_vlm_artifact(path: str | os.PathLike[str]) -> VLMArtifactInfo:
    """Validate a complete offline VLM artifact directory."""

    return read_vlm_artifact_manifest(path, require_payload=True)


def create_vlm_publication_evidence_template(
    checkpoint_dir: str | os.PathLike[str],
    base_snapshot_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    training_data: Mapping[str, Any],
    code: Mapping[str, Any],
    confidence_report: str | os.PathLike[str],
    repeatability_receipt: str | os.PathLike[str],
) -> Path:
    """Create an exact, deliberately unapproved publication-evidence template.

    The checkpoint, pinned base snapshot, strict confidence report, strict
    two-run repeatability receipt, and all byte-derived review bindings are
    validated before a complete canonical JSON file is published. Evaluation
    claims come only from the primary report and its exact-zero comparison
    receipt. Legal, privacy, evaluation, and provenance approvals remain false
    or unreviewed; the returned file therefore cannot authorize
    :func:`build_vlm_artifact` until a human reviews and edits those fields
    outside this helper.
    """

    source = _required_directory(checkpoint_dir, "VLM checkpoint")
    base_root = _required_directory(base_snapshot_dir, "VLM base snapshot")
    destination = _new_file_destination(
        output_path, "VLM publication evidence template output"
    )
    for protected_root, label in (
        (source, "checkpoint"),
        (base_root, "base snapshot"),
    ):
        if destination == protected_root or protected_root in destination.parents:
            raise VLMArtifactError(
                "publication evidence template must remain outside the " + label
            )

    checkpoint_identity = _inspect_checkpoint_identity(source)
    source_files = _validate_checkpoint_inventory(source)
    contract = _validate_contract(
        _load_json(source_files[_CONTRACT_FILENAME], "VLM checkpoint contract")
    )
    if (
        _sha256_bytes(_json_file_bytes(contract))
        != checkpoint_identity.checkpoint_contract_sha256
    ):
        raise VLMArtifactError(
            "VLM checkpoint contract changed after identity inspection"
        )
    adapter = _canonical_adapter_config(
        _load_json(source_files[_ADAPTER_CONFIG_FILENAME], "adapter_config.json"),
        contract,
    )
    if (
        _sha256_bytes(_json_file_bytes(adapter))
        != checkpoint_identity.adapter_config_sha256
    ):
        raise VLMArtifactError(
            "adapter_config.json changed after checkpoint identity inspection"
        )
    _validate_safetensors(source_files[_ADAPTER_WEIGHTS_FILENAME], contract["size"])
    processor_paths = set(source_files) & _PROCESSOR_FILES
    _validate_processor_files(source, processor_paths, contract["size"])

    base_identity = _canonical_base_snapshot(contract["size"])
    validate_vlm_base_snapshot(base_root, base_identity)
    training_record, code_record = _validate_template_context(training_data, code)
    (
        derived_evaluation_record,
        confidence_run_identity,
        primary_run,
    ) = _evaluation_from_confidence_report(confidence_report, checkpoint_identity)
    repeatability_claim, repeatability_identity = _repeatability_claim_from_receipt(
        repeatability_receipt, primary_run
    )
    dependencies = code_record["dependencies"]
    if dependencies["libreyolo"] != contract["libreyolo_version"]:
        raise VLMArtifactError(
            "code.dependencies.libreyolo does not match the checkpoint contract"
        )
    if dependencies["peft"] != adapter["peft_version"]:
        raise VLMArtifactError(
            "code.dependencies.peft does not match adapter_config.json"
        )

    weights_sha = checkpoint_identity.adapter_weights_sha256
    adapter_config_sha = checkpoint_identity.adapter_config_sha256
    contract_sha = checkpoint_identity.checkpoint_contract_sha256
    processor_sha = checkpoint_identity.processor_sha256
    recipe_sha = _recipe_sha256()
    repo, revision = _SUPPORTED_BASES[contract["size"]]
    training_manifest_sha = training_record["manifest_sha256"]
    evaluation_report_sha = derived_evaluation_record["report_sha256"]
    evaluation_envelope_sha = derived_evaluation_record["envelope_sha256"]
    code_revision = code_record["revision"]
    evaluation_record = {
        **derived_evaluation_record,
        "checkpoint_sha256": weights_sha,
        "repeatability": repeatability_claim,
        "passed": False,
    }
    evaluation_claim_sha = _evaluation_claim_sha256(evaluation_record)

    template = {
        "schema": PUBLICATION_EVIDENCE_SCHEMA,
        "artifact_license": {
            "spdx": "Apache-2.0",
            "redistribution_decision": "unreviewed",
        },
        "base_model": {
            "repo": repo,
            "revision": revision,
            "license_spdx": "Apache-2.0",
            "license_evidence_url": (
                f"https://huggingface.co/{repo}/blob/{revision}/README.md"
            ),
            "weights_redistribution_decision": "reference-only",
            "processor_redistribution_decision": "unreviewed",
            "snapshot": base_identity,
        },
        "training_data": {
            **training_record,
            "redistribution_decision": "unreviewed",
        },
        "evaluation": evaluation_record,
        "code": {
            "repository": "https://github.com/LibreYOLO/libreyolo",
            "revision": code_revision,
            "clean": code_record["clean"],
            "recipe": {"id": "qwen3vl-lora-v1", "sha256": recipe_sha},
            "dependencies": dependencies,
        },
        "review": {
            "approved": False,
            "reviewer": "",
            "reviewed_at": "",
            "bindings": {
                "base_snapshot_sha256": base_identity["sha256"],
                "training_data_manifest_sha256": training_manifest_sha,
                "evaluation_report_sha256": evaluation_report_sha,
                "evaluation_envelope_sha256": evaluation_envelope_sha,
                "evaluation_repeatability_receipt_sha256": repeatability_claim[
                    "receipt_sha256"
                ],
                "evaluation_repeatability_comparison_sha256": repeatability_claim[
                    "comparison_sha256"
                ],
                "evaluation_claim_sha256": evaluation_claim_sha,
                "code_revision": code_revision,
                "recipe_sha256": recipe_sha,
                "adapter_weights_sha256": weights_sha,
                "adapter_config_sha256": adapter_config_sha,
                "checkpoint_contract_sha256": contract_sha,
                "processor_sha256": processor_sha,
            },
            "gates": {key: False for key in sorted(_GATE_KEYS)},
        },
    }
    payload = _json_file_bytes(template)
    if len(payload) > _ARTIFACT_FILE_LIMITS[PUBLICATION_EVIDENCE_FILENAME]:
        raise VLMArtifactError("publication evidence template exceeds its safety limit")
    if _inspect_checkpoint_identity(source) != checkpoint_identity:
        raise VLMArtifactError(
            "VLM checkpoint changed while publication evidence was prepared"
        )
    (
        rechecked_evaluation,
        rechecked_run_identity,
        rechecked_primary_run,
    ) = _evaluation_from_confidence_report(confidence_report, checkpoint_identity)
    rechecked_repeatability, rechecked_repeatability_identity = (
        _repeatability_claim_from_receipt(repeatability_receipt, rechecked_primary_run)
    )
    if (
        rechecked_evaluation != derived_evaluation_record
        or rechecked_run_identity != confidence_run_identity
        or rechecked_repeatability != repeatability_claim
        or rechecked_repeatability_identity != repeatability_identity
    ):
        raise VLMArtifactError(
            "confidence benchmark or repeatability receipt changed while publication "
            "evidence was prepared"
        )
    _write_bytes_atomic_create_only(
        destination,
        payload,
        label="VLM publication evidence template output",
    )
    return destination


def build_vlm_artifact(
    checkpoint_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    publication_evidence: str | os.PathLike[str],
) -> VLMArtifactInfo:
    """Build a deterministic, create-only Qwen3-VL LoRA publication artifact."""

    source = _required_directory(checkpoint_dir, "VLM checkpoint")
    destination = _new_directory_destination(output_dir, "VLM artifact output")
    _assert_disjoint(source, destination)
    evidence_path = _required_file(publication_evidence, "publication evidence")
    if evidence_path == source or source in evidence_path.parents:
        raise VLMArtifactError(
            "publication evidence must be external to the checkpoint"
        )
    if (
        os.lstat(evidence_path).st_size
        > _ARTIFACT_FILE_LIMITS[PUBLICATION_EVIDENCE_FILENAME]
    ):
        raise VLMArtifactError("publication evidence exceeds its safety limit")

    source_files = _validate_checkpoint_inventory(source)
    contract = _validate_contract(
        _load_json(source_files[_CONTRACT_FILENAME], "VLM checkpoint contract")
    )
    evidence = _validate_publication_evidence(
        _load_json(evidence_path, "publication evidence"),
        expected_size=contract["size"],
        enforce_current_recipe=True,
    )
    if evidence["code"]["dependencies"]["libreyolo"] != contract["libreyolo_version"]:
        raise VLMArtifactError(
            "publication evidence LibreYOLO version does not match contract"
        )
    adapter = _canonical_adapter_config(
        _load_json(source_files[_ADAPTER_CONFIG_FILENAME], "adapter_config.json"),
        contract,
    )
    if adapter["peft_version"] != evidence["code"]["dependencies"]["peft"]:
        raise VLMArtifactError(
            "adapter_config.json peft_version does not match publication evidence"
        )
    _validate_safetensors(source_files[_ADAPTER_WEIGHTS_FILENAME], contract["size"])
    _validate_processor_files(
        source, set(source_files) & _PROCESSOR_FILES, contract["size"]
    )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    staging_identity = os.lstat(staging)
    staging_seal = (staging_identity.st_dev, staging_identity.st_ino)
    lock_path = destination.with_name(f".{destination.name}.create.lock")
    lock_fd: int | None = None
    lock_seal: tuple[int, int] | None = None
    published = False
    try:
        for name in sorted(source_files, key=str.casefold):
            source_path = source_files[name]
            target = staging / name
            if name == _CONTRACT_FILENAME:
                _write_create_only(target, _json_file_bytes(contract))
            elif name == _ADAPTER_CONFIG_FILENAME:
                _write_create_only(target, _json_file_bytes(adapter))
            elif name in _PROCESSOR_FILES:
                _copy_file_stable(source_path, target)
            elif name.endswith(".json"):
                _canonicalize_json_file(source_path, target, name)
            else:
                _copy_file_stable(source_path, target)

        _write_create_only(
            staging / PUBLICATION_EVIDENCE_FILENAME, _json_file_bytes(evidence)
        )
        _write_create_only(staging / _GITATTRIBUTES_FILENAME, _gitattributes_bytes())
        _write_create_only(staging / _APACHE_LICENSE_FILENAME, _apache_license_bytes())
        _write_create_only(
            staging / _NOTICE_FILENAME, _notice_bytes(contract, evidence)
        )
        _write_create_only(
            staging / _README_FILENAME, _readme_bytes(contract, evidence)
        )

        manifest = _build_manifest(staging, contract, evidence)
        _validate_reviewed_artifact_bindings(manifest, evidence)
        _write_create_only(staging / VLM_ARTIFACT_MANIFEST, _json_file_bytes(manifest))
        staged_info = validate_vlm_artifact(staging)

        try:
            lock_fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            locked = os.fstat(lock_fd)
            lock_seal = (locked.st_dev, locked.st_ino)
        except FileExistsError as exc:
            raise FileExistsError(
                f"artifact publication is already in progress: {destination}"
            ) from exc
        if _path_exists(destination):
            raise FileExistsError(f"VLM artifact output already exists: {destination}")
        try:
            _rename_create_only(staging, destination, staging_seal)
        except FileExistsError:
            raise
        except OSError as exc:
            if _path_exists(destination):
                raise FileExistsError(
                    f"VLM artifact output already exists: {destination}"
                ) from exc
            raise VLMArtifactError(
                f"Could not publish VLM artifact: {destination}"
            ) from exc
        published = True

        def require_published_seal() -> None:
            try:
                current = os.lstat(destination)
            except OSError as exc:
                raise VLMArtifactError(
                    "published artifact directory changed after publication"
                ) from exc
            if (
                _is_link_or_junction(destination)
                or not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != staging_seal
            ):
                raise VLMArtifactError(
                    "published artifact directory changed after publication"
                )

        require_published_seal()
        published_info = validate_vlm_artifact(destination)
        require_published_seal()
        if (
            published_info.manifest != staged_info.manifest
            or published_info.aggregate_sha256 != staged_info.aggregate_sha256
            or published_info.files != staged_info.files
            or published_info.base_snapshot != staged_info.base_snapshot
        ):
            raise VLMArtifactError(
                "published artifact identity differs from the validated staging artifact"
            )
        return published_info
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_seal is not None and _path_exists(lock_path):
            try:
                locked = os.lstat(lock_path)
                if (
                    not _is_link_or_junction(lock_path)
                    and (locked.st_dev, locked.st_ino) == lock_seal
                ):
                    lock_path.unlink()
            except OSError:
                pass
        if not published and _path_exists(staging):
            try:
                current = os.lstat(staging)
                if (
                    not _is_link_or_junction(staging)
                    and stat.S_ISDIR(current.st_mode)
                    and (current.st_dev, current.st_ino) == staging_seal
                ):
                    shutil.rmtree(staging, ignore_errors=True)
            except OSError:
                pass
