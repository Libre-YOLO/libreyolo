"""Strict local identity tests for publishable Qwen LoRA checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from libreyolo.models.vlm import artifact as artifact_module
from libreyolo.models.vlm.artifact import VLMArtifactError
from libreyolo.models.vlm.training.checkpoint import (
    VLM_CHECKPOINT_IDENTITY_SCHEMA,
    VLMCheckpointFileIdentity,
    VLMCheckpointIdentity,
    inspect_vlm_checkpoint_identity,
    save_vlm_checkpoint,
)

pytestmark = pytest.mark.unit

_REPO = "Qwen/Qwen3-VL-2B-Instruct"
_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def _canonical(value) -> bytes:
    return artifact_module._json_file_bytes(value)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _prompt(names: list[str]) -> str:
    labels = ", ".join(names)
    return (
        f"Detect all instances of: {labels}. "
        "Output the result as a JSON array, one object per instance: "
        '[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
        "Only include objects that are actually visible; if there are none, "
        "respond with an empty array []."
    )


def _contract() -> dict:
    names = ["ripe strawberry", "worker"]
    return {
        "schema": 1,
        "family": "qwen3vl",
        "size": "2b",
        "base_repo": _REPO,
        "base_revision": _REVISION,
        "names": names,
        "bbox_key": "bbox_2d",
        "coord_divisor": 1000.0,
        "box_format": "xyxy",
        "prompt": _prompt(names),
        "task": "detect",
        "metrics": {"epoch": 0, "train/loss": 0.25},
        "libreyolo_version": "1.6.0",
    }


def _adapter_config() -> dict:
    return {
        "alora_invocation_tokens": None,
        "alpha_pattern": {},
        "arrow_config": None,
        "auto_mapping": {
            "base_model_class": "Qwen3VLForConditionalGeneration",
            "parent_library": "transformers.models.qwen3_vl.modeling_qwen3_vl",
        },
        "base_model_name_or_path": "weights\\LibreQwen3VL2b",
        "bias": "none",
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
        "lora_alpha": 32,
        "lora_bias": False,
        "lora_dropout": 0.05,
        "lora_ga_config": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "peft_version": "0.19.1",
        "qalora_group_size": 16,
        "r": 16,
        "rank_pattern": {},
        "revision": None,
        "target_modules": (
            r".*language_model.*\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        ),
        "target_parameters": None,
        "task_type": None,
        "trainable_token_indices": None,
        "use_bdlora": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }


def _processor_payloads() -> dict[str, bytes]:
    template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    return {
        "chat_template.jinja": template.encode(),
        "processor_config.json": _canonical(
            {
                "image_processor": {"image_processor_type": "Qwen2VLImageProcessor"},
                "processor_class": "Qwen3VLProcessor",
                "video_processor": {"video_processor_type": "Qwen3VLVideoProcessor"},
            }
        ),
        "tokenizer.json": _canonical({"model": {}, "version": "1.0"}),
        "tokenizer_config.json": _canonical(
            {
                "chat_template": template,
                "tokenizer_class": "Qwen2TokenizerFast",
            }
        ),
    }


def _safetensor() -> bytes:
    header = {}
    offset = 0
    layout = artifact_module._QWEN_LORA_LAYOUT["2b"]
    for layer in range(layout["layers"]):
        for scope, modules in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        ):
            for module in modules:
                stem = (
                    "base_model.model.model.language_model.layers."
                    f"{layer}.{scope}.{module}"
                )
                shape_a, shape_b = artifact_module._expected_lora_shapes("2b", module)
                for side, shape in (("A", shape_a), ("B", shape_b)):
                    tensor_bytes = 2
                    for dimension in shape:
                        tensor_bytes *= dimension
                    header[f"{stem}.lora_{side}.weight"] = {
                        "dtype": "BF16",
                        "shape": shape,
                        "data_offsets": [offset, offset + tensor_bytes],
                    }
                    offset += tensor_bytes
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    return struct.pack("<Q", len(raw)) + raw + b"\x00" * offset


@pytest.fixture(autouse=True)
def _compact_strict_payloads(monkeypatch):
    layout = {"2b": {"layers": 1, "hidden": 1, "q": 1, "kv": 1, "intermediate": 1}}
    monkeypatch.setattr(artifact_module, "_QWEN_LORA_LAYOUT", layout)
    processor_records = tuple(
        (name, len(payload), _sha(payload))
        for name, payload in sorted(_processor_payloads().items())
    )
    monkeypatch.setattr(
        artifact_module, "_CANONICAL_PROCESSOR_FILES", {"2b": processor_records}
    )


def _write_checkpoint(root: Path) -> Path:
    root.mkdir()
    # Deliberately mirrors writer-shaped pretty JSON, not artifact-canonical JSON.
    (root / "libreyolo_vlm.json").write_text(
        json.dumps(_contract(), indent=2), encoding="utf-8"
    )
    (root / "adapter_config.json").write_text(
        json.dumps(_adapter_config(), indent=2), encoding="utf-8"
    )
    (root / "adapter_model.safetensors").write_bytes(_safetensor())
    for name, payload in _processor_payloads().items():
        (root / name).write_bytes(payload)
    (root / "README.md").write_text("# PEFT source card\n", encoding="utf-8")
    return root


@pytest.fixture
def checkpoint(tmp_path) -> Path:
    return _write_checkpoint(tmp_path / "checkpoint")


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}


def test_accepts_checkpoint_emitted_through_the_local_writer(tmp_path):
    class Saveable:
        def __init__(self, payloads):
            self.payloads = payloads

        def save_pretrained(self, directory):
            for name, payload in self.payloads.items():
                path = Path(directory) / name
                if isinstance(payload, bytes):
                    path.write_bytes(payload)
                else:
                    path.write_text(payload, encoding="utf-8")

    class Wrapper:
        FAMILY = "qwen3vl"
        HF_REPOS = {"2b": _REPO}
        HF_REVISIONS = {"2b": _REVISION}
        BBOX_KEY = "bbox_2d"
        COORD_DIVISOR = 1000.0
        BOX_FORMAT = "xyxy"
        size = "2b"
        names = {0: "ripe strawberry", 1: "worker"}

        @staticmethod
        def _detection_prompt():
            return _prompt(["ripe strawberry", "worker"])

    target = tmp_path / "writer-checkpoint"
    save_vlm_checkpoint(
        target,
        peft_model=Saveable(
            {
                "adapter_config.json": json.dumps(_adapter_config(), indent=2),
                "adapter_model.safetensors": _safetensor(),
                "README.md": "# PEFT source card\n",
            }
        ),
        processor=Saveable(_processor_payloads()),
        wrapper=Wrapper(),
        metrics={"train/loss": 0.25},
    )

    contract_raw = (target / "libreyolo_vlm.json").read_bytes()
    adapter_raw = (target / "adapter_config.json").read_bytes()
    assert contract_raw != _canonical(json.loads(contract_raw))
    assert adapter_raw != _canonical(json.loads(adapter_raw))
    identity = inspect_vlm_checkpoint_identity(target)
    assert (identity.family, identity.size, identity.base_revision) == (
        "qwen3vl",
        "2b",
        _REVISION,
    )


def test_inspect_returns_frozen_bounded_identity_without_side_effects(checkpoint):
    before = _tree_bytes(checkpoint)
    identity = inspect_vlm_checkpoint_identity(checkpoint)

    assert isinstance(identity, VLMCheckpointIdentity)
    assert identity.root == checkpoint.resolve()
    assert (identity.family, identity.size, identity.task) == (
        "qwen3vl",
        "2b",
        "detect",
    )
    assert (identity.base_repo, identity.base_revision) == (_REPO, _REVISION)
    assert all(
        isinstance(record, VLMCheckpointFileIdentity) for record in identity.files
    )
    assert tuple(record.path for record in identity.files) == tuple(
        sorted((set(before) - {"README.md"}), key=str.casefold)
    )
    assert {record.role for record in identity.files} == {
        "adapter_config",
        "adapter_weights",
        "checkpoint_contract",
        "processor",
    }
    assert "README.md" not in {record.path for record in identity.files}
    assert identity.adapter_weights_sha256 == _sha(before["adapter_model.safetensors"])
    assert identity.checkpoint_contract_sha256 == _sha(_canonical(_contract()))
    normalized_adapter = artifact_module._canonical_adapter_config(
        _adapter_config(), _contract()
    )
    assert identity.adapter_config_sha256 == _sha(_canonical(normalized_adapter))
    processor_records = [
        {"path": record.path, "size": record.size, "sha256": record.sha256}
        for record in identity.files
        if record.role == "processor"
    ]
    assert identity.processor_sha256 == artifact_module._aggregate_entries(
        processor_records
    )
    aggregate_payload = {
        "schema": VLM_CHECKPOINT_IDENTITY_SCHEMA,
        "family": identity.family,
        "size": identity.size,
        "task": identity.task,
        "base_repo": identity.base_repo,
        "base_revision": identity.base_revision,
        "files": [asdict(record) for record in identity.files],
        "adapter_weights_sha256": identity.adapter_weights_sha256,
        "adapter_config_sha256": identity.adapter_config_sha256,
        "checkpoint_contract_sha256": identity.checkpoint_contract_sha256,
        "processor_sha256": identity.processor_sha256,
    }
    assert identity.aggregate_sha256 == _sha(
        artifact_module._canonical_json(aggregate_payload)
    )
    assert _tree_bytes(checkpoint) == before
    with pytest.raises(FrozenInstanceError):
        identity.size = "4b"
    with pytest.raises(FrozenInstanceError):
        identity.files[0].sha256 = "0" * 64


def test_pretty_json_is_normalized_but_raw_bytes_remain_bound(checkpoint):
    pretty = inspect_vlm_checkpoint_identity(checkpoint)
    (checkpoint / "libreyolo_vlm.json").write_bytes(_canonical(_contract()))
    (checkpoint / "adapter_config.json").write_bytes(_canonical(_adapter_config()))
    canonical = inspect_vlm_checkpoint_identity(checkpoint)

    assert canonical.checkpoint_contract_sha256 == pretty.checkpoint_contract_sha256
    assert canonical.adapter_config_sha256 == pretty.adapter_config_sha256
    assert canonical.adapter_weights_sha256 == pretty.adapter_weights_sha256
    assert canonical.processor_sha256 == pretty.processor_sha256
    assert canonical.aggregate_sha256 != pretty.aggregate_sha256
    assert canonical.files != pretty.files


def test_semantic_validation_uses_the_same_isolated_bytes_as_raw_identity(
    checkpoint, monkeypatch
):
    transient = _contract()
    transient["metrics"]["epoch"] = 99
    transient_payload = json.dumps(transient, indent=2).encode()
    original_read = artifact_module._read_bounded

    def substitute_source_read(path, *, max_bytes, label):
        candidate = Path(path)
        if (
            candidate.name == "libreyolo_vlm.json"
            and candidate.parent.name == "primary"
        ):
            return transient_payload
        return original_read(path, max_bytes=max_bytes, label=label)

    monkeypatch.setattr(artifact_module, "_read_bounded", substitute_source_read)
    with pytest.raises(VLMArtifactError, match="semantic validation"):
        inspect_vlm_checkpoint_identity(checkpoint)


def test_source_readme_is_explicitly_ignored(checkpoint):
    first = inspect_vlm_checkpoint_identity(checkpoint)
    (checkpoint / "README.md").write_text("# Rewritten source card\n", encoding="utf-8")
    assert inspect_vlm_checkpoint_identity(checkpoint) == first


def test_rejects_full_model_checkpoint(checkpoint):
    (checkpoint / "adapter_config.json").unlink()
    (checkpoint / "adapter_model.safetensors").unlink()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"full model")
    with pytest.raises(VLMArtifactError, match="missing required files"):
        inspect_vlm_checkpoint_identity(checkpoint)


def test_rejects_legacy_unpinned_contract(checkpoint):
    contract = _contract()
    contract["base_revision"] = None
    (checkpoint / "libreyolo_vlm.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    with pytest.raises(VLMArtifactError, match="immutable base pin"):
        inspect_vlm_checkpoint_identity(checkpoint)


@pytest.mark.parametrize(
    ("filename", "change", "match"),
    [
        (
            "libreyolo_vlm.json",
            lambda value: value.__setitem__("unexpected", True),
            "invalid keys",
        ),
        (
            "adapter_config.json",
            lambda value: value.__setitem__("peft_type", "PREFIX_TUNING"),
            "peft_type='LORA'",
        ),
        (
            "adapter_config.json",
            lambda value: value.__setitem__("r", 8),
            "Qwen3-VL recipe",
        ),
    ],
)
def test_rejects_semantically_noncanonical_json(checkpoint, filename, change, match):
    value = _contract() if filename == "libreyolo_vlm.json" else _adapter_config()
    change(value)
    (checkpoint / filename).write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(VLMArtifactError, match=match):
        inspect_vlm_checkpoint_identity(checkpoint)


def test_rejects_extra_file(checkpoint):
    (checkpoint / "notes.txt").write_text(
        "not part of the checkpoint", encoding="utf-8"
    )
    with pytest.raises(VLMArtifactError, match="unsupported files"):
        inspect_vlm_checkpoint_identity(checkpoint)


def test_rejects_extra_processor_file(checkpoint):
    (checkpoint / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    with pytest.raises(VLMArtifactError, match="exact audited file set"):
        inspect_vlm_checkpoint_identity(checkpoint)


def test_rejects_processor_bytes_outside_the_audited_raw_identity(checkpoint):
    (checkpoint / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(VLMArtifactError, match="raw identity"):
        inspect_vlm_checkpoint_identity(checkpoint)


def test_rejects_safetensor_digest_not_bound_to_validated_bytes(
    checkpoint, monkeypatch
):
    original = artifact_module._validate_safetensors

    def mismatched_digest(path, size):
        original(path, size)
        return "0" * 64

    monkeypatch.setattr(artifact_module, "_validate_safetensors", mismatched_digest)
    with pytest.raises(VLMArtifactError, match="semantic validation"):
        inspect_vlm_checkpoint_identity(checkpoint)


def test_rejects_legacy_pickle_adapter(checkpoint):
    (checkpoint / "adapter_model.safetensors").unlink()
    (checkpoint / "adapter_model.bin").write_bytes(b"pickle")
    with pytest.raises(VLMArtifactError, match="missing required files"):
        inspect_vlm_checkpoint_identity(checkpoint)


def test_rejects_hardlinked_checkpoint_file(checkpoint, tmp_path):
    target = checkpoint / "tokenizer.json"
    original = tmp_path / "tokenizer-original.json"
    target.replace(original)
    os.link(original, target)
    with pytest.raises(VLMArtifactError, match="hard-linked"):
        inspect_vlm_checkpoint_identity(checkpoint)


def test_rejects_symlinked_checkpoint_root(checkpoint, tmp_path):
    linked = tmp_path / "linked-checkpoint"
    try:
        linked.symlink_to(checkpoint, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(VLMArtifactError, match="symlink or junction"):
        inspect_vlm_checkpoint_identity(linked)


def test_rejects_oversized_checkpoint_file(checkpoint, monkeypatch):
    limits = dict(artifact_module._ARTIFACT_FILE_LIMITS)
    limits["libreyolo_vlm.json"] = 8
    monkeypatch.setattr(artifact_module, "_ARTIFACT_FILE_LIMITS", limits)
    with pytest.raises(VLMArtifactError, match="8-byte safety limit"):
        inspect_vlm_checkpoint_identity(checkpoint)


def test_rejects_checkpoint_changed_during_identity_scan(checkpoint, monkeypatch):
    original = artifact_module._fingerprint_file
    semantic_files = len(_tree_bytes(checkpoint)) - 1
    calls = 0

    def mutate_between_scans(path, relative):
        nonlocal calls
        calls += 1
        if calls == semantic_files + 1:
            config = checkpoint / "adapter_config.json"
            config.write_bytes(config.read_bytes() + b" ")
        return original(path, relative)

    monkeypatch.setattr(artifact_module, "_fingerprint_file", mutate_between_scans)
    with pytest.raises(VLMArtifactError, match="changed while it was inspected"):
        inspect_vlm_checkpoint_identity(checkpoint)
