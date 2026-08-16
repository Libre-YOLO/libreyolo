"""Offline tests for the publishable LibreVLM artifact contract."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

import pytest

from libreyolo.models.vlm import artifact as artifact_module
from libreyolo.models.vlm.artifact import (
    PUBLICATION_EVIDENCE_SCHEMA,
    VLM_ARTIFACT_MANIFEST,
    VLM_ARTIFACT_SCHEMA,
    VLMArtifactError,
    build_vlm_artifact,
    create_vlm_publication_evidence_template,
    read_vlm_artifact_manifest,
    validate_vlm_artifact,
    validate_vlm_base_snapshot,
)

pytestmark = pytest.mark.unit

_BASES = {
    "2b": (
        "Qwen/Qwen3-VL-2B-Instruct",
        "89644892e4d85e24eaac8bacfd4f463576704203",
    ),
    "4b": (
        "Qwen/Qwen3-VL-4B-Instruct",
        "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    ),
}


@pytest.fixture(autouse=True)
def _compact_lora_layout(monkeypatch):
    """Keep offline artifact fixtures tiny; production dimensions are tested below."""

    toy = {
        "2b": {"layers": 1, "hidden": 1, "q": 1, "kv": 1, "intermediate": 1},
        "4b": {"layers": 1, "hidden": 1, "q": 1, "kv": 1, "intermediate": 1},
    }
    monkeypatch.setattr(artifact_module, "_QWEN_LORA_LAYOUT", toy)
    processor_records = tuple(
        (name, len(payload), _sha(payload))
        for name, payload in sorted(_processor_payloads().items())
    )
    monkeypatch.setattr(
        artifact_module,
        "_CANONICAL_PROCESSOR_FILES",
        {"2b": processor_records, "4b": processor_records},
    )


def _canonical(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safetensor(size: str = "2b") -> bytes:
    header = {}
    offset = 0
    layout = artifact_module._QWEN_LORA_LAYOUT[size]
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
                shape_a, shape_b = artifact_module._expected_lora_shapes(size, module)
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


def _mutate_safetensor(payload: bytes, mutate) -> bytes:
    header_size = struct.unpack("<Q", payload[:8])[0]
    header = json.loads(payload[8 : 8 + header_size])
    data = bytearray(payload[8 + header_size :])
    mutate(header, data)
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    return struct.pack("<Q", len(raw)) + raw + bytes(data)


def _prompt(names: list[str]) -> str:
    labels = ", ".join(names)
    return (
        f"Detect all instances of: {labels}. "
        "Output the result as a JSON array, one object per instance: "
        '[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
        "Only include objects that are actually visible; if there are none, "
        "respond with an empty array []."
    )


def _contract(size: str = "2b") -> dict:
    repo, revision = _BASES[size]
    names = ["ripe strawberry", "worker"]
    return {
        "schema": 1,
        "family": "qwen3vl",
        "size": size,
        "base_repo": repo,
        "base_revision": revision,
        "names": names,
        "bbox_key": "bbox_2d",
        "coord_divisor": 1000.0,
        "box_format": "xyxy",
        "prompt": _prompt(names),
        "task": "detect",
        "metrics": {"epoch": 0, "train/loss": 0.25, "val/loss": 0.5},
        "libreyolo_version": "1.6.0",
    }


def _adapter_config(size: str = "2b", *, real_peft_local_identity: bool = True) -> dict:
    repo, _ = _BASES[size]
    return {
        "alora_invocation_tokens": None,
        "alpha_pattern": {},
        "arrow_config": None,
        "auto_mapping": (
            {
                "base_model_class": "Qwen3VLForConditionalGeneration",
                "parent_library": "transformers.models.qwen3_vl.modeling_qwen3_vl",
            }
            if real_peft_local_identity
            else None
        ),
        "base_model_name_or_path": (
            f"weights\\LibreQwen3VL{size}" if real_peft_local_identity else repo
        ),
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


def _base_snapshot(size: str = "2b") -> dict:
    return artifact_module._canonical_base_snapshot(size)


def _processor_payloads() -> dict[str, bytes]:
    return {
        "chat_template.jinja": (
            "{% for message in messages %}{{ message.content }}{% endfor %}"
        ).encode(),
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
                "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}",
                "tokenizer_class": "Qwen2TokenizerFast",
            }
        ),
    }


def _processor_sha() -> str:
    entries = [
        {"path": name, "size": len(payload), "sha256": _sha(payload)}
        for name, payload in sorted(_processor_payloads().items())
    ]
    return _sha(_canonical(entries).rstrip(b"\n"))


def _evidence(size: str = "2b") -> dict:
    repo, revision = _BASES[size]
    snapshot = _base_snapshot(size)
    data_sha = "d" * 64
    report_sha = "e" * 64
    code_revision = "c" * 40
    adapter_sha = _sha(_safetensor(size))
    contract_sha = _sha(_canonical(_contract(size)))
    processor_sha = _processor_sha()
    return {
        "schema": PUBLICATION_EVIDENCE_SCHEMA,
        "artifact_license": {
            "spdx": "Apache-2.0",
            "redistribution_decision": "approved",
        },
        "base_model": {
            "repo": repo,
            "revision": revision,
            "license_spdx": "Apache-2.0",
            "license_evidence_url": (
                f"https://huggingface.co/{repo}/blob/{revision}/README.md"
            ),
            "weights_redistribution_decision": "reference-only",
            "processor_redistribution_decision": "approved",
            "snapshot": snapshot,
        },
        "training_data": {
            "source": "https://example.org/datasets/strawberries",
            "version": "2026.08",
            "split": "train-v1",
            "license_spdx": "CC-BY-4.0",
            "license_evidence_url": "https://example.org/datasets/strawberries/license",
            "manifest_sha256": data_sha,
            "redistribution_decision": "approved-for-derived-weights",
        },
        "evaluation": {
            "benchmark": "strawberries-holdout-v1",
            "report_sha256": report_sha,
            "checkpoint_sha256": adapter_sha,
            "metrics": {"metrics/mAP50": 0.8, "metrics/mAP50-95": 0.6},
            "passed": True,
        },
        "code": {
            "repository": "https://github.com/LibreYOLO/libreyolo",
            "revision": code_revision,
            "clean": True,
            "recipe": {
                "id": "qwen3vl-lora-v1",
                "sha256": artifact_module._recipe_sha256(),
            },
            "dependencies": {
                "libreyolo": "1.6.0",
                "peft": "0.19.1",
                "torch": "2.8.0+cu128",
                "transformers": "5.12.1",
            },
        },
        "review": {
            "approved": True,
            "reviewer": "Human Reviewer",
            "reviewed_at": "2026-08-16T12:34:56Z",
            "bindings": {
                "base_snapshot_sha256": snapshot["sha256"],
                "training_data_manifest_sha256": data_sha,
                "evaluation_report_sha256": report_sha,
                "code_revision": code_revision,
                "recipe_sha256": artifact_module._recipe_sha256(),
                "adapter_weights_sha256": adapter_sha,
                "checkpoint_contract_sha256": contract_sha,
                "processor_sha256": processor_sha,
            },
            "gates": {
                "artifact_license_approved": True,
                "base_model_verified": True,
                "training_data_approved": True,
                "privacy_approved": True,
                "evaluation_approved": True,
                "code_provenance_approved": True,
            },
        },
    }


def _template_context(size: str = "2b") -> tuple[dict, dict, dict]:
    evidence = _evidence(size)
    training_data = {
        key: value
        for key, value in evidence["training_data"].items()
        if key != "redistribution_decision"
    }
    evaluation = {
        key: value
        for key, value in evidence["evaluation"].items()
        if key not in {"checkpoint_sha256", "passed"}
    }
    code = {
        "revision": evidence["code"]["revision"],
        "clean": evidence["code"]["clean"],
        "dependencies": evidence["code"]["dependencies"],
    }
    return training_data, evaluation, code


def _write_checkpoint(root: Path, size: str = "2b") -> Path:
    root.mkdir()
    (root / "libreyolo_vlm.json").write_text(
        json.dumps(_contract(size), indent=2), encoding="utf-8"
    )
    (root / "adapter_config.json").write_text(
        json.dumps(_adapter_config(size), indent=2), encoding="utf-8"
    )
    (root / "adapter_model.safetensors").write_bytes(_safetensor(size))
    for name, payload in _processor_payloads().items():
        (root / name).write_bytes(payload)
    (root / "README.md").write_text(
        "# PEFT-generated source card (regenerated for publication)\n",
        encoding="utf-8",
    )
    return root


def _write_evidence(path: Path, size: str = "2b", *, value: dict | None = None) -> Path:
    path.write_text(json.dumps(value or _evidence(size), indent=2), encoding="utf-8")
    return path


def _artifact(tmp_path: Path, size: str = "2b"):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint", size)
    evidence = _write_evidence(tmp_path / "publication.json", size)
    output = tmp_path / "artifact"
    return build_vlm_artifact(checkpoint, output, publication_evidence=evidence)


def test_production_qwen_lora_layout_is_exact():
    assert artifact_module._PRODUCTION_QWEN_LORA_LAYOUT == {
        "2b": {
            "layers": 28,
            "hidden": 2048,
            "q": 2048,
            "kv": 1024,
            "intermediate": 6144,
        },
        "4b": {
            "layers": 36,
            "hidden": 2560,
            "q": 4096,
            "kv": 1024,
            "intermediate": 9728,
        },
    }


def test_official_base_snapshot_identities_are_exact():
    expected = {
        "2b": (
            12,
            "5bdeeb206b68d07937ee2d169adddad82206ace882f4c45728165040bd09a0a4",
            "f78a03f9fd52dbe77d3d71d7269ec3d5267f67da0334f224ef891e1031c14e05",
        ),
        "4b": (
            14,
            "a4f938b05ea1ccd0f0043ca9974d6a65b7bee1a4e19f28701d111e635fb8d468",
            "f341c77fe768f679d064c91adbd028dd8030d94ad24fdad677f10ca59dac9174",
        ),
    }
    for size, (count, aggregate, identity_sha) in expected.items():
        identity = artifact_module._canonical_base_snapshot(size)
        assert len(identity["files"]) == count
        assert identity["aggregate_sha256"] == aggregate
        assert identity["sha256"] == identity_sha


def test_trained_qwen_processor_serialization_identity_is_exact():
    assert artifact_module._TRAINED_PROCESSOR_FILES == (
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
    records = [
        {"path": path, "size": size, "sha256": digest}
        for path, size, digest in artifact_module._TRAINED_PROCESSOR_FILES
    ]
    assert artifact_module._aggregate_entries(records) == (
        "18f10e19ddad229e9c9f5cfc0c3cb437ce2cda43875671555415ff073724a2b3"
    )


@pytest.mark.parametrize("size", ["2b", "4b"])
def test_build_validate_round_trip_and_real_peft_identity(tmp_path, size):
    info = _artifact(tmp_path, size)

    assert info.root == tmp_path / "artifact"
    assert info.manifest["schema"] == VLM_ARTIFACT_SCHEMA
    assert info.manifest["identity"]["size"] == size
    assert info.base_snapshot["source"] == _BASES[size][0]
    assert info.files == tuple(sorted(info.files, key=str.casefold))
    assert validate_vlm_artifact(info.root).aggregate_sha256 == info.aggregate_sha256

    adapter = json.loads((info.root / "adapter_config.json").read_text())
    assert adapter["base_model_name_or_path"] == _BASES[size][0]
    assert adapter["revision"] == _BASES[size][1]
    assert adapter["auto_mapping"] is None
    assert (info.root / "LICENSE").read_text().lstrip().startswith("Apache License")
    assert len((info.root / "LICENSE").read_bytes()) == 11_357
    assert _sha((info.root / "LICENSE").read_bytes()) == (
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    )
    card = (info.root / "README.md").read_text()
    assert "license: apache-2.0" in card
    assert "pipeline_tag: image-text-to-text" in card
    assert f"base_model: {_BASES[size][0]}" in card
    assert "base_model_relation: adapter" in card
    assert "- object-detection\n- vlm\n- peft\n- lora\n- libreyolo" in card
    assert "processor, tokenizer, and chat-template assets" in card
    notice = (info.root / "NOTICE").read_text()
    assert "LoRA adapter trained with LibreYOLO" in notice
    assert "processor, tokenizer, and chat-template assets" in notice
    assert "Included processor assets redistribution: approved" in notice
    assert (info.root / ".gitattributes").read_bytes() == (
        b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
    )
    serialized_text = b"".join(
        path.read_bytes()
        for path in info.root.iterdir()
        if path.suffix in {".json", ".md", ""}
    )
    assert str(tmp_path).encode() not in serialized_text


def test_build_is_byte_deterministic_across_destinations(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence = _write_evidence(tmp_path / "publication.json")
    first = build_vlm_artifact(
        checkpoint, tmp_path / "first", publication_evidence=evidence
    )
    second = build_vlm_artifact(
        checkpoint, tmp_path / "second", publication_evidence=evidence
    )

    assert first.aggregate_sha256 == second.aggregate_sha256
    assert first.files == second.files
    for name in (*first.files, VLM_ARTIFACT_MANIFEST):
        assert (first.root / name).read_bytes() == (second.root / name).read_bytes()


def test_build_preserves_audited_raw_processor_serialization(tmp_path, monkeypatch):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    processor = json.loads((checkpoint / "processor_config.json").read_text())
    raw_processor = (
        json.dumps(processor, indent=4, ensure_ascii=False).replace("\n", "\r\n")
        + "\r\n"
    ).encode()
    (checkpoint / "processor_config.json").write_bytes(raw_processor)
    payloads = {
        name: (checkpoint / name).read_bytes() for name in _processor_payloads()
    }
    records = tuple(
        (name, len(payload), _sha(payload))
        for name, payload in sorted(payloads.items())
    )
    monkeypatch.setattr(
        artifact_module,
        "_CANONICAL_PROCESSOR_FILES",
        {"2b": records, "4b": records},
    )
    processor_identity = [
        {"path": name, "size": len(payload), "sha256": _sha(payload)}
        for name, payload in sorted(payloads.items())
    ]
    evidence_value = _evidence()
    evidence_value["review"]["bindings"]["processor_sha256"] = (
        artifact_module._aggregate_entries(processor_identity)
    )
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)

    info = build_vlm_artifact(
        checkpoint, tmp_path / "artifact", publication_evidence=evidence
    )
    for name, payload in payloads.items():
        assert (info.root / name).read_bytes() == payload


def test_manifest_only_reader_requires_no_payload(tmp_path):
    info = _artifact(tmp_path)
    manifest_only = tmp_path / "manifest-only"
    manifest_only.mkdir()
    shutil.copyfile(
        info.root / VLM_ARTIFACT_MANIFEST,
        manifest_only / VLM_ARTIFACT_MANIFEST,
    )

    parsed = read_vlm_artifact_manifest(manifest_only)
    assert parsed.files == info.files
    assert parsed.base_snapshot == info.base_snapshot
    with pytest.raises(VLMArtifactError, match="inventory mismatch"):
        read_vlm_artifact_manifest(manifest_only, require_payload=True)


def test_validated_info_is_recursively_immutable(tmp_path):
    info = _artifact(tmp_path)
    with pytest.raises(TypeError):
        info.manifest["identity"]["size"] = "4b"
    with pytest.raises(TypeError):
        info.base_snapshot["files"][0]["sha256"] = "0" * 64
    with pytest.raises(AttributeError):
        info.manifest["files"].append({})


def test_reader_does_not_depend_on_current_recipe_bytes(tmp_path, monkeypatch):
    info = _artifact(tmp_path)
    monkeypatch.setattr(artifact_module, "_recipe_sha256", lambda: "f" * 64)
    assert validate_vlm_artifact(info.root).aggregate_sha256 == info.aggregate_sha256


@pytest.mark.parametrize(
    "name",
    [
        "adapter_model.safetensors",
        "adapter_config.json",
        "libreyolo_vlm.json",
        "publication_evidence.json",
        "processor_config.json",
        "README.md",
        "LICENSE",
        "NOTICE",
    ],
)
def test_validation_rejects_every_tampered_file(tmp_path, name):
    info = _artifact(tmp_path)
    path = info.root / name
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(VLMArtifactError, match="does not match manifest"):
        validate_vlm_artifact(info.root)


def test_validation_rejects_extra_file(tmp_path):
    info = _artifact(tmp_path)
    (info.root / "model.py").write_text("raise SystemExit", encoding="utf-8")
    with pytest.raises(VLMArtifactError, match="unsupported|inventory mismatch"):
        validate_vlm_artifact(info.root)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda text: text.replace('"schema":', '"schema":"shadow", "schema":', 1),
            "duplicate",
        ),
        (
            lambda text: text.replace(
                '"representation":', '"unknown":1,"representation":', 1
            ),
            "invalid keys",
        ),
        (
            lambda text: text.replace('"files":', '"nonfinite":NaN,"files":', 1),
            "constant",
        ),
    ],
)
def test_manifest_rejects_duplicate_unknown_and_nonfinite_json(tmp_path, mutate, match):
    info = _artifact(tmp_path)
    manifest = info.root / VLM_ARTIFACT_MANIFEST
    manifest.write_text(mutate(manifest.read_text()), encoding="utf-8")
    with pytest.raises(VLMArtifactError, match=match):
        read_vlm_artifact_manifest(manifest)


def test_manifest_rejects_deep_and_oversized_json(tmp_path):
    deep = tmp_path / "deep"
    deep.mkdir()
    nested = "[" * 70 + "0" + "]" * 70
    (deep / VLM_ARTIFACT_MANIFEST).write_text(nested, encoding="utf-8")
    with pytest.raises(VLMArtifactError, match="nesting|object"):
        read_vlm_artifact_manifest(deep)

    huge = tmp_path / "huge"
    huge.mkdir()
    with (huge / VLM_ARTIFACT_MANIFEST).open("wb") as stream:
        stream.truncate(4 * 1024 * 1024 + 1)
    with pytest.raises(VLMArtifactError, match="safety limit"):
        read_vlm_artifact_manifest(huge)


def test_manifest_rejects_declared_file_over_role_limit_before_payload(tmp_path):
    info = _artifact(tmp_path)
    manifest = json.loads((info.root / VLM_ARTIFACT_MANIFEST).read_text())
    template = next(
        entry for entry in manifest["files"] if entry["path"] == "chat_template.jinja"
    )
    template["size"] = 256 * 1024 + 1
    (info.root / VLM_ARTIFACT_MANIFEST).write_bytes(_canonical(manifest))
    with pytest.raises(VLMArtifactError, match="safety limit"):
        read_vlm_artifact_manifest(info.root)


def test_build_rejects_oversized_non_json_processor_before_copy(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    with (checkpoint / "chat_template.jinja").open("wb") as stream:
        stream.truncate(256 * 1024 + 1)
    evidence = _write_evidence(tmp_path / "publication.json")
    with pytest.raises(VLMArtifactError, match="safety limit"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["review"].__setitem__("approved", False), "must be true"),
        (
            lambda value: value["evaluation"].__setitem__("passed", False),
            "must be true",
        ),
        (lambda value: value["code"].__setitem__("clean", False), "must be true"),
        (
            lambda value: value["artifact_license"].__setitem__(
                "redistribution_decision", "unknown"
            ),
            "redistribution_decision",
        ),
        (
            lambda value: value["review"]["gates"].__setitem__(
                "privacy_approved", False
            ),
            "must be true",
        ),
        (
            lambda value: value["review"]["bindings"].__setitem__(
                "evaluation_report_sha256", "a" * 64
            ),
            "bindings",
        ),
        (
            lambda value: value["code"]["recipe"].__setitem__("sha256", "a" * 64),
            "recipe.sha256",
        ),
    ],
)
def test_build_rejects_missing_publication_gate_without_partial_output(
    tmp_path, mutation, match
):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence_value = _evidence()
    mutation(evidence_value)
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)
    output = tmp_path / "artifact"

    with pytest.raises(VLMArtifactError, match=match):
        build_vlm_artifact(checkpoint, output, publication_evidence=evidence)
    assert not output.exists()
    assert not list(tmp_path.glob(".artifact.staging-*"))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda c: c.__setitem__("base_revision", "a" * 40), "immutable base pin"),
        (lambda c: c.__setitem__("task", "classify"), "task"),
        (lambda c: c.__setitem__("prompt", "custom"), "prompt"),
        (lambda c: c.__setitem__("family", "lfm2vl"), "Qwen3-VL"),
    ],
)
def test_build_rejects_invalid_checkpoint_contract(tmp_path, change, match):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    contract = _contract()
    change(contract)
    (checkpoint / "libreyolo_vlm.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    evidence = _write_evidence(tmp_path / "publication.json")
    with pytest.raises(VLMArtifactError, match=match):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


@pytest.mark.parametrize(
    "payload", [b"", b"adapter", b"\x08\x00\x00\x00\x00\x00\x00\x00{}      "]
)
def test_build_rejects_malformed_safetensors(tmp_path, payload):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    (checkpoint / "adapter_model.safetensors").write_bytes(payload)
    evidence = _write_evidence(tmp_path / "publication.json")
    with pytest.raises(VLMArtifactError, match="safetensors"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


@pytest.mark.parametrize("mutation", ["base-tensor", "integer", "shape", "layer-00"])
def test_build_rejects_non_lora_or_noncanonical_tensor_inventory(tmp_path, mutation):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")

    def alter(header, data):
        first = next(iter(header))
        if mutation == "base-tensor":
            start = len(data)
            header["base_model.model.model.embed_tokens.weight"] = {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [start, start + 2],
            }
            data.extend(b"\0\0")
        elif mutation == "integer":
            header[first]["dtype"] = "U16"
        elif mutation == "shape":
            header[first]["shape"] = [8, 2]
        else:
            additions = {}
            for name, tensor in header.items():
                if ".layers.0." not in name:
                    continue
                start = len(data)
                size_bytes = tensor["data_offsets"][1] - tensor["data_offsets"][0]
                copied = dict(tensor)
                copied["data_offsets"] = [start, start + size_bytes]
                additions[name.replace(".layers.0.", ".layers.00.")] = copied
                data.extend(b"\0" * size_bytes)
            header.update(additions)

    weight = checkpoint / "adapter_model.safetensors"
    weight.write_bytes(_mutate_safetensor(weight.read_bytes(), alter))
    evidence = _write_evidence(tmp_path / "publication.json")
    with pytest.raises(VLMArtifactError, match="safetensors|LoRA|dtype|canonical"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


def test_build_rejects_pickle_and_unknown_files(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    (checkpoint / "adapter_model.bin").write_bytes(b"pickle")
    evidence = _write_evidence(tmp_path / "publication.json")
    with pytest.raises(VLMArtifactError, match="unsupported files|safetensors"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_rslora", True),
        ("modules_to_save", ["lm_head"]),
        ("alpha_pattern", {"model.layers.0": 64}),
        ("rank_pattern", {"model.layers.0": 8}),
    ],
)
def test_build_rejects_behavior_changing_peft_options(tmp_path, field, value):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    config_path = checkpoint / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config[field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")
    evidence = _write_evidence(tmp_path / "publication.json")
    with pytest.raises(VLMArtifactError, match="fixed recipe"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


@pytest.mark.parametrize(
    ("dependency", "version"),
    [("peft", "0.18.0"), ("transformers", "5.11.0")],
)
def test_build_pins_publication_writer_dependencies(tmp_path, dependency, version):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    value = _evidence()
    value["code"]["dependencies"][dependency] = version
    evidence = _write_evidence(tmp_path / "publication.json", value=value)
    with pytest.raises(VLMArtifactError, match=f"dependencies.{dependency}"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


@pytest.mark.parametrize("swap", ["weights", "contract", "processor"])
def test_review_approval_is_not_transferable_to_different_checkpoint_bytes(
    tmp_path, swap
):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    if swap == "weights":
        weight = checkpoint / "adapter_model.safetensors"
        payload = bytearray(weight.read_bytes())
        payload[-1] ^= 1
        weight.write_bytes(payload)
    elif swap == "contract":
        contract = _contract()
        contract["names"] = ["different object"]
        contract["prompt"] = _prompt(contract["names"])
        (checkpoint / "libreyolo_vlm.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
    else:
        processor = json.loads((checkpoint / "processor_config.json").read_text())
        processor["image_processor"]["do_resize"] = False
        (checkpoint / "processor_config.json").write_text(
            json.dumps(processor), encoding="utf-8"
        )
    evidence = _write_evidence(tmp_path / "publication.json")
    expected = (
        "audited upstream assets" if swap == "processor" else "review does not bind"
    )
    with pytest.raises(VLMArtifactError, match=expected):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


def test_build_is_create_only_and_preserves_existing_destination(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence = _write_evidence(tmp_path / "publication.json")
    output = tmp_path / "artifact"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("mine", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_vlm_artifact(checkpoint, output, publication_evidence=evidence)
    assert sentinel.read_text() == "mine"


def test_atomic_publication_preserves_racing_empty_destination(tmp_path, monkeypatch):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence = _write_evidence(tmp_path / "publication.json")
    output = tmp_path / "artifact"
    original = artifact_module._rename_create_only

    def race(source, destination, seal):
        destination.mkdir()
        original(source, destination, seal)

    monkeypatch.setattr(artifact_module, "_rename_create_only", race)
    with pytest.raises(FileExistsError):
        build_vlm_artifact(checkpoint, output, publication_evidence=evidence)
    assert output.is_dir()
    assert not list(output.iterdir())
    assert not list(tmp_path.glob(".artifact.staging-*"))


def test_publication_rejects_staging_replacement_after_validation(
    tmp_path, monkeypatch
):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence = _write_evidence(tmp_path / "publication.json")
    output = tmp_path / "artifact"
    original = artifact_module._rename_create_only
    replacement: dict[str, Path] = {}

    def replace(source, destination, seal):
        moved = source.with_name(source.name + ".validated")
        source.rename(moved)
        source.mkdir()
        (source / "unvalidated.txt").write_text("preserve", encoding="utf-8")
        replacement["path"] = source
        original(source, destination, seal)

    monkeypatch.setattr(artifact_module, "_rename_create_only", replace)
    with pytest.raises(VLMArtifactError, match="staging directory changed"):
        build_vlm_artifact(checkpoint, output, publication_evidence=evidence)
    assert not output.exists()
    assert (replacement["path"] / "unvalidated.txt").read_text() == "preserve"


def test_publication_rejects_valid_new_inode_swapped_at_rename(tmp_path, monkeypatch):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence = _write_evidence(tmp_path / "publication.json")
    output = tmp_path / "artifact"
    displaced: dict[str, Path] = {}

    def replace_with_valid_copy(source, destination, _seal):
        original = source.with_name(source.name + ".validated-original")
        source.rename(original)
        shutil.copytree(original, source)
        displaced["path"] = original
        os.rename(source, destination)

    monkeypatch.setattr(artifact_module, "_rename_create_only", replace_with_valid_copy)
    with pytest.raises(VLMArtifactError, match="changed after publication"):
        build_vlm_artifact(checkpoint, output, publication_evidence=evidence)
    assert output.is_dir()
    assert validate_vlm_artifact(output).aggregate_sha256
    assert displaced["path"].is_dir()


def test_losing_builder_never_removes_another_publishers_lock(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence = _write_evidence(tmp_path / "publication.json")
    lock = tmp_path / ".artifact.create.lock"
    lock.write_text("winner", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already in progress"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )
    assert lock.read_text() == "winner"


def test_builder_never_unlinks_a_replaced_lock_inode(tmp_path, monkeypatch):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence = _write_evidence(tmp_path / "publication.json")
    lock = tmp_path / ".artifact.create.lock"

    def replace_lock(_source, _destination, _seal):
        try:
            lock.unlink()
        except PermissionError:
            pytest.skip("platform prevents replacing an open lock inode")
        lock.write_text("replacement", encoding="utf-8")
        raise VLMArtifactError("injected lock replacement")

    monkeypatch.setattr(artifact_module, "_rename_create_only", replace_lock)
    with pytest.raises(VLMArtifactError, match="lock replacement"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )
    assert lock.read_text() == "replacement"


def test_cleanup_preserves_replaced_staging_directory(tmp_path, monkeypatch):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence = _write_evidence(tmp_path / "publication.json")
    replacement: dict[str, Path] = {}

    def replace_staging(_source, destination):
        staging = destination.parent
        moved = staging.with_name(staging.name + ".original")
        staging.rename(moved)
        staging.mkdir()
        (staging / "attacker.txt").write_text("preserve", encoding="utf-8")
        replacement["path"] = staging
        raise VLMArtifactError("injected staging replacement")

    monkeypatch.setattr(artifact_module, "_copy_file_stable", replace_staging)
    with pytest.raises(VLMArtifactError, match="replacement"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )
    assert (replacement["path"] / "attacker.txt").read_text() == "preserve"


def test_publication_evidence_must_be_external(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence = _write_evidence(checkpoint / "publication.json")
    with pytest.raises(VLMArtifactError, match="external"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


def test_copy_failure_leaves_no_partial_destination(tmp_path, monkeypatch):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence = _write_evidence(tmp_path / "publication.json")

    def fail_copy(*_args, **_kwargs):
        raise VLMArtifactError("injected copy failure")

    monkeypatch.setattr(artifact_module, "_copy_file_stable", fail_copy)
    with pytest.raises(VLMArtifactError, match="injected"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )
    assert not (tmp_path / "artifact").exists()
    assert not list(tmp_path.glob(".artifact.staging-*"))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_build_rejects_internal_symlink(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    target = checkpoint / "tokenizer.json"
    target.unlink()
    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    try:
        target.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    evidence = _write_evidence(tmp_path / "publication.json")
    with pytest.raises(VLMArtifactError, match="link"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


def test_build_rejects_hardlinked_checkpoint_file(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    original = checkpoint / "tokenizer.json"
    outside = tmp_path / "outside.json"
    try:
        os.link(original, outside)
    except OSError as exc:
        pytest.skip(f"hardlink creation unavailable: {exc}")
    evidence = _write_evidence(tmp_path / "publication.json")
    with pytest.raises(VLMArtifactError, match="hard-linked"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


def test_build_and_validate_import_no_model_or_hub_packages(tmp_path, monkeypatch):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence = _write_evidence(tmp_path / "publication.json")
    forbidden = {"huggingface_hub", "peft", "transformers"}
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError(f"forbidden import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    info = build_vlm_artifact(
        checkpoint, tmp_path / "artifact", publication_evidence=evidence
    )
    validate_vlm_artifact(info.root)


def _materialize_base_snapshot(root: Path, identity: dict) -> None:
    root.mkdir()
    payloads = {
        "config.json": b'{"model_type":"qwen3_vl"}\n',
        "model.safetensors": b"immutable-base-weights",
        "preprocessor_config.json": b'{"processor_class":"Qwen3VLProcessor"}\n',
        "tokenizer.json": b'{"version":"1.0"}\n',
        "tokenizer_config.json": b'{"tokenizer_class":"Qwen2TokenizerFast"}\n',
    }
    assert set(payloads) == {item["path"] for item in identity["files"]}
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)


def _tiny_base_identity(monkeypatch, size: str = "2b") -> dict:
    payloads = {
        "config.json": b'{"model_type":"qwen3_vl"}\n',
        "model.safetensors": b"immutable-base-weights",
        "preprocessor_config.json": b'{"processor_class":"Qwen3VLProcessor"}\n',
        "tokenizer.json": b'{"version":"1.0"}\n',
        "tokenizer_config.json": b'{"tokenizer_class":"Qwen2TokenizerFast"}\n',
    }
    records = tuple(
        (name, len(payload), _sha(payload))
        for name, payload in sorted(payloads.items())
    )
    patched = dict(artifact_module._CANONICAL_BASE_FILES)
    patched[size] = records
    monkeypatch.setattr(artifact_module, "_CANONICAL_BASE_FILES", patched)
    return artifact_module._canonical_base_snapshot(size)


def test_validate_exact_base_snapshot_and_transport_metadata(tmp_path, monkeypatch):
    identity = _tiny_base_identity(monkeypatch)
    root = tmp_path / "base"
    _materialize_base_snapshot(root, identity)
    cache = root / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "download.lock").write_text("transport", encoding="utf-8")
    (root / ".libreyolo_snapshot_complete").write_text(
        json.dumps({"repo": identity["source"], "revision": identity["revision"]}),
        encoding="utf-8",
    )

    observed = validate_vlm_base_snapshot(root, identity)
    assert observed["sha256"] == identity["sha256"]


def test_artifact_info_base_identity_feeds_snapshot_validator(tmp_path, monkeypatch):
    tiny_identity = _tiny_base_identity(monkeypatch)
    info = _artifact(tmp_path)
    root = tmp_path / "base"
    _materialize_base_snapshot(root, dict(info.base_snapshot))
    observed = validate_vlm_base_snapshot(root, info.base_snapshot)
    assert observed["sha256"] == tiny_identity["sha256"]


@pytest.mark.parametrize("mutation", ["tamper", "extra", "missing", "marker"])
def test_validate_base_snapshot_rejects_tamper_and_inventory_drift(
    tmp_path, monkeypatch, mutation
):
    identity = _tiny_base_identity(monkeypatch)
    root = tmp_path / "base"
    _materialize_base_snapshot(root, identity)
    if mutation == "tamper":
        (root / "model.safetensors").write_bytes(b"changed")
    elif mutation == "extra":
        (root / "generation_config.json").write_text("{}", encoding="utf-8")
    elif mutation == "missing":
        (root / "tokenizer.json").unlink()
    else:
        (root / ".libreyolo_snapshot_complete").write_text(
            json.dumps({"repo": "wrong/repo", "revision": identity["revision"]}),
            encoding="utf-8",
        )

    with pytest.raises(VLMArtifactError, match="match|mismatch|marker"):
        validate_vlm_base_snapshot(root, identity)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_validate_base_snapshot_rejects_link(tmp_path, monkeypatch):
    identity = _tiny_base_identity(monkeypatch)
    root = tmp_path / "base"
    _materialize_base_snapshot(root, identity)
    weight = root / "model.safetensors"
    external = tmp_path / "external.safetensors"
    external.write_bytes(weight.read_bytes())
    weight.unlink()
    try:
        weight.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(VLMArtifactError, match="symlink|junction"):
        validate_vlm_base_snapshot(root, identity)


def _create_template_fixture(tmp_path: Path, monkeypatch):
    identity = _tiny_base_identity(monkeypatch)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    base = tmp_path / "base"
    _materialize_base_snapshot(base, identity)
    training_data, evaluation, code = _template_context()
    output = tmp_path / "publication-template.json"
    result = create_vlm_publication_evidence_template(
        checkpoint,
        base,
        output,
        training_data=training_data,
        evaluation=evaluation,
        code=code,
    )
    return checkpoint, base, result


def test_publication_template_derives_bindings_and_remains_unapproved(
    tmp_path, monkeypatch
):
    checkpoint, _base, output = _create_template_fixture(tmp_path, monkeypatch)
    payload = output.read_bytes()
    template = json.loads(payload)
    assert payload == _canonical(template)
    assert template["artifact_license"]["redistribution_decision"] == "unreviewed"
    assert template["base_model"]["processor_redistribution_decision"] == "unreviewed"
    assert template["training_data"]["redistribution_decision"] == "unreviewed"
    assert template["evaluation"]["passed"] is False
    assert template["review"]["approved"] is False
    assert template["review"]["reviewer"] == ""
    assert template["review"]["reviewed_at"] == ""
    assert set(template["review"]["gates"].values()) == {False}
    assert template["review"]["bindings"] == {
        "adapter_weights_sha256": _sha(_safetensor()),
        "base_snapshot_sha256": template["base_model"]["snapshot"]["sha256"],
        "checkpoint_contract_sha256": _sha(_canonical(_contract())),
        "code_revision": "c" * 40,
        "evaluation_report_sha256": "e" * 64,
        "processor_sha256": _processor_sha(),
        "recipe_sha256": artifact_module._recipe_sha256(),
        "training_data_manifest_sha256": "d" * 64,
    }

    with pytest.raises(VLMArtifactError, match="redistribution_decision"):
        build_vlm_artifact(
            checkpoint,
            tmp_path / "unapproved-artifact",
            publication_evidence=output,
        )


def test_human_approved_template_builds_without_rederiving_bindings(
    tmp_path, monkeypatch
):
    checkpoint, _base, output = _create_template_fixture(tmp_path, monkeypatch)
    evidence = json.loads(output.read_text(encoding="utf-8"))
    evidence["artifact_license"]["redistribution_decision"] = "approved"
    evidence["base_model"]["processor_redistribution_decision"] = "approved"
    evidence["training_data"]["redistribution_decision"] = (
        "approved-for-derived-weights"
    )
    evidence["evaluation"]["passed"] = True
    evidence["review"]["approved"] = True
    evidence["review"]["reviewer"] = "Human Reviewer"
    evidence["review"]["reviewed_at"] = "2026-08-16T12:34:56Z"
    evidence["review"]["gates"] = {key: True for key in evidence["review"]["gates"]}
    output.write_bytes(_canonical(evidence))

    info = build_vlm_artifact(
        checkpoint,
        tmp_path / "artifact",
        publication_evidence=output,
    )
    assert validate_vlm_artifact(info.root).aggregate_sha256 == info.aggregate_sha256


def test_publication_template_is_create_only_and_rejects_internal_output(
    tmp_path, monkeypatch
):
    identity = _tiny_base_identity(monkeypatch)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    base = tmp_path / "base"
    _materialize_base_snapshot(base, identity)
    training_data, evaluation, code = _template_context()
    existing = tmp_path / "publication-template.json"
    existing.write_text("racer", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            existing,
            training_data=training_data,
            evaluation=evaluation,
            code=code,
        )
    assert existing.read_text(encoding="utf-8") == "racer"

    with pytest.raises(VLMArtifactError, match="outside the checkpoint"):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            checkpoint / "publication-template.json",
            training_data=training_data,
            evaluation=evaluation,
            code=code,
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_publication_template_rejects_symlinked_output_parent(tmp_path, monkeypatch):
    identity = _tiny_base_identity(monkeypatch)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    base = tmp_path / "base"
    _materialize_base_snapshot(base, identity)
    training_data, evaluation, code = _template_context()
    real_parent = tmp_path / "real-output"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")

    with pytest.raises(VLMArtifactError, match="symlink|junction"):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            linked_parent / "publication-template.json",
            training_data=training_data,
            evaluation=evaluation,
            code=code,
        )


def test_publication_template_preserves_racing_destination(tmp_path, monkeypatch):
    identity = _tiny_base_identity(monkeypatch)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    base = tmp_path / "base"
    _materialize_base_snapshot(base, identity)
    training_data, evaluation, code = _template_context()
    output = tmp_path / "publication-template.json"
    original = artifact_module._link_create_only

    def race(source, destination):
        destination.write_bytes(b"racer")
        original(source, destination)

    monkeypatch.setattr(artifact_module, "_link_create_only", race)
    with pytest.raises(FileExistsError):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            output,
            training_data=training_data,
            evaluation=evaluation,
            code=code,
        )
    assert output.read_bytes() == b"racer"
    assert not list(tmp_path.glob(".publication-template.json.staging-*.tmp"))


def test_publication_template_rejects_non_plain_or_unknown_context(
    tmp_path, monkeypatch
):
    identity = _tiny_base_identity(monkeypatch)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    base = tmp_path / "base"
    _materialize_base_snapshot(base, identity)
    training_data, evaluation, code = _template_context()
    training_data["unknown"] = True
    with pytest.raises(VLMArtifactError, match="invalid keys"):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            tmp_path / "publication-template.json",
            training_data=training_data,
            evaluation=evaluation,
            code=code,
        )


@pytest.mark.parametrize(
    "expression",
    [
        "CC-BY-2.0 AND CC-BY-4.0",
        "(MIT OR Apache-2.0) AND LicenseRef-Dataset-Terms",
        "GPL-2.0-only WITH Classpath-exception-2.0",
    ],
)
def test_publication_accepts_syntactically_valid_spdx_expressions(tmp_path, expression):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence_value = _evidence()
    evidence_value["training_data"]["license_spdx"] = expression
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)
    info = build_vlm_artifact(
        checkpoint, tmp_path / "artifact", publication_evidence=evidence
    )
    assert expression in (info.root / "NOTICE").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "expression", ["MIT AND", "(Apache-2.0", "NONE", "MIT / Apache-2.0"]
)
def test_publication_rejects_invalid_spdx_expressions(tmp_path, expression):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence_value = _evidence()
    evidence_value["training_data"]["license_spdx"] = expression
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)
    with pytest.raises(VLMArtifactError, match="SPDX"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/data](https://evil.example)",
        "https://example.org/data set",
        "https://example.org/data%2fescape",
        "https://example.org/data\u202eevil",
        "https://example.org/data%E2%80%AEevil",
    ],
)
def test_publication_rejects_markdown_or_unicode_unsafe_urls(tmp_path, url):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence_value = _evidence()
    evidence_value["training_data"]["source"] = url
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)
    with pytest.raises(VLMArtifactError, match="safe|canonical|normalized"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


def test_publication_accepts_canonical_percent_encoded_url(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence_value = _evidence()
    evidence_value["training_data"]["source"] = "https://example.org/data%20set"
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)
    info = build_vlm_artifact(
        checkpoint, tmp_path / "artifact", publication_evidence=evidence
    )
    assert "https://example.org/data%20set" in (info.root / "README.md").read_text(
        encoding="utf-8"
    )


def test_model_card_uses_safe_code_spans_for_dynamic_labels_and_metrics(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    malicious = "worker`](https://evil.example)"
    contract = _contract()
    contract["names"] = [malicious]
    contract["prompt"] = _prompt(contract["names"])
    (checkpoint / "libreyolo_vlm.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    evidence_value = _evidence()
    evidence_value["evaluation"]["metrics"] = {malicious: 0.8}
    evidence_value["review"]["bindings"]["checkpoint_contract_sha256"] = _sha(
        _canonical(contract)
    )
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)

    info = build_vlm_artifact(
        checkpoint, tmp_path / "artifact", publication_evidence=evidence
    )
    card = (info.root / "README.md").read_text(encoding="utf-8")
    assert card.count("`` worker`](https://evil.example) ``") == 2


@pytest.mark.parametrize("surface", ["label", "metric"])
def test_publication_rejects_unicode_bidi_controls(tmp_path, surface):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence_value = _evidence()
    unsafe = "worker\u202eevil"
    if surface == "label":
        contract = _contract()
        contract["names"] = [unsafe]
        contract["prompt"] = _prompt(contract["names"])
        (checkpoint / "libreyolo_vlm.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
    else:
        evidence_value["evaluation"]["metrics"] = {unsafe: 0.8}
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)
    with pytest.raises(VLMArtifactError, match="safe|normalized"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )
