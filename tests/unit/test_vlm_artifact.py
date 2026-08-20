"""Offline tests for the publishable LibreVLM artifact contract."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
from types import SimpleNamespace

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
from libreyolo.models.vlm.training.checkpoint import inspect_vlm_checkpoint_identity
from libreyolo.validation import vlm_confidence_benchmark as confidence_benchmark
from libreyolo.validation.vlm_confidence import VLMDetection, build_confidence_run
from libreyolo.validation import vlm_confidence_report as confidence_report_module

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


def _publication_metrics() -> dict[str, float]:
    return {
        "metrics/vlm_confidence/auroc": 0.75,
        "metrics/vlm_confidence/candidate_mAP50": 0.75,
        "metrics/vlm_confidence/candidate_mAP50-95": 0.625,
        "metrics/vlm_confidence/constant_mAP50": 0.5,
        "metrics/vlm_confidence/constant_mAP50-95": 0.5,
        "metrics/vlm_confidence/default_conf_fp_retention": 0.5,
        "metrics/vlm_confidence/default_conf_prediction_retention": 0.625,
        "metrics/vlm_confidence/default_conf_tp_retention": 0.75,
        "metrics/vlm_confidence/delta_mAP50": 0.25,
        "metrics/vlm_confidence/delta_mAP50-95": 0.125,
        "metrics/vlm_confidence/detection_score_coverage": 0.75,
        "metrics/vlm_confidence/prediction_score_coverage": 0.75,
        "metrics/vlm_confidence/ranking_ap": 0.75,
        "metrics/vlm_confidence/response_score_coverage": 0.875,
        "metrics/vlm_confidence/scored_prediction_brier": 0.125,
        "metrics/vlm_confidence/scored_prediction_ece": 0.125,
        "metrics/vlm_confidence/scored_prediction_mce": 0.25,
    }


def _repeatability_claim(report_sha: str, envelope_sha: str) -> dict:
    return {
        "schema": artifact_module._REPEATABILITY_CLAIM_SCHEMA,
        "receipt_sha256": "7" * 64,
        "comparison_sha256": "8" * 64,
        "runs": [
            {
                "run_id": "1" * 32,
                "process_id": "2" * 32,
                "report_sha256": report_sha,
                "envelope_sha256": envelope_sha,
            },
            {
                "run_id": "3" * 32,
                "process_id": "4" * 32,
                "report_sha256": "9" * 64,
                "envelope_sha256": "a" * 64,
            },
        ],
        "tolerances": {"score_atol": 0.0, "metric_atol": 0.0, "map_atol": 0.0},
        "reproducible": True,
    }


def _evidence(size: str = "2b") -> dict:
    repo, revision = _BASES[size]
    snapshot = _base_snapshot(size)
    data_sha = "d" * 64
    report_sha = "e" * 64
    envelope_sha = "f" * 64
    code_revision = "c" * 40
    adapter_sha = _sha(_safetensor(size))
    canonical_adapter = artifact_module._canonical_adapter_config(
        _adapter_config(size), _contract(size)
    )
    adapter_config_sha = _sha(_canonical(canonical_adapter))
    contract_sha = _sha(_canonical(_contract(size)))
    processor_sha = _processor_sha()
    evaluation = {
        "benchmark": artifact_module._CONFIDENCE_BENCHMARK_ID,
        "report_sha256": report_sha,
        "envelope_sha256": envelope_sha,
        "checkpoint_sha256": adapter_sha,
        "metrics": _publication_metrics(),
        "repeatability": _repeatability_claim(report_sha, envelope_sha),
        "passed": True,
    }
    evidence = {
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
        "evaluation": evaluation,
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
                "evaluation_envelope_sha256": envelope_sha,
                "evaluation_repeatability_receipt_sha256": "7" * 64,
                "evaluation_repeatability_comparison_sha256": "8" * 64,
                "evaluation_claim_sha256": artifact_module._evaluation_claim_sha256(
                    evaluation
                ),
                "code_revision": code_revision,
                "recipe_sha256": artifact_module._recipe_sha256(),
                "adapter_weights_sha256": adapter_sha,
                "adapter_config_sha256": adapter_config_sha,
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
    return evidence


def _template_context(size: str = "2b") -> tuple[dict, dict]:
    evidence = _evidence(size)
    training_data = {
        key: value
        for key, value in evidence["training_data"].items()
        if key != "redistribution_decision"
    }
    code = {
        "revision": evidence["code"]["revision"],
        "clean": evidence["code"]["clean"],
        "dependencies": evidence["code"]["dependencies"],
    }
    return training_data, code


def _confidence_metrics() -> dict[str, float | None]:
    metrics: dict[str, float | None] = _publication_metrics()
    metrics.update(
        {
            "metrics/vlm_confidence/responses": 100.0,
            "speed/images_seen": 100.0,
            "speed/total_ms": 12.5,
            "test/optional_null": None,
        }
    )
    return metrics


def _bind_confidence_report(
    monkeypatch,
    checkpoint: Path,
    *,
    metrics: dict[str, float | None] | None = None,
    mutate_context=None,
    mutate_benchmark=None,
    report_sha: str = "e" * 64,
    envelope_sha: str = "f" * 64,
) -> Path:
    identity = inspect_vlm_checkpoint_identity(checkpoint)
    checkpoint_context = {
        "schema": "libreyolo.vlm-confidence-checkpoint-identity.v1",
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
        "files": [
            {
                "path": record.path,
                "role": record.role,
                "size": record.size,
                "sha256": record.sha256,
            }
            for record in identity.files
        ],
    }
    context = {
        "schema": "libreyolo.vlm-confidence-benchmark-context.v3",
        "git": {},
        "runtime": {},
        "determinism": {},
        "dataset": {
            "schema": "libreyolo.vlm-confidence-benchmark-dataset.v1",
            "manifest": {},
            "source": {},
            "partition": {
                "name": "holdout100",
                "role": "fine_tune_validation",
                "start": 0,
                "stop": 100,
                "image_count": 100,
                "annotation_artifact": (
                    "annotations/instances_val2017_holdout100.json"
                ),
                "annotation_size_bytes": 1,
                "annotation_sha256": "a" * 64,
            },
            "classes": {},
            "review": {},
        },
        "checkpoint": checkpoint_context,
    }
    if mutate_context is not None:
        mutate_context(context)
    benchmark = {
        "family": identity.family,
        "size": identity.size,
        "base_repo": identity.base_repo,
        "base_revision": identity.base_revision,
        "benchmark_run": context,
    }
    if mutate_benchmark is not None:
        mutate_benchmark(benchmark)
    report = checkpoint.parent / "vlm_confidence_report.json"
    report.write_bytes(b"strict report bytes")
    identity_metrics = _confidence_metrics() if metrics is None else metrics
    monkeypatch.setattr(
        artifact_module,
        "_read_confidence_benchmark_identity",
        lambda path: SimpleNamespace(
            run_id="1" * 32,
            process_id="2" * 32,
            report_sha256=report_sha,
            envelope_sha256=envelope_sha,
            execution_context=context,
            benchmark_config=benchmark,
            metrics=identity_metrics,
            nonfinite_metrics=tuple(
                sorted(key for key, value in identity_metrics.items() if value is None)
            ),
        ),
    )
    receipt = checkpoint.parent / "vlm_confidence_repeatability.json"
    receipt.write_text("mock receipt\n", encoding="utf-8")
    monkeypatch.setattr(
        artifact_module,
        "_read_confidence_repeatability_identity",
        lambda path: SimpleNamespace(
            receipt_sha256="7" * 64,
            comparison_sha256="8" * 64,
            tolerances={"score_atol": 0.0, "metric_atol": 0.0, "map_atol": 0.0},
            comparison=SimpleNamespace(reproducible=True),
            runs=(
                SimpleNamespace(
                    run_id="1" * 32,
                    process_id="2" * 32,
                    report_sha256=report_sha,
                    envelope_sha256=envelope_sha,
                ),
                SimpleNamespace(
                    run_id="3" * 32,
                    process_id="4" * 32,
                    report_sha256="9" * 64,
                    envelope_sha256="a" * 64,
                ),
            ),
        ),
    )
    return report


def _bound_repeatability_receipt(report: Path) -> Path:
    return report.with_name("vlm_confidence_repeatability.json")


def _write_strict_confidence_run(
    root: Path,
    checkpoint: Path,
    *,
    run_id: str = "9" * 32,
    process_id: str = "a" * 32,
) -> Path:
    """Write a real report/envelope pair accepted by the public strict reader."""

    root.mkdir()
    identity = inspect_vlm_checkpoint_identity(checkpoint)
    class_names = [f"class-{index}" for index in range(80)]
    category_ids = list(range(1, 81))
    review_checks = {
        "canonical_source": True,
        "image_attribution_sufficiency": True,
        "annotation_license_and_redistribution": True,
        "privacy_and_pii": True,
        "visual_quality": True,
        "selection_salt_freeze": True,
        "benchmark_suitability": True,
        "publication_upload_authorization": True,
    }
    checkpoint_context = artifact_module._checkpoint_report_context(identity)
    dataset_context = {
        "schema": "libreyolo.vlm-confidence-benchmark-dataset.v1",
        "manifest": {
            "schema": "libreyolo.vlm-benchmark-dataset.v1",
            "sha256": "1" * 64,
        },
        "source": {
            "canonical_annotation_sha256": "2" * 64,
            "file_sha256": "3" * 64,
            "file_size_bytes": 1,
            "selected_image_identity_sha256": "4" * 64,
        },
        "partition": {
            "name": "holdout100",
            "role": "fine_tune_validation",
            "start": 0,
            "stop": 100,
            "image_count": 100,
            "annotation_artifact": ("annotations/instances_val2017_holdout100.json"),
            "annotation_size_bytes": 1,
            "annotation_sha256": "5" * 64,
        },
        "classes": {
            "count": 80,
            "names": class_names,
            "category_ids": category_ids,
        },
        "review": {
            "schema": "libreyolo.vlm-benchmark-dataset-review.v1",
            "sha256": "6" * 64,
            "manifest_sha256": "1" * 64,
            "partition_role": "fine_tune_validation",
            "status": "approved",
            "reviewer": "Offline test reviewer",
            "reviewed_at": "2026-08-16T10:30:00Z",
            "checks": review_checks,
        },
    }
    execution_context = {
        "schema": "libreyolo.vlm-confidence-benchmark-context.v3",
        "git": {"commit": "a" * 40, "dirty": False},
        "runtime": {
            "python": "3.12.0",
            "implementation": "CPython",
            "platform": "offline-test",
            "torch": "2.8.0",
            "numpy": "2.0.0",
            "pillow": "11.0.0",
            "opencv": "4.10.0",
            "packages": {
                "transformers": "5.12.1",
                "huggingface_hub": "0.36.0",
                "tokenizers": "0.22.0",
                "safetensors": "0.6.0",
                "pycocotools": "2.0.10",
            },
            "cuda_runtime": None,
            "cudnn": None,
            "nvidia_driver": None,
            "cuda_available": False,
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "attention_backends": {"model": "sdpa"},
        },
        "determinism": {
            "seed": 0,
            "python_hash_seed": "0",
            "python_hash_randomization": False,
            "cublas_workspace_config": ":4096:8",
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
        "dataset": dataset_context,
        "checkpoint": checkpoint_context,
    }
    processor_files = sum(record.role == "processor" for record in identity.files)
    benchmark = {
        "family": identity.family,
        "size": identity.size,
        "base_repo": identity.base_repo,
        "base_revision": identity.base_revision,
        "checkpoint": json.loads(json.dumps(checkpoint_context)),
        "processor": {
            "source": "checkpoint",
            "revision": None,
            "sha256": identity.processor_sha256,
            "files": processor_files,
            "class": "offline",
        },
        "class_names": class_names,
        "generation_kwargs": {
            "do_sample": False,
            "max_new_tokens": 128,
            "num_beams": 1,
            "repetition_penalty": 1.1,
        },
        "confidence_method": "qwen_generation_policy_label_bbox_geomean_v1",
        "confidence_evaluation": {
            "iou_threshold": 0.5,
            "default_conf": 0.25,
            "fallback_score": 1.0,
            "calibration_bins": 10,
            "binning": "uniform_left_closed_v1",
            "population": "scored_postprocessed_predictions",
            "matching": "class_aware_max_cardinality_iou_v1",
        },
        "evaluation": {
            "max_det": 100,
            "faster_coco_eval": False,
            "imgsz": [1024, 1024],
            "label_to_category_id": {
                str(index): category_id
                for index, category_id in enumerate(category_ids)
            },
            "backend": "pycocotools 2.0.10",
        },
        "seed": 0,
        "backend": "transformers.Qwen3VLForConditionalGeneration",
        "device": "cpu",
        "dtype": "torch.bfloat16",
        "hardware": {"type": "cpu", "name": "offline-test"},
        "software": {
            "python": "3.12.0",
            "libreyolo": "1.6.0",
            "torch": "2.8.0",
            "transformers": "5.12.1",
            "pycocotools": "2.0.10",
        },
        "benchmark_run": execution_context,
    }
    target_box = (10.0, 10.0, 30.0, 30.0)
    wrong_box = (50.0, 50.0, 70.0, 70.0)
    predictions = []
    ground_truth = []
    images = []
    evaluator_images = []
    annotations = []
    ground_truth_rows = []
    generations = []
    for image_number in range(1, 101):
        image_id = str(image_number)
        class_id = (image_number - 1) % len(class_names)
        category_id = category_ids[class_id]
        predictions.extend(
            (
                VLMDetection(image_id, class_id, target_box, 0.75),
                VLMDetection(image_id, class_id, wrong_box, 0.125),
            )
        )
        ground_truth.append(VLMDetection(image_id, class_id, target_box))
        images.append(
            {
                "image_id": image_id,
                "file_name": f"{image_number:012d}.jpg",
                "sha256": f"{image_number:064x}",
                "width": 100,
                "height": 100,
            }
        )
        evaluator_images.append({"id": image_number, "width": 100, "height": 100})
        annotations.append(
            {
                "id": image_number,
                "image_id": image_number,
                "category_id": category_id,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
                "ignore": 0,
            }
        )
        ground_truth_rows.append(
            {"image_id": image_id, "class_id": class_id, "xyxy": list(target_box)}
        )
        generations.append(
            {
                "image_id": image_id,
                "sha256": f"{image_number + 100:064x}",
                "parsed_items": 2,
                "fallback_reason": None,
            }
        )
    dataset = {
        "split": "val",
        "class_names": class_names,
        "images": images,
        "evaluator_ground_truth": {
            "api": "offline.StubCOCO",
            "images": evaluator_images,
            "categories": [
                {"id": category_id, "name": class_name}
                for category_id, class_name in zip(category_ids, class_names)
            ],
            "annotations": annotations,
        },
        "ground_truth": ground_truth_rows,
    }
    evaluator = {
        "candidate_mAP50-95": 0.625,
        "constant_mAP50-95": 0.5,
        "candidate_mAP50": 0.75,
        "constant_mAP50": 0.5,
    }
    run = build_confidence_run(
        predictions,
        ground_truth,
        prompt="detect all COCO classes",
        dataset_manifest=dataset,
        benchmark_config=benchmark,
        generation_manifest=generations,
        evaluator_metrics=evaluator,
        iou_threshold=0.5,
        default_conf=0.25,
        fallback_score=1.0,
    )
    metrics = confidence_report_module._semantic_metrics(run, (100, 100, 200, 200))
    metrics.update(
        {
            "speed/preprocess_ms": 1.0,
            "speed/inference_ms": 2.0,
            "speed/postprocess_ms": 1.0,
            "speed/total_ms": 10.0,
            "speed/total_s": 1.0,
            "speed/images_seen": 100.0,
        }
    )
    report_payload = {
        "schema": "libreyolo.vlm-confidence-report.v2",
        "prompt": "detect all COCO classes",
        "benchmark_config": benchmark,
        "dataset_manifest": dataset,
        "generation_manifest": generations,
        "hashes": {
            "manifest": run.manifest_hash,
            "configuration": run.configuration_hash,
            "generation": run.generation_hash,
            "prediction_structure": run.prediction_structure_hash,
        },
        "confidence": {
            "iou_threshold": run.iou_threshold,
            "default_conf": run.default_conf,
            "fallback_score": run.fallback_score,
        },
        "diagnostics": confidence_report_module._diagnostics_surface(run),
        "calibration": confidence_report_module._calibration_surface(run),
        "evaluator_metrics": evaluator,
        "fallback_reasons": {},
        "predictions": [
            {
                "image_id": prediction.image_id,
                "class_id": prediction.class_id,
                "xyxy": list(prediction.xyxy),
                "candidate_score": prediction.score,
                "effective_score": prediction.score,
                "matched": matched,
            }
            for prediction, matched in zip(predictions, run.matches)
        ],
        "metrics": metrics,
        "artifacts": {"reliability_plot": None},
    }
    report_path = root / "vlm_confidence_report.json"
    report_path.write_bytes(_canonical(report_payload))

    def absolute(name: str) -> str:
        return str((root / name).resolve())

    envelope = {
        "schema": "libreyolo.vlm-confidence-benchmark-run.v3",
        "run_id": run_id,
        "process_id": process_id,
        "request": {
            "manifest": absolute("manifest.json"),
            "annotations": absolute("instances_val2017.json"),
            "images_dir": absolute("val2017"),
            "review_attestation": absolute("review.json"),
            "seed": 0,
            "model_family": "qwen3vl",
            "model_size": identity.size,
            "checkpoint_dir": str(checkpoint.resolve()),
            "device": "cpu",
            "imgsz": 1024,
            "default_conf": 0.25,
            "confidence_iou": 0.5,
        },
        "execution_context": execution_context,
        "report": {
            "path": "vlm_confidence_report.json",
            "sha256": _sha(report_path.read_bytes()),
        },
        "metrics": metrics,
        "nonfinite_metrics": [],
    }
    (root / "vlm_confidence_run.json").write_bytes(_canonical(envelope))
    return report_path


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
    assert PUBLICATION_EVIDENCE_SCHEMA == "libreyolo.vlm-publication-evidence.v2"
    assert (
        artifact_module._EVALUATION_CLAIM_SCHEMA
        == "libreyolo.vlm-publication-evaluation-claim.v2"
    )
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


def test_safetensors_validation_returns_the_validated_payload_digest(tmp_path):
    payload = _safetensor()
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(payload)
    assert artifact_module._validate_safetensors(path, "2b") == _sha(payload)


def test_publication_validation_metrics_match_strict_report_evaluation_surface():
    retention_and_coverage = {
        "metrics/vlm_confidence/default_conf_tp_retention",
        "metrics/vlm_confidence/default_conf_fp_retention",
        "metrics/vlm_confidence/default_conf_prediction_retention",
        "metrics/vlm_confidence/response_score_coverage",
        "metrics/vlm_confidence/detection_score_coverage",
        "metrics/vlm_confidence/prediction_score_coverage",
    }
    assert artifact_module._CONFIDENCE_VALIDATION_METRICS == (
        confidence_report_module._MAP_METRICS
        | confidence_report_module._QUALITY_METRICS
        | retention_and_coverage
    )
    assert artifact_module._CONFIDENCE_VALIDATION_METRICS <= (
        confidence_report_module._SEMANTIC_METRICS
    )
    assert artifact_module._CONFIDENCE_VALIDATION_METRICS.isdisjoint(
        confidence_report_module._SPEED_METRICS
    )


def test_publication_rejects_nonvalidation_benchmark_identity(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence_value = _evidence()
    evidence_value["evaluation"]["benchmark"] = "arbitrary-benchmark"
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)

    with pytest.raises(VLMArtifactError, match="evaluation.benchmark must be"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda metrics: metrics.pop("metrics/vlm_confidence/auroc"),
            "invalid keys",
        ),
        (lambda metrics: metrics.__setitem__("unknown", 0.5), "invalid keys"),
        (
            lambda metrics: metrics.__setitem__("metrics/vlm_confidence/auroc", -0.01),
            "between 0 and 1",
        ),
        (
            lambda metrics: metrics.__setitem__(
                "metrics/vlm_confidence/response_score_coverage", 1.01
            ),
            "between 0 and 1",
        ),
        (
            lambda metrics: metrics.__setitem__(
                "metrics/vlm_confidence/delta_mAP50", 1.01
            ),
            "between -1 and 1",
        ),
        (
            lambda metrics: metrics.__setitem__(
                "metrics/vlm_confidence/delta_mAP50", 0.2
            ),
            "must equal",
        ),
    ],
)
def test_publication_rejects_invalid_validation_metric_contract(
    tmp_path, mutation, message
):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence_value = _evidence()
    mutation(evidence_value["evaluation"]["metrics"])
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)

    with pytest.raises(VLMArtifactError, match=message):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda evidence: evidence["evaluation"]["metrics"].__setitem__(
            "metrics/vlm_confidence/auroc", 0.5
        ),
        lambda evidence: evidence["evaluation"].__setitem__(
            "envelope_sha256", "0" * 64
        ),
        lambda evidence: evidence["evaluation"]["repeatability"].__setitem__(
            "comparison_sha256", "0" * 64
        ),
    ],
)
def test_publication_rejects_evaluation_edit_with_stale_claim_binding(
    tmp_path, mutation
):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence_value = _evidence()
    mutation(evidence_value)
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)

    with pytest.raises(
        VLMArtifactError,
        match="review.bindings do not match|must match the primary evaluation run",
    ):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


def test_publication_review_must_bind_built_adapter_config(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence_value = _evidence()
    evidence_value["review"]["bindings"]["adapter_config_sha256"] = "0" * 64
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)

    with pytest.raises(VLMArtifactError, match="adapter_config_sha256"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )
    assert not (tmp_path / "artifact").exists()


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


def test_stable_copy_binds_the_opened_source_descriptor(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    alternate = tmp_path / "alternate.bin"
    destination = tmp_path / "copied.bin"
    source.write_bytes(b"trusted-source")
    alternate.write_bytes(b"forged-source!")
    assert source.stat().st_size == alternate.stat().st_size
    real_open = artifact_module.os.open

    def redirected_open(path, flags, *args, **kwargs):
        if Path(path) == source:
            return real_open(alternate, flags, *args, **kwargs)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(artifact_module.os, "open", redirected_open)
    with pytest.raises(VLMArtifactError, match="changed before it was opened"):
        artifact_module._copy_file_stable(source, destination)
    assert not destination.exists()


def test_bounded_read_binds_the_opened_source_descriptor(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    alternate = tmp_path / "alternate.json"
    source.write_bytes(b'{"trusted":1}')
    alternate.write_bytes(b'{"forged!":1}')
    assert source.stat().st_size == alternate.stat().st_size
    real_open = artifact_module.os.open

    def redirected_open(path, flags, *args, **kwargs):
        if Path(path) == source:
            return real_open(alternate, flags, *args, **kwargs)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(artifact_module.os, "open", redirected_open)
    with pytest.raises(VLMArtifactError, match="changed before it was opened"):
        artifact_module._read_bounded(source, max_bytes=1024, label="source JSON")


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
    training_data, code = _template_context()
    confidence_report = _bind_confidence_report(monkeypatch, checkpoint)
    output = tmp_path / "publication-template.json"
    result = create_vlm_publication_evidence_template(
        checkpoint,
        base,
        output,
        training_data=training_data,
        code=code,
        confidence_report=confidence_report,
        repeatability_receipt=_bound_repeatability_receipt(confidence_report),
    )
    return checkpoint, base, result


def _strict_template_inputs(tmp_path: Path, monkeypatch):
    identity = _tiny_base_identity(monkeypatch)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    base = tmp_path / "base"
    _materialize_base_snapshot(base, identity)
    report = _write_strict_confidence_run(tmp_path / "confidence-run", checkpoint)
    repeated_report = _write_strict_confidence_run(
        tmp_path / "confidence-run-repeat",
        checkpoint,
        run_id="b" * 32,
        process_id="c" * 32,
    )
    receipt = tmp_path / "confidence-repeatability.json"
    confidence_benchmark.create_benchmark_repeatability_receipt(
        report, repeated_report, receipt
    )
    training_data, code = _template_context()
    return checkpoint, base, report, receipt, training_data, code


def test_publication_template_reads_real_strict_report_and_envelope(
    tmp_path, monkeypatch
):
    checkpoint, base, report, receipt, training_data, code = _strict_template_inputs(
        tmp_path, monkeypatch
    )
    output = tmp_path / "publication-template.json"

    create_vlm_publication_evidence_template(
        checkpoint,
        base,
        output,
        training_data=training_data,
        code=code,
        confidence_report=report,
        repeatability_receipt=receipt,
    )

    evidence = json.loads(output.read_text(encoding="utf-8"))
    envelope = report.with_name("vlm_confidence_run.json")
    assert evidence["evaluation"]["report_sha256"] == _sha(report.read_bytes())
    assert evidence["evaluation"]["envelope_sha256"] == _sha(envelope.read_bytes())
    receipt_identity = confidence_benchmark.read_benchmark_repeatability_receipt(
        receipt
    )
    assert evidence["evaluation"]["repeatability"]["receipt_sha256"] == (
        receipt_identity.receipt_sha256
    )
    assert evidence["evaluation"]["repeatability"]["comparison_sha256"] == (
        receipt_identity.comparison_sha256
    )
    assert evidence["evaluation"]["repeatability"]["runs"][0]["report_sha256"] == _sha(
        report.read_bytes()
    )
    assert evidence["evaluation"]["benchmark"] == (
        artifact_module._CONFIDENCE_BENCHMARK_ID
    )
    assert set(evidence["evaluation"]["metrics"]) == (
        artifact_module._CONFIDENCE_VALIDATION_METRICS
    )
    assert evidence["review"]["bindings"]["evaluation_claim_sha256"] == (
        artifact_module._evaluation_claim_sha256(evidence["evaluation"])
    )


def test_publication_template_rejects_missing_envelope_without_output(
    tmp_path, monkeypatch
):
    checkpoint, base, report, receipt, training_data, code = _strict_template_inputs(
        tmp_path, monkeypatch
    )
    report.with_name("vlm_confidence_run.json").unlink()
    output = tmp_path / "publication-template.json"

    with pytest.raises(VLMArtifactError, match="missing companion"):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            output,
            training_data=training_data,
            code=code,
            confidence_report=report,
            repeatability_receipt=receipt,
        )
    assert not output.exists()


def test_publication_template_rejects_tampered_envelope_without_output(
    tmp_path, monkeypatch
):
    checkpoint, base, report, receipt, training_data, code = _strict_template_inputs(
        tmp_path, monkeypatch
    )
    envelope_path = report.with_name("vlm_confidence_run.json")
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["report"]["sha256"] = "0" * 64
    envelope_path.write_bytes(_canonical(envelope))
    output = tmp_path / "publication-template.json"

    with pytest.raises(VLMArtifactError, match="companion report bytes"):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            output,
            training_data=training_data,
            code=code,
            confidence_report=report,
            repeatability_receipt=receipt,
        )
    assert not output.exists()


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
    assert template["evaluation"]["benchmark"] == (
        artifact_module._CONFIDENCE_BENCHMARK_ID
    )
    assert set(template["evaluation"]["metrics"]) == (
        artifact_module._CONFIDENCE_VALIDATION_METRICS
    )
    assert "speed/total_ms" not in template["evaluation"]["metrics"]
    assert "metrics/vlm_confidence/responses" not in template["evaluation"]["metrics"]
    assert template["review"]["approved"] is False
    assert template["review"]["reviewer"] == ""
    assert template["review"]["reviewed_at"] == ""
    assert set(template["review"]["gates"].values()) == {False}
    assert template["review"]["bindings"] == {
        "adapter_config_sha256": inspect_vlm_checkpoint_identity(
            checkpoint
        ).adapter_config_sha256,
        "adapter_weights_sha256": _sha(_safetensor()),
        "base_snapshot_sha256": template["base_model"]["snapshot"]["sha256"],
        "checkpoint_contract_sha256": _sha(_canonical(_contract())),
        "code_revision": "c" * 40,
        "evaluation_claim_sha256": artifact_module._evaluation_claim_sha256(
            template["evaluation"]
        ),
        "evaluation_envelope_sha256": "f" * 64,
        "evaluation_repeatability_receipt_sha256": "7" * 64,
        "evaluation_repeatability_comparison_sha256": "8" * 64,
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


@pytest.mark.parametrize("mode", ["nonreproducible", "tolerant", "wrong_primary"])
def test_publication_template_rejects_invalid_repeatability_claim_source(
    tmp_path, monkeypatch, mode
):
    identity = _tiny_base_identity(monkeypatch)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    base = tmp_path / "base"
    _materialize_base_snapshot(base, identity)
    training_data, code = _template_context()
    confidence_report = _bind_confidence_report(monkeypatch, checkpoint)
    read_receipt = artifact_module._read_confidence_repeatability_identity

    def altered_receipt(path):
        current = read_receipt(path)
        values = vars(current).copy()
        if mode == "nonreproducible":
            values["comparison"] = SimpleNamespace(reproducible=False)
        elif mode == "tolerant":
            values["tolerances"] = {
                "score_atol": 0.01,
                "metric_atol": 0.0,
                "map_atol": 0.0,
            }
        else:
            runs = list(values["runs"])
            first = vars(runs[0]).copy()
            first["run_id"] = "5" * 32
            runs[0] = SimpleNamespace(**first)
            values["runs"] = tuple(runs)
        return SimpleNamespace(**values)

    monkeypatch.setattr(
        artifact_module,
        "_read_confidence_repeatability_identity",
        altered_receipt,
    )
    output = tmp_path / "publication-template.json"

    with pytest.raises(
        VLMArtifactError, match="reproducible|zero comparison tolerances|runs\[0\]"
    ):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            output,
            training_data=training_data,
            code=code,
            confidence_report=confidence_report,
            repeatability_receipt=_bound_repeatability_receipt(confidence_report),
        )
    assert not output.exists()


def test_publication_template_rechecks_repeatability_receipt_before_output(
    tmp_path, monkeypatch
):
    (
        checkpoint,
        base,
        confidence_report,
        receipt,
        training_data,
        code,
    ) = _strict_template_inputs(tmp_path, monkeypatch)
    read_receipt = artifact_module._read_confidence_repeatability_identity
    calls = 0

    def changed_second_read(path):
        nonlocal calls
        calls += 1
        current = read_receipt(path)
        if calls == 1:
            receipt.write_bytes(receipt.read_bytes() + b" ")
        return current

    monkeypatch.setattr(
        artifact_module,
        "_read_confidence_repeatability_identity",
        changed_second_read,
    )
    output = tmp_path / "publication-template.json"

    with pytest.raises(VLMArtifactError, match="repeatability receipt|canonical JSON"):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            output,
            training_data=training_data,
            code=code,
            confidence_report=confidence_report,
            repeatability_receipt=receipt,
        )
    assert calls == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda evidence: evidence["evaluation"]["repeatability"][
                "tolerances"
            ].__setitem__("map_atol", 1.0),
            "map_atol must equal 0",
        ),
        (
            lambda evidence: evidence["evaluation"]["repeatability"]["runs"][
                1
            ].__setitem__(
                "process_id",
                evidence["evaluation"]["repeatability"]["runs"][0]["process_id"],
            ),
            "process_id values must differ",
        ),
        (
            lambda evidence: evidence["evaluation"]["repeatability"].__setitem__(
                "reproducible", False
            ),
            "reproducible must be true",
        ),
    ],
)
def test_publication_evidence_rejects_invalid_repeatability_claim(
    tmp_path, mutation, message
):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence_value = _evidence()
    mutation(evidence_value)
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)

    with pytest.raises(VLMArtifactError, match=message):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda context: context["checkpoint"].__setitem__(
            "adapter_weights_sha256", "0" * 64
        ),
        lambda context: context["checkpoint"].__setitem__("base_revision", "0" * 40),
        lambda context: context["checkpoint"].__setitem__("processor_sha256", "0" * 64),
        lambda context: context["checkpoint"]["files"][0].__setitem__(
            "sha256", "0" * 64
        ),
    ],
)
def test_publication_template_rejects_report_checkpoint_identity_mismatch(
    tmp_path, monkeypatch, mutation
):
    identity = _tiny_base_identity(monkeypatch)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    base = tmp_path / "base"
    _materialize_base_snapshot(base, identity)
    training_data, code = _template_context()
    confidence_report = _bind_confidence_report(
        monkeypatch, checkpoint, mutate_context=mutation
    )
    output = tmp_path / "publication-template.json"

    with pytest.raises(VLMArtifactError, match="checkpoint identity"):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            output,
            training_data=training_data,
            code=code,
            confidence_report=confidence_report,
            repeatability_receipt=_bound_repeatability_receipt(confidence_report),
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda context: context.__setitem__("schema", "context.v2"),
            "execution_context schema",
        ),
        (
            lambda context: context["dataset"]["partition"].__setitem__(
                "role", "training"
            ),
            "holdout100",
        ),
    ],
)
def test_publication_template_rejects_nonvalidation_report_context(
    tmp_path, monkeypatch, mutation, message
):
    identity = _tiny_base_identity(monkeypatch)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    base = tmp_path / "base"
    _materialize_base_snapshot(base, identity)
    training_data, code = _template_context()
    confidence_report = _bind_confidence_report(
        monkeypatch, checkpoint, mutate_context=mutation
    )

    with pytest.raises(VLMArtifactError, match=message):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            tmp_path / "publication-template.json",
            training_data=training_data,
            code=code,
            confidence_report=confidence_report,
            repeatability_receipt=_bound_repeatability_receipt(confidence_report),
        )


@pytest.mark.parametrize("metric_value", [None, float("nan"), float("inf")])
def test_publication_template_rejects_unusable_validation_metric(
    tmp_path, monkeypatch, metric_value
):
    identity = _tiny_base_identity(monkeypatch)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    base = tmp_path / "base"
    _materialize_base_snapshot(base, identity)
    training_data, code = _template_context()
    metrics = _confidence_metrics()
    metrics["metrics/vlm_confidence/ranking_ap"] = metric_value
    confidence_report = _bind_confidence_report(
        monkeypatch, checkpoint, metrics=metrics
    )
    output = tmp_path / "publication-template.json"

    with pytest.raises(VLMArtifactError, match="validation metrics|finite"):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            output,
            training_data=training_data,
            code=code,
            confidence_report=confidence_report,
            repeatability_receipt=_bound_repeatability_receipt(confidence_report),
        )
    assert not output.exists()


def test_publication_template_rejects_checkpoint_mutation_during_report_read(
    tmp_path, monkeypatch
):
    identity = _tiny_base_identity(monkeypatch)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    base = tmp_path / "base"
    _materialize_base_snapshot(base, identity)
    training_data, code = _template_context()
    confidence_report = _bind_confidence_report(monkeypatch, checkpoint)
    read_identity = artifact_module._read_confidence_benchmark_identity

    def mutate_during_read(path):
        result = read_identity(path)
        contract_path = checkpoint / "libreyolo_vlm.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["metrics"]["epoch"] = 1
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        return result

    monkeypatch.setattr(
        artifact_module, "_read_confidence_benchmark_identity", mutate_during_read
    )
    output = tmp_path / "publication-template.json"
    with pytest.raises(VLMArtifactError, match="changed"):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            output,
            training_data=training_data,
            code=code,
            confidence_report=confidence_report,
            repeatability_receipt=_bound_repeatability_receipt(confidence_report),
        )
    assert not output.exists()


def test_publication_template_rechecks_run_identity_before_creating_output(
    tmp_path, monkeypatch
):
    (
        checkpoint,
        base,
        confidence_report,
        receipt,
        training_data,
        code,
    ) = _strict_template_inputs(tmp_path, monkeypatch)
    read_identity = artifact_module._read_confidence_benchmark_identity
    calls = 0

    def mutate_after_first_read(path):
        nonlocal calls
        calls += 1
        result = read_identity(path)
        if calls == 1:
            envelope_path = confidence_report.with_name("vlm_confidence_run.json")
            envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
            envelope["run_id"] = "b" * 32
            envelope_path.write_bytes(_canonical(envelope))
        return result

    monkeypatch.setattr(
        artifact_module,
        "_read_confidence_benchmark_identity",
        mutate_after_first_read,
    )
    output = tmp_path / "publication-template.json"

    with pytest.raises(
        VLMArtifactError, match="benchmark.*changed|must match the confidence_report"
    ):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            output,
            training_data=training_data,
            code=code,
            confidence_report=confidence_report,
            repeatability_receipt=receipt,
        )
    assert calls == 2
    assert not output.exists()


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
    training_data, code = _template_context()
    confidence_report = _bind_confidence_report(monkeypatch, checkpoint)
    existing = tmp_path / "publication-template.json"
    existing.write_text("racer", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            existing,
            training_data=training_data,
            code=code,
            confidence_report=confidence_report,
            repeatability_receipt=_bound_repeatability_receipt(confidence_report),
        )
    assert existing.read_text(encoding="utf-8") == "racer"

    with pytest.raises(VLMArtifactError, match="outside the checkpoint"):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            checkpoint / "publication-template.json",
            training_data=training_data,
            code=code,
            confidence_report=confidence_report,
            repeatability_receipt=_bound_repeatability_receipt(confidence_report),
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_publication_template_rejects_symlinked_output_parent(tmp_path, monkeypatch):
    identity = _tiny_base_identity(monkeypatch)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    base = tmp_path / "base"
    _materialize_base_snapshot(base, identity)
    training_data, code = _template_context()
    confidence_report = _bind_confidence_report(monkeypatch, checkpoint)
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
            code=code,
            confidence_report=confidence_report,
            repeatability_receipt=_bound_repeatability_receipt(confidence_report),
        )


def test_publication_template_preserves_racing_destination(tmp_path, monkeypatch):
    identity = _tiny_base_identity(monkeypatch)
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    base = tmp_path / "base"
    _materialize_base_snapshot(base, identity)
    training_data, code = _template_context()
    confidence_report = _bind_confidence_report(monkeypatch, checkpoint)
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
            code=code,
            confidence_report=confidence_report,
            repeatability_receipt=_bound_repeatability_receipt(confidence_report),
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
    training_data, code = _template_context()
    confidence_report = _bind_confidence_report(monkeypatch, checkpoint)
    training_data["unknown"] = True
    with pytest.raises(VLMArtifactError, match="invalid keys"):
        create_vlm_publication_evidence_template(
            checkpoint,
            base,
            tmp_path / "publication-template.json",
            training_data=training_data,
            code=code,
            confidence_report=confidence_report,
            repeatability_receipt=_bound_repeatability_receipt(confidence_report),
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


def test_model_card_uses_safe_code_spans_for_dynamic_labels(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    malicious = "worker`](https://evil.example)"
    contract = _contract()
    contract["names"] = [malicious]
    contract["prompt"] = _prompt(contract["names"])
    (checkpoint / "libreyolo_vlm.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    evidence_value = _evidence()
    evidence_value["review"]["bindings"]["checkpoint_contract_sha256"] = _sha(
        _canonical(contract)
    )
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)

    info = build_vlm_artifact(
        checkpoint, tmp_path / "artifact", publication_evidence=evidence
    )
    card = (info.root / "README.md").read_text(encoding="utf-8")
    assert card.count("`` worker`](https://evil.example) ``") == 1


def test_publication_rejects_unicode_bidi_controls_in_label(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence_value = _evidence()
    unsafe = "worker\u202eevil"
    contract = _contract()
    contract["names"] = [unsafe]
    contract["prompt"] = _prompt(contract["names"])
    (checkpoint / "libreyolo_vlm.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)
    with pytest.raises(VLMArtifactError, match="safe|normalized"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )


def test_publication_rejects_unknown_metric_key(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    evidence_value = _evidence()
    evidence_value["evaluation"]["metrics"] = {"unknown": 0.8}
    evidence = _write_evidence(tmp_path / "publication.json", value=evidence_value)
    with pytest.raises(VLMArtifactError, match="evaluation.metrics has invalid keys"):
        build_vlm_artifact(
            checkpoint, tmp_path / "artifact", publication_evidence=evidence
        )
