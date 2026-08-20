"""Offline tests for the internal VLM confidence benchmark runner."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import socket
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.validation import vlm_confidence_benchmark as benchmark
from libreyolo.validation.vlm_confidence_report import VLMConfidenceReportError

pytestmark = [pytest.mark.unit, pytest.mark.vlm]


_REVIEW_CHECKS = {
    "canonical_source": True,
    "image_attribution_sufficiency": True,
    "annotation_license_and_redistribution": True,
    "privacy_and_pii": True,
    "visual_quality": True,
    "selection_salt_freeze": True,
    "benchmark_suitability": True,
    "publication_upload_authorization": True,
}


def _verified_inputs(
    tmp_path: Path, *, required_role: str = benchmark._BASE_PARTITION_ROLE
):
    partition = benchmark._PARTITION_REQUIREMENTS[required_role]
    bundle = tmp_path / "bundle"
    annotation = bundle / partition["annotation_artifact"]
    annotation.parent.mkdir(parents=True)
    annotation.write_text("{}\n", encoding="utf-8")
    manifest = bundle / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "instances_val2017.json"
    source.write_text("{}\n", encoding="utf-8")
    images = tmp_path / "val2017"
    images.mkdir()
    review = tmp_path / "review.json"
    review.write_text("{}\n", encoding="utf-8")
    manifest_sha256 = "1" * 64
    return SimpleNamespace(
        manifest_path=manifest.resolve(),
        manifest_sha256=manifest_sha256,
        source_annotations=source.resolve(),
        source_canonical_sha256="2" * 64,
        source_file_sha256="6" * 64,
        source_file_size_bytes=source.stat().st_size,
        images_dir=images.resolve(),
        selected_image_identity_sha256="3" * 64,
        partition_name=partition["name"],
        partition_role=required_role,
        partition_start=partition["start"],
        partition_stop=partition["stop"],
        annotation_path=annotation.resolve(),
        annotation_sha256="4" * 64,
        annotation_size_bytes=annotation.stat().st_size,
        class_names=tuple(f"class-{index}" for index in range(80)),
        expected_images=(),
        expected_categories=tuple(
            {"id": index + 1, "name": f"class-{index}"} for index in range(80)
        ),
        expected_annotations=(),
        review_attestation_path=review.resolve(),
        review_attestation_sha256="5" * 64,
        review_attestation={
            "schema": "libreyolo.vlm-benchmark-dataset-review.v1",
            "manifest_sha256": manifest_sha256,
            "partition_role": required_role,
            "status": "approved",
            "reviewer": "Offline test reviewer",
            "reviewed_at": "2026-08-16T10:30:00Z",
            "checks": dict(_REVIEW_CHECKS),
        },
    )


def _strict_checkpoint_identity(tmp_path: Path, *, size: str = "4b", token: str = "a"):
    root = tmp_path / f"checkpoint-{size}"
    root.mkdir(exist_ok=True)
    repo, revision = benchmark._QWEN_BASE_PINS[size]
    weights_sha256 = "b" * 64
    files = (
        SimpleNamespace(
            path="adapter_config.json",
            role="adapter_config",
            size=101,
            sha256=token * 64,
        ),
        SimpleNamespace(
            path="adapter_model.safetensors",
            role="adapter_weights",
            size=102,
            sha256=weights_sha256,
        ),
        SimpleNamespace(
            path="libreyolo_vlm.json",
            role="checkpoint_contract",
            size=103,
            sha256="2" * 64,
        ),
        SimpleNamespace(
            path="processor_config.json",
            role="processor",
            size=104,
            sha256="3" * 64,
        ),
    )
    for entry in files:
        artifact = root / entry.path
        if not artifact.exists():
            artifact.write_bytes(f"offline-{entry.path}\n".encode())
    serialized_files = [
        {
            "path": entry.path,
            "role": entry.role,
            "size": entry.size,
            "sha256": entry.sha256,
        }
        for entry in files
    ]
    processor_sha256 = benchmark._checkpoint_processor_sha256(serialized_files)
    identity_values = {
        "family": "qwen3vl",
        "size": size,
        "task": "detect",
        "base_repo": repo,
        "base_revision": revision,
        "adapter_weights_sha256": weights_sha256,
        "adapter_config_sha256": "c" * 64,
        "checkpoint_contract_sha256": "d" * 64,
        "processor_sha256": processor_sha256,
    }
    return SimpleNamespace(
        root=root.resolve(),
        files=files,
        **identity_values,
        aggregate_sha256=benchmark._checkpoint_aggregate_sha256(
            identity_values, serialized_files
        ),
    )


def _verified_inputs_for_role(verified, required_role: str):
    if verified.partition_role == required_role:
        return verified
    partition = benchmark._PARTITION_REQUIREMENTS[required_role]
    annotation = verified.manifest_path.parent / partition["annotation_artifact"]
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text("{}\n", encoding="utf-8")
    values = vars(verified).copy()
    review = dict(verified.review_attestation)
    review["partition_role"] = required_role
    values.update(
        partition_name=partition["name"],
        partition_role=required_role,
        partition_start=partition["start"],
        partition_stop=partition["stop"],
        annotation_path=annotation.resolve(),
        annotation_size_bytes=annotation.stat().st_size,
        review_attestation=review,
    )
    return SimpleNamespace(**values)


def _checkpoint_identity_at(identity, root: Path):
    values = vars(identity).copy()
    values["root"] = Path(root).resolve()
    return SimpleNamespace(**values)


def _matching_checkpoint_inspector(identity, inspected=None):
    def inspect(path):
        root = Path(path).resolve()
        if inspected is not None:
            inspected.append(root)
        return _checkpoint_identity_at(identity, root)

    return inspect


def _install_checkpoint_cleanup_failure(monkeypatch):
    real_temporary_directory = benchmark.tempfile.TemporaryDirectory
    isolated_roots = []

    class FailingCleanup:
        def __init__(self, *args, **kwargs):
            self._temporary = real_temporary_directory(*args, **kwargs)
            self.name = self._temporary.name
            isolated_roots.append(Path(self.name).resolve() / "checkpoint")

        def cleanup(self):
            self._temporary.cleanup()
            raise OSError("injected isolated checkpoint cleanup failure")

    def temporary_directory(*args, **kwargs):
        if kwargs.get("prefix") == benchmark._CHECKPOINT_TEMP_PREFIX:
            return FailingCleanup(*args, **kwargs)
        return real_temporary_directory(*args, **kwargs)

    monkeypatch.setattr(benchmark.tempfile, "TemporaryDirectory", temporary_directory)
    return isolated_roots


def _install_base_cleanup_failure(monkeypatch):
    real_temporary_directory = benchmark.tempfile.TemporaryDirectory
    isolated_roots = []

    class FailingCleanup:
        def __init__(self, *args, **kwargs):
            self._temporary = real_temporary_directory(*args, **kwargs)
            self.name = self._temporary.name
            isolated_roots.append(Path(self.name).resolve() / "snapshot")

        def cleanup(self):
            self._temporary.cleanup()
            raise OSError("injected isolated base cleanup failure")

    def temporary_directory(*args, **kwargs):
        if kwargs.get("prefix") == benchmark._BASE_SNAPSHOT_TEMP_PREFIX:
            return FailingCleanup(*args, **kwargs)
        return real_temporary_directory(*args, **kwargs)

    monkeypatch.setattr(benchmark.tempfile, "TemporaryDirectory", temporary_directory)
    return isolated_roots


def _compact_base_snapshot(tmp_path, monkeypatch):
    source = (tmp_path / "compact-base").resolve()
    source.mkdir()
    (source / ".libreyolo_snapshot_complete").write_text(
        json.dumps(
            {
                "repo": benchmark._QWEN_2B_REPO,
                "revision": benchmark._QWEN_2B_REVISION,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"base-A")
    (source / "preprocessor_config.json").write_text("{}\n", encoding="utf-8")
    snapshot = {
        "kind": "pinned_hf_snapshot",
        "artifacts": [
            {"path": "config.json"},
            {"path": "model.safetensors"},
        ],
        "sha256": "a" * 64,
    }
    processor = {"files": 2, "sha256": "b" * 64}
    changed_snapshot = {**snapshot, "sha256": "c" * 64}

    def inspect(size="2b", *, expected_repo=None, expected_revision=None, root=None):
        del size, expected_repo, expected_revision
        actual_root = Path(root)
        selected = (
            snapshot
            if (actual_root / "model.safetensors").read_bytes() == b"base-A"
            else changed_snapshot
        )
        return actual_root, selected, processor

    monkeypatch.setattr(benchmark, "_snapshot_evidence", inspect)
    return source, snapshot, processor


def _run(verified, output, **kwargs):
    return benchmark.run_benchmark(
        verified.manifest_path,
        verified.source_annotations,
        verified.images_dir,
        verified.review_attestation_path,
        output,
        **kwargs,
    )


def _preflight(verified, output, **kwargs):
    return benchmark.preflight_benchmark(
        verified.manifest_path,
        verified.source_annotations,
        verified.images_dir,
        verified.review_attestation_path,
        output,
        **kwargs,
    )


def _cli_request_args(mode, verified, output):
    return [
        mode,
        "--manifest",
        str(verified.manifest_path),
        "--annotations",
        str(verified.source_annotations),
        "--images-dir",
        str(verified.images_dir),
        "--review-attestation",
        str(verified.review_attestation_path),
        "--output-root",
        str(output),
    ]


def _cli_run_args(verified, output):
    return _cli_request_args("run", verified, output)


def _cli_preflight_args(verified, output):
    return _cli_request_args("preflight", verified, output)


def _install_run_fakes(
    monkeypatch,
    events,
    verified,
    *,
    metrics=None,
    failure=None,
    model_hook=None,
):
    metrics = {"metric/finite": 0.5} if metrics is None else metrics
    report_identities = {}
    snapshot_identity = {
        "kind": "pinned_hf_snapshot",
        "sha256": "7" * 64,
    }
    processor_identity = {
        "source": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": "8" * 40,
        "sha256": "9" * 64,
        "files": 4,
    }

    def fake_determinism(seed):
        events.append(("determinism", seed))
        return {
            "seed": seed,
            "python_hash_seed": "0",
            "python_hash_randomization": False,
            "cublas_workspace_config": ":4096:8",
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }

    class FakeModel:
        def __init__(self, *, size, device, checkpoint_dir=None):
            events.append(("model", size, device, checkpoint_dir))
            assert [event[0] for event in events[:10]] == [
                "determinism",
                "offline",
                "dependencies",
                "pycocotools",
                "device",
                "verify",
                "data_config",
                "native_dataset",
                "snapshot",
                "verify",
            ]
            if model_hook is not None:
                model_hook(checkpoint_dir)
            self.device = torch.device("cpu" if device == "auto" else device)
            self.size = size
            self._checkpoint_dir = (
                None if checkpoint_dir is None else Path(checkpoint_dir)
            )

    class FakeValidator:
        _strict_local_directory_root = staticmethod(
            benchmark.VLMConfidenceValidator._strict_local_directory_root
        )

        def __init__(self, model, config, *, seed, **kwargs):
            events.append(("validator", seed, config, kwargs))
            self.model = model
            self.config = config
            self.seed = seed
            self.kwargs = kwargs

        def run(self):
            events.append(("run",))
            report = Path(self.config.save_dir) / "vlm_confidence_report.json"
            report.write_text("{}\n", encoding="utf-8")
            if failure is not None:
                raise failure
            normalized, _ = benchmark._normalized_metrics(metrics)
            checkpoint_context = self.kwargs["benchmark_context"]["checkpoint"]
            processor = (
                {
                    "source": benchmark._QWEN_BASE_PINS[self.model.size][0],
                    "revision": benchmark._QWEN_BASE_PINS[self.model.size][1],
                    "sha256": "9" * 64,
                    "files": 4,
                    "class": "offline",
                }
                if checkpoint_context is None
                else {
                    "source": "checkpoint",
                    "revision": None,
                    "sha256": checkpoint_context["processor_sha256"],
                    "files": sum(
                        entry["role"] == "processor"
                        for entry in checkpoint_context["files"]
                    ),
                    "class": "offline",
                }
            )
            report_identities[report.resolve()] = {
                "benchmark_config": {
                    "family": "qwen3vl",
                    "size": self.model.size,
                    "base_repo": benchmark._QWEN_BASE_PINS[self.model.size][0],
                    "base_revision": benchmark._QWEN_BASE_PINS[self.model.size][1],
                    "checkpoint": (
                        {"kind": "pinned_hf_snapshot"}
                        if self.model._checkpoint_dir is None
                        else json.loads(json.dumps(checkpoint_context))
                    ),
                    "processor": processor,
                    "seed": self.seed,
                    "device": str(self.model.device),
                    "class_names": list(verified.class_names),
                    "evaluation": {
                        "imgsz": [self.config.imgsz, self.config.imgsz],
                        "faster_coco_eval": self.config.faster_coco_eval,
                        "backend": "pycocotools offline",
                        "label_to_category_id": {
                            str(index): int(category["id"])
                            for index, category in enumerate(
                                verified.expected_categories
                            )
                        },
                    },
                    "confidence_evaluation": {
                        "default_conf": self.kwargs["default_conf"],
                        "iou_threshold": self.kwargs["confidence_iou"],
                    },
                    "benchmark_run": self.kwargs["benchmark_context"],
                },
                "metrics": normalized,
            }
            return metrics

    def fake_compare(first, second, **kwargs):
        events.append(("self_compare", Path(first), Path(second), kwargs))
        return SimpleNamespace(reproducible=True)

    def fake_report_identity(path, **kwargs):
        report = Path(path).resolve()
        events.append(("report_identity", report, kwargs))
        identity = report_identities.get(report)
        if identity is None:
            unique_identities = {
                id(value): value for value in report_identities.values()
            }
            if len(unique_identities) != 1:
                raise AssertionError(f"unexpected fake report path: {report}")
            identity = next(iter(unique_identities.values()))
            report_identities[report] = identity
        return (
            hashlib.sha256(report.read_bytes()).hexdigest(),
            identity["benchmark_config"],
            identity["metrics"],
        )

    monkeypatch.setattr(benchmark, "configure_determinism", fake_determinism)
    monkeypatch.setattr(
        benchmark,
        "_configure_offline_environment",
        lambda: (
            events.append(("offline",))
            or {
                "hf_hub_offline": "1",
                "transformers_offline": "1",
                "hf_hub_disable_telemetry": "1",
                "do_not_track": "1",
                "hub_runtime_offline": True,
                "transformers_runtime_offline": True,
                "hub_runtime_telemetry_disabled": True,
            }
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_required_package_versions",
        lambda: (
            events.append(("dependencies",))
            or {
                "transformers": "offline",
                "huggingface_hub": "offline",
                "tokenizers": "offline",
                "safetensors": "offline",
                "pycocotools": "offline",
            }
        ),
    )

    def fake_verify(manifest, annotations, images_dir, review_attestation, **kwargs):
        events.append(
            (
                "verify",
                manifest,
                annotations,
                images_dir,
                review_attestation,
                kwargs,
            )
        )
        return _verified_inputs_for_role(verified, kwargs["required_role"])

    monkeypatch.setattr(benchmark, "verify_benchmark_run_inputs", fake_verify)
    monkeypatch.setattr(
        benchmark,
        "_require_pycocotools",
        lambda: events.append(("pycocotools",)),
    )
    monkeypatch.setattr(
        benchmark,
        "_resolve_and_probe_device",
        lambda requested: (
            events.append(("device", requested))
            or (
                "cpu" if requested == "auto" else requested,
                {
                    "requested_device": requested,
                    "resolved_device": "cpu" if requested == "auto" else requested,
                    "tiny_tensor_probe": "ok",
                },
            )
        ),
    )

    def fake_load_data_config(data, *, autodownload, allow_scripts):
        payload = json.loads(Path(data).read_text(encoding="utf-8"))
        events.append(("data_config", Path(data), autodownload, allow_scripts, payload))
        return {
            **payload,
            "root": payload["path"],
            "val_annotation_file": payload["annotations"]["val"],
        }

    monkeypatch.setattr(benchmark, "load_data_config", fake_load_data_config)
    monkeypatch.setattr(
        benchmark,
        "_preflight_native_dataset",
        lambda actual: (
            events.append(("native_dataset", actual))
            or {
                "image_count": actual.partition_stop - actual.partition_start,
                "batch_size": 1,
                "num_workers": 0,
                "faster_coco_eval": False,
            }
        ),
    )

    def fake_snapshot(
        size="2b", *, expected_repo=None, expected_revision=None, root=None
    ):
        if root is None:
            events.append(("snapshot", size, expected_repo, expected_revision))
            root = verified.images_dir.parent / "weights" / f"LibreQwen3VL{size}"
        else:
            root = Path(root)
        weights = dict(snapshot_identity)
        weights.update(
            source=benchmark._QWEN_BASE_PINS[size][0],
            revision=benchmark._QWEN_BASE_PINS[size][1],
        )
        processor = dict(processor_identity)
        processor.update(
            source=benchmark._QWEN_BASE_PINS[size][0],
            revision=benchmark._QWEN_BASE_PINS[size][1],
        )
        return root, weights, processor

    monkeypatch.setattr(benchmark, "_snapshot_evidence", fake_snapshot)

    @contextmanager
    def fake_isolated_base(
        source_root,
        model_size,
        snapshot_identity,
        processor_identity,
        *,
        enabled,
    ):
        del source_root, model_size, snapshot_identity, processor_identity
        if not enabled:
            yield None
            return
        isolated = verified.images_dir.parent / ".fake-isolated-base"
        isolated.mkdir()
        try:
            yield isolated.resolve()
        finally:
            isolated.rmdir()

    monkeypatch.setattr(benchmark, "_isolated_base_snapshot", fake_isolated_base)
    monkeypatch.setattr(benchmark, "LibreQwen3VL", FakeModel)
    monkeypatch.setattr(benchmark, "VLMConfidenceValidator", FakeValidator)
    monkeypatch.setattr(benchmark, "compare_confidence_reports", fake_compare)
    monkeypatch.setattr(
        benchmark, "read_confidence_report_identity", fake_report_identity
    )
    monkeypatch.setattr(
        benchmark,
        "_git_context",
        lambda: {"commit": "a" * 40, "dirty": False},
    )

    def fake_runtime_context(
        *, requested_device, resolved_device, package_versions=None
    ):
        return {
            "python": "3.11.0",
            "implementation": "CPython",
            "platform": "offline",
            "torch": "offline",
            "numpy": "offline",
            "pillow": "offline",
            "opencv": "offline",
            "packages": dict(package_versions or {}),
            "cuda_runtime": None,
            "cudnn": None,
            "nvidia_driver": None,
            "cuda_available": False,
            "attention_backends": {"model": "offline"},
            "requested_device": requested_device,
            "resolved_device": resolved_device,
        }

    monkeypatch.setattr(benchmark, "_runtime_context", fake_runtime_context)
    monkeypatch.setattr(
        benchmark, "_attention_backends", lambda model: {"model": "offline"}
    )
    return report_identities


def _status(capsys):
    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert "NaN" not in output
    assert "Infinity" not in output
    return json.loads(output)


def test_preflight_runs_shared_checks_without_model_network_or_output(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    output = tmp_path / "preflight-output"

    def forbidden(*args, **kwargs):
        pytest.fail("preflight must not download or open a network connection")

    for name in benchmark._OFFLINE_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)

    result = _preflight(verified, output, seed=17, device="cuda:0")

    assert result.output_dir == output.resolve()
    assert result.snapshot_identity["sha256"] == "7" * 64
    assert result.processor_content_identity["sha256"] == "9" * 64
    assert result.runtime_context["resolved_device"] == "cuda:0"
    assert result.runtime_context["device_probe"]["tiny_tensor_probe"] == "ok"
    assert result.dataset_context["native"]["image_count"] == 500
    assert [event[0] for event in events] == [
        "determinism",
        "offline",
        "dependencies",
        "pycocotools",
        "device",
        "verify",
        "data_config",
        "native_dataset",
        "snapshot",
        "verify",
    ]
    assert not output.exists()
    assert not (tmp_path / ".preflight-output.lock").exists()
    assert not list(tmp_path.glob(".preflight-output.tmp-*"))
    assert not list(tmp_path.glob(".libreyolo-vlm-output-probe-*"))


def test_checkpoint_preflight_derives_4b_base_and_persists_path_free_identity(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    identity = _strict_checkpoint_identity(tmp_path, size="4b")
    inspected = []
    monkeypatch.setattr(
        benchmark,
        "_inspect_checkpoint_identity",
        _matching_checkpoint_inspector(identity, inspected),
    )

    result = _preflight(
        verified,
        tmp_path / "preflight-checkpoint",
        checkpoint_dir=identity.root,
    )

    assert result.model_size == "4b"
    assert result.checkpoint_root == identity.root
    assert result.checkpoint_identity["aggregate_sha256"] == identity.aggregate_sha256
    assert result.checkpoint_identity["adapter_weights_sha256"] == "b" * 64
    observed_partition = result.dataset_context["identity"]["partition"]
    for key, value in benchmark._PARTITION_REQUIREMENTS[
        benchmark._CHECKPOINT_PARTITION_ROLE
    ].items():
        assert observed_partition[key] == value
    assert observed_partition["annotation_size_bytes"] > 0
    assert observed_partition["annotation_sha256"] == "4" * 64
    assert result.dataset_context["native"]["image_count"] == 100
    assert {event[5]["required_role"] for event in events if event[0] == "verify"} == {
        benchmark._CHECKPOINT_PARTITION_ROLE
    }
    assert "root" not in result.checkpoint_identity
    assert str(identity.root) not in benchmark._json_text(result.checkpoint_identity)
    assert inspected[0] == identity.root
    assert inspected.count(identity.root) == 3
    isolated_roots = {path for path in inspected if path != identity.root}
    assert len(isolated_roots) == 1
    assert not next(iter(isolated_roots)).exists()
    snapshot_event = next(event for event in events if event[0] == "snapshot")
    assert snapshot_event[1:] == (
        "4b",
        benchmark._QWEN_4B_REPO,
        benchmark._QWEN_4B_REVISION,
    )
    assert "model" not in [event[0] for event in events]


def test_checkpoint_preflight_rejects_mismatched_isolated_copy_and_cleans_it(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    identity = _strict_checkpoint_identity(tmp_path, size="4b", token="a")
    changed = _strict_checkpoint_identity(tmp_path, size="4b", token="f")
    isolated_roots = []

    def inspect(path):
        root = Path(path).resolve()
        if root == identity.root:
            return _checkpoint_identity_at(identity, root)
        isolated_roots.append(root)
        return _checkpoint_identity_at(changed, root)

    monkeypatch.setattr(benchmark, "_inspect_checkpoint_identity", inspect)
    output = tmp_path / "mismatched-isolated-copy"

    with pytest.raises(
        benchmark.BenchmarkInputError,
        match="isolated checkpoint identity does not match",
    ):
        _preflight(verified, output, checkpoint_dir=identity.root)

    assert len(isolated_roots) == 1
    assert not isolated_roots[0].exists()
    assert "model" not in [event[0] for event in events]
    assert not output.exists()


def test_preflight_cli_emits_ready_evidence_without_persisting_it(
    tmp_path, monkeypatch, capsys
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    output = tmp_path / "preflight-output"

    code = benchmark.main(
        _cli_preflight_args(verified, output) + ["--seed", "19", "--device", "cuda:0"]
    )

    assert code == benchmark.EXIT_OK
    status = _status(capsys)
    assert status["schema"] == "libreyolo.vlm-confidence-benchmark-status.v1"
    assert status["status"] == "ready"
    assert status["mode"] == "preflight"
    assert status["code"] == benchmark.EXIT_OK
    evidence = status["preflight"]
    assert evidence["schema"] == "libreyolo.vlm-confidence-benchmark-preflight.v2"
    assert evidence["request"] == {
        "model_family": "qwen3vl",
        "model_size": "2b",
        "checkpoint_dir": None,
        "seed": 19,
        "device": "cuda:0",
        "resolved_device": "cuda:0",
        "output_root": str(output.resolve()),
    }
    assert evidence["snapshot"]["weights"]["sha256"] == "7" * 64
    assert evidence["snapshot"]["processor_content"]["sha256"] == "9" * 64
    assert evidence["checkpoint"] is None
    assert evidence["dataset"]["identity"]["review"]["status"] == "approved"
    assert evidence["offline"] == {
        "do_not_track": "1",
        "hf_hub_disable_telemetry": "1",
        "hf_hub_offline": "1",
        "hub_runtime_offline": True,
        "hub_runtime_telemetry_disabled": True,
        "transformers_offline": "1",
        "transformers_runtime_offline": True,
    }
    assert not output.exists()
    assert not (tmp_path / ".preflight-output.lock").exists()


def test_preflight_cli_exposes_checkpoint_path_only_in_request(
    tmp_path, monkeypatch, capsys
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    identity = _strict_checkpoint_identity(tmp_path, size="4b")
    monkeypatch.setattr(
        benchmark,
        "_inspect_checkpoint_identity",
        _matching_checkpoint_inspector(identity),
    )

    code = benchmark.main(
        _cli_preflight_args(verified, tmp_path / "checkpoint-preflight")
        + ["--checkpoint-dir", str(identity.root)]
    )

    assert code == benchmark.EXIT_OK
    evidence = _status(capsys)["preflight"]
    assert evidence["request"]["model_size"] == "4b"
    assert evidence["request"]["checkpoint_dir"] == str(identity.root)
    assert evidence["checkpoint"]["aggregate_sha256"] == identity.aggregate_sha256
    assert evidence["dataset"]["identity"]["partition"]["name"] == "holdout100"
    assert evidence["dataset"]["identity"]["partition"]["role"] == (
        benchmark._CHECKPOINT_PARTITION_ROLE
    )
    assert str(identity.root) not in benchmark._json_text(evidence["checkpoint"])


def test_run_benchmark_stages_complete_artifacts_and_records_context(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(
        monkeypatch,
        events,
        verified,
        metrics={
            "metric/finite": 0.5,
            "metric/nan": float("nan"),
            "metric/positive_infinity": float("inf"),
            "metric/negative_infinity": float("-inf"),
        },
    )
    output = tmp_path / "run-a"

    artifacts = _run(verified, output, seed=17, device="cuda:0")

    assert artifacts.output_dir == output.resolve()
    assert artifacts.report_path.is_file()
    assert artifacts.envelope_path.is_file()
    assert sorted(path.name for path in output.iterdir()) == [
        "vlm_confidence_report.json",
        "vlm_confidence_run.json",
    ]
    assert not list(tmp_path.glob(".run-a.tmp-*"))
    assert not (tmp_path / ".run-a.lock").exists()
    assert [event[0] for event in events[:13]] == [
        "determinism",
        "offline",
        "dependencies",
        "pycocotools",
        "device",
        "verify",
        "data_config",
        "native_dataset",
        "snapshot",
        "verify",
        "model",
        "validator",
        "run",
    ]
    verify_event = events[5]
    assert verify_event[1:5] == (
        verified.manifest_path,
        verified.source_annotations,
        verified.images_dir,
        verified.review_attestation_path,
    )
    assert verify_event[5] == {"required_role": "zero_shot_confidence_promotion"}
    data_event = events[6]
    assert data_event[2:4] == (False, False)
    assert data_event[4] == {
        "path": str(verified.images_dir),
        "val": str(verified.images_dir),
        "annotations": {"val": str(verified.annotation_path)},
        "nc": 80,
        "names": list(verified.class_names),
    }
    config = events[11][2]
    assert Path(config.data).name == "verified_dataset.yaml"
    assert not Path(config.data).exists()
    assert config.batch_size == 1
    assert config.num_workers == 0
    assert config.allow_download_scripts is False
    assert config.imgsz == 1024
    assert config.save_json is True
    assert config.save_plots is True
    assert config.faster_coco_eval is False
    assert events[11][3]["default_conf"] == 0.25
    assert events[11][3]["confidence_iou"] == 0.5
    assert events[11][3]["verified_dataset"] is verified
    assert events[11][3]["expected_snapshot_identity"]["sha256"] == "7" * 64
    assert events[11][3]["expected_processor_content_identity"]["sha256"] == "9" * 64
    assert Path(events[11][3]["expected_snapshot_root"]).name == (".fake-isolated-base")
    assert events[11][3]["benchmark_context"]["git"] == {
        "commit": "a" * 40,
        "dirty": False,
    }

    raw_envelope = artifacts.envelope_path.read_text(encoding="utf-8")
    assert "NaN" not in raw_envelope
    assert "Infinity" not in raw_envelope
    envelope = json.loads(raw_envelope)
    assert envelope["schema"] == "libreyolo.vlm-confidence-benchmark-run.v3"
    assert benchmark._RUN_IDENTIFIER.fullmatch(envelope["run_id"])
    assert benchmark._RUN_IDENTIFIER.fullmatch(envelope["process_id"])
    assert envelope["request"] == {
        "manifest": str(verified.manifest_path),
        "annotations": str(verified.source_annotations),
        "images_dir": str(verified.images_dir),
        "review_attestation": str(verified.review_attestation_path),
        "seed": 17,
        "model_family": "qwen3vl",
        "model_size": "2b",
        "checkpoint_dir": None,
        "device": "cuda:0",
        "imgsz": 1024,
        "default_conf": 0.25,
        "confidence_iou": 0.5,
    }
    assert envelope["execution_context"]["git"] == {
        "commit": "a" * 40,
        "dirty": False,
    }
    assert (
        envelope["execution_context"]["schema"]
        == "libreyolo.vlm-confidence-benchmark-context.v3"
    )
    assert envelope["execution_context"]["checkpoint"] is None
    dataset_context = envelope["execution_context"]["dataset"]
    assert dataset_context["manifest"] == {
        "schema": "libreyolo.vlm-benchmark-dataset.v1",
        "sha256": verified.manifest_sha256,
    }
    assert dataset_context["source"] == {
        "canonical_annotation_sha256": verified.source_canonical_sha256,
        "file_sha256": verified.source_file_sha256,
        "file_size_bytes": verified.source_file_size_bytes,
        "selected_image_identity_sha256": verified.selected_image_identity_sha256,
    }
    assert dataset_context["partition"] == {
        "name": "promotion500",
        "role": "zero_shot_confidence_promotion",
        "start": 0,
        "stop": 500,
        "image_count": 500,
        "annotation_artifact": ("annotations/instances_val2017_promotion500.json"),
        "annotation_size_bytes": verified.annotation_size_bytes,
        "annotation_sha256": verified.annotation_sha256,
    }
    assert dataset_context["classes"] == {
        "count": 80,
        "names": list(verified.class_names),
        "category_ids": [
            int(category["id"]) for category in verified.expected_categories
        ],
    }
    assert dataset_context["review"]["sha256"] == verified.review_attestation_sha256
    assert dataset_context["review"]["checks"] == _REVIEW_CHECKS
    assert str(tmp_path.resolve()) not in benchmark._json_text(dataset_context)
    assert envelope["execution_context"]["determinism"]["seed"] == 17
    assert (
        envelope["report"]["sha256"]
        == hashlib.sha256(artifacts.report_path.read_bytes()).hexdigest()
    )
    assert envelope["metrics"] == {
        "metric/finite": 0.5,
        "metric/nan": None,
        "metric/negative_infinity": None,
        "metric/positive_infinity": None,
    }
    assert envelope["nonfinite_metrics"] == [
        "metric/nan",
        "metric/negative_infinity",
        "metric/positive_infinity",
    ]


def test_runner_envelope_binds_companion_report_context(tmp_path, monkeypatch):
    events = []
    verified = _verified_inputs(tmp_path)
    report_identities = _install_run_fakes(monkeypatch, events, verified)
    artifacts = _run(verified, tmp_path / "run-a")
    envelope = json.loads(artifacts.envelope_path.read_text(encoding="utf-8"))

    validated = benchmark.read_benchmark_run_identity(
        artifacts.report_path, label="run"
    )

    assert validated.run_id == envelope["run_id"]
    assert validated.process_id == envelope["process_id"]
    assert validated.report_sha256 == envelope["report"]["sha256"]
    assert (
        validated.envelope_sha256
        == hashlib.sha256(artifacts.envelope_path.read_bytes()).hexdigest()
    )
    assert validated.execution_context == envelope["execution_context"]
    assert (
        validated.benchmark_config
        == next(iter(report_identities.values()))["benchmark_config"]
    )
    assert validated.metrics == envelope["metrics"]
    assert validated.nonfinite_metrics == tuple(envelope["nonfinite_metrics"])

    tampered = json.loads(json.dumps(envelope))
    tampered["execution_context"]["determinism"]["cudnn_benchmark"] = True
    benchmark._write_json_atomic(artifacts.envelope_path, tampered)
    with pytest.raises(VLMConfidenceReportError, match="cudnn_benchmark"):
        benchmark.read_benchmark_run_identity(artifacts.report_path, label="run")

    tampered = json.loads(json.dumps(envelope))
    tampered["report"]["sha256"] = "0" * 64
    benchmark._write_json_atomic(artifacts.envelope_path, tampered)
    with pytest.raises(VLMConfidenceReportError, match="companion report bytes"):
        benchmark.read_benchmark_run_identity(artifacts.report_path, label="run")

    for mutate, expected_error in (
        (
            lambda item: item["request"].__setitem__("imgsz", 1024.0),
            "request.imgsz",
        ),
        (
            lambda item: item["execution_context"]["determinism"].__setitem__(
                "seed", 0.0
            ),
            "determinism.seed",
        ),
        (
            lambda item: item["request"].__setitem__("default_conf", "0.25"),
            "default_conf",
        ),
        (
            lambda item: item["request"].__setitem__("manifest", "manifest.json"),
            "absolute operational path",
        ),
        (
            lambda item: item["execution_context"]["dataset"]["manifest"].__setitem__(
                "sha256", "0" * 64
            ),
            "review.manifest_sha256",
        ),
        (
            lambda item: item["execution_context"]["dataset"]["source"].__setitem__(
                "file_sha256", "invalid"
            ),
            "source.file_sha256",
        ),
        (
            lambda item: item["execution_context"]["dataset"]["source"].__setitem__(
                "file_size_bytes", 0
            ),
            "source.file_size_bytes",
        ),
        (
            lambda item: item["execution_context"]["dataset"]["partition"].__setitem__(
                "role", "fine_tune_training"
            ),
            "partition.role",
        ),
        (
            lambda item: item["execution_context"]["dataset"]["partition"].__setitem__(
                "image_count", 499
            ),
            "partition.image_count",
        ),
        (
            lambda item: item["execution_context"]["dataset"]["classes"].__setitem__(
                "category_ids", list(reversed(range(1, 81)))
            ),
            "classes.category_ids",
        ),
        (
            lambda item: item["execution_context"]["dataset"]["review"].__setitem__(
                "status", "pending"
            ),
            "review.status",
        ),
        (
            lambda item: item["execution_context"]["dataset"]["review"][
                "checks"
            ].__setitem__("privacy_and_pii", False),
            "review.checks",
        ),
        (
            lambda item: item["execution_context"]["dataset"].__setitem__(
                "absolute_path", "C:/forbidden"
            ),
            "unsupported absolute_path",
        ),
    ):
        tampered = json.loads(json.dumps(envelope))
        mutate(tampered)
        benchmark._write_json_atomic(artifacts.envelope_path, tampered)
        with pytest.raises(VLMConfidenceReportError, match=expected_error):
            benchmark.read_benchmark_run_identity(artifacts.report_path, label="run")

    identity = report_identities[artifacts.report_path.resolve()]
    original_config = json.loads(json.dumps(identity["benchmark_config"]))
    for mutate, expected_error in (
        (
            lambda item: item["evaluation"].__setitem__("imgsz", [640, 640]),
            "request.imgsz",
        ),
        (
            lambda item: item["evaluation"].__setitem__("imgsz", [1024.0, 1024.0]),
            "request.imgsz",
        ),
        (
            lambda item: item["confidence_evaluation"].__setitem__("default_conf", 0.2),
            "confidence thresholds",
        ),
        (
            lambda item: item["confidence_evaluation"].__setitem__(
                "iou_threshold", 0.6
            ),
            "confidence thresholds",
        ),
        (
            lambda item: item["evaluation"].__setitem__(
                "backend", "faster-coco-eval 1.7.2"
            ),
            "evaluation.backend",
        ),
        (
            lambda item: item["evaluation"].pop("label_to_category_id"),
            "classes.category_ids",
        ),
        (
            lambda item: item["evaluation"]["label_to_category_id"].__setitem__(
                "0", 80
            ),
            "classes.category_ids",
        ),
        (
            lambda item: item["evaluation"].__setitem__(
                "label_to_category_id",
                {
                    str(index): category_id
                    for index, category_id in enumerate(reversed(range(1, 81)))
                },
            ),
            "classes.category_ids",
        ),
        (
            lambda item: item.__setitem__(
                "class_names", [*item["class_names"][:-1], "wrong-class"]
            ),
            "classes.names",
        ),
        (
            lambda item: item["benchmark_run"]["dataset"]["review"].__setitem__(
                "sha256", "0" * 64
            ),
            "execution_context",
        ),
    ):
        benchmark._write_json_atomic(artifacts.envelope_path, envelope)
        identity["benchmark_config"] = json.loads(json.dumps(original_config))
        mutate(identity["benchmark_config"])
        with pytest.raises(VLMConfidenceReportError, match=expected_error):
            benchmark.read_benchmark_run_identity(artifacts.report_path, label="run")


def test_public_run_identity_requires_sibling_envelope(tmp_path):
    report = tmp_path / "vlm_confidence_report.json"
    report.write_text("{}\n", encoding="utf-8")

    with pytest.raises(VLMConfidenceReportError, match="missing companion"):
        benchmark.read_benchmark_run_identity(report)


def test_compare_requires_runner_companion_envelopes(tmp_path):
    first = tmp_path / "one" / "vlm_confidence_report.json"
    second = tmp_path / "two" / "vlm_confidence_report.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("{}\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")

    with pytest.raises(VLMConfidenceReportError, match="missing companion"):
        benchmark.compare_benchmarks(first, second)


def test_runner_envelope_rejects_duplicate_json_keys(tmp_path):
    report = tmp_path / "vlm_confidence_report.json"
    report.write_text("{}\n", encoding="utf-8")
    (tmp_path / "vlm_confidence_run.json").write_text(
        '{"schema":"first","schema":"second"}\n', encoding="utf-8"
    )

    with pytest.raises(VLMConfidenceReportError, match="duplicate JSON object key"):
        benchmark._load_runner_envelope(report, "run")


def test_runner_envelope_wraps_oversized_json_integer(tmp_path):
    report = tmp_path / "vlm_confidence_report.json"
    report.write_text("{}\n", encoding="utf-8")
    (tmp_path / "vlm_confidence_run.json").write_text(
        '{"value":' + "1" * 5000 + "}\n", encoding="utf-8"
    )

    with pytest.raises(
        VLMConfidenceReportError, match="invalid JSON value|canonical JSON"
    ):
        benchmark._load_runner_envelope(report, "run")


def test_run_benchmark_cleans_staging_artifacts_after_failure(tmp_path, monkeypatch):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(
        monkeypatch,
        events,
        verified,
        failure=RuntimeError("offline failure"),
    )
    output = tmp_path / "failed-run"

    with pytest.raises(RuntimeError, match="offline failure"):
        _run(verified, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".failed-run.tmp-*"))
    assert not (tmp_path / ".failed-run.lock").exists()


def test_run_benchmark_rejects_code_drift_during_generation(tmp_path, monkeypatch):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    contexts = iter(
        [
            {"commit": "a" * 40, "dirty": False},
            {"commit": "a" * 40, "dirty": False},
            {"commit": "b" * 40, "dirty": False},
        ]
    )
    monkeypatch.setattr(benchmark, "_git_context", lambda: next(contexts))
    output = tmp_path / "drifted-run"

    with pytest.raises(RuntimeError, match="changed during execution"):
        _run(verified, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".drifted-run.tmp-*"))
    assert not (tmp_path / ".drifted-run.lock").exists()


def test_run_rejects_snapshot_drift_immediately_after_model_construction(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    stable = benchmark._require_snapshot_stable

    def drifting_snapshot(*args, phase, **kwargs):
        if phase == "during model construction":
            raise RuntimeError(
                "VLM base snapshot identity changed during model construction"
            )
        return stable(*args, phase=phase, **kwargs)

    monkeypatch.setattr(benchmark, "_require_snapshot_stable", drifting_snapshot)
    output = tmp_path / "snapshot-drift"

    with pytest.raises(RuntimeError, match="changed during model construction"):
        _run(verified, output)

    assert "validator" not in [event[0] for event in events]
    assert not output.exists()
    assert not (tmp_path / ".snapshot-drift.lock").exists()


def test_run_rejects_checkpoint_drift_immediately_after_model_construction(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    expected = _strict_checkpoint_identity(tmp_path, size="4b", token="a")
    changed = _strict_checkpoint_identity(tmp_path, size="4b", token="f")
    source_inspections = 0

    def inspect(path):
        nonlocal source_inspections
        root = Path(path).resolve()
        if root == expected.root:
            source_inspections += 1
        identity = (
            changed if source_inspections == 5 and root == expected.root else expected
        )
        return _checkpoint_identity_at(identity, root)

    monkeypatch.setattr(benchmark, "_inspect_checkpoint_identity", inspect)
    output = tmp_path / "checkpoint-construction-drift"

    with pytest.raises(RuntimeError, match="during model construction"):
        _run(verified, output, checkpoint_dir=expected.root)

    assert "model" in [event[0] for event in events]
    assert "validator" not in [event[0] for event in events]
    assert not output.exists()
    assert not (tmp_path / ".checkpoint-construction-drift.lock").exists()


def test_run_rejects_checkpoint_drift_after_generation_without_publishing(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    expected = _strict_checkpoint_identity(tmp_path, size="4b", token="a")
    changed = _strict_checkpoint_identity(tmp_path, size="4b", token="f")
    source_inspections = 0

    def inspect(path):
        nonlocal source_inspections
        root = Path(path).resolve()
        if root == expected.root:
            source_inspections += 1
        identity = (
            changed if source_inspections == 6 and root == expected.root else expected
        )
        return _checkpoint_identity_at(identity, root)

    monkeypatch.setattr(benchmark, "_inspect_checkpoint_identity", inspect)
    output = tmp_path / "checkpoint-generation-drift"

    with pytest.raises(RuntimeError, match="during generation"):
        _run(verified, output, checkpoint_dir=expected.root)

    assert "run" in [event[0] for event in events]
    assert not output.exists()
    assert not (tmp_path / ".checkpoint-generation-drift.lock").exists()


def test_run_rejects_model_device_drift_before_second_snapshot(tmp_path, monkeypatch):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    monkeypatch.setattr(benchmark, "_resolved_model_device", lambda model: "cpu:1")
    output = tmp_path / "device-drift"

    with pytest.raises(RuntimeError, match="different device"):
        _run(verified, output)

    assert [event[0] for event in events].count("snapshot") == 1
    assert "validator" not in [event[0] for event in events]
    assert not output.exists()


def test_preflight_rejects_dataset_evidence_that_changes_during_checks(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    changed = SimpleNamespace(**vars(verified))
    changed.manifest_sha256 = "0" * 64
    values = iter((verified, changed))

    def changing_verify(*args, **kwargs):
        events.append(("verify",))
        return next(values)

    monkeypatch.setattr(benchmark, "verify_benchmark_run_inputs", changing_verify)
    output = tmp_path / "unstable-evidence"

    with pytest.raises(benchmark.BenchmarkInputError, match="changed during preflight"):
        _preflight(verified, output)

    assert "model" not in [event[0] for event in events]
    assert not output.exists()


def test_preflight_rejects_git_drift_after_local_evidence_checks(tmp_path, monkeypatch):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    contexts = iter(
        [
            {"commit": "a" * 40, "dirty": False},
            {"commit": "b" * 40, "dirty": False},
        ]
    )
    monkeypatch.setattr(benchmark, "_git_context", lambda: next(contexts))
    output = tmp_path / "git-drift"

    with pytest.raises(benchmark.BenchmarkInputError, match="during preflight"):
        _preflight(verified, output)

    assert "model" not in [event[0] for event in events]
    assert not output.exists()


def test_run_benchmark_refuses_overwrite_before_git_or_model(tmp_path, monkeypatch):
    verified = _verified_inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        benchmark,
        "_git_context",
        lambda: pytest.fail("git must not run for an occupied output"),
    )
    monkeypatch.setattr(
        benchmark,
        "LibreQwen3VL",
        lambda **kwargs: pytest.fail("model must not be constructed"),
    )

    with pytest.raises(benchmark.BenchmarkOutputExistsError):
        _run(verified, output)

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_checkpoint_rejection_stops_before_git_or_model_construction(
    tmp_path, monkeypatch
):
    verified = _verified_inputs(tmp_path)
    checkpoint = tmp_path / "invalid-checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(
        benchmark,
        "_inspect_checkpoint_identity",
        lambda path: (_ for _ in ()).throw(
            benchmark.BenchmarkInputError("invalid local VLM checkpoint: bad contract")
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_git_context",
        lambda: pytest.fail("git inspection must not start"),
    )
    monkeypatch.setattr(
        benchmark,
        "LibreQwen3VL",
        lambda **kwargs: pytest.fail("model construction must not start"),
    )
    output = tmp_path / "rejected-checkpoint"

    with pytest.raises(benchmark.BenchmarkInputError, match="bad contract"):
        _run(verified, output, checkpoint_dir=checkpoint)

    assert not output.exists()


def test_run_benchmark_refuses_broken_output_symlink(tmp_path, monkeypatch):
    verified = _verified_inputs(tmp_path)
    target = tmp_path / "missing-target"
    output = tmp_path / "broken-output"
    try:
        output.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    monkeypatch.setattr(
        benchmark,
        "_git_context",
        lambda: pytest.fail("git must not run for an occupied output"),
    )

    with pytest.raises(benchmark.BenchmarkOutputExistsError):
        _run(verified, output)

    assert output.is_symlink()
    assert not target.exists()


def test_run_benchmark_refuses_output_inside_git_worktree(tmp_path, monkeypatch):
    verified = _verified_inputs(tmp_path)
    monkeypatch.setattr(benchmark, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        benchmark,
        "_git_context",
        lambda: pytest.fail("git inspection must not start for in-tree output"),
    )
    monkeypatch.setattr(
        benchmark,
        "LibreQwen3VL",
        lambda **kwargs: pytest.fail("model must not be constructed"),
    )

    with pytest.raises(benchmark.BenchmarkInputError, match="outside the git"):
        _run(verified, tmp_path / "runs" / "run-a")


def test_preflight_rejects_regular_file_output_parent_without_side_effects(
    tmp_path, monkeypatch
):
    verified = _verified_inputs(tmp_path)
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("preserve\n", encoding="utf-8")
    output = parent_file / "run-a"
    monkeypatch.setattr(
        benchmark,
        "_git_context",
        lambda: pytest.fail("git must not run for an invalid output parent"),
    )

    with pytest.raises(benchmark.BenchmarkInputError, match="output parent"):
        _preflight(verified, output)

    assert parent_file.read_text(encoding="utf-8") == "preserve\n"
    assert not output.exists()
    assert not list(tmp_path.glob(".libreyolo-vlm-output-probe-*"))


def test_run_uses_validated_checkpoint_identity_for_construction_and_report(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    report_identities = _install_run_fakes(monkeypatch, events, verified)
    identity = _strict_checkpoint_identity(tmp_path, size="4b")
    monkeypatch.setattr(
        benchmark,
        "_inspect_checkpoint_identity",
        _matching_checkpoint_inspector(identity),
    )

    artifacts = _run(
        verified,
        tmp_path / "checkpoint-run",
        checkpoint_dir=identity.root,
    )

    model_event = next(event for event in events if event[0] == "model")
    checkpoint_load_root = Path(model_event[3])
    assert model_event[1:3] == ("4b", "auto")
    assert checkpoint_load_root != identity.root
    assert checkpoint_load_root.is_absolute()
    assert not checkpoint_load_root.exists()
    validator_event = next(event for event in events if event[0] == "validator")
    expectations = validator_event[3]
    isolated_identity = expectations["expected_checkpoint_identity"]
    assert isolated_identity.root == checkpoint_load_root
    assert isolated_identity is not identity
    assert benchmark._checkpoint_context(isolated_identity) == (
        benchmark._checkpoint_context(identity)
    )
    assert expectations["expected_snapshot_identity"]["source"] == (
        benchmark._QWEN_4B_REPO
    )
    assert expectations["expected_processor_content_identity"]["source"] == (
        benchmark._QWEN_4B_REPO
    )
    envelope = json.loads(artifacts.envelope_path.read_text(encoding="utf-8"))
    context_identity = envelope["execution_context"]["checkpoint"]
    assert envelope["request"]["checkpoint_dir"] == str(identity.root)
    assert envelope["request"]["model_size"] == "4b"
    assert context_identity["aggregate_sha256"] == identity.aggregate_sha256
    assert envelope["execution_context"]["dataset"]["partition"]["name"] == (
        "holdout100"
    )
    assert envelope["execution_context"]["dataset"]["partition"]["role"] == (
        benchmark._CHECKPOINT_PARTITION_ROLE
    )
    assert str(identity.root) not in benchmark._json_text(context_identity)
    report_identity = next(iter(report_identities.values()))
    assert (
        report_identity["benchmark_config"]["benchmark_run"]["checkpoint"]
        == context_identity
    )


def test_public_run_identity_rejects_promotion_context_for_checkpoint(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    report_identities = _install_run_fakes(monkeypatch, events, verified)
    identity = _strict_checkpoint_identity(tmp_path, size="4b")
    monkeypatch.setattr(
        benchmark,
        "_inspect_checkpoint_identity",
        _matching_checkpoint_inspector(identity),
    )
    artifacts = _run(
        verified,
        tmp_path / "wrong-checkpoint-partition",
        checkpoint_dir=identity.root,
    )
    envelope = json.loads(artifacts.envelope_path.read_text(encoding="utf-8"))
    fake_dataset = envelope["execution_context"]["dataset"]
    fake_dataset["partition"].update(
        benchmark._PARTITION_REQUIREMENTS[benchmark._BASE_PARTITION_ROLE]
    )
    fake_dataset["review"]["partition_role"] = benchmark._BASE_PARTITION_ROLE
    report_identity = next(iter(report_identities.values()))
    report_identity["benchmark_config"]["benchmark_run"]["dataset"] = json.loads(
        json.dumps(fake_dataset)
    )
    benchmark._write_json_atomic(artifacts.envelope_path, envelope)

    with pytest.raises(VLMConfidenceReportError, match="partition.name"):
        benchmark.read_benchmark_run_identity(artifacts.report_path)


def test_public_run_identity_rejects_forged_native_checkpoint_mapping(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    report_identities = _install_run_fakes(monkeypatch, events, verified)
    identity = _strict_checkpoint_identity(tmp_path, size="4b")
    monkeypatch.setattr(
        benchmark,
        "_inspect_checkpoint_identity",
        _matching_checkpoint_inspector(identity),
    )
    artifacts = _run(
        verified,
        tmp_path / "forged-native-checkpoint",
        checkpoint_dir=identity.root,
    )
    report_identity = next(iter(report_identities.values()))
    forged = json.loads(json.dumps(report_identity["benchmark_config"]["checkpoint"]))
    forged["aggregate_sha256"] = "0" * 64
    report_identity["benchmark_config"]["checkpoint"] = forged

    with pytest.raises(
        VLMConfidenceReportError,
        match="does not match benchmark_config.checkpoint",
    ):
        benchmark.read_benchmark_run_identity(artifacts.report_path)


def test_run_isolates_model_load_from_transient_requested_checkpoint_swap(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    identity = _strict_checkpoint_identity(tmp_path, size="4b", token="a")
    changed = _strict_checkpoint_identity(tmp_path, size="4b", token="f")
    source_file = identity.root / "adapter_config.json"
    source_payload = b'{"checkpoint":"A"}\n'
    source_file.write_bytes(source_payload)
    source_state = "a"
    isolated_roots = []

    def inspect(path):
        root = Path(path).resolve()
        selected = (
            changed if root == identity.root and source_state == "b" else identity
        )
        return _checkpoint_identity_at(selected, root)

    def transient_swap(checkpoint_dir):
        nonlocal source_state
        isolated_root = Path(checkpoint_dir).resolve()
        isolated_roots.append(isolated_root)
        assert isolated_root != identity.root
        source_state = "b"
        source_file.write_bytes(b'{"checkpoint":"B"}\n')
        try:
            assert (isolated_root / source_file.name).read_bytes() == source_payload
        finally:
            source_file.write_bytes(source_payload)
            source_state = "a"

    report_identities = _install_run_fakes(
        monkeypatch,
        events,
        verified,
        model_hook=transient_swap,
    )
    monkeypatch.setattr(benchmark, "_inspect_checkpoint_identity", inspect)

    artifacts = _run(
        verified,
        tmp_path / "transient-swap-run",
        checkpoint_dir=identity.root,
    )

    assert len(isolated_roots) == 1
    assert not isolated_roots[0].exists()
    assert source_file.read_bytes() == source_payload
    envelope = json.loads(artifacts.envelope_path.read_text(encoding="utf-8"))
    assert envelope["request"]["checkpoint_dir"] == str(identity.root)
    assert (
        envelope["execution_context"]["checkpoint"]["aggregate_sha256"]
        == identity.aggregate_sha256
    )
    report_identity = next(iter(report_identities.values()))
    assert (
        report_identity["benchmark_config"]["benchmark_run"]["checkpoint"]
        == (envelope["execution_context"]["checkpoint"])
    )


def test_isolated_checkpoint_is_cleaned_when_model_construction_fails(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    identity = _strict_checkpoint_identity(tmp_path, size="4b")
    isolated_roots = []
    cleanup_roots = _install_checkpoint_cleanup_failure(monkeypatch)

    def fail_construction(checkpoint_dir):
        isolated_roots.append(Path(checkpoint_dir).resolve())
        raise ValueError("offline model construction failed")

    _install_run_fakes(
        monkeypatch,
        events,
        verified,
        model_hook=fail_construction,
    )
    monkeypatch.setattr(
        benchmark,
        "_inspect_checkpoint_identity",
        _matching_checkpoint_inspector(identity),
    )
    output = tmp_path / "failed-isolated-run"

    with pytest.raises(ValueError, match="model construction failed"):
        _run(verified, output, checkpoint_dir=identity.root)

    assert len(isolated_roots) == 1
    assert cleanup_roots == isolated_roots
    assert not isolated_roots[0].exists()
    assert not output.exists()
    assert not (tmp_path / ".failed-isolated-run.lock").exists()


def test_isolated_checkpoint_cleanup_failure_prevents_output_publication(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    identity = _strict_checkpoint_identity(tmp_path, size="4b")
    _install_run_fakes(monkeypatch, events, verified)
    monkeypatch.setattr(
        benchmark,
        "_inspect_checkpoint_identity",
        _matching_checkpoint_inspector(identity),
    )
    isolated_roots = _install_checkpoint_cleanup_failure(monkeypatch)
    output = tmp_path / "cleanup-failed-run"

    with pytest.raises(
        benchmark.BenchmarkInputError,
        match="could not remove the isolated checkpoint copy",
    ):
        _run(verified, output, checkpoint_dir=identity.root)

    assert len(isolated_roots) == 1
    assert not isolated_roots[0].exists()
    assert not output.exists()
    assert not (tmp_path / ".cleanup-failed-run.lock").exists()


def test_isolated_base_snapshot_binds_model_load_during_transient_source_swap(
    tmp_path, monkeypatch
):
    source, snapshot, processor = _compact_base_snapshot(tmp_path, monkeypatch)
    isolated_roots = []

    class FakeQwen:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            source_file = source / "model.safetensors"
            source_file.write_bytes(b"base-B")
            load_root = Path(self._ensure_weights()).resolve()
            isolated_roots.append(load_root)
            self.loaded = (load_root / "model.safetensors").read_bytes()
            source_file.write_bytes(b"base-A")

    monkeypatch.setattr(benchmark, "LibreQwen3VL", FakeQwen)
    with benchmark._isolated_base_snapshot(
        source,
        "2b",
        snapshot,
        processor,
        enabled=True,
    ) as isolated:
        model = benchmark._construct_benchmark_model(
            model_size="2b",
            requested_device="cpu",
            snapshot_load_root=isolated,
            checkpoint_load_root=None,
        )
        assert model.loaded == b"base-A"
        assert isolated_roots == [isolated]
        assert (isolated / "model.safetensors").read_bytes() == b"base-A"

    assert (source / "model.safetensors").read_bytes() == b"base-A"
    assert not isolated_roots[0].exists()


def test_isolated_base_snapshot_mutation_is_rejected(tmp_path, monkeypatch):
    source, snapshot, processor = _compact_base_snapshot(tmp_path, monkeypatch)

    with benchmark._isolated_base_snapshot(
        source,
        "2b",
        snapshot,
        processor,
        enabled=True,
    ) as isolated:
        (isolated / "model.safetensors").write_bytes(b"base-B")
        with pytest.raises(RuntimeError, match="isolated copy"):
            benchmark._require_snapshot_stable(
                isolated,
                "2b",
                snapshot,
                processor,
                phase="during generation from the isolated copy",
                input_error=False,
            )


def test_isolated_base_cleanup_failure_is_reported_after_success(tmp_path, monkeypatch):
    source, snapshot, processor = _compact_base_snapshot(tmp_path, monkeypatch)
    isolated_roots = _install_base_cleanup_failure(monkeypatch)

    with (
        pytest.raises(
            benchmark.BenchmarkInputError,
            match="could not remove the isolated base snapshot",
        ),
        benchmark._isolated_base_snapshot(
            source,
            "2b",
            snapshot,
            processor,
            enabled=True,
        ),
    ):
        pass

    assert len(isolated_roots) == 1
    assert not isolated_roots[0].exists()


def test_isolated_base_cleanup_failure_preserves_primary_exception(
    tmp_path, monkeypatch
):
    source, snapshot, processor = _compact_base_snapshot(tmp_path, monkeypatch)
    isolated_roots = _install_base_cleanup_failure(monkeypatch)

    with (
        pytest.raises(ValueError, match="primary model failure"),
        benchmark._isolated_base_snapshot(
            source,
            "2b",
            snapshot,
            processor,
            enabled=True,
        ),
    ):
        raise ValueError("primary model failure")

    assert len(isolated_roots) == 1
    assert not isolated_roots[0].exists()


def test_checkpoint_envelope_rejects_component_digest_tampering(tmp_path, monkeypatch):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    identity = _strict_checkpoint_identity(tmp_path, size="4b")
    monkeypatch.setattr(
        benchmark,
        "_inspect_checkpoint_identity",
        _matching_checkpoint_inspector(identity),
    )
    artifacts = _run(
        verified,
        tmp_path / "checkpoint-run",
        checkpoint_dir=identity.root,
    )
    envelope = json.loads(artifacts.envelope_path.read_text(encoding="utf-8"))
    envelope["execution_context"]["checkpoint"]["adapter_weights_sha256"] = "0" * 64
    artifacts.envelope_path.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        VLMConfidenceReportError, match="adapter_weights_sha256.*file record"
    ):
        benchmark._load_runner_envelope(artifacts.report_path, "run")


def test_checkpoint_envelope_rejects_aggregate_digest_tampering(tmp_path, monkeypatch):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    identity = _strict_checkpoint_identity(tmp_path, size="4b")
    monkeypatch.setattr(
        benchmark,
        "_inspect_checkpoint_identity",
        _matching_checkpoint_inspector(identity),
    )
    artifacts = _run(
        verified,
        tmp_path / "checkpoint-run",
        checkpoint_dir=identity.root,
    )
    envelope = json.loads(artifacts.envelope_path.read_text(encoding="utf-8"))
    envelope["execution_context"]["checkpoint"]["aggregate_sha256"] = "0" * 64
    artifacts.envelope_path.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(VLMConfidenceReportError, match="aggregate_sha256.*payload"):
        benchmark._load_runner_envelope(artifacts.report_path, "run")


def test_checkpoint_envelope_rejects_processor_record_tampering(tmp_path, monkeypatch):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    identity = _strict_checkpoint_identity(tmp_path, size="4b")
    monkeypatch.setattr(
        benchmark,
        "_inspect_checkpoint_identity",
        _matching_checkpoint_inspector(identity),
    )
    artifacts = _run(
        verified,
        tmp_path / "checkpoint-run",
        checkpoint_dir=identity.root,
    )
    envelope = json.loads(artifacts.envelope_path.read_text(encoding="utf-8"))
    processor_file = next(
        entry
        for entry in envelope["execution_context"]["checkpoint"]["files"]
        if entry["role"] == "processor"
    )
    processor_file["sha256"] = "0" * 64
    artifacts.envelope_path.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        VLMConfidenceReportError, match="processor_sha256.*processor file records"
    ):
        benchmark._load_runner_envelope(artifacts.report_path, "run")


def test_output_preflight_accepts_existing_writable_parent_without_side_effects(
    tmp_path,
):
    output = tmp_path / "run-a"

    destination = benchmark._validate_output_destination(output)

    assert destination == output.resolve()
    assert list(tmp_path.iterdir()) == []


def test_atomic_json_write_does_not_leave_partial_file(tmp_path, monkeypatch):
    destination = tmp_path / "status.json"

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr(benchmark.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        benchmark._write_json_atomic(destination, {"status": "ok"})

    assert not destination.exists()
    assert not list(tmp_path.glob(".status.json.*.tmp"))


def test_configure_determinism_seeds_rngs_and_disables_nondeterministic_flags(
    monkeypatch,
):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    previous_hash_seed = os.environ.get("PYTHONHASHSEED")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(benchmark, "_hash_randomization_enabled", lambda: False)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    try:
        first = benchmark.configure_determinism(23)
        python_value = random.random()
        numpy_value = np.random.random()
        torch_value = torch.rand(1).item()
        second = benchmark.configure_determinism(23)

        assert random.random() == python_value
        assert np.random.random() == numpy_value
        assert torch.rand(1).item() == torch_value
        assert first == second
        assert first["python_hash_seed"] == "0"
        assert first["torch_deterministic_algorithms"] is True
        assert first["cudnn_benchmark"] is False
        assert first["cudnn_deterministic"] is True
        assert first["cuda_matmul_allow_tf32"] is False
        assert first["cudnn_allow_tf32"] is False
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        if previous_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace
        if previous_hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = previous_hash_seed


def test_configure_determinism_requires_process_start_hash_seed(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    with pytest.raises(benchmark.BenchmarkInputError, match="PYTHONHASHSEED"):
        benchmark.configure_determinism(0)


def test_configure_determinism_rejects_late_hash_seed_env_mutation(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(benchmark, "_hash_randomization_enabled", lambda: True)

    with pytest.raises(benchmark.BenchmarkInputError, match="before starting"):
        benchmark.configure_determinism(0)


def test_offline_environment_is_forced_before_hugging_face_work(monkeypatch):
    for name in benchmark._OFFLINE_ENVIRONMENT:
        monkeypatch.setenv(name, "0")
    real_import = benchmark.import_module

    def offline_import(name):
        if name == "huggingface_hub.constants":
            return SimpleNamespace(HF_HUB_OFFLINE=True, HF_HUB_DISABLE_TELEMETRY=True)
        if name == "transformers.utils.hub":
            return SimpleNamespace(is_offline_mode=lambda: True)
        return real_import(name)

    monkeypatch.setattr(benchmark, "import_module", offline_import)

    context = benchmark._configure_offline_environment()

    assert all(os.environ[name] == "1" for name in benchmark._OFFLINE_ENVIRONMENT)
    assert context == {
        "hf_hub_offline": "1",
        "transformers_offline": "1",
        "hf_hub_disable_telemetry": "1",
        "do_not_track": "1",
        "hub_runtime_offline": True,
        "transformers_runtime_offline": True,
        "hub_runtime_telemetry_disabled": True,
    }


def test_offline_environment_rejects_late_hugging_face_import(monkeypatch):
    for name in benchmark._OFFLINE_ENVIRONMENT:
        monkeypatch.setenv(name, "0")

    def cached_online_import(name):
        if name == "huggingface_hub.constants":
            return SimpleNamespace(HF_HUB_OFFLINE=False, HF_HUB_DISABLE_TELEMETRY=False)
        if name == "transformers.utils.hub":
            return SimpleNamespace(is_offline_mode=lambda: False)
        raise AssertionError(name)

    monkeypatch.setattr(benchmark, "import_module", cached_online_import)

    with pytest.raises(benchmark.BenchmarkInputError, match="before importing"):
        benchmark._configure_offline_environment()

    assert all(os.environ[name] == "1" for name in benchmark._OFFLINE_ENVIRONMENT)


def test_device_probe_normalizes_numeric_cuda_and_records_bounded_evidence(
    monkeypatch,
):
    synchronized = []

    class ProbeScalar:
        def __add__(self, value):
            assert value == 1.0
            return self

        def item(self):
            return 2.0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(name=f"GPU {index}", total_memory=24_000),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 9))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (12_000, 24_000))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda index: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)
    monkeypatch.setattr(torch, "ones", lambda *args, **kwargs: ProbeScalar())
    monkeypatch.setattr(benchmark, "_nvidia_driver_version", lambda: "999.1")

    resolved, evidence = benchmark._resolve_and_probe_device("0")

    assert resolved == "cuda:0"
    assert evidence == {
        "requested_device": "0",
        "resolved_device": "cuda:0",
        "type": "cuda",
        "index": 0,
        "name": "GPU 0",
        "capability": [8, 9],
        "total_memory_bytes": 24_000,
        "free_memory_bytes": 12_000,
        "bf16_supported": True,
        "cuda_runtime": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "nvidia_driver": "999.1",
        "tiny_tensor_probe": "ok",
    }
    assert synchronized == [0]


def test_device_probe_resolves_auto_to_cpu_without_model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(benchmark, "_mps_available", lambda: False)

    resolved, evidence = benchmark._resolve_and_probe_device("auto")

    assert resolved == "cpu"
    assert evidence["requested_device"] == "auto"
    assert evidence["tiny_tensor_probe"] == "ok"


def test_device_probe_rejects_unavailable_or_out_of_range_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(benchmark.BenchmarkInputError, match="unavailable"):
        benchmark._resolve_and_probe_device("cuda:0")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(benchmark.BenchmarkInputError, match="outside"):
        benchmark._resolve_and_probe_device("2")


@pytest.mark.parametrize("value", ["", " ", "cuda:nope", object()])
def test_device_syntax_rejected_before_output_or_git(value, tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_validate_output_destination",
        lambda *args, **kwargs: pytest.fail("output checks must not start"),
    )

    with pytest.raises(benchmark.BenchmarkInputError):
        benchmark.preflight_benchmark(
            "manifest",
            "annotations",
            "images",
            "review",
            tmp_path / "output",
            device=value,
        )


def test_resolved_model_device_canonicalizes_indexless_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    model = SimpleNamespace(model=None, device=torch.device("cuda"))

    assert benchmark._resolved_model_device(model) == "cuda:3"


def test_attention_backend_rejects_flash_attention():
    model = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="flash_attention_2")
        )
    )

    with pytest.raises(benchmark.BenchmarkInputError, match="FlashAttention"):
        benchmark._attention_backends(model)


def test_run_benchmark_refuses_dirty_worktree_before_model(tmp_path, monkeypatch):
    verified = _verified_inputs(tmp_path)
    monkeypatch.setattr(
        benchmark,
        "_git_context",
        lambda: {"commit": "a" * 40, "dirty": True},
    )
    monkeypatch.setattr(
        benchmark,
        "LibreQwen3VL",
        lambda **kwargs: pytest.fail("model must not run from dirty source"),
    )

    with pytest.raises(benchmark.BenchmarkInputError, match="clean git worktree"):
        _run(verified, tmp_path / "output")


@pytest.mark.parametrize("override", ["1", "true", "YES", "on"])
def test_run_benchmark_refuses_faster_coco_env_before_model(
    override, tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    monkeypatch.setenv("LIBREYOLO_FASTER_COCO_EVAL", override)

    with pytest.raises(benchmark.BenchmarkInputError, match="faster-coco-eval"):
        _run(verified, tmp_path / "run-a")

    assert [event[0] for event in events] == ["determinism", "offline"]


def test_run_benchmark_translates_dataset_rejection_before_any_execution(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    output = tmp_path / "rejected"
    _install_run_fakes(monkeypatch, events, verified)

    def reject_inputs(*args, **kwargs):
        events.append(("verify_rejected",))
        raise benchmark.BenchmarkDatasetError("review attestation is not approved")

    monkeypatch.setattr(
        benchmark,
        "verify_benchmark_run_inputs",
        reject_inputs,
    )
    monkeypatch.setattr(
        benchmark,
        "_staged_output",
        lambda *args, **kwargs: pytest.fail("staging must not run after rejection"),
    )

    with pytest.raises(
        benchmark.BenchmarkInputError, match="review attestation is not approved"
    ):
        _run(verified, output)

    assert [event[0] for event in events] == [
        "determinism",
        "offline",
        "dependencies",
        "pycocotools",
        "device",
        "verify_rejected",
    ]
    assert not output.exists()
    assert not (tmp_path / ".rejected.lock").exists()


def test_snapshot_evidence_uses_only_the_fixed_local_qwen_snapshot(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "weights" / "LibreQwen3VL2b"
    root.mkdir(parents=True)
    calls = []

    def weights(cls, actual_root, repo, revision):
        calls.append(("weights", actual_root, repo, revision))
        return json.loads(json.dumps(benchmark._QWEN_2B_SNAPSHOT_IDENTITY))

    def processor(cls, actual_root, repo, revision):
        calls.append(("processor", actual_root, repo, revision))
        return dict(benchmark._QWEN_2B_PROCESSOR_CONTENT_IDENTITY)

    monkeypatch.setattr(
        benchmark.VLMConfidenceValidator,
        "_base_snapshot_identity_from_root",
        classmethod(weights),
    )
    monkeypatch.setattr(
        benchmark.VLMConfidenceValidator,
        "_processor_content_identity_from_root",
        classmethod(processor),
    )

    actual_root, snapshot, processor_content = benchmark._snapshot_evidence()

    expected_repo = benchmark.LibreQwen3VL.HF_REPOS["2b"]
    expected_revision = benchmark.LibreQwen3VL.HF_REVISIONS["2b"]
    assert actual_root == root.resolve()
    assert calls == [
        ("weights", root.resolve(), expected_repo, expected_revision),
        ("processor", root.resolve(), expected_repo, expected_revision),
    ]
    assert snapshot == benchmark._QWEN_2B_SNAPSHOT_IDENTITY
    assert processor_content == benchmark._QWEN_2B_PROCESSOR_CONTENT_IDENTITY


def test_runner_qwen_base_pins_match_publication_contract():
    from libreyolo.models.vlm import artifact as artifact_module

    assert benchmark._QWEN_BASE_PINS == artifact_module._SUPPORTED_BASES


def test_snapshot_evidence_accepts_only_the_audited_4b_sharded_snapshot(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "weights" / "LibreQwen3VL4b"
    root.mkdir(parents=True)
    calls = []

    def weights(cls, actual_root, repo, revision):
        calls.append(("weights", actual_root, repo, revision))
        return json.loads(json.dumps(benchmark._QWEN_4B_SNAPSHOT_IDENTITY))

    def processor(cls, actual_root, repo, revision):
        calls.append(("processor", actual_root, repo, revision))
        return dict(benchmark._QWEN_4B_PROCESSOR_CONTENT_IDENTITY)

    monkeypatch.setattr(
        benchmark.VLMConfidenceValidator,
        "_base_snapshot_identity_from_root",
        classmethod(weights),
    )
    monkeypatch.setattr(
        benchmark.VLMConfidenceValidator,
        "_processor_content_identity_from_root",
        classmethod(processor),
    )

    actual_root, snapshot, processor_content = benchmark._snapshot_evidence(
        "4b",
        expected_repo=benchmark._QWEN_4B_REPO,
        expected_revision=benchmark._QWEN_4B_REVISION,
    )

    assert actual_root == root.resolve()
    assert calls == [
        (
            "weights",
            root.resolve(),
            benchmark._QWEN_4B_REPO,
            benchmark._QWEN_4B_REVISION,
        ),
        (
            "processor",
            root.resolve(),
            benchmark._QWEN_4B_REPO,
            benchmark._QWEN_4B_REVISION,
        ),
    ]
    assert snapshot == benchmark._QWEN_4B_SNAPSHOT_IDENTITY
    assert processor_content == benchmark._QWEN_4B_PROCESSOR_CONTENT_IDENTITY


@pytest.mark.parametrize("link_position", ["leaf", "ancestor"])
def test_snapshot_evidence_preserves_lexical_weights_path_for_validator_checks(
    tmp_path, monkeypatch, link_position
):
    monkeypatch.chdir(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    if link_position == "leaf":
        target = external / "snapshot"
        target.mkdir()
        link = tmp_path / "weights" / "LibreQwen3VL2b"
        link.parent.mkdir()
    else:
        target_parent = external / "weights"
        target = target_parent / "LibreQwen3VL2b"
        target.mkdir(parents=True)
        link = tmp_path / "weights"
    try:
        link.symlink_to(
            target if link_position == "leaf" else target_parent,
            target_is_directory=True,
        )
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    roots = []

    def checkpoint(cls, root, repo, revision):
        roots.append(root)
        return json.loads(json.dumps(benchmark._QWEN_2B_SNAPSHOT_IDENTITY))

    def processor(cls, root, repo, revision):
        roots.append(root)
        return dict(benchmark._QWEN_2B_PROCESSOR_CONTENT_IDENTITY)

    monkeypatch.setattr(
        benchmark.VLMConfidenceValidator,
        "_base_snapshot_identity_from_root",
        classmethod(checkpoint),
    )
    monkeypatch.setattr(
        benchmark.VLMConfidenceValidator,
        "_processor_content_identity_from_root",
        classmethod(processor),
    )

    root, _snapshot, _processor = benchmark._snapshot_evidence()

    lexical = tmp_path / "weights" / "LibreQwen3VL2b"
    assert root == lexical
    assert roots == [lexical, lexical]
    assert root.resolve(strict=True) == target.resolve(strict=True)
    assert root != root.resolve(strict=True)


def test_snapshot_evidence_wraps_local_identity_failures_as_input_errors(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "weights" / "LibreQwen3VL2b").mkdir(parents=True)
    monkeypatch.setattr(
        benchmark.VLMConfidenceValidator,
        "_base_snapshot_identity_from_root",
        classmethod(
            lambda cls, root, repo, revision: (_ for _ in ()).throw(
                RuntimeError("unreferenced shard")
            )
        ),
    )

    with pytest.raises(benchmark.BenchmarkInputError, match="unreferenced shard"):
        benchmark._snapshot_evidence()


@pytest.mark.parametrize(
    ("target", "malformed"),
    [
        ("snapshot", True),
        ("snapshot", False),
        ("processor", True),
        ("processor", False),
    ],
)
def test_snapshot_evidence_rejects_malformed_or_forged_official_identity(
    tmp_path, monkeypatch, target, malformed
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "weights" / "LibreQwen3VL2b").mkdir(parents=True)
    snapshot = json.loads(json.dumps(benchmark._QWEN_2B_SNAPSHOT_IDENTITY))
    processor = dict(benchmark._QWEN_2B_PROCESSOR_CONTENT_IDENTITY)
    if target == "snapshot":
        if malformed:
            snapshot = {"sha256": snapshot["sha256"]}
        else:
            snapshot["artifacts"][1]["sha256"] = "0" * 64
    elif malformed:
        processor = {"sha256": processor["sha256"]}
    else:
        processor["sha256"] = "0" * 64

    monkeypatch.setattr(
        benchmark.VLMConfidenceValidator,
        "_base_snapshot_identity_from_root",
        classmethod(lambda cls, root, repo, revision: snapshot),
    )
    monkeypatch.setattr(
        benchmark.VLMConfidenceValidator,
        "_processor_content_identity_from_root",
        classmethod(lambda cls, root, repo, revision: processor),
    )

    expected = "weights" if target == "snapshot" else "processor content"
    with pytest.raises(benchmark.BenchmarkInputError, match=expected):
        benchmark._snapshot_evidence()


def test_snapshot_evidence_rejects_runner_repo_or_revision_pin_drift(
    monkeypatch,
):
    monkeypatch.setattr(
        benchmark.LibreQwen3VL,
        "HF_REVISIONS",
        {"2b": "0" * 40},
    )
    monkeypatch.setattr(
        benchmark.VLMConfidenceValidator,
        "_base_snapshot_identity_from_root",
        lambda *args, **kwargs: pytest.fail("identity hashing must not start"),
    )

    with pytest.raises(benchmark.BenchmarkInputError, match="official pin"):
        benchmark._snapshot_evidence()


def test_verified_dataset_yaml_round_trips_through_real_local_loader(tmp_path):
    verified = _verified_inputs(tmp_path)

    with benchmark._temporary_verified_dataset_yaml(verified) as dataset_yaml:
        payload = json.loads(dataset_yaml.read_text(encoding="utf-8"))
        temporary_parent = dataset_yaml.parent
        assert payload == {
            "path": str(verified.images_dir),
            "val": str(verified.images_dir),
            "annotations": {"val": str(verified.annotation_path)},
            "nc": 80,
            "names": list(verified.class_names),
        }

    assert not temporary_parent.exists()


def test_native_preflight_iterates_real_coco_path_and_binds_ground_truth(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (4, 2), color=(20, 30, 40)).save(
        images_dir / "first.jpg", format="JPEG"
    )
    Image.new("RGB", (3, 5), color=(50, 60, 70)).save(
        images_dir / "second.jpg", format="JPEG"
    )
    categories = [{"id": index + 1, "name": f"class-{index}"} for index in range(80)]
    annotations = [
        {
            "id": 10,
            "image_id": 101,
            "category_id": 1,
            "bbox": [0.0, 0.0, 2.0, 1.0],
            "area": 2.0,
            "iscrowd": 0,
        },
        {
            "id": 11,
            "image_id": 202,
            "category_id": 80,
            "bbox": [0.0, 0.0, 1.0, 2.0],
            "area": 2.0,
            "iscrowd": 0,
            "ignore": 0,
        },
    ]
    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 101,
                        "file_name": "first.jpg",
                        "width": 4,
                        "height": 2,
                    },
                    {
                        "id": 202,
                        "file_name": "second.jpg",
                        "width": 3,
                        "height": 5,
                    },
                ],
                "annotations": annotations,
                "categories": categories,
            }
        ),
        encoding="utf-8",
    )
    verified = SimpleNamespace(
        images_dir=images_dir.resolve(),
        annotation_path=annotation_path.resolve(),
        partition_start=0,
        partition_stop=2,
        class_names=tuple(category["name"] for category in categories),
        expected_images=(
            {
                "image_id": 101,
                "file_name": "first.jpg",
                "width": 4,
                "height": 2,
            },
            {
                "image_id": 202,
                "file_name": "second.jpg",
                "width": 3,
                "height": 5,
            },
        ),
        expected_categories=tuple(categories),
        expected_annotations=tuple(annotations),
    )

    context = benchmark._preflight_native_dataset(verified)

    assert context["dataset_class"].endswith(".COCODataset")
    assert context["preprocessor_class"].endswith(".StandardValPreprocessor")
    assert context["evaluator_class"].endswith(".COCOEvaluator")
    assert context["batch_size"] == 1
    assert context["num_workers"] == 0
    assert context["faster_coco_eval"] is False
    assert context["evaluator_self_test"] == "passed"
    assert context["evaluator_self_test_backend"].startswith("pycocotools ")
    assert context["image_count"] == 2
    assert context["category_count"] == 80
    assert context["annotation_count"] == 2
    assert benchmark._HEX_DIGEST.fullmatch(context["image_order_sha256"])
    assert benchmark._HEX_DIGEST.fullmatch(context["ground_truth_sha256"])


def test_native_evaluator_self_test_converts_compute_failure(monkeypatch):
    class BrokenEvaluator:
        def __init__(self, *args, **kwargs):
            self.last_backend = None

        def update(self, predictions, image_id):
            pass

        def compute(self):
            raise RuntimeError("broken cocoeval")

    monkeypatch.setattr(benchmark, "COCOEvaluator", BrokenEvaluator)
    annotation = {
        "image_id": 101,
        "category_id": 1,
        "bbox": [0.0, 0.0, 2.0, 1.0],
        "iscrowd": 0,
        "ignore": 0,
    }

    with pytest.raises(ValueError, match="self-test failed: broken cocoeval"):
        benchmark._exercise_native_evaluator(
            SimpleNamespace(coco=object()), [annotation], {0: 1}
        )


def test_evaluator_self_test_failure_stops_before_model_or_output(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    output = tmp_path / "broken-evaluator"
    monkeypatch.setattr(
        benchmark,
        "_preflight_native_dataset",
        lambda verified: (_ for _ in ()).throw(
            benchmark.BenchmarkInputError("COCO evaluator self-test failed")
        ),
    )

    with pytest.raises(benchmark.BenchmarkInputError, match="self-test failed"):
        _run(verified, output)

    assert "model" not in [event[0] for event in events]
    assert not output.exists()
    assert not (tmp_path / ".broken-evaluator.lock").exists()


def test_run_benchmark_preflights_pycocotools_before_config_or_model(
    tmp_path, monkeypatch
):
    verified = _verified_inputs(tmp_path)
    events = []
    _install_run_fakes(monkeypatch, events, verified)

    def reject_dependency():
        events.append(("pycocotools",))
        raise benchmark.BenchmarkInputError("pycocotools is required")

    monkeypatch.setattr(benchmark, "_require_pycocotools", reject_dependency)
    for name in (
        "load_data_config",
        "LibreQwen3VL",
        "_staged_output",
    ):
        monkeypatch.setattr(
            benchmark,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"{_name} must not run after dependency rejection"
            ),
        )

    with pytest.raises(benchmark.BenchmarkInputError, match="pycocotools"):
        _run(verified, tmp_path / "missing-dependency")

    assert [event[0] for event in events] == [
        "determinism",
        "offline",
        "dependencies",
        "pycocotools",
    ]


def test_run_benchmark_rejects_resolved_path_drift_before_staging_or_model(
    tmp_path, monkeypatch
):
    verified = _verified_inputs(tmp_path)
    events = []
    _install_run_fakes(monkeypatch, events, verified)
    wrong_images = tmp_path / "wrong-images"
    wrong_images.mkdir()

    def wrong_config(data, *, autodownload, allow_scripts):
        events.append(("data_config_drift", autodownload, allow_scripts))
        return {
            "path": str(wrong_images),
            "root": str(wrong_images),
            "val": str(wrong_images),
            "annotations": {"val": str(verified.annotation_path)},
            "val_annotation_file": str(verified.annotation_path),
            "nc": 80,
            "names": list(verified.class_names),
        }

    monkeypatch.setattr(benchmark, "load_data_config", wrong_config)
    monkeypatch.setattr(
        benchmark,
        "_staged_output",
        lambda *args, **kwargs: pytest.fail("staging must not begin"),
    )

    with pytest.raises(benchmark.BenchmarkInputError, match="root does not match"):
        _run(verified, tmp_path / "drift")

    assert [event[0] for event in events] == [
        "determinism",
        "offline",
        "dependencies",
        "pycocotools",
        "device",
        "verify",
        "data_config_drift",
    ]


def test_run_cli_reports_dataset_rejection_as_input_error(
    tmp_path, monkeypatch, capsys
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    monkeypatch.setattr(
        benchmark,
        "verify_benchmark_run_inputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            benchmark.BenchmarkDatasetError("manifest mismatch")
        ),
    )

    code = benchmark.main(_cli_run_args(verified, tmp_path / "rejected"))

    assert code == benchmark.EXIT_USAGE
    status = _status(capsys)
    assert status["error"]["kind"] == "input"
    assert "manifest mismatch" in status["error"]["message"]


def test_parse_cli_args_covers_run_and_compare_contracts():
    run = benchmark.parse_cli_args(
        [
            "run",
            "--manifest",
            "bundle/manifest.json",
            "--annotations",
            "instances_val2017.json",
            "--images-dir",
            "val2017",
            "--review-attestation",
            "review.json",
            "--output-root",
            "run-a",
            "--checkpoint-dir",
            "checkpoint/best",
            "--seed",
            "19",
            "--device",
            "cuda:1",
        ]
    )
    assert run.mode == "run"
    assert run.manifest == Path("bundle/manifest.json")
    assert run.annotations == Path("instances_val2017.json")
    assert run.images_dir == Path("val2017")
    assert run.review_attestation == Path("review.json")
    assert run.output_root == Path("run-a")
    assert run.checkpoint_dir == Path("checkpoint/best")
    assert run.seed == 19
    assert run.device == "cuda:1"

    preflight = benchmark.parse_cli_args(
        [
            "preflight",
            "--manifest",
            "bundle/manifest.json",
            "--annotations",
            "instances_val2017.json",
            "--images-dir",
            "val2017",
            "--review-attestation",
            "review.json",
            "--output-root",
            "run-a",
            "--checkpoint-dir",
            "checkpoint/best",
            "--seed",
            "19",
            "--device",
            "cuda:1",
        ]
    )
    assert {
        key: getattr(preflight, key)
        for key in (
            "manifest",
            "annotations",
            "images_dir",
            "review_attestation",
            "output_root",
            "seed",
            "device",
            "checkpoint_dir",
        )
    } == {
        key: getattr(run, key)
        for key in (
            "manifest",
            "annotations",
            "images_dir",
            "review_attestation",
            "output_root",
            "seed",
            "device",
            "checkpoint_dir",
        )
    }

    compare = benchmark.parse_cli_args(
        [
            "compare",
            "first.json",
            "second.json",
            "--score-atol",
            "1e-6",
            "--metric-atol",
            "2e-6",
            "--map-atol",
            "3e-6",
        ]
    )
    assert compare.mode == "compare"
    assert compare.first_report == Path("first.json")
    assert compare.second_report == Path("second.json")
    assert compare.score_atol == 1e-6
    assert compare.metric_atol == 2e-6
    assert compare.map_atol == 3e-6

    with pytest.raises(benchmark._CLIUsageError):
        benchmark.parse_cli_args(
            [
                "run",
                "--manifest",
                "bundle/manifest.json",
                "--annotations",
                "instances_val2017.json",
                "--images-dir",
                "val2017",
                "--review-attestation",
                "review.json",
                "--output-root",
                "run-a",
                "--model-size",
                "4b",
            ]
        )

    with pytest.raises(benchmark._CLIUsageError):
        benchmark.parse_cli_args(
            [
                "run",
                "--data",
                "dataset.yaml",
                "--output-root",
                "run-a",
            ]
        )

    with pytest.raises(benchmark._CLIUsageError):
        benchmark.parse_cli_args(
            [
                "run",
                "--manifest",
                "bundle/manifest.json",
                "--annotations",
                "instances_val2017.json",
                "--images-dir",
                "val2017",
                "--review-attestation",
                "review.json",
                "--output-root",
                "run-a",
                "--partition-role",
                "confidence_smoke",
            ]
        )


@pytest.mark.parametrize("bad_tolerance", ["-1", "nan", "inf"])
def test_compare_cli_rejects_invalid_tolerances_as_json_usage(bad_tolerance, capsys):
    code = benchmark.main(
        ["compare", "first.json", "second.json", "--map-atol", bad_tolerance]
    )

    assert code == benchmark.EXIT_USAGE
    status = _status(capsys)
    assert status["status"] == "error"
    assert status["error"]["kind"] == "usage"


def test_run_cli_success_emits_one_strict_json_status(tmp_path, monkeypatch, capsys):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(
        monkeypatch,
        events,
        verified,
        metrics={"metric/undefined": float("nan")},
    )
    output = tmp_path / "run-a"

    code = benchmark.main(_cli_run_args(verified, output))

    assert code == benchmark.EXIT_OK
    status = _status(capsys)
    assert status["status"] == "ok"
    assert status["code"] == benchmark.EXIT_OK
    assert status["nonfinite_metrics"] == ["metric/undefined"]
    assert Path(status["report"]).is_file()


def test_run_cli_routes_library_stdout_away_from_json_status(
    tmp_path, monkeypatch, capsys
):
    verified = _verified_inputs(tmp_path)
    output = tmp_path / "run-a"

    def fake_run(*args, **kwargs):
        print("library chatter")
        return benchmark.BenchmarkArtifacts(
            output_dir=output,
            report_path=output / "vlm_confidence_report.json",
            envelope_path=output / "vlm_confidence_run.json",
            metrics={},
            nonfinite_metrics=(),
        )

    monkeypatch.setattr(benchmark, "run_benchmark", fake_run)

    code = benchmark.main(_cli_run_args(verified, output))

    assert code == benchmark.EXIT_OK
    captured = capsys.readouterr()
    assert "library chatter" in captured.err
    assert json.loads(captured.out)["status"] == "ok"
    assert captured.out.count("\n") == 1


def test_run_cli_reports_overwrite_with_distinct_exit_code(tmp_path, capsys):
    verified = _verified_inputs(tmp_path)
    output = tmp_path / "occupied"
    output.mkdir()

    code = benchmark.main(_cli_run_args(verified, output))

    assert code == benchmark.EXIT_OUTPUT_EXISTS
    status = _status(capsys)
    assert status["error"]["kind"] == "output_exists"


def test_preflight_cli_reports_existing_output_without_running_checks(
    tmp_path, monkeypatch, capsys
):
    verified = _verified_inputs(tmp_path)
    output = tmp_path / "occupied-preflight"
    output.mkdir()
    monkeypatch.setattr(
        benchmark,
        "_git_context",
        lambda: pytest.fail("git must not run for an occupied output"),
    )

    code = benchmark.main(_cli_preflight_args(verified, output))

    assert code == benchmark.EXIT_OUTPUT_EXISTS
    status = _status(capsys)
    assert status["mode"] == "preflight"
    assert status["error"]["kind"] == "output_exists"


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("configure_determinism", "process-start hash seed is not fixed"),
        ("_configure_offline_environment", "offline mode could not be forced"),
        ("_required_package_versions", "required dependency is missing"),
        ("_resolve_and_probe_device", "requested device probe failed"),
        ("_snapshot_evidence", "local snapshot is incomplete"),
    ],
)
def test_preflight_cli_maps_readiness_failures_to_input_exit(
    target, message, tmp_path, monkeypatch, capsys
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    output = tmp_path / "not-ready"
    monkeypatch.setattr(
        benchmark,
        target,
        lambda *args, **kwargs: (_ for _ in ()).throw(
            benchmark.BenchmarkInputError(message)
        ),
    )

    code = benchmark.main(_cli_preflight_args(verified, output))

    assert code == benchmark.EXIT_USAGE
    status = _status(capsys)
    assert status["mode"] == "preflight"
    assert status["error"] == {"kind": "input", "message": message}
    assert "model" not in [event[0] for event in events]
    assert not output.exists()
    assert not (tmp_path / ".not-ready.lock").exists()


def test_preflight_rejects_dirty_git_before_process_or_model_checks(
    tmp_path, monkeypatch, capsys
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    monkeypatch.setattr(
        benchmark, "_git_context", lambda: {"commit": "a" * 40, "dirty": True}
    )
    output = tmp_path / "dirty-preflight"

    code = benchmark.main(_cli_preflight_args(verified, output))

    assert code == benchmark.EXIT_USAGE
    status = _status(capsys)
    assert "clean git worktree" in status["error"]["message"]
    assert events == []
    assert not output.exists()


def test_preflight_rejects_faster_coco_override_before_dependencies_or_model(
    tmp_path, monkeypatch
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(monkeypatch, events, verified)
    monkeypatch.setenv("LIBREYOLO_FASTER_COCO_EVAL", "true")

    with pytest.raises(benchmark.BenchmarkInputError, match="faster-coco-eval"):
        _preflight(verified, tmp_path / "faster-preflight")

    assert [event[0] for event in events] == ["determinism", "offline"]


@pytest.mark.parametrize(
    ("reproducible", "expected_code", "expected_status"),
    [
        (True, benchmark.EXIT_OK, "reproducible"),
        (False, benchmark.EXIT_NOT_REPRODUCIBLE, "different"),
    ],
)
def test_compare_cli_exit_code_reflects_reproducibility(
    reproducible, expected_code, expected_status, monkeypatch, capsys
):
    captured = {}

    def fake_compare(first, second, **kwargs):
        captured.update(first=first, second=second, **kwargs)
        return {
            "reproducible": reproducible,
            "max_abs_semantic_metric_delta": 0.0,
        }

    monkeypatch.setattr(benchmark, "compare_benchmarks", fake_compare)
    code = benchmark.main(
        [
            "compare",
            "first.json",
            "second.json",
            "--score-atol",
            "0.1",
            "--metric-atol",
            "0.2",
            "--map-atol",
            "0.3",
        ]
    )

    assert code == expected_code
    status = _status(capsys)
    assert status["status"] == expected_status
    assert status["comparison"]["reproducible"] is reproducible
    assert status["tolerances"] == {
        "score_atol": 0.1,
        "metric_atol": 0.2,
        "map_atol": 0.3,
    }
    assert captured == {
        "first": Path("first.json"),
        "second": Path("second.json"),
        "score_atol": 0.1,
        "metric_atol": 0.2,
        "map_atol": 0.3,
    }


def test_compare_cli_reports_malformed_report_with_distinct_exit_code(
    monkeypatch, capsys
):
    def fail_compare(*args, **kwargs):
        raise VLMConfidenceReportError("first:$: invalid")

    monkeypatch.setattr(benchmark, "compare_benchmarks", fail_compare)

    code = benchmark.main(["compare", "first.json", "second.json"])

    assert code == benchmark.EXIT_INVALID_REPORT
    status = _status(capsys)
    assert status["error"] == {
        "kind": "invalid_report",
        "message": "first:$: invalid",
    }


def test_run_cli_reports_execution_failure_and_leaves_no_output(
    tmp_path, monkeypatch, capsys
):
    events = []
    verified = _verified_inputs(tmp_path)
    _install_run_fakes(
        monkeypatch,
        events,
        verified,
        failure=RuntimeError("generation failed"),
    )
    output = tmp_path / "failed"

    code = benchmark.main(_cli_run_args(verified, output))

    assert code == benchmark.EXIT_RUN_FAILED
    status = _status(capsys)
    assert status["error"] == {"kind": "execution", "message": "generation failed"}
    assert not output.exists()


def test_direct_compare_delegates_to_strict_persisted_comparator(monkeypatch):
    sentinel = SimpleNamespace(
        first_report_sha256="a" * 64,
        second_report_sha256="b" * 64,
    )
    captured = {}

    def fake_compare(first, second, **kwargs):
        captured.update(first=first, second=second, **kwargs)
        return sentinel

    monkeypatch.setattr(benchmark, "compare_confidence_reports", fake_compare)
    monkeypatch.setattr(
        benchmark,
        "_load_runner_envelope",
        lambda path, label: benchmark._ValidatedEnvelope(
            run_id="1" * 32 if label == "first_run" else "2" * 32,
            process_id="3" * 32 if label == "first_run" else "4" * 32,
            report_sha256="a" * 64 if label == "first_run" else "b" * 64,
            envelope_sha256="c" * 64,
            execution_context={},
            benchmark_config={},
            metrics={},
            nonfinite_metrics=(),
        ),
    )

    result = benchmark.compare_benchmarks(
        "first.json",
        "second.json",
        score_atol=0.01,
        metric_atol=0.02,
        map_atol=0.03,
    )

    assert result is sentinel
    assert captured == {
        "first": "first.json",
        "second": "second.json",
        "score_atol": 0.01,
        "metric_atol": 0.02,
        "map_atol": 0.03,
    }


def test_direct_compare_rejects_same_report_without_validation(tmp_path, monkeypatch):
    report = tmp_path / "vlm_confidence_report.json"
    report.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        benchmark,
        "_load_runner_envelope",
        lambda *args, **kwargs: pytest.fail("same report must fail first"),
    )

    with pytest.raises(VLMConfidenceReportError, match="two distinct"):
        benchmark.compare_benchmarks(report, report)


def test_direct_compare_requires_distinct_processes(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_load_runner_envelope",
        lambda path, label: benchmark._ValidatedEnvelope(
            run_id="1" * 32 if label == "first_run" else "2" * 32,
            process_id="3" * 32,
            report_sha256="a" * 64,
            envelope_sha256="c" * 64,
            execution_context={},
            benchmark_config={},
            metrics={},
            nonfinite_metrics=(),
        ),
    )

    with pytest.raises(VLMConfidenceReportError, match="fresh Python processes"):
        benchmark.compare_benchmarks("first.json", "second.json")


def _repeatability_comparison(
    first_sha256: str,
    second_sha256: str,
    *,
    reproducible: bool = True,
) -> benchmark.PersistedRepeatComparison:
    core = benchmark.RepeatComparison(
        same_manifest=True,
        same_configuration=True,
        same_generation=reproducible,
        same_prediction_structure=True,
        same_matches=True,
        same_score_availability=True,
        scores_within_tolerance=True,
        metrics_within_tolerance=True,
        same_calibration_bin_assignments=True,
        calibration_bins_within_tolerance=True,
        same_evaluator_metric_keys=True,
        evaluator_metrics_within_tolerance=True,
        same_diagnostics=True,
        max_abs_score_delta=0.0,
        max_abs_calibration_bin_delta=0.0,
        max_abs_evaluator_metric_delta=0.0,
        auroc_delta=0.0,
        ranking_ap_delta=0.0,
        brier_score_delta=0.0,
        expected_calibration_error_delta=0.0,
        maximum_calibration_error_delta=0.0,
        reproducible=reproducible,
    )
    return benchmark.PersistedRepeatComparison(
        first_report_sha256=first_sha256,
        second_report_sha256=second_sha256,
        core=core,
        same_response_diagnostics=True,
        same_fallback_reasons=True,
        same_semantic_metric_keys=True,
        map_metrics_within_tolerance=True,
        non_map_metrics_within_tolerance=True,
        semantic_metrics_within_tolerance=True,
        max_abs_semantic_metric_delta=0.0,
        differing_fields=() if reproducible else ("hashes.generation",),
        reproducible=reproducible,
    )


def _receipt_inputs(tmp_path: Path, monkeypatch, *, reproducible: bool = True):
    reports = []
    for index in (1, 2):
        run = tmp_path / f"run-{index}"
        run.mkdir()
        report = run / "vlm_confidence_report.json"
        report.write_text(json.dumps({"run": index}) + "\n", encoding="utf-8")
        (run / "vlm_confidence_run.json").write_text(
            json.dumps({"envelope": index}) + "\n", encoding="utf-8"
        )
        reports.append(report)
    report_digests = tuple(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in reports
    )
    envelope_digests = tuple(
        hashlib.sha256(
            path.with_name("vlm_confidence_run.json").read_bytes()
        ).hexdigest()
        for path in reports
    )
    comparison = _repeatability_comparison(
        report_digests[0], report_digests[1], reproducible=reproducible
    )
    captured = {}

    def fake_envelope(path, label):
        index = 0 if label == "first_run" else 1
        return benchmark._ValidatedEnvelope(
            run_id=("1" if index == 0 else "3") * 32,
            process_id=("2" if index == 0 else "4") * 32,
            report_sha256=report_digests[index],
            envelope_sha256=envelope_digests[index],
            execution_context={},
            benchmark_config={},
            metrics={},
            nonfinite_metrics=(),
        )

    def fake_compare(first, second, **kwargs):
        captured["reports"] = (Path(first), Path(second))
        captured["tolerances"] = kwargs
        return comparison

    monkeypatch.setattr(benchmark, "_load_runner_envelope", fake_envelope)
    monkeypatch.setattr(benchmark, "compare_benchmarks", fake_compare)
    return (reports[0], reports[1]), comparison, captured


def test_repeatability_receipt_is_canonical_path_free_and_immutable(
    tmp_path, monkeypatch
):
    reports, comparison, captured = _receipt_inputs(tmp_path, monkeypatch)
    output = tmp_path / "repeatability.json"

    identity = benchmark.create_benchmark_repeatability_receipt(
        reports[0], reports[1], output
    )

    payload = output.read_bytes()
    decoded = json.loads(payload)
    assert payload == benchmark._json_text(decoded).encode()
    assert decoded["schema"] == benchmark.REPEATABILITY_RECEIPT_SCHEMA
    assert decoded["comparison"]["reproducible"] is True
    assert decoded["tolerances"] == {
        "score_atol": 0.0,
        "metric_atol": 0.0,
        "map_atol": 0.0,
    }
    assert identity.receipt_sha256 == hashlib.sha256(payload).hexdigest()
    assert (
        identity.comparison_sha256
        == hashlib.sha256(
            benchmark._json_text(decoded["comparison"]).encode()
        ).hexdigest()
    )
    assert identity.comparison == comparison
    assert captured["tolerances"] == decoded["tolerances"]
    assert all(
        path.parent != report.parent
        for path, report in zip(captured["reports"], reports)
    )
    assert all(
        str(report.parent.resolve()) not in payload.decode() for report in reports
    )
    with pytest.raises(TypeError):
        identity.tolerances["score_atol"] = 1.0


@pytest.mark.parametrize(
    "run_index, filename",
    [
        (0, "vlm_confidence_report.json"),
        (0, "vlm_confidence_run.json"),
        (1, "vlm_confidence_report.json"),
        (1, "vlm_confidence_run.json"),
    ],
)
def test_repeatability_receipt_revalidates_all_sources_before_publication(
    tmp_path, monkeypatch, run_index, filename
):
    reports, _comparison, _captured = _receipt_inputs(tmp_path, monkeypatch)
    output = tmp_path / "repeatability.json"
    revalidate = benchmark._revalidate_repeatability_sources

    def mutate_then_revalidate(records):
        target = reports[run_index].with_name(filename)
        target.write_text('{"changed":true}\n', encoding="utf-8")
        revalidate(records)

    monkeypatch.setattr(
        benchmark, "_revalidate_repeatability_sources", mutate_then_revalidate
    )

    with pytest.raises(VLMConfidenceReportError, match="changed"):
        benchmark.create_benchmark_repeatability_receipt(reports[0], reports[1], output)
    assert not output.exists()


@pytest.mark.parametrize(
    "run_index, filename",
    [
        (0, "vlm_confidence_report.json"),
        (0, "vlm_confidence_run.json"),
        (1, "vlm_confidence_report.json"),
        (1, "vlm_confidence_run.json"),
    ],
)
def test_repeatability_receipt_revalidates_private_inputs_before_publication(
    tmp_path, monkeypatch, run_index, filename
):
    reports, _comparison, _captured = _receipt_inputs(tmp_path, monkeypatch)
    output = tmp_path / "repeatability.json"
    load_envelope = benchmark._load_runner_envelope
    target_label = "first_run" if run_index == 0 else "second_run"
    mutated = False

    def mutate_private_then_load(path, label):
        nonlocal mutated
        if label == target_label and not mutated:
            Path(path).with_name(filename).write_text(
                '{"substituted":true}\n', encoding="utf-8"
            )
            mutated = True
        return load_envelope(path, label)

    monkeypatch.setattr(benchmark, "_load_runner_envelope", mutate_private_then_load)

    with pytest.raises(VLMConfidenceReportError, match="private byte snapshot|private"):
        benchmark.create_benchmark_repeatability_receipt(reports[0], reports[1], output)
    assert not output.exists()


def test_repeatability_receipt_rejects_transient_private_identity(
    tmp_path, monkeypatch
):
    reports, _comparison, _captured = _receipt_inputs(tmp_path, monkeypatch)
    output = tmp_path / "repeatability.json"
    load_envelope = benchmark._load_runner_envelope

    def substitute_validated_identity(path, label):
        envelope = load_envelope(path, label)
        if label == "first_run":
            return dataclasses.replace(envelope, report_sha256="a" * 64)
        return envelope

    monkeypatch.setattr(
        benchmark, "_load_runner_envelope", substitute_validated_identity
    )

    with pytest.raises(VLMConfidenceReportError, match="private byte snapshot"):
        benchmark.create_benchmark_repeatability_receipt(reports[0], reports[1], output)
    assert not output.exists()


def test_repeatability_receipt_reader_rejects_duplicate_ids_and_noncanonical_json(
    tmp_path, monkeypatch
):
    reports, _comparison, _captured = _receipt_inputs(tmp_path, monkeypatch)
    valid = tmp_path / "valid.json"
    benchmark.create_benchmark_repeatability_receipt(reports[0], reports[1], valid)
    payload = json.loads(valid.read_text(encoding="utf-8"))
    payload["runs"][1]["process_id"] = payload["runs"][0]["process_id"]
    duplicate_process = tmp_path / "duplicate-process.json"
    duplicate_process.write_bytes(benchmark._json_text(payload).encode())
    with pytest.raises(VLMConfidenceReportError, match="process_id values must differ"):
        benchmark.read_benchmark_repeatability_receipt(duplicate_process)

    inconsistent_delta = json.loads(valid.read_text(encoding="utf-8"))
    inconsistent_delta["comparison"]["max_abs_semantic_metric_delta"] = 0.5
    inconsistent = tmp_path / "inconsistent.json"
    inconsistent.write_bytes(benchmark._json_text(inconsistent_delta).encode())
    with pytest.raises(VLMConfidenceReportError, match="metric tolerances"):
        benchmark.read_benchmark_repeatability_receipt(inconsistent)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(json.loads(valid.read_text())), encoding="utf-8")
    with pytest.raises(VLMConfidenceReportError, match="canonical JSON"):
        benchmark.read_benchmark_repeatability_receipt(noncanonical)


def test_repeatability_receipt_reader_rejects_untrusted_json_shapes(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema":"first","schema":"second"}\n')
    with pytest.raises(VLMConfidenceReportError, match="duplicate JSON object key"):
        benchmark.read_benchmark_repeatability_receipt(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_bytes(
        b'{"comparison":{},"runs":[],"schema":"x","tolerances":{"map_atol":1e400}}\n'
    )
    with pytest.raises(
        VLMConfidenceReportError, match="invalid JSON value|canonical JSON"
    ):
        benchmark.read_benchmark_repeatability_receipt(nonfinite)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (benchmark._MAX_REPEATABILITY_RECEIPT_BYTES + 1))
    with pytest.raises(VLMConfidenceReportError, match="exceeds"):
        benchmark.read_benchmark_repeatability_receipt(oversized)


def test_repeatability_receipt_records_failure_but_public_result_remains_false(
    tmp_path, monkeypatch
):
    reports, _comparison, _captured = _receipt_inputs(
        tmp_path, monkeypatch, reproducible=False
    )
    output = tmp_path / "different.json"

    identity = benchmark.create_benchmark_repeatability_receipt(
        reports[0], reports[1], output
    )

    assert not identity.comparison.reproducible
    assert identity.comparison.differing_fields == ("hashes.generation",)


def test_repeatability_receipt_preserves_comparator_diagnostic_order(
    tmp_path, monkeypatch
):
    reports, comparison, _captured = _receipt_inputs(
        tmp_path, monkeypatch, reproducible=False
    )
    comparison = dataclasses.replace(
        comparison,
        same_response_diagnostics=False,
        differing_fields=("response_diagnostics", "hashes.generation"),
    )
    monkeypatch.setattr(
        benchmark, "compare_benchmarks", lambda *args, **kwargs: comparison
    )
    output = tmp_path / "different.json"

    identity = benchmark.create_benchmark_repeatability_receipt(
        reports[0], reports[1], output
    )

    assert identity.comparison.differing_fields == (
        "response_diagnostics",
        "hashes.generation",
    )
    assert json.loads(output.read_bytes())["comparison"]["differing_fields"] == [
        "response_diagnostics",
        "hashes.generation",
    ]


def test_repeatability_receipt_rejects_path_in_differing_fields(tmp_path, monkeypatch):
    reports, _comparison, _captured = _receipt_inputs(
        tmp_path, monkeypatch, reproducible=False
    )
    valid = tmp_path / "valid.json"
    benchmark.create_benchmark_repeatability_receipt(reports[0], reports[1], valid)
    payload = json.loads(valid.read_bytes())
    payload["comparison"]["differing_fields"] = ["C:/private/user/path"]
    forged = tmp_path / "forged.json"
    forged.write_bytes(benchmark._json_text(payload).encode())

    with pytest.raises(VLMConfidenceReportError, match="supported comparator"):
        benchmark.read_benchmark_repeatability_receipt(forged)


def test_repeatability_receipt_is_create_only(tmp_path, monkeypatch):
    reports, _comparison, _captured = _receipt_inputs(tmp_path, monkeypatch)
    output = tmp_path / "existing.json"
    output.write_bytes(b"racer")

    with pytest.raises(benchmark.BenchmarkOutputExistsError):
        benchmark.create_benchmark_repeatability_receipt(reports[0], reports[1], output)
    assert output.read_bytes() == b"racer"


def test_repeatability_receipt_rejects_hardlinked_source(tmp_path, monkeypatch):
    reports, _comparison, _captured = _receipt_inputs(tmp_path, monkeypatch)
    outside = tmp_path / "hardlink.json"
    try:
        os.link(reports[0], outside)
    except OSError as exc:
        pytest.skip(f"hardlink creation unavailable: {exc}")

    with pytest.raises(VLMConfidenceReportError, match="hard-linked"):
        benchmark.create_benchmark_repeatability_receipt(
            reports[0], reports[1], tmp_path / "receipt.json"
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_repeatability_receipt_reader_rejects_symlink(tmp_path, monkeypatch):
    reports, _comparison, _captured = _receipt_inputs(tmp_path, monkeypatch)
    receipt = tmp_path / "receipt.json"
    benchmark.create_benchmark_repeatability_receipt(reports[0], reports[1], receipt)
    linked = tmp_path / "linked.json"
    try:
        linked.symlink_to(receipt)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(VLMConfidenceReportError, match="symlink|junction"):
        benchmark.read_benchmark_repeatability_receipt(linked)


def test_compare_cli_can_create_repeatability_receipt(monkeypatch, capsys, tmp_path):
    reports, _comparison, _captured = _receipt_inputs(tmp_path, monkeypatch)
    requested = tmp_path / "requested.json"

    code = benchmark.main(
        [
            "compare",
            str(reports[0]),
            str(reports[1]),
            "--receipt",
            str(requested),
        ]
    )

    assert code == benchmark.EXIT_OK
    identity = benchmark.read_benchmark_repeatability_receipt(requested)
    status = _status(capsys)
    assert status["receipt"] == {
        "path": str(requested.resolve()),
        "sha256": identity.receipt_sha256,
        "comparison_sha256": identity.comparison_sha256,
    }


@pytest.mark.parametrize("value", [True, -1, 2**32, 1.5, "1.5"])
def test_run_benchmark_rejects_invalid_seed_before_side_effects(value, tmp_path):
    verified = _verified_inputs(tmp_path)
    with pytest.raises(benchmark.BenchmarkInputError):
        _run(verified, tmp_path / "output", seed=value)


def test_metric_normalization_rejects_non_numeric_values():
    with pytest.raises(TypeError, match="must be numeric"):
        benchmark._normalized_metrics({"metric": object()})
    with pytest.raises(TypeError, match="must be numeric"):
        benchmark._normalized_metrics({"metric": True})
    assert benchmark._normalized_metrics({"metric": math.nan}) == (
        {"metric": None},
        ("metric",),
    )
