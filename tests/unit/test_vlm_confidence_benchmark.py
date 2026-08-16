"""Offline tests for the internal VLM confidence benchmark runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libreyolo.validation import vlm_confidence_benchmark as benchmark
from libreyolo.validation.vlm_confidence_report import VLMConfidenceReportError

pytestmark = [pytest.mark.unit, pytest.mark.vlm]


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text("names: [cat]\n", encoding="utf-8")
    return path


def _install_run_fakes(monkeypatch, events, *, metrics=None, failure=None):
    metrics = {"metric/finite": 0.5} if metrics is None else metrics
    report_identities = {}

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
        def __init__(self, *, size, device):
            events.append(("model", size, device))
            assert events[0][0] == "determinism"
            self.device = torch.device("cpu")

    class FakeValidator:
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
            report_identities[report.resolve()] = {
                "benchmark_config": {
                    "family": "qwen3vl",
                    "size": "2b",
                    "seed": self.seed,
                    "device": str(self.model.device),
                    "evaluation": {
                        "imgsz": [self.config.imgsz, self.config.imgsz],
                        "faster_coco_eval": self.config.faster_coco_eval,
                        "backend": "pycocotools offline",
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
    monkeypatch.setattr(
        benchmark,
        "_runtime_context",
        lambda **kwargs: {
            "python": "3.11.0",
            "implementation": "CPython",
            "platform": "offline",
            "torch": "offline",
            "numpy": "offline",
            "pillow": "offline",
            "opencv": "offline",
            "packages": {
                "transformers": "offline",
                "huggingface_hub": "offline",
                "tokenizers": "offline",
                "safetensors": "offline",
                "pycocotools": "offline",
            },
            "cuda_runtime": None,
            "cudnn": None,
            "nvidia_driver": None,
            "cuda_available": False,
            "attention_backends": {"model": "offline"},
            **kwargs,
        },
    )
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


def test_run_benchmark_stages_complete_artifacts_and_records_context(
    tmp_path, monkeypatch
):
    events = []
    _install_run_fakes(
        monkeypatch,
        events,
        metrics={
            "metric/finite": 0.5,
            "metric/nan": float("nan"),
            "metric/positive_infinity": float("inf"),
            "metric/negative_infinity": float("-inf"),
        },
    )
    dataset = _dataset(tmp_path)
    output = tmp_path / "run-a"

    artifacts = benchmark.run_benchmark(dataset, output, seed=17, device="cuda:0")

    assert artifacts.output_dir == output.resolve()
    assert artifacts.report_path.is_file()
    assert artifacts.envelope_path.is_file()
    assert sorted(path.name for path in output.iterdir()) == [
        "vlm_confidence_report.json",
        "vlm_confidence_run.json",
    ]
    assert not list(tmp_path.glob(".run-a.tmp-*"))
    assert not (tmp_path / ".run-a.lock").exists()
    assert [event[0] for event in events[:4]] == [
        "determinism",
        "model",
        "validator",
        "run",
    ]
    config = events[2][2]
    assert config.data == str(dataset.resolve())
    assert config.batch_size == 1
    assert config.num_workers == 0
    assert config.allow_download_scripts is False
    assert config.imgsz == 1024
    assert config.save_json is True
    assert config.save_plots is True
    assert config.faster_coco_eval is False
    assert events[2][3]["default_conf"] == 0.25
    assert events[2][3]["confidence_iou"] == 0.5
    assert events[2][3]["benchmark_context"]["git"] == {
        "commit": "a" * 40,
        "dirty": False,
    }

    raw_envelope = artifacts.envelope_path.read_text(encoding="utf-8")
    assert "NaN" not in raw_envelope
    assert "Infinity" not in raw_envelope
    envelope = json.loads(raw_envelope)
    assert envelope["schema"] == "libreyolo.vlm-confidence-benchmark-run.v1"
    assert benchmark._RUN_IDENTIFIER.fullmatch(envelope["run_id"])
    assert benchmark._RUN_IDENTIFIER.fullmatch(envelope["process_id"])
    assert envelope["request"] == {
        "dataset_yaml": str(dataset.resolve()),
        "seed": 17,
        "model_family": "qwen3vl",
        "model_size": "2b",
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
        == "libreyolo.vlm-confidence-benchmark-context.v1"
    )
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
    report_identities = _install_run_fakes(monkeypatch, events)
    artifacts = benchmark.run_benchmark(_dataset(tmp_path), tmp_path / "run-a")
    envelope = json.loads(artifacts.envelope_path.read_text(encoding="utf-8"))

    validated = benchmark._load_runner_envelope(artifacts.report_path, "run")

    assert validated.run_id == envelope["run_id"]
    assert validated.process_id == envelope["process_id"]
    assert validated.report_sha256 == envelope["report"]["sha256"]

    tampered = json.loads(json.dumps(envelope))
    tampered["execution_context"]["determinism"]["cudnn_benchmark"] = True
    benchmark._write_json_atomic(artifacts.envelope_path, tampered)
    with pytest.raises(VLMConfidenceReportError, match="cudnn_benchmark"):
        benchmark._load_runner_envelope(artifacts.report_path, "run")

    tampered = json.loads(json.dumps(envelope))
    tampered["report"]["sha256"] = "0" * 64
    benchmark._write_json_atomic(artifacts.envelope_path, tampered)
    with pytest.raises(VLMConfidenceReportError, match="companion report bytes"):
        benchmark._load_runner_envelope(artifacts.report_path, "run")

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
    ):
        tampered = json.loads(json.dumps(envelope))
        mutate(tampered)
        benchmark._write_json_atomic(artifacts.envelope_path, tampered)
        with pytest.raises(VLMConfidenceReportError, match=expected_error):
            benchmark._load_runner_envelope(artifacts.report_path, "run")

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
    ):
        benchmark._write_json_atomic(artifacts.envelope_path, envelope)
        identity["benchmark_config"] = json.loads(json.dumps(original_config))
        mutate(identity["benchmark_config"])
        with pytest.raises(VLMConfidenceReportError, match=expected_error):
            benchmark._load_runner_envelope(artifacts.report_path, "run")


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

    with pytest.raises(VLMConfidenceReportError, match="invalid JSON value"):
        benchmark._load_runner_envelope(report, "run")


def test_run_benchmark_cleans_staging_artifacts_after_failure(tmp_path, monkeypatch):
    events = []
    _install_run_fakes(
        monkeypatch,
        events,
        failure=RuntimeError("offline failure"),
    )
    output = tmp_path / "failed-run"

    with pytest.raises(RuntimeError, match="offline failure"):
        benchmark.run_benchmark(_dataset(tmp_path), output)

    assert not output.exists()
    assert not list(tmp_path.glob(".failed-run.tmp-*"))
    assert not (tmp_path / ".failed-run.lock").exists()


def test_run_benchmark_rejects_code_drift_during_generation(tmp_path, monkeypatch):
    events = []
    _install_run_fakes(monkeypatch, events)
    contexts = iter(
        [
            {"commit": "a" * 40, "dirty": False},
            {"commit": "b" * 40, "dirty": False},
        ]
    )
    monkeypatch.setattr(benchmark, "_git_context", lambda: next(contexts))
    output = tmp_path / "drifted-run"

    with pytest.raises(RuntimeError, match="changed during execution"):
        benchmark.run_benchmark(_dataset(tmp_path), output)

    assert not output.exists()
    assert not list(tmp_path.glob(".drifted-run.tmp-*"))
    assert not (tmp_path / ".drifted-run.lock").exists()


def test_run_benchmark_refuses_overwrite_before_git_or_model(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path)
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
        benchmark.run_benchmark(dataset, output)

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_run_benchmark_refuses_broken_output_symlink(tmp_path, monkeypatch):
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
        benchmark.run_benchmark(_dataset(tmp_path), output)

    assert output.is_symlink()
    assert not target.exists()


def test_run_benchmark_refuses_output_inside_git_worktree(tmp_path, monkeypatch):
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
        benchmark.run_benchmark(_dataset(tmp_path), tmp_path / "runs" / "run-a")


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
        benchmark.run_benchmark(_dataset(tmp_path), tmp_path / "output")


@pytest.mark.parametrize("override", ["1", "true", "YES", "on"])
def test_run_benchmark_refuses_faster_coco_env_before_model(
    override, tmp_path, monkeypatch
):
    events = []
    _install_run_fakes(monkeypatch, events)
    monkeypatch.setenv("LIBREYOLO_FASTER_COCO_EVAL", override)

    with pytest.raises(benchmark.BenchmarkInputError, match="faster-coco-eval"):
        benchmark.run_benchmark(_dataset(tmp_path), tmp_path / "run-a")

    assert events == []


def test_parse_cli_args_covers_run_and_compare_contracts():
    run = benchmark.parse_cli_args(
        [
            "run",
            "--data",
            "dataset.yaml",
            "--output-root",
            "run-a",
            "--seed",
            "19",
            "--device",
            "cuda:1",
        ]
    )
    assert run.mode == "run"
    assert run.data == Path("dataset.yaml")
    assert run.output_root == Path("run-a")
    assert run.seed == 19
    assert run.device == "cuda:1"

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
                "--data",
                "dataset.yaml",
                "--output-root",
                "run-a",
                "--model-size",
                "4b",
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
    _install_run_fakes(
        monkeypatch,
        events,
        metrics={"metric/undefined": float("nan")},
    )
    output = tmp_path / "run-a"

    code = benchmark.main(
        [
            "run",
            "--data",
            str(_dataset(tmp_path)),
            "--output-root",
            str(output),
        ]
    )

    assert code == benchmark.EXIT_OK
    status = _status(capsys)
    assert status["status"] == "ok"
    assert status["code"] == benchmark.EXIT_OK
    assert status["nonfinite_metrics"] == ["metric/undefined"]
    assert Path(status["report"]).is_file()


def test_run_cli_routes_library_stdout_away_from_json_status(
    tmp_path, monkeypatch, capsys
):
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

    code = benchmark.main(
        ["run", "--data", "dataset.yaml", "--output-root", str(output)]
    )

    assert code == benchmark.EXIT_OK
    captured = capsys.readouterr()
    assert "library chatter" in captured.err
    assert json.loads(captured.out)["status"] == "ok"
    assert captured.out.count("\n") == 1


def test_run_cli_reports_overwrite_with_distinct_exit_code(tmp_path, capsys):
    output = tmp_path / "occupied"
    output.mkdir()

    code = benchmark.main(
        [
            "run",
            "--data",
            str(_dataset(tmp_path)),
            "--output-root",
            str(output),
        ]
    )

    assert code == benchmark.EXIT_OUTPUT_EXISTS
    status = _status(capsys)
    assert status["error"]["kind"] == "output_exists"


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
    _install_run_fakes(
        monkeypatch,
        events,
        failure=RuntimeError("generation failed"),
    )
    output = tmp_path / "failed"

    code = benchmark.main(
        [
            "run",
            "--data",
            str(_dataset(tmp_path)),
            "--output-root",
            str(output),
        ]
    )

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
        ),
    )

    with pytest.raises(VLMConfidenceReportError, match="fresh Python processes"):
        benchmark.compare_benchmarks("first.json", "second.json")


@pytest.mark.parametrize("value", [True, -1, 2**32, 1.5, "1.5"])
def test_run_benchmark_rejects_invalid_seed_before_side_effects(value, tmp_path):
    with pytest.raises(benchmark.BenchmarkInputError):
        benchmark.run_benchmark(_dataset(tmp_path), tmp_path / "output", seed=value)


def test_metric_normalization_rejects_non_numeric_values():
    with pytest.raises(TypeError, match="must be numeric"):
        benchmark._normalized_metrics({"metric": object()})
    with pytest.raises(TypeError, match="must be numeric"):
        benchmark._normalized_metrics({"metric": True})
    assert benchmark._normalized_metrics({"metric": math.nan}) == (
        {"metric": None},
        ("metric",),
    )
