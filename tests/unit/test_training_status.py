"""Unit tests for the live training status writer and the monitor server."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from libreyolo.training.artifacts import TrainingStatusCallback
from libreyolo.training.callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from libreyolo.ui import train_monitor
from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit


def _start(save_dir, *, start_epoch=1):
    return TrainStartEvent(
        start_epoch=start_epoch,
        total_epochs=4,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir=str(save_dir),
        config={"epochs": 4},
    )


def _epoch(save_dir, *, epoch=1, metric=0.5, best_epoch=1):
    return TrainEpochEvent(
        epoch=epoch,
        total_epochs=4,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir=str(save_dir),
        train_loss=2.0 - epoch * 0.1,
        train_loss_items={"box": 0.2, "cls": 0.3},
        lr={"group0": 0.01},
        val_metrics={"metrics/mAP50-95": metric},
        validated=True,
        is_best=True,
        current_metric=metric,
        current_metric_name="metrics/mAP50-95",
        best_metric=metric,
        best_metric_name="metrics/mAP50-95",
        best_epoch=best_epoch,
        epoch_seconds=1.0,
    )


def _end(save_dir):
    return TrainEndEvent(
        total_epochs=4,
        completed_epochs=4,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir=str(save_dir),
        final_loss=1.6,
        best_metric=0.62,
        best_epoch=3,
        total_seconds=4.0,
        results={
            "best_checkpoint": str(save_dir / "weights" / "best.pt"),
            "last_checkpoint": str(save_dir / "weights" / "last.pt"),
        },
    )


def _load(save_dir):
    return json.loads((save_dir / "status.json").read_text())


def test_status_running_then_completed(tmp_path):
    cb = TrainingStatusCallback()
    cb.on_train_start(_start(tmp_path))
    running = _load(tmp_path)
    assert running["state"] == "running"
    assert running["schema_version"] == 2
    assert running["total_epochs"] == 4
    assert running["pid"] > 0

    cb.on_train_epoch_end(_epoch(tmp_path, epoch=1, metric=0.5))
    mid = _load(tmp_path)
    assert mid["state"] == "running"
    assert mid["current_epoch"] == 0
    assert mid["completed_epochs"] == 1
    assert mid["progress"] == pytest.approx(0.25)
    assert mid["metrics"]["mAP50-95"] == pytest.approx(0.5)
    # ETA = mean epoch time (1.0s) * remaining (3) = 3.0
    assert mid["eta_seconds"] == pytest.approx(3.0, abs=0.01)

    cb.on_train_end(_end(tmp_path))
    done = _load(tmp_path)
    assert done["state"] == "completed"
    assert done["best_metric"] == pytest.approx(0.62)
    assert done["best_epoch"] == 2
    assert done["checkpoints"]["best"].endswith("best.pt")


def test_status_failed_records_error(tmp_path):
    cb = TrainingStatusCallback()
    cb.on_train_start(_start(tmp_path))
    cb.on_train_epoch_end(_epoch(tmp_path, epoch=1))
    exc = RuntimeError("CUDA out of memory")
    cb.on_train_exception(
        TrainExceptionEvent(
            epoch=2,
            total_epochs=4,
            model_family="yolo9",
            model_size="s",
            task="detect",
            save_dir=str(tmp_path),
            exception=exc,
            exception_type="RuntimeError",
            exception_message="CUDA out of memory",
            elapsed_seconds=2.0,
        )
    )
    failed = _load(tmp_path)
    assert failed["state"] == "failed"
    assert failed["current_epoch"] == 1
    assert failed["completed_epochs"] == 1
    assert failed["progress"] == pytest.approx(0.25)
    assert failed["error"]["type"] == "RuntimeError"
    assert "out of memory" in failed["error"]["message"]


def test_status_atomic_write_is_valid_json_every_time(tmp_path):
    """status.json must always parse; never a half-written file."""
    cb = TrainingStatusCallback(write_log=False)
    cb.on_train_start(_start(tmp_path))
    for e in range(1, 5):
        cb.on_train_epoch_end(_epoch(tmp_path, epoch=e))
        json.loads((tmp_path / "status.json").read_text())  # raises if corrupt


def test_log_teeing_captures_libreyolo_output(tmp_path):
    cb = TrainingStatusCallback()
    cb.on_train_start(_start(tmp_path))
    logging.getLogger("libreyolo").info("epoch 1 done, mAP=0.5")
    cb.on_train_end(_end(tmp_path))
    log = (tmp_path / "train.log").read_text()
    assert "epoch 1 done" in log
    # handler must be detached after the run (no leak onto the global logger)
    assert cb._log_handler is None


def test_monitor_reads_status_and_results_csv_fallback(tmp_path):
    """With no metrics.jsonl, the monitor falls back to the family results.csv."""
    cb = TrainingStatusCallback(write_log=False)
    cb.on_train_start(_start(tmp_path))  # writes status.json, not metrics.jsonl

    # results.csv is written by TrainingArtifactsCallback; emulate a couple rows.
    (tmp_path / "results.csv").write_text(
        "epoch,train/loss,metrics/mAP50-95\n1,2.0,0.5\n2,1.9,0.55\n"
    )
    assert not (tmp_path / "metrics.jsonl").exists()

    assert train_monitor._read_status(tmp_path)["state"] == "running"
    metrics = train_monitor._read_metrics(tmp_path)
    assert "metrics/mAP50-95" in metrics["columns"]
    assert metrics["rows"][1]["metrics/mAP50-95"] == pytest.approx(0.55)


def test_metrics_jsonl_written_and_read(tmp_path):
    """Every family gets a universal, chart-ready metrics.jsonl history."""
    cb = TrainingStatusCallback(write_log=False)
    cb.on_train_start(_start(tmp_path))
    cb.on_train_epoch_end(_epoch(tmp_path, epoch=1, metric=0.5))
    cb.on_train_epoch_end(_epoch(tmp_path, epoch=2, metric=0.55))

    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2

    parsed = train_monitor._read_metrics(tmp_path)
    assert "epoch" in parsed["columns"]
    assert "train/loss" in parsed["columns"]
    assert "metrics/mAP50-95" in parsed["columns"]
    assert [row["epoch"] for row in parsed["rows"]] == [1, 2]
    assert parsed["rows"][1]["metrics/mAP50-95"] == pytest.approx(0.55)


def test_metrics_jsonl_reset_on_fresh_run(tmp_path):
    cb = TrainingStatusCallback(write_log=False)
    cb.on_train_start(_start(tmp_path))
    cb.on_train_epoch_end(_epoch(tmp_path, epoch=1))
    # A new fresh run (start_epoch=1) must clear stale history.
    cb.on_train_start(_start(tmp_path, start_epoch=1))
    cb.on_train_epoch_end(_epoch(tmp_path, epoch=1))
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_atomic_status_write_retries_transient_reader_sharing(monkeypatch, tmp_path):
    from libreyolo.training import artifacts

    real_replace = artifacts.os.replace
    calls = 0

    def transient_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("injected sharing violation")
        return real_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", transient_replace)
    artifacts._atomic_write_json(tmp_path / "status.json", {"state": "running"})

    assert calls == 3
    assert json.loads((tmp_path / "status.json").read_text()) == {
        "state": "running"
    }


@pytest.mark.skipif(os.name != "nt", reason="Windows reader sharing semantics")
def test_resume_waits_for_monitor_readers_before_rewriting_status_and_metrics(
    tmp_path,
):
    cb = TrainingStatusCallback(write_log=False)
    cb.on_train_start(_start(tmp_path))
    for epoch in range(1, 5):
        cb.on_train_epoch_end(_epoch(tmp_path, epoch=epoch))

    readers = [
        open(tmp_path / "status.json", encoding="utf-8"),
        open(tmp_path / "metrics.jsonl", encoding="utf-8"),
    ]

    def release_readers():
        time.sleep(0.05)
        for reader in readers:
            reader.close()

    release = threading.Thread(target=release_readers)
    release.start()
    try:
        cb.on_train_start(_start(tmp_path, start_epoch=3))
    finally:
        for reader in readers:
            if not reader.closed:
                reader.close()
        release.join(timeout=2)

    resumed = _load(tmp_path)
    assert resumed["state"] == "running"
    assert resumed["completed_epochs"] == 2
    assert resumed["current_epoch"] is None
    cb.on_train_epoch_end(_epoch(tmp_path, epoch=3))
    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in rows] == [1, 2, 3]


@pytest.mark.skipif(os.name != "nt", reason="Windows reader sharing semantics")
def test_fresh_status_start_waits_for_monitor_reader_before_resetting_log(tmp_path):
    cb = TrainingStatusCallback()
    cb.on_train_start(_start(tmp_path))
    logging.getLogger("libreyolo").info("stale log line")
    cb._close_log()
    reader = open(tmp_path / "train.log", encoding="utf-8")

    def release_reader():
        time.sleep(0.05)
        reader.close()

    release = threading.Thread(target=release_reader)
    release.start()
    try:
        cb.on_train_start(_start(tmp_path))
    finally:
        if not reader.closed:
            reader.close()
        release.join(timeout=2)
    logging.getLogger("libreyolo").info("fresh log line")
    cb._close_log()

    log = (tmp_path / "train.log").read_text()
    assert "fresh log line" in log
    assert "stale log line" not in log


def test_metrics_jsonl_trims_stale_resume_rows(tmp_path):
    cb = TrainingStatusCallback(write_log=False)
    cb.on_train_start(_start(tmp_path))
    for epoch in range(1, 5):
        cb.on_train_epoch_end(_epoch(tmp_path, epoch=epoch))

    cb.on_train_start(_start(tmp_path, start_epoch=3))
    cb.on_train_epoch_end(_epoch(tmp_path, epoch=3))

    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in rows] == [1, 2, 3]


def test_resume_status_keeps_absolute_progress_through_train_end(tmp_path):
    cb = TrainingStatusCallback(write_log=False)
    cb.on_train_start(_start(tmp_path, start_epoch=3))
    started = _load(tmp_path)
    assert started["completed_epochs"] == 2
    assert started["progress"] == pytest.approx(0.5)

    cb.on_train_epoch_end(_epoch(tmp_path, epoch=3, best_epoch=3))
    cb.on_train_epoch_end(_epoch(tmp_path, epoch=4, best_epoch=4))
    end = _end(tmp_path)
    end = TrainEndEvent(
        total_epochs=end.total_epochs,
        completed_epochs=2,  # current invocation only, as emitted by BaseTrainer
        model_family=end.model_family,
        model_size=end.model_size,
        task=end.task,
        save_dir=end.save_dir,
        final_loss=end.final_loss,
        best_metric=end.best_metric,
        best_epoch=end.best_epoch,
        total_seconds=end.total_seconds,
        results=end.results,
    )
    cb.on_train_end(end)

    done = _load(tmp_path)
    assert done["completed_epochs"] == 4
    assert done["current_epoch"] == 3
    assert done["progress"] == pytest.approx(1.0)


def test_concurrent_training_logs_are_isolated_and_level_restores(tmp_path):
    lib_logger = logging.getLogger("libreyolo")
    original_level = lib_logger.level
    lib_logger.setLevel(logging.WARNING)
    ready = threading.Barrier(3)
    release_first = threading.Event()
    first_closed = threading.Event()
    errors = []

    def run(name: str, wait_for_release: bool):
        cb = TrainingStatusCallback()
        run_dir = tmp_path / name
        try:
            cb.on_train_start(_start(run_dir))
            lib_logger.info("message-%s", name)
            ready.wait(timeout=2)
            if wait_for_release:
                assert release_first.wait(timeout=2)
            cb.on_train_end(_end(run_dir))
            if not wait_for_release:
                first_closed.set()
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)
            cb._close_log()

    first = threading.Thread(target=run, args=("first", False))
    second = threading.Thread(target=run, args=("second", True))
    try:
        first.start()
        second.start()
        ready.wait(timeout=2)
        assert lib_logger.level == logging.INFO
        assert first_closed.wait(timeout=2)
        assert lib_logger.level == logging.INFO
        release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)
        assert not first.is_alive() and not second.is_alive()
        assert not errors
        assert lib_logger.level == logging.WARNING
        first_log = (tmp_path / "first" / "train.log").read_text()
        second_log = (tmp_path / "second" / "train.log").read_text()
        assert "message-first" in first_log and "message-second" not in first_log
        assert "message-second" in second_log and "message-first" not in second_log
    finally:
        release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)
        lib_logger.setLevel(original_level)


def test_training_run_directory_reservation_is_atomic(tmp_path):
    trainers = [
        SimpleNamespace(
            config=SimpleNamespace(
                project=str(tmp_path / "runs"), name="exp", exist_ok=False
            )
        )
        for _ in range(12)
    ]
    barrier = threading.Barrier(len(trainers))

    def reserve(trainer):
        barrier.wait(timeout=5)
        return BaseTrainer._get_save_dir(trainer)

    with ThreadPoolExecutor(max_workers=len(trainers)) as pool:
        paths = list(pool.map(reserve, trainers))

    assert len(set(paths)) == len(trainers)
    assert all(path.is_dir() for path in paths)
    assert {path.name for path in paths} == {
        "exp",
        *(f"exp{index}" for index in range(2, len(trainers) + 1)),
    }


def test_training_run_directory_supports_nested_names(tmp_path):
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            project=str(tmp_path / "runs"), name="sweep/exp", exist_ok=False
        )
    )

    first = BaseTrainer._get_save_dir(trainer)
    second = BaseTrainer._get_save_dir(trainer)

    assert first == tmp_path / "runs" / "sweep" / "exp"
    assert second == tmp_path / "runs" / "sweep" / "exp2"


def test_training_run_directory_preserves_empty_name(tmp_path):
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            project=str(tmp_path / "runs"), name="", exist_ok=False
        )
    )

    first = BaseTrainer._get_save_dir(trainer)
    second = BaseTrainer._get_save_dir(trainer)

    assert first == tmp_path / "runs"
    assert second == tmp_path / "runs2"
    assert first.is_dir() and second.is_dir()


def test_training_run_directory_reuses_resume_target(tmp_path):
    resume_dir = tmp_path / "original" / "exp"
    trainer = SimpleNamespace(
        _resume_save_dir=resume_dir,
        config=SimpleNamespace(
            project=str(tmp_path / "other"), name="new", exist_ok=False
        ),
    )

    assert BaseTrainer._get_save_dir(trainer) == resume_dir
    assert resume_dir.is_dir()
    assert not (tmp_path / "other").exists()


def test_metrics_jsonl_skips_torn_line(tmp_path):
    (tmp_path / "metrics.jsonl").write_text(
        '{"epoch": 0, "train/loss": 2.0}\n{"epoch": 1, "train/loss": 1.9}\n{"epoch": 2, "trai'
    )
    parsed = train_monitor._read_metrics(tmp_path)
    assert len(parsed["rows"]) == 2  # torn final line dropped


def test_monitor_missing_status_is_graceful(tmp_path):
    assert train_monitor._read_status(tmp_path)["state"] == "missing"
    assert train_monitor._read_metrics(tmp_path) == {"columns": [], "rows": []}
    assert train_monitor._read_log_tail(tmp_path) == ""


def test_find_latest_run_prefers_status(tmp_path):
    old = tmp_path / "runs" / "exp"
    new = tmp_path / "runs" / "exp2"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "status.json").write_text("{}")
    (new / "status.json").write_text("{}")
    # make `new` newer
    import os
    import time

    now = time.time()
    os.utime(old / "status.json", (now - 100, now - 100))
    os.utime(new / "status.json", (now, now))
    assert train_monitor.find_latest_run(tmp_path / "runs") == new


def test_image_path_sandbox_blocks_escape(tmp_path):
    (tmp_path / "val.png").write_bytes(b"x")
    assert train_monitor._resolve_in_run(tmp_path, "val.png") is not None
    assert train_monitor._resolve_in_run(tmp_path, "../secret.txt") is None
    assert train_monitor._resolve_in_run(tmp_path, "../../etc/passwd") is None


def _make_run(root, rel, state="running"):
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps({"state": state, "save_dir": str(d)}))
    return d


def test_discover_runs_finds_all_and_newest_first(tmp_path):
    import os
    import time

    root = tmp_path / "runs"
    a = _make_run(root, "train/exp")
    b = _make_run(root, "train/exp2")
    c = _make_run(root, "detect/predict")  # different subtree
    now = time.time()
    os.utime(a / "status.json", (now - 300, now - 300))
    os.utime(b / "status.json", (now, now))
    os.utime(c / "status.json", (now - 100, now - 100))

    runs = train_monitor.discover_runs(root)
    assert set(runs) == {a, b, c}
    assert runs[0] == b  # newest first


def test_discover_runs_includes_root_when_root_is_a_run(tmp_path):
    _make_run(tmp_path, ".")  # tmp_path itself is a run
    runs = train_monitor.discover_runs(tmp_path)
    assert tmp_path in runs


def test_run_id_and_resolution_roundtrip(tmp_path):
    root = tmp_path / "runs"
    run = _make_run(root, "train/exp2")
    run_id = train_monitor.run_id_for(root, run)
    assert run_id == "train/exp2"
    assert train_monitor._run_dir(root, run_id) == run.resolve()


def test_run_dir_rejects_escape_and_unknown(tmp_path):
    root = tmp_path / "runs"
    _make_run(root, "train/exp")
    assert train_monitor._run_dir(root, "../../etc") is None
    assert train_monitor._run_dir(root, "train/does_not_exist") is None
    assert train_monitor._run_dir(root, "") is None
    assert train_monitor._run_dir(root, None) is None


def test_run_summary_reports_state(tmp_path):
    root = tmp_path / "runs"
    run = _make_run(root, "train/exp", state="completed")
    (run / "status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "model_family": "yolo9",
                "model_size": "t",
                "task": "detect",
                "progress": 1.0,
                "total_epochs": 8,
                "best_metric": 0.5,
                "best_metric_name": "mAP50-95",
            }
        )
    )
    summ = train_monitor._run_summary(root, run)
    assert summ["id"] == "train/exp"
    assert summ["state"] == "completed"
    assert summ["model"] == "yolo9t"
    assert summ["best_metric"] == 0.5


def test_server_serves_multiple_runs_by_id(tmp_path):
    """One server, two runs, each addressable by its own ?run= id."""
    import threading
    import urllib.request

    root = tmp_path / "runs"
    _make_run(root, "train/a", state="running")
    _make_run(root, "train/b", state="completed")

    httpd, url = train_monitor.serve(root, port=0, open_browser=False)
    httpd.server_port = httpd.server_address[1]  # port=0 -> OS-assigned
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        runs = json.load(urllib.request.urlopen(base + "/api/runs", timeout=5))["runs"]
        ids = {r["id"] for r in runs}
        assert ids == {"train/a", "train/b"}

        sa = json.load(urllib.request.urlopen(base + "/api/status?run=train/a", timeout=5))
        sb = json.load(urllib.request.urlopen(base + "/api/status?run=train/b", timeout=5))
        assert sa["state"] == "running"
        assert sb["state"] == "completed"

        # unknown / escaping run ids are refused, not served
        code = urllib.request.urlopen(base + "/", timeout=5).status
        assert code == 200
        try:
            urllib.request.urlopen(base + "/api/status?run=../../etc", timeout=5)
            raise AssertionError("escape not blocked")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
