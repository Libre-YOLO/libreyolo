"""Unit tests for Comet, ClearML, Neptune and DVCLive loggers."""

from __future__ import annotations

import builtins
import pickle
from typing import ClassVar

import pytest

from libreyolo.training.callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from libreyolo.training.loggers import (
    ClearMLLogger,
    CometLogger,
    DVCLiveLogger,
    NeptuneLogger,
    resolve_loggers,
)
from libreyolo.training.loggers import (
    clearml_logger as clearml_module,
)
from libreyolo.training.loggers import (
    comet_logger as comet_module,
)
from libreyolo.training.loggers import (
    dvclive_logger as dvclive_module,
)
from libreyolo.training.loggers import (
    neptune_logger as neptune_module,
)

pytestmark = pytest.mark.unit


def _start_event(save_dir: str, *, start_epoch: int = 1) -> TrainStartEvent:
    return TrainStartEvent(
        start_epoch=start_epoch,
        total_epochs=2,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir=save_dir,
        config={"epochs": 2, "lr0": 0.01, "data": None},
    )


def _epoch_event(save_dir: str, *, epoch: int = 1) -> TrainEpochEvent:
    return TrainEpochEvent(
        epoch=epoch,
        total_epochs=2,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir=save_dir,
        train_loss=1.5,
        train_loss_items={"box": 0.2},
        lr={"group0": 0.01},
        val_metrics={"metrics/mAP50": 0.6},
        validated=True,
        is_best=True,
        current_metric=0.6,
        current_metric_name="metrics/mAP50",
        best_metric=0.6,
        best_metric_name="metrics/mAP50",
        best_epoch=1,
        epoch_seconds=2.5,
    )


def _end_event(save_dir: str) -> TrainEndEvent:
    return TrainEndEvent(
        total_epochs=2,
        completed_epochs=2,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir=save_dir,
        final_loss=1.0,
        best_metric=0.6,
        best_epoch=1,
        total_seconds=5.0,
        results={"final_loss": 1.0},
    )


def _exception_event(save_dir: str) -> TrainExceptionEvent:
    exc = RuntimeError("boom")
    return TrainExceptionEvent(
        epoch=1,
        total_epochs=2,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir=save_dir,
        exception=exc,
        exception_type="RuntimeError",
        exception_message="boom",
        elapsed_seconds=1.0,
    )


class FakeCometExperiment:
    def __init__(self):
        self.calls = []

    def set_name(self, name):
        self.calls.append(("set_name", name))

    def log_parameters(self, params):
        self.calls.append(("log_parameters", params))

    def log_metrics(self, metrics, step=None, epoch=None):
        self.calls.append(("log_metrics", metrics, step, epoch))

    def log_others(self, values):
        self.calls.append(("log_others", values))

    def log_asset(self, path, file_name=None):
        self.calls.append(("log_asset", path, file_name))

    def end(self):
        self.calls.append(("end",))


class FakeComet:
    def __init__(self):
        self.start_kwargs = None
        self.experiment = FakeCometExperiment()

    def start(self, **kwargs):
        self.start_kwargs = kwargs
        return self.experiment


class FakeClearMLLogger:
    def __init__(self):
        self.scalars = []
        self.single_values = []

    def report_scalar(self, **kwargs):
        self.scalars.append(kwargs)

    def report_single_value(self, name, value):
        self.single_values.append((name, value))


class FakeClearMLTask:
    instances: ClassVar[list] = []

    def __init__(self, init_kwargs):
        self.init_kwargs = init_kwargs
        self.logger = FakeClearMLLogger()
        self.configurations = []
        self.artifacts = []
        self.closed = False
        self.failed = None
        self.__class__.instances.append(self)

    @classmethod
    def init(cls, **kwargs):
        return cls(kwargs)

    def get_logger(self):
        return self.logger

    def connect_configuration(self, config, **kwargs):
        self.configurations.append((config, kwargs))

    def upload_artifact(self, **kwargs):
        self.artifacts.append(kwargs)

    def close(self):
        self.closed = True

    def mark_failed(self, **kwargs):
        self.failed = kwargs


class FakeNeptuneRun:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.configs = []
        self.metrics = []
        self.tags = []
        self.files = []
        self.closed = False
        self.terminated = False
        self.__class__.instances.append(self)

    def log_configs(self, **kwargs):
        self.configs.append(kwargs)

    def log_metrics(self, **kwargs):
        self.metrics.append(kwargs)

    def add_tags(self, **kwargs):
        self.tags.append(kwargs)

    def assign_files(self, **kwargs):
        self.files.append(kwargs)

    def close(self):
        self.closed = True

    def terminate(self):
        self.terminated = True


class FakeDVCLive:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.params = []
        self.metrics = []
        self.summary = {}
        self.step = 0
        self.artifacts = []
        self.summary_calls = 0
        self.ended = False
        self.__class__.instances.append(self)

    def log_params(self, params):
        self.params.append(params)

    def log_metric(self, name, value):
        self.metrics.append((name, value, self.step))

    def make_summary(self):
        self.summary_calls += 1

    def log_artifact(self, path, **kwargs):
        self.artifacts.append((path, kwargs))

    def end(self):
        self.ended = True


@pytest.fixture
def fake_backends(monkeypatch):
    comet = FakeComet()
    FakeClearMLTask.instances = []
    FakeNeptuneRun.instances = []
    FakeDVCLive.instances = []
    monkeypatch.setattr(comet_module, "_import_comet", lambda: comet)
    monkeypatch.setattr(clearml_module, "_import_clearml_task", lambda: FakeClearMLTask)
    monkeypatch.setattr(neptune_module, "_import_neptune_run", lambda: FakeNeptuneRun)
    monkeypatch.setattr(dvclive_module, "_import_dvclive", lambda: FakeDVCLive)
    return comet


def _write_artifacts(tmp_path):
    (tmp_path / "results.csv").write_text("epoch\n1\n")
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "best.pt").write_bytes(b"00")


def test_resolve_all_extended_logger_names_and_dvc_alias(fake_backends):
    resolved = resolve_loggers(["comet", "clearml", "neptune", "dvc", "dvclive"])
    assert [type(item) for item in resolved] == [
        CometLogger,
        ClearMLLogger,
        NeptuneLogger,
        DVCLiveLogger,
        DVCLiveLogger,
    ]


@pytest.mark.parametrize(
    ("module", "helper_name", "package_name", "extra_name"),
    [
        (comet_module, "_import_comet", "comet_ml", "comet"),
        (clearml_module, "_import_clearml_task", "clearml", "clearml"),
        (neptune_module, "_import_neptune_run", "neptune_scale", "neptune"),
        (dvclive_module, "_import_dvclive", "dvclive", "dvclive"),
    ],
)
def test_extended_logger_import_errors_name_install_extra(
    monkeypatch, module, helper_name, package_name, extra_name
):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == package_name:
            raise ImportError(f"blocked {package_name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=rf"libreyolo\[{extra_name}\]"):
        getattr(module, helper_name)()


def test_comet_logger_full_lifecycle(fake_backends, tmp_path):
    _write_artifacts(tmp_path)
    logger = CometLogger(project_name="project", workspace="team", log_checkpoints=True)
    logger.on_train_start(_start_event(str(tmp_path)))
    logger.on_train_epoch_end(_epoch_event(str(tmp_path)))
    logger.on_train_end(_end_event(str(tmp_path)))

    assert fake_backends.start_kwargs == {
        "project_name": "project",
        "mode": "create",
        "workspace": "team",
    }
    calls = fake_backends.experiment.calls
    assert ("set_name", "yolo9s-detect") in calls
    metrics = next(call for call in calls if call[0] == "log_metrics")
    assert metrics[1]["val/mAP50"] == 0.6
    assert metrics[2:] == (1, 1)
    assert any(call[0] == "log_asset" and call[2] == "best.pt" for call in calls)
    assert calls[-1] == ("end",)


def test_clearml_logger_full_lifecycle(fake_backends, tmp_path):
    _write_artifacts(tmp_path)
    logger = ClearMLLogger(tags=["nightly"], log_checkpoints=True)
    logger.on_train_start(_start_event(str(tmp_path)))
    logger.on_train_epoch_end(_epoch_event(str(tmp_path)))
    logger.on_train_end(_end_event(str(tmp_path)))

    task = FakeClearMLTask.instances[0]
    assert task.init_kwargs["task_name"] == "yolo9s-detect"
    assert task.init_kwargs["reuse_last_task_id"] is False
    assert task.configurations[0][1]["ignore_remote_overrides"] is True
    val_metric = next(
        item
        for item in task.logger.scalars
        if item["title"] == "val" and item["series"] == "mAP50"
    )
    assert val_metric["value"] == 0.6
    assert any(item["name"] == "best.pt" for item in task.artifacts)
    assert task.closed is True


def test_clearml_marks_training_exception_failed(fake_backends, tmp_path):
    logger = ClearMLLogger()
    logger.on_train_start(_start_event(str(tmp_path)))
    logger.on_train_exception(_exception_event(str(tmp_path)))

    task = FakeClearMLTask.instances[0]
    assert task.failed["status_reason"] == "RuntimeError"
    assert task.failed["status_message"] == "boom"


def test_neptune_logger_uses_current_scale_api(fake_backends, tmp_path):
    _write_artifacts(tmp_path)
    logger = NeptuneLogger(
        project="team/project", tags=["nightly"], log_checkpoints=True
    )
    logger.on_train_start(_start_event(str(tmp_path)))
    logger.on_train_epoch_end(_epoch_event(str(tmp_path)))
    logger.on_train_end(_end_event(str(tmp_path)))

    run = FakeNeptuneRun.instances[0]
    assert run.kwargs["project"] == "team/project"
    assert run.kwargs["enable_console_log_capture"] is False
    assert run.configs[0]["flatten"] is True
    assert run.metrics[0]["data"]["train/loss"] == 1.5
    assert run.metrics[0]["step"] == 1
    assert run.tags == [{"tags": ("nightly",)}]
    assert any("artifacts/best.pt" in item["files"] for item in run.files)
    assert run.closed is True


def test_neptune_backend_failure_terminates_run(fake_backends, tmp_path):
    logger = NeptuneLogger()
    logger.on_train_start(_start_event(str(tmp_path)))
    run = FakeNeptuneRun.instances[0]

    def explode(**kwargs):
        raise ConnectionError("server down")

    run.log_metrics = explode
    logger.on_train_epoch_end(_epoch_event(str(tmp_path)))
    assert run.terminated is True


def test_dvclive_safe_defaults_resume_and_lifecycle(fake_backends, tmp_path):
    _write_artifacts(tmp_path)
    logger = DVCLiveLogger(log_checkpoints=True)
    logger.on_train_start(_start_event(str(tmp_path), start_epoch=2))
    logger.on_train_epoch_end(_epoch_event(str(tmp_path), epoch=2))
    logger.on_train_end(_end_event(str(tmp_path)))

    live = FakeDVCLive.instances[0]
    assert live.kwargs["dir"] == str(tmp_path / "dvclive")
    assert live.kwargs["resume"] is True
    assert live.kwargs["save_dvc_exp"] is False
    assert live.kwargs["dvcyaml"] is None
    assert live.params[0]["data"] == "None"
    assert ("train/loss", 1.5, 2) in live.metrics
    assert ("train/loss.box", 0.2, 2) in live.metrics
    assert live.summary_calls == 1
    assert live.summary == {"best_metric": 0.6, "best_epoch": 1}
    assert live.artifacts[0][1]["cache"] is False
    assert live.ended is True


def test_extended_logger_instances_are_picklable(fake_backends):
    instances = (
        CometLogger(),
        ClearMLLogger(),
        NeptuneLogger(),
        DVCLiveLogger(),
    )
    for instance in instances:
        assert pickle.loads(pickle.dumps(instance)) is not None
