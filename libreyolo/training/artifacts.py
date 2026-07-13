"""Built-in training artifact writers."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils.logging import LoggerLevelLease, ThreadLogFilter
from .callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)

logger = logging.getLogger("libreyolo")

_FILE_SHARING_RETRY_SECONDS = 1.0
_FILE_SHARING_RETRY_INITIAL_SECONDS = 0.005
_FILE_SHARING_RETRY_MAX_SECONDS = 0.05


def _retry_transient_file_access(operation):
    """Retry bounded Windows-style sharing violations from short-lived readers."""
    deadline = time.monotonic() + _FILE_SHARING_RETRY_SECONDS
    delay = _FILE_SHARING_RETRY_INITIAL_SECONDS
    while True:
        try:
            return operation()
        except PermissionError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2.0, _FILE_SHARING_RETRY_MAX_SECONDS)


def _replace_with_retry(source, destination) -> None:
    _retry_transient_file_access(lambda: os.replace(source, destination))


def _unlink_with_retry(path: Path) -> None:
    _retry_transient_file_access(path.unlink)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write ``value`` to ``path`` atomically (tmp file + ``os.replace``).

    A reader (the monitor UI or a polling agent) therefore never observes a
    half-written file; it sees either the previous version or the new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(
                _json_safe(value), f, allow_nan=False, indent=2, sort_keys=True
            )
            f.write("\n")
        _replace_with_retry(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class TrainingArtifactsCallback:
    """Write durable training artifacts from normalized trainer events."""

    def __init__(
        self,
        *,
        enabled_families: Iterable[str] | None = None,
        results_name: str = "results.csv",
        summary_name: str = "summary.json",
    ):
        self.enabled_families = (
            {family.lower() for family in enabled_families}
            if enabled_families is not None
            else None
        )
        self.results_name = results_name
        self.summary_name = summary_name

    def on_train_start(self, event: TrainStartEvent) -> None:
        if not self._enabled(event):
            return

        save_dir = self._save_dir(event)
        if event.start_epoch <= 1:
            for filename in (self.results_name, self.summary_name):
                path = save_dir / filename
                if path.exists():
                    _unlink_with_retry(path)
        else:
            self._trim_csv_before_epoch(
                save_dir / self.results_name,
                start_epoch=event.start_epoch,
            )

    def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
        if not self._enabled(event):
            return

        self._append_csv_row(
            self._save_dir(event) / self.results_name,
            self._epoch_row(event),
        )

    def on_train_end(self, event: TrainEndEvent) -> None:
        if not self._enabled(event):
            return

        save_dir = self._save_dir(event)
        results_path = save_dir / self.results_name
        logged_epochs = self._read_logged_epochs(results_path)
        summary = {
            "total_epochs": event.total_epochs,
            "completed_epochs": max(event.completed_epochs, len(logged_epochs)),
            "invocation_completed_epochs": event.completed_epochs,
            "logged_epochs": logged_epochs,
            "model_family": event.model_family,
            "model_size": event.model_size,
            "task": event.task,
            "save_dir": event.save_dir,
            "final_loss": event.final_loss,
            "best_metric": event.best_metric,
            "best_epoch": event.best_epoch,
            "total_seconds": event.total_seconds,
            "checkpoints": {
                "best": event.results.get("best_checkpoint"),
                "last": event.results.get("last_checkpoint"),
            },
            "results_scope": "current_invocation",
            "results": dict(event.results),
        }
        self._write_json(save_dir / self.summary_name, summary)

    def on_train_exception(self, event: TrainExceptionEvent) -> None:
        return None

    def _enabled(
        self,
        event: TrainStartEvent | TrainEpochEvent | TrainEndEvent | TrainExceptionEvent,
    ) -> bool:
        if self.enabled_families is None:
            return True
        return event.model_family.lower() in self.enabled_families

    @staticmethod
    def _save_dir(
        event: TrainStartEvent | TrainEpochEvent | TrainEndEvent | TrainExceptionEvent,
    ) -> Path:
        save_dir = Path(event.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    @classmethod
    def _epoch_row(cls, event: TrainEpochEvent) -> dict[str, Any]:
        row: dict[str, Any] = {
            "epoch": event.epoch,
            "time": event.epoch_seconds,
            "train/loss": event.train_loss,
            "validated": event.validated,
            "is_best": event.is_best,
            "current_metric": event.current_metric,
            "current_metric_name": event.current_metric_name,
            "best_metric": event.best_metric,
            "best_metric_name": event.best_metric_name,
            "best_epoch": event.best_epoch,
        }

        for name, value in event.train_loss_items.items():
            row[cls._train_loss_column(name)] = value
        for name, value in event.val_metrics.items():
            row[cls._metric_column(name)] = value
        for name, value in event.lr.items():
            row[f"lr/{name}"] = value

        return row

    @staticmethod
    def _train_loss_column(name: str) -> str:
        normalized = name.strip().replace(" ", "_")
        if normalized.startswith("train/"):
            return normalized
        if normalized.endswith("_loss"):
            return f"train/{normalized}"
        return f"train/{normalized}_loss"

    @staticmethod
    def _metric_column(name: str) -> str:
        normalized = name.strip().replace(" ", "_")
        if "/" in normalized:
            return normalized
        return f"metrics/{normalized}"

    @classmethod
    def _append_csv_row(cls, path: Path, row: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized_row = {key: cls._csv_value(value) for key, value in row.items()}

        if not path.exists() or path.stat().st_size == 0:
            cls._write_csv(path, list(normalized_row), [normalized_row])
            return

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        new_columns = [key for key in normalized_row if key not in fieldnames]
        if new_columns:
            fieldnames.extend(new_columns)
            rows.append(normalized_row)
            cls._write_csv(path, fieldnames, rows)
            return

        cls._append_csv(path, fieldnames, normalized_row)

    @staticmethod
    def _write_csv(
        path: Path, fieldnames: list[str], rows: list[Mapping[str, Any]]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        try:
            with os.fdopen(fd, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            _replace_with_retry(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _append_csv(
        path: Path,
        fieldnames: list[str],
        row: Mapping[str, Any],
    ) -> None:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    @classmethod
    def _write_json(cls, path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(
                    cls._json_value(value),
                    f,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                f.write("\n")
            _replace_with_retry(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _read_logged_epochs(path: Path) -> list[int]:
        if not path.exists() or path.stat().st_size == 0:
            return []

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            epochs = []
            for row in reader:
                try:
                    epochs.append(int(row.get("epoch", "")))
                except (TypeError, ValueError):
                    continue
            return epochs

    @classmethod
    def _trim_csv_before_epoch(cls, path: Path, *, start_epoch: int) -> None:
        if not path.exists() or path.stat().st_size == 0:
            return

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            if "epoch" not in fieldnames:
                return

            rows = []
            for row in reader:
                try:
                    epoch = int(row.get("epoch", ""))
                except (TypeError, ValueError):
                    continue
                if epoch < start_epoch:
                    rows.append(row)

        cls._write_csv(path, fieldnames, rows)

    @classmethod
    def _json_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): cls._json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_value(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        if isinstance(value, bool):
            return int(value)
        return value


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TrainingStatusCallback:
    """Write a live, machine-readable ``status.json`` for every training run.

    Unlike :class:`TrainingArtifactsCallback` (which is family-gated and writes
    the per-epoch ``results.csv``), this callback is always on for every model
    family. It exists to serve two starved consumers of an agent-launched run:

    - an **agent** polling ``status.json`` answers "what epoch / is it alive /
      best metric so far / did it crash" in a few tokens instead of tailing a
      log, and
    - the ``libreyolo monitor`` **web UI**, which relays the same file to a
      browser so a human gets live feedback without the agent in the loop.

    The file is rewritten atomically on every epoch and carries ``state``
    (``running`` / ``completed`` / ``failed``), progress, an ETA derived from
    mean epoch time, the latest and best metrics, and, on failure, the
    exception message. It also tees the ``libreyolo`` console log to
    ``train.log`` in the run directory so the monitor can show the terminal.
    """

    def __init__(
        self,
        *,
        status_name: str = "status.json",
        metrics_name: str = "metrics.jsonl",
        log_name: str = "train.log",
        write_log: bool = True,
    ):
        self.status_name = status_name
        self.metrics_name = metrics_name
        self.log_name = log_name
        self.write_log = write_log
        self._start_time: float | None = None
        self._epoch_time_sum = 0.0
        self._epoch_time_count = 0
        self._base: dict[str, Any] = {}
        self._status: dict[str, Any] = {}
        self._start_completed_epochs = 0
        self._completed_epochs = 0
        self._log_handler: logging.Handler | None = None
        self._log_path: Path | None = None
        self._log_level_lease: LoggerLevelLease | None = None

    # -- events ------------------------------------------------------------

    def on_train_start(self, event: TrainStartEvent) -> None:
        self._close_log()
        save_dir = self._save_dir(event)
        self._start_time = time.time()
        self._epoch_time_sum = 0.0
        self._epoch_time_count = 0
        self._status = {}
        self._start_completed_epochs = max(int(event.start_epoch) - 1, 0)
        self._completed_epochs = self._start_completed_epochs
        self._base = {
            "schema_version": 2,
            "pid": os.getpid(),
            "model_family": event.model_family,
            "model_size": event.model_size,
            "task": event.task,
            "save_dir": event.save_dir,
            "total_epochs": event.total_epochs,
            "start_epoch": event.start_epoch,
            "started_at": _utcnow_iso(),
        }
        # A fresh run (not a resume) starts the universal metric history clean.
        if event.start_epoch <= 1:
            metrics_path = save_dir / self.metrics_name
            if metrics_path.exists():
                try:
                    _unlink_with_retry(metrics_path)
                except OSError:
                    logger.debug("Could not reset %s", self.metrics_name, exc_info=True)
        else:
            try:
                self._trim_metrics_before_epoch(
                    save_dir / self.metrics_name,
                    start_epoch=event.start_epoch,
                )
            except OSError:
                logger.debug("Could not trim %s", self.metrics_name, exc_info=True)
        self._open_log(save_dir, fresh=event.start_epoch <= 1)
        self._write(
            save_dir,
            state="running",
            completed_epochs=self._completed_epochs,
            current_epoch=None,
        )

    def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
        save_dir = self._save_dir(event)
        self._epoch_time_sum += event.epoch_seconds
        self._epoch_time_count += 1
        mean_epoch = self._epoch_time_sum / max(self._epoch_time_count, 1)
        # Public callback events number completed epochs from 1. status.json
        # keeps ``current_epoch`` zero-based for its UI/API consumers and stores
        # an absolute completed count, including epochs from a resumed run.
        completed = max(int(event.epoch), self._start_completed_epochs)
        self._completed_epochs = max(self._completed_epochs, completed)
        remaining = max(event.total_epochs - completed, 0)
        metrics = {
            name.removeprefix("metrics/"): value
            for name, value in event.val_metrics.items()
        }
        # Append the full epoch row to a universal, chart-ready history. Reuses
        # the exact schema of the family-gated results.csv so the monitor can
        # chart every family, gated or not, from a single append-only file.
        self._append_metrics(save_dir, TrainingArtifactsCallback._epoch_row(event))
        self._write(
            save_dir,
            state="running",
            current_epoch=max(int(event.epoch) - 1, 0),
            completed_epochs=self._completed_epochs,
            epoch_seconds=event.epoch_seconds,
            mean_epoch_seconds=mean_epoch,
            eta_seconds=mean_epoch * remaining,
            train_loss=event.train_loss,
            metrics=metrics,
            validated=event.validated,
            current_metric=event.current_metric,
            current_metric_name=event.current_metric_name,
            best_metric=event.best_metric,
            best_metric_name=event.best_metric_name,
            best_epoch=(event.best_epoch - 1 if event.best_epoch is not None else None),
        )

    def on_train_end(self, event: TrainEndEvent) -> None:
        save_dir = self._save_dir(event)
        absolute_completed = max(
            self._completed_epochs,
            self._start_completed_epochs + max(int(event.completed_epochs), 0),
        )
        absolute_completed = min(absolute_completed, max(int(event.total_epochs), 0))
        self._completed_epochs = absolute_completed
        self._write(
            save_dir,
            state="completed",
            completed_epochs=absolute_completed,
            total_seconds=event.total_seconds,
            train_loss=event.final_loss,
            best_metric=event.best_metric,
            best_epoch=(event.best_epoch - 1 if event.best_epoch is not None else None),
            checkpoints={
                "best": event.results.get("best_checkpoint"),
                "last": event.results.get("last_checkpoint"),
            },
        )
        self._close_log()

    def on_train_exception(self, event: TrainExceptionEvent) -> None:
        save_dir = self._save_dir(event)
        self._write(
            save_dir,
            state="failed",
            current_epoch=(max(int(event.epoch) - 1, 0) if event.epoch is not None else None),
            completed_epochs=self._completed_epochs,
            elapsed_seconds=event.elapsed_seconds,
            error={
                "type": event.exception_type,
                "message": event.exception_message,
            },
        )
        self._close_log()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _save_dir(event: Any) -> Path:
        save_dir = Path(event.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _write(self, save_dir: Path, **fields: Any) -> None:
        self._status.update(fields)
        payload = dict(self._base)
        payload.update(self._status)
        if self._start_time is not None:
            payload.setdefault(
                "elapsed_seconds", round(time.time() - self._start_time, 3)
            )
        total = payload.get("total_epochs") or 0
        completed = payload.get("completed_epochs") or 0
        payload["progress"] = min(max(completed / total, 0.0), 1.0) if total else 0.0
        payload["updated_at"] = _utcnow_iso()
        try:
            _atomic_write_json(save_dir / self.status_name, payload)
        except Exception:
            # Status is best-effort telemetry: never let it break a run.
            logger.debug("Failed to write %s", self.status_name, exc_info=True)

    def _append_metrics(self, save_dir: Path, row: Mapping[str, Any]) -> None:
        try:
            line = json.dumps(_json_safe(row), allow_nan=False)
            with open(save_dir / self.metrics_name, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            logger.debug("Failed to append %s", self.metrics_name, exc_info=True)

    @staticmethod
    def _trim_metrics_before_epoch(path: Path, *, start_epoch: int) -> None:
        """Atomically discard stale/duplicate JSONL rows at the resume boundary."""
        if not path.exists() or path.stat().st_size == 0:
            return
        kept = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    epoch = int(row.get("epoch")) if isinstance(row, Mapping) else None
                except (TypeError, ValueError):
                    continue
                if epoch is not None and epoch < start_epoch:
                    kept.append(json.dumps(_json_safe(row), allow_nan=False))
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                if kept:
                    handle.write("\n".join(kept) + "\n")
            _replace_with_retry(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    def _open_log(self, save_dir: Path, *, fresh: bool) -> None:
        if not self.write_log:
            return
        try:
            self._log_path = save_dir / self.log_name
            if fresh and self._log_path.exists():
                _unlink_with_retry(self._log_path)
            handler = logging.FileHandler(self._log_path, encoding="utf-8")
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            handler.addFilter(ThreadLogFilter())
            lib_logger = logging.getLogger("libreyolo")
            self._log_level_lease = LoggerLevelLease(
                lib_logger, logging.INFO
            ).acquire()
            lib_logger.addHandler(handler)
            self._log_handler = handler
        except Exception:
            if self._log_level_lease is not None:
                self._log_level_lease.release()
                self._log_level_lease = None
            logger.debug("Failed to open %s", self.log_name, exc_info=True)
            self._log_handler = None

    def _close_log(self) -> None:
        handler = self._log_handler
        if handler is None:
            if self._log_level_lease is not None:
                self._log_level_lease.release()
                self._log_level_lease = None
            return
        self._log_handler = None
        try:
            lib_logger = logging.getLogger("libreyolo")
            lib_logger.removeHandler(handler)
            handler.close()
        except Exception:
            logger.debug("Failed to close %s", self.log_name, exc_info=True)
        finally:
            if self._log_level_lease is not None:
                self._log_level_lease.release()
                self._log_level_lease = None
