"""Collective-safety tests for rank-zero-only training phases."""

from __future__ import annotations

import contextlib
import json
import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.unit


def _free_port() -> int:
    socket_pair = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with contextlib.closing(socket_pair) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rank_zero_phase_worker(
    rank: int,
    world_size: int,
    port: int,
    output_dir: str,
) -> None:
    """Exercise success, root failure, and serialization failure on Gloo."""
    output = Path(output_dir)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=2),
    )
    try:
        from libreyolo.training.distributed import (
            RankZeroPhaseError,
            run_rank_zero_phase,
        )

        def successful_phase() -> dict:
            (output / f"success-called-rank-{rank}").write_text(
                "complete", encoding="utf-8"
            )
            return {"owner": rank, "items": [1, 2, 3]}

        success = run_rank_zero_phase("setup", successful_phase)
        marker_complete = (output / "success-called-rank-0").read_text(
            encoding="utf-8"
        ) == "complete"

        def failing_phase() -> None:
            (output / f"failure-called-rank-{rank}").write_text(
                "called", encoding="utf-8"
            )
            raise ValueError("rank zero exploded")

        try:
            run_rank_zero_phase("validation", failing_phase)
        except RankZeroPhaseError as exc:
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "phase": exc.phase,
                "failure_stage": exc.failure_stage,
                "root_type": exc.root_exception_type,
                "root_message": exc.root_exception_message,
                "traceback": exc.rank_zero_traceback,
                "cause_type": type(exc.__cause__).__name__
                if exc.__cause__ is not None
                else None,
            }
        else:
            raise AssertionError("rank-zero failure was not propagated")

        # A collective after the error proves every rank received the outcome
        # and remained in a usable, synchronized process group.
        reached_after_failure = torch.tensor(1, dtype=torch.int64)
        dist.all_reduce(reached_after_failure)

        def unserializable_phase():
            return lambda: None

        try:
            run_rank_zero_phase("checkpoint", unserializable_phase)
        except RankZeroPhaseError as exc:
            serialization_failure = {
                "message": str(exc),
                "phase": exc.phase,
                "failure_stage": exc.failure_stage,
                "root_type": exc.root_exception_type,
                "root_message": exc.root_exception_message,
            }
        else:
            raise AssertionError("unserializable rank-zero result was accepted")

        recovered = run_rank_zero_phase("callback", lambda: "recovered")
        record = {
            "success": success,
            "marker_complete": marker_complete,
            "failure": failure,
            "serialization_failure": serialization_failure,
            "reached_after_failure": int(reached_after_failure.item()),
            "recovered": recovered,
        }
        (output / f"rank-{rank}.json").write_text(
            json.dumps(record, sort_keys=True), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


def _rank_zero_validation_worker(
    rank: int,
    world_size: int,
    port: int,
    output_dir: str,
) -> None:
    """Prove rank-zero validation bypasses the live DDP forward wrapper."""
    output = Path(output_dir)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=2),
    )
    try:
        from libreyolo.training.trainer import BaseTrainer

        class _BufferModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1, bias=False)
                self.register_buffer("offset", torch.tensor([0.5]))

            def forward(self, inputs):
                return self.linear(inputs) + self.offset

        torch.manual_seed(7)
        ddp_model = torch.nn.parallel.DistributedDataParallel(_BufferModel())
        ddp_model.train()

        class _Wrapper:
            def __init__(self, model):
                self.model = model

        class _ValidationProbe:
            _validate_epoch = BaseTrainer._validate_epoch
            _run_rank_zero_validation = BaseTrainer._run_rank_zero_validation

            def _run_validation(self, epoch, *, save_plots=None):
                del epoch, save_plots
                if isinstance(
                    self.model,
                    torch.nn.parallel.DistributedDataParallel,
                ):
                    raise AssertionError("rank-zero validation entered DDP.forward")
                if self.model is not ddp_model.module:
                    raise AssertionError("validation did not receive the DDP module")
                if self.wrapper_model.model is not self.model:
                    raise AssertionError("wrapper still references the DDP model")
                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(torch.ones(1, 2))
                (output / f"validation-called-rank-{rank}").write_text(
                    "complete",
                    encoding="utf-8",
                )
                return {
                    "owner": rank,
                    "prediction": float(prediction.item()),
                }

        trainer = _ValidationProbe()
        trainer.model = ddp_model
        trainer.wrapper_model = _Wrapper(ddp_model)
        result = trainer._validate_epoch(0)

        restored = (
            trainer.model is ddp_model
            and trainer.wrapper_model.model is ddp_model
            and trainer.model.training
            and trainer.model.module.training
        )
        with torch.no_grad():
            post_validation = ddp_model(torch.ones(1, 2))
        collective = torch.tensor(1, dtype=torch.int64)
        dist.all_reduce(collective)
        (output / f"validation-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "result": result,
                    "restored": restored,
                    "post_validation": float(post_validation.item()),
                    "collective": int(collective.item()),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def test_rank_zero_phase_preserves_non_distributed_behavior():
    from libreyolo.training.distributed import run_rank_zero_phase

    calls: list[str] = []
    result = run_rank_zero_phase("setup", lambda: calls.append("called") or 42)

    assert result == 42
    assert calls == ["called"]

    root_error = ValueError("local failure")
    with pytest.raises(ValueError) as caught:
        run_rank_zero_phase("validation", lambda: (_ for _ in ()).throw(root_error))
    assert caught.value is root_error


def test_rank_zero_phase_broadcasts_success_and_failure_on_cpu_gloo(tmp_path):
    world_size = 2
    mp.spawn(
        _rank_zero_phase_worker,
        args=(world_size, _free_port(), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    records = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    assert (
        records[0]["success"]
        == records[1]["success"]
        == {
            "owner": 0,
            "items": [1, 2, 3],
        }
    )
    assert all(record["marker_complete"] for record in records)
    assert sorted(path.name for path in tmp_path.glob("success-called-rank-*")) == [
        "success-called-rank-0"
    ]
    assert sorted(path.name for path in tmp_path.glob("failure-called-rank-*")) == [
        "failure-called-rank-0"
    ]

    failures = [record["failure"] for record in records]
    assert failures[0]["message"] == failures[1]["message"]
    assert failures[0]["phase"] == failures[1]["phase"] == "validation"
    assert failures[0]["failure_stage"] == failures[1]["failure_stage"] == "execution"
    assert failures[0]["root_type"] == failures[1]["root_type"] == "ValueError"
    assert (
        failures[0]["root_message"]
        == failures[1]["root_message"]
        == "rank zero exploded"
    )
    assert "ValueError: rank zero exploded" in failures[0]["traceback"]
    assert failures[0]["cause_type"] == "ValueError"
    assert failures[1]["cause_type"] is None

    serialization_failures = [record["serialization_failure"] for record in records]
    assert serialization_failures[0] == serialization_failures[1]
    assert serialization_failures[0]["phase"] == "checkpoint"
    assert serialization_failures[0]["failure_stage"] == "result serialization"
    assert "local object" in serialization_failures[0]["root_message"].lower()
    assert all(record["reached_after_failure"] == world_size for record in records)
    assert all(record["recovered"] == "recovered" for record in records)


def test_rank_zero_validation_unwraps_ddp_on_cpu_gloo(tmp_path):
    world_size = 2
    mp.spawn(
        _rank_zero_validation_worker,
        args=(world_size, _free_port(), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    records = [
        json.loads(
            (tmp_path / f"validation-rank-{rank}.json").read_text(encoding="utf-8")
        )
        for rank in range(world_size)
    ]
    assert records[0]["result"] == records[1]["result"]
    assert records[0]["result"]["owner"] == 0
    assert all(record["restored"] for record in records)
    assert all(record["collective"] == world_size for record in records)
    assert records[0]["post_validation"] == pytest.approx(records[1]["post_validation"])
    assert sorted(path.name for path in tmp_path.glob("validation-called-rank-*")) == [
        "validation-called-rank-0"
    ]
