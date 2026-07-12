"""Tests for atomic training checkpoint persistence."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
import torch

from libreyolo.training import trainer as trainer_module
from libreyolo.utils.serialization import (
    load_trusted_torch_file,
    wrap_libreyolo_checkpoint,
)

pytestmark = pytest.mark.unit


def _checkpoint(value=1.0):
    return wrap_libreyolo_checkpoint(
        {"layer.weight": torch.tensor([value])},
        model_family="yolo9",
        size="t",
        task="detect",
        nc=1,
        names={0: "object"},
        imgsz=64,
    )


def test_atomic_checkpoint_save_publishes_valid_identical_targets(tmp_path):
    last = tmp_path / "last.pt"
    best = tmp_path / "best.pt"

    trainer_module._atomic_save_checkpoint(_checkpoint(), [last, best])

    last_checkpoint = load_trusted_torch_file(last, context="unit test")
    best_checkpoint = load_trusted_torch_file(best, context="unit test")
    assert torch.equal(
        last_checkpoint["model"]["layer.weight"],
        best_checkpoint["model"]["layer.weight"],
    )
    assert last.read_bytes() == best.read_bytes()
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_checkpoint_write_failure_preserves_previous_target(
    tmp_path, monkeypatch
):
    target = tmp_path / "last.pt"
    previous = b"previous valid checkpoint"
    target.write_bytes(previous)

    def interrupted_save(checkpoint, file):
        del checkpoint
        file.write(b"truncated replacement")
        raise OSError("disk full")

    monkeypatch.setattr(trainer_module.torch, "save", interrupted_save)

    with pytest.raises(OSError, match="disk full"):
        trainer_module._atomic_save_checkpoint(_checkpoint(), [target])

    assert target.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [Path(target)]


def test_reader_sees_complete_old_checkpoint_until_atomic_replace(
    tmp_path, monkeypatch
):
    target = tmp_path / "last.pt"
    trainer_module._atomic_save_checkpoint(_checkpoint(1.0), [target])
    replace_started = Event()
    allow_replace = Event()
    real_replace = trainer_module.os.replace

    def paused_replace(source, destination):
        if Path(destination) == target:
            replace_started.set()
            assert allow_replace.wait(timeout=5)
        real_replace(source, destination)

    monkeypatch.setattr(trainer_module.os, "replace", paused_replace)

    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(
            trainer_module._atomic_save_checkpoint,
            _checkpoint(2.0),
            [target],
        )
        assert replace_started.wait(timeout=5)
        visible = load_trusted_torch_file(target, context="concurrent reader")
        assert visible["model"]["layer.weight"].item() == pytest.approx(1.0)
        allow_replace.set()
        writer.result(timeout=5)

    replaced = load_trusted_torch_file(target, context="post-replace reader")
    assert replaced["model"]["layer.weight"].item() == pytest.approx(2.0)


def test_unreadable_serialization_is_not_promoted(tmp_path, monkeypatch):
    target = tmp_path / "last.pt"
    previous = b"previous valid checkpoint"
    target.write_bytes(previous)

    def corrupt_save(checkpoint, file):
        del checkpoint
        file.write(b"not a torch checkpoint")

    monkeypatch.setattr(trainer_module.torch, "save", corrupt_save)

    with pytest.raises(Exception):
        trainer_module._atomic_save_checkpoint(_checkpoint(), [target])

    assert target.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [target]
