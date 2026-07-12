from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from libreyolo.training.artifacts import TrainingStatusCallback
from libreyolo.ui.server import _UIState

pytestmark = pytest.mark.unit


def test_new_run_reservation_is_atomic_across_server_states(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    states = [_UIState(device="cpu") for _ in range(12)]
    barrier = threading.Barrier(len(states))

    def reserve(state: _UIState) -> Path:
        barrier.wait(timeout=5)
        return state.new_run()

    with ThreadPoolExecutor(max_workers=len(states)) as pool:
        paths = list(pool.map(reserve, states))

    assert len(set(paths)) == len(states)
    assert all(path.is_dir() for path in paths)
    assert {path.name for path in paths} == {
        "predict",
        *(f"predict{index}" for index in range(2, len(states) + 1)),
    }


def test_new_run_waits_for_inference_and_does_not_rewrite_response_dir(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    state = _UIState(device="cpu")
    first_run = state.new_run()
    inference_entered = threading.Event()
    release_inference = threading.Event()
    new_run_started = threading.Event()

    class FakeModel:
        def __call__(self, _source, *, conf, save, output_path):
            assert conf == pytest.approx(0.25)
            assert save is True
            inference_entered.set()
            assert release_inference.wait(timeout=5)
            saved = Path(output_path) / "image.jpg"
            saved.write_bytes(b"rendered")
            return SimpleNamespace(saved_path=str(saved), boxes=[])

    state._get_model = lambda _name: FakeModel()

    def start_new_run():
        new_run_started.set()
        return state.new_run()

    with ThreadPoolExecutor(max_workers=2) as pool:
        inference = pool.submit(
            state.infer, "fake", 0.25, "image.jpg", b"input"
        )
        assert inference_entered.wait(timeout=5)
        next_run = pool.submit(start_new_run)
        assert new_run_started.wait(timeout=5)
        assert not next_run.done()
        release_inference.set()
        result = inference.result(timeout=5)
        second_run = next_run.result(timeout=5)

    assert result["dir"] == str(first_run)
    assert result["saved"] == str(first_run / "image.jpg")
    assert second_run != first_run
    assert state.run_dir == second_run


def test_ui_and_training_log_capture_are_isolated_when_interleaved(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    state = _UIState(device="cpu")
    state.new_run()
    ui_entered = threading.Event()
    release_ui = threading.Event()
    training_started = threading.Event()
    release_training = threading.Event()
    emitted: list[str] = []
    run_dir = tmp_path / "train"

    class FakeModel:
        def __call__(self, _source, *, conf, save, output_path):
            logging.getLogger("libreyolo").info("ui-only-message")
            ui_entered.set()
            assert release_ui.wait(timeout=5)
            saved = Path(output_path) / "image.jpg"
            saved.write_bytes(b"rendered")
            return SimpleNamespace(saved_path=str(saved), boxes=[])

    state._get_model = lambda _name: FakeModel()
    start = SimpleNamespace(
        start_epoch=1,
        total_epochs=1,
        model_family="yolo9",
        model_size="t",
        task="detect",
        save_dir=str(run_dir),
        config={},
    )
    end = SimpleNamespace(
        total_epochs=1,
        completed_epochs=1,
        model_family="yolo9",
        model_size="t",
        task="detect",
        save_dir=str(run_dir),
        final_loss=0.0,
        best_metric=None,
        best_epoch=None,
        total_seconds=0.1,
        results={},
    )
    lib_logger = logging.getLogger("libreyolo")
    original_level = lib_logger.level
    lib_logger.setLevel(logging.WARNING)

    def train():
        callback = TrainingStatusCallback()
        callback.on_train_start(start)
        logging.getLogger("libreyolo").info("training-only-message")
        training_started.set()
        assert release_training.wait(timeout=5)
        callback.on_train_end(end)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            inference = pool.submit(
                state.infer,
                "fake",
                0.25,
                "image.jpg",
                b"input",
                emitted.append,
            )
            assert ui_entered.wait(timeout=5)
            training = pool.submit(train)
            assert training_started.wait(timeout=5)
            assert lib_logger.level == logging.INFO

            release_ui.set()
            inference.result(timeout=5)
            # The training lease remains active after UI capture closes.
            assert lib_logger.level == logging.INFO
            release_training.set()
            training.result(timeout=5)

        assert lib_logger.level == logging.WARNING
        ui_output = "\n".join(emitted)
        training_output = (run_dir / "train.log").read_text()
        assert "ui-only-message" in ui_output
        assert "training-only-message" not in ui_output
        assert "training-only-message" in training_output
        assert "ui-only-message" not in training_output
    finally:
        release_ui.set()
        release_training.set()
        lib_logger.setLevel(original_level)
