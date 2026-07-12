"""AMP optimizer-step gating for D-FINE and DEIM trainer overrides."""

from __future__ import annotations

import contextlib
import importlib
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.unit


class _Loader:
    def __init__(self, num_batches: int):
        self.dataset = SimpleNamespace()
        self.collate_fn = None
        self._batches = [
            (
                torch.zeros(1, 1),
                torch.zeros(1, 1, 5),
                ((1, 1),),
                (index,),
            )
            for index in range(num_batches)
        ]

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


class _VariableLoader(_Loader):
    def __init__(self):
        self.dataset = SimpleNamespace()
        self.collate_fn = None
        self._batches = [
            (
                torch.ones(4, 1),
                torch.zeros(4, 1, 5),
                ((1, 1),) * 4,
                tuple(range(4)),
            ),
            (
                torch.full((1, 1), 10.0),
                torch.zeros(1, 1, 5),
                ((1, 1),),
                (4,),
            ),
        ]


class _Progress:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, values) -> None:
        del values


class _SequencedScaler:
    """Minimal GradScaler double: skip one step, then apply one step."""

    def __init__(self):
        self._scale = 8.0
        self._outcomes = iter((False, True))
        self._last_step_succeeded = False
        self.optimizer_steps = 0
        self.update_calls = 0

    def scale(self, loss):
        return loss

    def get_scale(self) -> float:
        return self._scale

    def unscale_(self, optimizer) -> None:
        del optimizer

    def step(self, optimizer) -> None:
        self._last_step_succeeded = next(self._outcomes)
        if self._last_step_succeeded:
            optimizer.step()
            self.optimizer_steps += 1

    def update(self) -> None:
        self.update_calls += 1
        if not self._last_step_succeeded:
            self._scale /= 2.0


class _EMA:
    def __init__(self):
        self.updates = 0

    def update(self, model) -> None:
        del model
        self.updates += 1


class _Scheduler:
    def __init__(self):
        self.calls: list[int] = []

    def update_lr(self, step: int) -> float:
        self.calls.append(step)
        return 0.05


@pytest.mark.parametrize("family", ["dfine", "deim"])
@pytest.mark.parametrize("accum_steps", [1, 2])
def test_amp_skip_does_not_advance_optimizer_dependent_state(
    family,
    accum_steps,
    monkeypatch,
):
    module = importlib.import_module(f"libreyolo.models.{family}.trainer")
    trainer_class = getattr(module, f"{family.upper()}Trainer")
    monkeypatch.setattr(
        module,
        "autocast",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        module,
        "tqdm",
        lambda iterable, **kwargs: _Progress(iterable),
    )

    trainer = trainer_class.__new__(trainer_class)
    trainer.config = SimpleNamespace(
        epochs=1,
        clip_max_norm=0.0,
        eval_interval=-1,
        batch=1,
        nbs=accum_steps,
    )
    trainer.device = torch.device("cpu")
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        trainer.model.weight.fill_(1.0)
    trainer.optimizer = torch.optim.SGD(
        [
            {
                "params": trainer.model.parameters(),
                "lr": 0.2,
                "lr_mult": 0.5,
            }
        ],
        lr=0.2,
    )
    trainer.scaler = _SequencedScaler()
    trainer.ema_model = _EMA()
    trainer.lr_scheduler = _Scheduler()
    trainer.optimizer_step_count = 0
    trainer._frozen_bn_modules = []
    trainer.train_loader = _Loader(num_batches=2 if accum_steps == 1 else 4)
    trainer.get_loss_components = lambda outputs: {}
    trainer.on_forward = lambda imgs, targets, polygons=None: {
        "total_loss": trainer.model.weight.sum()
    }

    _, val_metrics, loss_components, lrs = trainer_class._train_epoch(trainer, 0)

    assert val_metrics is None
    assert loss_components == {}
    assert trainer.scaler.update_calls == 2
    assert trainer.scaler.optimizer_steps == 1
    assert trainer.optimizer_step_count == 1
    assert trainer.ema_model.updates == 1
    assert trainer.lr_scheduler.calls == [1]
    assert trainer.model.weight.item() == pytest.approx(0.8)
    assert lrs == {"group0": pytest.approx(0.025)}


@pytest.mark.parametrize("family", ["dfine", "deim"])
def test_variable_microbatches_are_sample_weighted(family, monkeypatch):
    module = importlib.import_module(f"libreyolo.models.{family}.trainer")
    trainer_class = getattr(module, f"{family.upper()}Trainer")
    monkeypatch.setattr(
        module,
        "tqdm",
        lambda iterable, **kwargs: _Progress(iterable),
    )

    trainer = trainer_class.__new__(trainer_class)
    trainer.config = SimpleNamespace(
        epochs=1,
        clip_max_norm=0.0,
        eval_interval=-1,
        batch=1,
        nbs=2,
    )
    trainer.device = torch.device("cpu")
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        trainer.model.weight.fill_(1.0)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scaler = None
    trainer.ema_model = None
    trainer.lr_scheduler = _Scheduler()
    trainer.optimizer_step_count = 0
    trainer._frozen_bn_modules = []
    trainer.train_loader = _VariableLoader()
    trainer.get_loss_components = lambda outputs: {}
    trainer.on_forward = lambda imgs, targets, polygons=None: {
        "total_loss": trainer.model(imgs).mean()
    }

    trainer_class._train_epoch(trainer, 0)

    # Combined gradient is (4 * 1 + 1 * 10) / 5 = 2.8.
    assert trainer.model.weight.item() == pytest.approx(0.72)
