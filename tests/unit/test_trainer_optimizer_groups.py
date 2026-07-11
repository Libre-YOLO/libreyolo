"""BaseTrainer optimizer param-group weight-decay semantics (issue #566)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit


class TinyTrainer(BaseTrainer):
    def get_model_family(self) -> str:
        return "tiny"

    def get_model_tag(self) -> str:
        return "tiny"

    def create_transforms(self):
        raise NotImplementedError

    def create_scheduler(self, iters_per_epoch: int):
        raise NotImplementedError

    def get_loss_components(self, outputs):
        return {}


def _make_trainer(optimizer: str) -> TinyTrainer:
    trainer = TinyTrainer.__new__(TinyTrainer)
    trainer.model = nn.Sequential(
        nn.Conv2d(3, 4, 3, bias=True),
        nn.BatchNorm2d(4),
        nn.Conv2d(4, 2, 1, bias=False),
    )
    trainer.config = SimpleNamespace(
        optimizer=optimizer,
        lr0=1e-3,
        weight_decay=0.05,
        momentum=0.9,
        nesterov=True,
    )
    return trainer


def test_adamw_norm_and_bias_groups_get_zero_weight_decay():
    """Param groups without an explicit weight_decay inherit the optimizer
    default, which is 0.01 for AdamW. Norms and biases must carry an explicit
    0.0 (upstream paramwise norm/bias_decay_mult=0)."""
    trainer = _make_trainer("adamw")
    opt = trainer._setup_optimizer()

    decays = sorted(g["weight_decay"] for g in opt.param_groups)
    assert decays == [0.0, 0.0, 0.05]

    # The decayed group is the conv/linear weights, not the BN/bias params.
    bn_weight = trainer.model[1].weight
    bias = trainer.model[0].bias
    for group in opt.param_groups:
        params = {id(p) for p in group["params"]}
        if id(bn_weight) in params or id(bias) in params:
            assert group["weight_decay"] == 0.0


def test_sgd_groups_unchanged_by_explicit_zero():
    """SGD's default decay is already 0, so the explicit 0.0 is a no-op."""
    trainer = _make_trainer("sgd")
    opt = trainer._setup_optimizer()
    decays = sorted(g["weight_decay"] for g in opt.param_groups)
    assert decays == [0.0, 0.0, 0.05]
