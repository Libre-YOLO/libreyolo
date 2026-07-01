"""Tests for the reserved (not-yet-implemented) knowledge-distillation API.

The ``distill_model`` / ``dis`` training arguments are accepted by every
trainable family so the public training API surface is stable, but the feature
itself is not implemented. Requesting a teacher must raise ``NotImplementedError``
rather than silently training an ordinary (non-distilled) model.
"""

import warnings

import pytest
import torch.nn as nn

from libreyolo.training.config import (
    TrainConfig,
    check_distillation_not_implemented,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Guard helper
# ---------------------------------------------------------------------------


def test_guard_noop_when_teacher_not_requested():
    # None means distillation was not requested — must not raise.
    assert check_distillation_not_implemented(None) is None


def test_guard_raises_when_teacher_requested():
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        check_distillation_not_implemented("teacher.pt")


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_config_reserves_distill_fields_with_defaults():
    cfg = TrainConfig()
    assert cfg.distill_model is None
    assert cfg.dis == 6.0


def test_config_accepts_distill_kwargs_without_unknown_key_warning():
    # These are real fields now, so from_kwargs must not warn about them.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = TrainConfig.from_kwargs(distill_model="teacher.pt", dis=10.0)
    assert cfg.distill_model == "teacher.pt"
    assert cfg.dis == 10.0


# ---------------------------------------------------------------------------
# Trainer wiring (covers every family via BaseTrainer.__init__)
# ---------------------------------------------------------------------------


def test_trainer_raises_when_teacher_requested():
    """A real flagship trainer must raise at construction when a teacher is set.

    The guard is the first line of ``BaseTrainer.__init__`` (all 17 trainable
    families subclass it), so this fires before any device/model/data work.
    """
    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        YOLO9Trainer(model=nn.Identity(), distill_model="teacher.pt")


# ---------------------------------------------------------------------------
# Python-API early guard (every family's train() is auto-wrapped)
# ---------------------------------------------------------------------------


class _FakeWrapper:
    pass


def _make_fake_train(captured: dict):
    def train(self, data, *, epochs=10, **kwargs):
        captured["called"] = True
        return {"ok": True}

    return train


def test_wrapped_train_raises_before_running_body():
    """The distill guard fires before the family train() body (no data work)."""
    from libreyolo.models.base.model import _wrap_train_with_cfg

    captured: dict = {}
    wrapped = _wrap_train_with_cfg(_make_fake_train(captured))
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        wrapped(_FakeWrapper(), "data.yaml", distill_model="teacher.pt")
    assert "called" not in captured  # body never ran


def test_wrapped_train_raises_when_teacher_comes_from_cfg_yaml(tmp_path):
    """A teacher supplied via ``cfg=`` yaml is caught after the merge."""
    from libreyolo.models.base.model import _wrap_train_with_cfg

    cfg = tmp_path / "train.yaml"
    cfg.write_text("epochs: 50\ndistill_model: teacher.pt\n")
    captured: dict = {}
    wrapped = _wrap_train_with_cfg(_make_fake_train(captured))
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        wrapped(_FakeWrapper(), "data.yaml", cfg=str(cfg))
    assert "called" not in captured


def test_wrapped_train_passes_through_when_no_teacher():
    """No teacher → guard is a no-op and the body runs normally."""
    from libreyolo.models.base.model import _wrap_train_with_cfg

    captured: dict = {}
    wrapped = _wrap_train_with_cfg(_make_fake_train(captured))
    assert wrapped(_FakeWrapper(), "data.yaml", epochs=5) == {"ok": True}
    assert captured["called"] is True
