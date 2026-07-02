"""Tests for the public knowledge-distillation training API.

``distill_model`` (teacher checkpoint path) turns distillation on; ``dis`` is
the global loss weight (None = per-loss-type default); ``distill_loss_type``
selects the feature loss. The arguments are accepted by every trainable family;
families without ``get_distill_config()`` raise a clear error at setup time.
"""

import warnings

import pytest

from libreyolo.training.config import TrainConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_config_distill_defaults():
    cfg = TrainConfig()
    assert cfg.distill_model is None
    assert cfg.dis is None
    assert cfg.distill_loss_type == "mgd"
    assert cfg.distill_mask_ratio == 0.65
    assert cfg.distill_tau == 1.0


def test_config_accepts_distill_kwargs_without_unknown_key_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = TrainConfig.from_kwargs(
            distill_model="teacher.pt", dis=10.0, distill_loss_type="cwd"
        )
    assert cfg.distill_model == "teacher.pt"
    assert cfg.dis == 10.0
    assert cfg.distill_loss_type == "cwd"


# ---------------------------------------------------------------------------
# Trainer wiring
# ---------------------------------------------------------------------------


def test_trainer_accepts_teacher_at_construction():
    """Constructing a trainer with a teacher no longer raises; the teacher is
    only loaded at setup() time."""
    import torch.nn as nn

    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    trainer = YOLO9Trainer(model=nn.Identity(), distill_model="teacher.pt")
    assert trainer.config.distill_model == "teacher.pt"
    assert trainer.distiller is None  # not set up until setup()


def test_setup_distillation_noop_without_teacher():
    import torch.nn as nn

    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    trainer = YOLO9Trainer(model=nn.Identity())
    trainer._setup_distillation()
    assert trainer.distiller is None


def test_setup_distillation_requires_wrapper_model():
    import torch.nn as nn

    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    trainer = YOLO9Trainer(model=nn.Identity(), distill_model="teacher.pt")
    assert trainer.wrapper_model is None
    with pytest.raises(ValueError, match="wrapper_model"):
        trainer._setup_distillation()


def test_unsupported_family_raises_clear_error():
    """Families without get_distill_config() fail with a message naming them."""
    from libreyolo.models.base.model import BaseModel

    class _Stub:
        FAMILY = "stubfamily"
        get_distill_config = BaseModel.get_distill_config

    with pytest.raises(NotImplementedError, match="stubfamily"):
        _Stub().get_distill_config()


# ---------------------------------------------------------------------------
# cfg-yaml passthrough (every family's train() is auto-wrapped)
# ---------------------------------------------------------------------------


class _FakeWrapper:
    pass


def test_wrapped_train_forwards_distill_kwargs_from_cfg_yaml(tmp_path):
    from libreyolo.models.base.model import _wrap_train_with_cfg

    cfg = tmp_path / "train.yaml"
    cfg.write_text("epochs: 50\ndistill_model: teacher.pt\ndis: 2.0\n")
    captured: dict = {}

    def train(self, data, *, epochs=10, **kwargs):
        captured.update(kwargs, epochs=epochs)
        return {"ok": True}

    wrapped = _wrap_train_with_cfg(train)
    assert wrapped(_FakeWrapper(), "data.yaml", cfg=str(cfg)) == {"ok": True}
    assert captured["distill_model"] == "teacher.pt"
    assert captured["dis"] == 2.0
    assert captured["epochs"] == 50
