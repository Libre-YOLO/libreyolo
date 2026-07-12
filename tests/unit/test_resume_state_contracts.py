"""Strict resume ordering and checkpoint identity contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit


class _Scheduler:
    def __init__(self):
        self.position = 0

    def update_lr(self, step: int) -> float:
        del step
        return 0.01

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = int(state["position"])


class _ResumeTrainer(BaseTrainer):
    def get_model_family(self) -> str:
        return "resume-test"

    def get_model_tag(self) -> str:
        return "resume-test"

    def create_transforms(self):
        return None, None

    def create_scheduler(self, iters_per_epoch):
        del iters_per_epoch
        return _Scheduler()

    def get_loss_components(self, outputs):
        del outputs
        return {}

    def on_forward(self, imgs, targets, polygons=None):
        del targets, polygons
        return {"total_loss": self.model(imgs).mean()}

    def on_setup(self):
        self.events.append("on_setup")

    def _setup_data(self):
        self.events.append("data")
        self.model = torch.nn.Linear(2, 1, bias=False)
        self.train_loader = [object()]

    def _setup_optimizer(self):
        self.events.append("optimizer")
        torch.testing.assert_close(
            self.model.weight,
            self.expected_weight,
        )
        return torch.optim.SGD(self.model.parameters(), lr=0.01)


def _trainer(tmp_path: Path) -> _ResumeTrainer:
    trainer = _ResumeTrainer(
        model=torch.nn.Linear(1, 1, bias=False),
        device="cpu",
        amp=False,
        ema=False,
        data=None,
        project=str(tmp_path),
        name="run",
        exist_ok=True,
    )
    trainer.events = []
    return trainer


def test_resume_defers_model_load_until_after_architecture_setup(tmp_path):
    expected_weight = torch.tensor([[2.0, 3.0]])
    checkpoint = tmp_path / "last.pt"
    torch.save(
        {
            "model": {"weight": expected_weight.clone()},
            "epoch": 0,
        },
        checkpoint,
    )
    trainer = _trainer(tmp_path)
    trainer.expected_weight = expected_weight

    trainer.resume(str(checkpoint))

    assert trainer.model.weight.shape == (1, 1)
    assert trainer._resume_model_state is not None
    trainer.setup()
    assert trainer.events == ["on_setup", "data", "optimizer"]
    torch.testing.assert_close(trainer.model.weight, expected_weight)


def test_resume_with_distillation_disabled_keeps_student_optimizer_state(
    tmp_path,
    caplog,
):
    source_model = torch.nn.Linear(2, 1, bias=False)
    adapter_weight = torch.nn.Parameter(torch.tensor([[3.0]]))
    adapter_bias = torch.nn.Parameter(torch.tensor([4.0]))
    optimizer = torch.optim.SGD(
        [
            {"params": list(source_model.parameters()), "lr": 0.07},
            {"params": [adapter_weight], "lr": 0.03},
            {"params": [adapter_bias], "lr": 0.03},
        ],
        momentum=0.9,
    )
    source_model.weight.grad = torch.tensor([[2.0, 4.0]])
    adapter_weight.grad = torch.tensor([[5.0]])
    adapter_bias.grad = torch.tensor([6.0])
    optimizer.step()
    optimizer_state = optimizer.state_dict()
    student_param_id = optimizer_state["param_groups"][0]["params"][0]
    expected_momentum = optimizer_state["state"][student_param_id][
        "momentum_buffer"
    ].clone()
    expected_ema_weight = torch.tensor([[7.0, 11.0]])

    trainer = _ResumeTrainer(
        model=torch.nn.Linear(1, 1, bias=False),
        device="cpu",
        amp=False,
        ema=True,
        distill_model=None,
        data=None,
        project=str(tmp_path),
        name="run",
        exist_ok=True,
    )
    trainer.events = []
    saved_config = trainer.config.to_dict()
    saved_config["distill_model"] = "teacher.pt"
    checkpoint = tmp_path / "distilled.pt"
    torch.save(
        {
            "model": source_model.state_dict(),
            "epoch": 2,
            "optimizer": optimizer_state,
            "optimizer_step_count": 13,
            "scheduler": {"position": 13},
            "distiller": {
                "adapter.weight": adapter_weight.detach().clone(),
                "adapter.bias": adapter_bias.detach().clone(),
            },
            "ema": {"weight": expected_ema_weight.clone()},
            "ema_updates": 9,
            "config": saved_config,
        },
        checkpoint,
    )
    trainer.expected_weight = source_model.weight.detach().clone()

    trainer.resume(str(checkpoint))
    trainer.setup()

    assert trainer.distiller is None
    assert len(trainer.optimizer.param_groups) == 1
    assert len(trainer.optimizer.state) == 1
    assert trainer.optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)
    torch.testing.assert_close(
        trainer.optimizer.state[trainer.model.weight]["momentum_buffer"],
        expected_momentum,
    )
    assert trainer.optimizer_step_count == 13
    assert trainer.lr_scheduler.position == 13
    assert trainer.ema_model.updates == 9
    torch.testing.assert_close(trainer.ema_model.ema.weight, expected_ema_weight)
    assert "distillation is disabled" in caplog.text


def test_disabled_distillation_rejects_ambiguous_optimizer_topology(tmp_path):
    current_param = torch.nn.Parameter(torch.ones(1))
    trainer = _trainer(tmp_path)
    trainer.optimizer = torch.optim.SGD([current_param], lr=0.01)
    trainer._discard_resume_distiller_optimizer_state = True

    model_param_a = torch.nn.Parameter(torch.ones(1))
    model_param_b = torch.nn.Parameter(torch.ones(1))
    distiller_param = torch.nn.Parameter(torch.ones(1))
    saved_optimizer = torch.optim.SGD(
        [
            {"params": [model_param_a, model_param_b]},
            {"params": [distiller_param]},
        ],
        lr=0.01,
    ).state_dict()

    with pytest.raises(RuntimeError, match="model optimizer group 0"):
        trainer._optimizer_state_for_resume(saved_optimizer)


def test_resume_inherits_saved_distillation_when_not_explicitly_disabled(tmp_path):
    trainer = _trainer(tmp_path)
    saved_config = trainer.config.to_dict()
    saved_config["distill_model"] = "teacher.pt"
    checkpoint = tmp_path / "distilled.pt"
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "epoch": 0,
            "distiller": {"adapter.weight": torch.ones(1)},
            "config": saved_config,
        },
        checkpoint,
    )

    trainer.resume(str(checkpoint))

    assert trainer.config.distill_model == "teacher.pt"
    assert trainer._resume_distiller_state is not None
    assert not getattr(trainer, "_discard_resume_distiller_optimizer_state", False)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("model_family", "another-family"),
        ("size", "m"),
        ("task", "segment"),
    ],
)
def test_resume_rejects_checkpoint_identity_mismatch(
    tmp_path,
    field,
    wrong_value,
):
    checkpoint = tmp_path / "wrong.pt"
    payload = {
        "model": {"weight": torch.ones(1, 1)},
        "epoch": 0,
        "model_family": "resume-test",
        "size": "s",
        "task": "detect",
    }
    payload[field] = wrong_value
    torch.save(
        payload,
        checkpoint,
    )
    trainer = _trainer(tmp_path)

    with pytest.raises(RuntimeError, match=field):
        trainer.resume(str(checkpoint))


def test_resume_rejects_inference_only_checkpoint(tmp_path):
    checkpoint = tmp_path / "inference.pt"
    torch.save({"model": {"weight": torch.ones(1, 1)}}, checkpoint)
    trainer = _trainer(tmp_path)

    with pytest.raises(RuntimeError, match="no training epoch"):
        trainer.resume(str(checkpoint))


@pytest.mark.parametrize(
    "state_name",
    ["optimizer", "scheduler", "distiller", "ema", "scaler"],
)
def test_resume_rejects_null_component_state(tmp_path, state_name):
    checkpoint = tmp_path / f"null-{state_name}.pt"
    torch.save(
        {
            "model": {"weight": torch.ones(1, 1)},
            "epoch": 0,
            state_name: None,
        },
        checkpoint,
    )
    trainer = _trainer(tmp_path)

    with pytest.raises(RuntimeError, match=rf"{state_name} state"):
        trainer.resume(str(checkpoint))


def test_resume_rejects_every_checkpoint_after_setup(tmp_path):
    checkpoint = tmp_path / "different-nc.pt"
    torch.save(
        {
            "model": {"weight": torch.ones(2, 1)},
            "epoch": 0,
            "nc": 2,
        },
        checkpoint,
    )
    trainer = _trainer(tmp_path)

    class _Wrapper:
        task = "detect"
        nb_classes = 1

        def __init__(self, model):
            self.model = model
            self.rebuild_calls = 0

        def _rebuild_for_checkpoint_classes(self, nc, model_state):
            del nc, model_state
            self.rebuild_calls += 1

    wrapper = _Wrapper(trainer.model)
    trainer.wrapper_model = wrapper
    trainer._is_setup = True
    original_model = trainer.model

    with pytest.raises(RuntimeError, match=r"before setup\(\)"):
        trainer.resume(str(checkpoint))

    assert trainer.model is original_model
    assert wrapper.model is original_model
    assert wrapper.rebuild_calls == 0


def test_omitted_recipe_value_inherits_checkpoint_config(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.config.lr0 = 0.005
    trainer._explicit_train_config_keys = frozenset()

    trainer._restore_checkpoint_config(
        {
            "config": {
                **trainer.config.to_dict(),
                "lr0": 0.000123,
            }
        }
    )

    assert trainer.config.lr0 == pytest.approx(0.000123)
