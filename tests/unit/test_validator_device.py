"""BaseValidator._setup_device normalisation tests."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from libreyolo.validation.base import BaseValidator
from libreyolo.validation.config import ValidationConfig

pytestmark = pytest.mark.unit


class _StubValidator(BaseValidator):
    def _setup_dataloader(self): pass
    def _init_metrics(self): pass
    def _preprocess_batch(self, b): pass
    def _postprocess_predictions(self, p, b): pass
    def _update_metrics(self, p, t, i, ids=None): pass
    def _compute_metrics(self): return {}


class _TransactionalValidator(_StubValidator):
    def _setup(self, **kwargs):
        del kwargs

    def _run_validation(self):
        assert self.model.device == self.device
        expected = getattr(self, "expected_actual_device", self.device)
        assert next(self.model.model.parameters()).device == expected
        assert not self.model.model.training
        if self.raise_during_validation:
            raise RuntimeError("validation failed")

    def _finalize(self):
        return {"ok": 1.0}


class _TrackingLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1)
        self.to_calls = []

    def to(self, device, *args, **kwargs):
        del args, kwargs
        self.to_calls.append(torch.device(device))
        return self


def _setup_device(device: str) -> "torch.device":
    config = ValidationConfig(data="x.yaml", device=device)
    v = object.__new__(_StubValidator)
    v.config = config
    return v._setup_device()


def _stub_validator(*, half: bool, device: str):
    validator = object.__new__(_StubValidator)
    validator.config = ValidationConfig(data="x.yaml", device=device, half=half)
    validator.device = torch.device(device)
    return validator


def _transactional_validator(*, raise_during_validation: bool):
    validator = object.__new__(_TransactionalValidator)
    validator.model = type("Wrapper", (), {})()
    validator.model.model = torch.nn.Linear(1, 1).train()
    validator.model.model.add_module("dropout", torch.nn.Dropout().eval())
    validator.model.device = torch.device("cpu")
    validator.device = torch.device("cpu")
    validator.raise_during_validation = raise_during_validation
    return validator


def test_bare_integer_device_string_normalised():
    with patch("torch.cuda.is_available", return_value=True):
        device = _setup_device("0")
    assert device.type == "cuda"
    assert str(device) == "cuda:0"


def test_bare_integer_string_two_digit():
    with patch("torch.cuda.is_available", return_value=True):
        device = _setup_device("10")
    assert device.type == "cuda"
    assert str(device) == "cuda:10"


def test_named_device_strings_pass_through():
    assert _setup_device("cpu").type == "cpu"
    assert str(_setup_device("cuda:0")) == "cuda:0"


@pytest.mark.parametrize(
    ("device", "expected_calls"),
    [("cuda", ["cuda"]), ("cpu", [])],
)
def test_half_validation_uses_cuda_autocast_only(monkeypatch, device, expected_calls):
    calls = []

    @contextmanager
    def fake_autocast(device_type):
        calls.append(device_type)
        yield

    monkeypatch.setattr("libreyolo.validation.base.torch.amp.autocast", fake_autocast)

    validator = _stub_validator(half=True, device=device)

    with validator._autocast_context():
        pass

    assert calls == expected_calls


def test_validation_restores_model_device_and_mode_after_success():
    validator = _transactional_validator(raise_during_validation=False)

    assert validator.run() == {"ok": 1.0}
    assert validator.model.device == torch.device("cpu")
    assert next(validator.model.model.parameters()).device.type == "cpu"
    assert validator.model.model.training
    assert not validator.model.model.dropout.training


def test_validation_restores_model_device_and_mode_after_exception():
    validator = _transactional_validator(raise_during_validation=True)

    with pytest.raises(RuntimeError, match="validation failed"):
        validator.run()

    assert validator.model.device == torch.device("cpu")
    assert next(validator.model.model.parameters()).device.type == "cpu"
    assert validator.model.model.training
    assert not validator.model.model.dropout.training


def test_validation_device_override_is_temporary():
    validator = _transactional_validator(raise_during_validation=False)
    module = _TrackingLinear().train()
    validator.model.model = module
    validator.device = torch.device("cuda:7")
    validator.expected_actual_device = torch.device("cpu")

    validator.run()

    assert module.to_calls == [torch.device("cuda:7"), torch.device("cpu")]
    assert validator.model.device == torch.device("cpu")


def test_exported_backend_eval_proxy_skips_native_module_migration():
    class _EvalProxy:
        def eval(self):
            return self

    validator = object.__new__(_StubValidator)
    validator.model = type("Backend", (), {"model": _EvalProxy()})()
    validator.device = torch.device("cpu")

    with validator._validation_model_state():
        pass


def test_warmup_uses_effective_dataloader_batch_size():
    seen_shapes = []

    class _Wrapper:
        model = torch.nn.Identity()

        def _forward(self, tensor):
            seen_shapes.append(tuple(tensor.shape))
            return tensor

    validator = object.__new__(_StubValidator)
    validator.model = _Wrapper()
    validator.device = torch.device("cpu")
    validator.config = ValidationConfig(
        data="x.yaml",
        imgsz=32,
        batch_size=8,
        device="cpu",
        verbose=False,
    )
    validator.dataloader = type("Loader", (), {"batch_size": 1})()

    validator._warmup_model(n_warmup=1)

    assert seen_shapes == [(1, 3, 32, 32)]


def test_pose_validator_override_uses_transactional_model_state():
    from libreyolo.validation.pose_validator import PoseValidator

    validator = object.__new__(PoseValidator)
    validator.model = type("Wrapper", (), {})()
    validator.model.model = torch.nn.Sequential(
        torch.nn.Linear(1, 1),
        torch.nn.Dropout(),
    ).train()
    validator.model.model[1].eval()
    validator.model.device = torch.device("cpu")
    validator.device = torch.device("cpu")
    original_modes = [module.training for module in validator.model.model.modules()]

    def fail_inside_pose(**kwargs):
        assert kwargs == {"sentinel": True}
        assert not validator.model.model.training
        assert not validator.model.model[1].training
        raise RuntimeError("pose validation failed")

    validator._run_pose = fail_inside_pose

    with pytest.raises(RuntimeError, match="pose validation failed"):
        validator.run(sentinel=True)

    assert [module.training for module in validator.model.model.modules()] == original_modes
    assert validator.model.device == torch.device("cpu")
