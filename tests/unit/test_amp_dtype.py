"""Contract tests for explicit automatic mixed-precision dtypes."""

from types import SimpleNamespace

import pytest
import torch

from libreyolo.training.config import TrainConfig
from libreyolo.training.trainer import BaseTrainer
from libreyolo.utils.amp import normalize_amp_dtype, torch_amp_dtype
from libreyolo.validation.config import ValidationConfig


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "canonical", "torch_dtype"),
    [
        ("float16", "float16", torch.float16),
        ("fp16", "float16", torch.float16),
        ("bfloat16", "bfloat16", torch.bfloat16),
        ("bf16", "bfloat16", torch.bfloat16),
    ],
)
def test_amp_dtype_aliases(value, canonical, torch_dtype):
    assert normalize_amp_dtype(value) == canonical
    assert torch_amp_dtype(value) == torch_dtype


@pytest.mark.unit
def test_invalid_amp_dtype_fails_in_train_and_validation_configs():
    with pytest.raises(ValueError, match="amp_dtype"):
        TrainConfig(amp_dtype="float32")
    with pytest.raises(ValueError, match="amp_dtype"):
        ValidationConfig(data="unused.yaml", amp_dtype="float32")


@pytest.mark.unit
def test_trainer_autocast_context_uses_configured_dtype(monkeypatch):
    calls = []

    class _Context:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return False

    def fake_autocast(device_type, *, dtype):
        calls.append((device_type, dtype))
        return _Context()

    monkeypatch.setattr("libreyolo.training.trainer.autocast", fake_autocast)
    trainer = SimpleNamespace(config=SimpleNamespace(amp_dtype="bfloat16"))

    with BaseTrainer._autocast_context(trainer):
        pass

    assert calls == [("cuda", torch.bfloat16)]
