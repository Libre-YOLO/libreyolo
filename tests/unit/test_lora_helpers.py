"""Small LoRA helper tests that stay inside the PR gate."""

from __future__ import annotations

import builtins

import pytest
import torch

from libreyolo.training import lora as lora_helpers

pytestmark = pytest.mark.unit


def test_state_dict_has_lora_detects_adapter_tensors():
    assert lora_helpers.state_dict_has_lora({"encoder.lora_A.default.weight": torch.ones(1)})
    assert lora_helpers.state_dict_has_lora({"encoder.lora_B.default.weight": torch.ones(1)})
    assert lora_helpers.state_dict_has_lora({"encoder.lora_magnitude_vector": torch.ones(1)})
    assert not lora_helpers.state_dict_has_lora({"encoder.weight": torch.ones(1)})


def test_module_has_lora_detects_wrapped_or_adapter_modules():
    wrapped = torch.nn.Module()
    wrapped.peft_config = {}
    assert lora_helpers.module_has_lora(wrapped)

    module = torch.nn.Module()
    module.register_parameter("lora_A_default", torch.nn.Parameter(torch.ones(1)))
    assert lora_helpers.module_has_lora(module)

    plain = torch.nn.Linear(1, 1)
    assert not lora_helpers.module_has_lora(plain)


def test_apply_lora_to_detr_rejects_non_detr_models():
    # Raises before any peft import, so this stays inside the PR gate.
    with pytest.raises(ValueError, match=r"no \.backbone"):
        lora_helpers.apply_lora_to_detr(torch.nn.Linear(4, 4))


def test_detr_recipe_constants_are_exported():
    assert lora_helpers.DETR_LORA_RANK == 16
    assert lora_helpers.DETR_LORA_ALPHA == 16
    assert "linear1" in lora_helpers.DETR_TARGET_LINEAR_NAMES
    assert "TransformerDecoderLayer" in lora_helpers.DETR_BLOCK_CLASSES


def test_missing_peft_error_mentions_lora_extra(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "peft":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r'pip install "libreyolo\[lora\]"'):
        lora_helpers._require_peft()
