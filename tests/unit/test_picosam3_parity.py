"""Gated tensor parity against the pinned Apache-2.0 upstream checkout."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from libreyolo.models.picosam3.nn import PicoSAM3Network

pytestmark = [pytest.mark.unit, pytest.mark.external_data, pytest.mark.sam]


def test_picosam3_upstream_tensor_parity():
    upstream = os.environ.get("LIBREYOLO_PICOSAM3_UPSTREAM")
    checkpoint = os.environ.get("LIBREYOLO_PICOSAM3_CHECKPOINT")
    if not upstream or not checkpoint:
        pytest.skip("set LIBREYOLO_PICOSAM3_UPSTREAM and LIBREYOLO_PICOSAM3_CHECKPOINT")

    model_file = Path(upstream) / "model_compression" / "model.py"
    spec = importlib.util.spec_from_file_location("picosam3_upstream_model", model_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference = module.PicoSAM3().eval()
    port = PicoSAM3Network().eval()
    reference.load_state_dict(state_dict, strict=True)
    port.load_state_dict(state_dict, strict=True)

    torch.manual_seed(573)
    inputs = torch.randn((3, 3, 96, 96))
    with torch.no_grad():
        expected = reference(inputs)
        actual = port(inputs)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
