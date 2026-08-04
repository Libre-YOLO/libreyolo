"""Gated tensor parity against pinned MIT TEED and DexiNed checkouts."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from libreyolo.models.dexined.nn import DexiNedCore
from libreyolo.models.teed.nn import TEEDCore

pytestmark = [pytest.mark.unit, pytest.mark.external_data]

TEED_COMMIT = "40fa4b1391dc6424f88989d0ca75d5b592c8681d"
DEXINED_COMMIT = "08ed67ad0579f3969536a9719cdc1b829fb74fc1"


def _require_checkout(environment_name: str, expected_commit: str) -> Path:
    value = os.environ.get(environment_name)
    if not value:
        pytest.skip(f"set {environment_name} to a pinned upstream checkout")
    checkout = Path(value)
    if (checkout / ".git").exists():
        head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert head == expected_commit
    return checkout


def test_teed_upstream_checkpoint_tensor_parity(monkeypatch):
    upstream = _require_checkout("LIBREYOLO_TEED_UPSTREAM", TEED_COMMIT)
    checkpoint = os.environ.get("LIBREYOLO_TEED_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set LIBREYOLO_TEED_CHECKPOINT to a local checkpoint")

    # ted.py imports count_parameters from a utility module whose unrelated
    # plotting dependencies are not needed to construct or run the network.
    utility_stub = types.ModuleType("utils.img_processing")
    utility_stub.count_parameters = lambda model: sum(
        parameter.numel() for parameter in model.parameters()
    )
    monkeypatch.setitem(sys.modules, "utils.img_processing", utility_stub)
    monkeypatch.syspath_prepend(str(upstream))
    spec = importlib.util.spec_from_file_location(
        "teed_pinned_reference",
        upstream / "ted.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference = module.TED().eval()
    port = TEEDCore().eval()
    reference.load_state_dict(state_dict, strict=True)
    port.load_state_dict(state_dict, strict=True)

    torch.manual_seed(41)
    inputs = torch.randn(2, 3, 64, 80)
    with torch.no_grad():
        expected = reference(inputs)
        actual = port(inputs)
    assert len(actual) == len(expected) == 4
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=0.0,
            atol=0.0,
        )


def test_dexined_upstream_architecture_tensor_parity():
    upstream = _require_checkout("LIBREYOLO_DEXINED_UPSTREAM", DEXINED_COMMIT)
    spec = importlib.util.spec_from_file_location(
        "dexined_pinned_reference",
        upstream / "model.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    reference = module.DexiNed().eval()
    port = DexiNedCore().eval()
    port.load_state_dict(reference.state_dict(), strict=True)

    torch.manual_seed(42)
    inputs = torch.randn(1, 3, 32, 48)
    with torch.no_grad():
        expected = reference(inputs)
        actual = port(inputs)
    assert len(actual) == len(expected) == 7
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=0.0,
            atol=0.0,
        )
