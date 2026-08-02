"""CenterNet native-architecture and official-conversion contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from libreyolo.models.centernet.nn import DCN, build_centernet

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("size", "entries"), (("resdcn18", 177), ("dla34", 400))
)
def test_official_parameter_layout_is_preserved(size, entries):
    model = build_centernet(size)
    assert len(model.state_dict()) == entries
    assert model.state_dict()["hm.2.weight"].shape[0] == 80
    assert any(isinstance(module, DCN) for module in model.modules())


@pytest.mark.parametrize("size", ("resdcn18", "dla34"))
def test_raw_heads_have_stride_four_shapes(size):
    model = build_centernet(size).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 128, 128))
    assert set(output) == {"hm", "wh", "reg"}
    assert output["hm"].shape == (1, 80, 32, 32)
    assert output["wh"].shape == output["reg"].shape == (1, 2, 32, 32)


@pytest.mark.external_data
def test_pinned_upstream_raw_outputs_match_exactly():
    if not os.environ.get("CENTERNET_UPSTREAM_DIR") or not os.environ.get(
        "CENTERNET_OFFICIAL_CKPT_DIR"
    ):
        pytest.skip("set CenterNet upstream and checkpoint environment variables")
    root = Path(__file__).resolve().parents[2]
    script = root / "weights" / "parity_centernet.py"
    completed = subprocess.run(
        [sys.executable, str(script)], cwd=root, check=False, text=True
    )
    assert completed.returncode == 0
