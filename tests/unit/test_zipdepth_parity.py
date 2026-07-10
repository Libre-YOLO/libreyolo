"""ZipDepth inference parity vs the official implementation.

Cross-loads the official checkpoints into both the upstream ``ZipDepth`` module
and LibreYOLO's port and asserts bitwise-identical outputs in eval mode, both
unfused (training form) and after each side's own reparameterization.

Skipped unless two env vars point at local artifacts (not fetched in CI):

- ``ZIPDEPTH_UPSTREAM_DIR``: clone of https://github.com/fabiotosi92/ZipDepth
  (provides the reference implementation and ``checkpoints/*.pth``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

UPSTREAM_DIR = os.environ.get("ZIPDEPTH_UPSTREAM_DIR")

pytestmark = [
    pytest.mark.external_data,
    pytest.mark.skipif(
        not UPSTREAM_DIR or not Path(UPSTREAM_DIR).is_dir(),
        reason="Set ZIPDEPTH_UPSTREAM_DIR to a clone of fabiotosi92/ZipDepth",
    ),
]

_CHECKPOINTS = {
    "b": "zipdepth_base.pth",
    "bnpu": "zipdepth_base_npu.pth",
}


def _load_upstream(size: str):
    sys.path.insert(0, UPSTREAM_DIR)
    try:
        from zipdepth.model.architecture import create_model
        from zipdepth.utils.model_utils import strip_state_dict_prefixes
    finally:
        sys.path.pop(0)

    ckpt_path = Path(UPSTREAM_DIR) / "checkpoints" / _CHECKPOINTS[size]
    if not ckpt_path.exists():
        pytest.skip(f"missing upstream checkpoint {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = strip_state_dict_prefixes(sd.get("model_state_dict", sd))

    model = create_model(variant="base", upsample_unfold=(size == "b"))
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    return model.eval(), sd


def _load_ours(size: str, sd: dict):
    from libreyolo.models.zipdepth.nn import LibreZipDepthNet

    model = LibreZipDepthNet(size=size)
    model.load_state_dict(sd, strict=True)
    return model.eval()


@pytest.mark.parametrize("size", sorted(_CHECKPOINTS))
@pytest.mark.parametrize("fused", [False, True])
def test_zipdepth_forward_parity(size: str, fused: bool):
    upstream, sd = _load_upstream(size)
    ours = _load_ours(size, sd)

    if fused:
        upstream.fuse_for_inference()
        ours.fuse_for_inference()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 256, 320)

    with torch.no_grad():
        up_out = upstream(x)
        our_out = ours(x)

    assert up_out.shape == our_out.shape
    diff = (up_out - our_out).abs().max().item()
    assert diff == 0.0, f"size={size} fused={fused} max_abs_diff={diff}"
