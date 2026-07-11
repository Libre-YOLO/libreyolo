"""Golden parity against the pinned official DA3MONO-LARGE implementation.

Recipe:

1. Clone ``https://github.com/ByteDance-Seed/Depth-Anything-3`` at commit
   ``41736238f5bced4debf3f2a12375d2466874866d``.
2. Download ``depth-anything/DA3MONO-LARGE/model.safetensors`` at revision
   ``f465978e618db8cc79c83b8bbf24964857db1875``.
3. Install the upstream runtime dependencies and set ``DA3_UPSTREAM_DIR`` and
   ``DA3_SAFETENSORS`` to those local artifacts.

The test cross-loads the same 406 official tensors into both implementations,
applies the official sky handling to the reference depth, converts that depth
to the LibreYOLO inverse-depth contract, and compares every output pixel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

UPSTREAM_DIR = os.environ.get("DA3_UPSTREAM_DIR")
SAFETENSORS = os.environ.get("DA3_SAFETENSORS")

pytestmark = [
    pytest.mark.external_data,
    pytest.mark.skipif(
        not UPSTREAM_DIR
        or not Path(UPSTREAM_DIR).is_dir()
        or not SAFETENSORS
        or not Path(SAFETENSORS).is_file(),
        reason="Set DA3_UPSTREAM_DIR and DA3_SAFETENSORS to pinned local artifacts",
    ),
]


def test_da3mono_large_contract_output_matches_official():
    source_root = str(Path(UPSTREAM_DIR) / "src")
    sys.path.insert(0, source_root)
    try:
        from depth_anything_3.model.dinov2.dinov2 import DinoV2
        from depth_anything_3.model.dpt import DPT
    finally:
        sys.path.pop(0)

    official_backbone = DinoV2(
        name="vitl",
        out_layers=[4, 11, 17, 23],
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        cat_token=False,
    )
    official_head = DPT(
        dim_in=1024,
        output_dim=1,
        features=256,
        out_channels=[256, 512, 1024, 1024],
    )
    state = load_file(SAFETENSORS, device="cpu")
    official_backbone.load_state_dict(
        {
            key.removeprefix("model.backbone."): value
            for key, value in state.items()
            if key.startswith("model.backbone.")
        },
        strict=True,
    )
    official_head.load_state_dict(
        {
            key.removeprefix("model.head."): value
            for key, value in state.items()
            if key.startswith("model.head.")
        },
        strict=True,
    )

    from libreyolo.models.depth_anything3.nn import LibreDepthAnything3Net

    ours = LibreDepthAnything3Net()
    ours.load_state_dict(
        {key.removeprefix("model."): value for key, value in state.items()},
        strict=True,
    )
    del state

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    official_backbone.eval().to(device)
    official_head.eval().to(device)
    ours.eval().to(device)
    torch.manual_seed(123)
    image = torch.rand(1, 3, 56, 70, device=device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    with torch.inference_mode():
        features, _ = official_backbone(
            ((image - mean) / std).unsqueeze(1), export_feat_layers=[]
        )
        output = official_head(features, 56, 70, patch_start_idx=0)
        depth = LibreDepthAnything3Net._apply_mono_sky(output.depth, output.sky)
        reference = torch.reciprocal(depth.clamp_min(1e-6))[:, 0].unsqueeze(1)
        actual = ours(image)

    assert actual.shape == reference.shape == (1, 1, 56, 70)
    assert torch.allclose(actual, reference, rtol=1e-6, atol=1e-7)
