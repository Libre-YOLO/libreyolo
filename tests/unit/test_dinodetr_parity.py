# SPDX-License-Identifier: Apache-2.0
# Reference implementation: IDEA-Research/DINO at
# d84a491d41898b3befd8294d1cf2614661fc0953 (Apache-2.0).
"""Exact tensor parity against the pinned standalone DINO implementation.

Set ``DINODETR_UPSTREAM_DIR`` to the audited upstream checkout and
``DINODETR_OFFICIAL_WEIGHTS_DIR`` to a directory containing the three official
Google Drive checkpoints. The test verifies the commit, license, and weight
hashes before importing any reference code.
"""

from __future__ import annotations

import gc
import hashlib
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torchvision

from libreyolo.models.dinodetr.nn import LibreDINODETRModel

pytestmark = [pytest.mark.unit, pytest.mark.dinodetr, pytest.mark.external_data]

UPSTREAM_COMMIT = "d84a491d41898b3befd8294d1cf2614661fc0953"
UPSTREAM_LICENSE_SHA256 = (
    "465ed5f2f9d61880f2f37c7b8de6c7342813c80c92bc3542ea7dd55422c4637c"
)
OFFICIAL_CASES = {
    "r50": (
        "checkpoint0011_4scale.pth",
        "0bcd6b0c33d60ed33461ce6f02ce5797a819c7c02eb7e15b76adfb6df307955a",
        240,
    ),
    "r50s5": (
        "checkpoint0011_5scale.pth",
        "1ccc1b6b7139813e4d3bfbeecfcf88347ebc226829769a0bf16c4a114c275cc0",
        128,
    ),
    "swinl": (
        "checkpoint0027_5scale_swin.pth",
        "17ddce1592816a0c63a2edc94d4a0877ffeb086f397a6657e151c703a4c850b5",
        128,
    ),
}


class _Config(SimpleNamespace):
    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_inputs() -> tuple[Path, Path]:
    upstream_value = os.environ.get("DINODETR_UPSTREAM_DIR")
    weights_value = os.environ.get("DINODETR_OFFICIAL_WEIGHTS_DIR")
    if not upstream_value or not weights_value:
        pytest.skip("set DINODETR_UPSTREAM_DIR and DINODETR_OFFICIAL_WEIGHTS_DIR")
    upstream = Path(upstream_value).resolve()
    weights = Path(weights_value).resolve()
    if not (upstream / ".git").exists():
        pytest.fail(f"not a git checkout: {upstream}")
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == UPSTREAM_COMMIT
    license_bytes = (upstream / "LICENSE").read_text(encoding="utf-8").encode()
    assert hashlib.sha256(license_bytes).hexdigest() == UPSTREAM_LICENSE_SHA256
    for filename, expected_hash, _ in OFFICIAL_CASES.values():
        checkpoint = weights / filename
        assert checkpoint.is_file(), checkpoint
        assert _sha256(checkpoint) == expected_hash
    return upstream, weights


def _install_upstream_pure_attention(upstream: Path, monkeypatch):
    """Import upstream while replacing only its unavailable compiled operator."""
    # util.misc has a legacy minor-version comparison. 0.7 selects its modern
    # branch without changing model math on current torchvision releases.
    monkeypatch.setattr(torchvision, "__version__", "0.7.0")
    original_resnet50 = torchvision.models.resnet50

    def offline_resnet50(*args, **kwargs):
        kwargs.pop("pretrained", None)
        kwargs["weights"] = None
        return original_resnet50(*args, **kwargs)

    monkeypatch.setattr(torchvision.models, "resnet50", offline_resnet50)
    compiled_stub = types.ModuleType("MultiScaleDeformableAttention")
    compiled_stub.ms_deform_attn_forward = lambda *args, **kwargs: None
    compiled_stub.ms_deform_attn_backward = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "MultiScaleDeformableAttention", compiled_stub)
    monkeypatch.syspath_prepend(str(upstream))

    from models.dino.backbone import build_backbone
    from models.dino.deformable_transformer import build_deformable_transformer
    from models.dino.dino import DINO
    from models.dino.ops.functions.ms_deform_attn_func import (
        ms_deform_attn_core_pytorch,
    )
    from models.dino.ops.modules.ms_deform_attn import MSDeformAttn

    def pure_forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        del input_level_start_index
        batch, query_length, _ = query.shape
        _, input_length, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(
            batch, input_length, self.n_heads, self.d_model // self.n_heads
        )
        sampling_offsets = self.sampling_offsets(query).view(
            batch,
            query_length,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        )
        attention_weights = self.attention_weights(query).view(
            batch,
            query_length,
            self.n_heads,
            self.n_levels * self.n_points,
        )
        attention_weights = torch.nn.functional.softmax(attention_weights, -1).view(
            batch,
            query_length,
            self.n_heads,
            self.n_levels,
            self.n_points,
        )
        if reference_points.shape[-1] == 2:
            normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = reference_points[:, :, None, :, None, :] + (
                sampling_offsets / normalizer[None, None, None, :, None, :]
            )
        else:
            sampling_locations = reference_points[:, :, None, :, None, :2] + (
                sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        output = ms_deform_attn_core_pytorch(
            value,
            input_spatial_shapes,
            sampling_locations,
            attention_weights,
        )
        return self.output_proj(output)

    monkeypatch.setattr(MSDeformAttn, "forward", pure_forward)
    return build_backbone, build_deformable_transformer, DINO


def _load_config(upstream: Path, size: str) -> _Config:
    filename = "DINO_4scale.py" if size == "r50" else "DINO_5scale.py"
    raw = runpy.run_path(str(upstream / "config" / "DINO" / filename))
    config = _Config(
        **{key: value for key, value in raw.items() if not key.startswith("_")}
    )
    if size == "swinl":
        config.backbone = "swin_L_384_22k"
        # Checkpointing changes memory use during training, not inference math.
        config.use_checkpoint = False
    return config


def _build_upstream(upstream: Path, size: str, builders):
    build_backbone, build_transformer, dino = builders
    config = _load_config(upstream, size)
    return dino(
        build_backbone(config),
        build_transformer(config),
        num_classes=91,
        num_queries=900,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=False,
        fix_refpoints_hw=-1,
        num_feature_levels=config.num_feature_levels,
        nheads=8,
        two_stage_type="standard",
        two_stage_add_query_num=0,
        dec_pred_class_embed_share=True,
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=False,
        two_stage_bbox_embed_share=False,
        decoder_sa_type="sa",
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=91,
    )


def _tensor_leaves(value, prefix="output"):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensor_leaves(item, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensor_leaves(item, f"{prefix}[{index}]")


def test_all_three_variants_match_pinned_upstream_exactly(monkeypatch):
    upstream, weights = _verified_inputs()
    builders = _install_upstream_pure_attention(upstream, monkeypatch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for size, (filename, _, input_size) in OFFICIAL_CASES.items():
        checkpoint = torch.load(
            weights / filename, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint["model"]
        reference = _build_upstream(upstream, size, builders).eval()
        candidate = LibreDINODETRModel(size=size, nc=91).eval()
        reference.load_state_dict(state_dict, strict=True)
        candidate.load_state_dict(state_dict, strict=True)
        del checkpoint, state_dict
        reference.to(device)
        candidate.to(device)
        torch.manual_seed(637)
        input_tensor = torch.randn(1, 3, input_size, input_size, device=device)

        with torch.inference_mode():
            reference_output = reference(input_tensor)
            candidate_output = candidate(input_tensor)

        reference_leaves = dict(_tensor_leaves(reference_output))
        candidate_leaves = dict(_tensor_leaves(candidate_output))
        assert reference_leaves.keys() == candidate_leaves.keys()
        for key in reference_leaves:
            difference = (reference_leaves[key] - candidate_leaves[key]).abs().max()
            assert difference.item() == 0.0, (
                f"size={size} tensor={key} max_abs_diff={difference.item()}"
            )

        del (
            reference,
            candidate,
            input_tensor,
            reference_output,
            candidate_output,
            reference_leaves,
            candidate_leaves,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
