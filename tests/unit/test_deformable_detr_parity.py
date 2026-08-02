# SPDX-License-Identifier: Apache-2.0
# Reference implementation: fundamentalvision/Deformable-DETR at
# 11169a60c33333af00a4849f1808023eba96a931 (Apache-2.0).
"""Exact tensor parity against the pinned upstream implementation.

Set ``DEFORMABLE_DETR_UPSTREAM_DIR`` to a checkout at ``UPSTREAM_COMMIT``.
The test verifies the local checkout's commit and LICENSE hash before importing
it, downloads only the five pinned Apache-2.0 SenseTime safetensors, converts
them to native keys, and exercises upstream's own pure-PyTorch attention core.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torchvision

from libreyolo.models.deformable_detr.conversion import (
    convert_hf_deformable_detr_state_dict,
)
from libreyolo.models.deformable_detr.nn import LibreDeformableDETRModel

pytestmark = [pytest.mark.unit, pytest.mark.external_data, pytest.mark.network]

UPSTREAM_COMMIT = "11169a60c33333af00a4849f1808023eba96a931"
UPSTREAM_LICENSE_SHA256 = (
    "068413f8cf5e42e34d6e171e45a779df6dbea0a18249c0a223a3878e6de3cb27"
)

# size: (repo, revision, model.safetensors SHA256)
OFFICIAL_CASES = {
    "r50ss": (
        "SenseTime/deformable-detr-single-scale",
        "e880a4ca7bbe47b33d37ed90e2948efbbdad0d44",
        "82eeb57bbcdd02408afc53d5f5c874e3a7f27b5034194ae2c4475d06fceaa59b",
    ),
    "r50ssdc5": (
        "SenseTime/deformable-detr-single-scale-dc5",
        "c23332913d0ae1a8c98725e308eccba65a5933cc",
        "e71afa5f5900e2e769275156494195508efcadaab4275b0cd4c80f10369dc090",
    ),
    "r50": (
        "SenseTime/deformable-detr",
        "83ecd26945199939cb82806f988debdb71e6f43e",
        "caf1e3e61283c6ce35cd2d9adaa7033cf40997d4dfe434003bcdb9085cc8cf9b",
    ),
    "r50refine": (
        "SenseTime/deformable-detr-with-box-refine",
        "2e9e461623a8fdc296e19666c46c8a4389a3a6fe",
        "4113700fe8aade398808424b7c5c1304cfbf886adc6450a6ca5d50a702be3373",
    ),
    "r50twostage": (
        "SenseTime/deformable-detr-with-box-refine-two-stage",
        "e74bff70d69f3e825f6cefaf179bfba707f92054",
        "411bb4238a834d40fff651b1b5b7d6dd80c2dd28be1747eec7b6918674e85de6",
    ),
}

ARCHITECTURES = {
    "r50ss": (1, False, False, False),
    "r50ssdc5": (1, True, False, False),
    "r50": (4, False, False, False),
    "r50refine": (4, False, True, False),
    "r50twostage": (4, False, True, True),
}


def _verified_upstream_dir() -> Path:
    configured = os.environ.get("DEFORMABLE_DETR_UPSTREAM_DIR")
    if not configured:
        pytest.skip("set DEFORMABLE_DETR_UPSTREAM_DIR to the pinned upstream checkout")
    upstream = Path(configured).resolve()
    if not (upstream / ".git").exists():
        pytest.fail(f"not a git checkout: {upstream}")

    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == UPSTREAM_COMMIT
    # Git may materialize CRLF on Windows; hash canonical LF content so this
    # stays equal to the audited raw GitHub blob on every platform.
    license_bytes = (upstream / "LICENSE").read_text(encoding="utf-8").encode()
    license_hash = hashlib.sha256(license_bytes).hexdigest()
    assert license_hash == UPSTREAM_LICENSE_SHA256
    return upstream


def _install_upstream_pure_attention(upstream: Path, monkeypatch):
    """Import upstream while replacing only its unavailable compiled operator."""
    # util.misc parses only the first three version characters and mistakes
    # torchvision 0.26 for 0.2. A neutral version string selects its modern
    # branch without changing any model math.
    monkeypatch.setattr(torchvision, "__version__", "1.0.0")
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

    from models.backbone import build_backbone
    from models.deformable_detr import DeformableDETR
    from models.deformable_transformer import DeformableTransformer
    from models.ops.functions.ms_deform_attn_func import (
        ms_deform_attn_core_pytorch,
    )
    from models.ops.modules.ms_deform_attn import MSDeformAttn

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
    return build_backbone, DeformableDETR, DeformableTransformer


def _build_upstream(size: str, builders):
    build_backbone, deformable_detr, deformable_transformer = builders
    levels, dilation, refine, two_stage = ARCHITECTURES[size]
    args = SimpleNamespace(
        hidden_dim=256,
        position_embedding="sine",
        lr_backbone=0,
        masks=False,
        num_feature_levels=levels,
        backbone="resnet50",
        dilation=dilation,
    )
    backbone = build_backbone(args)
    transformer = deformable_transformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=levels,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=two_stage,
        two_stage_num_proposals=300,
    )
    return deformable_detr(
        backbone,
        transformer,
        num_classes=91,
        num_queries=300,
        num_feature_levels=levels,
        aux_loss=True,
        with_box_refine=refine,
        two_stage=two_stage,
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


def test_all_five_variants_match_pinned_upstream_exactly(monkeypatch):
    upstream = _verified_upstream_dir()
    builders = _install_upstream_pure_attention(upstream, monkeypatch)
    huggingface_hub = pytest.importorskip("huggingface_hub")
    safetensors = pytest.importorskip("safetensors.torch")

    torch.manual_seed(2020)
    input_tensor = torch.randn(1, 3, 128, 128)
    for size, (repo_id, revision, expected_sha256) in OFFICIAL_CASES.items():
        checkpoint_path = Path(
            huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
                revision=revision,
                token=False,
            )
        )
        assert (
            hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == expected_sha256
        )
        native_state = convert_hf_deformable_detr_state_dict(
            safetensors.load_file(str(checkpoint_path), device="cpu")
        )

        reference = _build_upstream(size, builders).eval()
        candidate = LibreDeformableDETRModel(size=size, nc=91).eval()
        reference.load_state_dict(native_state, strict=True)
        candidate.load_state_dict(native_state, strict=True)

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

        del reference, candidate, native_state, reference_output, candidate_output
